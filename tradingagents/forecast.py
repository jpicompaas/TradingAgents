"""GARCH(1,1)-t Monte Carlo forecast — three-flavor (headwinds / same / tailwinds).

References (see ``/references`` in the webapp for full bibliography):

  - Bollerslev (1986) — Generalized Autoregressive Conditional Heteroskedasticity
  - Bollerslev (1987) — A conditionally heteroskedastic time series model for
    speculative prices [Student-t innovations]
  - Glosten, Jagannathan, Runkle (1993) — leverage-effect GARCH (GJR)
  - Cont (2001) — Empirical properties of asset returns: stylized facts
  - Andersen, Bollerslev, Diebold, Labys (2001) — realized volatility

Why this replaces ``today × exp(±k·σ·√t)``: the prior model assumed constant
volatility and a deterministic envelope, which contradicts the single most
robust stylized fact about equity returns — volatility clusters (Cont 2001).
GARCH directly models that clustering; Student-t innovations (Bollerslev 1987)
capture the fat tails that a Gaussian model under-represents in the bear case.

Pipeline:

  1. Fit GARCH(1,1)-t on ~1y daily log returns. With ``GARCH_LEVERAGE=1`` the
     fit becomes GJR-GARCH-t to capture the leverage effect (vol asymmetry —
     equities drop faster than they recover).
  2. Forecast 90 days of conditional variance via Monte Carlo simulation
     (``arch.forecast(method='simulation', simulations=N)``).
  3. Optional drift adjustment: when the Portfolio Manager's ``expected`` at
     +90d is within ±50% of today's close, shift simulated returns by a
     constant ``δ`` so the median path lands on that target. This preserves
     the prior "same = LLM-anchored" behavior without contaminating the
     vol-driven dispersion of the bull/bear curves.
  4. Compute per-day percentiles. Each scenario gets a central percentile
     plus an inner-percentile uncertainty band:

       Scenario   path   low    high
       headwinds  P10    P2.5   P25
       same       P50    P25    P75
       tailwinds  P90    P75    P97.5

  Bands have probabilistic meaning: the bear ``low`` is the 2.5th-percentile
  path — only a 2.5% chance of finishing below it under the fitted model.

Outputs (unchanged contract — drop-in replacement):
  - ``forecast.csv`` — daily rows with all three paths and their bands
  - ``forecast.png`` — fan chart visualizing the three scenarios + bands
  - A summary block injected into ``complete_report.md``
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forecast configuration
# ---------------------------------------------------------------------------

# Forecast horizon (days). The CSV always covers 0..MAX_HORIZON inclusive.
MAX_HORIZON = 90

# Number of Monte Carlo paths. 5000 is enough for stable percentile
# estimates at 90d horizon while still finishing in <2s on a laptop.
DEFAULT_N_SIMS = 5000

# Per-scenario percentile choices — see module docstring.
_SCENARIO_PERCENTILES: dict[str, tuple[float, float, float]] = {
    "headwinds": (2.5, 10.0, 25.0),    # (low, central, high)
    "same":      (25.0, 50.0, 75.0),
    "tailwinds": (75.0, 90.0, 97.5),
}

_SCENARIO_COLORS = {
    "headwinds": "#cc4444",
    "same":      "#666666",
    "tailwinds": "#229922",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _fetch_history(ticker: str, end_date: str, lookback_days: int = 365) -> Optional[pd.DataFrame]:
    """Pull ~1y of daily OHLCV ending on ``end_date`` from yfinance.

    Returns None on any failure — forecast generation is best-effort and
    must never crash the report bundle.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.info("forecast: yfinance not installed; skipping forecast")
        return None
    try:
        end = datetime.strptime(end_date, "%Y-%m-%d")
        start = end - timedelta(days=lookback_days + 30)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        return df.dropna()
    except Exception as exc:
        logger.warning("forecast: yfinance fetch failed (%s); skipping forecast", exc)
        return None


# ---------------------------------------------------------------------------
# Drift adjustment toward the LLM's central view
# ---------------------------------------------------------------------------


def _llm_value(
    levels: Optional[Mapping[str, Any]], horizon_days: int, attr: str
) -> Optional[float]:
    """Extract the LLM's value for one horizon × attribute, or None."""
    if not levels:
        return None
    est = levels.get(f"estimate_{horizon_days}d")
    if not est:
        return None
    val = est.get(attr)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _llm_drift_target(
    last_close: float, levels: Optional[Mapping[str, Any]]
) -> Optional[float]:
    """Daily log-return drift δ that lands the median on PM's expected_90d.

    Returns None if no usable LLM expected exists, or if the value is
    implausible (>50% off today's close — typically stale training-data
    prices).
    """
    expected = _llm_value(levels, MAX_HORIZON, "expected")
    if expected is None:
        # Fall back to +60d, then +30d.
        for h in (60, 30):
            v = _llm_value(levels, h, "expected")
            if v is not None and 0.5 * last_close <= v <= 2.0 * last_close:
                return float(np.log(v / last_close)) / float(h)
        return None
    if not (0.5 * last_close <= expected <= 2.0 * last_close):
        return None
    return float(np.log(expected / last_close)) / float(MAX_HORIZON)


# ---------------------------------------------------------------------------
# GARCH-MC core
# ---------------------------------------------------------------------------


def _garch_simulate(
    log_returns: np.ndarray,
    n_sims: int,
    horizon: int,
    use_leverage: bool,
) -> Optional[np.ndarray]:
    """Fit GARCH(1,1)-t (or GJR-GARCH-t) and return ``(n_sims, horizon)``
    array of simulated daily log returns (decimal scale).

    Returns None on any failure so the caller can fall back to a plain
    historical-vol Monte Carlo.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.info("forecast: 'arch' not installed; falling back to historical-vol MC")
        return None

    # arch optimizes much more stably when returns are scaled into the
    # "≈ 1 in magnitude" range — pass percentage returns and rescale on
    # the way out. (See Sheppard's documentation; standard practice.)
    pct = log_returns * 100.0

    try:
        model = arch_model(
            pct,
            mean="Constant",
            vol="GARCH",
            p=1,
            o=1 if use_leverage else 0,
            q=1,
            dist="t",
            rescale=False,
        )
        fit = model.fit(disp="off", show_warning=False)
    except Exception as exc:
        logger.warning("forecast: GARCH fit failed (%s); falling back", exc)
        return None

    try:
        fcast = fit.forecast(
            horizon=horizon,
            method="simulation",
            simulations=n_sims,
            reindex=False,
        )
        # `simulations.values` shape: (1, n_sims, horizon)
        sims_pct = np.asarray(fcast.simulations.values)[0]
    except Exception as exc:
        logger.warning("forecast: GARCH simulation failed (%s); falling back", exc)
        return None

    return sims_pct / 100.0  # back to decimal log-return scale


def _historical_vol_simulate(
    log_returns: np.ndarray, n_sims: int, horizon: int
) -> np.ndarray:
    """Constant-σ Monte Carlo fallback: t-distributed iid daily returns.

    Used when GARCH is unavailable or the fit fails. Loses vol clustering
    but preserves the percentile-based scenario interpretation.
    """
    sigma = float(np.std(log_returns, ddof=1))
    # Use Student-t with 6 d.o.f. — a reasonable default for daily equity
    # returns per Bollerslev (1987); not estimated to keep the fallback fast.
    rng = np.random.default_rng(seed=42)
    df = 6
    eps = rng.standard_t(df=df, size=(n_sims, horizon))
    # Standardize so var(eps) = 1 (Student-t variance is df/(df-2)).
    eps *= float(np.sqrt((df - 2) / df))
    return eps * sigma


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def generate_three_flavor_forecast(
    ticker: str,
    analysis_date: str,
    levels: Optional[Mapping[str, Any]],
    output_dir: Path,
    n_sims: int = DEFAULT_N_SIMS,
) -> Optional[dict]:
    """Produce headwinds / same / tailwinds forecast curves and persist them.

    Returns a metadata dict on success or None when there is nothing to
    forecast (typically yfinance returned no usable history).
    """
    history = _fetch_history(ticker, analysis_date)
    if history is None or len(history) < 60:
        logger.info("forecast: insufficient history for %s; skipping", ticker)
        return None

    last_close = float(history["Close"].iloc[-1])
    log_returns = np.log(history["Close"]).diff().dropna().to_numpy()
    realized_daily_sigma = float(np.std(log_returns, ddof=1))

    use_leverage = os.environ.get("GARCH_LEVERAGE", "").lower() in ("1", "true", "yes")
    horizon = MAX_HORIZON

    # 1. GARCH-MC, with constant-vol fallback so the run never empty-handed.
    sim_returns = _garch_simulate(log_returns, n_sims, horizon, use_leverage)
    method = "garch_t"
    if sim_returns is None:
        sim_returns = _historical_vol_simulate(log_returns, n_sims, horizon)
        method = "historical_vol_t"

    # 2. Drift adjustment toward the LLM's expected (only the median moves
    #    meaningfully; the spread is preserved).
    drift = _llm_drift_target(last_close, levels)
    if drift is not None:
        sim_returns = sim_returns + drift

    # 3. Path construction: P_t = P_0 · exp(Σ r_t).
    cum_log = np.cumsum(sim_returns, axis=1)
    sim_prices = last_close * np.exp(cum_log)  # shape (n_sims, horizon)

    # 4. Per-day percentiles for each scenario + bands.
    paths: dict[str, dict[str, Any]] = {}
    for scenario, (p_lo, p_mid, p_hi) in _SCENARIO_PERCENTILES.items():
        path_mid = np.percentile(sim_prices, p_mid, axis=0)
        path_lo = np.percentile(sim_prices, p_lo, axis=0)
        path_hi = np.percentile(sim_prices, p_hi, axis=0)
        # Day-0 anchor is today's close for all three paths.
        paths[scenario] = {
            "color": _SCENARIO_COLORS[scenario],
            "percentile": p_mid,
            "days": list(range(horizon + 1)),
            "path": np.concatenate([[last_close], path_mid]),
            "lower": np.concatenate([[last_close], path_lo]),
            "upper": np.concatenate([[last_close], path_hi]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV -----------------------------------------------------------
    base_date = datetime.strptime(analysis_date, "%Y-%m-%d")
    rows = []
    for d in range(horizon + 1):
        row: dict[str, Any] = {
            "day": d,
            "date": (base_date + timedelta(days=d)).strftime("%Y-%m-%d"),
        }
        for scenario, data in paths.items():
            row[scenario] = round(float(data["path"][d]), 4)
            row[f"{scenario}_low"] = round(float(data["lower"][d]), 4)
            row[f"{scenario}_high"] = round(float(data["upper"][d]), 4)
        rows.append(row)
    csv_path = output_dir / "forecast.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ---- Chart ---------------------------------------------------------
    png_path: Optional[Path] = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 6.5))
        for scenario, data in paths.items():
            ax.plot(
                data["days"],
                data["path"],
                color=data["color"],
                label=f"{scenario.capitalize()} (P{int(data['percentile'])})",
                linewidth=2,
            )
            ax.fill_between(
                data["days"],
                data["lower"],
                data["upper"],
                alpha=0.13,
                color=data["color"],
            )
        ax.axhline(
            last_close,
            color="black",
            linestyle=":",
            alpha=0.5,
            label=f"Today ({last_close:.2f})",
        )
        ax.set_xlabel("Days from analysis")
        ax.set_ylabel("Price")
        algo_label = "GJR-GARCH(1,1)-t" if use_leverage else "GARCH(1,1)-t"
        if method != "garch_t":
            algo_label = "Historical-vol-t (GARCH unavailable)"
        ax.set_title(
            f"{ticker} — {algo_label} Monte Carlo, {n_sims} paths, percentile bands"
        )
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        png_path = output_dir / "forecast.png"
        fig.savefig(png_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        logger.info("forecast: matplotlib unavailable; skipping chart")
    except Exception as exc:
        logger.warning("forecast: chart generation failed: %s", exc)

    # 5. Probability of finishing below today's close — convenient summary
    #    that the prior fixed-envelope model couldn't express.
    prob_below_today = float(np.mean(sim_prices[:, -1] < last_close))

    return {
        "csv_path": csv_path,
        "png_path": png_path,
        "method": method,
        "leverage": use_leverage,
        "n_sims": n_sims,
        "horizon": horizon,
        "last_close": last_close,
        "daily_sigma": realized_daily_sigma,
        "annualized_vol_pct": realized_daily_sigma * float(np.sqrt(252)) * 100.0,
        "drift_applied": drift is not None,
        "prob_below_today_at_horizon": prob_below_today,
        "scenarios": {
            scenario: {
                "percentile": data["percentile"],
                "max_day": horizon,
                "expected_at_max_day": float(data["path"][-1]),
                "lower_at_max_day": float(data["lower"][-1]),
                "upper_at_max_day": float(data["upper"][-1]),
            }
            for scenario, data in paths.items()
        },
    }


def render_forecast_section(meta: Optional[Mapping[str, Any]], ticker: str) -> str:
    """Compact markdown summary of the forecast for ``complete_report.md``."""
    if not meta:
        return ""
    method_label = {
        "garch_t": "GJR-GARCH(1,1)-t" if meta.get("leverage") else "GARCH(1,1)-t",
        "historical_vol_t": "Historical-vol Student-t (GARCH unavailable)",
    }.get(meta.get("method", "garch_t"), "GARCH(1,1)-t")

    sections = ["## VII. Forecast Curves"]
    sections.append(
        f"Today's close: **{meta['last_close']:.2f}** · "
        f"Realized annualized vol: **{meta['annualized_vol_pct']:.1f}%** · "
        f"Method: **{method_label}** ({meta.get('n_sims', 0)} Monte Carlo paths) · "
        f"P(below today at +{meta.get('horizon', 90)}d): **{meta['prob_below_today_at_horizon'] * 100:.0f}%**"
    )
    sections.append(
        "_**Scenarios are percentile paths** of the simulated distribution: "
        "Headwinds = P10 (10% chance of being worse), Same = P50 (median), "
        "Tailwinds = P90 (10% chance of being better). Bands show the inner-"
        "percentile uncertainty around each scenario — e.g. the bear-case "
        "low is the 2.5th-percentile path. See `/references` in the webapp "
        "for the GARCH/MC papers behind this approach._"
    )
    if meta.get("png_path"):
        sections.append(f"![{ticker} forecast](forecast.png)")
    sections.append("")
    sections.append("| Scenario | Percentile | +90d price | +90d band |")
    sections.append("|---|---|---|---|")
    for scenario, info in meta["scenarios"].items():
        sections.append(
            f"| {scenario.capitalize()} | P{int(info['percentile'])} | "
            f"{info['expected_at_max_day']:.2f} | "
            f"{info['lower_at_max_day']:.2f} – {info['upper_at_max_day']:.2f} |"
        )
    sections.append("")
    sections.append("Daily resolution path lives in `forecast.csv`.")
    return "\n".join(sections)
