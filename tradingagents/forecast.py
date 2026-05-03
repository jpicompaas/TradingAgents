"""Three-flavor (headwinds / same / tailwinds) forward-price forecast.

Anchors the curves on the LLM's +30 / +60 / +90 day estimates that the
Portfolio Manager (and Trader) emit in their ``LevelEstimates.estimate_*``
fields, then interpolates a daily path between today and each anchor.
±1σ bands widen with √t using realized volatility from the trailing year
of OHLCV — so the *uncertainty* is grounded in real data even though the
*direction* comes from the LLM.

Three scenarios:

- **headwinds** — uses each horizon's ``low`` (or ``expected`` if low is
  unset). Bearish anchor.
- **same** — uses ``expected``. Base case.
- **tailwinds** — uses ``high`` (or ``expected``). Bullish anchor.

Outputs in the report bundle directory:
- ``forecast.csv`` — daily rows with all three paths and their ±1σ bands
- ``forecast.png`` — fan chart visualizing all three (matplotlib, optional)
- A summary block injected into ``complete_report.md``
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_SCENARIOS = (
    ("headwinds", "low", "#cc4444"),
    ("same", "expected", "#666666"),
    ("tailwinds", "high", "#229922"),
)


def _fetch_history(ticker: str, end_date: str, lookback_days: int = 365) -> Optional[pd.DataFrame]:
    """Pull ~1 year of daily OHLCV ending on ``end_date`` from yfinance.

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


def _piecewise_log_path(
    last_close: float, anchors: list[tuple[int, float]], days_max: int
) -> np.ndarray:
    """Log-linearly interpolate a price path from day 0 (last_close) through anchors."""
    if not anchors:
        return np.array([last_close])
    sorted_anchors = sorted(anchors)
    x = np.array([0] + [a[0] for a in sorted_anchors])
    y = np.log(np.array([last_close] + [a[1] for a in sorted_anchors]))
    grid = np.arange(0, days_max + 1)
    return np.exp(np.interp(grid, x, y))


def _bands(path: np.ndarray, daily_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """1σ confidence bands widening with √t."""
    days = np.arange(len(path))
    sigma_t = daily_sigma * np.sqrt(days)
    return path * np.exp(-sigma_t), path * np.exp(+sigma_t)


def _scenario_anchors(
    levels: Mapping[str, Any], scenario_attr: str
) -> list[tuple[int, float]]:
    """Build (days, anchor_price) pairs for one scenario.

    Falls back to ``expected`` when the scenario-specific field (low/high)
    is unset, so a single point estimate still drives all three flavors.
    """
    out: list[tuple[int, float]] = []
    for days, key in ((30, "estimate_30d"), (60, "estimate_60d"), (90, "estimate_90d")):
        est = levels.get(key)
        if not est:
            continue
        val = est.get(scenario_attr)
        if val is None:
            val = est.get("expected")
        if val is not None:
            try:
                out.append((days, float(val)))
            except (TypeError, ValueError):
                continue
    return out


def generate_three_flavor_forecast(
    ticker: str,
    analysis_date: str,
    levels: Optional[Mapping[str, Any]],
    output_dir: Path,
) -> Optional[dict]:
    """Produce headwinds / same / tailwinds forecast curves and persist them.

    Returns a metadata dict on success (paths to outputs, vol stats),
    or None when there is nothing to forecast — typically because the
    Portfolio Manager left the +30/+60/+90 estimates unset, or because
    historical OHLCV could not be fetched.
    """
    if not levels:
        return None

    history = _fetch_history(ticker, analysis_date)
    if history is None or len(history) < 30:
        logger.info("forecast: insufficient history for %s; skipping", ticker)
        return None

    last_close = float(history["Close"].iloc[-1])
    log_returns = np.log(history["Close"]).diff().dropna()
    daily_sigma = float(log_returns.std())

    paths: dict[str, dict] = {}
    for scenario, attr, color in _SCENARIOS:
        anchors = _scenario_anchors(levels, attr)
        if not anchors:
            continue
        max_day = max(d for d, _ in anchors)
        path = _piecewise_log_path(last_close, anchors, max_day)
        lower, upper = _bands(path, daily_sigma)
        paths[scenario] = {
            "color": color,
            "anchors": anchors,
            "days": list(range(max_day + 1)),
            "path": path,
            "lower": lower,
            "upper": upper,
        }

    if not paths:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Daily CSV ----------------------------------------------------------
    base_date = datetime.strptime(analysis_date, "%Y-%m-%d")
    max_day = max(max(p["days"]) for p in paths.values())
    rows = []
    for d in range(max_day + 1):
        row: dict[str, Any] = {
            "day": d,
            "date": (base_date + timedelta(days=d)).strftime("%Y-%m-%d"),
        }
        for scenario, data in paths.items():
            if d <= max(data["days"]):
                row[scenario] = round(float(data["path"][d]), 4)
                row[f"{scenario}_low"] = round(float(data["lower"][d]), 4)
                row[f"{scenario}_high"] = round(float(data["upper"][d]), 4)
        rows.append(row)
    csv_path = output_dir / "forecast.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Chart --------------------------------------------------------------
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
                label=scenario.capitalize(),
                linewidth=2,
            )
            ax.fill_between(
                data["days"],
                data["lower"],
                data["upper"],
                alpha=0.12,
                color=data["color"],
            )
            for d, v in data["anchors"]:
                ax.scatter([d], [v], color=data["color"], s=36, zorder=5, edgecolors="white")

        ax.axhline(
            last_close,
            color="black",
            linestyle=":",
            alpha=0.5,
            label=f"Today ({last_close:.2f})",
        )
        ax.set_xlabel("Days from analysis")
        ax.set_ylabel("Price")
        ax.set_title(
            f"{ticker} — three-flavor forecast (anchored at +30/+60/+90d, ±1σ bands)"
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

    return {
        "csv_path": csv_path,
        "png_path": png_path,
        "last_close": last_close,
        "daily_sigma": daily_sigma,
        "annualized_vol_pct": daily_sigma * float(np.sqrt(252)) * 100.0,
        "scenarios": {
            scenario: {
                "anchors": data["anchors"],
                "max_day": max(data["days"]),
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
    sections = ["## VII. Forecast Curves"]
    sections.append(
        f"Today's close: **{meta['last_close']:.2f}** · "
        f"Realized annualized vol: **{meta['annualized_vol_pct']:.1f}%** · "
        f"Bands shown are ±1σ widening with √t."
    )
    if meta.get("png_path"):
        sections.append(f"![{ticker} forecast](forecast.png)")
    sections.append("")
    sections.append("| Scenario | +30d / +60d / +90d (anchor) |")
    sections.append("|---|---|")
    for scenario, info in meta["scenarios"].items():
        anchors_str = " / ".join(f"{v:.0f}" for _, v in info["anchors"])
        sections.append(f"| {scenario.capitalize()} | {anchors_str} |")
    sections.append("")
    sections.append("Daily resolution path lives in `forecast.csv`.")
    return "\n".join(sections)
