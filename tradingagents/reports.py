"""Persist the full analysis bundle (analysts, research, trading, risk,
portfolio + raw state) to a per-run directory.

Used by both the CLI and ``main.py``. The default root is
``trading-reports/`` in the current working directory; callers can override.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from tradingagents.agents.utils.personas import get_persona
from tradingagents.dataflows.config import get_config
from tradingagents.forecast import (
    generate_three_flavor_forecast,
    render_forecast_section,
)


DEFAULT_REPORTS_ROOT = Path("trading-reports")


def _strategy_header(ticker: str) -> str:
    """Build the top-of-report block describing the active persona/strategy.

    Reads the live dataflow config (set by ``propagate()``) so it reflects
    what was actually used in the run — provider, model pair, debate depth,
    and the optional investor-persona overlay applied to the Trader and
    Portfolio Manager.
    """
    cfg = get_config()
    provider = cfg.get("llm_provider") or "?"
    deep = cfg.get("deep_think_llm") or "?"
    quick = cfg.get("quick_think_llm") or "?"
    debate_rounds = cfg.get("max_debate_rounds")
    risk_rounds = cfg.get("max_risk_discuss_rounds")

    persona = get_persona(cfg.get("trading_persona"))
    if persona is not None:
        persona_block = (
            f"**Investor persona:** {persona.display_name} "
            f"(`{persona.key}`)\n\n"
            f"**Strategy / philosophy applied to Trader & Portfolio Manager:**\n\n"
            f"> {persona.fragment}\n"
        )
    else:
        persona_block = (
            "**Investor persona:** _none_ (neutral analysis — set "
            "`TRADINGAGENTS_PERSONA` in `.env` to apply an investor lens to "
            "the Trader and Portfolio Manager).\n"
        )

    return (
        f"# Trading Analysis Report: {ticker}\n\n"
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Configuration\n\n"
        f"- **LLM provider:** `{provider}`\n"
        f"- **Deep-thinking model:** `{deep}` (Research Manager, Portfolio Manager)\n"
        f"- **Quick-thinking model:** `{quick}` (analysts, researchers, risk debators, trader)\n"
        f"- **Debate rounds:** {debate_rounds}, **Risk-discussion rounds:** {risk_rounds}\n\n"
        f"{persona_block}\n"
        f"---\n\n"
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def save_run_bundle(
    final_state: Mapping[str, Any],
    ticker: str,
    *,
    root: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Write a complete report bundle for a single run.

    Folder name includes the active investor-persona key when one is set,
    so it's easy to compare runs of the same ticker under different lenses
    side-by-side: ``<TICKER>_<persona>_<timestamp>`` (or ``<TICKER>_<timestamp>``
    if no persona overlay is active).

    Layout::

        <root>/<TICKER>_<persona>_<timestamp>/
            complete_report.md
            full_state.json
            1_analysts/{market,sentiment,news,fundamentals}.md
            2_research/{bull,bear,manager}.md
            3_trading/trader.md
            4_risk/{aggressive,conservative,neutral}.md
            5_portfolio/decision.md

    Returns the bundle directory path.
    """
    root = Path(root) if root is not None else DEFAULT_REPORTS_ROOT
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    persona = get_persona(get_config().get("trading_persona"))
    persona_slug = persona.key if persona is not None else ""
    folder = (
        f"{ticker}_{persona_slug}_{timestamp}" if persona_slug else f"{ticker}_{timestamp}"
    )
    save_path = root / folder
    save_path.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []

    # 1. Analysts
    analyst_specs = [
        ("market_report", "market.md", "Market Analyst"),
        ("sentiment_report", "sentiment.md", "Social Analyst"),
        ("news_report", "news.md", "News Analyst"),
        ("fundamentals_report", "fundamentals.md", "Fundamentals Analyst"),
    ]
    analyst_parts: list[tuple[str, str]] = []
    for state_key, filename, display in analyst_specs:
        body = final_state.get(state_key)
        if body:
            _write(save_path / "1_analysts" / filename, body)
            analyst_parts.append((display, body))
    if analyst_parts:
        sections.append(
            "## I. Analyst Team Reports\n\n"
            + "\n\n".join(f"### {n}\n{t}" for n, t in analyst_parts)
        )

    # 2. Research
    debate = final_state.get("investment_debate_state") or {}
    research_specs = [
        ("bull_history", "bull.md", "Bull Researcher"),
        ("bear_history", "bear.md", "Bear Researcher"),
        ("judge_decision", "manager.md", "Research Manager"),
    ]
    research_parts: list[tuple[str, str]] = []
    for state_key, filename, display in research_specs:
        body = debate.get(state_key)
        if body:
            _write(save_path / "2_research" / filename, body)
            research_parts.append((display, body))
    if research_parts:
        sections.append(
            "## II. Research Team Decision\n\n"
            + "\n\n".join(f"### {n}\n{t}" for n, t in research_parts)
        )

    # 3. Trading
    trader_plan = final_state.get("trader_investment_plan")
    if trader_plan:
        _write(save_path / "3_trading" / "trader.md", trader_plan)
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{trader_plan}")

    # 4. Risk Management
    risk = final_state.get("risk_debate_state") or {}
    risk_specs = [
        ("aggressive_history", "aggressive.md", "Aggressive Analyst"),
        ("conservative_history", "conservative.md", "Conservative Analyst"),
        ("neutral_history", "neutral.md", "Neutral Analyst"),
    ]
    risk_parts: list[tuple[str, str]] = []
    for state_key, filename, display in risk_specs:
        body = risk.get(state_key)
        if body:
            _write(save_path / "4_risk" / filename, body)
            risk_parts.append((display, body))
    if risk_parts:
        sections.append(
            "## IV. Risk Management Team Decision\n\n"
            + "\n\n".join(f"### {n}\n{t}" for n, t in risk_parts)
        )

    # 5. Portfolio Manager
    if risk.get("judge_decision"):
        _write(save_path / "5_portfolio" / "decision.md", risk["judge_decision"])
        sections.append(
            "## V. Portfolio Manager Decision\n\n### Portfolio Manager\n"
            + risk["judge_decision"]
        )

    # Top-level final decision (separate from Portfolio Manager debate)
    final_decision = final_state.get("final_trade_decision")
    if final_decision:
        _write(save_path / "5_portfolio" / "final_trade_decision.md", final_decision)
        sections.append(
            "## VI. Final Trade Decision\n\n" + final_decision
        )

    # Three-flavor forecast (headwinds / same / tailwinds).
    # Best-effort: skipped silently if the PM didn't populate `pm_levels`,
    # if yfinance is unavailable, or if matplotlib isn't installed.
    forecast_meta = generate_three_flavor_forecast(
        ticker=ticker,
        analysis_date=str(final_state.get("trade_date") or ""),
        levels=final_state.get("pm_levels"),
        output_dir=save_path,
    )
    forecast_section = render_forecast_section(forecast_meta, ticker)
    if forecast_section:
        sections.append(forecast_section)

    # Consolidated report — header includes the active persona and the
    # strategy/philosophy fragment that was actually applied during the run.
    (save_path / "complete_report.md").write_text(
        _strategy_header(ticker) + "\n\n".join(sections), encoding="utf-8"
    )

    # Raw state for downstream tooling / re-analysis
    try:
        (save_path / "full_state.json").write_text(
            json.dumps(_jsonable(final_state), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:  # never block the human-readable bundle on JSON quirks
        pass

    return save_path


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of LangChain message objects to plain JSON."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
