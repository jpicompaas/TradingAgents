"""Per-ticker scenario synthesis: headwinds / same / tailwinds with probabilities.

Runs once at the end of each analysis. Reads the bull / bear / research-
manager debate plus a snapshot of the analyst reports, and asks the LLM
for a structured ``ScenarioBreakdown`` — concrete, ticker-specific
factor lists, each tagged with its own probability and impact.

Best-effort: if the structured call fails (weak provider, transient
error) we silently skip — the run still produces ``forecast.csv`` /
``forecast.png``, and the webapp's forecast view falls back to showing
just the curves and the bull/bear assumptions extracted from disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from tradingagents.agents.schemas import ScenarioBreakdown

logger = logging.getLogger(__name__)


_SYSTEM = """You are a scenario analyst synthesizing forward-looking risk factors.

Given a bull case, bear case, and research-manager decision for one ticker,
produce three breakdowns over the next 90 days:

  - HEADWINDS: 3-6 concrete factors that would push the stock LOWER
  - SAME / CURRENT: 3-6 conditions whose persistence keeps the stock FLAT
  - TAILWINDS: 3-6 concrete factors that would push the stock HIGHER

Each factor must be specific to this ticker — reference real catalysts,
named segments, customer concentrations, regulators, products. Do not
write generic phrases like "macro uncertainty" or "competitive pressure"
without naming the actual competitor or macro driver.

For each factor assign:
  - probability_pct (0-100): your subjective probability it materializes
    within 90 days. Independent across factors — they need not sum to 100.
  - impact: 'low' (<3%), 'medium' (3-8%), 'high' (>8%) effect on price.
  - rationale: one sentence sourcing the call from the debate.
"""


def _trim(text: Optional[str], n: int = 4000) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _build_prompt(ticker: str, state: Mapping[str, Any]) -> list[dict]:
    debate = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}

    user = f"""Ticker: {ticker}

=== Bull case ===
{_trim(debate.get("bull_history"))}

=== Bear case ===
{_trim(debate.get("bear_history"))}

=== Research Manager decision ===
{_trim(debate.get("judge_decision"))}

=== Trader plan ===
{_trim(state.get("trader_investment_plan"))}

=== Portfolio Manager final decision ===
{_trim(risk.get("judge_decision") or state.get("final_trade_decision"))}

=== Fundamentals snapshot ===
{_trim(state.get("fundamentals_report"), 2500)}

=== Recent news ===
{_trim(state.get("news_report"), 2000)}

Produce the structured ScenarioBreakdown for {ticker}.
"""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def generate_scenario_breakdown(
    state: Mapping[str, Any],
    ticker: str,
    llm: Any,
    output_dir: Path,
) -> Optional[dict]:
    """Generate ``scenarios.json`` for the bundle. Returns the dict or None.

    ``llm`` is any LangChain chat model that supports ``with_structured_output``.
    """
    if llm is None:
        return None
    try:
        structured = llm.with_structured_output(ScenarioBreakdown)
    except Exception as exc:
        logger.info("scenarios: with_structured_output unsupported (%s); skipping", exc)
        return None

    prompt = _build_prompt(ticker, state)
    try:
        result = structured.invoke(prompt)
    except Exception as exc:
        logger.warning("scenarios: structured call failed (%s); skipping", exc)
        return None

    try:
        # Pydantic v2 — model_dump; v1 fallback to dict().
        data = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    except Exception:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "scenarios.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("scenarios: wrote %s", out_path)
    return data
