"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import logging

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.personas import get_persona, persona_system_preamble
from tradingagents.agents.utils.structured import bind_structured
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        persona = get_persona(get_config().get("trading_persona"))
        persona_preamble = persona_system_preamble(persona)

        prompt = f"""{persona_preamble}As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

**Style — concise and useful, no slop.** Do not restate facts the reader can find
in the research plan, the trader proposal, or the risk debate above. The
executive summary is at most three sentences and the investment thesis adds
new synthesis or judgment, not a recap. If you have nothing concrete to add to
a section, say less. Empty is better than filler — every sentence must add a
signal a reader cannot get from the source material.

**Price levels — populate the `levels` field** with TWO required pieces:

**(1) State-change thresholds** valid only until next earnings:
- `next_earnings_date` (ISO YYYY-MM-DD if known; leave None if not)
- `accumulate_below`: add-to-position dip-buy zone
- `hold_zone_low` / `hold_zone_high`: no-action range
- `trim_above`: take-profit level
- `exit_below`: stop / thesis-broken level

**(2) Forward price estimates — ALWAYS POPULATE** at +30, +60, and +90
calendar days from the analysis date. The Trader and the downstream
forecast curve depend on these.
- For each horizon (`estimate_30d`, `estimate_60d`, `estimate_90d`),
  populate `expected` (the most-likely close at that horizon) anchored in:
  the current close from the Market Analyst report, the directional
  conviction of your rating, recent realized volatility, and any known
  catalysts (the +60d window typically spans the next earnings print —
  widen the range there).
- Also populate `low` (bearish-scenario close) and `high` (bullish-scenario
  close). These bracket your expected: ±0.5 to ±1.5 standard deviations
  is a reasonable range, sized by the recent realized volatility.
- These estimates are decision-grade approximations, not predictions —
  your job is to give a falsifiable forward view a reader can track
  against. A ±10% expected for a typical equity 90 days out, with a
  ±20–30% bull/bear range, is well within reason. **Do not leave the
  expected fields unset.**

State-change levels (group 1) and `next_earnings_date` should be left
unset only when the analyst reports give you no defensible basis.

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        # Run structured output and capture both the rendered markdown AND
        # the parsed object so downstream code (forecast generation in
        # save_run_bundle) can read the structured levels without re-parsing
        # markdown. Mirrors invoke_structured_or_freetext but keeps the obj.
        decision_obj = None
        final_trade_decision = None
        if structured_llm is not None:
            try:
                decision_obj = structured_llm.invoke(prompt)
                final_trade_decision = render_pm_decision(decision_obj)
            except Exception as exc:
                logger.warning(
                    "Portfolio Manager: structured-output invocation failed (%s); "
                    "retrying once as free text",
                    exc,
                )
        if final_trade_decision is None:
            final_trade_decision = llm.invoke(prompt).content

        pm_levels_dict = None
        if decision_obj is not None and decision_obj.levels is not None:
            pm_levels_dict = decision_obj.levels.model_dump()

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "pm_levels": pm_levels_dict,
        }

    return portfolio_manager_node
