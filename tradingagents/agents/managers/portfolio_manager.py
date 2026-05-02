"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.personas import get_persona, persona_system_preamble
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.config import get_config


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

**Price levels — populate the `levels` field** with two complementary pieces:

1. **State-change thresholds** valid only until next earnings:
   - `next_earnings_date` (ISO YYYY-MM-DD if known)
   - `accumulate_below`: add-to-position dip-buy zone
   - `hold_zone_low` / `hold_zone_high`: no-action range
   - `trim_above`: take-profit level
   - `exit_below`: stop / thesis-broken level

2. **Forward price estimates** at +30, +60, and +90 calendar days from the
   analysis date (`estimate_30d`, `estimate_60d`, `estimate_90d`). For each
   horizon, give a low / expected / high range anchored in current price,
   base-rate volatility, and known catalysts. The +60d horizon typically
   spans the next earnings print — reflect that uncertainty in the range.
   If you only have a point estimate, populate just `expected`.

Anchor every number in something specific (technical level, prior support,
valuation floor, post-event reaction). **Leave any field unset if you cannot
defend it** — inventing numbers to fill the schema is worse than omitting them.

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

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
        }

    return portfolio_manager_node
