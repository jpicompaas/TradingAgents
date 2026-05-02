"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.personas import get_persona, persona_system_preamble
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.config import get_config


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]

        persona = get_persona(get_config().get("trading_persona"))
        persona_preamble = persona_system_preamble(persona)

        messages = [
            {
                "role": "system",
                "content": (
                    persona_preamble
                    + "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan.\n\n"
                    "STYLE — be concise and useful. Do not restate facts already covered in the "
                    "research plan or analyst reports. Do not pad. If you do not have a defensible "
                    "basis for a price level, leave that field unset rather than inventing a number. "
                    "Empty is better than filler. Every sentence must add a new signal a reader cannot "
                    "get from the source material.\n\n"
                    "PRICE LEVELS — populate the `levels` field with two things:\n"
                    "(a) State-change thresholds VALID ONLY UNTIL next earnings:\n"
                    "    - next_earnings_date (ISO YYYY-MM-DD if known)\n"
                    "    - accumulate_below: dip-buy zone where the thesis says 'add'\n"
                    "    - hold_zone_low / hold_zone_high: do-nothing range\n"
                    "    - trim_above: take-profit level\n"
                    "    - exit_below: stop / thesis-broken level\n"
                    "(b) Forward price estimates at +30 / +60 / +90 days from the analysis date "
                    "(`estimate_30d`, `estimate_60d`, `estimate_90d`). For each horizon, give a "
                    "low / expected / high range anchored in current price, base-rate volatility, "
                    "and known catalysts (earnings prints typically fall in the +60d window). "
                    "If you only have a point estimate, populate just `expected`.\n"
                    "Anchor every number in something specific (technical level, prior support, "
                    "valuation floor, post-event reaction). Omit any field you cannot defend."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{instrument_context}\n\n"
                    f"Research Manager's investment plan:\n{investment_plan}\n\n"
                    f"Decide the transaction (Buy/Hold/Sell), give a tight reasoning that adds "
                    f"signal beyond the plan above, and fill the `levels` block with pre-earnings "
                    f"price levels you can actually justify."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
