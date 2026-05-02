"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Pydantic 2.13 strict-validates ``(str, Enum)`` types in some langchain
# structured-output paths (rejecting bare strings with `is_instance_of`).
# Typing the rating/action fields as ``Literal[...]`` keeps the same JSON
# schema for the model but avoids that validator class entirely. The Enum
# classes below are kept as named constants for any non-schema callers.

PortfolioRatingValue = Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]
TraderActionValue = Literal["Buy", "Hold", "Sell"]


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """


class HorizonEstimate(BaseModel):
    """Point/range price estimate over a fixed time horizon.

    All fields optional. If the model only has a single point estimate,
    populate ``expected``. If it has a low/high range as well, populate
    those. Inventing values to fill the schema is worse than omitting.
    """

    low: Optional[float] = Field(
        default=None,
        description="Bearish-scenario price for this horizon; conservative downside.",
    )
    expected: Optional[float] = Field(
        default=None,
        description="Most-likely price at the end of this horizon.",
    )
    high: Optional[float] = Field(
        default=None,
        description="Bullish-scenario price for this horizon; reasonable upside.",
    )


class LevelEstimates(BaseModel):
    """Pre-earnings price levels and forward price estimates.

    Two purposes in one block:

    1. **Levels at which the thesis changes state** (accumulate / hold / trim
       / exit), valid only until ``next_earnings_date``.
    2. **Forward price estimates** at 30, 60, and 90 days from the analysis
       date — anchor the thesis to concrete numbers a reader can track.

    All fields are optional. **Leave any field unset** if you do not have a
    defensible basis from the analyst reports — inventing numbers is worse
    than omitting them.
    """

    next_earnings_date: Optional[str] = Field(
        default=None,
        description=(
            "ISO date (YYYY-MM-DD) of the next scheduled earnings event, if "
            "known from the analyst reports. State-change levels below expire "
            "on this date."
        ),
    )
    accumulate_below: Optional[float] = Field(
        default=None,
        description="Add-to-position level: dip-buy zone. Price under which the thesis says accumulate.",
    )
    hold_zone_low: Optional[float] = Field(
        default=None,
        description="Lower bound of the no-action range — between this and hold_zone_high, simply hold.",
    )
    hold_zone_high: Optional[float] = Field(
        default=None,
        description="Upper bound of the no-action range — between hold_zone_low and this, simply hold.",
    )
    trim_above: Optional[float] = Field(
        default=None,
        description="Take-profit level: price above which to trim or take partial profits.",
    )
    exit_below: Optional[float] = Field(
        default=None,
        description=(
            "Stop / thesis-broken level. If price closes below this, the bull "
            "thesis has failed and the position should be exited."
        ),
    )

    estimate_30d: Optional[HorizonEstimate] = Field(
        default=None,
        description=(
            "Forward price estimate ~30 calendar days from the analysis "
            "date (low / expected / high). Anchor in current technical setup, "
            "near-term catalysts, and base-rate volatility."
        ),
    )
    estimate_60d: Optional[HorizonEstimate] = Field(
        default=None,
        description=(
            "Forward price estimate ~60 calendar days from the analysis date. "
            "Typically spans the next earnings window — reflect that in the range."
        ),
    )
    estimate_90d: Optional[HorizonEstimate] = Field(
        default=None,
        description=(
            "Forward price estimate ~90 calendar days from the analysis date. "
            "Reflects the post-earnings reaction band."
        ),
    )


def _format_level(prefix: str, value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{prefix}{value:g}"


def _format_horizon(label: str, est: Optional[HorizonEstimate]) -> Optional[str]:
    """One bullet line for a single horizon, e.g. '+30d: 280 (range 260–305)'.

    Omits parts that are unset; returns None if nothing useful to show.
    """
    if est is None:
        return None
    parts: list[str] = []
    if est.expected is not None:
        parts.append(f"{est.expected:g}")
    if est.low is not None and est.high is not None:
        parts.append(f"range {est.low:g}–{est.high:g}")
    elif est.low is not None:
        parts.append(f"low {est.low:g}")
    elif est.high is not None:
        parts.append(f"high {est.high:g}")
    if not parts:
        return None
    return f"{label}: " + " (".join(parts) + (")" if len(parts) > 1 else "")


def render_level_estimates(levels: Optional[LevelEstimates]) -> str:
    """Render a LevelEstimates block, or empty string if nothing to show.

    Output is a single compact bulleted block. Any field left None is
    omitted entirely — empty is better than filler.
    """
    if levels is None:
        return ""
    lines: list[str] = []

    # State-change levels (valid until earnings)
    if levels.accumulate_below is not None:
        lines.append(_format_level("Accumulate below ", levels.accumulate_below))
    if levels.hold_zone_low is not None and levels.hold_zone_high is not None:
        lines.append(f"Hold {levels.hold_zone_low:g} – {levels.hold_zone_high:g}")
    elif levels.hold_zone_low is not None:
        lines.append(_format_level("Hold above ", levels.hold_zone_low))
    elif levels.hold_zone_high is not None:
        lines.append(_format_level("Hold below ", levels.hold_zone_high))
    if levels.trim_above is not None:
        lines.append(_format_level("Trim above ", levels.trim_above))
    if levels.exit_below is not None:
        lines.append(_format_level("Exit below ", levels.exit_below))

    horizon_lines: list[str] = []
    for label, est in (
        ("+30d", levels.estimate_30d),
        ("+60d", levels.estimate_60d),
        ("+90d", levels.estimate_90d),
    ):
        line = _format_horizon(label, est)
        if line:
            horizon_lines.append(line)

    if not lines and not horizon_lines:
        return ""

    blocks: list[str] = []
    if lines:
        suffix = (
            f" (valid until next earnings: {levels.next_earnings_date})"
            if levels.next_earnings_date
            else " (valid until next earnings)"
        )
        blocks.append("**Levels**" + suffix + ":\n- " + "\n- ".join(lines))
    if horizon_lines:
        blocks.append("**Forward price estimates**:\n- " + "\n- ".join(horizon_lines))

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRatingValue = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderActionValue = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )
    levels: Optional[LevelEstimates] = Field(
        default=None,
        description=(
            "Pre-earnings price levels at which the thesis changes state "
            "(accumulate / hold / trim / exit), valid until the next earnings "
            "event. Leave None or omit individual fields if no defensible "
            "basis exists in the analyst reports."
        ),
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    levels_md = render_level_estimates(proposal.levels)
    if levels_md:
        parts.extend(["", levels_md])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRatingValue = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    levels: Optional[LevelEstimates] = Field(
        default=None,
        description=(
            "Pre-earnings price levels at which the position changes state "
            "(accumulate / hold / trim / exit), valid until the next earnings "
            "event. Leave None or omit individual fields if no defensible "
            "basis exists in the analyst reports."
        ),
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    levels_md = render_level_estimates(decision.levels)
    if levels_md:
        parts.extend(["", levels_md])
    return "\n".join(parts)
