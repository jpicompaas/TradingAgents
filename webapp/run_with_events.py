"""Subprocess entry that runs the trading-agents graph and emits structured events.

This is the webapp's equivalent of ``main.py``. The webapp spawns
``python -m webapp.run_with_events`` instead of ``main.py`` so the
browser can render the same kind of structured progress display the
terminal CLI shows (per-agent status, current report, stats) — not
just a stdout dump.

Output protocol (one line per record on stdout):

    EVENT|<json>          structured snapshot — see ``_emit_state``
    <anything else>       treated as a log line by the webapp

Reuses ``MessageBuffer`` and the chunk-handling helpers from
``cli.main`` so the progress logic stays in one place.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# .env must be loaded BEFORE importing default_config (it reads
# TRADINGAGENTS_PERSONA at module import time).
load_dotenv()
load_dotenv(".env.enterprise", override=False)

from cli.main import (  # noqa: E402
    ANALYST_ORDER,
    MessageBuffer,
    classify_message_type,
    update_analyst_statuses,
    update_research_team_status,
)
from cli.stats_handler import StatsCallbackHandler  # noqa: E402
from tradingagents.agents.utils.personas import get_persona  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402
from tradingagents.reports import save_run_bundle  # noqa: E402


# How many recent messages / tool calls to include in each snapshot.
# Bound the payload so a long run doesn't grow unbounded SSE frames.
MAX_RECENT_MESSAGES = 30
MAX_RECENT_TOOLS = 30
# Truncate per-section previews so a 5KB report doesn't get dumped on
# every chunk; the final state is sent in full at the end.
PREVIEW_CHARS = 1200


def _emit(event: str, payload: dict) -> None:
    """Emit a single structured record. Prefix is parsed by webapp/runs.py."""
    line = f"EVENT|{json.dumps({'event': event, **payload}, default=str)}"
    print(line, flush=True)


def _truncate(text: str, n: int = PREVIEW_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _snapshot(
    buf: MessageBuffer,
    stats: StatsCallbackHandler,
    started_at: float,
) -> dict:
    """Build a state snapshot to ship to the browser."""
    msgs = list(buf.messages)[-MAX_RECENT_MESSAGES:]
    tools = list(buf.tool_calls)[-MAX_RECENT_TOOLS:]
    sections = {
        name: _truncate(content) if isinstance(content, str) else content
        for name, content in buf.report_sections.items()
        if content
    }
    return {
        "agent_status": dict(buf.agent_status),
        "current_agent": buf.current_agent,
        "report_sections": sections,
        "messages": [
            {"ts": ts, "type": t, "content": _truncate(c, 400)}
            for ts, t, c in msgs
        ],
        "tool_calls": [
            {"ts": ts, "name": name, "args": _truncate(str(args), 200)}
            for ts, name, args in tools
        ],
        "stats": stats.get_stats(),
        "elapsed": int(time.time() - started_at),
        "persona_label": buf.persona_label,
    }


def main() -> int:
    ticker = os.environ.get("TRADINGAGENTS_TICKER", "AMZN").strip().upper()
    trade_date = os.environ.get("TRADINGAGENTS_TRADE_DATE", "2026-05-03").strip()

    config = DEFAULT_CONFIG.copy()
    # Mirror main.py defaults — Groq stays the cheap clone-and-run path.
    config["llm_provider"] = os.environ.get(
        "TRADINGAGENTS_LLM_PROVIDER", config.get("llm_provider", "groq")
    ).lower()
    config["max_debate_rounds"] = int(os.environ.get("TRADINGAGENTS_DEBATE_ROUNDS", 1))

    selected_analysts = ["market", "social", "news", "fundamentals"]
    stats = StatsCallbackHandler()

    graph = TradingAgentsGraph(
        selected_analysts,
        config=config,
        debug=True,
        callbacks=[stats],
    )

    buf = MessageBuffer()
    buf.init_for_analysis(selected_analysts)
    persona = get_persona(config.get("trading_persona"))
    buf.persona_label = persona.display_name if persona else ""

    started_at = time.time()
    _emit(
        "start",
        {
            "ticker": ticker,
            "trade_date": trade_date,
            "persona": config.get("trading_persona") or "",
            "persona_label": buf.persona_label,
            "selected_analysts": selected_analysts,
            "agent_status": dict(buf.agent_status),
        },
    )

    # Drive the same loop as cli/main.py's run_analysis, minus the Rich layout.
    init_state = graph.propagator.create_initial_state(ticker, trade_date)
    args = graph.propagator.get_graph_args(callbacks=[stats])

    # Kick off with the first analyst marked in_progress for a non-empty
    # initial UI (matches CLI behavior).
    if selected_analysts:
        first = f"{selected_analysts[0].capitalize()} Analyst"
        buf.update_agent_status(first, "in_progress")
        _emit("state", _snapshot(buf, stats, started_at))

    trace = []
    last_emit = 0.0
    EMIT_INTERVAL = 0.5  # don't flood the browser; coalesce updates

    for chunk in graph.graph.stream(init_state, **args):
        # ---- mirror cli/main.py chunk handling --------------------------------
        for message in chunk.get("messages", []):
            msg_id = getattr(message, "id", None)
            if msg_id is not None:
                if msg_id in buf._processed_message_ids:
                    continue
                buf._processed_message_ids.add(msg_id)

            msg_type, content = classify_message_type(message)
            if content and content.strip():
                buf.add_message(msg_type, content)

            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    if isinstance(tc, dict):
                        buf.add_tool_call(tc["name"], tc["args"])
                    else:
                        buf.add_tool_call(tc.name, tc.args)

        update_analyst_statuses(buf, chunk)

        if chunk.get("investment_debate_state"):
            ds = chunk["investment_debate_state"]
            bull = (ds.get("bull_history") or "").strip()
            bear = (ds.get("bear_history") or "").strip()
            judge = (ds.get("judge_decision") or "").strip()
            if bull or bear:
                update_research_team_status("in_progress")
            if bull:
                buf.update_report_section(
                    "investment_plan", f"### Bull Researcher Analysis\n{bull}"
                )
            if bear:
                buf.update_report_section(
                    "investment_plan", f"### Bear Researcher Analysis\n{bear}"
                )
            if judge:
                buf.update_report_section(
                    "investment_plan", f"### Research Manager Decision\n{judge}"
                )
                update_research_team_status("completed")
                buf.update_agent_status("Trader", "in_progress")

        if chunk.get("trader_investment_plan"):
            buf.update_report_section(
                "trader_investment_plan", chunk["trader_investment_plan"]
            )
            if buf.agent_status.get("Trader") != "completed":
                buf.update_agent_status("Trader", "completed")
                buf.update_agent_status("Aggressive Analyst", "in_progress")

        if chunk.get("risk_debate_state"):
            rs = chunk["risk_debate_state"]
            agg = (rs.get("aggressive_history") or "").strip()
            con = (rs.get("conservative_history") or "").strip()
            neu = (rs.get("neutral_history") or "").strip()
            judge = (rs.get("judge_decision") or "").strip()
            if agg:
                if buf.agent_status.get("Aggressive Analyst") != "completed":
                    buf.update_agent_status("Aggressive Analyst", "in_progress")
                buf.update_report_section(
                    "final_trade_decision",
                    f"### Aggressive Analyst Analysis\n{agg}",
                )
            if con:
                if buf.agent_status.get("Conservative Analyst") != "completed":
                    buf.update_agent_status("Conservative Analyst", "in_progress")
                buf.update_report_section(
                    "final_trade_decision",
                    f"### Conservative Analyst Analysis\n{con}",
                )
            if neu:
                if buf.agent_status.get("Neutral Analyst") != "completed":
                    buf.update_agent_status("Neutral Analyst", "in_progress")
                buf.update_report_section(
                    "final_trade_decision",
                    f"### Neutral Analyst Analysis\n{neu}",
                )
            if judge:
                if buf.agent_status.get("Portfolio Manager") != "completed":
                    buf.update_agent_status("Portfolio Manager", "in_progress")
                    buf.update_report_section(
                        "final_trade_decision",
                        f"### Portfolio Manager Decision\n{judge}",
                    )
                    buf.update_agent_status("Aggressive Analyst", "completed")
                    buf.update_agent_status("Conservative Analyst", "completed")
                    buf.update_agent_status("Neutral Analyst", "completed")
                    buf.update_agent_status("Portfolio Manager", "completed")

        # Coalesce snapshots — graph.stream chunks can arrive faster than
        # the browser needs them.
        now = time.time()
        if now - last_emit >= EMIT_INTERVAL:
            _emit("state", _snapshot(buf, stats, started_at))
            last_emit = now
        trace.append(chunk)

    final_state = trace[-1] if trace else {}

    # Mark every agent completed and pull final report sections from state.
    for agent in buf.agent_status:
        buf.update_agent_status(agent, "completed")
    for section in buf.report_sections.keys():
        if section in final_state:
            buf.update_report_section(section, final_state[section])

    _emit("state", _snapshot(buf, stats, started_at))

    # Persist bundle and tell the webapp where it landed.
    bundle_dir = save_run_bundle(final_state, ticker)
    bundle_name = bundle_dir.name
    print(f"Report saved to: {bundle_dir.resolve()}", flush=True)

    # Per-ticker scenario breakdown with explicit probabilities. Best-effort
    # — if it fails the bundle still has the forecast curves; the webapp
    # forecast view degrades gracefully.
    try:
        from tradingagents.scenarios import generate_scenario_breakdown

        scen = generate_scenario_breakdown(
            final_state, ticker, graph.quick_thinking_llm, bundle_dir
        )
        if scen:
            print(f"Scenario breakdown written to scenarios.json", flush=True)
    except Exception as exc:
        print(f"Scenario breakdown skipped: {exc}", flush=True)

    # Push to R2 if configured. Synchronous so the "View full report" link
    # the browser shows next is guaranteed to resolve.
    from webapp import storage  # local import to keep the cli/main.py import path light
    if storage.is_enabled():
        try:
            uploaded, _ = storage.put_bundle_dir(bundle_dir, bundle_name)
            print(f"R2 uploaded {uploaded} files to reports/{bundle_name}/", flush=True)
        except Exception as exc:
            print(f"R2 upload failed: {exc}", flush=True)

    _emit(
        "done",
        {
            "bundle_name": bundle_name,
            "elapsed": int(time.time() - started_at),
            "stats": stats.get_stats(),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _emit("done", {"bundle_name": "", "cancelled": True})
        sys.exit(130)
    except Exception as exc:  # surface failures to the browser
        _emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        raise
