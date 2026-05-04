"""Spawn the trading-agents graph runner and pump structured events to the webapp.

Each ``Run`` owns a subprocess (``python -m webapp.run_with_events``) and an
asyncio queue of events. The runner emits two flavors of stdout:

    EVENT|<json>      — structured snapshot consumed by the live progress UI
    <other text>      — captured as a log line (Groq retries, "Report saved
                        to: ..." line, traceback frames, etc)

Both flavors are forwarded to subscribers as SSE events so the browser can
render either or both.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


# Ship at most this many recent log lines back to a late-joining client to
# avoid blowing up the SSE replay payload on long runs.
MAX_REPLAY_LINES = 400


@dataclass
class Run:
    run_id: str
    ticker: str
    trade_date: str
    persona: str
    started_at: float
    proc: subprocess.Popen
    queue: "asyncio.Queue[Optional[dict]]"
    lines: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # ordered raw events
    state: dict = field(default_factory=dict)         # latest structured snapshot
    status: str = "running"                           # running | done | failed | cancelled
    bundle_name: Optional[str] = None
    error: Optional[str] = None


_RUNS: dict[str, Run] = {}


def get_run(run_id: str) -> Optional[Run]:
    return _RUNS.get(run_id)


def list_runs(limit: int = 20) -> list[Run]:
    items = sorted(_RUNS.values(), key=lambda r: r.started_at, reverse=True)
    return items[:limit]


async def start_run(
    ticker: str,
    trade_date: str,
    persona: str = "",
    *,
    cwd: str = ".",
    extra_env: Optional[dict[str, str]] = None,
) -> Run:
    """Spawn the runner subprocess with the requested env and return the Run handle.

    ``extra_env`` lets the caller forward form-supplied overrides
    (model selection, analyst list, debate rounds, ...) without the
    runner having to know about every form field shape.
    """
    run_id = secrets.token_hex(4)

    env = os.environ.copy()
    env["TRADINGAGENTS_TICKER"] = ticker
    env["TRADINGAGENTS_TRADE_DATE"] = trade_date
    if persona:
        env["TRADINGAGENTS_PERSONA"] = persona
    if extra_env:
        # Skip empty values so blanks don't clobber DEFAULT_CONFIG.
        env.update({k: v for k, v in extra_env.items() if v})
    # PYTHONUNBUFFERED makes stdout flush immediately so the SSE stream
    # shows progress in real time rather than in a single dump at the end.
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "webapp.run_with_events"],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )

    run = Run(
        run_id=run_id,
        ticker=ticker,
        trade_date=trade_date,
        persona=persona,
        started_at=time.time(),
        proc=proc,
        queue=asyncio.Queue(),
    )
    _RUNS[run_id] = run
    asyncio.create_task(_pump_output(run))
    return run


async def _pump_output(run: Run) -> None:
    """Drain subprocess stdout, classify each line, and broadcast events."""
    loop = asyncio.get_running_loop()
    assert run.proc.stdout is not None

    def _readline() -> str:
        return run.proc.stdout.readline()

    while True:
        line = await loop.run_in_executor(None, _readline)
        if not line:
            break
        line = line.rstrip("\n")

        if line.startswith("EVENT|"):
            payload = line[len("EVENT|"):]
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                # Malformed event — fall through and treat as a log line.
                run.lines.append(line)
                await run.queue.put({"kind": "log", "data": line})
                continue
            _apply_event(run, evt)
            run.events.append(evt)
            await run.queue.put({"kind": "event", "data": evt})
        else:
            run.lines.append(line)
            # Belt-and-suspenders: also pick the bundle name out of the
            # human-readable line in case the EVENT|done was missed.
            if "Report saved to:" in line and not run.bundle_name:
                tail = line.split("Report saved to:", 1)[1].strip()
                run.bundle_name = tail.rstrip("/").split("/")[-1]
            await run.queue.put({"kind": "log", "data": line})

    rc = await loop.run_in_executor(None, run.proc.wait)
    if run.status == "running":  # not already cancelled / errored
        run.status = "done" if rc == 0 else "failed"
    await run.queue.put(None)  # sentinel — close the stream


def _apply_event(run: Run, evt: dict) -> None:
    """Update the Run's mirrored state from a structured event."""
    kind = evt.get("event")
    if kind == "start":
        run.state = {
            "agent_status": evt.get("agent_status", {}),
            "report_sections": {},
            "messages": [],
            "tool_calls": [],
            "stats": {"llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0},
            "elapsed": 0,
            "persona_label": evt.get("persona_label", ""),
            "selected_analysts": evt.get("selected_analysts", []),
        }
    elif kind == "state":
        # Replace fields wholesale — the runner sends a fresh full snapshot.
        for key in (
            "agent_status",
            "report_sections",
            "messages",
            "tool_calls",
            "stats",
            "elapsed",
            "persona_label",
            "current_agent",
        ):
            if key in evt:
                run.state[key] = evt[key]
    elif kind == "done":
        if evt.get("bundle_name"):
            run.bundle_name = evt["bundle_name"]
        if "stats" in evt:
            run.state["stats"] = evt["stats"]
        if evt.get("cancelled"):
            run.status = "cancelled"
    elif kind == "error":
        run.error = evt.get("message")


def cancel_run(run: Run) -> bool:
    """Send SIGTERM to a running subprocess. Returns True if we sent it."""
    if run.status != "running" or run.proc.poll() is not None:
        return False
    try:
        run.proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return False
    run.status = "cancelled"
    return True


async def stream_run(run: Run) -> AsyncIterator[str]:
    """Yield SSE-formatted events for one Run.

    On (re)connect, replays buffered state + recent lines so a refresh
    mid-run doesn't lose history.
    """
    # Replay current state first so the page paints immediately.
    if run.state:
        yield _sse("state", json.dumps(run.state))
    for line in run.lines[-MAX_REPLAY_LINES:]:
        yield _sse("log", line)

    if run.status != "running":
        yield _sse(
            "done",
            json.dumps(
                {
                    "status": run.status,
                    "bundle_name": run.bundle_name or "",
                    "error": run.error,
                }
            ),
        )
        return

    while True:
        item = await run.queue.get()
        if item is None:
            yield _sse(
                "done",
                json.dumps(
                    {
                        "status": run.status,
                        "bundle_name": run.bundle_name or "",
                        "error": run.error,
                    }
                ),
            )
            return
        if item["kind"] == "log":
            yield _sse("log", item["data"])
        else:
            yield _sse("state", json.dumps(item["data"]))


def snapshot(run: Run) -> dict[str, Any]:
    """Plain-dict view of a run for JSON endpoints."""
    return {
        "run_id": run.run_id,
        "ticker": run.ticker,
        "trade_date": run.trade_date,
        "persona": run.persona,
        "started_at": run.started_at,
        "status": run.status,
        "bundle_name": run.bundle_name,
        "error": run.error,
        "state": run.state,
    }


def _sse(event: str, data: str) -> str:
    # SSE requires multi-line payloads to be re-prefixed with "data:" on
    # each line; our state JSON is single-line so a single "data:" works.
    return f"event: {event}\ndata: {data}\n\n"
