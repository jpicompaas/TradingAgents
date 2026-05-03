"""Spawn ``main.py`` as a subprocess and stream its stdout to the webapp.

Single-process MVP — runs are kept in an in-memory dict keyed by run_id.
A subprocess inherits the webapp's environment plus the user's
ticker/date/persona overrides so we don't have to depend on the
trading-agents graph internals from the web layer. The subprocess
writes its bundle to the same ``trading-reports/`` directory the
browser views.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class Run:
    run_id: str
    ticker: str
    trade_date: str
    persona: str
    started_at: float
    proc: subprocess.Popen
    queue: "asyncio.Queue[Optional[str]]"
    lines: list[str] = field(default_factory=list)
    status: str = "running"  # "running" | "done" | "failed"
    bundle_name: Optional[str] = None  # set when we detect "Report saved to:"


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
) -> Run:
    """Spawn ``python main.py`` with the requested env and return the Run handle."""
    run_id = secrets.token_hex(4)

    env = os.environ.copy()
    env["TRADINGAGENTS_TICKER"] = ticker
    env["TRADINGAGENTS_TRADE_DATE"] = trade_date
    if persona:
        env["TRADINGAGENTS_PERSONA"] = persona
    # PYTHONUNBUFFERED makes stdout flush immediately so the SSE stream
    # shows progress in real time rather than in a single dump at the end.
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "main.py"],
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
    """Drain the subprocess stdout into the run's queue and detect completion."""
    loop = asyncio.get_running_loop()
    assert run.proc.stdout is not None

    def _readline() -> str:
        return run.proc.stdout.readline()

    while True:
        line = await loop.run_in_executor(None, _readline)
        if not line:
            break
        line = line.rstrip("\n")
        run.lines.append(line)
        # Detect the "Report saved to: /home/.../trading-reports/<NAME>" line
        # so the UI can link to the finished bundle even though the process
        # writes paths from inside the container.
        if "Report saved to:" in line:
            tail = line.split("Report saved to:", 1)[1].strip()
            run.bundle_name = tail.rstrip("/").split("/")[-1]
        await run.queue.put(line)

    rc = await loop.run_in_executor(None, run.proc.wait)
    run.status = "done" if rc == 0 else "failed"
    await run.queue.put(None)  # sentinel


async def stream_run(run: Run) -> AsyncIterator[str]:
    """Yield SSE-formatted events for one Run.

    On reconnect, replays buffered lines first so the page can recover
    if the user reloaded mid-run.
    """
    # Replay anything we've already produced (so refreshes don't lose history).
    for line in list(run.lines):
        yield _sse("line", line)

    if run.status != "running":
        yield _sse(
            "done",
            f"status={run.status} bundle={run.bundle_name or ''}",
        )
        return

    while True:
        item = await run.queue.get()
        if item is None:
            yield _sse(
                "done",
                f"status={run.status} bundle={run.bundle_name or ''}",
            )
            return
        yield _sse("line", item)


def _sse(event: str, data: str) -> str:
    # Multi-line data must be re-prefixed; data is single-line for our usage.
    return f"event: {event}\ndata: {data}\n\n"
