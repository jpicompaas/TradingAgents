"""TradingAgents webapp — live runs and report browser.

Two screens:

- ``/run`` — kick off a fresh analysis (ticker, date, persona) and
  watch the streaming output, same content the terminal CLI shows.
- ``/reports`` — browse the persisted ``trading-reports/`` tree by
  ticker. Each ticker page has tabs (one per persona / "neutral") and
  a dropdown to flip between past runs of that persona.

Single FastAPI app, single Docker image. SSE for the live stream;
markdown rendered server-side for the report viewer.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import markdown as md
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp import reports as reports_mod
from webapp import runs as runs_mod
from tradingagents.agents.utils.personas import list_personas


HERE = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))

REPORTS_ROOT = Path(os.environ.get("TRADINGAGENTS_REPORTS_ROOT", "trading-reports"))

app = FastAPI(title="TradingAgents")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
# Bundle assets (forecast.png etc) served directly from the reports tree.
if REPORTS_ROOT.is_dir():
    app.mount(
        "/bundles",
        StaticFiles(directory=str(REPORTS_ROOT)),
        name="bundles",
    )


# ---------------------------------------------------------------------------
# Markdown rendering — rewrite relative image / link paths so the rendered
# HTML can find ``forecast.png`` etc under ``/bundles/<bundle_name>/...``.
# ---------------------------------------------------------------------------

_MD = md.Markdown(extensions=["tables", "fenced_code", "sane_lists"])


def render_bundle_md(text: str, bundle_name: str) -> str:
    """Render a bundle's markdown, rewriting relative URLs to /bundles/<name>/..."""
    if not text:
        return ""
    # Rewrite image src and link href targets that don't start with http/https/#/. /
    def rewrite(match: re.Match) -> str:
        prefix, target = match.group(1), match.group(2)
        if (
            target.startswith(("http://", "https://", "/", "#", "mailto:"))
        ):
            return match.group(0)
        return f"{prefix}/bundles/{bundle_name}/{target}"

    rewritten = re.sub(r"(\!\[[^\]]*\]\()([^)\s]+)", rewrite, text)
    rewritten = re.sub(r"(\[[^\]]*\]\()([^)\s]+)", rewrite, rewritten)
    _MD.reset()
    return _MD.convert(rewritten)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    tickers = reports_mod.list_tickers(REPORTS_ROOT)
    runs = runs_mod.list_runs()
    return TEMPLATES.TemplateResponse(
        request,
        "home.html",
        {"tickers": tickers, "runs": runs},
    )


# ---------- Run a new analysis -----------------------------------------------


@app.get("/run", response_class=HTMLResponse)
def run_form(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "run_form.html",
        {
            "personas": list_personas(),
            "default_persona": os.environ.get("TRADINGAGENTS_PERSONA", ""),
            "default_ticker": "MSFT",
            "default_date": os.environ.get("TRADINGAGENTS_TRADE_DATE", "2026-05-03"),
        },
    )


@app.post("/run")
async def run_start(
    ticker: str = Form(...),
    trade_date: str = Form(...),
    persona: str = Form(""),
):
    ticker = ticker.strip().upper()
    trade_date = trade_date.strip()
    persona = persona.strip()
    if not ticker or not trade_date:
        raise HTTPException(400, "ticker and trade_date are required")
    run = await runs_mod.start_run(ticker, trade_date, persona)
    return RedirectResponse(f"/run/{run.run_id}", status_code=303)


@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_view(request: Request, run_id: str):
    run = runs_mod.get_run(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return TEMPLATES.TemplateResponse(
        request,
        "run_view.html",
        {"run": run},
    )


@app.get("/run/{run_id}/stream")
async def run_stream(run_id: str):
    run = runs_mod.get_run(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return StreamingResponse(
        runs_mod.stream_run(run),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- Browse reports ---------------------------------------------------


@app.get("/reports", response_class=HTMLResponse)
def reports_index(request: Request):
    bundles = reports_mod.list_bundles(REPORTS_ROOT)
    tickers = sorted({b.ticker for b in bundles})
    # Latest bundle per ticker for the index card
    latest_by_ticker: dict[str, reports_mod.Bundle] = {}
    for b in bundles:
        latest_by_ticker.setdefault(b.ticker, b)
    return TEMPLATES.TemplateResponse(
        request,
        "reports_index.html",
        {
            "tickers": tickers,
            "latest": latest_by_ticker,
        },
    )


@app.get("/reports/{ticker}", response_class=HTMLResponse)
def reports_ticker(
    request: Request,
    ticker: str,
    persona: Optional[str] = None,
    run: Optional[str] = None,
):
    view = reports_mod.get_ticker_view(ticker, REPORTS_ROOT)
    if not view.by_persona:
        raise HTTPException(404, f"no reports found for {ticker}")

    personas = view.personas
    active_persona = persona if (persona in personas) else personas[0]

    runs_for_persona = view.by_persona.get(active_persona) or []
    if run:
        active_bundle = next((b for b in runs_for_persona if b.name == run), None)
    else:
        active_bundle = runs_for_persona[0] if runs_for_persona else None

    rendered = ""
    section_files: dict[str, list] = {}
    if active_bundle is not None:
        text = reports_mod.read_complete_report(active_bundle)
        rendered = render_bundle_md(text, active_bundle.name)
        section_files = reports_mod.list_section_files(active_bundle)

    return TEMPLATES.TemplateResponse(
        request,
        "ticker_view.html",
        {
            "ticker": ticker,
            "view": view,
            "personas": personas,
            "active_persona": active_persona,
            "active_bundle": active_bundle,
            "runs_for_persona": runs_for_persona,
            "rendered_html": rendered,
            "section_files": section_files,
        },
    )


@app.get("/reports/{ticker}/{run}/section/{subdir}/{filename}", response_class=HTMLResponse)
def section_file(request: Request, ticker: str, run: str, subdir: str, filename: str):
    bundle = reports_mod.get_bundle(ticker, run, REPORTS_ROOT)
    if bundle is None:
        raise HTTPException(404, "no such run")
    target = bundle.path / subdir / filename
    if not target.is_file() or target.suffix != ".md":
        raise HTTPException(404)
    rendered = render_bundle_md(target.read_text(encoding="utf-8"), bundle.name)
    return TEMPLATES.TemplateResponse(
        request,
        "section_view.html",
        {
            "ticker": ticker,
            "bundle": bundle,
            "title": f"{subdir} / {filename}",
            "rendered_html": rendered,
        },
    )


# Healthcheck for compose --------------------------------------------------


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"
