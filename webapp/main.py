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

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Optional

import markdown as md
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp import reports as reports_mod
from webapp import runs as runs_mod
from webapp import storage as storage_mod
from tradingagents.agents.utils.personas import list_personas


HERE = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))

REPORTS_ROOT = Path(os.environ.get("TRADINGAGENTS_REPORTS_ROOT", "trading-reports"))

app = FastAPI(title="TradingAgents")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
# Note: /bundles/* is served by ``bundle_asset`` below — it streams from R2
# when configured, falling back to local disk. We can't use StaticFiles
# because the canonical store is remote.


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
            "default_ticker": os.environ.get("TRADINGAGENTS_DEFAULT_TICKER", "AMZN"),
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


@app.get("/run/{run_id}/state")
def run_state(run_id: str):
    """JSON view of a run's current state — useful for clients that don't SSE."""
    run = runs_mod.get_run(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    return JSONResponse(runs_mod.snapshot(run))


@app.post("/run/{run_id}/cancel")
def run_cancel(run_id: str):
    run = runs_mod.get_run(run_id)
    if run is None:
        raise HTTPException(404, "no such run")
    sent = runs_mod.cancel_run(run)
    return JSONResponse({"cancelled": sent, "status": run.status})


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
    if persona in personas:
        active_persona = persona
    else:
        # Default to the persona of the most-recent run for this ticker
        # rather than alphabetically first — opening /reports/MSFT should
        # land on the latest analysis the user actually ran.
        latest = max(
            (lst[0] for lst in view.by_persona.values() if lst),
            key=lambda b: b.timestamp,
            default=None,
        )
        active_persona = latest.persona_label if latest else personas[0]

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
    if not filename.endswith(".md") or "/" in subdir or "/" in filename:
        raise HTTPException(404)
    text = reports_mod.read_text(bundle, f"{subdir}/{filename}")
    if not text:
        raise HTTPException(404)
    rendered = render_bundle_md(text, bundle.name)
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


# ---------- Forecast viewer --------------------------------------------------


def _parse_forecast_csv(text: str) -> tuple[list[dict], list[str]]:
    """Return (rows, columns) parsed from forecast.csv text. Columns preserves order."""
    if not text:
        return [], []
    reader = csv.DictReader(io.StringIO(text))
    rows = [dict(r) for r in reader]
    columns = reader.fieldnames or []
    # Cast numeric columns to float so the template can format them.
    for r in rows:
        for k, v in list(r.items()):
            if k in ("day", "date") or v in (None, ""):
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows, list(columns)


def _build_chart_svg(rows: list[dict], width: int = 760, height: int = 320) -> str:
    """Tiny dependency-free SVG line chart for the three forecast paths."""
    if not rows:
        return ""
    keys = ("headwinds", "same", "tailwinds")
    color = {"headwinds": "#cc4444", "same": "#9b9b9b", "tailwinds": "#229922"}

    points: dict[str, list[tuple[float, float]]] = {k: [] for k in keys}
    bands: dict[str, list[tuple[float, float, float]]] = {k: [] for k in keys}
    for r in rows:
        try:
            d = int(r["day"])
        except Exception:
            continue
        for k in keys:
            v = r.get(k)
            if isinstance(v, (int, float)):
                points[k].append((d, v))
            lo, hi = r.get(f"{k}_low"), r.get(f"{k}_high")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                bands[k].append((d, lo, hi))

    all_y = [y for series in points.values() for _, y in series] + [
        b for series in bands.values() for _, lo, hi in series for b in (lo, hi)
    ]
    if not all_y:
        return ""

    days_max = max(d for series in points.values() for d, _ in series)
    y_min = min(all_y) * 0.985
    y_max = max(all_y) * 1.015
    padding_l, padding_r, padding_t, padding_b = 56, 16, 14, 30
    plot_w = width - padding_l - padding_r
    plot_h = height - padding_t - padding_b

    def sx(d: float) -> float:
        return padding_l + (d / max(days_max, 1)) * plot_w

    def sy(y: float) -> float:
        return padding_t + (1 - (y - y_min) / max(y_max - y_min, 1e-9)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'class="forecast-chart" role="img" aria-label="Forecast chart">'
    ]
    # Background
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0e1117"/>'
    )
    # Y-axis ticks (5 lines)
    for i in range(5):
        v = y_min + (y_max - y_min) * i / 4
        y = sy(v)
        parts.append(
            f'<line x1="{padding_l}" x2="{width - padding_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="#30363d" stroke-width="0.5"/>'
            f'<text x="{padding_l - 6}" y="{y + 3:.1f}" fill="#8b949e" font-size="10" '
            f'text-anchor="end">{v:.0f}</text>'
        )
    # X-axis day labels (every ~15)
    step = max(1, days_max // 6)
    for d in range(0, days_max + 1, step):
        x = sx(d)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 10}" fill="#8b949e" font-size="10" '
            f'text-anchor="middle">d{d}</text>'
        )

    # Bands (low/high) as semi-transparent polygons
    for k in keys:
        if not bands[k]:
            continue
        forward = [f"{sx(d):.1f},{sy(hi):.1f}" for d, _, hi in bands[k]]
        backward = [f"{sx(d):.1f},{sy(lo):.1f}" for d, lo, _ in reversed(bands[k])]
        poly = " ".join(forward + backward)
        parts.append(
            f'<polygon points="{poly}" fill="{color[k]}" fill-opacity="0.10" stroke="none"/>'
        )

    # Lines
    for k in keys:
        pts = points[k]
        if not pts:
            continue
        d_str = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        parts.append(
            f'<polyline points="{d_str}" fill="none" stroke="{color[k]}" stroke-width="1.8"/>'
        )

    # Legend
    legend_x = padding_l
    for i, k in enumerate(keys):
        x = legend_x + i * 110
        parts.append(
            f'<rect x="{x}" y="2" width="10" height="3" fill="{color[k]}"/>'
            f'<text x="{x + 14}" y="6" fill="#c9d1d9" font-size="10">{k}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


@app.get("/reports/{ticker}/{run}/forecast", response_class=HTMLResponse)
def forecast_view(request: Request, ticker: str, run: str):
    bundle = reports_mod.get_bundle(ticker, run, REPORTS_ROOT)
    if bundle is None:
        raise HTTPException(404, "no such run")

    csv_text = reports_mod.read_text(bundle, "forecast.csv")
    if not csv_text:
        raise HTTPException(404, "no forecast.csv in this bundle")

    rows, columns = _parse_forecast_csv(csv_text)
    chart_svg = _build_chart_svg(rows)

    scenarios_text = reports_mod.read_text(bundle, "scenarios.json")
    scenarios = None
    if scenarios_text:
        try:
            scenarios = json.loads(scenarios_text)
        except json.JSONDecodeError:
            scenarios = None

    return TEMPLATES.TemplateResponse(
        request,
        "forecast_view.html",
        {
            "ticker": ticker,
            "bundle": bundle,
            "rows": rows,
            "columns": columns,
            "chart_svg": chart_svg,
            "scenarios": scenarios,
            "csv_url": f"/bundles/{bundle.name}/forecast.csv",
            "png_url": f"/bundles/{bundle.name}/forecast.png",
        },
    )


# ---------- Bundle assets (forecast.png, raw section md, etc) -----------------


@app.get("/bundles/{bundle_name}/{rest:path}")
def bundle_asset(bundle_name: str, rest: str):
    """Stream a single file out of a bundle.

    Tries R2 first (the canonical store); falls back to the local
    ``trading-reports/`` mirror so old un-backfilled bundles keep
    working and so the brief window between save-and-upload doesn't
    404. Lazy by design — nothing is fetched until this URL is hit.
    """
    if ".." in rest or rest.startswith("/"):
        raise HTTPException(400, "invalid path")

    streamed = storage_mod.stream_object(bundle_name, rest)
    if streamed is not None:
        body, content_type, length = streamed
        headers = {}
        if length is not None:
            headers["Content-Length"] = str(length)
        return StreamingResponse(body, media_type=content_type, headers=headers)

    local = REPORTS_ROOT / bundle_name / rest
    if local.is_file():
        return FileResponse(local)
    raise HTTPException(404, "no such asset")


# Healthcheck for compose --------------------------------------------------


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.get("/storage/status", response_class=JSONResponse)
def storage_status():
    """Quick visibility into which backend is in use, for debugging."""
    return JSONResponse(
        {
            "r2_enabled": storage_mod.is_enabled(),
            "bundle_count_r2": len(storage_mod.list_bundle_names())
            if storage_mod.is_enabled()
            else 0,
        }
    )
