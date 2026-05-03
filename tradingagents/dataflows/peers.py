"""Peer-comparable valuation snapshot.

For a given ticker, returns a small markdown table of named peers with
their trailing PE, forward PE, market cap, and P/S — pulled from yfinance
``Ticker.info``. Used by the bull and bear researchers so their cases
must engage with concrete relative-valuation numbers instead of arguing
in a vacuum.

Two-step resolution:

1. **Static peer map** — for the most-traded large caps we hard-code the
   peer set so the comparison is sensible (e.g. MSFT → AAPL/GOOGL/AMZN/
   META/ORCL/CRM). This avoids the brittle "first 5 hits in the same
   industry" approach.
2. **Sector fallback** — for unknown tickers we look up the target's
   sector and industry in yfinance, then try a small sector-default
   list. If we still can't find peers, the bull/bear prompts get an
   explicit "no peer data available" string and must reason without it.

Best-effort: any failure logs a warning and returns a graceful "data
unavailable" message rather than blocking the run.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


PEER_MAP: dict[str, list[str]] = {
    # Mega-cap tech
    "MSFT": ["AAPL", "GOOGL", "AMZN", "META", "ORCL", "CRM"],
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "GOOGL": ["MSFT", "AAPL", "META", "AMZN", "NFLX"],
    "GOOG": ["MSFT", "AAPL", "META", "AMZN", "NFLX"],
    "AMZN": ["MSFT", "GOOGL", "WMT", "META", "NFLX"],
    "META": ["GOOGL", "MSFT", "AAPL", "SNAP", "PINS"],
    "NFLX": ["DIS", "AMZN", "GOOGL", "PARA", "WBD"],
    # Semis
    "NVDA": ["AMD", "AVGO", "INTC", "QCOM", "TSM", "ASML"],
    "AMD": ["NVDA", "INTC", "QCOM", "AVGO", "TSM"],
    "INTC": ["AMD", "NVDA", "QCOM", "AVGO", "TSM"],
    "AVGO": ["NVDA", "QCOM", "AMD", "TXN", "MRVL"],
    "QCOM": ["AVGO", "NVDA", "AMD", "TXN", "MRVL"],
    "TSM": ["NVDA", "AVGO", "INTC", "ASML", "AMD"],
    # Cloud / SaaS / Comms
    "CRM": ["MSFT", "ORCL", "NOW", "ADBE", "WDAY"],
    "ORCL": ["MSFT", "CRM", "SAP", "NOW", "ADBE"],
    "ADBE": ["MSFT", "CRM", "NOW", "INTU", "ORCL"],
    "NOW": ["CRM", "MSFT", "ADBE", "WDAY", "ORCL"],
    "TWLO": ["ZM", "RNG", "FIVN", "VG", "BAND"],
    "ZM": ["TWLO", "RNG", "FIVN", "TEAM", "DOCN"],
    # Fintech / payments
    "V": ["MA", "PYPL", "AXP", "FIS", "FISV"],
    "MA": ["V", "PYPL", "AXP", "FIS", "FISV"],
    "PYPL": ["V", "MA", "SQ", "AFRM", "GPN"],
    # Consumer
    "TSLA": ["F", "GM", "RIVN", "LCID", "STLA"],
    "DIS": ["NFLX", "PARA", "WBD", "CMCSA", "FOXA"],
    # Banks
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "BAC": ["JPM", "WFC", "C", "GS", "MS"],
    "GS": ["MS", "JPM", "BAC", "C", "WFC"],
    # Healthcare
    "JNJ": ["PFE", "ABBV", "MRK", "BMY", "LLY"],
    "PFE": ["JNJ", "ABBV", "MRK", "BMY", "LLY"],
    "LLY": ["NVO", "JNJ", "PFE", "MRK", "ABBV"],
    "UNH": ["CVS", "HUM", "ELV", "CI", "MOH"],
    # Energy
    "XOM": ["CVX", "COP", "BP", "SHEL", "TTE"],
    "CVX": ["XOM", "COP", "BP", "SHEL", "TTE"],
}


def _resolve_peers(ticker: str, max_peers: int = 5) -> list[str]:
    """Return a peer list for ``ticker``. Empty list if we can't find any."""
    upper = ticker.upper()
    static = PEER_MAP.get(upper)
    if static:
        return [p for p in static[:max_peers] if p != upper]
    # Fallback: ask yfinance for the sector and try to find sector peers.
    # Disabled by default because yfinance's recommendation endpoints are
    # flaky; static map is our source of truth. A user can always extend
    # PEER_MAP for tickers they care about.
    return []


def _fmt_num(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_market_cap(v) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    return f"{v:.0f}"


def _fetch_one(ticker: str) -> Optional[dict]:
    """Return a row dict for one ticker, or None if yfinance can't reach it."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("peers: yfinance.info failed for %s: %s", ticker, exc)
        return None
    return {
        "ticker": ticker,
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "peg": info.get("pegRatio"),
        "market_cap": info.get("marketCap"),
    }


def get_peer_valuations(ticker: str, max_peers: int = 6) -> str:
    """Build a markdown table comparing the target's PE/Forward PE/PS/PEG/MC
    against named peers. Always returns a string suitable for prompt
    injection — never raises.

    The table puts the target ticker on the first row and bolds it so the
    LLM can't miss it. Empty cells show ``—`` rather than ``None`` so the
    LLM doesn't try to interpret a missing value as a real signal.
    """
    target = ticker.upper().strip()
    peers = _resolve_peers(target, max_peers=max_peers)

    rows = []
    for tk in [target] + peers:
        row = _fetch_one(tk)
        if row is not None:
            rows.append(row)

    # Filter out totally-empty rows except the target (always keep target row).
    target_row = rows[0] if rows and rows[0]["ticker"] == target else None
    peer_rows = [
        r
        for r in rows[1:]
        if any(r.get(k) is not None for k in ("trailing_pe", "forward_pe", "ps"))
    ]

    if not target_row and not peer_rows:
        return (
            f"_Peer-valuation table unavailable for {target}: yfinance returned "
            "no usable PE / Forward PE / P/S data for the target or its peers. "
            "Reason about valuation qualitatively — do not invent specific peer "
            "ratios._"
        )

    if not peer_rows:
        return (
            f"_No peer-valuation data available for {target}'s comparables. "
            f"Target: trailing PE {_fmt_num(target_row['trailing_pe'])}, "
            f"forward PE {_fmt_num(target_row['forward_pe'])}, "
            f"P/S {_fmt_num(target_row['ps'])}, "
            f"market cap {_fmt_market_cap(target_row['market_cap'])}. "
            "Argue valuation against the target's own historical multiples; do "
            "not invent peer ratios._"
        )

    lines = [
        "| Ticker | Trailing PE | Forward PE | P/S | PEG | Mkt Cap |",
        "|---|---|---|---|---|---|",
    ]
    target_label = f"**{target}** (target)"
    if target_row:
        lines.append(
            f"| {target_label} | {_fmt_num(target_row['trailing_pe'])} | "
            f"{_fmt_num(target_row['forward_pe'])} | {_fmt_num(target_row['ps'])} | "
            f"{_fmt_num(target_row['peg'])} | {_fmt_market_cap(target_row['market_cap'])} |"
        )
    for r in peer_rows:
        lines.append(
            f"| {r['ticker']} | {_fmt_num(r['trailing_pe'])} | "
            f"{_fmt_num(r['forward_pe'])} | {_fmt_num(r['ps'])} | "
            f"{_fmt_num(r['peg'])} | {_fmt_market_cap(r['market_cap'])} |"
        )

    return "\n".join(lines)
