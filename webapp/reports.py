"""Filesystem helpers for browsing the persisted ``trading-reports/`` tree.

Every analysis run writes a folder named ``<TICKER>[_<persona>]_<YYYYMMDD>_<HHMMSS>``
containing the full bundle. This module discovers, parses, and groups
those folders so the webapp can render ticker-tab-and-dropdown views
without knowing the deep file layout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


REPORTS_ROOT = Path("trading-reports")

# Bundle name format: TICKER[_PERSONA_TOKENS]_YYYYMMDD_HHMMSS
# The TICKER itself never contains underscores (Yahoo uses dots: BRK.B,
# CNC.TO, 7203.T) so split by ``_`` is unambiguous.
_BUNDLE_RE = re.compile(
    r"^(?P<ticker>[^_]+?)(?:_(?P<persona>.+?))?_(?P<date>\d{8})_(?P<time>\d{6})$"
)

NEUTRAL_LABEL = "neutral"


@dataclass
class Bundle:
    """Single completed run on disk."""

    name: str                                  # raw folder name
    ticker: str
    persona: str                               # "" for neutral runs
    timestamp: datetime
    path: Path

    @property
    def persona_label(self) -> str:
        return self.persona or NEUTRAL_LABEL

    @property
    def display_timestamp(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class TickerView:
    """A single ticker grouped by persona, newest first within each group."""

    ticker: str
    by_persona: dict[str, list[Bundle]] = field(default_factory=dict)

    @property
    def personas(self) -> list[str]:
        # Stable order: neutral first, then alphabetical
        ps = list(self.by_persona.keys())
        ps.sort(key=lambda p: (p != NEUTRAL_LABEL, p))
        return ps

    def latest(self, persona: str) -> Optional[Bundle]:
        items = self.by_persona.get(persona) or []
        return items[0] if items else None


def parse_bundle_name(name: str) -> Optional[dict]:
    m = _BUNDLE_RE.match(name)
    if not m:
        return None
    try:
        ts = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return {
        "ticker": m.group("ticker"),
        "persona": m.group("persona") or "",
        "timestamp": ts,
    }


def list_bundles(root: Path = REPORTS_ROOT) -> list[Bundle]:
    """Discover all bundles, newest first."""
    if not root.is_dir():
        return []
    out: list[Bundle] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        meta = parse_bundle_name(p.name)
        if meta is None:
            continue
        out.append(
            Bundle(
                name=p.name,
                ticker=meta["ticker"],
                persona=meta["persona"],
                timestamp=meta["timestamp"],
                path=p,
            )
        )
    out.sort(key=lambda b: b.timestamp, reverse=True)
    return out


def list_tickers(root: Path = REPORTS_ROOT) -> list[str]:
    seen: set[str] = set()
    for b in list_bundles(root):
        seen.add(b.ticker)
    return sorted(seen)


def get_ticker_view(ticker: str, root: Path = REPORTS_ROOT) -> TickerView:
    """Return all bundles for ``ticker`` grouped by persona, newest first."""
    view = TickerView(ticker=ticker)
    for b in list_bundles(root):
        if b.ticker.lower() != ticker.lower():
            continue
        view.by_persona.setdefault(b.persona_label, []).append(b)
    # Each persona list is already newest-first because list_bundles is.
    return view


def get_bundle(ticker: str, name: str, root: Path = REPORTS_ROOT) -> Optional[Bundle]:
    """Look up a specific bundle folder by exact name (after verifying ticker)."""
    p = root / name
    if not p.is_dir():
        return None
    meta = parse_bundle_name(name)
    if meta is None or meta["ticker"].lower() != ticker.lower():
        return None
    return Bundle(
        name=name,
        ticker=meta["ticker"],
        persona=meta["persona"],
        timestamp=meta["timestamp"],
        path=p,
    )


def read_complete_report(bundle: Bundle) -> str:
    """Return the rendered ``complete_report.md`` text, or empty if missing."""
    f = bundle.path / "complete_report.md"
    if not f.is_file():
        return ""
    return f.read_text(encoding="utf-8")


def list_section_files(bundle: Bundle) -> dict[str, list[Path]]:
    """Group the per-section markdown files (analysts/research/trading/risk/portfolio)."""
    out: dict[str, list[Path]] = {}
    for sub in ("1_analysts", "2_research", "3_trading", "4_risk", "5_portfolio"):
        d = bundle.path / sub
        if d.is_dir():
            out[sub] = sorted(d.glob("*.md"))
    return out
