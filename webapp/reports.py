"""Filesystem + R2 helpers for browsing the persisted report bundles.

When R2 is configured (``webapp.storage.is_enabled()``) the bundle list
is the union of R2 keys and any local-only bundles, and per-file reads
go through R2 first, falling back to local. With R2 disabled, behavior
is identical to the original local-only implementation.

Bundle-name format: ``<TICKER>[_<persona>]_<YYYYMMDD>_<HHMMSS>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from webapp import storage


REPORTS_ROOT = Path("trading-reports")

# Bundle name format: TICKER[_PERSONA_TOKENS]_YYYYMMDD_HHMMSS
# The TICKER itself never contains underscores (Yahoo uses dots: BRK.B,
# CNC.TO, 7203.T) so split by ``_`` is unambiguous.
_BUNDLE_RE = re.compile(
    r"^(?P<ticker>[^_]+?)(?:_(?P<persona>.+?))?_(?P<date>\d{8})_(?P<time>\d{6})$"
)

NEUTRAL_LABEL = "neutral"

_SECTION_DIRS = ("1_analysts", "2_research", "3_trading", "4_risk", "5_portfolio")


@dataclass
class Bundle:
    """A single completed run.

    ``path`` is the local-disk location when present; ``None`` for bundles
    that live only in R2. Reads should go through ``read_complete_report``
    / ``list_section_files`` rather than touching ``path`` directly.
    """

    name: str
    ticker: str
    persona: str  # "" for neutral runs
    timestamp: datetime
    path: Optional[Path]

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


def _bundle_from_name(name: str, path: Optional[Path]) -> Optional[Bundle]:
    meta = parse_bundle_name(name)
    if meta is None:
        return None
    return Bundle(
        name=name,
        ticker=meta["ticker"],
        persona=meta["persona"],
        timestamp=meta["timestamp"],
        path=path,
    )


def list_bundles(root: Path = REPORTS_ROOT) -> list[Bundle]:
    """Discover all bundles, newest first.

    Returns the union of R2 (when enabled) and local-disk bundles. A
    bundle present in both keeps its local ``path`` so the static-file
    fallback works even if R2 is briefly unreachable.
    """
    bundles: dict[str, Bundle] = {}

    # Local first — gives us a path attribute for fallback reads.
    if root.is_dir():
        for p in root.iterdir():
            if not p.is_dir():
                continue
            b = _bundle_from_name(p.name, p)
            if b is not None:
                bundles[p.name] = b

    # R2 next — adds bundles that don't exist locally; keeps existing
    # ``path`` for ones that do.
    for name in storage.list_bundle_names():
        if name in bundles:
            continue
        b = _bundle_from_name(name, None)
        if b is not None:
            bundles[name] = b

    out = list(bundles.values())
    out.sort(key=lambda b: b.timestamp, reverse=True)
    return out


def list_tickers(root: Path = REPORTS_ROOT) -> list[str]:
    seen: set[str] = set()
    for b in list_bundles(root):
        seen.add(b.ticker)
    return sorted(seen)


def get_ticker_view(ticker: str, root: Path = REPORTS_ROOT) -> TickerView:
    view = TickerView(ticker=ticker)
    for b in list_bundles(root):
        if b.ticker.lower() != ticker.lower():
            continue
        view.by_persona.setdefault(b.persona_label, []).append(b)
    return view


def get_bundle(
    ticker: str, name: str, root: Path = REPORTS_ROOT
) -> Optional[Bundle]:
    """Look up a bundle by exact name. Checks local then R2."""
    meta = parse_bundle_name(name)
    if meta is None or meta["ticker"].lower() != ticker.lower():
        return None

    p = root / name
    if p.is_dir():
        return Bundle(
            name=name,
            ticker=meta["ticker"],
            persona=meta["persona"],
            timestamp=meta["timestamp"],
            path=p,
        )
    if storage.bundle_exists(name):
        return Bundle(
            name=name,
            ticker=meta["ticker"],
            persona=meta["persona"],
            timestamp=meta["timestamp"],
            path=None,
        )
    return None


# ---------------------------------------------------------------------------
# Per-file reads — R2 first, local fallback
# ---------------------------------------------------------------------------


def read_complete_report(bundle: Bundle) -> str:
    return read_text(bundle, "complete_report.md")


def read_text(bundle: Bundle, relpath: str) -> str:
    """Fetch a text file from the bundle. Empty string on miss."""
    text = storage.get_text(bundle.name, relpath)
    if text is not None:
        return text
    if bundle.path is not None:
        f = bundle.path / relpath
        if f.is_file():
            return f.read_text(encoding="utf-8")
    return ""


def list_section_files(bundle: Bundle) -> dict[str, list[str]]:
    """Group per-section markdown filenames (analysts/research/...).

    Values are bare filenames so templates can build URLs without
    worrying about whether the source is R2 or disk.
    """
    out: dict[str, list[str]] = {}

    # Prefer R2 when the bundle exists there; otherwise fall back to disk.
    if storage.is_enabled() and storage.bundle_exists(bundle.name):
        for sub in _SECTION_DIRS:
            files = [
                rel.split("/", 1)[1]
                for rel in storage.list_bundle_files(bundle.name, sub)
                if rel.startswith(f"{sub}/") and rel.endswith(".md")
            ]
            if files:
                out[sub] = sorted(files)
        if out:
            return out

    if bundle.path is not None:
        for sub in _SECTION_DIRS:
            d = bundle.path / sub
            if d.is_dir():
                files = sorted(p.name for p in d.glob("*.md"))
                if files:
                    out[sub] = files
    return out
