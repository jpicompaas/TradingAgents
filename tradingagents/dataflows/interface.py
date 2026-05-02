from typing import Annotated

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

import logging as _logging
import time as _time

_logger = _logging.getLogger(__name__)

# Methods whose output the agents reason heavily on — analysts/researchers
# can't recover from missing values here, so we retry these aggressively
# before failing over to a different vendor.
_KEY_METRIC_METHODS = frozenset({
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
})

# Per-call retry budget. Key metrics get more attempts because a missing
# data point cascades into "no clear trend" placeholder reasoning.
_RETRIES_KEY = 5
_RETRIES_OTHER = 2
_BASE_DELAY = 1.0  # seconds; doubled each attempt


def _is_transient_data_error(exc: Exception) -> bool:
    """Errors that warrant a retry on the SAME vendor before failover.

    Network blips, upstream 5xx, connection refused, timeouts — anything
    that could plausibly succeed on the next try. Permanent errors (404,
    invalid ticker, missing ticker symbol) should NOT be retried.
    """
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return True
    if isinstance(exc, AlphaVantageRateLimitError):
        # Rate limits — short retry, then failover so we don't burn the budget.
        return True
    msg = str(exc).lower()
    transient_markers = (
        "rate limit",
        "rate-limit",
        "timed out",
        "timeout",
        "connection",
        "temporarily",
        "try again",
        "503",
        "502",
        "504",
        "429",
    )
    return any(m in msg for m in transient_markers)


def _is_permanent_data_error(exc: Exception) -> bool:
    """Errors that should fail over immediately rather than retry."""
    msg = str(exc).lower()
    return any(m in msg for m in ("not found", "404", "delisted", "no data found"))


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor with retry + failover.

    Resilience order:
    1. Try the primary vendor with up to N retries (exponential backoff)
       on transient errors. Permanent errors (404/delisted) skip retry.
    2. On exhausted retries, fail over to the next vendor and retry there.
    3. Return as soon as any attempt succeeds.

    For key-metric methods (price/indicators/fundamentals) the retry
    budget is higher so analysts always get the data they need to reason.
    """
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    max_retries = _RETRIES_KEY if method in _KEY_METRIC_METHODS else _RETRIES_OTHER

    last_error: Exception | None = None
    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        delay = _BASE_DELAY
        for attempt in range(max_retries + 1):
            try:
                return impl_func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — catalogued below
                last_error = exc
                if _is_permanent_data_error(exc):
                    # 404 / delisted — no point retrying or failing over with
                    # the same args; return the error string so the agent can
                    # see it and choose a different tool/ticker reasoning path.
                    _logger.info(
                        "%s/%s: permanent error, skipping retries: %s",
                        vendor, method, exc,
                    )
                    break
                if not _is_transient_data_error(exc):
                    # Unknown error class — retry once or twice but don't
                    # spend the full budget guessing.
                    if attempt >= 1:
                        break
                if attempt < max_retries:
                    _logger.warning(
                        "%s/%s attempt %d/%d failed (%s); retrying in %.1fs",
                        vendor, method, attempt + 1, max_retries, exc, delay,
                    )
                    _time.sleep(delay)
                    delay *= 2
                    continue
                # Exhausted retries on this vendor; fall through to next.
                _logger.warning(
                    "%s/%s exhausted %d retries; failing over",
                    vendor, method, max_retries,
                )
                break

    if last_error is not None:
        raise RuntimeError(
            f"All vendors failed for '{method}' after retries; last error: {last_error}"
        ) from last_error
    raise RuntimeError(f"No available vendor for '{method}'")