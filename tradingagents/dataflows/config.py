import tradingagents.default_config as default_config
from typing import Dict, Optional

# Use default config but allow it to be overridden
_config: Optional[Dict] = None


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()


def set_config(config: Dict):
    """Update the configuration with custom values."""
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    _config.update(config)


def get_config() -> Dict:
    """Get the current configuration."""
    if _config is None:
        initialize_config()
    return _config.copy()


def canonical_ticker(llm_supplied: str) -> str:
    """Return the user-entered ticker, ignoring whatever the LLM hallucinated.

    LLMs (especially smaller ones like llama-3.1-8b-instant) frequently invent
    exchange suffixes (`TWLO` -> `TWLO.TO`/`TWLO.TW`). The user already told us
    the exact ticker via the CLI/propagate(); trust that, not the model.

    Falls back to the LLM-supplied value if no canonical ticker is set
    (e.g. when tools are exercised outside a propagate() flow).
    """
    pinned = (get_config().get("company_of_interest") or "").strip()
    return pinned or (llm_supplied or "").strip()


# Initialize with default config
initialize_config()
