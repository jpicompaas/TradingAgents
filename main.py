from dotenv import load_dotenv

# Load .env BEFORE importing default_config — DEFAULT_CONFIG reads
# os.getenv("TRADINGAGENTS_PERSONA") at import time, so the load order
# matters. With .env loaded first, the persona overlay actually takes effect.
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.reports import save_run_bundle  # noqa: E402

# Create a custom config (defaults to Groq via DEFAULT_CONFIG)
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "groq"
config["deep_think_llm"] = "llama-3.3-70b-versatile"
config["quick_think_llm"] = "llama-3.3-70b-versatile"
config["max_debate_rounds"] = 1

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance
    "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance
    "news_data": "yfinance",                 # Options: alpha_vantage, yfinance
}

TICKER = "TWLO"
TRADE_DATE = "2026-05-02"

# Initialize with custom config
ta = TradingAgentsGraph(debug=False, config=config)

# forward propagate
final_state, decision = ta.propagate(TICKER, TRADE_DATE)

# Auto-save full bundle (analysts + research + trading + risk + portfolio + raw state)
bundle_dir = save_run_bundle(final_state, TICKER)
print(f"\nReport saved to: {bundle_dir.resolve()}")
print(f"\n--- Final Decision ---\n{decision}")
