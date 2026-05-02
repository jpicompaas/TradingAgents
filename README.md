<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-TauricResearch-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/Join_GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents: Multi-Agents LLM Financial Trading Framework

## News
- [2026-04] **TradingAgents v0.2.4** released with structured-output agents (Research Manager, Trader, Portfolio Manager), LangGraph checkpoint resume, persistent decision log, DeepSeek/Qwen/GLM/Azure provider support, Docker, and a Windows UTF-8 encoding fix. See [CHANGELOG.md](CHANGELOG.md) for the full list.
- [2026-03] **TradingAgents v0.2.3** released with multi-language support, GPT-5.4 family models, unified model catalog, backtesting date fidelity, and proxy support.
- [2026-03] **TradingAgents v0.2.2** released with GPT-5.4/Gemini 3.1/Claude 4.6 model coverage, five-tier rating scale, OpenAI Responses API, Anthropic effort control, and cross-platform stability.
- [2026-02] **TradingAgents v0.2.0** released with multi-provider LLM support (GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x) and improved system architecture.
- [2026-01] **Trading-R1** [Technical Report](https://arxiv.org/abs/2509.11420) released, with [Terminal](https://github.com/TauricResearch/Trading-R1) expected to land soon.

<div align="center">
<a href="https://www.star-history.com/#TauricResearch/TradingAgents&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" />
   <img alt="TradingAgents Star History" src="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" style="width: 80%; height: auto;" />
 </picture>
</a>
</div>

> 🎉 **TradingAgents** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

<div align="center">

🚀 [TradingAgents](#tradingagents-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#tradingagents-package) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles. This ensures the system achieves a robust, scalable approach to market analysis and decision-making.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Sentiment Analyst: Analyzes social media and public sentiment using sentiment scoring algorithms to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions. It determines the timing and magnitude of trades based on comprehensive market insights.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

TradingAgents runs in Docker — no Python toolchain or `pip install` on the host.

### 1. Clone

```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

### 2. Configure API keys

```bash
cp .env.example .env   # then edit .env and fill in keys for the providers you'll use
```

### 3. Run the CLI

```bash
docker compose run --rm tradingagents                          # interactive CLI
docker compose run --rm tradingagents analyze --checkpoint     # enable LangGraph resume
docker compose run --rm tradingagents analyze --clear-checkpoints   # wipe per-ticker checkpoint DBs
```

The first invocation builds the image; subsequent runs reuse it. Run `docker compose build tradingagents` to force a rebuild after pulling code changes. Per-run state (`~/.tradingagents/...`) is persisted across runs in the `tradingagents_data` named volume.

### Local models with Ollama

```bash
docker compose --profile ollama run --rm tradingagents-ollama
```

### Programmatic / `main.py` example

The Dockerfile's entrypoint is the CLI, so override it to run a script:

```bash
docker compose run --rm --entrypoint python tradingagents main.py
```

### Developing without Docker (optional)

Contributors who prefer a host install can still:
```bash
pip install .
tradingagents
```
This is not required to use the framework.

### Required APIs

The default provider is **Groq** (`llama-3.3-70b-versatile`), so the minimum-viable `.env` is:

```bash
GROQ_API_KEY=gsk_...
TRADINGAGENTS_DISABLE_ANNOUNCEMENTS=1
```

Pick a different provider by setting its key — `.env.example` lists all of them:

| Provider | Env var | Models surfaced in CLI |
| --- | --- | --- |
| **Groq** (default) | `GROQ_API_KEY` | `llama-3.3-70b-versatile`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, Llama 4 Scout/Maverick, Gemma2 |
| OpenAI | `OPENAI_API_KEY` | GPT-5.x family |
| Anthropic | `ANTHROPIC_API_KEY` | Claude 4.x family |
| Google | `GOOGLE_API_KEY` | Gemini 3.x |
| xAI | `XAI_API_KEY` | Grok 4.x |
| DeepSeek | `DEEPSEEK_API_KEY` | DeepSeek V4 (incl. thinking-mode round-trip) |
| Qwen (DashScope) | `DASHSCOPE_API_KEY` | Qwen 3.x |
| GLM (Zhipu) | `ZHIPU_API_KEY` | GLM 5 |
| OpenRouter | `OPENROUTER_API_KEY` | Any OpenRouter model |
| Ollama (local) | _none_ | Qwen3, GPT-OSS 20B, GLM-4.7-Flash |

Optional:
- `ALPHA_VANTAGE_API_KEY` — alternative market-data vendor (default is `yfinance`, no key needed)
- `TRADINGAGENTS_PERSONA` — investor/economist overlay (`buffett`, `dalio`, `cathie_wood`, `taleb`, …) applied to the Trader and Portfolio Manager

For enterprise providers (Azure OpenAI, AWS Bedrock), copy `.env.enterprise.example` to `.env.enterprise` and fill in credentials.

### CLI Usage

Launch the interactive CLI:
```bash
docker compose run --rm tradingagents             # canonical path
tradingagents                                     # only if you also did the optional host install
```
You will see a screen where you can select your desired tickers, analysis date, LLM provider, research depth, and more.

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

## TradingAgents Package

### Tech stack

| Concern | Library / tool |
| --- | --- |
| Graph orchestration | `langgraph` (StateGraph, ToolNode, SqliteSaver checkpointer) |
| LLM wrappers | `langchain-core`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` |
| Structured output | `pydantic` schemas + `bind_tools(method="function_calling")` |
| Market data | `yfinance` + `curl_cffi`, Alpha Vantage |
| Indicators | `stockstats`, `pandas` |
| Interactive CLI | `typer`, `rich`, `questionary` |
| Persistence | `langgraph-checkpoint-sqlite`, plain Markdown memory log |
| Containerization | Docker, Docker Compose |

### Resilience layers

Trading-agent runs are long, multi-step, and depend on several flaky external services. The pipeline is built so a single transient failure cannot crash a run:

| Layer | What it catches |
| --- | --- |
| **DNS pinning** in `docker-compose.yml` (1.1.1.1 / 8.8.8.8) | ISP DNS sinkholes (e.g. `fc.yahoo.com` returning a black-hole IP) |
| **`yf_retry`** | yfinance rate limits, connection errors, 5xx, timeouts (6 retries, exponential backoff) |
| **`route_to_vendor`** | Per-vendor retries (5× for key metrics like price/indicators/fundamentals, 2× otherwise), then failover to a different vendor |
| **`canonical_ticker`** | LLMs hallucinating exchange suffixes (`TWLO` → `TWLO.TO`) — every data tool uses the user-entered ticker, ignoring whatever the LLM passes |
| **`GroqChatOpenAI` recovery** | Groq's `tool_use_failed` quirk — parses Groq's malformed `<function=...>` payload directly into a valid `tool_calls` AIMessage, so the graph never even sees the error; final fallback returns an empty AIMessage so the run still finishes |
| **`ToolNode(handle_tool_errors=True)`** | Any unhandled tool exception (404, malformed args, etc.) becomes a `ToolMessage` containing the error string |
| **`invoke_structured_or_freetext`** | Falls back to plain `llm.invoke` if structured-output binding fails (e.g. `deepseek-reasoner` has no `tool_choice`) |

Permanent errors (404 / "delisted") short-circuit the retry budget so a wrong ticker fails fast.

### Implementation Details

Built with LangGraph for flexibility and modularity. Defaults to Groq for a clone-and-run experience. Supports OpenAI, Anthropic, Google, xAI, DeepSeek, Qwen (DashScope), GLM (Zhipu), OpenRouter, Ollama (local), Azure OpenAI, and AWS Bedrock.

### Python Usage

To use TradingAgents inside your code, you can import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from dotenv import load_dotenv
load_dotenv()  # IMPORTANT: must run before importing default_config (it reads
               # TRADINGAGENTS_PERSONA via os.getenv at import time)

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.reports import save_run_bundle

ta = TradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())
final_state, decision = ta.propagate("NVDA", "2026-01-15")

# Auto-save full bundle (analysts + research + trading + risk + portfolio + raw state)
bundle_dir = save_run_bundle(final_state, "NVDA")
print(f"Report saved to: {bundle_dir.resolve()}")
print(decision)
```

Override the defaults to switch provider/models/debate depth:

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "groq"                          # default; pick another from the table above
config["deep_think_llm"] = "openai/gpt-oss-120b"         # Research Manager + Portfolio Manager
config["quick_think_llm"] = "llama-3.3-70b-versatile"    # analysts, researchers, risk debators, trader
config["max_debate_rounds"] = 2
config["trading_persona"] = "warren_buffett"             # optional investor overlay

ta = TradingAgentsGraph(debug=False, config=config)
```

See `tradingagents/default_config.py` for all configuration options.

## Investor / Economist Personas

You can optionally filter the **final trading decision** through the worldview of a renowned investor or economist. When a persona is configured, its philosophy is prepended to the system prompt of the **Trader** (transaction proposal) and the **Portfolio Manager** (final approve/reject). Analysts and bull/bear debators stay neutral so tool calls and the adversarial debate aren't skewed.

### How to enable

Set `TRADINGAGENTS_PERSONA` in your `.env`:

```bash
TRADINGAGENTS_PERSONA=warren_buffett
```

…or in code:

```python
config = DEFAULT_CONFIG.copy()
config["trading_persona"] = "cathie_wood"
ta = TradingAgentsGraph(config=config)
```

Leave it blank or unset for neutral behavior (the default). Unknown names log a warning and the run continues with no persona — typos won't crash a long run.

### Available personas

| Style | Personas |
| --- | --- |
| **Value / Quality** | `warren_buffett`, `charlie_munger`, `ben_graham`, `mohnish_pabrai` |
| **Growth / Innovation** | `cathie_wood`, `peter_lynch`, `phil_fisher` |
| **Macro / Trader** | `stanley_druckenmiller`, `george_soros`, `ray_dalio` |
| **Contrarian / Risk** | `michael_burry`, `nassim_taleb`, `bill_ackman` |
| **Academic / Indexing** | `aswath_damodaran`, `john_bogle` |
| **Indian markets** | `rakesh_jhunjhunwala` |

Short aliases are accepted (`buffett`, `munger`, `graham`, `pabrai`, `wood`, `lynch`, `fisher`, `druck`, `soros`, `dalio`, `burry`, `ackman`, `taleb`, `damodaran`, `bogle`, `rakesh`). Lookup is case-insensitive and tolerant of spaces, hyphens, and underscores — `"Warren Buffett"`, `"warren-buffett"`, and `"warren_buffett"` all resolve to the same persona.

### What changes

The persona overlay is a **prompt prefix**, not a separate pipeline. The Trader and Portfolio Manager still see the same analyst reports, the same bull/bear debate, and the same risk debate — but they weigh that evidence through the persona's lens. For example, `warren_buffett` will downweight stories that depend on multiple expansion and demand a margin of safety; `cathie_wood` will lead with TAM and S-curve adoption and tolerate volatility; `nassim_taleb` will reject anything fragile and prefer convex payoffs.

Personas do not change which analysts run, which tools are called, or which structured-output schema is used. They steer style and emphasis only. To add a new persona, edit `tradingagents/agents/utils/personas.py` and add a `Persona(...)` entry plus any aliases.

## Outputs

Every run writes three things to disk:

### 1. Auto-saved report bundle

`trading-reports/<TICKER>_<YYYYMMDD_HHMMSS>/` (gitignored, bind-mounted into the container):

```
complete_report.md              ← consolidated, with persona/strategy block at top
full_state.json                 ← raw state for downstream tooling
1_analysts/{market,sentiment,news,fundamentals}.md
2_research/{bull,bear,manager}.md      ← bull/bear debate (each with required-assumptions block)
3_trading/trader.md
4_risk/{aggressive,conservative,neutral}.md   ← three-way risk debate
5_portfolio/{decision.md, final_trade_decision.md}
```

Bull and bear writeups end with a structured **`## Required Assumptions for the Bull/Bear Case`** section enumerating eight categories (inflation regime, interest-rate path, geopolitical/war risk, sector demand, regulatory, FX, operational milestones, liquidity) — each with an explicit invalidation signal so the thesis is falsifiable.

### 2. Decision log (always on)

Each completed run appends to `~/.tradingagents/memory/trading_memory.md`. On the next run for the same ticker, TradingAgents fetches the realised return (raw and alpha vs SPY), generates a one-paragraph reflection, and injects the most recent same-ticker decisions plus recent cross-ticker lessons into the Portfolio Manager prompt — so each analysis carries forward what worked and what didn't. Override the path with `TRADINGAGENTS_MEMORY_LOG_PATH`.

### 3. Checkpoint resume (opt-in)

`--checkpoint` saves state after each LangGraph node, so a crashed or interrupted run resumes from the last successful step instead of starting over. Per-ticker SQLite databases live at `~/.tradingagents/cache/checkpoints/<TICKER>.db` (override with `TRADINGAGENTS_CACHE_DIR`). Use `--clear-checkpoints` to reset.

```bash
docker compose run --rm tradingagents analyze --checkpoint           # enable
docker compose run --rm tradingagents analyze --clear-checkpoints    # reset
```

```python
config = DEFAULT_CONFIG.copy()
config["checkpoint_enabled"] = True
ta = TradingAgentsGraph(config=config)
```

## Contributing

We welcome contributions from the community! Whether it's fixing a bug, improving documentation, or suggesting a new feature, your input helps make this project better. If you are interested in this line of research, please consider joining our open-source financial AI research community [Tauric Research](https://tauric.ai/).

Past contributions, including code, design feedback, and bug reports, are credited per release in [`CHANGELOG.md`](CHANGELOG.md).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help :)

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
