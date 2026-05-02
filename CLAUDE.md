# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**TradingAgents** is a multi-agent LLM financial trading framework built on **LangGraph**. A team of LLM agents (analysts → researchers → trader → risk debators → portfolio manager) collaboratively analyzes a ticker for a given date and emits a final trade decision. Defaults to **Groq** (`llama-3.3-70b-versatile`) so a clone-and-run takes one API key. Entry points: `cli/main.py` (interactive Rich/Typer CLI, exposed as the `tradingagents` command) and `main.py` / `tradingagents.graph.trading_graph.TradingAgentsGraph` (programmatic).

## Tech stack

| Concern | Library / tool |
| --- | --- |
| Graph orchestration | `langgraph` (StateGraph, ToolNode, SqliteSaver checkpointer) |
| LLM wrappers | `langchain-core`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` |
| Structured output | `pydantic` schemas + `bind_tools(method="function_calling")` |
| LLM providers | OpenAI, Groq, Anthropic, Google (Gemini), xAI (Grok), DeepSeek, Qwen (DashScope), GLM (Zhipu), OpenRouter, Ollama (local), Azure OpenAI, AWS Bedrock |
| Market data | `yfinance` + `curl_cffi` (Chrome TLS impersonation), Alpha Vantage |
| Indicators | `stockstats`, `pandas` |
| CLI | `typer`, `rich`, `questionary`, `python-dotenv` |
| Persistence | `langgraph-checkpoint-sqlite`, plain Markdown memory log |
| Containerization | Docker, Docker Compose |
| Testing | `pytest`, autouse fixtures in `tests/conftest.py` (offline by default) |

## Commands

### Install / run (Docker is the canonical path — no host `pip install` required)

```bash
cp .env.example .env                                                # one-time: fill in provider keys
docker compose run --rm tradingagents                               # interactive CLI
docker compose run --rm tradingagents analyze --checkpoint          # enable LangGraph resume
docker compose run --rm tradingagents analyze --clear-checkpoints   # wipe all per-ticker checkpoint DBs
docker compose run --rm --entrypoint python tradingagents main.py   # programmatic example (TWLO)
docker compose --profile ollama run --rm tradingagents-ollama       # local-models variant
docker compose build tradingagents                                  # rebuild after code changes
```

Optional host install (for contributors only):
```bash
pip install .
tradingagents
```

### Tests
```bash
pytest                                         # full suite
pytest tests/test_signal_processing.py         # single file
pytest -m unit                                 # markers: unit, integration, smoke
python scripts/smoke_structured_output.py      # live-API smoke (needs real keys)
```

`tests/conftest.py` autouses a `_dummy_api_keys` fixture that injects placeholder values for every provider env var, and a `mock_llm_client` fixture that patches `tradingagents.llm_clients.factory.create_llm_client`. Unit tests must stay offline — do not call real APIs.

### Environment

Minimum-viable `.env`:
```
GROQ_API_KEY=gsk_...
TRADINGAGENTS_DISABLE_ANNOUNCEMENTS=1
```

Other supported keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `ZHIPU_API_KEY`, `OPENROUTER_API_KEY`, `ALPHA_VANTAGE_API_KEY`. For Azure / AWS Bedrock, copy `.env.enterprise.example` to `.env.enterprise` (the CLI loads it after `.env` with `override=False`).

`TRADINGAGENTS_PERSONA` selects an investor/economist overlay (`buffett`, `dalio`, `soros`, `taleb`, `cathie_wood`, …). Surfaced in the CLI Progress panel title and at the top of `complete_report.md`.

`TRADINGAGENTS_LLM_PROVIDER` (and `LLM_PROVIDER` as a fallback) and `TRADINGAGENTS_OUTPUT_LANGUAGE` skip the corresponding interactive picker steps. When set, the CLI logs `Step N: ... pinned via env → <value>` and proceeds.

Runtime overrides:
- `TRADINGAGENTS_RESULTS_DIR` — per-run JSON state and reports (default `~/.tradingagents/logs`).
- `TRADINGAGENTS_CACHE_DIR` — base for checkpoint SQLite DBs (default `~/.tradingagents/cache`).
- `TRADINGAGENTS_MEMORY_LOG_PATH` — append-only Markdown decision log (default `~/.tradingagents/memory/trading_memory.md`).

`docker-compose.yml` bind-mounts `./.tradingagents` and `./trading-reports` to the host. It also pins DNS to `1.1.1.1` / `8.8.8.8` to bypass ISP resolvers that sinkhole `fc.yahoo.com` (yfinance's cookie endpoint).

## Architecture

### Graph orchestration (`tradingagents/graph/`)

`TradingAgentsGraph` (`graph/trading_graph.py`) is the public façade. Construction wires:
1. **LLM clients** via `create_llm_client(provider, model, base_url, **kwargs)` — both a `deep_thinking_llm` and a `quick_thinking_llm` are produced. Provider-specific thinking knobs (`google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`) are gated in `_get_provider_kwargs` so they only flow to the right client.
2. **Tool nodes** — four `langgraph.prebuilt.ToolNode`s keyed by analyst type (`market`, `social`, `news`, `fundamentals`), constructed with `handle_tool_errors=True` so a failed tool call becomes a `ToolMessage` containing the error string instead of crashing the graph.
3. **Workflow** — `GraphSetup.setup_graph(selected_analysts)` builds a `StateGraph(AgentState)`. Analyst nodes are connected sequentially with conditional `should_continue_*` edges back to their tool node and a `Msg Clear *` (a `RemoveMessage` node) before handing off to the next analyst or to **Bull Researcher**. The debate loop alternates Bull/Bear via `should_continue_debate` until `Research Manager` resolves it, which then flows Trader → Aggressive → Conservative → Neutral risk debators (round-robin via `should_continue_risk_analysis`) → Portfolio Manager → END.
4. **Checkpointer** — when `config["checkpoint_enabled"]`, `propagate` recompiles `self.workflow` with a per-ticker `SqliteSaver` (`graph/checkpointer.py`). Thread ID is `sha256("<TICKER>:<date>")[:16]` so the same ticker+date resumes and a different date starts fresh. Checkpoints are cleared on successful completion.

`propagate(ticker, date)` is the entry point. It:
1. **Pins the canonical ticker** into dataflow config (`set_config({"company_of_interest": ticker})`) so the data tools can override any LLM-hallucinated symbol.
2. Resolves any **pending** memory-log entries for the ticker (fetches realized return + SPY alpha via yfinance, runs `Reflector.reflect_on_final_decision`, writes reflections in a single batch).
3. Injects `past_context` (recent same-ticker decisions + cross-ticker lessons) from the memory log into the initial state so the Portfolio Manager prompt carries forward what worked.
4. Runs the graph (`graph.stream` in debug mode, otherwise `graph.invoke`).
5. Logs full state to `<results_dir>/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json`, writes a new pending entry to the memory log, and returns `(final_state, processed_signal)`.

### Agent state (`agents/utils/agent_states.py`)
`AgentState` extends `MessagesState` with named report fields (`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`, `investment_plan`, `trader_investment_plan`, `final_trade_decision`) plus nested debate states (`InvestDebateState`, `RiskDebateState`). Every analyst writes to its dedicated report field; the CLI keys off these field names to drive its live progress display.

### Agents (`tradingagents/agents/`)
Organized by team — `analysts/` (market, social, news, fundamentals), `researchers/` (bull, bear), `managers/` (research_manager, portfolio_manager), `risk_mgmt/` (aggressive, conservative, neutral debators), `trader/`. Each module exposes a `create_<role>(llm)` factory that returns a LangGraph node function.

The **Research Manager**, **Trader**, and **Portfolio Manager** use structured output via `agents/utils/structured.py` — `bind_structured(llm, Schema)` (Pydantic schemas in `agents/schemas.py`) is wrapped with `invoke_structured_or_freetext`, which falls back to plain `llm.invoke` if either binding or invocation fails (e.g. `deepseek-reasoner` has no `tool_choice`).

**Bull and Bear researchers** must end their writeup with a `## Required Assumptions for the Bull/Bear Case` section enumerating eight categories (inflation regime, interest-rate path, geopolitical/war risk, sector demand, regulatory, FX, operational milestones, liquidity) with explicit invalidation signals. This makes each thesis falsifiable.

### LLM clients (`tradingagents/llm_clients/`)
`factory.create_llm_client` dispatches by provider:
- `_OPENAI_COMPATIBLE = ("openai", "xai", "deepseek", "qwen", "glm", "ollama", "openrouter", "groq")` → `OpenAIClient`
- `anthropic` → `AnthropicClient`, `google` → `GoogleClient`, `azure` → `AzureOpenAIClient`

Imports inside the factory are **lazy** so importing the factory (e.g. during test collection) does not force-load every provider SDK or trip on missing keys.

`OpenAIClient` returns one of three `ChatOpenAI` subclasses depending on provider:
- `NormalizedChatOpenAI` (default) — flattens Responses-API typed-block content to a string, defaults `with_structured_output` to `method="function_calling"` to avoid the noisy parse path.
- `DeepSeekChatOpenAI` — round-trips `reasoning_content` between turns (the API returns 400 if you drop it) and refuses `with_structured_output` for `deepseek-reasoner`.
- `GroqChatOpenAI` — see resilience section below; recovers from Groq's `tool_use_failed` quirk.

Native OpenAI uses the Responses API (`use_responses_api=True`); third-party OpenAI-compatible providers use Chat Completions.

`backend_url` in config is **provider-agnostic and defaults to `None`** — the per-provider client picks its own default endpoint. Do not hard-code an OpenAI base URL into the default config.

### Data layer (`tradingagents/dataflows/`)
Vendor-pluggable. `dataflows/interface.py` is the routing surface; `dataflows/config.py` holds the runtime config. Two vendors: `yfinance` (default, no extra key) and `alpha_vantage` (needs `ALPHA_VANTAGE_API_KEY`) — selectable per-category in `config["data_vendors"]` (`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`) with optional per-tool overrides via `config["tool_vendors"]`.

The agent-facing tool functions live in:
- `agents/utils/core_stock_tools.py` — `get_stock_data`
- `agents/utils/technical_indicators_tools.py` — `get_indicators`
- `agents/utils/fundamental_data_tools.py` — `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`
- `agents/utils/news_data_tools.py` — `get_news`, `get_global_news`, `get_insider_transactions`

**Every ticker-taking tool calls `canonical_ticker(symbol_or_ticker)` from `dataflows/config.py`.** This ignores the LLM-supplied symbol and uses the ticker that `propagate()` pinned. Without this, smaller models hallucinate exchange suffixes (`TWLO` → `TWLO.TO` / `TWLO.TW`); with it, tool calls always use the user's input.

`dataflows/utils.safe_ticker_component(ticker)` is mandatory whenever a ticker is used as a path component (results dir, checkpoint DB filename).

### Resilience layers

| Layer | Mechanism | What it catches |
| --- | --- | --- |
| **DNS override** | `dns: 1.1.1.1, 8.8.8.8` in `docker-compose.yml` | ISP DNS sinkholes (e.g. `fc.yahoo.com` returning a black-hole IP) |
| **`yf_retry`** (`dataflows/stockstats_utils.py`) | Up to 6 retries with exponential backoff | yfinance rate limits, connection errors, 5xx, timeouts; permanent errors (404/delisted) skip retry |
| **`route_to_vendor`** (`dataflows/interface.py`) | Per-vendor retries (5× for key metrics, 2× otherwise) before failover | Network blips, rate limits, transient HTTP markers; permanent errors skip retry |
| **Vendor failover** | `yfinance → alpha_vantage` (or vice versa) | Sustained vendor outage |
| **`canonical_ticker`** | Tool layer overrides LLM-supplied symbol | LLMs hallucinating exchange suffixes |
| **`GroqChatOpenAI`** (`llm_clients/openai_client.py`) | Parses Groq's malformed `<function=...>{json}</function>` payload and synthesizes a valid `tool_calls` AIMessage; retries 5× as backup; final fallback returns empty AIMessage so the graph still finishes | Groq's `tool_use_failed` 400 (Llama models intermittently emit legacy syntax instead of OpenAI tool-call schema) |
| **`ToolNode(handle_tool_errors=True)`** | Tool exceptions become `ToolMessage` content | Any unhandled tool exception (404, malformed args, etc.) |
| **`invoke_structured_or_freetext`** (`agents/utils/structured.py`) | Falls back to plain `llm.invoke` if structured-output binding/invocation fails | Provider quirks (e.g. `deepseek-reasoner` has no `tool_choice`) |

The Groq recovery layer is the most subtle: when Groq returns a 400 with `code: 'tool_use_failed'` and a `failed_generation` blob containing `<function=NAME{json}</function>`, `_recover_tool_call_message` parses the function name and JSON args directly and returns an `AIMessage` with structured `tool_calls` — bypassing the API's strict parser entirely. The graph never sees the error.

### Persona overlay (`agents/utils/personas.py`)
Optional investor/economist persona is read from `config["trading_persona"]` (env var `TRADINGAGENTS_PERSONA`). When set, `persona_system_preamble(persona)` returns a system-prompt prefix (display name + ~100-word philosophy fragment) prepended **only** to the Trader and Portfolio Manager system prompts — the two decision-making nodes. Analysts (data extraction) and bull/bear/risk debators (kept neutral for argument quality) deliberately ignore the persona.

The active persona surfaces in three places:
1. **CLI Progress panel title** — `Progress · Persona: <Display Name>`.
2. **CLI header** — `Strategy / Persona: <Display Name>` line.
3. **`complete_report.md`** — top-of-file `## Configuration` block names the persona and quotes its strategy fragment.

Lookup is case-insensitive, accepts spaces/hyphens/underscores, and supports short aliases (`buffett`, `soros`, `druck`, `taleb`). Unknown names log a warning and run with no persona — typos surface but never crash. To add a persona, add a `Persona(...)` entry plus any aliases.

### Persistence (`agents/utils/memory.py`)
`TradingMemoryLog` is an append-only Markdown file (`<!-- ENTRY_END -->` is the entry separator — chosen because LLM prose can't emit HTML comments). Entries are written in `pending` state at the end of `propagate`; they are **resolved** on the next same-ticker run by fetching the realized 5-day return and SPY alpha and writing a one-paragraph reflection. Pending entries never get pruned; resolved entries can rotate via `memory_log_max_entries`.

### Reports bundle (`tradingagents/reports.py`)
`save_run_bundle(final_state, ticker)` writes a structured directory tree under `trading-reports/<TICKER>_<YYYYMMDD_HHMMSS>/`:
```
complete_report.md          # consolidated, with Configuration + Persona block at top
full_state.json             # raw state for downstream tooling
1_analysts/{market,sentiment,news,fundamentals}.md
2_research/{bull,bear,manager}.md
3_trading/trader.md
4_risk/{aggressive,conservative,neutral}.md
5_portfolio/{decision.md, final_trade_decision.md}
```

Both the CLI and `main.py` call this automatically on every run — no save prompts. `trading-reports/` is gitignored and bind-mounted into the container.

### CLI (`cli/`)
`cli/main.py` builds a Rich live-updating layout (progress table, message stream, current-report panel, stats footer). Key pieces:
- `MessageBuffer` initializes `agent_status` and `report_sections` dynamically from the user's analyst selection (`init_for_analysis`). `REPORT_SECTIONS` maps each report field to (controlling analyst, finalizing agent) so a report only counts as "complete" when its finalizing agent transitions to `completed`.
- `MessageBuffer.persona_label` is set after config is finalized; rendered in the Progress panel title and the header.
- `update_analyst_statuses` is called on every chunk and computes status from accumulated state, not just the current chunk.
- `StatsCallbackHandler` (`cli/stats_handler.py`) is passed both as an LLM constructor `callbacks=` and as graph-stream `callbacks=` to track LLM calls, tool calls, and token usage separately.
- After the run, the CLI auto-saves via `save_run_bundle()` and prints the full report to the terminal — no interactive save prompt.

## Conventions worth knowing

- **`load_dotenv()` must run before importing `default_config`.** `DEFAULT_CONFIG["trading_persona"] = os.getenv("TRADINGAGENTS_PERSONA")` is evaluated at import time. `cli/main.py` and `main.py` both load `.env` first; preserve this if you add new entry points.
- **Internal debate stays in English** even when `output_language` is non-English; the language instruction (`get_language_instruction`) is only applied to user-facing agents (analysts and Portfolio Manager) for reasoning quality.
- **Tickers are pass-through.** `build_instrument_context` says "no dot in → no dot out". The data-tool layer additionally enforces this via `canonical_ticker` in case the LLM misbehaves.
- **Msg Clear nodes between analysts** wipe the per-analyst tool-call history (via `RemoveMessage` ids) and inject a `HumanMessage(content="Continue")` placeholder — required for Anthropic compatibility (it rejects empty histories) and prevents tool-message bleed between analysts.
- **Two LLMs per run.** `deep_think_llm` is used by Research Manager and Portfolio Manager; everything else uses `quick_think_llm`. Default is `llama-3.3-70b-versatile` for both.
- **Don't use `llama-3.1-8b-instant` on Groq.** It intermittently emits `<function=...>` syntax instead of OpenAI tool-call schema. The catalog labels it "NOT recommended" and the recovery layer handles it, but choosing a more capable model produces better reasoning end-to-end.
- **Adding a new provider:** add a branch in `llm_clients/factory.py` (lazy import), a client class implementing `BaseLLMClient.get_llm()`, an env-var entry in `_API_KEY_ENV_VARS` in `tests/conftest.py`, and the model list in `llm_clients/model_catalog.py` — `validate_model` auto-derives valid IDs from the catalog.
- **Adding a new analyst:** add a `create_<name>_analyst` factory under `agents/analysts/`, export it in `agents/__init__.py`, register a tool list and `ToolNode` in `TradingAgentsGraph._create_tool_nodes` (with `handle_tool_errors=True`), add a `should_continue_<name>` to `graph/conditional_logic.py`, wire its branch in `GraphSetup.setup_graph`, and add the report field to `AgentState` plus the CLI's `REPORT_SECTIONS` / `ANALYST_MAPPING`.
- **Adding a new data tool:** define the `@tool` function under `agents/utils/<category>_tools.py`, route it through `route_to_vendor` (for retries + failover), use `canonical_ticker(symbol)` to normalize the ticker, and register the vendor implementations in `dataflows/interface.py`'s `VENDOR_METHODS`.
