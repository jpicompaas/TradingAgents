# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TradingAgents is a multi-agent LLM financial trading framework built on **LangGraph**. A team of LLM agents (analysts → researchers → trader → risk debators → portfolio manager) collaboratively analyzes a ticker for a given date and emits a final trade decision. Entry points: `cli/main.py` (interactive Rich/Typer CLI, exposed as the `tradingagents` command) and `main.py` / `tradingagents.graph.trading_graph.TradingAgentsGraph` (programmatic).

## Commands

### Install / run
```bash
pip install .                                  # install package + CLI
tradingagents                                  # interactive CLI (alias for cli.main:app)
tradingagents analyze --checkpoint             # enable LangGraph resume
tradingagents analyze --clear-checkpoints      # wipe all per-ticker checkpoint DBs
python main.py                                 # programmatic example (NVDA, hard-coded date)
docker compose run --rm tradingagents          # containerized run (reads .env)
docker compose --profile ollama run --rm tradingagents-ollama   # local-models variant
```

### Tests
```bash
pytest                                         # full suite (pyproject sets testpaths=tests, -ra --strict-markers)
pytest tests/test_signal_processing.py         # single file
pytest tests/test_structured_agents.py::test_name -v
pytest -m unit                                 # markers: unit, integration, smoke
python scripts/smoke_structured_output.py      # live-API smoke for structured-output agents (needs real keys)
```

`tests/conftest.py` autouses a `_dummy_api_keys` fixture that injects placeholder values for every provider env var, and exposes a `mock_llm_client` fixture that patches `tradingagents.llm_clients.factory.create_llm_client`. Use these to keep tests offline — do not reach out to real APIs in unit tests.

### Environment
Copy `.env.example` to `.env` and set the keys for the providers you use (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `ZHIPU_API_KEY`, `OPENROUTER_API_KEY`, `ALPHA_VANTAGE_API_KEY`). For Azure / AWS Bedrock, copy `.env.enterprise.example` to `.env.enterprise` (the CLI loads it after `.env` with `override=False`).

Runtime overrides:
- `TRADINGAGENTS_RESULTS_DIR` — where per-run JSON state and reports are written (default `~/.tradingagents/logs`).
- `TRADINGAGENTS_CACHE_DIR` — base for checkpoint SQLite DBs (default `~/.tradingagents/cache`; checkpoints land under `<cache>/checkpoints/<TICKER>.db`).
- `TRADINGAGENTS_MEMORY_LOG_PATH` — append-only markdown decision log (default `~/.tradingagents/memory/trading_memory.md`).

## Architecture

### Graph orchestration (`tradingagents/graph/`)
`TradingAgentsGraph` (`graph/trading_graph.py`) is the public façade. Construction wires:
1. **LLM clients** via `create_llm_client(provider, model, base_url, **kwargs)` — both a `deep_thinking_llm` and a `quick_thinking_llm` are produced. Provider-specific thinking knobs (`google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`) are gated in `_get_provider_kwargs` so they only flow to the right client.
2. **Tool nodes** — four `langgraph.prebuilt.ToolNode`s keyed by analyst type (`market`, `social`, `news`, `fundamentals`) wrapping the abstract tool functions in `agents/utils/agent_utils.py`.
3. **Workflow** — `GraphSetup.setup_graph(selected_analysts)` builds a `StateGraph(AgentState)`. Analyst nodes are connected sequentially with conditional `should_continue_*` edges back to their tool node and a `Msg Clear *` (a `RemoveMessage` node) before handing off to the next analyst or to **Bull Researcher**. The debate loop alternates Bull/Bear via `should_continue_debate` until `Research Manager` resolves it, which then flows Trader → Aggressive → Conservative → Neutral risk debators (round-robin via `should_continue_risk_analysis`) → Portfolio Manager → END.
4. **Checkpointer** — when `config["checkpoint_enabled"]`, `propagate` recompiles `self.workflow` with a per-ticker `SqliteSaver` (`graph/checkpointer.py`). The thread ID is `sha256("<TICKER>:<date>")[:16]` so the same ticker+date resumes and a different date starts fresh. Checkpoints are cleared on successful completion.

`propagate(ticker, date)` is the entry point. It:
1. Resolves any **pending** memory-log entries for the ticker (fetches realized return + SPY alpha via yfinance, runs `Reflector.reflect_on_final_decision`, writes reflections in a single batch).
2. Injects `past_context` (recent same-ticker decisions + cross-ticker lessons) from the memory log into the initial state so the Portfolio Manager prompt carries forward what worked.
3. Runs the graph (`graph.stream` in debug mode, otherwise `graph.invoke`).
4. Logs full state to `<results_dir>/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json`, writes a new pending entry to the memory log, and returns `(final_state, processed_signal)`.

### Agent state (`agents/utils/agent_states.py`)
`AgentState` extends `MessagesState` with named report fields (`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`, `investment_plan`, `trader_investment_plan`, `final_trade_decision`) plus nested debate states (`InvestDebateState`, `RiskDebateState`). Every analyst writes to its dedicated report field; the CLI keys off these field names to drive its live progress display.

### Agents (`tradingagents/agents/`)
Organized by team — `analysts/` (market, social, news, fundamentals), `researchers/` (bull, bear), `managers/` (research_manager, portfolio_manager), `risk_mgmt/` (aggressive, conservative, neutral debators), `trader/`. Each module exposes a `create_<role>(llm)` factory that returns a LangGraph node function.

The **Research Manager**, **Trader**, and **Portfolio Manager** use structured output via `agents/utils/structured.py` — they call `bind_structured(llm, Schema)` (Pydantic schemas live in `agents/schemas.py`) and run through `invoke_structured_or_freetext`, which falls back to plain `llm.invoke` if either binding or invocation fails (e.g. `deepseek-reasoner` has no `tool_choice`). The fallback is intentional and silent at the call site; warnings are logged once at the helper.

### LLM clients (`tradingagents/llm_clients/`)
`factory.create_llm_client` dispatches by provider:
- `_OPENAI_COMPATIBLE = ("openai", "xai", "deepseek", "qwen", "glm", "ollama", "openrouter")` → `OpenAIClient`
- `anthropic` → `AnthropicClient`, `google` → `GoogleClient`, `azure` → `AzureOpenAIClient`

Important: imports inside the factory are **lazy** so importing the factory (e.g. during test collection) does not force-load every provider SDK or trip on missing keys.

`OpenAIClient` returns `NormalizedChatOpenAI` — a `ChatOpenAI` subclass that (a) flattens Responses-API typed-block content to a string in `invoke`, and (b) defaults `with_structured_output` to `method="function_calling"` to avoid the noisy parse path. Native OpenAI uses the Responses API (`use_responses_api=True`); third-party OpenAI-compatible providers use Chat Completions. DeepSeek thinking models go through `DeepSeekChatOpenAI`, which round-trips `reasoning_content` between turns (the API returns 400 if you drop it) and refuses `with_structured_output` for `deepseek-reasoner`.

`backend_url` in config is **provider-agnostic and defaults to `None`** — the per-provider client picks its own default endpoint. Do not hard-code an OpenAI base URL into the default config; previously this leaked OpenAI's `/v1` into Gemini and produced malformed URLs (see commit 4016fd4).

### Data layer (`tradingagents/dataflows/`)
Vendor-pluggable. `dataflows/interface.py` is the routing surface; `dataflows/config.py` holds the runtime config (set via `set_config(config)` from `TradingAgentsGraph.__init__`). Two vendors today — `yfinance` (default, no extra key) and `alpha_vantage` (needs `ALPHA_VANTAGE_API_KEY`) — selectable per-category in `config["data_vendors"]` (`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`) with optional per-tool overrides via `config["tool_vendors"]`. The agent-facing tool functions in `agents/utils/agent_utils.py` (`get_stock_data`, `get_indicators`, `get_fundamentals`, `get_news`, …) dispatch through this routing layer.

`dataflows/utils.safe_ticker_component(ticker)` is mandatory whenever a ticker is used as a path component (results dir, checkpoint DB filename) — see commit 2c97bad. Don't bypass it.

### Persona overlay (`agents/utils/personas.py`)
Optional investor/economist persona is read from `config["trading_persona"]` (env var `TRADINGAGENTS_PERSONA`). When set, `persona_system_preamble(persona)` returns a short system-prompt prefix (display name + ~100-word philosophy fragment) that is prepended **only** to the Trader and Portfolio Manager system prompts — the two decision-making nodes. Analysts (data extraction) and bull/bear/risk debators (kept neutral for argument quality) deliberately ignore the persona. Lookup is case-insensitive, accepts spaces/hyphens/underscores, and supports short aliases (e.g. `buffett`, `soros`, `druck`, `taleb`). Unknown names log a warning and run with no persona — typos surface but never crash. The registry is the single source of truth; to add a persona, add a `Persona(...)` entry plus any aliases in that file.

### Persistence (`agents/utils/memory.py`)
`TradingMemoryLog` is an append-only Markdown file (`<!-- ENTRY_END -->` is the entry separator — chosen because LLM prose can't emit HTML comments). Entries are written in `pending` state at the end of `propagate`; they are **resolved** on the next same-ticker run by fetching the realized 5-day return and SPY alpha and writing a one-paragraph reflection. Pending entries never get pruned; resolved entries can rotate via `memory_log_max_entries`. Cross-ticker entries accumulate until that ticker is run again — by design, to avoid stampeding yfinance on every run.

### CLI (`cli/`)
`cli/main.py` builds a Rich live-updating layout (progress table, message stream, current-report panel, stats footer). Key pieces:
- `MessageBuffer` initializes `agent_status` and `report_sections` dynamically from the user's analyst selection (`init_for_analysis`). `REPORT_SECTIONS` maps each report field to (controlling analyst, finalizing agent) so a report only counts as "complete" when its finalizing agent transitions to `completed` — this prevents interim debate updates from being mis-counted (see `get_completed_reports_count`).
- `update_analyst_statuses` is called on every chunk and computes status from accumulated state, not just the current chunk, so resumed/streamed states converge correctly.
- `StatsCallbackHandler` (`cli/stats_handler.py`) is passed both as an LLM constructor `callbacks=` and as graph-stream `callbacks=` to track LLM calls, tool calls, and token usage separately.
- After the run, the CLI prompts to save a structured report (`reports/<TICKER>_<timestamp>/{1_analysts,2_research,3_trading,4_risk,5_portfolio}/...md` plus a consolidated `complete_report.md`) and then optionally prints the full report to the terminal.

## Conventions worth knowing

- **Internal debate stays in English** even when `output_language` is non-English; the language instruction (`get_language_instruction`) is only applied to user-facing agents (analysts and Portfolio Manager) for reasoning quality.
- **Exchange-suffixed tickers must be preserved end-to-end.** `build_instrument_context` injects an explicit instruction into agent prompts; agents must use the exact ticker (e.g. `CNC.TO`, `7203.T`, `0700.HK`) in every tool call.
- **Msg Clear nodes between analysts** wipe the per-analyst tool-call history (via `RemoveMessage` ids) and inject a `HumanMessage(content="Continue")` placeholder — this is required for Anthropic compatibility (it rejects empty histories) and prevents tool-message bleed between analysts.
- **Two LLMs per run.** `deep_think_llm` is used by Research Manager and Portfolio Manager; everything else uses `quick_think_llm`. Pick a stronger model for `deep_think_llm` when reasoning quality matters more than latency/cost.
- **Adding a new provider:** add a branch in `llm_clients/factory.py` (lazy import), a client class implementing `BaseLLMClient.get_llm()`, an env-var entry in `_API_KEY_ENV_VARS` in `tests/conftest.py`, and the model list in `llm_clients/model_catalog.py` for `validate_model`.
- **Adding a new analyst:** add a `create_<name>_analyst` factory under `agents/analysts/`, export it in `agents/__init__.py`, register a tool list and `ToolNode` in `TradingAgentsGraph._create_tool_nodes`, add a `should_continue_<name>` to `graph/conditional_logic.py`, and wire its branch in `GraphSetup.setup_graph` — also add the report field to `AgentState` and to the CLI's `REPORT_SECTIONS` / `ANALYST_MAPPING`.
