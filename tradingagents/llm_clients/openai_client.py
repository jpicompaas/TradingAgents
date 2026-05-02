import json
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

logger = logging.getLogger(__name__)


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling. ``with_structured_output`` defaults
    to function-calling so the Responses-API parse path is avoided
    (langchain-openai's parse path emits noisy
    PydanticSerializationUnexpectedValue warnings per call without
    affecting correctness).

    Provider-specific quirks (e.g. DeepSeek's thinking mode) live in
    purpose-built subclasses below so this base class stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Two quirks that don't apply to other OpenAI-compatible providers:

    1. **Thinking-mode round-trip.** When DeepSeek's thinking models return
       a response with ``reasoning_content``, that field must be echoed
       back as part of the assistant message on the next turn or the API
       fails with HTTP 400. ``_create_chat_result`` captures the field on
       receive and ``_get_request_payload`` re-attaches it on send.

    2. **deepseek-reasoner has no tool_choice.** Structured output via
       function-calling is unavailable, so we raise NotImplementedError
       and let the agent factories fall back to free-text generation
       (see ``tradingagents/agents/utils/structured.py``).
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_)):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", [])
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if self.model_name == "deepseek-reasoner":
            raise NotImplementedError(
                "deepseek-reasoner does not support tool_choice; structured "
                "output is unavailable. Agent factories fall back to "
                "free-text generation automatically."
            )
        return super().with_structured_output(schema, method=method, **kwargs)


def _is_groq_tool_use_failed(exc: Exception) -> bool:
    """Detect Groq's `tool_use_failed` 400 error.

    Groq's Llama models intermittently emit tool calls in the legacy
    `<function=name>{json}</function>` text format instead of the OpenAI
    `tool_calls` JSON, and Groq rejects this with HTTP 400 / code
    'tool_use_failed'.
    """
    body = getattr(exc, "body", None) or {}
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict) and err.get("code") == "tool_use_failed":
            return True
    # Fallback: inspect message text. The SDK sometimes surfaces the body
    # only in the stringified message.
    return "tool_use_failed" in str(exc)


def _extract_failed_generation(exc: Exception) -> Optional[str]:
    """Pull the `failed_generation` payload Groq returns alongside the 400."""
    body = getattr(exc, "body", None) or {}
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict):
            fg = err.get("failed_generation")
            if isinstance(fg, str) and fg:
                return fg
    # SDK sometimes only stringifies the body. Try a regex fallback.
    m = re.search(r"'failed_generation':\s*'(.+?)'\s*}\s*}", str(exc), re.DOTALL)
    return m.group(1) if m else None


# Match the Llama legacy tool-call syntax in any of the buggy variants Groq
# has been seen to emit, e.g.:
#   <function=get_news>{"ticker": "NVDA"}</function>
#   <function=get_news{"ticker": "NVDA"}</function>          (no closing >)
#   <function=get_news>"description"{"ticker": "NVDA"}</function>
_LEGACY_FN_PATTERN = re.compile(
    r"<function\s*=\s*([\w_./-]+)[^{<]*?(\{.*?\})\s*</function>",
    re.DOTALL,
)


def _recover_tool_call_message(failed_generation: str) -> Optional[AIMessage]:
    """Parse Groq's malformed `<function=...>{json}</function>` and rebuild a
    proper AIMessage with structured `tool_calls`.

    This sidesteps Groq's strict parser entirely: we extract the model's
    intent ourselves and feed it back into the LangGraph as if the model had
    emitted valid tool calls. Returns None if the payload can't be parsed.
    """
    tool_calls = []
    for match in _LEGACY_FN_PATTERN.finditer(failed_generation):
        name = match.group(1).strip()
        json_blob = match.group(2)
        try:
            args = json.loads(json_blob)
        except json.JSONDecodeError:
            # Try to repair common Llama mistakes (single quotes, trailing
            # commas) before giving up.
            try:
                repaired = json_blob.replace("'", '"')
                repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
                args = json.loads(repaired)
            except json.JSONDecodeError:
                continue
        tool_calls.append({
            "name": name,
            "args": args if isinstance(args, dict) else {},
            "id": f"groq_recovered_{uuid.uuid4().hex[:12]}",
            "type": "tool_call",
        })

    if not tool_calls:
        return None

    return AIMessage(content="", tool_calls=tool_calls)


class GroqChatOpenAI(NormalizedChatOpenAI):
    """Groq-specific ChatOpenAI hardened against `tool_use_failed`.

    Resilience strategy:
    1. **Recover** — when Groq rejects a malformed tool call, parse the
       model's `failed_generation` ourselves and synthesize a valid
       AIMessage with structured `tool_calls`. The LangGraph never even
       sees the error.
    2. **Retry** — if recovery fails (no parseable tool call in the
       failed_generation), retry the underlying call with exponential
       backoff. The model's next sample usually emits valid syntax.
    3. **Degrade gracefully** — if every retry also fails to recover,
       return an empty AIMessage so the analyst node moves on with the
       data it already has rather than crashing the entire run.

    This means a `tool_use_failed` from Groq is, at worst, a single
    skipped tool call — never a fatal error.
    """

    _GROQ_TOOL_USE_RETRIES = 5
    _GROQ_TOOL_USE_BACKOFF = 1.0  # seconds, doubled each attempt (max ~31s total)

    def invoke(self, input, config=None, **kwargs):
        delay = self._GROQ_TOOL_USE_BACKOFF
        last_failed_gen: Optional[str] = None

        for attempt in range(self._GROQ_TOOL_USE_RETRIES + 1):
            try:
                return normalize_content(
                    ChatOpenAI.invoke(self, input, config, **kwargs)
                )
            except Exception as exc:
                if not _is_groq_tool_use_failed(exc):
                    # Not the Groq quirk — let other handlers see it.
                    raise

                last_failed_gen = _extract_failed_generation(exc) or last_failed_gen

                # Step 1: try to recover the model's intent from the
                # malformed payload. Most reliable path — succeeds on
                # the very first failure with no retries needed.
                if last_failed_gen:
                    recovered = _recover_tool_call_message(last_failed_gen)
                    if recovered is not None:
                        logger.warning(
                            "Groq tool_use_failed on attempt %d — recovered %d "
                            "tool call(s) by parsing failed_generation; continuing.",
                            attempt + 1,
                            len(recovered.tool_calls),
                        )
                        return recovered

                # Step 2: couldn't parse — retry the API call.
                if attempt < self._GROQ_TOOL_USE_RETRIES:
                    logger.warning(
                        "Groq tool_use_failed (attempt %d/%d); could not parse "
                        "failed_generation, retrying in %.1fs",
                        attempt + 1,
                        self._GROQ_TOOL_USE_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                # Step 3: exhausted retries AND couldn't parse. Don't crash
                # the graph — return whatever text the model produced so the
                # analyst can move on with what it already has.
                logger.error(
                    "Groq tool_use_failed: exhausted %d retries and could not "
                    "parse failed_generation. Returning empty AIMessage so the "
                    "graph continues. Last failed_generation: %r",
                    self._GROQ_TOOL_USE_RETRIES,
                    (last_failed_gen or "")[:500],
                )
                return AIMessage(
                    content=last_failed_gen or "",
                    additional_kwargs={"groq_tool_use_failed": True},
                )

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4/", "ZHIPU_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth. An explicit base_url on the
        # client (e.g. a corporate proxy) takes precedence over the
        # provider default so users can route through their own gateway.
        if self.provider in _PROVIDER_CONFIG:
            default_base, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = self.base_url or default_base
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        # Provider-specific quirks live in their own subclasses so the base
        # NormalizedChatOpenAI stays free of provider-specific branches.
        if self.provider == "deepseek":
            chat_cls = DeepSeekChatOpenAI
        elif self.provider == "groq":
            chat_cls = GroqChatOpenAI
        else:
            chat_cls = NormalizedChatOpenAI
        return chat_cls(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
