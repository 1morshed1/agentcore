"""Raw LLM clients — talking to OpenAI and Anthropic with no SDK.

This is the module that proves the central point of the project: an "LLM API
call" is just an HTTP POST with a JSON body. There is no framework magic. We use
``httpx`` (a modern ``requests``-like HTTP client) and nothing else.

Design: one interface, many providers
-------------------------------------
Every provider speaks a slightly different dialect (different URLs, auth headers,
request shapes, and — most annoyingly — different ways of representing *tool
calls*). We hide all of that behind a single abstract base class,
:class:`LLMClient`, whose :meth:`~LLMClient.complete` method always returns the
same normalised :class:`LLMResponse`, no matter which provider produced it.

The rest of the library (the Agent, the Workflow) only ever sees the normalised
shape. That is the value of an abstraction layer: the ReAct loop is written once
and works with any provider.

Providers implemented here:

* :class:`OpenAIClient`     — the Chat Completions API.
* :class:`AnthropicClient`  — the Messages API.
* :class:`MockLLM`          — a scriptable fake for tests and offline demos.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .errors import LLMError

# ---------------------------------------------------------------------------
# Normalised data shapes
# ---------------------------------------------------------------------------
# These are the vendor-neutral objects the rest of the library works with. Each
# provider client is responsible for translating its own wire format into these.


@dataclass
class ToolCall:
    """A single request from the model to run one tool.

    Attributes:
        id: A unique identifier for this call. When we send the tool's result
            back, we tag it with this id so the model can pair request↔result.
        name: The name of the tool to run.
        arguments: The arguments, already parsed into a dict.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    """Token accounting for one LLM call.

    ``prompt_tokens`` are what we sent (the whole conversation so far);
    ``completion_tokens`` are what the model generated. Cost is driven almost
    entirely by these two numbers, so we track them everywhere.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        # Lets us sum usage across many calls with the built-in `sum()` idiom.
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class LLMResponse:
    """The normalised result of one call to any provider.

    Attributes:
        content: The model's text output (may be empty if it only made tool
            calls).
        tool_calls: Any tools the model wants to run this turn. Empty list means
            "no tools requested" — which, in a ReAct loop, signals a final
            answer.
        usage: Token counts for this call.
        raw_message: The provider's original assistant message, in that
            provider's own format. We keep it so it can be appended verbatim to
            the conversation history (providers are picky about getting their own
            message shape back on the next turn).
        model: The model id that produced this response.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw_message: dict[str, Any] = field(default_factory=dict)
    model: str = ""


# ---------------------------------------------------------------------------
# Pricing table (USD per 1,000,000 tokens)
# ---------------------------------------------------------------------------
# Cost tracking needs prices. We keep a small, editable table here. These are
# illustrative list prices; update them as vendor pricing changes. Keys are model
# ids (prefix-matched, so "gpt-4o-2024-08-06" matches "gpt-4o").
_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    # model prefix        (input $/Mtok, output $/Mtok)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic — current generation
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    # Anthropic — older ids
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
}


def estimate_cost(model: str, usage: Usage) -> float:
    """Estimate the USD cost of one call from its token usage.

    We match ``model`` against the longest matching prefix in the pricing table,
    so dated model ids (``gpt-4o-2024-08-06``) resolve to their family price.
    Unknown models cost ``0.0`` — better to under-report than to invent a number.
    """
    match = None
    for prefix in sorted(_PRICING_PER_MTOK, key=len, reverse=True):
        if model.startswith(prefix):
            match = prefix
            break
    if match is None:
        return 0.0
    in_price, out_price = _PRICING_PER_MTOK[match]
    return (usage.prompt_tokens / 1_000_000) * in_price + (
        usage.completion_tokens / 1_000_000
    ) * out_price


# ---------------------------------------------------------------------------
# The abstract base client
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract interface every provider client implements.

    Subclasses must implement:

    * :meth:`complete` — send messages + tool schemas, return an
      :class:`LLMResponse`.
    * :meth:`format_tool_result_message` — build the provider-specific message
      that carries a tool's result back to the model.
    * :attr:`tool_schema_dialect` — ``"openai"`` or ``"anthropic"``, so the agent
      knows which ``Tool`` serialiser to use.

    The base class provides :meth:`_request_with_retries`, shared retry logic that
    every HTTP-backed provider benefits from.
    """

    #: which Tool.to_*_schema() the agent should call for this provider
    tool_schema_dialect: str = "openai"

    def __init__(self, model: str, *, max_retries: int = 3, timeout: float = 60.0):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Run one completion. Return a normalised :class:`LLMResponse`."""

    @abstractmethod
    def format_tool_result_message(self, tool_call: ToolCall, result: str) -> dict[str, Any]:
        """Build the message that reports a tool's result back to the model.

        OpenAI expects a ``{"role": "tool", ...}`` message; Anthropic expects a
        ``{"role": "user", "content": [{"type": "tool_result", ...}]}`` message.
        The agent calls this without needing to know which.
        """

    # -- shared retry helper -------------------------------------------------
    def _request_with_retries(self, do_request: Callable[[], httpx.Response]) -> httpx.Response:
        """Execute an HTTP request, retrying transient failures with backoff.

        We retry on:
        * connection/timeout errors (network blips), and
        * HTTP 429 (rate limit) and 5xx (server errors).

        We do **not** retry 4xx auth/validation errors (e.g. 401) — those will
        never succeed on retry, so we fail fast with a clear message.

        Backoff is exponential: 1s, 2s, 4s, ... This is the same pattern every
        production HTTP client uses; a fixed delay would hammer a struggling
        server, while exponential backoff gives it room to recover.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = do_request()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc  # network-level failure -> retry
            else:
                if response.status_code < 400:
                    return response
                if response.status_code in (429,) or response.status_code >= 500:
                    last_exc = LLMError(
                        f"HTTP {response.status_code} from provider: {response.text[:200]}"
                    )
                else:
                    # Non-retryable (401 auth, 400 bad request, ...). Fail now.
                    raise LLMError(
                        f"HTTP {response.status_code} from provider: {response.text[:500]}"
                    )

            if attempt < self.max_retries:
                time.sleep(2**attempt)  # 1s, 2s, 4s, ...

        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# OpenAI (Chat Completions API)
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    """Talks to OpenAI's ``/v1/chat/completions`` endpoint with raw HTTP.

    The request body is just ``{"model", "messages", "tools", "temperature"}``.
    The interesting translation work is in reading the response: tool calls come
    back nested under ``choices[0].message.tool_calls`` with arguments as a JSON
    *string* that we must parse.
    """

    tool_schema_dialect = "openai"
    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None, **kwargs: Any):
        super().__init__(model, **kwargs)
        # Read the key from the environment by default so it never lives in code.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set. Export it or pass api_key=...")

    def complete(self, messages, tools=None, *, temperature=0.1) -> LLMResponse:
        # `None` tools must be omitted entirely; sending "tools": null is an error.
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools

        def do_request() -> httpx.Response:
            with httpx.Client(timeout=self.timeout) as client:
                return client.post(
                    self.BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )

        data = self._request_with_retries(do_request).json()
        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Translate OpenAI's response JSON into our normalised LLMResponse."""
        # Import here to avoid a circular import (tools imports nothing from us,
        # but keeping this local is tidy and cheap).
        from .tools import parse_tool_arguments

        message = data["choices"][0]["message"]
        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            tool_calls.append(
                ToolCall(
                    id=call["id"],
                    name=fn["name"],
                    arguments=parse_tool_arguments(fn.get("arguments", "{}")),
                )
            )

        usage_raw = data.get("usage", {}) or {}
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
            ),
            raw_message=message,
            model=data.get("model", self.model),
        )

    def format_tool_result_message(self, tool_call: ToolCall, result: str) -> dict[str, Any]:
        # OpenAI: one message per tool result, role="tool", linked by id.
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        }


# ---------------------------------------------------------------------------
# Anthropic (Messages API)
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    """Talks to Anthropic's ``/v1/messages`` endpoint with raw HTTP.

    Anthropic differs from OpenAI in several ways we must handle:

    * The **system prompt is a top-level field**, not a message with
      ``role: "system"``. So we lift any system message out of ``messages``.
    * ``max_tokens`` is **required**.
    * **Sampling parameters are not sent.** Current Claude models reject
      ``temperature`` / ``top_p`` / ``top_k`` with HTTP 400, so ``temperature``
      is accepted for interface compatibility and ignored here.
    * Current models may return ``thinking`` blocks alongside text. They are
      skipped when extracting text but kept in ``raw_message``, because the API
      expects them echoed back unchanged on the next turn.
    * Content is a **list of typed blocks** (``text`` blocks and ``tool_use``
      blocks), not a single string. Tool-call arguments arrive already parsed as a
      dict (nice — no JSON string to decode).
    * A tool result is sent as a ``user`` message containing a ``tool_result``
      block.
    """

    tool_schema_dialect = "anthropic"
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        *,
        api_key: str | None = None,
        max_tokens: int = 16000,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Export it or pass api_key=...")
        self.max_tokens = max_tokens

    def complete(self, messages, tools=None, *, temperature=0.1) -> LLMResponse:
        # Separate the system prompt from the conversation turns.
        system_prompt = ""
        convo: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                convo.append(msg)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": convo,
            "max_tokens": self.max_tokens,
            # No "temperature": rejected by current Claude models (see class doc).
        }
        if system_prompt:
            body["system"] = system_prompt
        if tools:
            body["tools"] = tools

        def do_request() -> httpx.Response:
            with httpx.Client(timeout=self.timeout) as client:
                return client.post(
                    self.BASE_URL,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": self.API_VERSION,
                        "Content-Type": "application/json",
                    },
                    json=body,
                )

        data = self._request_with_retries(do_request).json()
        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Translate Anthropic's response JSON into our normalised LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}) or {},
                    )
                )

        usage_raw = data.get("usage", {}) or {}
        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            usage=Usage(
                prompt_tokens=usage_raw.get("input_tokens", 0),
                completion_tokens=usage_raw.get("output_tokens", 0),
            ),
            # Anthropic wants the assistant's *content blocks* echoed back, so we
            # reconstruct the assistant message from them.
            raw_message={"role": "assistant", "content": data.get("content", [])},
            model=data.get("model", self.model),
        )

    def format_tool_result_message(self, tool_call: ToolCall, result: str) -> dict[str, Any]:
        # Anthropic: a user message carrying a tool_result content block.
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result,
                }
            ],
        }


# ---------------------------------------------------------------------------
# Mock client — for tests and offline demos
# ---------------------------------------------------------------------------


class MockLLM(LLMClient):
    """A scriptable fake LLM. Never makes a network call.

    Why this exists: you must be able to test an agent framework **without** an
    API key, without a network, and *deterministically*. So ``MockLLM`` returns a
    pre-programmed sequence of responses.

    You give it a list of :class:`LLMResponse` objects (or a callable that
    computes the next one from the message history). Each call to
    :meth:`complete` returns the next scripted response.

    Example::

        llm = MockLLM(responses=[
            LLMResponse(content="", tool_calls=[ToolCall("1", "add", {"a": 2, "b": 3})]),
            LLMResponse(content="The answer is 5."),
        ])

    The first call asks to run the ``add`` tool; after the agent feeds back the
    result, the second call gives the final answer. This is exactly one ReAct
    iteration, fully under test control.
    """

    tool_schema_dialect = "openai"

    def __init__(
        self,
        responses: list[LLMResponse] | Callable[[list[dict[str, Any]]], LLMResponse] | None = None,
        model: str = "mock-model",
    ):
        super().__init__(model)
        self._responses = responses or []
        self._index = 0
        #: every message list passed to complete(), captured for assertions
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages, tools=None, *, temperature=0.1) -> LLMResponse:
        # Record the input so tests can assert on what the agent sent.
        self.calls.append(list(messages))

        if callable(self._responses):
            return self._responses(messages)

        if self._index >= len(self._responses):
            raise LLMError(
                "MockLLM ran out of scripted responses. Did the agent loop more "
                "times than you scripted for?"
            )
        response = self._responses[self._index]
        self._index += 1
        if not response.model:
            response.model = self.model
        return response

    def format_tool_result_message(self, tool_call: ToolCall, result: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call.id, "content": result}
