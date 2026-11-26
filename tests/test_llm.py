"""Tests for the LLM clients.

We never hit a real API. Instead we:

* use `MockLLM` directly, and
* mount a fake `httpx` transport (`httpx.MockTransport`) to test that the real
  `OpenAIClient` / `AnthropicClient` correctly *parse* provider responses and
  *build* the right request bodies — all offline.
"""

from __future__ import annotations

import httpx
import pytest

from agentcore import (
    AnthropicClient,
    LLMResponse,
    MockLLM,
    OpenAIClient,
    ToolCall,
    Usage,
    estimate_cost,
)
from agentcore.errors import LLMError


# ---------------------------------------------------------------------------
# Usage & cost
# ---------------------------------------------------------------------------
def test_usage_addition_and_total():
    u = Usage(prompt_tokens=10, completion_tokens=5) + Usage(prompt_tokens=2, completion_tokens=3)
    assert u.prompt_tokens == 12
    assert u.completion_tokens == 8
    assert u.total_tokens == 20


def test_estimate_cost_matches_longest_prefix():
    # gpt-4o-mini is cheaper than gpt-4o; the dated id must resolve to gpt-4o.
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=0)
    assert estimate_cost("gpt-4o-mini", usage) == pytest.approx(0.15)
    assert estimate_cost("gpt-4o-2024-08-06", usage) == pytest.approx(2.50)
    # Unknown model -> zero, never a guess.
    assert estimate_cost("totally-unknown", usage) == 0.0


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------
def test_mock_returns_scripted_responses_in_order():
    llm = MockLLM(responses=[LLMResponse(content="first"), LLMResponse(content="second")])
    assert llm.complete([{"role": "user", "content": "hi"}]).content == "first"
    assert llm.complete([{"role": "user", "content": "hi"}]).content == "second"


def test_mock_records_calls():
    llm = MockLLM(responses=[LLMResponse(content="ok")])
    msgs = [{"role": "user", "content": "hi"}]
    llm.complete(msgs)
    assert llm.calls == [msgs]


def test_mock_runs_out_of_responses():
    llm = MockLLM(responses=[LLMResponse(content="only one")])
    llm.complete([])
    with pytest.raises(LLMError):
        llm.complete([])


def test_mock_callable_responses():
    # A callable can compute the next response from the message history.
    llm = MockLLM(responses=lambda messages: LLMResponse(content=f"saw {len(messages)} msgs"))
    assert llm.complete([{"role": "user", "content": "a"}]).content == "saw 1 msgs"


# ---------------------------------------------------------------------------
# OpenAIClient — parse a canned response through a fake transport
# ---------------------------------------------------------------------------
def _make_openai_client(handler) -> OpenAIClient:
    client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
    # Monkeypatch: replace the retry helper to use a MockTransport-backed client.
    transport = httpx.MockTransport(handler)

    def _request_with_retries(do_request):
        with httpx.Client(transport=transport) as c:
            return c.post("https://api.openai.com/v1/chat/completions", json={})

    client._request_with_retries = _request_with_retries  # type: ignore[assignment]
    return client


def test_openai_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )

    client = _make_openai_client(handler)
    resp = client.complete([{"role": "user", "content": "add 2 and 3"}])
    assert resp.content == ""
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "add"
    assert call.arguments == {"a": 2, "b": 3}  # JSON string was parsed to a dict
    assert resp.usage.prompt_tokens == 11


def test_openai_tool_result_message_shape():
    client = OpenAIClient(model="gpt-4o-mini", api_key="k")
    msg = client.format_tool_result_message(ToolCall("call_1", "add", {}), "5")
    assert msg == {"role": "tool", "tool_call_id": "call_1", "content": "5"}


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------
def test_anthropic_parses_content_blocks_and_tool_use():
    client = AnthropicClient(model="claude-haiku-5-5", api_key="k")
    data = {
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "text", "text": "Let me calculate."},
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2, "b": 3}},
        ],
        "usage": {"input_tokens": 20, "output_tokens": 9},
    }
    resp = client._parse_response(data)
    assert resp.content == "Let me calculate."
    assert resp.tool_calls[0].name == "add"
    assert resp.tool_calls[0].arguments == {"a": 2, "b": 3}  # already a dict
    assert resp.usage.prompt_tokens == 20  # input_tokens -> prompt_tokens
    assert resp.usage.completion_tokens == 9


def test_anthropic_tool_result_message_shape():
    client = AnthropicClient(model="claude-haiku-5-5", api_key="k")
    msg = client.format_tool_result_message(ToolCall("toolu_1", "add", {}), "5")
    assert msg["role"] == "user"
    assert msg["content"][0]["type"] == "tool_result"
    assert msg["content"][0]["tool_use_id"] == "toolu_1"


def test_missing_api_key_raises():
    with pytest.raises(LLMError):
        OpenAIClient(model="gpt-4o-mini", api_key="")


def test_anthropic_request_body_omits_sampling_params(monkeypatch):
    """Current Claude models 400 on temperature/top_p/top_k, so none are sent."""
    sent: dict = {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers, json):
            sent.update(json)
            return httpx.Response(
                200,
                json={"model": json["model"], "content": [{"type": "text", "text": "hi"}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    client = AnthropicClient(api_key="k")
    resp = client.complete(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}],
        temperature=0.7,
    )
    assert resp.content == "hi"
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["system"] == "be brief"
    assert all(m["role"] != "system" for m in sent["messages"])
    for param in ("temperature", "top_p", "top_k"):
        assert param not in sent


def test_anthropic_thinking_blocks_skipped_in_text_but_kept_for_echo():
    client = AnthropicClient(api_key="k")
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Answer."},
    ]
    resp = client._parse_response({"content": blocks})
    assert resp.content == "Answer."
    assert resp.raw_message == {"role": "assistant", "content": blocks}


def test_current_claude_pricing():
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert estimate_cost("claude-haiku-5-5", usage) == pytest.approx(0.60)
    assert estimate_cost("claude-sonnet-5-5", usage) == pytest.approx(12.00)
    assert estimate_cost("claude-opus-5-5", usage) == pytest.approx(24.00)
