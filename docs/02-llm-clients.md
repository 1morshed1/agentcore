# 02 — Talking to the LLM by Hand

Pairs with: `agentcore/llm.py`

## The whole "API call" is one HTTP POST

This is the myth-busting module. When you strip away every SDK, calling an LLM is:

```
POST https://api.openai.com/v1/chat/completions
Authorization: Bearer sk-...
Content-Type: application/json

{ "model": "gpt-4o-mini", "messages": [...], "tools": [...] }
```

You get back JSON. That's the entire interaction. `agentcore` uses `httpx` to
send that POST and Python's built-in JSON handling to read the reply. No
`openai`, no `anthropic`, no `langchain`.

## One interface, many providers

Different vendors format things differently, so we define an **abstract base
class** `LLMClient` with three things every provider must supply:

```python
class LLMClient(ABC):
    tool_schema_dialect: str          # "openai" or "anthropic"
    def complete(messages, tools) -> LLMResponse: ...
    def format_tool_result_message(tool_call, result) -> dict: ...
```

Everything else in the library talks only to this interface and only ever
receives a **normalised `LLMResponse`**. That's the payoff: the ReAct loop is
written once and works with OpenAI, Anthropic, or the mock — none of them leak
their quirks upward.

### The normalised shapes

- `LLMResponse` — `{ content, tool_calls, usage, raw_message, model }`
- `ToolCall` — `{ id, name, arguments(dict) }`
- `Usage` — `{ prompt_tokens, completion_tokens }`

The single most important normalisation is **tool calls**, because that's where
the two providers diverge the most.

## OpenAI vs Anthropic: the differences we absorb

| Concern | OpenAI (Chat Completions) | Anthropic (Messages) |
|---|---|---|
| System prompt | a message with `role: "system"` | a **top-level `system` field** |
| `max_tokens` | optional | **required** |
| Assistant content | a single string | a **list of typed blocks** (`text`, `tool_use`) |
| Tool call args | a **JSON string** you must `json.loads` | an **already-parsed dict** |
| Tool result | `{role: "tool", tool_call_id, content}` | `{role: "user", content: [{type: "tool_result", ...}]}` |
| Auth header | `Authorization: Bearer <key>` | `x-api-key: <key>` + `anthropic-version` |

`OpenAIClient` and `AnthropicClient` each translate their wire format into the
same `LLMResponse`. If you ever add a third provider (say, a local model server),
you write one more subclass and nothing else changes.

### Why keep `raw_message`?

When the model makes a tool call, we must append **its own assistant message**
back into the conversation before we send the tool result — providers validate
that the shapes match what they emitted. Rather than try to reconstruct it
perfectly, we stash the provider's original message on `LLMResponse.raw_message`
and append that verbatim. Small detail, saves a lot of "invalid message" errors.

## Retries and backoff

Networks and rate limits are facts of life, so `_request_with_retries` wraps
every call:

- **Retry** on connection/timeout errors, HTTP `429` (rate limit), and `5xx`
  (server errors).
- **Don't retry** on `401`/`400` — an auth or validation error won't fix itself,
  so we fail fast with a readable message.
- **Backoff** is exponential: 1s, 2s, 4s. A fixed retry delay would hammer an
  already-struggling server; exponential backoff is the standard, polite pattern.

## Cost tracking

`estimate_cost(model, usage)` multiplies token counts by a per-model price from a
small table (`_PRICING_PER_MTOK`), matching the **longest model-id prefix** so
that a dated id like `gpt-4o-2024-08-06` still resolves to the `gpt-4o` price.
Unknown models cost `0.0` — we'd rather under-report than invent a number. This
feeds the tracer (note 04).

## The `MockLLM` — the most useful class for *you*

You cannot (and must not) hit a paid API in a test suite. `MockLLM` is a fake
that returns a **scripted list of `LLMResponse`s**, one per call:

```python
llm = MockLLM(responses=[
    LLMResponse(content="", tool_calls=[ToolCall("1", "add", {"a": 2, "b": 3})]),
    LLMResponse(content="The answer is 5."),
])
```

Call 1 asks to run `add`; after the agent feeds back the result, call 2 gives the
final answer. That's one full ReAct iteration under total test control. `MockLLM`
also records every `messages` list it received (`.calls`), so a test can assert
*exactly* what the agent sent — including that the tool result was appended
correctly. Every test in this repo runs offline because of this class.

> **Design lesson:** design for a fake from the start. Because `complete()` is the
> single choke point for all model I/O, faking it makes the entire library
> testable. If model calls were scattered across the codebase, this would be
> impossible.

Next: [03 — Memory & message shapes](./03-memory.md).
