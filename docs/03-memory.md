# 03 — Memory & Message Shapes

Pairs with: `agentcore/memory.py`

## The one thing to internalise

**An LLM has no memory.** The API is stateless. The model does not "remember"
your previous message — you resend the entire conversation on every single call.
What we call "memory" is just:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's 2 + 2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And times 10?"},
]
```

Each turn we append to this list and POST the whole thing again. `Memory` is a
thin wrapper around exactly this list.

## Why wrap a list at all?

Three reasons, all about giving the rest of the code *one place* to depend on:

1. **A single owner of conversation state.** The agent reads and writes memory in
   one spot, so the ReAct loop stays clean.
2. **Serialisation.** `to_dict()` / `from_dict()` let us checkpoint a
   conversation to disk and resume it later (see note 07). If memory were a bare
   list scattered around, this would be fiddly.
3. **A future extension point.** Real systems eventually need to *trim* or
   *summarise* history to stay under the context-window limit. Because everything
   goes through `Memory`, you can add that later in one method without touching
   the agent. (We haven't implemented trimming — it's noted as a deliberate
   extension point, not an oversight.)

## The message roles

| Role | Who it's from | Notes |
|------|---------------|-------|
| `system` | you (the developer) | the persona/instructions; sent once, at the top |
| `user` | the human, or an upstream agent handing off | the task/input |
| `assistant` | the model | includes its tool-call *requests* |
| `tool` | your code | the *result* of running a tool (OpenAI dialect) |

The ordering matters and providers validate it: an assistant tool-call message
must be followed by the matching tool-result message(s) before the next assistant
turn. The agent (note 05) is what maintains that ordering; `Memory` just stores
whatever it's given, in order.

## A subtle design choice: `to_list()` returns the *live* list

`to_list()` hands back the actual internal list, not a copy, because the agent
appends the provider's raw assistant message straight onto it during the loop.
When you want a safe, immutable snapshot (for inspection or storage), call
`snapshot()` instead. Naming the two operations differently makes the sharp edge
explicit rather than hidden.

## `last_assistant_text()` and content blocks

Pulling out "the final answer" sounds trivial, but Anthropic stores assistant
content as a **list of blocks**, not a string. So `last_assistant_text()` handles
both: a plain string is returned as-is; a block list has its `text` blocks joined.
This is the kind of provider difference we absorb low in the stack so callers
never think about it.

Next: [04 — Tracing & cost tracking](./04-tracing.md).
