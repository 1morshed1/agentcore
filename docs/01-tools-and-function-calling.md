# 01 — Tools & Function Calling

Pairs with: `agentcore/tools.py`

## The problem

An LLM outputs text. That's all it does. So how does "the model called my
`web_search` function" actually work? The model *cannot* run your code. Instead,
there's a protocol both sides agree on.

## The protocol, step by step

1. **You describe your functions to the model** as JSON Schema. For each tool the
   model learns: its `name`, a `description`, and the typed shape of its
   arguments. Example (OpenAI dialect):

   ```json
   {
     "type": "function",
     "function": {
       "name": "web_search",
       "description": "Search the web and return the top results",
       "parameters": {
         "type": "object",
         "properties": { "query": { "type": "string" } },
         "required": ["query"]
       }
     }
   }
   ```

2. **The model responds with a tool call** instead of prose. It doesn't run
   anything — it emits a structured request: *"call `web_search` with
   `{"query": "crm pricing"}`"* plus a unique `id`.

3. **Your code runs the real function** with those arguments.

4. **You send the return value back** as a `tool` message, tagged with the same
   `id`, so the model can match the result to its request.

5. The model reads the result and either answers or calls another tool.

That's function calling. No magic — just a well-defined request/response dance.

## What our `@tool` decorator automates

Hand-writing that JSON Schema for every function would be miserable and
error-prone. So `@tool` **generates it from the function itself**:

```python
@tool(description="Search the web and return the top results")
def web_search(query: str, limit: int = 5) -> str: ...
```

From this we derive:

- `name` → `"web_search"` (the function name)
- `description` → what you passed (or the docstring's first paragraph)
- `parameters` → built by **introspection**:
  - `query: str` → `{"type": "string"}`, and it's **required** (no default)
  - `limit: int = 5` → `{"type": "integer"}`, **not required** (has a default)

### How the introspection works (the three tools)

Python gives us everything we need in the standard library:

| Tool | What it gives us |
|------|------------------|
| `inspect.signature(func)` | the parameters, in order, with their defaults |
| `typing.get_type_hints(func)` | the resolved type annotations |
| `inspect.getdoc(func)` | the docstring, for descriptions |

The rule for "required": a parameter is required **iff it has no default value**
(`param.default is inspect.Parameter.empty`). This mirrors exactly how Python
itself decides whether an argument is mandatory — a nice, intuitive mapping.

### Type mapping

JSON Schema has only a few primitive types, so we map Python → JSON:

| Python | JSON Schema |
|--------|-------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list[X]` | `array` with `items` |
| `dict` | `object` |
| `Optional[X]` / `X \| None` | the schema of `X` (optionality is handled by `required`) |
| anything else | `string` (safe fallback) |

The `Optional` case is a subtle but important decision: **optionality is not a
type**. `Optional[str]` doesn't mean "a nullable string type" in JSON Schema — it
means "this field may be absent." So we unwrap `Optional[str]` to just
`{"type": "string"}` and express optionality by *leaving the field out of the
`required` list*. Getting this wrong is a common bug in hand-rolled tool layers.

### Docstring → parameter descriptions

Good parameter descriptions dramatically improve how reliably a model calls a
tool. Rather than force you to pass them separately, we parse a **Google-style**
docstring's `Args:` block:

```python
@tool()
def web_search(query: str, limit: int = 5) -> str:
    """Search the web and return the top results.

    Args:
        query: The search terms to look up.
        limit: Maximum number of results to return.
    """
```

`query`'s schema now carries `"description": "The search terms to look up."` —
generated for free.

## Two dialects: OpenAI vs Anthropic

The two vendors format tool schemas slightly differently:

- **OpenAI** wraps each tool in `{"type": "function", "function": {...}}` and puts
  arguments under `parameters`.
- **Anthropic** puts the tool at top level and calls the argument schema
  `input_schema`.

So `Tool` has two serialisers: `to_openai_schema()` and `to_anthropic_schema()`.
The `Tool` object is the single source of truth; the dialect is just a
presentation of it. The agent asks its LLM client which dialect to use.

## Executing tools safely

When the model asks to run a tool, three things can go wrong, and we handle each:

1. **Bad arguments** (missing required field, unknown field). We validate cheaply
   in `validate_arguments()` and raise a `ToolExecutionError` with a clear
   message — which the agent feeds *back to the model* so it can correct itself.
2. **The tool raises.** We wrap any exception from the tool body in
   `ToolExecutionError` (keeping the original on `.original`). Again the agent
   reports the message to the model rather than crashing the whole run.
3. **Async tools.** `Tool.execute()` detects a coroutine function and runs it to
   completion, so a synchronous agent can still use async tools.

### Design decision: why *not* pull in a JSON Schema validator?

We validate arguments by hand (presence of required keys, no extras) rather than
adding `jsonschema` as a dependency. Two reasons: (1) the dependency rule — we
stay at one runtime dependency; (2) the goal isn't spec-perfect validation, it's a
*helpful, correctable error message* the model can act on. A full validator is
overkill for that.

Next: [02 — Talking to the LLM by hand](./02-llm-clients.md).
