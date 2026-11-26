# 04 — Tracing & Cost Tracking

Pairs with: `agentcore/tracing.py`

## Why observability is not optional

The moment an agent does more than one step, "it didn't work" becomes impossible
to debug without a record of *what it actually did*. Did the model call the wrong
tool? Did a tool error out? Did it loop 9 times burning tokens? A **trace**
answers all of this, and an HTML timeline of an agent's reasoning makes a run
easy to inspect and share.

## The data model: a flat, time-ordered event list

A `Trace` is a list of `TraceEvent`s. Each event has a `type`:

| Type | Recorded when | Key fields |
|------|---------------|-----------|
| `agent_start` | an agent begins a task | `agent`, `message` (the task) |
| `llm_call` | one model completion returns | `model`, `usage`, `cost_usd`, `duration_s` |
| `tool_call` | a tool finishes running | `tool_name`, `arguments`, `result`, `duration_s` |
| `handoff` | one agent passes to another | `message` ("a -> b") |
| `agent_end` | an agent produces its answer | `agent`, `message` (the output) |
| `error` | something failed | `message` |

### Design decision: flat list, not a nested tree

You could model a trace as a tree (agent → steps → tool calls). We chose a **flat
list** because:

- it's trivial to append to from anywhere, with no parent-tracking bookkeeping;
- it serialises to JSON directly;
- each event carries an `agent` label, so even a multi-agent workflow trace stays
  readable when flattened;
- rolling several agents' traces into one workflow trace is just list
  concatenation (`merge()`).

Simplicity wins here. A tree buys you little that the `agent` label doesn't
already give you.

## Timing: measure, don't guess

The agent wraps each LLM/tool call and records the **wall-clock duration** plus,
for LLM calls, the **token usage** (which the provider returns) and the
**estimated cost** (usage × price, from note 02). So every row on the timeline
tells you *how long* and *how much*.

## The three exports

1. **`to_dict()` / `to_json()`** — the full trace plus a computed `summary`
   block (total cost, tokens, call counts, duration). Log it, diff it, analyse it.
2. **`summary()`** — a one-line summary:
   `steps: 3 LLM / 2 tool | tokens: 1240 | cost: $0.0031 | duration: 4.20s`.
3. **`to_html()`** — a **self-contained** HTML timeline (all CSS inline, no external
   files). Colour-coded rows, a summary header, and truncated tool results. Open
   it in any browser; commit it as an artifact.

## Aggregate metrics

`total_cost()`, `total_usage()`, `total_duration()`, and `count(type)` compute
the roll-ups by scanning the event list. Cheap, and always consistent with the
raw events because they're derived from them rather than tracked separately (no
chance of the summary drifting out of sync with reality).

## A note on truncation

Tool results can be huge (a whole web page). `_truncate()` clips long strings in
the trace to keep JSON/HTML readable, appending `… (+N chars)` so you know
something was cut. The *agent* still gets the full result — only the trace record
is trimmed.

Next: [05 — The ReAct loop](./05-react-loop.md).
