# 05 — The ReAct Loop

Pairs with: `agentcore/agent.py`

This is the most important note. Everything else was setup; this is the algorithm
that *is* an agent.

## ReAct in one breath

**ReAct = Reason + Act.** The model alternates between thinking about what to do
and acting (calling a tool), using each observation to inform the next thought.
The loop:

```
build conversation = [system prompt, task]
for step in range(max_steps):
    response = LLM(conversation, tools)
    conversation.append(response)         # keep the model's own message
    if response has tool_calls:
        for call in response.tool_calls:
            result = run_the_tool(call)
            conversation.append(tool_result(call, result))
        continue                          # loop: let the model see the results
    else:
        return response.content           # no tools => final answer
raise MaxStepsExceeded
```

That's the entire idea behind `create_react_agent`. Read `Agent.run()` — it's
this loop with tracing and error handling layered on.

## The single most important line

```python
memory.add(response.raw_message)  # BEFORE handling the tool calls
```

Order matters and providers enforce it: the assistant's tool-call message must
appear in the history **before** the tool-result messages that answer it. If you
run the tools first and forget to append the assistant message, the next API call
fails with a validation error. We append the provider's *own* message
(`raw_message`) to guarantee the shape is exactly what it expects.

## "No tool call" is the exit condition

There's no special "I'm done" signal from the model. The loop ends simply when a
response comes back with **no tool calls** — that plain text *is* the answer. This
is elegant: the same mechanism that lets the model act also, by its absence,
signals completion.

## Why `max_steps` exists

A ReAct loop is unbounded by construction — the model decides when to stop. A
confused model can loop forever (call tool, ignore result, call it again...),
burning money. `max_steps` is the circuit breaker. Hitting it raises
`MaxStepsExceeded` — an *error*, because a well-behaved run should always reach a
final answer first.

## Turning every failure into an observation

Look at `_execute_tool`. Three things can go wrong, and **none of them crash the
run**:

| Failure | What the model gets back |
|---------|--------------------------|
| Unknown tool name | `"Error: unknown tool 'foo'. Available: [...]"` |
| Bad/missing arguments | `"Tool error: Missing required argument(s): query"` |
| Tool body raises | `"Tool error: <the exception message>"` |

Each becomes a **string observation** appended to the conversation. The model
reads it and can *correct itself* on the next step — retry with the right
arguments, pick a different tool, or give up gracefully. This "errors are just
observations" pattern is a huge part of why ReAct agents are robust. If we let
tool exceptions propagate, a single bad model guess would kill the whole task.

## Context injection (for workflows)

`_build_task_prompt(task, context)` folds a `context` dict into the user message:

```
Research competitor pricing.

Context from previous steps:
- researcher: <the researcher agent's findings>
```

This is how an agent later in a workflow *sees the output of agents before it*.
Handoff is nothing more exotic than "put the previous result in the next prompt."

## Sync and async twins

`run()` is synchronous — easiest to read and reason about, and the default.
`arun()` is the identical loop, except:

- tools are awaited (`tool.aexecute`), and
- the blocking HTTP call is pushed to a worker thread with
  `asyncio.to_thread`, so it doesn't stall the event loop.

`arun()` is what lets the workflow engine (note 06) run independent agents
*concurrently*. Keeping the two in lockstep is a small maintenance cost for a big
usability win: simple code stays simple, concurrent code is available when you
need it.

## `AgentResult` — a clean return type

Rather than return a bare string, `run()` returns an `AgentResult` dataclass:
`output`, `steps`, `tracer` (full observability), `memory` (final state), plus
`cost_usd`/`total_tokens` convenience properties. Typed, self-documenting, and it
gives you everything you need to debug or display a run.

Next: [06 — The workflow engine (DAGs)](./06-workflow-engine.md).
