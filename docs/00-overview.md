# 00 — Overview & Mental Model

## What is an "agent," really?

Strip away the marketing and an LLM agent is a **loop around a chat completion
call**. That's it. The loop is:

```
1. Send the conversation so far to the model.
2. The model replies. Either:
     (a) it produces a final answer  -> we're done, or
     (b) it asks to call a tool      -> we run the tool, append the result to
                                          the conversation, and go back to step 1.
```

This loop has a name: **ReAct** ("Reason + Act"), from a 2022 paper. The model
*reasons* about what to do, *acts* by calling a tool, observes the result, and
repeats. Frameworks like LangGraph's `create_react_agent` wrap this loop in a lot
of abstraction. We implement it directly in `agentcore/agent.py` — and it's about
40 lines.

Everything else in this library exists to make that loop **usable, observable,
and composable**:

- **Tools** (`tools.py`) — so the model can *do* things, not just talk.
- **LLM clients** (`llm.py`) — so we can actually send that HTTP request, to
  either OpenAI or Anthropic, with retries.
- **Memory** (`memory.py`) — so the conversation is stored in one place.
- **Tracing** (`tracing.py`) — so we can *see* what happened: every step, its
  latency, its token cost.
- **Workflow** (`workflow.py`) — so we can chain *many* agents together with
  dependencies (a DAG), not just run one.
- **Persistence** (`persistence.py`) — so a long workflow can survive a crash and
  resume.

## The dependency rule

The single most important design constraint: **no LLM framework dependencies.**
The only third-party runtime package is `httpx` (an HTTP client). We are proving
that the "magic" of agent frameworks is ordinary code you can write yourself.

```
agentcore
  └── httpx        <- the ONLY runtime dependency
```

## How the pieces fit together

```
        ┌─────────────────────────────────────────────┐
        │                 Workflow                     │   many agents, DAG-ordered
        │   researcher ──► analyst ──► writer          │
        └───────────────────┬─────────────────────────┘
                            │ each node is one…
                            ▼
        ┌─────────────────────────────────────────────┐
        │                   Agent                      │   the ReAct loop
        │   ┌─────────┐   ┌──────────┐   ┌──────────┐  │
        │   │ Memory  │   │  Tools   │   │  Tracer  │  │
        │   └─────────┘   └──────────┘   └──────────┘  │
        │            uses an LLMClient ▼                │
        └───────────────────┬─────────────────────────┘
                            ▼
        ┌─────────────────────────────────────────────┐
        │        LLMClient (OpenAI / Anthropic)        │   raw httpx calls
        └─────────────────────────────────────────────┘
```

## Design decisions taken here (and why)

1. **Provider abstraction from day one.** The plan mentioned OpenAI, but real
   portfolios impress more when they show you can support *both* OpenAI and
   Anthropic behind one interface. So `LLMClient` is an abstract base with
   `OpenAIClient` and `AnthropicClient` implementations. See note 02.

2. **A `MockLLM` client.** Tests must run with no API key and no network. So we
   ship a scriptable fake client. This is also how you'd unit-test *any*
   LLM app — never hit a paid API in your test suite.

3. **Dataclasses over dicts for results.** `AgentResult`, `Trace`, `Step`, etc.
   are typed dataclasses. They're self-documenting and give you autocomplete,
   which matters when the goal is a *clean public API*.

4. **Sync-first, async-capable.** The ReAct loop is easiest to read
   synchronously, so that's the default. But tools can be `async`, and the
   workflow engine runs independent agents *in parallel* using threads. See
   note 06.

Next: [01 — Tools & function calling](./01-tools-and-function-calling.md).
