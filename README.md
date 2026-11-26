# agentcore

[![Tests](https://github.com/1morshed1/agentcore/actions/workflows/tests.yml/badge.svg)](https://github.com/1morshed1/agentcore/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/runtime%20deps-httpx%20only-success)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A multi-agent orchestrator built on raw LLM APIs — no LangChain, no LangGraph, no CrewAI.**

`agentcore` is a small, readable Python library that gives you agents, tools, and
DAG-based multi-agent workflows on top of *raw* OpenAI/Anthropic HTTP calls. Its
only runtime dependency is [`httpx`](https://www.python-httpx.org/). It covers the
core of what LangGraph is used for — tool-calling agents, multi-agent DAGs, state
handoffs, tracing and checkpointing — in a codebase small enough to read in an
afternoon.

```python
from agentcore import Agent, tool, OpenAIClient


@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b


agent = Agent(
    name="calculator",
    role="You are a precise calculator. Use tools for arithmetic.",
    llm=OpenAIClient("gpt-4o-mini"),
    tools=[add],
)

result = agent.run("What is 2 + 3?")
print(result.output)  # -> "5"
print(result.summary())  # steps: 2 LLM / 1 tool | tokens: 210 | cost: $0.0001 | duration: 1.4s
```

---

## Why I built this

I've worked with LangChain, LangGraph, CrewAI, AutoGen, Google ADK, and MCP. Using
a framework well is not the same as knowing what it does, so I wanted to understand what `create_react_agent`
**actually does** under the hood — how tool calling works at the protocol level,
how conversation state survives across agent handoffs, how a ReAct loop is really
structured, and how to make all of it observable and reliable. So I built it from
scratch with nothing but Python and an HTTP client.

The result taught me more about agents than any framework tutorial. Every design
decision is written up in [`docs/`](./docs/README.md), so the repository doubles
as a guided tour of how agent frameworks work internally.

## What it does

- 🧠 **Agents** — a role, a set of tools, and a ReAct loop (`agent.py`)
- 🔧 **Tools** — `@tool` auto-generates JSON Schema from your type hints & docstrings
- 🔌 **Providers** — one interface over OpenAI *and* Anthropic (+ a `MockLLM` for tests)
- 🕸️ **Workflows** — DAG orchestration with parallel branches and shared context
- 🤝 **Handoffs** — each agent receives its dependencies' outputs as context
- 📊 **Tracing** — per-step latency, token usage, and cost; JSON + HTML timeline export
- 💾 **Persistence** — atomic checkpoints so a crashed workflow resumes where it stopped
- ⚡ **Async** — an async agent loop and thread-parallel workflow execution

## Install

```bash
git clone https://github.com/1morshed1/agentcore
cd agentcore
pip install -e .          # installs the package + httpx
```

## Run it in 2 minutes (no API key needed)

The offline demo uses a scripted mock model, so it runs instantly with no key and
no network — perfect for seeing the whole pipeline work:

```bash
python examples/01_offline_demo.py
```

```
Running workflow 'market_research'...

  ✓ researcher finished in 1 step(s)
  ✓ analyst finished in 2 step(s)
  ✓ writer finished in 1 step(s)

--- FINAL REPORT ---
PRICING BRIEF
Competitor CRM pricing increased from $80 to $92 per seat (+15.0%).
Recommendation: revisit our own pricing to stay competitive.

--- TRACE SUMMARY ---
steps: 4 LLM / 1 tool | tokens: 0 | cost: $0.0000 | duration: 0.00s

Wrote HTML timeline to market_research.trace.html (open it in a browser).
```

Then, with an API key, run a real workflow:

```bash
export OPENAI_API_KEY=sk-...
python examples/03_market_research.py
```

See [`examples/`](./examples/README.md) for all four examples.

## The same agent: LangGraph vs agentcore

The same "call a tool, then answer" agent, side by side:

<table>
<tr><th>LangGraph</th><th>agentcore</th></tr>
<tr valign="top"><td>

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


@tool
def add(a: int, b: int) -> int:
    "Add two numbers"
    return a + b


model = ChatOpenAI(model="gpt-4o-mini")
graph = create_react_agent(model, tools=[add])

result = graph.invoke({"messages": [("user", "What is 2 + 3?")]})
print(result["messages"][-1].content)
```

Depends on `langgraph`, `langchain-core`,
`langchain-openai`, and their transitive tree.
The ReAct loop, tool schema generation, and
state handling are hidden inside the framework.

</td><td>

```python
from agentcore import Agent, tool, OpenAIClient


@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b


agent = Agent(
    name="calc",
    role="Use tools for arithmetic.",
    llm=OpenAIClient("gpt-4o-mini"),
    tools=[add],
)

result = agent.run("What is 2 + 3?")
print(result.output)
```

Depends on `httpx`. That's it.
The ReAct loop is ~40 readable lines in
`agent.py`; the schema generation is in
`tools.py`. Nothing is hidden.

</td></tr></table>

The APIs are comparably concise — but with `agentcore` you can open the source and
see *exactly* what each call does, and `result` is a typed object carrying the full
trace, cost, and token counts.

## Execution trace visualization

Every run produces a trace. Export it as a self-contained HTML timeline:

```python
result = workflow.run()
with open("trace.html", "w") as f:
    f.write(result.tracer.to_html("My run"))
```

The timeline colour-codes each event (agent start/end, LLM calls, tool calls,
handoffs, errors) and shows a header with total cost, tokens, and duration. You can
also get the raw data:

```python
result.tracer.save_json("trace.json")  # full structured trace + summary
print(result.summary())  # one-line summary
```

## How it works (the short version)

An "agent" is a loop around a chat-completion call — the **ReAct** pattern:

```
build the conversation (system prompt + task)
repeat up to max_steps:
    response = LLM(conversation, tools)
    if response asks to call tools:
        run each tool, append its result to the conversation
        continue            # let the model reason about the results
    else:
        return response.content   # no tool calls == final answer
```

A **workflow** wires many agents into a DAG, topologically sorts them into levels,
runs independent nodes in parallel, and passes each agent its dependencies'
outputs as context. Full write-ups, one per subsystem, live in
[`docs/`](./docs/README.md):

1. [Overview & mental model](./docs/00-overview.md)
2. [Tools & function calling](./docs/01-tools-and-function-calling.md)
3. [Talking to the LLM by hand](./docs/02-llm-clients.md)
4. [Memory & message shapes](./docs/03-memory.md)
5. [Tracing & cost tracking](./docs/04-tracing.md)
6. [The ReAct loop](./docs/05-react-loop.md)
7. [The workflow engine (DAGs)](./docs/06-workflow-engine.md)
8. [Persistence: checkpoint & resume](./docs/07-persistence.md)

## Performance & footprint vs LangGraph

The honest, useful comparison for a library like this is **dependency footprint**
and **transparency**, not micro-benchmarks — an agent's wall-clock time is
dominated by the LLM's own response latency, which is identical whichever library
wraps it.

| | agentcore | LangGraph (typical) |
|---|---|---|
| Runtime dependencies | **1** (`httpx`) | dozens (langgraph + langchain-core + integrations + their deps) |
| Core ReAct loop | ~40 lines you can read | hidden across the framework |
| Tool schema generation | ~80 lines in `tools.py` | inside `langchain-core` |
| Per-run cost/token tracking | built in (`result.summary()`) | via callbacks/LangSmith |
| Time to read the whole library | an afternoon | — |

> Both call the same model over the same HTTP, so per-step latency is equivalent.
> `agentcore` adds negligible overhead (a few dict operations and a timer per
> step). What you gain is a tiny install and a codebase you can fully understand.

## Testing

The library is designed to be fully testable **offline** via `MockLLM`, so the
whole suite runs with no API key and no network:

```bash
pip install -e ".[dev]"
pytest
# 60 passed in ~0.3s
```

## Project layout

```
agentcore/
├── agentcore/            # the library
│   ├── tools.py          # @tool decorator + JSON Schema generation
│   ├── llm.py            # raw OpenAI/Anthropic clients + MockLLM + cost table
│   ├── memory.py         # conversation history
│   ├── tracing.py        # trace events, cost, JSON + HTML export
│   ├── agent.py          # the ReAct loop (sync + async)
│   ├── workflow.py       # DAG orchestration with parallelism
│   ├── persistence.py    # atomic checkpoint/resume
│   └── errors.py         # the exception hierarchy
├── docs/                 # design notes — the "why" behind every module
├── examples/             # runnable examples (start with 01_offline_demo.py)
└── tests/                # 63 offline tests
```

## License

[MIT](./LICENSE)
