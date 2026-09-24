# agentcore — Learning Notes

These notes are written **for learning**. The code in `agentcore/` is heavily
commented, but code comments explain *what a line does*. These documents explain
the bigger picture: *why the design is shaped the way it is*, what the
alternatives were, and what a framework like LangChain/LangGraph is doing under
the same hood.

Read them roughly in order — each builds on the last.

| # | Note | What you'll learn |
|---|------|-------------------|
| 00 | [Overview & mental model](./00-overview.md) | What an "agent" really is, and the shape of the whole library |
| 01 | [Tools & function calling](./01-tools-and-function-calling.md) | How an LLM calls your Python code, and how we auto-build JSON Schema |
| 02 | [Talking to the LLM by hand](./02-llm-clients.md) | Raw HTTP calls to OpenAI & Anthropic, retries, and why we abstract the provider |
| 03 | [Memory & message shapes](./03-memory.md) | How conversation state is stored and why message roles matter |
| 04 | [Tracing & cost tracking](./04-tracing.md) | Observability: timing every step, counting tokens, computing cost |
| 05 | [The ReAct loop](./05-react-loop.md) | The core algorithm every agent framework implements |
| 06 | [The workflow engine (DAGs)](./06-workflow-engine.md) | Orchestrating many agents with dependencies and parallelism |
| 07 | [Persistence: checkpoint & resume](./07-persistence.md) | Surviving crashes by saving state to disk |

> **Tip:** open a note side-by-side with the module it describes. Note 01 pairs
> with `agentcore/tools.py`, note 05 with `agentcore/agent.py`, and so on.
