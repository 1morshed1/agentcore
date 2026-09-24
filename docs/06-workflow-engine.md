# 06 — The Workflow Engine (DAGs)

Pairs with: `agentcore/workflow.py`

## From one agent to many

One agent is a loop. A *workflow* is a graph of agents, each specialised, wired
together by dependencies:

```
researcher ──► analyst ──► writer
```

The arrows are dependencies: `analyst` needs `researcher`'s output; `writer` needs
`analyst`'s. This shape is a **DAG** — a Directed Acyclic Graph:

- **Directed** — dependencies point one way.
- **Acyclic** — nothing can (transitively) depend on itself. If it could, you'd
  have a chicken-and-egg deadlock, so we detect and reject cycles.

## Registering the graph

```python
wf = Workflow("market_research")
wf.add_agent(researcher, task="Research competitor pricing")
wf.add_agent(analyst,    task="Find pricing trends", depends_on=["researcher"])
wf.add_agent(writer,     task="Write a report",      depends_on=["analyst"])
```

Each node is keyed by its agent's `name`; dependencies refer to those names.
`add_agent` returns `self`, so you can chain calls fluently if you prefer.

## The key algorithm: topological sort into *levels*

To run the graph correctly we must run every node **after** its dependencies. The
classic way to order a DAG is a **topological sort**. We use **Kahn's algorithm**,
but with a twist: instead of a flat order, we group nodes into **levels**.

```
level 0: [researcher]          # nothing depends on -> can start immediately
level 1: [analyst]             # ready once researcher is done
level 2: [writer]              # ready once analyst is done
```

A more branched graph shows why levels matter:

```
        ┌─► news_researcher ─┐
topic ─►┤                    ├─► analyst ─► writer
        └─► filings_reader  ─┘
level 0: [filings_reader, news_researcher]   <- these two run IN PARALLEL
level 1: [analyst]
level 2: [writer]
```

Everything in one level is independent **by construction**, so we can run a whole
level concurrently.

### How Kahn's algorithm works (in `_execution_levels`)

1. Compute each node's **in-degree** = number of dependencies it's waiting on.
2. Every node with in-degree `0` forms the **current level** (nothing left to wait
   for).
3. "Remove" that level: for each node in it, decrement the in-degree of the nodes
   that depended on it. Any that hit `0` form the **next level**.
4. Repeat until no nodes are ready.
5. If nodes remain unplaced, they form a **cycle** → not a DAG → raise
   `WorkflowError`.

We `sorted()` the ready sets so execution order is deterministic (important for
reproducible traces and tests).

## Parallelism: threads for I/O-bound work

Within a level, independent nodes run in a `ThreadPoolExecutor`. "But Python has
the GIL!" — right, and it doesn't matter here, because an agent's time is spent
**waiting on an HTTP response**, not burning CPU. The GIL is released during I/O,
so threads give real, wall-clock speedups for I/O-bound work like API calls. (A
single-node level skips the pool entirely — no thread overhead when there's
nothing to parallelise.)

### A concurrency subtlety: one tracer per node

Parallel threads must not scribble on the same tracer's event list at once. So
each node runs with its **own** tracer, and we `merge()` its events into the
shared workflow tracer after the node finishes. Each thread only ever touches its
own object → no data races, and the merged trace still tells the whole story.

## Shared context = handoff

Before a node runs, `_gather_context` collects its dependencies' outputs into a
dict, which the agent folds into its prompt (note 05). That's the entire "handoff"
mechanism: **the previous agent's answer becomes part of the next agent's task.**
No shared mutable blackboard, no magic — just passing results forward.

## `WorkflowResult`

You get back `outputs` (name → text), `final_output` (the last topological node's
output — the natural "answer" for a linear pipeline), `results` (name → full
`AgentResult` for deep inspection), and one merged `tracer` for the whole run.

## Checkpointing hooks

`run(checkpoint=...)` saves completed node outputs after each node and skips
already-done nodes when resuming. That's the subject of the next note.

Next: [07 — Persistence: checkpoint & resume](./07-persistence.md).
