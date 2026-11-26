"""The Workflow engine — orchestrating many agents as a DAG.

A single agent is useful. Real work usually needs *several*, each specialised, in
a pipeline: a researcher gathers facts, an analyst crunches them, a writer turns
the analysis into prose. Those dependencies form a **DAG** — a Directed Acyclic
Graph. "Directed" because dependencies point one way; "Acyclic" because a task
cannot (transitively) depend on itself.

This module runs such a graph:

* You register agents as **nodes**, each with the tasks it should do and the nodes
  it ``depends_on``.
* We compute a valid execution order with a **topological sort**.
* Independent nodes (same "level" of the graph) run **in parallel**.
* Each node receives the outputs of its dependencies as **shared context**.
* The whole run is captured in one merged trace, and can be **checkpointed** so a
  crash mid-workflow can be resumed.

This is the 80/20 of what LangGraph's graph executor gives you, in a file you can
read top to bottom.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, AgentResult
from .errors import WorkflowError
from .persistence import Checkpoint
from .tracing import ExecutionTracer


@dataclass
class Node:
    """One step in the workflow: an agent plus what it should do and after whom.

    Attributes:
        agent: The :class:`~agentcore.agent.Agent` that runs this step.
        task: The instruction for this step.
        depends_on: Names of nodes that must finish before this one starts. Their
            outputs are passed into this node as context.
    """

    agent: Agent
    task: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """The outcome of running a whole workflow.

    Attributes:
        outputs: Map of node name -> that agent's final text output.
        final_output: The output of the last node to complete in topological
            order (a convenient "the answer" for linear pipelines).
        results: Map of node name -> full :class:`AgentResult` (for deep
            inspection).
        tracer: One merged trace spanning every agent in the workflow.
    """

    outputs: dict[str, str]
    final_output: str
    results: dict[str, AgentResult]
    tracer: ExecutionTracer

    def summary(self) -> str:
        return self.tracer.summary()


class Workflow:
    """A DAG of agents with shared context, parallel execution, and checkpointing.

    Example::

        wf = Workflow("market_research")
        wf.add_agent(researcher, task="Research competitor pricing")
        wf.add_agent(analyst, task="Find pricing trends", depends_on=["researcher"])
        wf.add_agent(writer, task="Write a 1-page report", depends_on=["analyst"])
        result = wf.run()
        print(result.final_output)

    Nodes are keyed by their agent's ``name``. Dependencies refer to those names.
    """

    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, Node] = {}

    # -- graph construction --------------------------------------------------
    def add_agent(self, agent: Agent, task: str, depends_on: list[str] | None = None) -> Workflow:
        """Register an agent as a node. Returns ``self`` for fluent chaining.

        Raises:
            WorkflowError: if a node with this agent's name already exists.
        """
        if agent.name in self.nodes:
            raise WorkflowError(f"Duplicate node name {agent.name!r} in workflow {self.name!r}.")
        self.nodes[agent.name] = Node(agent=agent, task=task, depends_on=list(depends_on or []))
        return self

    # -- graph analysis ------------------------------------------------------
    def _validate(self) -> None:
        """Check that every dependency exists (dangling edges are a common bug)."""
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise WorkflowError(
                        f"Node {node.agent.name!r} depends on {dep!r}, which is not in the workflow."
                    )

    def _execution_levels(self) -> list[list[str]]:
        """Group nodes into levels via a topological (Kahn's) sort.

        Each *level* is a set of nodes with no remaining unsatisfied dependencies —
        i.e. nodes that can all run **at the same time**. Returning levels (rather
        than a flat order) is what enables parallelism: everything in one level is
        independent by construction.

        Algorithm (Kahn's, level-by-level):
        1. Compute each node's in-degree (number of dependencies).
        2. The nodes with in-degree 0 form the next level.
        3. "Remove" them; decrement the in-degree of everything that depended on
           them. Repeat.
        4. If we finish with nodes still unplaced, the graph has a **cycle** — not
           a DAG — so we raise.
        """
        # in-degree = how many deps each node is still waiting on
        indegree: dict[str, int] = {name: len(node.depends_on) for name, node in self.nodes.items()}
        # dependents = reverse edges: who is waiting on me
        dependents: dict[str, list[str]] = {name: [] for name in self.nodes}
        for name, node in self.nodes.items():
            for dep in node.depends_on:
                dependents[dep].append(name)

        levels: list[list[str]] = []
        placed = 0
        ready = sorted(name for name, deg in indegree.items() if deg == 0)

        while ready:
            levels.append(ready)
            placed += len(ready)
            next_ready: list[str] = []
            for name in ready:
                for child in dependents[name]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        next_ready.append(child)
            ready = sorted(next_ready)

        if placed != len(self.nodes):
            unplaced = [n for n, d in indegree.items() if d > 0]
            raise WorkflowError(
                f"Workflow {self.name!r} has a cycle involving: {unplaced}. A DAG must be acyclic."
            )
        return levels

    def _gather_context(self, node: Node, outputs: dict[str, str]) -> dict[str, Any]:
        """Collect the outputs of a node's dependencies into its context dict."""
        return {dep: outputs[dep] for dep in node.depends_on if dep in outputs}

    # -- execution -----------------------------------------------------------
    def run(
        self,
        *,
        max_workers: int = 4,
        checkpoint: Checkpoint | None = None,
        on_node_complete: Callable[[str, AgentResult], None] | None = None,
    ) -> WorkflowResult:
        """Execute the whole workflow and return a :class:`WorkflowResult`.

        Args:
            max_workers: Thread-pool size for running independent nodes in
                parallel. Because each agent's work is an HTTP call (I/O-bound),
                threads give real speedups despite the GIL.
            checkpoint: Optional :class:`~agentcore.persistence.Checkpoint`. If
                given, completed node outputs are saved after each node, and any
                already-completed nodes are skipped on a resumed run.
            on_node_complete: Optional callback fired as each node finishes —
                useful for progress bars / streaming UIs.
        """
        self._validate()
        levels = self._execution_levels()

        # A single tracer for the whole workflow; each agent writes into it.
        tracer = ExecutionTracer(self.name)

        # Restore any previously-completed work when resuming from a checkpoint.
        outputs: dict[str, str] = {}
        results: dict[str, AgentResult] = {}
        if checkpoint is not None:
            outputs = checkpoint.load_outputs()

        last_completed = ""

        for level in levels:
            # Skip nodes already done in a previous (checkpointed) run.
            pending = [name for name in level if name not in outputs]

            # Log handoffs: every dependency edge into this level is a handoff.
            for name in pending:
                for dep in self.nodes[name].depends_on:
                    tracer.log_handoff(dep, name)

            if len(pending) == 1:
                # Single node: run inline (no thread overhead).
                name = pending[0]
                result = self._run_node(name, outputs, tracer)
                outputs[name] = result.output
                results[name] = result
                last_completed = name
                if checkpoint is not None:
                    checkpoint.save_outputs(outputs)
                if on_node_complete:
                    on_node_complete(name, result)
            elif pending:
                # Multiple independent nodes: run them concurrently.
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(self._run_node, name, outputs, tracer): name for name in pending
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        result = future.result()  # re-raises any exception from the node
                        outputs[name] = result.output
                        results[name] = result
                        last_completed = name
                        if checkpoint is not None:
                            checkpoint.save_outputs(outputs)
                        if on_node_complete:
                            on_node_complete(name, result)

        # For linear pipelines the "final" output is the last topological node.
        # (levels[-1][-1] is deterministic thanks to the sorted() ordering above.)
        final_name = levels[-1][-1] if levels else ""
        final_output = outputs.get(final_name, outputs.get(last_completed, ""))

        return WorkflowResult(
            outputs=outputs,
            final_output=final_output,
            results=results,
            tracer=tracer,
        )

    def _run_node(self, name: str, outputs: dict[str, str], tracer: ExecutionTracer) -> AgentResult:
        """Run a single node's agent, feeding it its dependencies' outputs.

        Note the per-node tracer trick: each agent runs with its *own* tracer so
        its timing is measured independently, then we merge those events into the
        shared workflow tracer. This keeps parallel writes from interleaving on one
        tracer object (each thread touches only its own).
        """
        node = self.nodes[name]
        context = self._gather_context(node, outputs)
        node_tracer = ExecutionTracer(name)
        result = node.agent.run(node.task, context=context, tracer=node_tracer)
        tracer.merge(node_tracer)
        return result
