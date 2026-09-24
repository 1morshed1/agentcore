"""Tests for the DAG workflow engine: ordering, parallelism, context, cycles."""

from __future__ import annotations

import threading
import time

import pytest

from agentcore import Agent, LLMResponse, MockLLM, Workflow
from agentcore.errors import WorkflowError


def _answering_agent(name: str, answer: str) -> Agent:
    """An agent that ignores tools and just returns a fixed answer."""
    return Agent(name, f"you are {name}", MockLLM(responses=[LLMResponse(content=answer)]))


def test_linear_pipeline_runs_in_order_with_handoff():
    researcher = _answering_agent("researcher", "price is 100")
    writer = _answering_agent("writer", "REPORT: price is 100")

    wf = Workflow("demo")
    wf.add_agent(researcher, task="research")
    wf.add_agent(writer, task="write", depends_on=["researcher"])
    result = wf.run()

    assert result.outputs["researcher"] == "price is 100"
    assert result.final_output == "REPORT: price is 100"
    # Writer must have received the researcher's output as context.
    writer_prompt = writer.llm.calls[0][-1]["content"]
    assert "price is 100" in writer_prompt


def test_execution_levels_grouping():
    wf = Workflow("diamond")
    wf.add_agent(_answering_agent("a", "a"), task="t")
    wf.add_agent(_answering_agent("b", "b"), task="t", depends_on=["a"])
    wf.add_agent(_answering_agent("c", "c"), task="t", depends_on=["a"])
    wf.add_agent(_answering_agent("d", "d"), task="t", depends_on=["b", "c"])

    levels = wf._execution_levels()
    assert levels[0] == ["a"]
    assert levels[1] == ["b", "c"]  # both depend only on a -> same level
    assert levels[2] == ["d"]


def test_independent_nodes_run_in_parallel():
    # Each agent sleeps 0.2s inside its (mock) LLM call. Run serially that's
    # 0.4s; in parallel it should be ~0.2s. We assert well under the serial time.
    def slow_response(_messages):
        time.sleep(0.2)
        return LLMResponse(content="done")

    a = Agent("a", "role", MockLLM(responses=slow_response))
    b = Agent("b", "role", MockLLM(responses=slow_response))
    wf = Workflow("parallel")
    wf.add_agent(a, task="t")
    wf.add_agent(b, task="t")  # no dependency -> same level -> parallel

    start = time.time()
    wf.run(max_workers=2)
    elapsed = time.time() - start
    assert elapsed < 0.35, f"expected parallel (~0.2s), took {elapsed:.2f}s"


def test_cycle_is_detected():
    wf = Workflow("cyclic")
    a = _answering_agent("a", "a")
    b = _answering_agent("b", "b")
    wf.add_agent(a, task="t", depends_on=["b"])
    wf.add_agent(b, task="t", depends_on=["a"])
    with pytest.raises(WorkflowError, match="cycle"):
        wf.run()


def test_dangling_dependency_is_rejected():
    wf = Workflow("dangling")
    wf.add_agent(_answering_agent("a", "a"), task="t", depends_on=["ghost"])
    with pytest.raises(WorkflowError, match="ghost"):
        wf.run()


def test_duplicate_node_name_rejected():
    wf = Workflow("dupes")
    wf.add_agent(_answering_agent("a", "1"), task="t")
    with pytest.raises(WorkflowError):
        wf.add_agent(_answering_agent("a", "2"), task="t")


def test_on_node_complete_callback_fires():
    seen: list[str] = []
    lock = threading.Lock()

    def cb(name, result):
        with lock:
            seen.append(name)

    wf = Workflow("cb")
    wf.add_agent(_answering_agent("a", "a"), task="t")
    wf.add_agent(_answering_agent("b", "b"), task="t", depends_on=["a"])
    wf.run(on_node_complete=cb)
    assert set(seen) == {"a", "b"}


def test_merged_trace_covers_all_agents():
    wf = Workflow("trace")
    wf.add_agent(_answering_agent("a", "a"), task="t")
    wf.add_agent(_answering_agent("b", "b"), task="t", depends_on=["a"])
    result = wf.run()
    # Two agents, one handoff edge -> the merged trace has events from both.
    agents_in_trace = {e.agent for e in result.tracer.events}
    assert {"a", "b"} <= agents_in_trace
