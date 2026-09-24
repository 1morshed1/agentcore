"""Tests for the execution tracer: aggregation, summary, and exports."""

from __future__ import annotations

import json

from agentcore import EventType, ExecutionTracer, Usage


def _populated_tracer() -> ExecutionTracer:
    t = ExecutionTracer("agent")
    t.log_agent_start("agent", "do the thing")
    t.log_llm_call("agent", "gpt-4o", Usage(prompt_tokens=1_000_000, completion_tokens=0), 1.5)
    t.log_tool_call("agent", "add", {"a": 1, "b": 2}, 3, 0.01)
    t.log_agent_end("agent", "done")
    return t


def test_counts_and_totals():
    t = _populated_tracer()
    assert t.count(EventType.LLM_CALL) == 1
    assert t.count(EventType.TOOL_CALL) == 1
    # 1M input tokens of gpt-4o at $2.50/Mtok = $2.50.
    assert t.total_cost() == 2.5
    assert t.total_usage().prompt_tokens == 1_000_000


def test_summary_is_one_line_and_informative():
    t = _populated_tracer()
    s = t.summary()
    assert "1 LLM" in s and "1 tool" in s and "$" in s


def test_to_json_has_summary_and_events():
    t = _populated_tracer()
    data = json.loads(t.to_json())
    assert data["summary"]["llm_calls"] == 1
    assert data["summary"]["tool_calls"] == 1
    assert len(data["events"]) == 4
    # Enum serialised to its string value.
    assert data["events"][0]["type"] == "agent_start"


def test_html_export_is_self_contained():
    t = _populated_tracer()
    html = t.to_html("my run")
    assert "<html" in html and "</html>" in html
    assert "my run" in html
    # No external stylesheet/script links -> self-contained.
    assert "http://" not in html.replace("http-equiv", "")
    assert "add" in html  # the tool call shows up on the timeline


def test_merge_combines_events():
    a = ExecutionTracer("a")
    a.log_agent_start("a", "t")
    b = ExecutionTracer("b")
    b.log_agent_start("b", "t")
    a.merge(b)
    assert len(a.events) == 2


def test_truncation_of_long_tool_results():
    t = ExecutionTracer("agent")
    t.log_tool_call("agent", "big", {}, "x" * 1000, 0.0)
    result_text = t.events[0].result
    assert "+" in result_text and "chars" in result_text  # was truncated
