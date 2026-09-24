"""Tests for the Agent ReAct loop.

Every test drives the agent with a `MockLLM` scripted to produce a specific
sequence of tool calls / final answers, so the loop's behaviour is fully
deterministic and offline.
"""

from __future__ import annotations

import pytest

from agentcore import Agent, LLMResponse, MockLLM, ToolCall, tool
from agentcore.errors import MaxStepsExceeded


@tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b


@tool(description="Always fails")
def broken() -> str:
    raise RuntimeError("nope")


def test_direct_answer_no_tools():
    llm = MockLLM(responses=[LLMResponse(content="42")])
    agent = Agent("a", "answer questions", llm)
    result = agent.run("meaning of life?")
    assert result.output == "42"
    assert result.steps == 1


def test_single_tool_call_then_answer():
    llm = MockLLM(responses=[
        LLMResponse(content="", tool_calls=[ToolCall("1", "add", {"a": 2, "b": 3})]),
        LLMResponse(content="The sum is 5."),
    ])
    agent = Agent("calc", "use tools", llm, tools=[add])
    result = agent.run("2+3?")
    assert result.output == "The sum is 5."
    assert result.steps == 2
    # The tool result (5) must have been appended to the conversation the agent
    # sent on its SECOND call.
    second_call_messages = llm.calls[1]
    assert any("5" in str(m.get("content")) for m in second_call_messages)


def test_unknown_tool_becomes_observation_not_crash():
    llm = MockLLM(responses=[
        LLMResponse(content="", tool_calls=[ToolCall("1", "does_not_exist", {})]),
        LLMResponse(content="recovered"),
    ])
    agent = Agent("a", "role", llm, tools=[add])
    result = agent.run("go")
    assert result.output == "recovered"
    # The model's 2nd call should include an "unknown tool" observation.
    assert any("unknown tool" in str(m.get("content")) for m in llm.calls[1])


def test_tool_that_raises_becomes_observation():
    llm = MockLLM(responses=[
        LLMResponse(content="", tool_calls=[ToolCall("1", "broken", {})]),
        LLMResponse(content="handled the error"),
    ])
    agent = Agent("a", "role", llm, tools=[broken])
    result = agent.run("go")
    assert result.output == "handled the error"
    assert any("Tool error" in str(m.get("content")) for m in llm.calls[1])


def test_bad_arguments_become_observation():
    llm = MockLLM(responses=[
        # Missing required 'b'.
        LLMResponse(content="", tool_calls=[ToolCall("1", "add", {"a": 1})]),
        LLMResponse(content="ok"),
    ])
    agent = Agent("a", "role", llm, tools=[add])
    result = agent.run("go")
    assert result.output == "ok"
    assert any("Tool error" in str(m.get("content")) for m in llm.calls[1])


def test_max_steps_exceeded():
    # The model keeps calling a tool forever; max_steps must stop it.
    looping = MockLLM(responses=lambda msgs: LLMResponse(
        content="", tool_calls=[ToolCall("1", "add", {"a": 1, "b": 1})]
    ))
    agent = Agent("a", "role", looping, tools=[add], max_steps=3)
    with pytest.raises(MaxStepsExceeded):
        agent.run("loop forever")


def test_context_injected_into_prompt():
    llm = MockLLM(responses=[LLMResponse(content="done")])
    agent = Agent("a", "role", llm)
    agent.run("write report", context={"researcher": "the sky is blue"})
    # The user message should carry the upstream context.
    user_msg = llm.calls[0][-1]["content"]
    assert "the sky is blue" in user_msg
    assert "researcher" in user_msg


def test_result_exposes_cost_and_tokens():
    llm = MockLLM(responses=[LLMResponse(content="hi")])
    agent = Agent("a", "role", llm)
    result = agent.run("q")
    # Convenience accessors delegate to the tracer.
    assert result.cost_usd == result.tracer.total_cost()
    assert result.total_tokens == result.tracer.total_usage().total_tokens


async def test_async_run_matches_sync():
    llm = MockLLM(responses=[
        LLMResponse(content="", tool_calls=[ToolCall("1", "add", {"a": 4, "b": 5})]),
        LLMResponse(content="nine"),
    ])
    agent = Agent("a", "role", llm, tools=[add])
    result = await agent.arun("4+5?")
    assert result.output == "nine"
    assert result.steps == 2
