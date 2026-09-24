"""agentcore — a dependency-minimal multi-agent orchestrator.

Build agents, tools, and DAG workflows on top of *raw* LLM APIs — no LangChain,
no LangGraph, no CrewAI. The only runtime dependency is ``httpx``.

Quick start
-----------
::

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
    print(result.output)          # -> "5"
    print(result.summary())       # -> steps/tokens/cost/duration one-liner

This module re-exports the whole public API so you can import everything from the
top-level package. Everything else in the package is an implementation detail you
*can* reach into, but shouldn't need to.
"""

from __future__ import annotations

# -- core building blocks ---------------------------------------------------
from .agent import Agent, AgentResult
from .errors import (
    AgentCoreError,
    CheckpointError,
    LLMError,
    MaxStepsExceeded,
    ToolError,
    ToolExecutionError,
    WorkflowError,
)
from .llm import (
    AnthropicClient,
    LLMClient,
    LLMResponse,
    MockLLM,
    OpenAIClient,
    ToolCall,
    Usage,
    estimate_cost,
)
from .memory import Memory
from .persistence import Checkpoint
from .tools import Tool, build_tool, tool
from .tracing import EventType, ExecutionTracer, TraceEvent
from .workflow import Node, Workflow, WorkflowResult

__version__ = "0.1.0"

__all__ = [
    # agent
    "Agent",
    "AgentResult",
    # tools
    "tool",
    "build_tool",
    "Tool",
    # llm
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "MockLLM",
    "LLMResponse",
    "ToolCall",
    "Usage",
    "estimate_cost",
    # memory
    "Memory",
    # tracing
    "ExecutionTracer",
    "TraceEvent",
    "EventType",
    # workflow
    "Workflow",
    "WorkflowResult",
    "Node",
    # persistence
    "Checkpoint",
    # errors
    "AgentCoreError",
    "ToolError",
    "ToolExecutionError",
    "MaxStepsExceeded",
    "LLMError",
    "WorkflowError",
    "CheckpointError",
    # meta
    "__version__",
]
