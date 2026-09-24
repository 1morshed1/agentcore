"""The Agent — a ReAct loop around an LLM.

This is the core of the whole library. Everything else (tools, clients, memory,
tracing) exists to serve the loop implemented here. If you read one file to
understand how agents work, read this one.

The ReAct loop
--------------
"ReAct" = *Reason + Act*. The algorithm is tiny::

    build the conversation (system prompt + task)
    repeat up to max_steps times:
        response = LLM(conversation, tools)
        if response asks to call tools:
            run each tool, append its result to the conversation
            continue          # let the model reason about the results
        else:
            return response.content    # no tool calls == final answer
    raise MaxStepsExceeded

That's it. `create_react_agent` in LangGraph does a great deal more bookkeeping,
but the beating heart is these ~15 lines. Below, the same loop is fleshed out
with tracing, error handling, and both sync and async variants.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import MaxStepsExceeded, ToolExecutionError
from .llm import LLMClient, LLMResponse, ToolCall
from .memory import Memory
from .tools import Tool, validate_arguments
from .tracing import ExecutionTracer


@dataclass
class AgentResult:
    """The outcome of an :meth:`Agent.run` call.

    Attributes:
        output: The agent's final text answer.
        agent: The name of the agent that produced it.
        steps: How many ReAct iterations (LLM calls) it took.
        tracer: The full :class:`ExecutionTracer` for the run — inspect it for
            timing, cost, and the exact tool calls made.
        memory: The final conversation state (handy for handoffs / debugging).
    """

    output: str
    agent: str
    steps: int
    tracer: ExecutionTracer
    memory: Memory

    # Convenience pass-throughs so callers don't have to reach into the tracer.
    @property
    def cost_usd(self) -> float:
        return self.tracer.total_cost()

    @property
    def total_tokens(self) -> int:
        return self.tracer.total_usage().total_tokens

    def summary(self) -> str:
        return self.tracer.summary()


class Agent:
    """A single autonomous worker: a role, some tools, and a ReAct loop.

    Args:
        name: Unique, human-readable identifier (used in traces and handoffs).
        role: The system prompt — the agent's persona and instructions.
        llm: An :class:`~agentcore.llm.LLMClient` (OpenAI, Anthropic, or Mock).
            This is how we stay provider-agnostic: the agent never knows which
            vendor it's talking to.
        tools: The tools this agent may call. May be empty (a "pure reasoning"
            agent that just answers).
        max_steps: Safety limit on ReAct iterations. Prevents infinite tool loops.
        temperature: Sampling temperature for the model (low = more deterministic).

    Example::

        agent = Agent(
            name="mathematician",
            role="You are a careful mathematician. Use tools for arithmetic.",
            llm=OpenAIClient("gpt-4o-mini"),
            tools=[add, multiply],
        )
        result = agent.run("What is (2 + 3) * 4?")
        print(result.output)
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm: LLMClient,
        tools: list[Tool] | None = None,
        *,
        max_steps: int = 10,
        temperature: float = 0.1,
    ):
        self.name = name
        self.role = role
        self.llm = llm
        # Index tools by name for O(1) lookup when the model calls one.
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.max_steps = max_steps
        self.temperature = temperature

    # -- prompt construction -------------------------------------------------
    def _build_system_prompt(self) -> str:
        """The system message. Kept simple: the role is the instruction."""
        return self.role

    def _build_task_prompt(self, task: str, context: dict[str, Any] | None) -> str:
        """Combine the task with any shared context passed from upstream agents.

        In a workflow, an agent often needs the *output of the agents before it*.
        We serialise that context into the user message so the model can see it.
        """
        if not context:
            return task
        context_lines = "\n".join(f"- {key}: {value}" for key, value in context.items())
        return f"{task}\n\nContext from previous steps:\n{context_lines}"

    def _tool_schemas(self) -> list[dict[str, Any]] | None:
        """Serialise our tools into the dialect this agent's provider expects."""
        if not self.tools:
            return None
        if self.llm.tool_schema_dialect == "anthropic":
            return [t.to_anthropic_schema() for t in self.tools.values()]
        return [t.to_openai_schema() for t in self.tools.values()]

    # -- the synchronous ReAct loop -----------------------------------------
    def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        *,
        memory: Memory | None = None,
        tracer: ExecutionTracer | None = None,
    ) -> AgentResult:
        """Run the agent to completion on ``task`` and return an :class:`AgentResult`.

        Args:
            task: What to do.
            context: Optional dict of upstream results to include in the prompt.
            memory: Optional pre-seeded conversation (used to *resume* from a
                checkpoint). If omitted, a fresh conversation is started.
            tracer: Optional shared tracer (the workflow passes one so all agents
                write to a single trace). If omitted, the agent makes its own.

        Raises:
            MaxStepsExceeded: if the loop runs ``max_steps`` times without the
                model producing a final answer.
        """
        # 1. Set up conversation + observability. Reuse any passed-in memory so a
        #    resumed run continues exactly where it left off.
        if memory is None:
            memory = Memory()
            memory.add_system(self._build_system_prompt())
            memory.add_user(self._build_task_prompt(task, context))
        tracer = tracer or ExecutionTracer(self.name)
        tracer.log_agent_start(self.name, task)

        tool_schemas = self._tool_schemas()

        # 2. The loop.
        for step in range(self.max_steps):
            response = self._call_llm(memory.to_list(), tool_schemas, tracer)

            # Append the model's own message verbatim BEFORE handling tool calls —
            # providers require their assistant message to precede the tool result.
            memory.add(response.raw_message or {"role": "assistant", "content": response.content})

            if not response.tool_calls:
                # No tools requested => this is the final answer. We're done.
                tracer.log_agent_end(self.name, response.content)
                return AgentResult(
                    output=response.content,
                    agent=self.name,
                    steps=step + 1,
                    tracer=tracer,
                    memory=memory,
                )

            # 3. The model wants to act: run every requested tool and feed results
            #    back into the conversation.
            for call in response.tool_calls:
                result = self._execute_tool(call, tracer)
                memory.add(self.llm.format_tool_result_message(call, result))
            # loop back around — the model will now reason about the observations

        # 4. Ran out of steps without a final answer.
        tracer.log_error(self.name, f"hit max_steps={self.max_steps}")
        raise MaxStepsExceeded(
            f"Agent {self.name!r} exceeded max_steps={self.max_steps} without a final answer."
        )

    # -- LLM + tool execution helpers (shared shape, timed) ------------------
    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None,
        tracer: ExecutionTracer,
    ) -> LLMResponse:
        """One timed LLM call, recorded on the trace."""
        start = time.time()
        response = self.llm.complete(messages, tool_schemas, temperature=self.temperature)
        tracer.log_llm_call(self.name, response.model or self.llm.model, response.usage, time.time() - start)
        return response

    def _execute_tool(self, call: ToolCall, tracer: ExecutionTracer) -> str:
        """Run one tool the model requested, returning a string observation.

        Every failure mode is turned into a *string* fed back to the model rather
        than an exception that crashes the run — this is what lets an agent
        recover from its own mistakes (wrong tool name, bad arguments, tool that
        raised). That resilience is a big part of what makes ReAct robust.
        """
        start = time.time()

        # Unknown tool: tell the model, don't crash.
        tool = self.tools.get(call.name)
        if tool is None:
            message = f"Error: unknown tool {call.name!r}. Available: {list(self.tools)}"
            tracer.log_tool_call(self.name, call.name, call.arguments, message, time.time() - start)
            return message

        # Validate arguments; a clear message lets the model self-correct.
        try:
            validate_arguments({**tool.parameters, "name": tool.name}, call.arguments)
            result = tool.execute(**call.arguments)
            result_str = str(result)
        except ToolExecutionError as exc:
            result_str = f"Tool error: {exc.original}"
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            result_str = f"Tool error: {exc}"

        tracer.log_tool_call(self.name, call.name, call.arguments, result_str, time.time() - start)
        return result_str

    # -- the asynchronous ReAct loop ----------------------------------------
    async def arun(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        *,
        memory: Memory | None = None,
        tracer: ExecutionTracer | None = None,
    ) -> AgentResult:
        """Async twin of :meth:`run`.

        Identical logic, but tools are awaited via :meth:`Tool.aexecute`, and the
        (blocking) LLM HTTP call is pushed to a thread so it doesn't block the
        event loop. Use this when you want to run many agents concurrently on one
        thread via ``asyncio.gather``.
        """
        if memory is None:
            memory = Memory()
            memory.add_system(self._build_system_prompt())
            memory.add_user(self._build_task_prompt(task, context))
        tracer = tracer or ExecutionTracer(self.name)
        tracer.log_agent_start(self.name, task)

        tool_schemas = self._tool_schemas()

        for step in range(self.max_steps):
            start = time.time()
            # Run the sync HTTP client in a worker thread to stay non-blocking.
            response = await asyncio.to_thread(
                self.llm.complete, memory.to_list(), tool_schemas, temperature=self.temperature
            )
            tracer.log_llm_call(self.name, response.model or self.llm.model, response.usage, time.time() - start)

            memory.add(response.raw_message or {"role": "assistant", "content": response.content})

            if not response.tool_calls:
                tracer.log_agent_end(self.name, response.content)
                return AgentResult(response.content, self.name, step + 1, tracer, memory)

            for call in response.tool_calls:
                result = await self._aexecute_tool(call, tracer)
                memory.add(self.llm.format_tool_result_message(call, result))

        tracer.log_error(self.name, f"hit max_steps={self.max_steps}")
        raise MaxStepsExceeded(
            f"Agent {self.name!r} exceeded max_steps={self.max_steps} without a final answer."
        )

    async def _aexecute_tool(self, call: ToolCall, tracer: ExecutionTracer) -> str:
        start = time.time()
        tool = self.tools.get(call.name)
        if tool is None:
            message = f"Error: unknown tool {call.name!r}. Available: {list(self.tools)}"
            tracer.log_tool_call(self.name, call.name, call.arguments, message, time.time() - start)
            return message
        try:
            validate_arguments({**tool.parameters, "name": tool.name}, call.arguments)
            result = await tool.aexecute(**call.arguments)
            result_str = str(result)
        except ToolExecutionError as exc:
            result_str = f"Tool error: {exc.original}"
        except Exception as exc:  # noqa: BLE001
            result_str = f"Tool error: {exc}"
        tracer.log_tool_call(self.name, call.name, call.arguments, result_str, time.time() - start)
        return result_str
