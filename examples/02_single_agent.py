"""Single agent with real tools against a live model.

Requires an API key:

    export OPENAI_API_KEY=sk-...      # or ANTHROPIC_API_KEY for AnthropicClient
    python examples/02_single_agent.py

This is the smallest "real" example: one agent, three tools, a live ReAct loop.
Swap the provider by changing the ONE marked line below.
"""

from __future__ import annotations

import math

from agentcore import Agent, OpenAIClient, tool

# To use Anthropic instead, swap the import and the client line:
#   from agentcore import AnthropicClient
#   llm = AnthropicClient("claude-haiku-5-5")


@tool(description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="Multiply two numbers")
def multiply(a: float, b: float) -> float:
    return a * b


@tool(description="Compute the square root of a number")
def sqrt(x: float) -> float:
    """Compute a square root.

    Args:
        x: A non-negative number.
    """
    return math.sqrt(x)


def main() -> None:
    llm = OpenAIClient("gpt-4o-mini")  # <-- the one line to change for another provider

    agent = Agent(
        name="mathematician",
        role=(
            "You are a careful mathematician. Break problems into steps and use "
            "the provided tools for every arithmetic operation — never compute in "
            "your head."
        ),
        llm=llm,
        tools=[add, multiply, sqrt],
        max_steps=8,
    )

    task = "What is the square root of (12 + 4) multiplied by 3?"
    print(f"Task: {task}\n")

    result = agent.run(task)

    print("Answer:", result.output)
    print("\nTrace summary:", result.summary())

    # Print the tool calls the model actually made, in order.
    print("\nTool calls made:")
    for event in result.tracer.events:
        if event.type.value == "tool_call":
            print(f"  {event.tool_name}({event.arguments}) -> {event.result}")


if __name__ == "__main__":
    main()
