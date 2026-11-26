"""Offline demo — the whole library working, with NO API key and NO network.

Run it:

    pip install -e .
    python examples/01_offline_demo.py

How it works
------------
We drive the agents with `MockLLM`, a fake client that returns a *scripted*
sequence of responses instead of calling a real model. That lets you watch the
exact mechanics — tool calls, handoffs, tracing — deterministically. Read the
scripted responses below alongside `docs/05-react-loop.md`.

This is the "market research" workflow: a researcher gathers data, an
analyst computes a number with a tool, and a writer produces the final report.
"""

from __future__ import annotations

from agentcore import (
    Agent,
    LLMResponse,
    MockLLM,
    ToolCall,
    Workflow,
    tool,
)


# --- a tool the analyst will call ------------------------------------------
@tool(description="Compute the percentage change from an old value to a new value")
def pct_change(old: float, new: float) -> float:
    """Compute percentage change.

    Args:
        old: The original value.
        new: The new value.
    """
    return round((new - old) / old * 100, 1)


def build_workflow() -> Workflow:
    # Each agent gets its own MockLLM scripted to play its part in the pipeline.

    # 1) Researcher: just "reports" findings (no tools) in a single step.
    researcher = Agent(
        name="researcher",
        role="You research competitor pricing.",
        llm=MockLLM(
            responses=[
                LLMResponse(
                    content="Competitor CRM pricing: last year $80/seat, this year $92/seat."
                ),
            ]
        ),
    )

    # 2) Analyst: calls the pct_change tool, then reports the trend.
    #    Response 1 asks to run the tool; response 2 (after seeing "15.0")
    #    gives the analysis. That's one full ReAct iteration.
    analyst = Agent(
        name="analyst",
        role="You analyse pricing data. Use tools for arithmetic.",
        llm=MockLLM(
            responses=[
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall("call_1", "pct_change", {"old": 80, "new": 92})],
                ),
                LLMResponse(content="Pricing rose 15.0% year over year — an aggressive increase."),
            ]
        ),
        tools=[pct_change],
    )

    # 3) Writer: turns the analysis into a short report (no tools).
    writer = Agent(
        name="writer",
        role="You write concise, executive-ready reports.",
        llm=MockLLM(
            responses=[
                LLMResponse(
                    content=(
                        "PRICING BRIEF\n"
                        "Competitor CRM pricing increased from $80 to $92 per seat (+15.0%).\n"
                        "Recommendation: revisit our own pricing to stay competitive."
                    )
                ),
            ]
        ),
    )

    wf = Workflow("market_research")
    wf.add_agent(researcher, task="Research competitor CRM pricing")
    wf.add_agent(analyst, task="Analyse the pricing trend", depends_on=["researcher"])
    wf.add_agent(writer, task="Write a one-paragraph pricing brief", depends_on=["analyst"])
    return wf


def main() -> None:
    wf = build_workflow()

    # A progress callback so we can see nodes complete in real time.
    def on_done(name: str, result) -> None:
        print(f"  ✓ {name} finished in {result.steps} step(s)")

    print("Running workflow 'market_research'...\n")
    result = wf.run(on_node_complete=on_done)

    print("\n--- FINAL REPORT ---")
    print(result.final_output)

    print("\n--- TRACE SUMMARY ---")
    print(result.summary())

    # Write a visual HTML timeline you can open in a browser.
    out = "market_research.trace.html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(result.tracer.to_html("Market Research — agentcore trace"))
    print(f"\nWrote HTML timeline to {out} (open it in a browser).")


if __name__ == "__main__":
    main()
