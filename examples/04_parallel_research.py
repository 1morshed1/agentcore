"""A branched DAG: two researchers run in parallel, then merge.

Requires an API key:

    export OPENAI_API_KEY=sk-...
    python examples/04_parallel_research.py

Graph shape:

    news_researcher  ─┐
                      ├─► synthesizer ─► editor
    filings_reader   ─┘

`news_researcher` and `filings_reader` have no dependency on each other, so the
workflow engine puts them in the same level and runs them **concurrently** in
threads. The synthesizer waits for both and receives both outputs as context.
This is where the DAG engine earns its keep versus a simple linear chain.
"""

from __future__ import annotations

from agentcore import Agent, OpenAIClient, Workflow, tool


@tool(description="Fetch recent news headlines about a company")
def get_news(company: str) -> str:
    """Return recent news snippets for a company (stubbed for the example)."""
    return f"[news] {company} announced a new product line and beat earnings estimates."


@tool(description="Fetch summary figures from a company's latest financial filing")
def get_filings(company: str) -> str:
    """Return filing highlights for a company (stubbed for the example)."""
    return f"[filings] {company}: revenue +18% YoY, operating margin 24%, cash reserves strong."


def main() -> None:
    llm = OpenAIClient("gpt-4o-mini")

    news_researcher = Agent(
        name="news_researcher",
        role="You gather and summarise recent news. Use get_news.",
        llm=llm,
        tools=[get_news],
    )
    filings_reader = Agent(
        name="filings_reader",
        role="You extract key figures from financial filings. Use get_filings.",
        llm=llm,
        tools=[get_filings],
    )
    synthesizer = Agent(
        name="synthesizer",
        role="You combine news and financial data into a coherent picture.",
        llm=llm,
    )
    editor = Agent(
        name="editor",
        role="You polish the synthesis into a crisp investor note (max 5 sentences).",
        llm=llm,
    )

    wf = Workflow("company_briefing")
    wf.add_agent(news_researcher, task="Summarise recent news for Acme Corp")
    wf.add_agent(filings_reader, task="Summarise the latest filing for Acme Corp")
    wf.add_agent(
        synthesizer,
        task="Combine the news and filing summaries into one picture of Acme Corp",
        depends_on=["news_researcher", "filings_reader"],  # <-- waits for BOTH
    )
    wf.add_agent(editor, task="Write the final investor note", depends_on=["synthesizer"])

    print("Levels (each inner list runs in parallel):")
    for i, level in enumerate(wf._execution_levels()):
        print(f"  level {i}: {level}")

    print("\nRunning...\n")
    result = wf.run(max_workers=4, on_node_complete=lambda n, r: print(f"  ✓ {n}"))

    print("\n--- INVESTOR NOTE ---")
    print(result.final_output)
    print("\n--- SUMMARY ---")
    print(result.summary())


if __name__ == "__main__":
    main()
