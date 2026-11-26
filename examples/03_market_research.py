"""The market-research workflow — with a live model.

Requires an API key:

    export OPENAI_API_KEY=sk-...
    python examples/03_market_research.py

Three specialised agents in a linear pipeline:

    researcher --> analyst --> writer

Each has its own role and tools; the workflow passes each agent the previous
agent's output as context. Includes checkpointing, so if it crashes partway you
can re-run and it resumes.

Note on the web_search tool: to keep the example self-contained and free, the
"search" here returns canned data. Swap in a real search API (Tavily, SerpAPI,
Bing, ...) inside `web_search` and everything else works unchanged — that's the
point of the tool abstraction.
"""

from __future__ import annotations

from agentcore import Agent, Checkpoint, OpenAIClient, Workflow, tool


@tool(description="Search the web for information and return relevant snippets")
def web_search(query: str) -> str:
    """Search the web.

    Args:
        query: The search query.
    """
    # ---- Replace this stub with a real search API call in production. ----
    canned = {
        "crm pricing": (
            "HubSpot Sales: $90/seat/mo. Salesforce: $165/seat/mo. "
            "Pipedrive: $49/seat/mo. Zoho CRM: $40/seat/mo."
        ),
    }
    for key, value in canned.items():
        if key in query.lower():
            return value
    return "No results found for that query."


@tool(description="Compute the average of a list of numbers")
def average(numbers: list[float]) -> float:
    return round(sum(numbers) / len(numbers), 2) if numbers else 0.0


def main() -> None:
    llm = OpenAIClient("gpt-4o-mini")

    researcher = Agent(
        name="researcher",
        role="You are a market researcher. Use web_search to gather competitor pricing data.",
        llm=llm,
        tools=[web_search],
    )
    analyst = Agent(
        name="analyst",
        role="You are a pricing analyst. Use the average tool to compute stats and identify trends.",
        llm=llm,
        tools=[average],
    )
    writer = Agent(
        name="writer",
        role="You are a business writer. Produce a concise one-page pricing report.",
        llm=llm,
    )

    wf = Workflow("market_research")
    wf.add_agent(researcher, task="Research competitor pricing for CRM software")
    wf.add_agent(
        analyst,
        task="Analyse the pricing data and identify the average and trends",
        depends_on=["researcher"],
    )
    wf.add_agent(
        writer, task="Write a one-page CRM pricing analysis report", depends_on=["analyst"]
    )

    # Checkpoint so a crash mid-run can be resumed by re-running this script.
    checkpoint = Checkpoint("checkpoints/market_research.json")

    print("Running market_research workflow (this makes real API calls)...\n")
    result = wf.run(checkpoint=checkpoint, on_node_complete=lambda n, r: print(f"  ✓ {n}"))

    print("\n--- FINAL REPORT ---")
    print(result.final_output)
    print("\n--- SUMMARY ---")
    print(result.summary())

    result.tracer.save_json("market_research.trace.json")
    with open("market_research.trace.html", "w", encoding="utf-8") as fh:
        fh.write(result.tracer.to_html("Market Research"))
    print("\nWrote market_research.trace.json and .html")

    # Clear the checkpoint now that we finished successfully.
    checkpoint.clear()


if __name__ == "__main__":
    main()
