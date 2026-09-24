# agentcore examples

Four runnable examples, from "no API key needed" to "real multi-agent workflow".

| File | Needs an API key? | Shows |
|------|-------------------|-------|
| [`01_offline_demo.py`](./01_offline_demo.py) | **No** — uses `MockLLM` | The whole pipeline (agent → tools → workflow → trace) with scripted model responses. Run this first. |
| [`02_single_agent.py`](./02_single_agent.py) | Yes | A single ReAct agent with real tools against a live model |
| [`03_market_research.py`](./03_market_research.py) | Yes | The 3-agent research→analysis→report workflow from the plan |
| [`04_parallel_research.py`](./04_parallel_research.py) | Yes | A branched DAG where two researchers run in parallel |

## Running the offline demo (no setup)

```bash
pip install -e .          # installs agentcore + httpx
python examples/01_offline_demo.py
```

It prints each agent's steps, the final report, a cost/latency summary, and writes
an HTML trace timeline you can open in a browser.

## Running the live examples

Set whichever key matches the client you use:

```bash
export OPENAI_API_KEY=sk-...        # for OpenAIClient
export ANTHROPIC_API_KEY=sk-ant-... # for AnthropicClient
python examples/03_market_research.py
```

Each live example is written to swap providers by changing a single line — see the
comment near the top of each file.
