## Project B: Multi-Agent Orchestrator from Scratch (No LangChain)

### Why This Project

Your CV lists LangChain, LangGraph, CrewAI, AutoGen, Google ADK, and MCP. That is five agent frameworks. The immediate question an interviewer has is: "Can you build without them, or do you only know how to configure them?"

This project answers that question definitively. You build a multi-agent orchestration system using raw OpenAI/Anthropic API calls — no LangChain, no LangGraph, no CrewAI. Just Python, `httpx`, and your own abstractions.

This is not about avoiding frameworks for the sake of it. It's about proving that you understand the internals: how tool-calling actually works at the API level, how conversation state is managed across agent handoffs, how to implement ReAct loops, how to handle streaming with tool interrupts, and how to do all of this reliably with retries and state recovery.

The result is a lightweight, dependency-minimal library that does 80% of what LangGraph does in 1/10th the code. It also becomes a talking point in every interview: "I used LangGraph at work, but I wanted to understand what it does under the hood, so I built my own."

### What You Build

A Python library (`agentcore`) that provides:
- Agent definitions with roles, tools, and memory
- A ReAct execution loop using raw API function calling
- Multi-agent workflows with DAG-based task orchestration
- Agent-to-agent handoffs with shared context
- Streaming output with tool-call interrupts
- Structured state persistence and recovery
- Full execution tracing and cost tracking

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  agentcore (Python library, zero LLM framework dependencies)   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Agent                                                   │   │
│  │                                                         │   │
│  │  class Agent:                                           │   │
│  │      name: str                                          │   │
│  │      role: str              # system prompt             │   │
│  │      model: str             # "gpt-4o", "claude-..."    │   │
│  │      tools: list[Tool]      # callable functions        │   │
│  │      max_steps: int         # safety limit              │   │
│  │      memory: Memory         # conversation history      │   │
│  │                                                         │   │
│  │      def run(task, context) -> AgentResult               │   │
│  │      # internally: ReAct loop with raw API calls        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Tool System                                             │   │
│  │                                                         │   │
│  │  @tool(description="Search the web for information")    │   │
│  │  def web_search(query: str) -> str:                     │   │
│  │      ...                                                │   │
│  │                                                         │   │
│  │  - Auto-generates JSON Schema from type hints           │   │
│  │  - Validates tool arguments before execution            │   │
│  │  - Handles tool errors gracefully                       │   │
│  │  - Supports async tools                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Workflow Engine                                         │   │
│  │                                                         │   │
│  │  workflow = Workflow("research_report")                  │   │
│  │  workflow.add_agent(researcher, depends_on=[])          │   │
│  │  workflow.add_agent(analyst, depends_on=[researcher])   │   │
│  │  workflow.add_agent(writer, depends_on=[analyst])       │   │
│  │  result = workflow.run("Analyze Q4 sales trends")       │   │
│  │                                                         │   │
│  │  - DAG execution with parallel branches                 │   │
│  │  - Shared context passed between agents                 │   │
│  │  - Checkpoint/resume on failure                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Execution Tracer                                        │   │
│  │                                                         │   │
│  │  Every LLM call, tool call, and agent handoff is logged │   │
│  │  - Latency per step                                     │   │
│  │  - Tokens used per call                                 │   │
│  │  - Cost per agent, per workflow                         │   │
│  │  - Full conversation history exportable as JSON         │   │
│  │  - Visual timeline (HTML export)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Implementation: The ReAct Loop Without LangChain

This is the core. Most engineers have never seen what LangGraph's `create_react_agent` actually does. Here it is, raw:

```python
import httpx
import json
from typing import Any

class Agent:
    def __init__(self, name: str, role: str, model: str, tools: list, max_steps: int = 10):
        self.name = name
        self.role = role
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.tracer = ExecutionTracer()
    
    def run(self, task: str, context: dict | None = None) -> AgentResult:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_task_prompt(task, context)},
        ]
        
        tool_schemas = [t.to_openai_schema() for t in self.tools.values()]
        
        for step in range(self.max_steps):
            # Raw API call — no framework
            response = self._call_llm(messages, tool_schemas)
            
            message = response["choices"][0]["message"]
            messages.append(message)
            
            # Check: did the model want to call a tool?
            if tool_calls := message.get("tool_calls"):
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])
                    
                    # Execute the tool
                    self.tracer.log_tool_call(func_name, func_args)
                    try:
                        result = self.tools[func_name].execute(**func_args)
                    except Exception as e:
                        result = f"Tool error: {str(e)}"
                    
                    self.tracer.log_tool_result(func_name, result)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(result),
                    })
                
                continue  # loop back — let the LLM think about the results
            
            # No tool call = final answer
            return AgentResult(
                output=message["content"],
                trace=self.tracer.export(),
                total_steps=step + 1,
                total_cost=self.tracer.total_cost(),
            )
        
        raise MaxStepsExceeded(f"Agent '{self.name}' hit {self.max_steps} step limit")
    
    def _call_llm(self, messages: list, tools: list) -> dict:
        """Raw HTTP call to OpenAI/Anthropic. No SDK, no framework."""
        with httpx.Client() as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools if tools else None,
                    "temperature": 0.1,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()
```

**The Tool Decorator (auto-generating JSON Schema from type hints):**

This is what LangChain's `@tool` does. Build it yourself:

```python
import inspect
from typing import get_type_hints

def tool(description: str):
    def decorator(func):
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        
        # Build OpenAI function schema from Python type hints
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            param_type = hints.get(param_name, str)
            json_type = {str: "string", int: "integer", float: "number", bool: "boolean"}
            
            properties[param_name] = {
                "type": json_type.get(param_type, "string"),
                "description": "",  # can be enriched with docstring parsing
            }
            
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        
        func._tool_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        func._tool_name = func.__name__
        func._tool_execute = func
        
        return Tool(
            name=func.__name__,
            description=description,
            execute=func,
            schema=func._tool_schema,
        )
    return decorator
```

**Usage (the developer experience should be this clean):**

```python
from agentcore import Agent, Workflow, tool

@tool(description="Search the web and return results")
def web_search(query: str) -> str:
    # Your search implementation
    ...

@tool(description="Read the contents of a webpage")
def read_page(url: str) -> str:
    ...

@tool(description="Perform mathematical calculations")
def calculator(expression: str) -> str:
    return str(eval(expression))  # sandboxed in production

# Define agents
researcher = Agent(
    name="researcher",
    role="You are a research analyst. Find information using web search.",
    model="gpt-4o",
    tools=[web_search, read_page],
)

analyst = Agent(
    name="analyst",
    role="You analyze data and extract key insights. Use the calculator for numerical analysis.",
    model="gpt-4o-mini",
    tools=[calculator],
)

writer = Agent(
    name="writer",
    role="You write clear, structured reports based on provided research and analysis.",
    model="gpt-4o",
    tools=[],
)

# Build workflow
workflow = Workflow("market_research")
workflow.add_agent(researcher, task="Research competitor pricing for CRM software")
workflow.add_agent(analyst, task="Analyze the pricing data and identify trends", depends_on=["researcher"])
workflow.add_agent(writer, task="Write a 1-page pricing analysis report", depends_on=["analyst"])

result = workflow.run()
print(result.final_output)
print(result.trace.summary())
# Total steps: 14 | Total cost: $0.12 | Duration: 34s
```

### What This Proves to Hiring Managers

- You understand LLM function calling at the protocol level, not just the framework level.
- You can build production-quality abstractions (clean API, type-safe, well-documented).
- You know what LangChain/LangGraph actually do under the hood.
- You can work without your preferred frameworks if the project requires it.

### README Structure

1. "Why I Built This" section explaining that you wanted to understand agent internals, not just use frameworks.
2. Side-by-side code comparison: the same agent in LangGraph vs. `agentcore`, showing that yours is simpler and more transparent.
3. A working example (market research workflow) that someone can run in 2 minutes.
4. Execution trace visualization (an HTML timeline showing each agent's steps, tool calls, and handoffs).
5. Performance comparison: latency and cost vs equivalent LangGraph implementation.

---
---