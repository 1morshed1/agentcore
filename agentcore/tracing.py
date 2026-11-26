"""Execution tracing — seeing exactly what an agent did.

An agent that you cannot observe is impossible to debug or trust. This module
records a structured **trace** of everything that happens during a run: every LLM
call (with its latency and token cost), every tool call (with arguments and
result), and every agent handoff in a workflow.

The trace is:

* **inspectable in code** — it's a tree of dataclasses,
* **exportable as JSON** — for logging or later analysis, and
* **exportable as a self-contained HTML timeline** — a visual you can open in a
  browser and put in a portfolio README.

Structure of a trace
---------------------
::

    Trace
     └── events: list[TraceEvent]      (flat, time-ordered)
          - llm_call     (model, latency, usage, cost)
          - tool_call    (name, arguments, result, latency)
          - agent_start / agent_end / handoff / error

We keep events as a **flat, time-ordered list** rather than a nested tree. Flat is
simpler to append to from anywhere, trivially serialisable, and still easy to
render — each event carries an ``agent`` label so a workflow's multi-agent trace
stays readable.
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .llm import Usage, estimate_cost


class EventType(str, Enum):
    """The kinds of things that can happen during a run.

    Subclassing ``str`` means these serialise straight to their string value in
    JSON, so a trace file is human-readable.
    """

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    HANDOFF = "handoff"
    ERROR = "error"


@dataclass
class TraceEvent:
    """A single recorded event on the timeline.

    Not every field applies to every event type — an ``llm_call`` uses
    ``usage``/``cost``, a ``tool_call`` uses ``tool_name``/``arguments``/``result``.
    Unused fields simply stay at their defaults, which keeps the structure flat and
    the JSON self-describing.

    Attributes:
        type: The :class:`EventType`.
        agent: Which agent produced the event (for multi-agent workflows).
        timestamp: Unix time when the event was recorded.
        duration_s: How long the operation took, in seconds.
        model: For LLM calls, the model id used.
        usage: For LLM calls, the token usage.
        cost_usd: For LLM calls, the estimated dollar cost.
        tool_name: For tool calls, the tool that ran.
        arguments: For tool calls, the arguments passed.
        result: For tool calls, a (possibly truncated) string of the return value.
        message: Free-form text (used for handoffs, errors, and notes).
    """

    type: EventType
    agent: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_s: float = 0.0
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value  # Enum -> its string value
        return data


class ExecutionTracer:
    """Collects :class:`TraceEvent` objects during a single agent or workflow run.

    The tracer is deliberately dumb: it just appends events. The clever part is in
    how it's *used* — the agent wraps each LLM/tool call with a timer and hands the
    measurements here.

    Typical usage inside the agent::

        with tracer.time_llm_call(agent="researcher", model="gpt-4o") as record:
            response = llm.complete(...)
            record(response.usage)     # attach usage once we have it

    The context manager measures wall-clock latency automatically, so the caller
    only has to supply the usage.
    """

    def __init__(self, agent_name: str = ""):
        self.agent_name = agent_name
        self.events: list[TraceEvent] = []
        self._start_time = time.time()

    # -- low-level append ----------------------------------------------------
    def record(self, event: TraceEvent) -> None:
        self.events.append(event)

    # -- convenience recorders ----------------------------------------------
    def log_agent_start(self, agent: str, task: str) -> None:
        self.record(TraceEvent(type=EventType.AGENT_START, agent=agent, message=task))

    def log_agent_end(self, agent: str, output: str) -> None:
        self.record(TraceEvent(type=EventType.AGENT_END, agent=agent, message=_truncate(output)))

    def log_handoff(self, from_agent: str, to_agent: str) -> None:
        self.record(
            TraceEvent(
                type=EventType.HANDOFF,
                agent=from_agent,
                message=f"{from_agent} -> {to_agent}",
            )
        )

    def log_error(self, agent: str, message: str) -> None:
        self.record(TraceEvent(type=EventType.ERROR, agent=agent, message=message))

    def log_llm_call(self, agent: str, model: str, usage: Usage, duration_s: float) -> None:
        self.record(
            TraceEvent(
                type=EventType.LLM_CALL,
                agent=agent,
                model=model,
                usage=usage,
                cost_usd=estimate_cost(model, usage),
                duration_s=duration_s,
            )
        )

    def log_tool_call(
        self,
        agent: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        duration_s: float,
    ) -> None:
        self.record(
            TraceEvent(
                type=EventType.TOOL_CALL,
                agent=agent,
                tool_name=tool_name,
                arguments=arguments,
                result=_truncate(str(result)),
                duration_s=duration_s,
            )
        )

    # -- aggregate metrics ---------------------------------------------------
    def total_cost(self) -> float:
        """Sum of estimated cost across every LLM call."""
        return round(sum(e.cost_usd for e in self.events), 6)

    def total_usage(self) -> Usage:
        """Sum of token usage across every LLM call."""
        total = Usage()
        for event in self.events:
            if event.type == EventType.LLM_CALL:
                total = total + event.usage
        return total

    def total_duration(self) -> float:
        """Wall-clock seconds from tracer creation to now."""
        return round(time.time() - self._start_time, 3)

    def count(self, event_type: EventType) -> int:
        return sum(1 for e in self.events if e.type == event_type)

    # -- exports -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Full trace as a JSON-serialisable dict, including a summary block."""
        usage = self.total_usage()
        return {
            "agent": self.agent_name,
            "summary": {
                "total_cost_usd": self.total_cost(),
                "total_duration_s": self.total_duration(),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "llm_calls": self.count(EventType.LLM_CALL),
                "tool_calls": self.count(EventType.TOOL_CALL),
            },
            "events": [e.to_dict() for e in self.events],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    def summary(self) -> str:
        """A one-line human summary, à la the plan's example output."""
        s = self.to_dict()["summary"]
        return (
            f"steps: {s['llm_calls']} LLM / {s['tool_calls']} tool | "
            f"tokens: {s['total_tokens']} | "
            f"cost: ${s['total_cost_usd']:.4f} | "
            f"duration: {s['total_duration_s']:.2f}s"
        )

    def merge(self, other: ExecutionTracer) -> None:
        """Absorb another tracer's events (used to roll agent traces up to a
        workflow-level trace)."""
        self.events.extend(other.events)

    # -- HTML timeline -------------------------------------------------------
    def to_html(self, title: str = "agentcore trace") -> str:
        """Render a self-contained HTML timeline of the run.

        No external CSS/JS — everything is inline so the file opens anywhere and
        can be committed as an artifact. Each event becomes a colour-coded row on
        a vertical timeline, with per-event details and a summary header.
        """
        return _render_html(self, title)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = 500) -> str:
    """Keep traces readable — long tool outputs are clipped with an ellipsis."""
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


# Colours per event type for the HTML timeline (kept muted and readable).
_EVENT_COLOURS = {
    EventType.AGENT_START: "#2563eb",
    EventType.AGENT_END: "#16a34a",
    EventType.LLM_CALL: "#7c3aed",
    EventType.TOOL_CALL: "#ea580c",
    EventType.HANDOFF: "#0891b2",
    EventType.ERROR: "#dc2626",
}


def _render_html(tracer: ExecutionTracer, title: str) -> str:
    data = tracer.to_dict()
    summary = data["summary"]

    rows: list[str] = []
    for event in tracer.events:
        colour = _EVENT_COLOURS.get(event.type, "#64748b")
        # Build the detail line depending on the event type.
        if event.type == EventType.LLM_CALL:
            detail = (
                f"model <code>{html.escape(event.model)}</code> · "
                f"{event.usage.total_tokens} tok · "
                f"${event.cost_usd:.5f} · {event.duration_s:.2f}s"
            )
        elif event.type == EventType.TOOL_CALL:
            args = html.escape(json.dumps(event.arguments))
            detail = (
                f"<code>{html.escape(event.tool_name)}({args})</code> "
                f"→ {html.escape(event.result)} · {event.duration_s:.2f}s"
            )
        else:
            detail = html.escape(event.message)

        label = event.type.value.replace("_", " ")
        agent = html.escape(event.agent or "")
        rows.append(
            f"""
            <div class="event">
              <div class="dot" style="background:{colour}"></div>
              <div class="body">
                <div class="head">
                  <span class="badge" style="background:{colour}">{label}</span>
                  <span class="agent">{agent}</span>
                </div>
                <div class="detail">{detail}</div>
              </div>
            </div>"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 2rem; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 1rem; }}
  .summary {{ display: flex; flex-wrap: wrap; gap: .75rem; margin-bottom: 1.5rem; }}
  .stat {{ background: #1e293b; border-radius: .5rem; padding: .6rem .9rem; }}
  .stat b {{ display: block; font-size: 1.1rem; }}
  .stat span {{ font-size: .75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }}
  .timeline {{ border-left: 2px solid #334155; margin-left: .5rem; }}
  .event {{ position: relative; padding: .4rem 0 .4rem 1.5rem; }}
  .dot {{ position: absolute; left: -7px; top: .8rem; width: 12px; height: 12px; border-radius: 50%; }}
  .head {{ display: flex; align-items: center; gap: .5rem; }}
  .badge {{ color: white; font-size: .7rem; padding: .1rem .5rem; border-radius: .3rem; text-transform: uppercase; letter-spacing: .03em; }}
  .agent {{ font-size: .8rem; color: #94a3b8; }}
  .detail {{ font-size: .85rem; margin-top: .2rem; color: #cbd5e1; }}
  code {{ background: #1e293b; padding: .05rem .3rem; border-radius: .25rem; font-size: .8rem; }}
</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="summary">
    <div class="stat"><b>${summary["total_cost_usd"]:.4f}</b><span>cost</span></div>
    <div class="stat"><b>{summary["total_tokens"]}</b><span>tokens</span></div>
    <div class="stat"><b>{summary["llm_calls"]}</b><span>llm calls</span></div>
    <div class="stat"><b>{summary["tool_calls"]}</b><span>tool calls</span></div>
    <div class="stat"><b>{summary["total_duration_s"]:.2f}s</b><span>duration</span></div>
  </div>
  <div class="timeline">
    {"".join(rows)}
  </div>
</body>
</html>"""
