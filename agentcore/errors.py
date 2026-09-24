"""Custom exception hierarchy for agentcore.

Why a dedicated module for exceptions?
--------------------------------------
When you build a library, the *errors you raise are part of your public API*.
Callers write ``try/except`` blocks against them, so giving each failure mode a
named, well-documented class is just as important as naming your functions well.

Every exception here inherits from a single base class, :class:`AgentCoreError`.
That means a user who just wants to catch "anything this library throws" can
write::

    try:
        agent.run(task)
    except AgentCoreError as exc:
        ...  # handle any agentcore-specific failure

while a user who wants to react to one *specific* situation can catch the
narrower subclass instead.
"""

from __future__ import annotations


class AgentCoreError(Exception):
    """Base class for every exception raised by agentcore.

    Catch this to handle *any* library-specific error. It intentionally adds no
    behaviour of its own — it exists purely to give the hierarchy a common root.
    """


class ToolError(AgentCoreError):
    """Raised when something goes wrong while *defining* or *registering* a tool.

    This covers static/setup problems — e.g. two tools sharing the same name, or
    a tool being defined with an unsupported parameter type. It is distinct from
    a tool *raising at runtime*: a runtime failure inside a tool's body is caught
    by the agent and fed back to the model as an observation, not surfaced here.
    """


class ToolExecutionError(AgentCoreError):
    """Wraps an exception raised *inside* a tool's body during execution.

    The agent catches the original error, records it on the trace, and (by
    default) reports the message back to the model so it can recover. We keep the
    original exception around on :attr:`original` for debugging.
    """

    def __init__(self, tool_name: str, original: BaseException):
        self.tool_name = tool_name
        self.original = original
        super().__init__(f"Tool {tool_name!r} raised: {original!r}")


class MaxStepsExceeded(AgentCoreError):
    """Raised when an agent's ReAct loop hits its ``max_steps`` safety limit.

    A ReAct loop is, by construction, unbounded — the model keeps calling tools
    until it decides to answer. ``max_steps`` is the circuit breaker that stops a
    misbehaving agent from looping forever (and burning tokens). Hitting it is an
    error, not a normal termination.
    """


class LLMError(AgentCoreError):
    """Raised when a raw LLM API call fails in a way we cannot recover from.

    Recoverable failures (timeouts, HTTP 429/5xx) are retried internally with
    exponential backoff; this is only raised once retries are exhausted, or for
    non-retryable failures such as an authentication error (HTTP 401).
    """


class WorkflowError(AgentCoreError):
    """Raised for structural problems with a :class:`~agentcore.workflow.Workflow`.

    Examples: adding two agents under the same node name, declaring a dependency
    on an agent that was never added, or building a graph that contains a cycle
    (a DAG, by definition, must be *acyclic*).
    """


class CheckpointError(AgentCoreError):
    """Raised when saving or loading a workflow checkpoint fails.

    For example: trying to resume from a checkpoint file that does not exist, or
    one whose format version we do not understand.
    """
