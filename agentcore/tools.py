"""The tool system: turn a plain Python function into an LLM-callable tool.

This module is the heart of "function calling" done by hand. When you use
LangChain's ``@tool`` decorator, this is roughly what happens under the hood —
except here you can read every line.

The big idea
------------
An LLM cannot call your Python function directly. Instead, the API works like
this:

1. You describe your functions to the model as **JSON Schema** (name,
   description, and a typed list of parameters).
2. The model, instead of answering in prose, may respond with a *tool call*:
   "please run ``web_search`` with ``{"query": "..."}``".
3. Your code runs the real Python function, then sends the return value back to
   the model as an observation.
4. The model uses that observation to decide its next move.

So a "tool" is really two things glued together:

* a **schema** the model reads, and
* a **callable** your code runs.

The :func:`tool` decorator builds the schema *automatically* from the function's
type hints and docstring, so you never have to hand-write JSON Schema. That is
the whole developer-experience win.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from .errors import ToolError, ToolExecutionError

# ---------------------------------------------------------------------------
# Mapping Python types -> JSON Schema types
# ---------------------------------------------------------------------------
# JSON Schema only knows a handful of primitive types. We map the common Python
# annotations onto them. Anything we do not recognise falls back to "string",
# which is the safest default because the model can put any text there and we can
# coerce it later.
_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Translate a single Python type annotation into a JSON Schema fragment.

    Handles the cases that show up constantly in real tools:

    * plain primitives (``str``, ``int``, ``float``, ``bool``)
    * ``list[X]`` / ``dict`` containers
    * ``Optional[X]`` (i.e. ``X | None``) — we unwrap to the inner type, because
      "optional" is expressed via the ``required`` array, not the type itself.

    Returns a dict like ``{"type": "string"}`` or
    ``{"type": "array", "items": {"type": "integer"}}``.
    """
    # `Optional[X]` and `X | None` both show up as a typing.Union whose args
    # include NoneType. Strip the None and recurse on the remaining type.
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        # A genuine multi-type union (e.g. int | str) has no single JSON type,
        # so we leave the type open rather than guess wrong.
        return {}

    # Parameterised generics like list[int] / dict[str, int].
    if origin in (list, tuple, set):
        args = get_args(annotation)
        item_schema = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema or {"type": "string"}}
    if origin is dict:
        return {"type": "object"}

    # Plain, unparameterised types.
    json_type = _PYTHON_TO_JSON_TYPE.get(annotation)
    if json_type is not None:
        return {"type": json_type}

    # Unknown annotation -> treat as free-form string.
    return {"type": "string"}


def _parse_param_docs(docstring: str | None) -> dict[str, str]:
    """Extract per-parameter descriptions from a Google-style docstring.

    We look for an ``Args:`` (or ``Arguments:`` / ``Parameters:``) section and
    read lines of the form ``name: description``. This lets a tool author write
    one ordinary docstring and get rich, per-argument descriptions in the schema
    for free — exactly the kind of context that makes a model call a tool
    correctly.

    Example docstring understood by this parser::

        Search the web.

        Args:
            query: The search terms to look up.
            limit: Maximum number of results to return.
    """
    if not docstring:
        return {}

    lines = docstring.splitlines()
    param_docs: dict[str, str] = {}
    in_args = False
    current: str | None = None

    for raw in lines:
        line = raw.strip()
        header = line.rstrip(":").lower()
        if header in ("args", "arguments", "parameters"):
            in_args = True
            continue
        if in_args:
            # A new section header (e.g. "Returns:") ends the Args block.
            if header in ("returns", "return", "raises", "yields", "examples", "example", "note", "notes"):
                break
            if ":" in line:
                name, _, desc = line.partition(":")
                current = name.strip()
                param_docs[current] = desc.strip()
            elif current and line:
                # Continuation of the previous parameter's description.
                param_docs[current] += " " + line
    return param_docs


def _summary_from_docstring(docstring: str | None) -> str:
    """Return the first paragraph of a docstring, used as a fallback description."""
    if not docstring:
        return ""
    summary_lines: list[str] = []
    for raw in docstring.strip().splitlines():
        line = raw.strip()
        if not line:  # blank line ends the summary paragraph
            break
        summary_lines.append(line)
    return " ".join(summary_lines)


@dataclass
class Tool:
    """A single callable exposed to an LLM.

    You rarely construct this directly — the :func:`tool` decorator builds it for
    you. But it is a plain dataclass, so you *can* assemble one by hand (handy for
    wrapping a function you do not own, or for tests).

    Attributes:
        name: The identifier the model uses to call the tool. Must be unique
            within an agent.
        description: Natural-language explanation of what the tool does. The
            model reads this to decide *when* to use the tool, so make it good.
        func: The actual Python callable to run. May be sync or ``async``.
        parameters: JSON Schema ``object`` describing the tool's arguments.
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})

    # -- Schema serialisation ------------------------------------------------
    # The two big model vendors format tool schemas slightly differently, so we
    # provide one method per dialect. The agent picks the right one based on the
    # provider it is talking to.

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialise to OpenAI's ``tools`` format (a ``{"type": "function"}`` wrapper)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Serialise to Anthropic's ``tools`` format (``input_schema`` at top level)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    # -- Execution -----------------------------------------------------------
    def execute(self, **kwargs: Any) -> Any:
        """Run the tool synchronously and return its result.

        If the wrapped function is ``async``, we spin up an event loop to run it
        to completion so that a synchronous agent can still use async tools.

        Any exception raised *inside* the tool is wrapped in
        :class:`ToolExecutionError` so the caller can distinguish a tool failure
        from a framework bug.
        """
        try:
            if inspect.iscoroutinefunction(self.func):
                return asyncio.run(self.func(**kwargs))
            return self.func(**kwargs)
        except Exception as exc:  # noqa: BLE001 - we intentionally catch all tool errors
            raise ToolExecutionError(self.name, exc) from exc

    async def aexecute(self, **kwargs: Any) -> Any:
        """Run the tool from within an existing event loop (async agents).

        Sync tools are executed directly; async tools are awaited. We use the
        same :class:`ToolExecutionError` wrapping as :meth:`execute`.
        """
        try:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**kwargs)
            return self.func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(self.name, exc) from exc


def build_tool(func: Callable[..., Any], *, description: str | None = None, name: str | None = None) -> Tool:
    """Construct a :class:`Tool` from a function by introspecting it.

    This is the engine behind the :func:`tool` decorator, exposed separately so
    you can wrap functions programmatically too.

    The steps, in order:

    1. Read the function's resolved type hints (``get_type_hints`` evaluates
       string/`from __future__` annotations for us).
    2. Walk its signature parameter by parameter, converting each annotation to a
       JSON Schema fragment and pulling a description out of the docstring.
    3. Mark a parameter ``required`` if — and only if — it has no default value.
    4. Assemble the final ``{"type": "object", ...}`` schema.
    """
    hints = get_type_hints(func)
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)
    param_docs = _parse_param_docs(docstring)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in signature.parameters.items():
        # `self`/`cls` are bound implicitly and never part of the tool schema.
        if param_name in ("self", "cls"):
            continue
        # *args / **kwargs cannot be described by JSON Schema, so we reject them
        # early with a clear message rather than silently dropping them.
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolError(
                f"Tool {func.__name__!r} uses *args/**kwargs, which cannot be "
                f"expressed as a tool schema. Use explicit, typed parameters."
            )

        annotation = hints.get(param_name, str)
        schema = _annotation_to_schema(annotation)
        if param_name in param_docs:
            schema["description"] = param_docs[param_name]
        properties[param_name] = schema

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    resolved_description = description or _summary_from_docstring(docstring)
    if not resolved_description:
        raise ToolError(
            f"Tool {func.__name__!r} has no description. Pass "
            f"@tool(description=...) or add a docstring so the model knows what "
            f"the tool does."
        )

    return Tool(
        name=name or func.__name__,
        description=resolved_description,
        func=func,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


def tool(
    description: str | None = None,
    *,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator that turns a function into a :class:`Tool`.

    Usage::

        @tool(description="Search the web and return the top results")
        def web_search(query: str) -> str:
            ...

    Or lean on the docstring for the description::

        @tool()
        def web_search(query: str) -> str:
            \"\"\"Search the web and return the top results.\"\"\"
            ...

    Args:
        description: What the tool does. If omitted, the first paragraph of the
            function's docstring is used instead.
        name: Override the tool name (defaults to the function's own name).

    Returns:
        A decorator that replaces the function with a ready-to-use ``Tool``.
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return build_tool(func, description=description, name=name)

    return decorator


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    """Lightweight validation of model-provided arguments against a tool schema.

    Models occasionally hallucinate arguments — a missing required field, or an
    extra one that does not exist. We do a *cheap* check here (presence of
    required keys, no unexpected keys) rather than pulling in a full JSON Schema
    validator, because the goal is to give the model a clear, correctable error
    message, not to enforce a spec to the letter.

    Raises:
        ToolExecutionError: if the arguments are structurally invalid. Raising
            this (rather than a plain ValueError) means the agent's normal tool
            error handling reports it back to the model, which can then retry.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolExecutionError(
            schema.get("name", "tool"),
            ValueError(f"Missing required argument(s): {', '.join(missing)}"),
        )

    unexpected = [key for key in arguments if key not in properties]
    if unexpected:
        raise ToolExecutionError(
            schema.get("name", "tool"),
            ValueError(f"Unexpected argument(s): {', '.join(unexpected)}"),
        )


def parse_tool_arguments(raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
    """Coerce the model's raw tool arguments into a dict.

    OpenAI returns tool-call arguments as a JSON *string*; Anthropic returns them
    as an already-parsed dict. This helper normalises both so the agent does not
    have to care which provider produced them.
    """
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not raw_arguments:
        return {}
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError("tool", ValueError(f"Invalid JSON arguments: {exc}")) from exc
