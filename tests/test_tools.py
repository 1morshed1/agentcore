"""Tests for the tool system (schema generation, validation, execution).

These tests document the *contract* of `@tool`: what schema a given function
should produce, and how tools behave when called with good and bad arguments.
"""

from __future__ import annotations

from typing import Optional

import pytest

from agentcore import Tool, tool
from agentcore.errors import ToolError, ToolExecutionError
from agentcore.tools import parse_tool_arguments, validate_arguments


def test_tool_generates_schema_from_type_hints():
    @tool(description="Add two integers")
    def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two integers"
    # Both params are required (no defaults) and typed as integers.
    assert add.parameters == {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }


def test_default_values_make_params_optional():
    @tool(description="Search")
    def search(query: str, limit: int = 5) -> str:
        return query

    # `query` has no default -> required; `limit` has a default -> not required.
    assert search.parameters["required"] == ["query"]
    assert "limit" in search.parameters["properties"]


def test_optional_type_is_unwrapped():
    @tool(description="Maybe")
    def maybe(x: Optional[str] = None) -> str:
        return x or ""

    # Optional[str] should become a plain string type, optionality via `required`.
    assert maybe.parameters["properties"]["x"] == {"type": "string"}
    assert maybe.parameters["required"] == []


def test_list_type_becomes_array_with_items():
    @tool(description="Sum a list")
    def total(values: list[int]) -> int:
        return sum(values)

    assert total.parameters["properties"]["values"] == {
        "type": "array",
        "items": {"type": "integer"},
    }


def test_docstring_provides_description_and_param_docs():
    @tool()  # no explicit description -> use the docstring
    def web_search(query: str, limit: int = 5) -> str:
        """Search the web and return the top results.

        Args:
            query: The search terms to look up.
            limit: Maximum number of results to return.
        """
        return query

    assert web_search.description == "Search the web and return the top results."
    assert web_search.parameters["properties"]["query"]["description"] == "The search terms to look up."


def test_missing_description_raises():
    with pytest.raises(ToolError):
        @tool()
        def no_desc(x: int) -> int:  # no description arg AND no docstring
            return x


def test_var_args_rejected():
    with pytest.raises(ToolError):
        @tool(description="bad")
        def variadic(*args: int) -> int:
            return sum(args)


def test_openai_and_anthropic_schemas_differ():
    @tool(description="Echo")
    def echo(text: str) -> str:
        return text

    openai = echo.to_openai_schema()
    anthropic = echo.to_anthropic_schema()
    assert openai["type"] == "function"
    assert openai["function"]["name"] == "echo"
    # Anthropic uses input_schema at the top level, no "function" wrapper.
    assert anthropic["name"] == "echo"
    assert "input_schema" in anthropic
    assert "function" not in anthropic


def test_execute_runs_the_function():
    @tool(description="Add")
    def add(a: int, b: int) -> int:
        return a + b

    assert add.execute(a=2, b=3) == 5


def test_execute_wraps_exceptions():
    @tool(description="Boom")
    def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ToolExecutionError) as exc:
        boom.execute()
    # The original exception is preserved for debugging.
    assert isinstance(exc.value.original, ValueError)


async def test_async_tool_execution():
    @tool(description="Async add")
    async def aadd(a: int, b: int) -> int:
        return a + b

    # Sync execute() should still work by running the coroutine to completion.
    assert aadd.execute(a=1, b=1) == 2
    # And aexecute() awaits it directly.
    assert await aadd.aexecute(a=4, b=5) == 9


def test_validate_arguments_missing_required():
    schema = {"properties": {"q": {"type": "string"}}, "required": ["q"], "name": "t"}
    with pytest.raises(ToolExecutionError):
        validate_arguments(schema, {})


def test_validate_arguments_unexpected():
    schema = {"properties": {"q": {"type": "string"}}, "required": [], "name": "t"}
    with pytest.raises(ToolExecutionError):
        validate_arguments(schema, {"nope": 1})


def test_parse_tool_arguments_handles_string_and_dict():
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}
    assert parse_tool_arguments({"a": 1}) == {"a": 1}
    assert parse_tool_arguments("") == {}
