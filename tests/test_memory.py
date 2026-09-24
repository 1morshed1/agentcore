"""Tests for the Memory conversation store."""

from __future__ import annotations

from agentcore import Memory


def test_add_helpers_build_expected_roles():
    mem = Memory()
    mem.add_system("you are helpful")
    mem.add_user("hi")
    mem.add_assistant("hello")
    assert [m["role"] for m in mem.to_list()] == ["system", "user", "assistant"]
    assert len(mem) == 3


def test_last_assistant_text_plain_string():
    mem = Memory()
    mem.add_user("q")
    mem.add_assistant("the answer")
    assert mem.last_assistant_text() == "the answer"


def test_last_assistant_text_from_content_blocks():
    # Anthropic-style assistant content is a list of blocks.
    mem = Memory()
    mem.add({"role": "assistant", "content": [{"type": "text", "text": "block answer"}]})
    assert mem.last_assistant_text() == "block answer"


def test_snapshot_is_a_copy():
    mem = Memory()
    mem.add_user("hi")
    snap = mem.snapshot()
    mem.add_user("again")
    # Mutating memory after snapshotting must not change the snapshot.
    assert len(snap) == 1
    assert len(mem) == 2


def test_round_trip_serialisation():
    mem = Memory()
    mem.add_system("sys")
    mem.add_user("u")
    restored = Memory.from_dict(mem.to_dict())
    assert restored.to_list() == mem.to_list()
