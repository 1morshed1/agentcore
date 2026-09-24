"""Memory — where an agent keeps its conversation.

An LLM is *stateless*. Every API call must include the entire conversation so
far; the model remembers nothing between calls on its own. "Memory," therefore,
is nothing more mystical than **a list of messages we keep appending to and
resend every turn.**

This module wraps that list in a small class so that:

* the agent has one obvious place to read/write conversation state,
* we can serialise it (for persistence / checkpoints), and
* we can later add smarter behaviour (trimming, summarisation) in one spot
  without touching the ReAct loop.

Message shape
-------------
A message is a dict with at least a ``role``. The roles you'll see:

* ``system``    — the instructions/persona, sent once at the top.
* ``user``      — input from the human (or an upstream agent).
* ``assistant`` — the model's own replies (including its tool-call requests).
* ``tool``      — the result of running a tool (OpenAI dialect).

We store provider-native message dicts as-is, because each provider is fussy
about getting its own message shapes back on the next turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    """A conversation history: an ordered list of message dicts.

    Attributes:
        messages: The raw message list, in the order they occurred. This is
            exactly what gets sent to the LLM each turn.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)

    # -- adding messages -----------------------------------------------------
    def add(self, message: dict[str, Any]) -> None:
        """Append a fully-formed provider-native message dict."""
        self.messages.append(message)

    def add_system(self, content: str) -> None:
        """Add (or you may prepend) a system instruction message."""
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        """Add a user turn."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Add a plain assistant text turn (no tool calls)."""
        self.messages.append({"role": "assistant", "content": content})

    # -- reading -------------------------------------------------------------
    def to_list(self) -> list[dict[str, Any]]:
        """Return the message list to send to the LLM.

        We return the live list (not a copy) because the agent appends the
        provider's raw assistant message to it directly during the loop. If you
        need an immutable snapshot, use :meth:`snapshot`.
        """
        return self.messages

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the messages — safe to store or inspect."""
        return list(self.messages)

    def last_assistant_text(self) -> str:
        """Return the text of the most recent assistant message, if any.

        Useful for pulling out an agent's final answer without re-parsing the
        whole history.
        """
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    return content
                # Anthropic-style content blocks: join the text blocks.
                if isinstance(content, list):
                    return "".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
        return ""

    def __len__(self) -> int:
        return len(self.messages)

    # -- persistence helpers -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for JSON checkpointing)."""
        return {"messages": self.messages}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Rebuild a Memory from :meth:`to_dict` output."""
        return cls(messages=list(data.get("messages", [])))
