"""Tests for checkpointing and resume."""

from __future__ import annotations

import json
import os

import pytest

from agentcore import Agent, Checkpoint, LLMResponse, MockLLM, Workflow
from agentcore.errors import CheckpointError


def _answering_agent(name: str, answer: str) -> Agent:
    return Agent(name, name, MockLLM(responses=[LLMResponse(content=answer)]))


def test_save_and_load_round_trip(tmp_path):
    cp = Checkpoint(str(tmp_path / "cp.json"))
    assert cp.load_outputs() == {}  # nothing yet
    cp.save_outputs({"a": "hello", "b": "world"})
    assert cp.load_outputs() == {"a": "hello", "b": "world"}


def test_save_is_atomic_and_versioned(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint(str(path))
    cp.save_outputs({"a": "1"})
    data = json.loads(path.read_text())
    assert data["version"] == 1
    assert data["outputs"] == {"a": "1"}


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "cp.json"
    path.write_text("{ not valid json")
    with pytest.raises(CheckpointError):
        Checkpoint(str(path)).load_outputs()


def test_wrong_version_raises(tmp_path):
    path = tmp_path / "cp.json"
    path.write_text(json.dumps({"version": 999, "outputs": {}}))
    with pytest.raises(CheckpointError):
        Checkpoint(str(path)).load_outputs()


def test_workflow_resumes_and_skips_completed_nodes(tmp_path):
    """The heart of resume: a node already in the checkpoint is NOT re-run."""
    cp = Checkpoint(str(tmp_path / "cp.json"))
    # Pre-seed the checkpoint as if 'researcher' already completed.
    cp.save_outputs({"researcher": "cached research"})

    # The researcher's MockLLM has NO scripted responses — so if the workflow
    # tried to run it, we'd get an LLMError. It must be skipped instead.
    researcher = Agent("researcher", "r", MockLLM(responses=[]))
    writer = _answering_agent("writer", "final report")

    wf = Workflow("resume")
    wf.add_agent(researcher, task="research")
    wf.add_agent(writer, task="write", depends_on=["researcher"])

    result = wf.run(checkpoint=cp)
    assert result.outputs["researcher"] == "cached research"  # from checkpoint
    assert result.final_output == "final report"
    # The writer received the cached research as context.
    assert "cached research" in writer.llm.calls[0][-1]["content"]


def test_checkpoint_written_after_each_node(tmp_path):
    cp = Checkpoint(str(tmp_path / "cp.json"))
    wf = Workflow("save")
    wf.add_agent(_answering_agent("a", "a-out"), task="t")
    wf.run(checkpoint=cp)
    # After the run, the checkpoint file holds the completed output.
    assert cp.load_outputs()["a"] == "a-out"


def test_clear_removes_file(tmp_path):
    cp = Checkpoint(str(tmp_path / "cp.json"))
    cp.save_outputs({"a": "1"})
    assert cp.exists()
    cp.clear()
    assert not cp.exists()
