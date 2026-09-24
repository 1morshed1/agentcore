"""Persistence — checkpoint and resume a workflow.

Long multi-agent workflows are expensive: each node may make several paid LLM
calls. If the process crashes at node 4 of 5, you do **not** want to re-run (and
re-pay for) nodes 1–3. A *checkpoint* saves completed work to disk so a rerun can
skip straight to where it failed.

Design
------
We keep this deliberately small: a checkpoint is a single JSON file holding the
map of ``{node_name: output}`` for every node that has completed. On a resumed
run, the workflow loads that map, treats those nodes as done, and only executes
the rest.

Why JSON on disk (and not a database)? Because it's dependency-free, human
-readable (you can open the file and see exactly what's cached), and perfectly
adequate for the job. The interface is small enough that swapping in Redis or S3
later would touch only this file.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from .errors import CheckpointError

#: Bumped if the on-disk format ever changes, so we can refuse to load files we
#: don't understand rather than misinterpret them.
CHECKPOINT_VERSION = 1


class Checkpoint:
    """A file-backed store of completed workflow node outputs.

    Usage::

        cp = Checkpoint("checkpoints/market_research.json")
        result = workflow.run(checkpoint=cp)   # saves after each node
        # ... if it crashes, just run again with the same Checkpoint:
        result = workflow.run(checkpoint=cp)   # skips already-done nodes

    Args:
        path: Where to read/write the checkpoint JSON. Parent directories are
            created on first save.
    """

    def __init__(self, path: str):
        self.path = path

    # -- reading -------------------------------------------------------------
    def exists(self) -> bool:
        return os.path.exists(self.path)

    def load_outputs(self) -> dict[str, str]:
        """Return the saved ``{node: output}`` map, or ``{}`` if none exists yet.

        Raises:
            CheckpointError: if the file exists but is corrupt or from an
                incompatible format version.
        """
        if not self.exists():
            return {}
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"Could not read checkpoint {self.path!r}: {exc}") from exc

        version = data.get("version")
        if version != CHECKPOINT_VERSION:
            raise CheckpointError(
                f"Checkpoint {self.path!r} has version {version}, but this build "
                f"understands version {CHECKPOINT_VERSION}."
            )
        return dict(data.get("outputs", {}))

    # -- writing -------------------------------------------------------------
    def save_outputs(self, outputs: dict[str, str]) -> None:
        """Persist the ``{node: output}`` map atomically.

        We write to a temp file in the same directory and then ``os.replace`` it
        over the target. ``os.replace`` is atomic on every major OS, so a crash
        mid-write can never leave a half-written, corrupt checkpoint — you either
        have the old file or the new one, never a torn one. This "write-temp-then
        -rename" pattern is the standard way to update a file safely.
        """
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)

        payload: dict[str, Any] = {"version": CHECKPOINT_VERSION, "outputs": outputs}

        # NamedTemporaryFile in the target dir guarantees os.replace stays on the
        # same filesystem (a cross-device rename is not atomic).
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, self.path)
        except OSError as exc:
            # Clean up the temp file if the rename failed.
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise CheckpointError(f"Could not write checkpoint {self.path!r}: {exc}") from exc

    def clear(self) -> None:
        """Delete the checkpoint file (e.g. once a workflow has fully succeeded)."""
        if self.exists():
            os.unlink(self.path)
