# 07 — Persistence: Checkpoint & Resume

Pairs with: `agentcore/persistence.py`

## The problem: crashes are expensive

A 5-node workflow might make 15 LLM calls. If it dies at node 4, re-running from
scratch means paying again for nodes 1–3 and waiting through them again. A
**checkpoint** saves completed work so a rerun resumes where it failed.

## The design: a tiny JSON file

A checkpoint is one JSON file holding the map of completed node outputs:

```json
{
  "version": 1,
  "outputs": {
    "researcher": "found: competitor pricing is ...",
    "analyst": "trend: prices rose 12% YoY ..."
  }
}
```

Usage is deliberately effortless:

```python
cp = Checkpoint("checkpoints/market_research.json")
result = workflow.run(checkpoint=cp)  # saves after each node completes
# ...if it crashes, run the exact same line again:
result = workflow.run(checkpoint=cp)  # researcher & analyst are skipped
```

On resume, `workflow.run` calls `checkpoint.load_outputs()`, seeds the `outputs`
map, and its per-level loop skips any node already present. So finished nodes are
never re-executed.

### Why JSON on disk, not a database?

- **Zero dependencies** — stays true to the project's one-dependency rule.
- **Human-readable** — open the file and see exactly what's cached.
- **Sufficient** — the whole state is a small string→string map.

The interface (`load_outputs` / `save_outputs` / `clear` / `exists`) is small
enough that swapping in Redis or S3 later would touch only this one file. That's
the point of keeping persistence behind a narrow class.

## The important detail: atomic writes

Look at `save_outputs`. It does **not** open the target file and write into it
directly. Instead it:

1. writes to a temporary file in the *same directory*, then
2. `os.replace(tmp, target)` — renames it over the target.

Why bother? Because if the process is killed **mid-write**, a direct write leaves a
half-written, corrupt JSON file — and now your checkpoint is worse than useless.
`os.replace` is **atomic** on every major OS: at any instant the target is either
the complete old file or the complete new file, never a torn one. The temp file
lives in the same directory so the rename stays on one filesystem (a cross-device
rename isn't atomic). This "write-temp-then-rename" trick is the standard way to
update any file safely, and it's worth internalising — it applies far beyond this
project.

## Version stamping

Each file carries `"version": 1`. If the format ever changes, `load_outputs`
refuses a file it doesn't understand (raising `CheckpointError`) rather than
silently misreading old data. Cheap insurance against a whole class of
"mysterious corruption" bugs.

## What is *not* checkpointed (and why)

We checkpoint node **outputs**, not the mid-flight conversation *inside* a single
agent's ReAct loop. Rationale: node granularity is the natural, cheap unit — if an
agent dies mid-loop, re-running just that one agent is a small, acceptable cost,
whereas serialising every partial loop state would add a lot of complexity for
little gain. (The `Agent.run` API *does* accept a pre-seeded `Memory`, so
finer-grained resume is possible to build on top — it's a deliberate extension
point, left as one.)

This is the last note. Head back to the [index](./README.md), or open the
[main README](../README.md) to run the examples.
