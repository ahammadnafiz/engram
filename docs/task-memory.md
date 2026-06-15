# Task Memory

Task memory is the part of Engram built for agents that keep working across
many turns, tool calls, large prompts, and process restarts.

Engram keeps two memory planes connected:

| Plane | Tables | Purpose |
|-------|--------|---------|
| Fact memory | `agent_memory`, `memory_relations` | searchable, typed, deduped facts and graph relations |
| Task memory | `agent_task_runs`, `agent_events`, `agent_checkpoints`, `memory_jobs` | durable task state, raw ledger, checkpoints, and background derivation |

Fact memory is optimized for retrieval. Task memory is optimized for continuity
and auditability.

## When To Use Task Memory

Use task memory when:

- work spans more than one turn
- the agent should resume after process restart
- raw user, assistant, tool, and artifact events matter
- memory extraction should run after the user-facing response
- you need compact checkpoints instead of replaying every event
- large prompts or documents need source anchors

For a short stateless chatbot, direct `add()` and `search()` may be enough.

## Start A Task

```python
task = await engram.start_task(
    "Ship persistent memory for the coding agent",
    "codex",
    user_id="nafiz",
    metadata={"repo": "engram"},
)

print(task.task_run_id)
print(task.status)
```

Task status can move through `active`, `paused`, `completed`, `failed`, and
`cancelled`.

```python
await engram.pause_task(task.task_run_id, outcome="Waiting on review")
await engram.complete_task(task.task_run_id, outcome="Docs and tests passed")
```

Terminal statuses are guarded by the task manager.

## Record Turns And Tool Output

`record_turn()` writes user and assistant messages, optional tool calls, optional
artifacts, and a background `turn_ingest` job in one transaction.

```python
events = await engram.record_turn(
    task.task_run_id,
    user_message="Implement deterministic recall and failure traces.",
    assistant_response="I will add policy metadata, trace_recall, and tests.",
    tool_calls=[{"name": "pytest", "result": "273 passed"}],
    artifacts=[{"path": "docs/api-reference.md", "type": "markdown"}],
)
```

Use `record_event()` for single ledger entries.

```python
event = await engram.record_event(
    agent_id="codex",
    task_run_id=task.task_run_id,
    user_id="nafiz",
    role="tool",
    event_type="tool_result",
    content="ruff check src tests: All checks passed",
    payload={"command": "ruff check src tests", "exit_code": 0},
)
```

Use `redact_event()` when raw event content must be removed while preserving
audit metadata.

```python
redacted = await engram.redact_event(event.event_id)
print(redacted.redacted_at)
```

## Process Memory Jobs

`record_turn()` queues a `turn_ingest` job by default. Processing that job can
derive facts with `add_conversation()` when an LLM is configured, and it always
updates task checkpoints.

```python
jobs = await engram.process_memory_jobs(limit=10)

for job in jobs:
    print(job.job_id, job.status, job.error)
```

For production, run a worker process:

```python
async with Engram(memory_policy="coding_agent") as engram:
    await engram.run_memory_worker(batch_size=20, interval_seconds=1.0)
```

For tests, scripts, or local demos, inline `process_memory_jobs()` is easier to
inspect.

## Create Checkpoints

Checkpoints are compact task state. Use them for resumability; use events for
audit history; use fact memory for retrieval.

```python
checkpoint = await engram.create_checkpoint(
    task.task_run_id,
    "Ruff is clean and unit tests pass.",
    completed_steps=["Clean Ruff", "Restore CI lint gates"],
    pending_steps=["Update docs", "Run final package checks"],
    decisions=["Keep public API examples in docs syntax-checked"],
    source_event_ids=[event.event_id],
)
```

## Build Resume Context

`build_context()` combines task metadata, recent events, checkpoints, memory
search, and optional graph expansion.

```python
context = await engram.build_context(
    task.task_run_id,
    query="resume implementation",
    max_tokens=120000,
    recent_event_limit=40,
    memory_limit=25,
    checkpoint_limit=3,
    include_graph=True,
)

print(context.text)
print(context.sections.keys())
```

The result is deterministic enough for prompt assembly and cache-friendly
context building.

## Recommended Turn Loop

```python
async def handle_turn(engram, task_id, user_message):
    context = await engram.build_context(
        task_id,
        query=user_message,
        max_tokens=120000,
        include_graph=True,
    )

    response = await call_your_llm(context.text, user_message)

    await engram.record_turn(
        task_id,
        user_message=user_message,
        assistant_response=response,
        enqueue_processing=True,
    )

    return response
```

Run `run_memory_worker()` separately so the user-facing request does not wait on
fact extraction.

## Long Input

Use `record_long_input()` for large prompts, legal documents, specs, or any
source where exact text matters.

```python
report = await engram.record_long_input(
    task.task_run_id,
    text=huge_prompt_or_document,
    title="Vendor agreement review",
    max_chunk_tokens=700,
    extract_with_llm=True,
)

print(report.source_event_id)
print(report.manifest)
```

`record_long_input()`:

1. stores the raw input as a source event
2. splits it into chunks with character spans and quote hashes
3. stores each chunk as an artifact event
4. extracts or heuristically derives anchored facts
5. creates a manifest checkpoint

Build source-aware context when answering:

```python
context = await engram.build_long_input_context(
    task.task_run_id,
    query="What are the termination obligations?",
    expected_terms=["termination", "notice"],
    max_tokens=4000,
)

print(context.text)
print(context.trace["source_chunk_event_ids"])
print(context.trace["missing_expected_terms"])
```

For legal or exact-document answers, prefer source chunks over distilled memory
summaries and require source chunk IDs in the application response.

## Critical Recall In Long Tasks

Vector search alone can miss broad but important facts. Policy-backed critical
memory gives you deterministic recall.

```python
trace = await engram.trace_recall(
    "Can we order dinner and continue the repo work?",
    "assistant",
    user_id="user_123",
    expected_terms=["shellfish", "never revert"],
    max_tokens=1200,
)

print(trace.context)
print(trace.critical_memory_ids)
print(trace.missing_expected_terms)
```

Use the trace to distinguish:

| Symptom | Trace field |
|---------|-------------|
| fact was never stored | absent from `critical_memory_ids` and `search_memory_ids` |
| fact was not critical | present in `search_memory_ids`, absent from `critical_memory_ids` |
| fact ranked but did not fit | present in `trimmed_memory_ids` |
| old fact was corrected | present in `superseded_memory_ids` |
| caller-required term is missing | present in `missing_expected_terms` |

## Evidence Retrieval For Aggregation

Aggregation questions often need coverage across sessions rather than the
single highest-ranked memory. Compose this from the public primitives:
`deep_search()` for high-recall retrieval, `get_memories()` to pull surrounding
turns from a session group, and `engram.llm` for a custom reader.

```python
hits = await engram.deep_search(
    "Which payments did Sarah make in March?",
    "assistant",
    user_id="sarah",
    limit=12,
)

# Expand a hit with the rest of its source session.
session_id = hits[0].memory.metadata.get("original_session_id")
group = await engram.get_memories(
    "assistant",
    user_id="sarah",
    metadata_filter={"original_session_id": session_id},
)
context = "\n".join(m.content for m in group)

if engram.llm is not None:
    answer = await engram.llm.complete(
        f"Context:\n{context}\n\n"
        "Which payments did Sarah make in March? Answer concisely.",
    )
```

A fuller reference implementation of this pattern — session-diversified
evidence selection, turn-window expansion, and a multi-call evidence-ledger
reader — lives in `scripts/longmemeval_harness.py`
(`search_evidence_set`, `get_neighboring_context_block`,
`answer_from_evidence`). It is QA-harness machinery built on these same public
APIs, useful for LongMemEval-style workloads, support history, and
multi-session personal memory.

## Current Limits

- Search is scoped by `agent_id` and optionally `user_id`; authorization is the
  application layer's responsibility.
- `trace_recall()` only reports missing terms that callers pass through
  `expected_terms`.
- Long-input anchoring records text spans and quote hashes. PDF page numbers,
  OCR coordinates, and external citation IDs must be supplied by the caller.
- Job processing is durable, but applications still need monitoring and retry
  policy around failed `memory_jobs`.
- The schema is alpha. Back up data before migrations.
