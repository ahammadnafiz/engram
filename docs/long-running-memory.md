# Long-Running Agent Memory

This page documents the memory framework added for agents that run across long
sessions, large prompts, tool calls, and multi-day tasks. It is the canonical
design document for the new architecture.

Engram now has two connected memory planes:

| Plane | Tables | Purpose |
|-------|--------|---------|
| Fact memory | `agent_memory`, `memory_relations` | Searchable, typed, deduped facts and graph relations |
| Task memory | `agent_tasks`, `agent_events`, `task_checkpoints`, `memory_jobs` | Durable task state, raw ledger, checkpoints, and background derivation |

The fact plane is optimized for retrieval. The task plane is optimized for
continuity and auditability.

## Why This Exists

Vector search alone is not reliable enough for production agent memory. A broad
query can rank the wrong thing above an allergy, a repo constraint, a corrected
deadline, or a user preference. Long-running agents also need raw event history,
source anchors, and traceability when recall fails.

The new design addresses four requirements:

1. Deterministic recall for critical facts.
2. First-class conflict resolution for old versus corrected facts.
3. Memory typing with different retrieval policies.
4. Failure observability for missed retrievals.

## Memory Types

Every memory has a `memory_type`.

| Type | Use |
|------|-----|
| `semantic` | General durable fact |
| `episodic` | Event or dated narrative |
| `procedural` | Behavioral rule or process |
| `profile` | Identity, health, relationships, location |
| `project` | Project facts, codenames, owners, metrics |
| `task` | Requirements and current work state |
| `preference` | Stable user preferences and style |
| `constraint` | Hard rules, repo constraints, deadlines, safety limits |
| `decision` | Explicit decisions, corrections, approvals |
| `tool_result` | Test results, tool output, external observations |

`search(..., memory_types=[...])`, `deep_search(...)`, `trace_recall(...)`, and
context builders can restrict retrieval to one or more types.

## Memory Policy

`MemoryPolicy` controls automatic typing, critical slots, and conflict keys.

```python
from engram import Engram

async with Engram(memory_policy="coding_agent") as engram:
    await engram.add(
        "Repo constraint: never revert user changes without explicit approval",
        agent_id="codex",
        user_id="nafiz",
    )
```

Built-in policies:

| Policy | Use |
|--------|-----|
| `default` | Personal assistants and general agents |
| `legal` | Legal or exact-document review where citations and source chunks matter |
| `coding_agent` | Coding agents with repo constraints, tool results, and implementation decisions |

You can pass a custom policy:

```python
from engram.policy import MemoryPolicy, SlotRule, TypeRule

policy = MemoryPolicy(
    name="support",
    type_rules=(
        TypeRule("profile", (r"\baccount id\b", r"\bplan\b")),
        TypeRule("constraint", (r"\bSLA\b", r"\bescalate\b")),
    ),
    slot_rules=(
        SlotRule("support:account_plan", (r"\bplan\b",), ("profile",)),
        SlotRule("support:sla", (r"\bSLA\b",), ("constraint",)),
    ),
)

async with Engram(memory_policy=policy) as engram:
    ...
```

## Critical Slots And Conflict Keys

Critical memories get deterministic metadata:

```json
{
  "critical": true,
  "critical_slot": "profile:allergy:cashews",
  "conflict_key": "assistant:user_123:profile:allergy:cashews",
  "status": "active",
  "version": 1
}
```

When a new memory has the same `conflict_key`, older active memories are marked:

```json
{
  "status": "superseded",
  "superseded_by": "mem_new",
  "superseded_at": "2026-06-09T..."
}
```

Active search filters superseded memories by default. `trace_recall()` can still
report superseded memory IDs so debugging can answer, "Was the old fact hidden by
conflict resolution?"

## Deterministic Critical Recall

`recall_critical()` bypasses vector ranking and retrieves active critical facts
directly by policy metadata:

```python
critical = await engram.recall_critical(
    agent_id="assistant",
    user_id="user_123",
    memory_types=["profile", "preference", "constraint"],
)
```

`trace_recall()` always places critical memories before vector-ranked memories in
the prompt budget:

```python
trace = await engram.trace_recall(
    "Can we order dinner and continue the repo work?",
    agent_id="assistant",
    user_id="user_123",
    expected_terms=["shellfish", "never revert"],
    max_tokens=1200,
)

print(trace.context)
print(trace.critical_memory_ids)
print(trace.missing_expected_terms)
```

## Recall Trace

`RecallTrace` is the observability record for one retrieval:

| Field | Meaning |
|-------|---------|
| `critical_memory_ids` | Deterministically recalled critical facts |
| `search_memory_ids` | Memories returned by vector/keyword search |
| `ranked_memory_ids` | Critical + search memories after dedupe |
| `kept_memory_ids` | Memories included in the final prompt block |
| `trimmed_memory_ids` | Ranked memories dropped by token budget |
| `superseded_memory_ids` | Old conflict losers hidden from active recall |
| `missing_expected_terms` | Operator-supplied expected terms not present in final context |
| `notes` | Human-readable debugging flags |

This is designed for incident review. If an agent misses a fact, inspect whether
the fact was never stored, stored but not critical, ranked but trimmed, or
superseded.

## Long-Running Task Flow

Use task memory when work spans many turns or must survive process restarts.

```python
async with Engram(memory_policy="coding_agent") as engram:
    task = await engram.start_task(
        "Ship persistent memory for the coding agent",
        agent_id="codex",
        user_id="nafiz",
    )

    await engram.record_turn(
        task.task_run_id,
        user_message="Implement deterministic recall and failure traces.",
        assistant_response="I will add policy metadata, trace_recall, and tests.",
        tool_calls=[{"name": "pytest", "result": "190 passed"}],
    )

    await engram.process_memory_jobs(limit=10)

    context = await engram.build_context(
        task.task_run_id,
        query="resume implementation",
        max_tokens=200000,
    )
```

`record_turn()` writes raw user/assistant/tool/artifact events to
`agent_events`, then queues a `memory_jobs` item. `process_memory_jobs()` derives
facts and checkpoints. `build_context()` combines task metadata, recent events,
checkpoints, typed memory search, and optional graph expansion.

## Event Ledger

`agent_events` is append-oriented. It records:

| Role | Event types |
|------|-------------|
| `user` | `user_message`, source documents, long prompts |
| `assistant` | `assistant_message` |
| `tool` | `tool_call`, `tool_result` |
| `agent` | `agent_action`, `decision`, `artifact`, `observation` |
| `system` | `system_note`, `error` |

Events can be redacted with `redact_event(event_id)`. Redaction clears content
and payload but preserves audit metadata.

## Checkpoints

Checkpoints are compact task state:

```python
await engram.create_checkpoint(
    task.task_run_id,
    "Ruff is clean and unit tests pass.",
    completed_steps=["Clean Ruff", "Restore full CI lint gates"],
    pending_steps=["Update docs", "Run final package checks"],
    decisions=["Keep Pydantic type aliases as runtime imports"],
    source_event_ids=[...],
)
```

Use checkpoints for resumability. Use events for audit history. Use fact memory
for search and prompt recall.

## Long Input And Legal-Style Prompts

For large prompts, legal docs, specs, or multi-thousand-token task statements,
use `record_long_input()` instead of stuffing the full prompt directly into a
single memory.

```python
report = await engram.record_long_input(
    task.task_run_id,
    huge_prompt_or_document,
    title="Vendor agreement review",
    max_chunk_tokens=700,
    extract_with_llm=True,
)

context = await engram.build_long_input_context(
    task.task_run_id,
    query="What are the termination obligations?",
    expected_terms=["termination", "notice"],
    max_tokens=4000,
)
```

`record_long_input()` does five things:

1. Stores the raw input as a source event.
2. Splits it into anchored chunks with character spans and `quote_hash`.
3. Stores each chunk as an artifact event.
4. Extracts or heuristically derives facts from chunks.
5. Creates a manifest checkpoint.

`build_long_input_context()` combines `trace_recall()` with the most relevant
source chunks and the manifest. For legal or exact-document answers, instruct the
agent to cite source chunks and prefer source text over distilled memories.

## What Works For 200k+ Context Tasks

Engram does not try to put everything into the prompt. It keeps durable state in
Postgres and builds a bounded context at every turn:

| Problem | Mechanism |
|---------|-----------|
| Critical facts disappear from vector ranking | `recall_critical()` and `trace_recall()` |
| Old facts conflict with corrections | `conflict_key`, `status=superseded`, `superseded_by` |
| Huge input cannot fit in every prompt | Chunk events + anchored memories + manifest |
| Multi-day work needs resumability | `agent_tasks`, `agent_events`, `task_checkpoints` |
| Broad prompt misses a required item | `expected_terms` and trace fields |
| Prompt budget is exceeded | Context builders trim deterministically and report what was trimmed |

For high-stakes document work, pair this with application-level citation checks:
before answering, call `build_long_input_context()`, require source chunk IDs in
the answer, and reject answers whose cited chunks are missing.

## Recommended Production Loop

```python
async def handle_turn(engram, task_id, user_message):
    task = await engram.get_task(task_id)

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

    # Run inline for simple apps; run in a worker for production.
    await engram.process_memory_jobs(limit=10)
    return response
```

For production, run `run_memory_worker()` in a background process and monitor
failed `memory_jobs`.

## Current Limits

- Search is scoped by `agent_id` and optionally `user_id`; tenant isolation is
  an application responsibility.
- `trace_recall()` reports missing expected terms only when callers provide
  `expected_terms`.
- Long-input source anchoring is text-based; PDF page numbers or OCR coordinates
  must be added by the caller in metadata.
- The schema is alpha. Back up data before running migrations.

