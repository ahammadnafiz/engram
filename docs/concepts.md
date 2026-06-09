# Core Concepts

This page explains the current Engram memory model.

## Memory Is Not One Thing

Engram separates durable memory into layers:

| Layer | Stored in | Optimized for |
|-------|-----------|---------------|
| Fact memory | `agent_memory` | Search, deterministic recall, conflict resolution |
| Graph memory | `memory_relations` | Multi-hop associations |
| Session memory | `agent_sessions` | Conversation grouping and rolling summaries |
| Task memory | `agent_tasks`, `agent_events`, `task_checkpoints`, `memory_jobs` | Long-running work and resumability |

For a short chatbot, fact memory may be enough. For a coding agent, legal review
agent, or personal assistant running for days, use task memory plus fact memory.

## Two-Column Fact Memory

A memory stores both a concise fact and optional source context.

```python
memory = await engram.add(
    content="User is allergic to shellfish",
    main_content="[USER]: I found out in Maine that shellfish makes me sick.",
    agent_id="assistant",
    user_id="user_123",
)
```

| Field | Embedded | Search role | Prompt role |
|-------|----------|-------------|-------------|
| `content` / `fact` | Yes | Semantic + keyword retrieval | Concise fact |
| `main_content` | No | Stored context, not vectorized | Evidence/context when needed |

This keeps embedding cost low while preserving the conversation or source text
that produced the fact.

## Memory Types

Each memory has a `memory_type`.

| Type | Examples |
|------|----------|
| `profile` | name, health, allergy, city, manager |
| `preference` | communication style, UI preference, meeting preference |
| `constraint` | "never revert user changes", deadlines, safety limits |
| `project` | codenames, owners, launch date, metrics |
| `task` | requirements, pending work, completed work |
| `decision` | approved approach, correction, changed target |
| `tool_result` | pytest output, load test result, API response |
| `semantic` | general durable fact |
| `episodic` | event or dated narrative |
| `procedural` | rule or process |

Use typed retrieval when the prompt has a narrow purpose:

```python
constraints = await engram.search(
    "repo rules",
    agent_id="codex",
    user_id="nafiz",
    memory_types=["constraint", "decision"],
)
```

## Memory Policy

`MemoryPolicy` applies domain rules before storage:

1. Infer a more specific `memory_type` from text or metadata.
2. Decide whether the memory is critical.
3. Assign a deterministic `critical_slot`.
4. Build a `conflict_key` scoped by agent and user.

```python
async with Engram(memory_policy="coding_agent") as engram:
    await engram.add(
        "Repo constraint: do not edit generated migrations manually",
        agent_id="codex",
        user_id="nafiz",
    )
```

Built-in policies:

| Policy | Focus |
|--------|-------|
| `default` | Personal facts, preferences, projects, requirements |
| `legal` | citations, source chunks, clauses, deadlines, audit logs |
| `coding_agent` | repo constraints, implementation decisions, tool results |

## Critical Memory

Critical facts should not depend only on vector search. Engram stores policy
metadata so they can be recalled deterministically:

```json
{
  "critical": true,
  "critical_slot": "constraint:repo",
  "conflict_key": "codex:nafiz:constraint:repo",
  "status": "active",
  "version": 1
}
```

Use `recall_critical()` to retrieve them directly:

```python
critical = await engram.recall_critical(
    agent_id="codex",
    user_id="nafiz",
    memory_types=["constraint", "preference", "profile"],
)
```

`trace_recall()` includes critical memories first, then vector-ranked memories.

## Conflict Resolution

When a new memory has the same `conflict_key` as an older active memory, the
older memory is not deleted. It is marked superseded:

```json
{
  "status": "superseded",
  "superseded_by": "mem_new",
  "superseded_at": "2026-06-09T..."
}
```

Normal search hides superseded memories. Trace APIs can still report them.

This matters for corrections:

- "I am allergic to cashews."
- Later: "Correction: I am not allergic to cashews."

Both records can exist for auditability, but active recall should use the latest
slot winner.

## Hybrid Search

`search()` combines:

| Signal | Source |
|--------|--------|
| Semantic | pgvector cosine similarity on `embedding` |
| Keyword | PostgreSQL full-text search on `fact_tsv` |
| Decay | recency/access score |
| Importance | explicit importance and reinforcement |

Search supports metadata and type filters:

```python
results = await engram.search(
    "black friday latency target",
    agent_id="codex",
    user_id="nafiz",
    metadata_filter={"project": "atlas_checkout"},
    memory_types=["project", "constraint", "tool_result"],
)
```

Use `deep_search()` for broad or multi-part prompts. It expands the query with
the configured LLM, searches each variant, and dedupes by memory ID.

## Recall Observability

`trace_recall()` builds a prompt block and explains what happened.

```python
trace = await engram.trace_recall(
    "Can we launch today?",
    agent_id="assistant",
    user_id="user_123",
    expected_terms=["rollback owner", "error rate"],
    max_tokens=1000,
)
```

Important fields:

| Field | Debug question |
|-------|----------------|
| `critical_memory_ids` | Was it pinned as critical? |
| `search_memory_ids` | Did vector/keyword retrieval find it? |
| `ranked_memory_ids` | Was it eligible for the context? |
| `kept_memory_ids` | Did it make the final prompt? |
| `trimmed_memory_ids` | Was it cut by token budget? |
| `superseded_memory_ids` | Was an old fact hidden by correction rules? |
| `missing_expected_terms` | Did required terms fail to appear? |

## Graph Relationships

Graph relations connect memories for multi-hop context:

```python
await engram.relate(
    source_id=task.memory_id,
    target_id=constraint.memory_id,
    relation_type="supports",
    weight=0.8,
)

graph = await engram.traverse_many(
    [task.memory_id, constraint.memory_id],
    max_depth=2,
    direction="any",
)

block = engram.render_graph_context(graph, max_tokens=800)
```

Graph traversal is useful when one retrieved fact should bring along related
decisions, tool results, or constraints.

## Task Memory

Task memory is for long-running work:

| Model | Meaning |
|-------|---------|
| `TaskRun` | Goal, status, owner, outcome |
| `AgentEvent` | Raw ledger entry for user, assistant, tool, artifact, decision |
| `TaskCheckpoint` | Compact state snapshot |
| `MemoryJob` | Durable background derivation work |

Typical loop:

```python
task = await engram.start_task("Refactor the memory framework", "codex")
context = await engram.build_context(task.task_run_id, query="resume")
response = await call_llm(context.text)
await engram.record_turn(task.task_run_id, user_message, response)
await engram.process_memory_jobs(limit=10)
```

For production, run `run_memory_worker()` in a worker process.

## Long Input

Long prompts and documents should be chunked and anchored:

```python
report = await engram.record_long_input(
    task.task_run_id,
    text=large_prompt,
    title="Legal review packet",
)

context = await engram.build_long_input_context(
    task.task_run_id,
    query="termination obligations",
    expected_terms=["termination", "notice"],
)
```

Each chunk records:

- chunk ID
- heading/kind
- character span
- quote hash
- source event ID

Use source chunks for legal, financial, compliance, or exact-document answers.
Use distilled memory for speed and continuity.

## When To Use Which API

| Need | API |
|------|-----|
| Store one fact | `add()` |
| Store facts from a conversation | `add_conversation()` |
| Retrieve relevant facts | `search()` |
| Retrieve broad/multi-part facts | `deep_search()` |
| Build debuggable prompt memory | `trace_recall()` |
| Build compact prompt block | `get_context_block()` |
| Start resumable agent work | `start_task()` |
| Record raw turns/tools | `record_turn()` / `record_event()` |
| Build task-resume context | `build_context()` |
| Ingest huge prompt/document | `record_long_input()` |
| Build source-anchored answer context | `build_long_input_context()` |

