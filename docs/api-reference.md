# API Reference

This page documents the public API exported by `engram.Engram` in the current
alpha release. The examples are syntax-checked by the test suite.

## Client Lifecycle

```text
Engram(
    settings: EngramSettings | None = None,
    *,
    database_url: str | None = None,
    openai_api_key: str | None = None,
    memory_policy: str | MemoryPolicy | None = None,
)
```

`memory_policy` accepts `"default"`, `"legal"`, `"coding_agent"`, a custom
`MemoryPolicy`, or `None` for the default policy.

```python
from engram import Engram

async with Engram(memory_policy="coding_agent") as engram:
    health = await engram.health_check()
    print(health["status"])
```

Manual lifecycle is useful in web application startup and shutdown hooks.

```python
engram = Engram(database_url="postgresql://engram:engram_secret@localhost:5432/engram")
await engram.connect()
try:
    results = await engram.search("repo constraints", "codex")
finally:
    await engram.close()
```

## Memory API

### `add()`

```text
await engram.add(
    content,
    agent_id,
    *,
    main_content=None,
    user_id=None,
    session_id=None,
    memory_type="semantic",
    metadata=None,
) -> Memory
```

`content` is the concise fact that gets embedded. `main_content` stores source
conversation or document context and is not embedded.

```python
memory = await engram.add(
    "User is allergic to shellfish",
    "assistant",
    user_id="user_123",
    main_content="[USER]: Shellfish makes me sick.\n[AI]: I will remember that.",
)

print(memory.memory_id)
print(memory.memory_type)
print(memory.metadata)
```

The active policy can infer a more specific `memory_type`, mark critical facts,
and assign a `conflict_key`.

### `add_batch()`

```text
await engram.add_batch(memories: list[dict[str, Any]]) -> list[Memory]
```

Each dictionary can contain `content`, `agent_id`, `main_content`, `user_id`,
`session_id`, `memory_type`, and `metadata`.

```python
memories = await engram.add_batch(
    [
        {"content": "User prefers concise answers", "agent_id": "assistant"},
        {
            "content": "Project Atlas launch date is July 18",
            "agent_id": "assistant",
            "memory_type": "project",
            "metadata": {"project": "atlas"},
        },
    ]
)
```

### `add_conversation()`

```text
await engram.add_conversation(
    user_message,
    assistant_response,
    agent_id,
    *,
    user_id=None,
    session_id=None,
    conversation_history=None,
    conversation_summary=None,
    metadata=None,
    search_limit=10,
    update_summary=True,
) -> list[Memory]
```

This method requires a configured LLM provider. It extracts facts, compares them
with existing memories, stores add/update/delete/noop decisions, and writes the
raw exchange to `main_content`.

```python
memories = await engram.add_conversation(
    user_message="I'm Sarah. I am allergic to shellfish.",
    assistant_response="I will avoid shellfish recommendations.",
    agent_id="assistant",
    user_id="sarah",
)

for memory in memories:
    print(memory.memory_type, memory.content)
```

### Read And Write Helpers

```text
await engram.get(memory_id) -> Memory
await engram.update(memory_id, *, content=None, importance=None, metadata=None) -> Memory
await engram.reinforce(memory_id, importance_boost=0.1) -> Memory
await engram.forget(memory_id) -> bool
await engram.purge(agent_id, user_id=None) -> int
await engram.list_recent(agent_id, user_id=None, limit=10) -> list[Memory]
await engram.get_memories(agent_id, *, user_id=None, session_id=None, metadata_filter=None, memory_types=None, limit=200) -> list[Memory]
```

`get()` updates access metadata. `get_memories()` is a plain filtered read and
does not rank by relevance.

```python
memory = await engram.update(
    memory.memory_id,
    content="User is allergic to shellfish and cashews",
    metadata={"source": "manual_correction"},
)

await engram.reinforce(memory.memory_id, importance_boost=0.2)
recent = await engram.list_recent("assistant", user_id="sarah", limit=5)
deleted = await engram.forget(memory.memory_id)
```

## Search And Recall

### `search()`

```text
await engram.search(
    query,
    agent_id,
    *,
    user_id=None,
    limit=10,
    min_score=0.0,
    metadata_filter=None,
    memory_types=None,
    mode="hybrid",
    rerank=False,
) -> list[SearchResult]
```

Modes are `"hybrid"`, `"semantic"`, and `"keyword"`. Hybrid search combines
pgvector similarity, PostgreSQL full-text search, decay, and importance. Normal
search hides memories where `metadata.status` is `"superseded"`.

```python
results = await engram.search(
    "rollback owner for Atlas checkout",
    "codex",
    user_id="nafiz",
    memory_types=["project", "constraint", "tool_result"],
    metadata_filter={"project": "atlas_checkout"},
    limit=5,
)

for result in results:
    print(result.score, result.memory.content)
```

Set `rerank=True` to reorder overfetched candidates with the configured local
cross-encoder. This requires the optional `sentence-transformers` dependency.

### `deep_search()`

```text
await engram.deep_search(
    query,
    agent_id,
    *,
    user_id=None,
    limit=10,
    min_score=0.0,
    metadata_filter=None,
    memory_types=None,
    mode="hybrid",
    n_queries=4,
    rerank=False,
) -> list[SearchResult]
```

`deep_search()` expands broad queries with the configured LLM, runs each variant,
and fuses rankings with Reciprocal Rank Fusion. Without an LLM provider, it
falls back to one `search()` call.

### `recall_critical()`

```text
await engram.recall_critical(
    agent_id,
    *,
    user_id=None,
    limit=50,
    memory_types=None,
) -> list[Memory]
```

This reads active critical memories by metadata. It does not use vector rank.

```python
critical = await engram.recall_critical(
    "codex",
    user_id="nafiz",
    memory_types=["constraint", "decision", "preference"],
)
```

### `trace_recall()`

```text
await engram.trace_recall(
    query,
    agent_id,
    *,
    user_id=None,
    limit=20,
    min_score=0.0,
    max_tokens=2000,
    expected_terms=None,
    use_deep_search=True,
    memory_types=None,
    token_counter=None,
) -> RecallTrace
```

`trace_recall()` builds a prompt-ready memory block and tells you what happened:
critical hits, search hits, ranked IDs, kept IDs, trimmed IDs, superseded IDs,
missing expected terms, notes, and metadata counts.

```python
trace = await engram.trace_recall(
    "resume repository work",
    "codex",
    user_id="nafiz",
    expected_terms=["never revert", "pytest"],
    max_tokens=1200,
)

print(trace.context)
print(trace.missing_expected_terms)
print(trace.trimmed_memory_ids)
```

### `get_context_block()`

```text
await engram.get_context_block(
    query,
    agent_id,
    *,
    user_id=None,
    session_id=None,
    limit=10,
    min_score=0.0,
    max_tokens=None,
    header="## Relevant memories",
    token_counter=None,
    memory_types=None,
    group_by_type=False,
    rerank=False,
) -> str
```

Use this when you want a compact memory block for a system or context prompt.
If `session_id` has a rolling summary, the summary is prepended.

```python
block = await engram.get_context_block(
    query=user_message,
    agent_id="assistant",
    user_id="sarah",
    max_tokens=800,
    group_by_type=True,
)
```

## Evidence APIs

These methods support aggregation and evidence-reading workloads where one
ranked memory is not enough.

```text
await engram.search_evidence_set(
    query,
    agent_id,
    *,
    user_id=None,
    limit=10,
    candidate_limit=None,
    min_score=0.0,
    metadata_filter=None,
    memory_types=None,
    mode="hybrid",
    use_deep_search=True,
    rerank=True,
    diversify_metadata_key="original_session_id",
    max_per_group=3,
    preferred_role=None,
    role_metadata_key="turn_role",
) -> list[SearchResult]
```

`search_evidence_set()` overfetches, optionally deep-searches and reranks, then
diversifies by session or metadata group.

```text
await engram.get_neighboring_context_block(
    results,
    agent_id,
    *,
    user_id=None,
    before=2,
    after=2,
    include_session_start=False,
    max_tokens=None,
    token_counter=None,
    memory_types=None,
    session_metadata_key="original_session_id",
    turn_metadata_key="turn_index",
    date_metadata_key="haystack_date",
    role_metadata_key="turn_role",
    group_limit=200,
    priority_window_results=3,
    prior_user_turns=0,
    context_order="chronological",
) -> tuple[str, list[dict[str, Any]]]
```

`get_neighboring_context_block()` expands retrieved turn memories with nearby
turns from the same session or metadata group.

```python
hits = await engram.search_evidence_set(
    "Where did the user buy the replacement charger?",
    "assistant",
    user_id="sarah",
    limit=8,
    preferred_role="user",
)

context, sources = await engram.get_neighboring_context_block(
    hits,
    "assistant",
    user_id="sarah",
    before=2,
    after=1,
    max_tokens=3000,
)
```

```text
await engram.answer_from_evidence(
    *,
    question,
    context,
    question_date=None,
    max_tokens=256,
    reading="direct",
) -> str
```

`answer_from_evidence()` uses the configured LLM to answer from a supplied
context. Use `reading="con"` for aggregation questions that need an evidence
ledger before the final answer.

## Task Memory

### Task Lifecycle

```text
await engram.start_task(goal, agent_id, *, user_id=None, session_id=None, metadata=None) -> TaskRun
await engram.get_task(task_run_id, *, include_deleted=False) -> TaskRun
await engram.list_tasks(*, agent_id=None, user_id=None, status=None, limit=100, include_deleted=False) -> list[TaskRun]
await engram.pause_task(task_run_id, *, outcome=None) -> TaskRun
await engram.complete_task(task_run_id, *, outcome=None) -> TaskRun
await engram.fail_task(task_run_id, *, outcome=None) -> TaskRun
await engram.cancel_task(task_run_id, *, outcome=None) -> TaskRun
await engram.soft_delete_task(task_run_id) -> TaskRun
```

```python
task = await engram.start_task(
    "Ship the memory docs",
    "codex",
    user_id="nafiz",
    metadata={"repo": "engram"},
)

await engram.pause_task(task.task_run_id, outcome="Waiting for review")
```

### Events And Turns

```text
await engram.record_event(
    *,
    agent_id,
    role,
    event_type,
    content="",
    task_run_id=None,
    session_id=None,
    user_id=None,
    payload=None,
    metadata=None,
) -> AgentEvent

await engram.list_events(*, task_run_id=None, session_id=None, agent_id=None, limit=100, include_deleted=False) -> list[AgentEvent]
await engram.redact_event(event_id) -> AgentEvent
```

Roles are `user`, `assistant`, `agent`, `tool`, and `system`. Event types are
`user_message`, `assistant_message`, `tool_call`, `tool_result`,
`agent_action`, `decision`, `observation`, `artifact`, `error`, and
`system_note`.

```text
await engram.record_turn(
    task_run_id,
    user_message,
    assistant_response,
    *,
    agent_id=None,
    user_id=None,
    session_id=None,
    tool_calls=None,
    artifacts=None,
    metadata=None,
    enqueue_processing=True,
) -> list[AgentEvent]
```

`record_turn()` writes user, assistant, tool, and artifact events in one
transaction and queues a `turn_ingest` job by default.

```python
events = await engram.record_turn(
    task.task_run_id,
    user_message="Update the API reference.",
    assistant_response="I rewrote it from the current client signatures.",
    tool_calls=[{"name": "pytest", "result": "docs examples passed"}],
)
```

### Checkpoints, Context, And Jobs

```text
await engram.create_checkpoint(
    task_run_id,
    summary,
    *,
    completed_steps=None,
    pending_steps=None,
    decisions=None,
    blockers=None,
    artifacts=None,
    source_event_ids=None,
    metadata=None,
) -> TaskCheckpoint

await engram.build_context(
    task_run_id,
    *,
    query="",
    max_tokens=200000,
    token_counter=None,
    recent_event_limit=40,
    memory_limit=25,
    checkpoint_limit=3,
    include_graph=True,
) -> ContextBuildResult

await engram.process_memory_jobs(*, limit=10) -> list[MemoryJob]
await engram.run_memory_worker(*, batch_size=10, interval_seconds=1.0, stop_event=None, max_iterations=None) -> int
```

```python
await engram.create_checkpoint(
    task.task_run_id,
    "API docs now match the Engram client signatures.",
    completed_steps=["Read client.py", "Rewrote API reference"],
    pending_steps=["Run mkdocs build"],
)

context = await engram.build_context(task.task_run_id, query="resume docs work")
jobs = await engram.process_memory_jobs(limit=10)
```

## Long Input

```text
await engram.record_long_input(
    task_run_id,
    text,
    *,
    title=None,
    agent_id=None,
    user_id=None,
    session_id=None,
    metadata=None,
    max_chunk_tokens=700,
    extract_with_llm=True,
    max_facts_per_chunk=6,
) -> LongInputIngestionReport

await engram.build_long_input_context(
    task_run_id,
    *,
    query,
    max_tokens=4000,
    source_chunk_limit=6,
    expected_terms=None,
    token_counter=None,
) -> LongInputContextResult
```

```python
report = await engram.record_long_input(
    task.task_run_id,
    text=contract_text,
    title="Vendor MSA",
    max_chunk_tokens=700,
)

context = await engram.build_long_input_context(
    task.task_run_id,
    query="termination notice requirements",
    expected_terms=["termination", "notice"],
    max_tokens=4000,
)

print(report.manifest)
print(context.trace["missing_expected_terms"])
```

## Graph API

```text
await engram.relate(source_id, target_id, relation_type="related_to", weight=1.0, metadata=None) -> None
await engram.traverse(start_memory_id, max_depth=3, direction="outbound", relation_types=None, min_weight=0.0, limit=50) -> list[TraversalResult]
await engram.traverse_many(start_memory_ids, *, max_depth=2, direction="any", relation_types=None, min_weight=0.0, limit_per_seed=25, total_limit=100, skip_missing=True) -> list[TraversalResult]
engram.render_graph_context(results, *, max_tokens=None, token_counter=None, include_paths=False, header="## Related memory graph") -> str
```

```python
await engram.relate(
    source_id=requirement.memory_id,
    target_id=decision.memory_id,
    relation_type="supports",
    weight=0.8,
)

graph = await engram.traverse_many(
    [requirement.memory_id, decision.memory_id],
    max_depth=2,
    direction="any",
)

graph_context = engram.render_graph_context(graph, max_tokens=800)
```

## Sessions

```text
async with engram.session(agent_id, user_id=None, metadata=None) -> Session
```

```python
async with engram.session("assistant", user_id="sarah") as session:
    await engram.add(
        "User asked about deployment",
        "assistant",
        user_id="sarah",
        session_id=session.session_id,
    )
```

`add_conversation(..., update_summary=True)` can maintain the session rolling
summary when an LLM provider is configured.

## Models

Key models exported from `engram`:

| Model | Purpose |
|-------|---------|
| `Memory` | Stored fact, source context, embedding, type, importance, metadata |
| `SearchResult` | Memory plus ranking scores |
| `RecallTrace` | Retrieval observability record |
| `TaskRun` | Durable unit of agent work |
| `AgentEvent` | Raw event ledger entry |
| `TaskCheckpoint` | Compact task state snapshot |
| `MemoryJob` | Durable background derivation job |
| `ContextBuildResult` | Rendered task context plus sections |
| `LongInputChunk` | Source-anchored chunk |
| `LongInputIngestionReport` | Result of `record_long_input()` |
| `LongInputContextResult` | Prompt context for long-input answers |
| `TraversalResult` | Memory graph traversal hit |
| `Session` | Agent/user conversation session |

## Services

`EmbeddingService.from_settings(settings)` creates the configured embedding
provider. `LLMService.from_settings(settings)` returns `None` when no
`llm_provider` is configured.

```python
from engram import EmbeddingService, LLMService, get_settings

settings = get_settings()
embedding = EmbeddingService.from_settings(settings)
vector = await embedding.embed("hello")

llm = LLMService.from_settings(settings)
if llm is not None:
    facts = await llm.extract_facts("I like tea.", "I will remember that.")
```

## Health

```text
await engram.health_check() -> dict[str, Any]
```

```python
health = await engram.health_check()
print(health["status"])
print(health["components"])
```
