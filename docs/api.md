# API Reference

This page documents the public API exposed by `Engram`.

## Client Lifecycle

```python
from engram import Engram

async with Engram(
    settings=None,
    embedding_service=None,
    llm_service=None,
    memory_policy="default",
) as engram:
    ...
```

`memory_policy` may be `"default"`, `"legal"`, `"coding_agent"`, or a custom
`MemoryPolicy`.

Manual lifecycle:

```python
engram = Engram(memory_policy="coding_agent")
await engram.connect()
try:
    ...
finally:
    await engram.close()
```

## Memory Operations

### `add()`

```python
memory = await engram.add(
    content: str,
    agent_id: str,
    *,
    main_content: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    memory_type: MemoryType = "semantic",
    metadata: dict | None = None,
) -> Memory
```

Stores one fact. `content` is embedded and searchable. `main_content` is stored
but not embedded. The active `MemoryPolicy` may override `memory_type` and add
critical/conflict metadata.

```python
memory = await engram.add(
    "User is allergic to shellfish",
    "assistant",
    user_id="user_123",
    main_content="[USER]: I found out in Maine that shellfish makes me sick.",
)
```

### `add_batch()`

```python
memories = await engram.add_batch(
    memories: list[dict[str, Any]],
) -> list[Memory]
```

Each dict accepts `content`, `agent_id`, `main_content`, `user_id`,
`session_id`, `memory_type`, and `metadata`.

### `add_conversation()`

```python
memories = await engram.add_conversation(
    user_message: str,
    assistant_response: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    conversation_summary: str | None = None,
    metadata: dict | None = None,
    search_limit: int = 10,
    update_summary: bool = True,
) -> list[Memory]
```

Requires an LLM provider. Extracts atomic facts, compares them with existing
memories, applies add/update/delete/noop decisions, stores the exchange in
`main_content`, and optionally updates a session summary.

### `get()`, `update()`, `reinforce()`, `forget()`, `purge()`

```python
memory = await engram.get(memory_id)

memory = await engram.update(
    memory_id,
    content="Corrected fact",
    importance=0.9,
    metadata={"source": "manual_correction"},
)

memory = await engram.reinforce(memory_id, importance_boost=0.1)
deleted = await engram.forget(memory_id)
count = await engram.purge(agent_id="assistant", user_id="user_123")
```

`update()` re-embeds when `content` changes. `reinforce()` raises importance and
updates access data. `purge()` deletes all memories for an agent or agent/user
scope.

### `list_recent()`

```python
memories = await engram.list_recent(
    agent_id: str,
    user_id: str | None = None,
    limit: int = 10,
) -> list[Memory]
```

## Search And Recall

### `search()`

```python
results = await engram.search(
    query: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    metadata_filter: dict | None = None,
    memory_types: list[MemoryType] | None = None,
) -> list[SearchResult]
```

Hybrid search combines vector similarity, keyword search, time decay, and
importance. Superseded memories are hidden from normal search.

### `deep_search()`

```python
results = await engram.deep_search(
    query: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    metadata_filter: dict | None = None,
    memory_types: list[MemoryType] | None = None,
    n_queries: int = 4,
) -> list[SearchResult]
```

Uses the configured LLM to expand a broad query into variants, runs concurrent
searches, and dedupes by `memory_id`. Falls back to `search()` when no LLM is
configured.

### `recall_critical()`

```python
memories = await engram.recall_critical(
    agent_id: str,
    *,
    user_id: str | None = None,
    limit: int = 50,
    memory_types: list[MemoryType] | None = None,
) -> list[Memory]
```

Retrieves active critical memories directly from metadata. This is deterministic
and does not rely on vector rank.

### `trace_recall()`

```python
trace = await engram.trace_recall(
    query: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    limit: int = 20,
    min_score: float = 0.0,
    max_tokens: int = 2000,
    expected_terms: list[str] | None = None,
    use_deep_search: bool = True,
    memory_types: list[MemoryType] | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> RecallTrace
```

Builds a prompt-ready memory block and returns trace fields:

- `critical_memory_ids`
- `search_memory_ids`
- `ranked_memory_ids`
- `kept_memory_ids`
- `trimmed_memory_ids`
- `superseded_memory_ids`
- `missing_expected_terms`
- `notes`

### `get_context_block()`

```python
block = await engram.get_context_block(
    query: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    max_tokens: int | None = None,
    header: str = "## Relevant memories",
    token_counter: Callable[[str], int] | None = None,
    memory_types: list[MemoryType] | None = None,
    group_by_type: bool = False,
) -> str
```

Returns an injection-ready block for prompts. If `session_id` has a stored
rolling summary, it is prepended.

## Long-Running Task Memory

### `start_task()`, `get_task()`, `list_tasks()`

```python
task = await engram.start_task(
    goal: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> TaskRun

task = await engram.get_task(task_run_id, include_deleted=False)

tasks = await engram.list_tasks(
    agent_id: str | None = None,
    user_id: str | None = None,
    status: str | list[str] | None = None,
    limit: int = 100,
    include_deleted: bool = False,
) -> list[TaskRun]
```

### Task status

```python
await engram.pause_task(task_run_id, outcome="Waiting on review")
await engram.complete_task(task_run_id, outcome="Released v0.3.0a1")
await engram.fail_task(task_run_id, outcome="Blocked by missing credentials")
await engram.cancel_task(task_run_id, outcome="User cancelled")
await engram.soft_delete_task(task_run_id)
```

### Events

```python
event = await engram.record_event(
    agent_id="codex",
    role="tool",
    event_type="tool_result",
    content="pytest tests/unit -q: 190 passed",
    task_run_id=task.task_run_id,
    user_id="nafiz",
    payload={"command": "pytest tests/unit -q", "exit_code": 0},
)

events = await engram.list_events(task_run_id=task.task_run_id, limit=100)
redacted = await engram.redact_event(event.event_id)
```

Roles: `user`, `assistant`, `agent`, `tool`, `system`.

Event types: `user_message`, `assistant_message`, `tool_call`, `tool_result`,
`agent_action`, `decision`, `observation`, `artifact`, `error`, `system_note`.

### `record_turn()`

```python
events = await engram.record_turn(
    task_run_id: str,
    user_message: str,
    assistant_response: str,
    *,
    agent_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tool_calls: list[dict] | None = None,
    artifacts: list[dict] | None = None,
    metadata: dict | None = None,
    enqueue_processing: bool = True,
) -> list[AgentEvent]
```

Records user and assistant events, optional tool/artifact events, and queues a
`turn_ingest` job by default.

### `create_checkpoint()`

```python
checkpoint = await engram.create_checkpoint(
    task_run_id: str,
    summary: str,
    *,
    completed_steps: list[str] | None = None,
    pending_steps: list[str] | None = None,
    decisions: list[str] | None = None,
    blockers: list[str] | None = None,
    artifacts: list[dict] | None = None,
    source_event_ids: list[str] | None = None,
    metadata: dict | None = None,
) -> TaskCheckpoint
```

### `build_context()`

```python
context = await engram.build_context(
    task_run_id: str,
    *,
    query: str = "",
    max_tokens: int = 200000,
    token_counter: Callable[[str], int] | None = None,
    recent_event_limit: int = 40,
    memory_limit: int = 25,
    checkpoint_limit: int = 3,
    include_graph: bool = True,
) -> ContextBuildResult
```

Builds deterministic resume context from task metadata, recent events,
checkpoints, memory search, and optional graph expansion.

### Background jobs

```python
jobs = await engram.process_memory_jobs(limit=10)

count = await engram.run_memory_worker(
    batch_size=10,
    interval_seconds=1.0,
    stop_event=None,
    max_iterations=None,
)
```

## Long Input APIs

### `record_long_input()`

```python
report = await engram.record_long_input(
    task_run_id: str,
    text: str,
    *,
    title: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict | None = None,
    max_chunk_tokens: int = 700,
    extract_with_llm: bool = True,
    max_facts_per_chunk: int = 6,
) -> LongInputIngestionReport
```

Stores the raw input as a source event, splits it into anchored chunks, stores
chunk events, extracts anchored memories, and creates a manifest checkpoint.

### `build_long_input_context()`

```python
context = await engram.build_long_input_context(
    task_run_id: str,
    *,
    query: str,
    max_tokens: int = 4000,
    source_chunk_limit: int = 6,
    expected_terms: list[str] | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> LongInputContextResult
```

Combines recall trace, relevant source chunks, and the long-input manifest.

## Graph Operations

```python
await engram.relate(
    source_id: str,
    target_id: str,
    relation_type: RelationType = "related_to",
    weight: float = 1.0,
    metadata: dict | None = None,
)

results = await engram.traverse(
    start_memory_id: str,
    max_depth: int = 3,
    direction: str = "outbound",
    relation_types: list[RelationType] | None = None,
    min_weight: float = 0.0,
    limit: int = 50,
) -> list[TraversalResult]

results = await engram.traverse_many(
    start_memory_ids: list[str],
    *,
    max_depth: int = 2,
    direction: str = "any",
    relation_types: list[RelationType] | None = None,
    min_weight: float = 0.0,
    limit_per_seed: int = 25,
    total_limit: int = 100,
    skip_missing: bool = True,
) -> list[TraversalResult]

block = engram.render_graph_context(results, max_tokens=800)
```

## Sessions

```python
async with engram.session(
    agent_id="assistant",
    user_id="user_123",
    metadata={"channel": "chat"},
) as session:
    await engram.add(
        "User asked about deployment",
        agent_id="assistant",
        user_id="user_123",
        session_id=session.session_id,
    )
```

Sessions now include optional rolling summaries used by `get_context_block()`.

## Models

### `Memory`

```python
class Memory(BaseModel):
    memory_id: str
    agent_id: str
    user_id: str | None
    session_id: str | None
    content: str
    fact: str | None
    main_content: str | None
    memory_type: MemoryType
    embedding: list[float] | None
    importance: float
    access_count: int
    created_at: datetime
    last_accessed_at: datetime
    metadata: dict
```

### `RecallTrace`

```python
class RecallTrace(BaseModel):
    query: str
    agent_id: str
    user_id: str | None
    critical_memory_ids: list[str]
    search_memory_ids: list[str]
    ranked_memory_ids: list[str]
    kept_memory_ids: list[str]
    trimmed_memory_ids: list[str]
    superseded_memory_ids: list[str]
    missing_expected_terms: list[str]
    context: str
    notes: list[str]
    metadata: dict
```

### Task and long-input models

- `TaskRun`
- `AgentEvent`
- `TaskCheckpoint`
- `MemoryJob`
- `ContextBuildResult`
- `LongInputChunk`
- `LongInputIngestionReport`
- `LongInputContextResult`

See `src/engram/task/models.py` for exact fields.

## Services

### `EmbeddingService`

Providers: `openai`, `sentence-transformers`, `cohere`, `ollama`,
`huggingface`.

```python
embedding = EmbeddingService.from_settings()
vector = await embedding.embed("hello")
vectors = await embedding.embed_batch(["a", "b"])
```

### `LLMService`

Providers: `openai`, `anthropic`, `ollama`, `groq`, `litellm`.

```python
llm = LLMService.from_settings()
facts = await llm.extract_facts(user_message, assistant_response)
summary = await llm.summarize(long_text)
expanded = await llm.expand_query("broad query", n_queries=4)
```

## Health

```python
health = await engram.health_check()
print(health["status"])
print(health["components"])
```

