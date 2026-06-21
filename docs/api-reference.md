# Engram Client API Reference

This document provides a comprehensive, OSS-grade reference for the `engram.Engram` asynchronous client. 

> [!NOTE]
> All methods (except context managers) are asynchronous and must be awaited. The code snippets assume you have instantiated an `engram` client.

---

## 1. Client Lifecycle

### `Engram()`

```py
Engram(
    settings: EngramSettings | None = None,
    *,
    database_url: str | None = None,
    openai_api_key: str | None = None,
    memory_policy: str | MemoryPolicy | None = None,
)
```
Initializes the Engram client. Providers are lazy-loaded upon `connect()`. Use `memory_policy` to configure how facts are typed and slotted (valid strings: `"default"`, `"legal"`, `"coding_agent"`).

### `connect()`

```py
await engram.connect() -> None
```
Establishes the database connection pool and initializes configured providers.

### `close()`

```py
await engram.close() -> None
```
Safely tears down the database connection pool and releases provider resources.

---

## 2. Memory Operations

### `add()`

```py
await engram.add(
    content: str,
    agent_id: str,
    *,
    main_content: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    memory_type: str = "semantic",
    metadata: dict | None = None,
) -> Memory
```
Stores a single atomic memory fact. The `content` is the exact fact to embed and index (mapped to `.fact` internally). `main_content` is used for storing the raw context the fact was extracted from, but it is not embedded for search.

### `add_batch()`

```py
await engram.add_batch(memories: list[dict[str, Any]]) -> list[Memory]
```
Efficiently adds multiple memories in a single transaction with batch-vectorization. The `memories` list should contain dictionaries with `content`, `agent_id`, and other optional arguments corresponding to `add()`.

### `add_conversation()`

```py
await engram.add_conversation(
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
    extract_assistant_response: bool = False,
) -> ConversationResult
```
Processes a conversation turn to intelligently extract, compare, and store new memories using the configured LLM provider. Set `extract_assistant_response=True` only if you want the assistant's own statements converted into stored memories.

`ConversationResult` is **list-compatible** — iterating, `len()`-ing, indexing, or truth-testing it yields the written memories, exactly like the previous `list[Memory]` return, so existing callers need no change. Its `.decisions` field additionally exposes a `FactDecision(fact, operation, applied, reason, memory_id, target_id)` for **every** extracted fact, including those that resolved to `NOOP` or could not be applied. This makes a skipped update *visible and debuggable* (e.g. `operation="NOOP", applied=False, reason="Duplicate of …"`) instead of silently absent from the returned list. If you write on this path, inspect `.decisions` to detect a fact that was extracted but not stored.

> **Use as the sole writer to a memory space.** `add_conversation()` searches the agent's existing memories to decide ADD/UPDATE/DELETE. If you also store raw turns in the same space with `add_batch()`, the freshly-extracted fact is found already present verbatim in its own raw row, so the decision step judges it identical and NOOPs it — supersession never fires. Either let `add_conversation()` own a memory space, or keep the two writers in separate namespaces. (The `examples/chatbot.py` example deliberately uses `add_batch()` only for this reason.)

---

## 3. Memory Helpers & Lineage

### `get()`

```py
await engram.get(memory_id: str) -> Memory
```
Fetches a single memory by ID and updates its internal access count.

### `update()`

```py
await engram.update(
    memory_id: str, 
    *, 
    content: str | None = None, 
    importance: float | None = None, 
    metadata: dict | None = None
) -> Memory
```
Edits a memory directly in-place. Use with caution, as it bypasses version lineage tracking.

### `revise()`

```py
await engram.revise(
    memory_id: str, 
    *, 
    content: str | None = None, 
    importance: float | None = None, 
    metadata: dict | None = None, 
    reason: str | None = None
) -> Memory
```
Updates a memory by creating a new active head in the lineage and marking the old memory as superseded. Essential for user corrections.

### `get_current()`

```py
await engram.get_current(memory_id: str) -> Memory
```
Fetches the active head of a memory's lineage. If passed an old (superseded) ID, it resolves forward to the current version.

### `get_lineage()`

```py
await engram.get_lineage(memory_id: str) -> MemoryLineage
```
Retrieves the full version history (all updates and supersessions) of a specific memory.

### `explain_memory()`

```py
await engram.explain_memory(memory_id: str) -> MemoryExplanation
```
Returns an LLM-generated explanation of a memory's intent and context based on its graph relations.

### `reinforce()`

```py
await engram.reinforce(memory_id: str, importance_boost: float = 0.1) -> Memory
```
Increases the importance score of a memory, typically used when a recalled memory was proven highly useful.

### `forget()`

```py
await engram.forget(memory_id: str) -> bool
```
Soft-deletes a memory from the active search index. Returns `True` if successfully deleted.

### `purge()`

```py
await engram.purge(agent_id: str, user_id: str | None = None) -> int
```
Hard-deletes all memories belonging to an agent (and optionally a specific user). Returns the number of deleted records.

### `list_recent()`

```py
await engram.list_recent(agent_id: str, user_id: str | None = None, limit: int = 10) -> list[Memory]
```
Returns the most recently created or updated active memories.

### `get_history()`

```py
await engram.get_history(
    agent_id: str, 
    *, 
    user_id: str | None = None, 
    limit: int = 50, 
    include_superseded: bool = True, 
    memory_types: list[str] | None = None, 
    since: datetime | None = None, 
    until: datetime | None = None
) -> list[MemoryHistoryEvent]
```
Retrieves a chronological event timeline of added, revised, and superseded facts. Useful for rendering user-facing memory audit logs.

### `get_memories()`

```py
await engram.get_memories(
    agent_id: str, 
    *, 
    user_id: str | None = None, 
    session_id: str | None = None, 
    metadata_filter: dict | None = None, 
    memory_types: list[str] | None = None, 
    limit: int = 200
) -> list[Memory]
```
A plain filtered read operation without vector ranking. Ideal for bulk exporting or aggregating specific memory segments.

---

## 4. Search and Recall

### `search()`

```py
await engram.search(
    query: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    metadata_filter: dict | None = None,
    memory_types: list[str] | None = None,
    mode: str = "hybrid", 
    rerank: bool = False,
    include_superseded: bool = False,
) -> list[SearchResult]
```
Core hybrid search combining semantic similarity, full-text keywords, and importance/decay. Setting `rerank=True` activates local cross-encoder re-sorting (requires the `sentence-transformers` dependency).

### `deep_search()`

```py
await engram.deep_search(
    query: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    metadata_filter: dict | None = None,
    memory_types: list[str] | None = None,
    mode: str = "hybrid", 
    n_queries: int = 4,
    rerank: bool = False,
) -> list[SearchResult]
```
Expands broad queries by asking the LLM to generate `n_queries` permutations, runs them concurrently, and fuses the results using Reciprocal Rank Fusion (RRF).

### `recall()`

```py
await engram.recall(
    question: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    question_date: datetime | None = None,
    limit: int = 10,
    compose_answer: bool = True,
) -> RecallAnswer
```
The high-level autonomous operator. Classifies user intent (e.g. current vs historical), maps natural language timeframes (e.g., "yesterday"), and composes a fully grounded answer. Set `compose_answer=False` to skip the final LLM composition.

### `recall_critical()`

```py
await engram.recall_critical(
    agent_id: str, 
    *, 
    user_id: str | None = None, 
    limit: int = 50, 
    memory_types: list[str] | None = None
) -> list[Memory]
```
Reads active critical memories (like constraints, base logic, or profile info) directly via metadata, bypassing vector similarity rank.

### `trace_recall()`

```py
await engram.trace_recall(
    query: str, 
    agent_id: str, 
    *, 
    user_id: str | None = None, 
    limit: int = 20, 
    min_score: float = 0.0,
    max_tokens: int = 2000, 
    expected_terms: list[str] | None = None, 
    use_deep_search: bool = True, 
    memory_types: list[str] | None = None, 
    token_counter: Callable | None = None
) -> RecallTrace
```
A diagnostic utility. Returns a full observability trace of retrieval performance, including hit/miss metrics against `expected_terms` and truncation flags.

### `get_context_block()`

```py
await engram.get_context_block(
    query: str, 
    agent_id: str, 
    *, 
    user_id: str | None = None, 
    session_id: str | None = None, 
    limit: int = 10, 
    min_score: float = 0.0,
    max_tokens: int | None = None, 
    header: str = "## Relevant memories", 
    token_counter: Callable | None = None,
    memory_types: list[str] | None = None, 
    group_by_type: bool = False, 
    rerank: bool = False
) -> str
```
Formats retrieved search results directly into a budgeted, prompt-ready markdown block. If `session_id` is provided and has a rolling summary, the summary is prepended.

---

## 5. Graph API

### `relate()`

```py
await engram.relate(
    source_id: str, 
    target_id: str, 
    relation_type: str = "related_to", 
    weight: float = 1.0, 
    metadata: dict | None = None
) -> None
```
Creates a directional edge between two memories to form associative networks. Common relation types: `related_to`, `supports`, `causes`.

### `traverse()`

```py
await engram.traverse(
    start_memory_id: str, 
    max_depth: int = 3, 
    direction: str = "outbound", 
    relation_types: list[str] | None = None, 
    min_weight: float = 0.0, 
    limit: int = 50
) -> list[TraversalResult]
```
Performs a multi-hop traversal starting from a single memory using recursive Postgres CTEs.

### `traverse_many()`

```py
await engram.traverse_many(
    start_memory_ids: list[str], 
    *, 
    max_depth: int = 2, 
    direction: str = "any", 
    relation_types: list[str] | None = None, 
    min_weight: float = 0.0, 
    limit_per_seed: int = 25, 
    total_limit: int = 100, 
    skip_missing: bool = True
) -> list[TraversalResult]
```
Traverses the memory graph concurrently from multiple starting seeds and returns deduplicated, ranked results.

### `render_graph_context()`

```py
engram.render_graph_context(
    results: list[TraversalResult], 
    *, 
    max_tokens: int | None = None, 
    token_counter: Callable | None = None, 
    include_paths: bool = False, 
    header: str = "## Related memory graph"
) -> str
```
Utility to render traversal results into a markdown context block suitable for LLM injection.

---

## 6. Sessions

### `session()`

```py
async with engram.session(
    agent_id: str, 
    user_id: str | None = None, 
    metadata: dict | None = None
) -> Session
```
Context manager that creates a conversational session boundary. Memory additions scoped to the active session can automatically maintain a rolling conversational summary.

---

## 7. Task Memory

### `start_task()`

```py
await engram.start_task(
    goal: str, 
    agent_id: str, 
    *, 
    user_id: str | None = None, 
    session_id: str | None = None, 
    metadata: dict | None = None
) -> TaskRun
```
Initializes a durable event ledger for a long-running autonomous agent.

### `get_task()` / `list_tasks()`

```py
await engram.get_task(task_run_id: str, *, include_deleted: bool = False) -> TaskRun
await engram.list_tasks(*, agent_id: str | None = None, user_id: str | None = None, status: str | None = None, limit: int = 100, include_deleted: bool = False) -> list[TaskRun]
```
Query existing tasks based on their status or ownership.

### State Transitions

```py
await engram.pause_task(task_run_id: str, *, outcome: str | None = None) -> TaskRun
await engram.complete_task(task_run_id: str, *, outcome: str | None = None) -> TaskRun
await engram.fail_task(task_run_id: str, *, outcome: str | None = None) -> TaskRun
await engram.cancel_task(task_run_id: str, *, outcome: str | None = None) -> TaskRun
await engram.soft_delete_task(task_run_id: str) -> TaskRun
```
Methods to manage the lifecycle and terminal outcomes of a task.

### `record_event()`

```py
await engram.record_event(
    *, 
    agent_id: str, 
    role: str, 
    event_type: str, 
    content: str = "", 
    task_run_id: str | None = None, 
    session_id: str | None = None, 
    user_id: str | None = None, 
    payload: dict | None = None, 
    metadata: dict | None = None
) -> AgentEvent
```
Logs a single, immutable action, observation, or tool call to the event ledger.

### `record_turn()`

```py
await engram.record_turn(
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
    enqueue_processing: bool = True
) -> list[AgentEvent]
```
Writes a complete conversation turn (user, assistant, and tools) in one transaction and optionally queues a background analysis job for derivation.

### `search_events()`

```py
await engram.search_events(
    query: str, 
    *, 
    agent_id: str | None = None, 
    task_run_id: str | None = None, 
    session_id: str | None = None, 
    user_id: str | None = None, 
    event_types: list[str] | None = None, 
    roles: list[str] | None = None, 
    since: datetime | None = None, 
    until: datetime | None = None, 
    limit: int = 50, 
    include_deleted: bool = False, 
    mode: str = "hybrid"
) -> list[AgentEvent]
```
Perform hybrid search directly on the event ledger to recall prior actions or specific tool calls.

### `create_checkpoint()`

```py
await engram.create_checkpoint(
    task_run_id: str, 
    summary: str, 
    *, 
    completed_steps: list[str] | None = None, 
    pending_steps: list[str] | None = None, 
    decisions: list[str] | None = None, 
    blockers: list[str] | None = None, 
    artifacts: list[dict] | None = None, 
    source_event_ids: list[str] | None = None, 
    metadata: dict | None = None
) -> TaskCheckpoint
```
Saves a compact state snapshot summarizing the agent's progress, pending goals, and blockers.

### `build_context()`

```py
await engram.build_context(
    task_run_id: str, 
    *, 
    query: str = "", 
    max_tokens: int = 200000, 
    token_counter: Callable | None = None, 
    recent_event_limit: int = 40, 
    memory_limit: int = 25, 
    checkpoint_limit: int = 3, 
    include_graph: bool = True
) -> ContextBuildResult
```
Assembles a unified prompt context encompassing the latest checkpoints, recent events, relevant semantic memories, and graph traversal logic.

---

## 8. Long Input Processing

### `record_long_input()`

```py
await engram.record_long_input(
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
    max_facts_per_chunk: int = 6
) -> LongInputIngestionReport
```
Ingests massive text contexts (e.g., source code, PR diffs, or contracts) and recursively chunks them, extracting actionable and citable facts.

### `build_long_input_context()`

```py
await engram.build_long_input_context(
    task_run_id: str, 
    *, 
    query: str, 
    max_tokens: int = 4000, 
    source_chunk_limit: int = 6, 
    expected_terms: list[str] | None = None, 
    token_counter: Callable | None = None
) -> LongInputContextResult
```
Retrieves targeted, source-anchored chunks from a previously recorded long input for precise answering.

---

## 9. Background Jobs

### `process_memory_jobs()`

```py
await engram.process_memory_jobs(*, limit: int = 10) -> list[MemoryJob]
```
Manually processes a batch of deferred derivation jobs queued by operations like `record_turn`.

### `run_memory_worker()`

```py
await engram.run_memory_worker(
    *, 
    batch_size: int = 10, 
    interval_seconds: float = 1.0, 
    stop_event: asyncio.Event | None = None, 
    max_iterations: int | None = None
) -> int
```
Runs a continuous loop to consume background memory jobs. Suitable for launching in an `asyncio.create_task()` daemon.

---

## 10. Health & Diagnostics

### `health_check()`

```py
await engram.health_check() -> dict[str, Any]
```
Runs diagnostic checks verifying the database connection, schema version, and accessibility of configured LLM/Embedding providers.
