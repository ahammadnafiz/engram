# Task Memory & Checkpointing

Task Memory is the stateful half of the Engram architecture. While **Fact Memory** stores distilled, semantic knowledge across time, **Task Memory** provides the immutable ledger of what happened, when it happened, and what decisions were made during a specific job.

| Plane | Tables | Primary Purpose |
|-------|--------|-----------------|
| **Fact Memory** | `agent_memory`, `memory_relations` | Fast semantic retrieval, semantic deduplication, and policy pinning. |
| **Task Memory** | `agent_task_runs`, `agent_events`, `agent_checkpoints` | Auditability, resumability, and raw conversational ledgers. |

## 1. When to use Task Memory

You should use Task Memory for any agent that:
- Runs for more than one single conversational turn.
- Needs to pause and resume work after process restarts.
- Requires auditability (knowing exactly what prompt the LLM saw).
- Generates artifacts or executes tool calls.
- Ingests massive documents that need to be chunked and cited.

> [!NOTE]
> If you are building a simple, stateless Slack bot that only answers isolated questions without needing to "resume" ongoing work, you can bypass Task Memory and just use `add()` and `search()`.

---

## 2. Managing the Task Lifecycle

### Starting a Task
Tasks are scoped to an agent and an optional user. You can pass arbitrary metadata to track app-specific identifiers (like repository names or ticket IDs).

```python
task = await engram.start_task(
    goal="Ship persistent memory for the coding agent",
    agent_id="codex",
    user_id="nafiz",
    metadata={"repo": "engram"},
)

print(f"Task ID: {task.task_run_id}")
```

### Managing Status
The Task Manager guards status transitions. A task moves from `active` -> `paused` -> `completed` (or `failed` / `cancelled`).

```python
await engram.pause_task(task.task_run_id, outcome="Waiting on PR review")
await engram.complete_task(task.task_run_id, outcome="PR merged successfully")
```

---

## 3. The Immutable Ledger

Instead of manually updating the database, agents append events to the Task Ledger.

### Recording Turns
The `record_turn()` helper writes user messages, assistant responses, tool executions, and artifact generation into a single atomic transaction.

```python
await engram.record_turn(
    task_run_id=task.task_run_id,
    user_message="Implement deterministic recall.",
    assistant_response="I will add the trace_recall method.",
    tool_calls=[{"name": "pytest", "result": "273 passed"}],
    enqueue_processing=True  # Important: Queues semantic extraction!
)
```

> [!WARNING]
> By default, `record_turn` does **not** extract semantic facts (like "User prefers pytest") immediately. If `enqueue_processing=True`, it writes a job to `memory_jobs`. You must run the background worker to actually convert these raw events into semantic Facts.

### Searching the Ledger
If you need to know what was *said* recently, rather than what was *learned*, you can search the event ledger using hybrid (vector + keyword) search.

```python
hits = await engram.search_events(
    query="chatbot memory jobs",
    agent_id="codex",
    roles=["user"],
    event_types=["user_message"],
    limit=5,
)
```

---

## 4. Checkpoints & Resumability

If an agent has 500 events in its ledger, sending all 500 back to the LLM will exhaust the context window. Instead, the background worker periodically compresses the ledger into a **Checkpoint**.

You can also create checkpoints manually:

```python
checkpoint = await engram.create_checkpoint(
    task_run_id=task.task_run_id,
    summary="Ruff is clean and unit tests pass.",
    completed_steps=["Clean Ruff", "Restore CI lint gates"],
    pending_steps=["Update docs", "Run final package checks"],
    decisions=["Keep public API examples in docs syntax-checked"],
)
```

---

## 5. The Production Turn Loop

In production, you should combine Task Memory (the ledger) with Fact Memory (retrieval) using the intelligent `engram.recall()` operator. 

`recall()` uses an LLM intent classifier to decide whether the user is asking about the current task state, broad historical facts, or an exact past event.

```python
async def handle_turn(engram, task_id, user_message):
    
    # 1. Fetch cognitive context (Checkpoints + Semantic Facts)
    #    build_context assembles a deterministic block scoped to the task:
    #    recent events, checkpoints, and search-ranked memories.
    context = await engram.build_context(task_id, query=user_message)

    # 2. Call your LLM
    response = await call_your_llm(
        memory_context=context.text,
        user_message=user_message
    )

    # 3. Append to the ledger
    await engram.record_turn(
        task_id,
        user_message=user_message,
        assistant_response=response,
        enqueue_processing=True,
    )

    return response
```

> [!TIP]
> Run `engram.run_memory_worker()` in a separate background process to digest the `record_turn` events without blocking the user's web request.

---

## 6. Long Documents & Exact Citation

If you pass a 100-page Legal PDF to an LLM, you must be able to cite the exact paragraph it used to make a decision. 

Use `record_long_input()` to chunk the document into Artifact Events.

```python
report = await engram.record_long_input(
    task.task_run_id,
    text=massive_legal_document,
    title="Vendor agreement review",
    max_chunk_tokens=700,
    extract_with_llm=True, # Extracts searchable facts from the chunks
)
```

When building context to answer a question about this document, Engram returns the exact source chunk IDs.

```python
context = await engram.build_long_input_context(
    task.task_run_id,
    query="What are the termination obligations?",
    expected_terms=["termination", "notice"],
)

print(context.trace["source_chunk_event_ids"])
```

> [!IMPORTANT]
> For legal or medical applications, configure your LLM System Prompt to explicitly cite the `source_chunk_event_ids` provided in the context block. Fail closed if the `missing_expected_terms` array is populated.
