# Chatbot Example Concept

`examples/chatbot.py` is the end-to-end demonstration app for the current Engram
API. It is not a minimal chatbot. It intentionally exercises persistent memory,
typed recall, task memory, graph traversal, and background job processing.

## What It Demonstrates

| Capability | API |
|------------|-----|
| Persistent facts | `add()`, `search()`, `list_recent()` |
| Typed/critical recall | `trace_recall()` |
| Conflict-aware updates | `add()` policy metadata, `update()`, `forget()` |
| Long-running task state | `start_task()`, `build_context()` |
| Raw turn ledger | `record_turn()` |
| Background derivation | `process_memory_jobs()` |
| Graph context | `relate()`, `traverse()`, `traverse_many()`, `render_graph_context()` |
| Health/status | `health_check()` |
| Cleanup | `purge()` |

## Runtime Flow

```text
User message
    |
    v
trace_recall()
    - critical memories first
    - deep/vector search
    - trace missing/trimmed/superseded
    |
    v
build_context()
    - active task
    - recent events
    - checkpoints
    - typed memory search
    - graph expansion
    |
    v
LLM response
    |
    v
record_turn()
    - user event
    - assistant event
    - optional tool/artifact events
    - enqueue memory job
    |
    v
process_memory_jobs()
    - extract facts
    - create/update memories
    - create checkpoints
```

## Why It Uses A Task

The chatbot starts a task so the conversation can survive process restarts and
multi-day usage. Short-term chat history is still kept in memory for convenience,
but durable continuity lives in:

- `agent_tasks`
- `agent_events`
- `task_checkpoints`
- `agent_memory`

## In-Chat Commands

| Command | Behavior |
|---------|----------|
| `/memories` | Shows recent stored facts |
| `/search <q>` | Runs hybrid search |
| `/graph` | Shows related graph memories and rendered graph context |
| `/task` | Shows active task context |
| `/worker` | Processes queued memory jobs |
| `/forget` | Purges memories for the configured agent/user |
| `/help` | Shows command help |
| `/quit` | Exits |

## Memory Prompt Assembly

The chatbot builds prompt context from multiple sources:

1. System prompt.
2. `trace_recall()` memory block.
3. `build_context()` task memory block.
4. Recent in-process sliding window.
5. Current user message.

This combination keeps the prompt useful when a session is short, long,
interrupted, or resumed later.

## Failure Observability

The chatbot stores the most recent `RecallTrace`. When recall looks wrong, inspect:

- `critical_memory_ids`
- `search_memory_ids`
- `kept_memory_ids`
- `trimmed_memory_ids`
- `superseded_memory_ids`
- `missing_expected_terms`

That gives a concrete answer to whether a fact was never stored, not ranked,
trimmed from the prompt, or superseded by a correction.

## Configuration

Local embeddings:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
python examples/chatbot.py
```

OpenAI LLM and embeddings:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...
python examples/chatbot.py
```

## Production Pattern

For a production chatbot, split the example into two processes:

- API/chat process: builds context, calls the model, records turns.
- Worker process: runs `run_memory_worker()` and handles memory derivation.

The example uses `/worker` and inline calls so the behavior is easy to inspect
locally.

