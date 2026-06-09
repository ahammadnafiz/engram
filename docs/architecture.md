# Architecture

Engram is an async Python memory layer backed by PostgreSQL + pgvector. The
current alpha architecture is built for agents that need persistent fact memory,
source-aware long-input handling, and resumable long-running task state.

## System View

```text
Application / Agent
        |
        v
Engram Client
  |-- Memory API: add, search, deep_search, trace_recall
  |-- Policy: memory typing, critical slots, conflict keys
  |-- Task API: tasks, events, checkpoints, jobs
  |-- Long Input API: source events, chunks, anchored memories
  |-- Graph API: relations, traversal, prompt graph rendering
        |
        v
Services
  |-- MemoryStore
  |-- TaskMemoryManager
  |-- ContextBuilder
  |-- GraphTraversal
  |-- EmbeddingService
  |-- LLMService
        |
        v
PostgreSQL + pgvector
```

## Core Design Choices

| Decision | Reason |
|----------|--------|
| PostgreSQL as the only required store | ACID writes, vector search, full-text search, JSONB, recursive CTEs |
| Two-column memories | Embed compact facts, preserve full context without extra embedding cost |
| Policy metadata | Critical facts and conflicts need deterministic retrieval rules |
| Append-only event ledger | Long-running agents need auditability and replayable context |
| Checkpoints | Resuming a task should not require replaying every raw event |
| Background memory jobs | Fact derivation can be decoupled from the user-facing turn |
| Recall traces | Missed retrievals must be diagnosable |

## Component Responsibilities

### `Engram`

`src/engram/client.py` is the facade. It owns service lifecycle and exposes the
public API:

- memory CRUD
- search and trace recall
- task/event/checkpoint APIs
- long-input APIs
- graph APIs
- sessions and health checks

### `MemoryPolicy`

`src/engram/policy.py` controls:

- type inference
- critical memory selection
- deterministic critical slots
- conflict keys

Policies do not store data by themselves. They enrich metadata before the memory
is handed to `MemoryStore`.

### `MemoryStore`

`src/engram/memory/store.py` handles:

- embedding facts
- inserting/updating memories
- near-duplicate detection
- superseding older conflict winners
- hybrid search
- listing critical/superseded policy memories

### `TaskMemoryManager`

`src/engram/task/manager.py` owns task persistence:

- task lifecycle
- event ledger
- event redaction
- checkpoints
- memory job queue

### `ContextBuilder`

`src/engram/task/context.py` builds bounded task context from:

- task state
- recent events
- checkpoints
- typed memory search
- optional graph traversal

### `GraphTraversal`

`src/engram/graph/traversal.py` creates and traverses typed memory relations.
`traverse_many()` is used for prompt assembly when several search results should
pull in related decisions, constraints, or tool outputs.

## Database Tables

| Table | Purpose |
|-------|---------|
| `agents` | agent namespace |
| `users` | optional user namespace |
| `agent_memory` | fact memory with embeddings, type, metadata |
| `memory_relations` | directed graph edges between memories |
| `agent_sessions` | conversation sessions and rolling summaries |
| `agent_tasks` | long-running task runs |
| `agent_events` | raw user/assistant/tool/agent/system event ledger |
| `task_checkpoints` | compact task summaries |
| `memory_jobs` | background queue for derivation work |

## Memory Write Flow

```text
engram.add(content, main_content, memory_type, metadata)
        |
        v
MemoryPolicy.apply_metadata()
  - infer type
  - critical_slot
  - conflict_key
        |
        v
MemoryStore.add()
  - embed content/fact
  - near-duplicate guard
  - insert into agent_memory
        |
        v
supersede_conflicts()
  - mark older active memories with same conflict_key as superseded
```

## Recall Flow

```text
trace_recall(query)
        |
        +--> recall_critical()
        |      deterministic metadata lookup
        |
        +--> deep_search() or search()
        |      vector + keyword + decay + importance
        |
        +--> list superseded policy memories
        |
        +--> dedupe critical + ranked
        |
        +--> trim to prompt budget
        |
        v
RecallTrace(context, kept, trimmed, missing, superseded)
```

This makes retrieval debuggable. If a fact did not appear in the prompt, the
trace tells whether it was missing, unranked, trimmed, or superseded.

## Task Flow

```text
start_task(goal)
        |
record_turn(user, assistant, tools, artifacts)
        |
agent_events append user/assistant/tool/artifact records
        |
memory_jobs enqueue turn_ingest
        |
process_memory_jobs() or run_memory_worker()
        |
add_conversation-like fact derivation + checkpoint updates
        |
build_context(task_id, query)
```

The raw ledger is authoritative. Derived memories are optimized views used for
recall.

## Long-Input Flow

```text
record_long_input(task_id, text)
        |
raw source event
        |
chunk by heading/token estimate
        |
artifact event per chunk with char span + quote_hash
        |
extract facts per chunk
        |
anchored memories with chunk metadata
        |
manifest checkpoint
```

`build_long_input_context()` later combines:

- recall trace
- selected source chunks
- long-input manifest

## Search Implementation

Hybrid search uses:

- pgvector cosine similarity over `agent_memory.embedding`
- PostgreSQL full-text search over `fact_tsv`
- recency/access decay
- memory importance
- optional `metadata_filter`
- optional `memory_types`

Superseded memories are excluded from normal search:

```sql
COALESCE(metadata->>'status', 'active') <> 'superseded'
```

## Provider Architecture

Embedding providers:

- OpenAI
- Sentence Transformers
- Cohere
- Ollama
- HuggingFace Inference

LLM providers:

- OpenAI
- Anthropic
- Ollama
- Groq
- LiteLLM

Providers are registered through the provider registry system and created from
`EngramSettings`.

## Reliability Boundaries

Engram provides durable storage, retrieval traces, and resumable task state. It
does not by itself guarantee:

- tenant authorization
- PII detection
- legal citation verification
- exactly-once background processing across all failure modes
- perfect fact extraction from an LLM

Applications should add auth, audit policy, source-citation checks, job
monitoring, and human review where the domain requires it.

