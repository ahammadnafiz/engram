# Architecture Diagrams

## Two-Plane Memory Architecture

```mermaid
flowchart TD
    App["Agent / LLM app"] --> Client["Engram client"]

    Client --> Policy["MemoryPolicy<br/>type + critical slot + conflict key"]
    Client --> MemoryAPI["Memory APIs<br/>add/search/trace"]
    Client --> EvidenceAPI["Evidence APIs<br/>diverse hits + neighbors + reader"]
    Client --> TaskAPI["Task APIs<br/>tasks/events/checkpoints/jobs"]
    Client --> LongInput["Long Input APIs<br/>source + chunks + manifest"]
    Client --> Graph["GraphTraversal"]

    Policy --> MemoryStore["MemoryStore"]
    MemoryAPI --> MemoryStore
    EvidenceAPI --> MemoryStore
    MemoryStore --> Embed["EmbeddingService"]
    EvidenceAPI --> LLM["LLMService"]
    LongInput --> LLM
    Embed --> Providers["Embedding providers"]
    LLM --> LLMProviders["LLM providers"]

    TaskAPI --> TaskMgr["TaskMemoryManager"]
    LongInput --> TaskMgr
    Graph --> DB[("PostgreSQL + pgvector + pg_trgm")]
    MemoryStore --> DB
    TaskMgr --> DB

    DB --> Fact["agent_memory<br/>typed facts + embeddings + metadata"]
    DB --> Relations["memory_relations"]
    DB --> Sessions["agent_sessions"]
    DB --> Tasks["agent_task_runs"]
    DB --> Events["agent_events"]
    DB --> Checkpoints["agent_checkpoints"]
    DB --> Jobs["memory_jobs"]
```

## Connect And Schema Initialization

```mermaid
sequenceDiagram
    participant A as Application
    participant E as Engram
    participant ES as EmbeddingService
    participant DB as PostgreSQL
    participant L as LLMService

    A->>E: connect()
    E->>ES: from_settings()
    ES-->>E: embedding dimension
    E->>DB: connect()
    E->>DB: init_schema(embedding_dimension)
    DB-->>E: schema ready or dimension error
    E->>L: from_settings()
    L-->>E: optional LLM or None
    E-->>A: connected client
```

## Memory Write With Policy

```mermaid
sequenceDiagram
    participant A as Application
    participant E as Engram
    participant P as MemoryPolicy
    participant S as MemoryStore
    participant V as EmbeddingService
    participant DB as PostgreSQL

    A->>E: add(content, agent_id, main_content)
    E->>P: apply_metadata()
    P-->>E: inferred type, critical_slot, conflict_key
    E->>S: add(MemoryCreate)
    S->>V: embed(content/fact)
    V-->>S: vector
    S->>DB: duplicate guard + INSERT agent_memory
    S->>DB: supersede older active rows with same conflict_key
    DB-->>S: memory row
    E-->>A: Memory
```

## Trace Recall

```mermaid
flowchart TD
    Q["trace_recall(query)"] --> Critical["recall_critical<br/>metadata lookup"]
    Q --> Search["deep_search/search<br/>vector + keyword + decay + importance"]
    Q --> Old["list superseded<br/>for observability"]

    Critical --> Merge["dedupe and rank<br/>critical first"]
    Search --> Merge
    Merge --> Budget["trim to max_tokens"]
    Budget --> Trace["RecallTrace<br/>kept / trimmed / missing / superseded"]
    Old --> Trace
```

## Evidence Retrieval

```mermaid
flowchart TD
    Question["aggregation question"] --> Retrieval["deep_search (high-recall)"]
    Retrieval --> Group["get_memories (session/group neighbors)"]
    Group --> Context["get_context_block (budgeted block)"]
    Context --> Reader["engram.llm reader prompt"]
    Reader --> Answer["grounded answer"]
```

The benchmark's reference reader (`scripts/longmemeval_harness.py`) composes
this same flow with session diversification and an evidence ledger.

## Long-Running Task Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant E as Engram
    participant DB as PostgreSQL
    participant W as Memory Worker

    A->>E: start_task(goal)
    E->>DB: INSERT agent_task_runs
    A->>E: build_context(task_id, query)
    E->>DB: SELECT task, events, checkpoints, memories
    E-->>A: ContextBuildResult
    A->>E: record_turn(user, assistant, tools)
    E->>DB: INSERT agent_events
    E->>DB: INSERT memory_jobs(turn_ingest)
    W->>E: process_memory_jobs()
    E->>DB: claim memory_jobs
    E->>DB: INSERT derived agent_memory if LLM exists
    E->>DB: INSERT agent_checkpoints
    E->>DB: mark memory_jobs completed
```

## Long Input

```mermaid
flowchart TD
    Input["large prompt / legal doc / spec"] --> Source["source event<br/>raw text"]
    Source --> Split["chunk by heading and token estimate"]
    Split --> ChunkEvents["artifact events<br/>chunk_id, char span, quote_hash"]
    ChunkEvents --> Extract["LLM or heuristic fact extraction"]
    Extract --> Memories["anchored agent_memory rows<br/>source_event_id, chunk_id, char span"]
    ChunkEvents --> Manifest["manifest checkpoint"]
    Memories --> Context["build_long_input_context"]
    Manifest --> Context
    Source --> Context
```

## Database Entity View

```mermaid
erDiagram
    agents ||--o{ agent_memory : owns
    users ||--o{ agent_memory : scopes
    agent_memory ||--o{ memory_relations : source
    agent_memory ||--o{ memory_relations : target
    agents ||--o{ agent_sessions : owns
    agents ||--o{ agent_task_runs : owns
    agent_task_runs ||--o{ agent_events : records
    agent_task_runs ||--o{ agent_checkpoints : snapshots
    memory_jobs }o--|| agent_task_runs : derives

    agent_memory {
        text memory_id PK
        text agent_id
        text user_id
        text session_id
        text content
        text fact
        text main_content
        text memory_type
        vector embedding
        float importance
        int access_count
        jsonb metadata
    }

    agent_task_runs {
        text task_run_id PK
        text agent_id
        text user_id
        text session_id
        text goal
        text status
        text outcome
        jsonb metadata
    }

    agent_events {
        text event_id PK
        text task_run_id
        text role
        text event_type
        text content
        jsonb payload
        jsonb metadata
    }

    agent_checkpoints {
        text checkpoint_id PK
        text task_run_id
        text summary
        text[] completed_steps
        text[] pending_steps
        jsonb metadata
    }

    memory_jobs {
        text job_id PK
        text job_type
        text status
        int attempts
        jsonb payload
        text error
    }
```

## Two-Column Cost Model

```mermaid
flowchart LR
    subgraph Fact["fact / content"]
        F1["concise extracted fact"]
        F2["embedded"]
        F3["searched by vector + keyword"]
        F4["returned in prompt context"]
    end

    subgraph Main["main_content"]
        M1["source conversation or document chunk"]
        M2["not embedded"]
        M3["not used for vector ranking"]
        M4["returned as supporting context"]
    end

    F1 --> F2 --> F3 --> F4
    M1 --> M2 --> M3 --> M4

    F2 -. "embedding cost" .-> Cost["paid provider call or local compute"]
    M2 -. "storage only" .-> Savings["no extra embedding cost"]
```
