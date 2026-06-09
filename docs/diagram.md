# Architecture Diagrams

## Two-Plane Memory Architecture

```mermaid
flowchart TD
    App["Agent / LLM App"] --> Client["Engram Client"]

    Client --> Policy["MemoryPolicy<br/>type + critical slot + conflict key"]
    Client --> TaskAPI["Task APIs<br/>tasks/events/checkpoints/jobs"]
    Client --> LongInput["Long Input APIs<br/>source + chunks + manifest"]
    Client --> Graph["GraphTraversal"]

    Policy --> MemoryStore["MemoryStore"]
    MemoryStore --> Embed["EmbeddingService"]
    Embed --> Providers["Embedding Providers"]

    TaskAPI --> TaskMgr["TaskMemoryManager"]
    LongInput --> TaskMgr
    Graph --> DB[("PostgreSQL + pgvector")]
    MemoryStore --> DB
    TaskMgr --> DB

    DB --> Fact["agent_memory<br/>typed facts + embeddings + metadata"]
    DB --> Relations["memory_relations"]
    DB --> Tasks["agent_task_runs"]
    DB --> Events["agent_events"]
    DB --> Checkpoints["agent_checkpoints"]
    DB --> Jobs["memory_jobs"]
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

    A->>E: add(content, memory_type, metadata)
    E->>P: apply_metadata()
    P-->>E: inferred type, critical_slot, conflict_key
    E->>S: add(MemoryCreate)
    S->>V: embed(content/fact)
    V-->>S: vector
    S->>DB: INSERT agent_memory
    DB-->>S: memory_id
    E->>S: supersede_conflicts(conflict_key, memory_id)
    S->>DB: UPDATE old active memories status=superseded
    E-->>A: Memory
```

## Trace Recall

```mermaid
flowchart TD
    Q["trace_recall(query)"] --> Critical["recall_critical<br/>metadata lookup"]
    Q --> Search["deep_search/search<br/>vector + keyword + decay + importance"]
    Q --> Old["list superseded<br/>for observability"]

    Critical --> Merge["Dedupe and rank<br/>critical first"]
    Search --> Merge
    Merge --> Budget["Trim to max_tokens"]
    Budget --> Trace["RecallTrace<br/>kept / trimmed / missing / superseded"]
    Old --> Trace
```

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
    E->>DB: SELECT task, recent events, checkpoints, memories
    E-->>A: ContextBuildResult
    A->>E: record_turn(user, assistant, tools)
    E->>DB: INSERT agent_events
    E->>DB: INSERT memory_jobs(turn_ingest)
    W->>E: process_memory_jobs()
    E->>DB: claim memory_jobs
    E->>DB: INSERT derived agent_memory
    E->>DB: INSERT agent_checkpoints
    E->>DB: mark memory_jobs completed
```

## Long Input

```mermaid
flowchart TD
    Input["2k+ token prompt / legal doc / spec"] --> Source["source event<br/>raw text"]
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
        F1["Concise extracted fact"]
        F2["Embedded"]
        F3["Searched by vector + keyword"]
        F4["Returned in prompt context"]
    end

    subgraph Main["main_content"]
        M1["Source conversation or document chunk"]
        M2["Not embedded"]
        M3["Not used for vector ranking"]
        M4["Returned as supporting context"]
    end

    F1 --> F2 --> F3 --> F4
    M1 --> M2 --> M3 --> M4

    F2 -. "embedding cost" .-> Cost["Paid provider call or local compute"]
    M2 -. "storage only" .-> Savings["No extra embedding cost"]
```
