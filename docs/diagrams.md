# Architecture Diagrams

## Two-Plane Memory Architecture

```mermaid
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#fffde7'
    tertiaryColor: '#f3e5f5'
    fourthColor: '#e8f5e9'
---
flowchart TD
    %% Input
    App{{<b>Agent / LLM app</b>}}

    %% Client Plane
    subgraph CLIENT_PLANE ["🔵 Engram Client Plane"]
        direction TB
        Client([<b>Engram Client</b>])
        Policy([<b>MemoryPolicy</b><br/>type + critical slot + conflict key])
        MemoryAPI([<b>Memory APIs</b><br/>add/search/trace])
        EvidenceAPI([<b>Evidence APIs</b><br/>diverse hits + neighbors + reader])
        TaskAPI([<b>Task APIs</b><br/>tasks/events/checkpoints/jobs])
        LongInput([<b>Long Input APIs</b><br/>source + chunks + manifest])
        Graph([<b>GraphTraversal</b>])
        
        Client --> Policy & MemoryAPI & EvidenceAPI & TaskAPI & LongInput & Graph
    end

    %% Services & DB Plane
    subgraph DATA_PLANE ["🟢 Storage & Services Plane"]
        direction TB
        MemoryStore(MemoryStore)
        TaskMgr(TaskMemoryManager)
        Embed(EmbeddingService)
        LLM(LLMService)
        DB[(<b>PostgreSQL</b><br/>pgvector + pg_trgm)]
    end
    
    %% Connections
    App ==> Client
    Policy & MemoryAPI & EvidenceAPI --> MemoryStore
    TaskAPI & LongInput --> TaskMgr
    EvidenceAPI & LongInput --> LLM
    MemoryStore --> Embed
    
    MemoryStore & TaskMgr & Graph ==> DB

    %% DB Tables
    subgraph TABLES ["🟡 Tables"]
        direction TB
        Fact[(agent_memory)]
        Relations[(memory_relations)]
        Sessions[(agent_sessions)]
        Tasks[(agent_task_runs)]
        Events[(agent_events)]
        Checkpoints[(agent_checkpoints)]
        Jobs[(memory_jobs)]
    end
    
    DB -.-> TABLES

    %% Styling
    style CLIENT_PLANE fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style DATA_PLANE fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10
    style TABLES fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;
    classDef dbnode fill:#fff,stroke:#fbc02d,stroke-width:2px;

    class App source;
    class Client,Policy,MemoryAPI,EvidenceAPI,TaskAPI,LongInput,Graph api;
    class MemoryStore,TaskMgr,Embed,LLM process;
    class DB,Fact,Relations,Sessions,Tasks,Events,Checkpoints,Jobs dbnode;
```

## Connect And Schema Initialization

```mermaid
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#fffde7'
    actorBkg: '#fff'
    actorBorder: '#01579b'
    signalColor: '#333'
---
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
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#fffde7'
    actorBkg: '#fff'
    actorBorder: '#01579b'
---
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
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#f3e5f5'
---
flowchart TD
    Q{{"<b>trace_recall(query)</b>"}}
    
    subgraph RETRIEVAL ["🔵 1 · Retrieval Pipelines"]
        direction TB
        Critical([<b>recall_critical</b><br/>metadata lookup])
        Search([<b>deep_search/search</b><br/>vector + keyword + decay])
        Old(<b>list superseded</b><br/>for observability)
    end
    
    subgraph FILTERING ["🟣 2 · Fusion & Trimming"]
        direction TB
        Merge(Dedupe and rank<br/>critical first)
        Budget(Trim to max_tokens)
    end
    
    Trace((<b>RecallTrace</b><br/>kept / trimmed / missing / superseded))
    
    Q ==> Critical & Search & Old
    Critical & Search ==> Merge
    Merge ==> Budget
    Budget ==> Trace
    Old -.-> Trace

    style RETRIEVAL fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style FILTERING fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;

    class Critical,Search api;
    class Old,Merge,Budget process;
    class Q source;
```

## Evidence Retrieval

```mermaid
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#fffde7'
    tertiaryColor: '#f3e5f5'
---
flowchart TD
    Question{{<b>aggregation question</b>}}
    
    subgraph FETCH ["🔵 1 · High-Recall Fetch"]
        direction TB
        Retrieval([<b>deep_search</b>])
        Group([<b>get_memories</b><br/>session/group neighbors])
    end
    
    subgraph ASSEMBLE ["🟡 2 · Context Assembly"]
        direction TB
        Context([<b>get_context_block</b><br/>budgeted block])
    end
    
    subgraph GENERATE ["🟣 3 · Machine Reading"]
        direction TB
        Reader([<b>engram.llm reader</b>])
    end
    
    Answer((<b>grounded answer</b>))
    
    Question ==> Retrieval
    Retrieval ==> Group
    Group ==> Context
    Context ==> Reader
    Reader ==> Answer

    style FETCH fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style ASSEMBLE fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10
    style GENERATE fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;

    class Retrieval,Group,Context,Reader api;
    class Question source;
```

The benchmark's reference reader (`scripts/longmemeval_harness.py`) composes
this same flow with session diversification and an evidence ledger.

## Long-Running Task Flow

```mermaid
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#fffde7'
    actorBkg: '#fff'
    actorBorder: '#01579b'
---
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
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#fffde7'
---
flowchart TD
    Input{{<b>large prompt / legal doc / spec</b>}}
    
    subgraph PREP ["🔵 1 · Chunking"]
        direction TB
        Source(source event: raw text)
        Split(chunk by heading and token estimate)
        ChunkEvents[(artifact events<br/>chunk_id, char span, quote_hash)]
    end
    
    subgraph INGEST ["🟡 2 · Fact Extraction"]
        direction TB
        Extract([<b>LLM or heuristic fact extraction</b>])
        Memories[(anchored agent_memory rows<br/>source_event_id, chunk_id, char span)]
        Manifest(manifest checkpoint)
    end
    
    Context([<b>build_long_input_context</b>])
    
    Input ==> Source
    Source ==> Split
    Split ==> ChunkEvents
    ChunkEvents ==> Extract
    Extract ==> Memories
    ChunkEvents -.-> Manifest
    
    Memories ==> Context
    Manifest -.-> Context
    Source -.-> Context

    style PREP fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style INGEST fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;
    classDef dbnode fill:#fff,stroke:#fbc02d,stroke-width:2px;

    class Source,Split,Manifest process;
    class Extract,Context api;
    class Input source;
    class ChunkEvents,Memories dbnode;
```

## Database Entity View

```mermaid
---
config:
  theme: base
  look: handDrawn
---
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
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#fffde7'
---
flowchart LR
    subgraph FACT ["🔵 fact / content"]
        direction TB
        F1(concise extracted fact)
        F2([embedded])
        F3(searched by vector + keyword)
        F4(returned in prompt context)
        F1 --> F2 --> F3 --> F4
    end

    subgraph MAIN ["🟡 main_content"]
        direction TB
        M1(source conversation or document chunk)
        M2([not embedded])
        M3(not used for vector ranking)
        M4(returned as supporting context)
        M1 --> M2 --> M3 --> M4
    end

    Cost((Paid provider call<br/>or local compute))
    Savings((No extra<br/>embedding cost))

    F2 -. "embedding cost" .-> Cost
    M2 -. "storage only" .-> Savings

    style FACT fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style MAIN fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    
    class F1,F3,F4,M1,M3,M4 process;
    class F2,M2 api;
```
