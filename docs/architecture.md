# Architecture

Engram is an async Python memory layer backed by PostgreSQL + pgvector. The
current alpha architecture is built for persistent fact memory, source-aware
long-input handling, graph expansion, and resumable long-running task state.

## System View

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
    App{{<b>Application / Agent</b>}}

    %% Engram Client Subgraph
    subgraph CLIENT ["🔵 Engram Client"]
        direction TB
        MemAPI([<b>Memory API</b><br/>add, search, trace])
        CtxAPI([<b>Context API</b><br/>block builder])
        Pol([<b>Policy</b><br/>metadata & conflict])
        TaskAPI([<b>Task API</b><br/>events, jobs])
        LongAPI([<b>Long Input API</b><br/>chunking])
        GraphAPI([<b>Graph API</b><br/>relations])
    end

    %% Services Subgraph
    subgraph SERVICES ["🟢 Backend Services"]
        direction TB
        MemStore(MemoryStore)
        TaskMgr(TaskMemoryManager)
        CtxBld(ContextBuilder)
        GraphTrav(GraphTraversal)
        EmbSvc(EmbeddingService)
        LLMSvc(LLMService)
    end

    %% Database Subgraph
    subgraph DB_LAYER ["🟡 Data Layer"]
        direction TB
        DB[(<b>PostgreSQL</b><br/>pgvector + pg_trgm)]
    end

    %% Flow
    App ==> CLIENT
    CLIENT ==> SERVICES
    SERVICES ==> DB_LAYER

    %% Styling
    style CLIENT fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style SERVICES fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10
    style DB_LAYER fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;
    classDef dbnode fill:#fff,stroke:#fbc02d,stroke-width:2px;

    class MemStore,TaskMgr,CtxBld,GraphTrav,EmbSvc,LLMSvc process;
    class MemAPI,CtxAPI,Pol,TaskAPI,LongAPI,GraphAPI api;
    class App source;
    class DB dbnode;
```

## Design Choices

| Decision | Reason |
|----------|--------|
| PostgreSQL as the required store | ACID writes, vector search, full-text search, JSONB, recursive CTEs |
| Two-column memories | embed compact facts and preserve source context without extra embedding cost |
| Policy metadata | critical facts and conflicts need deterministic retrieval rules |
| Append-oriented event ledger | long-running agents need auditability and replayable context |
| Checkpoints | resuming a task should not require replaying every raw event |
| Durable memory jobs | fact derivation can be decoupled from the user-facing turn |
| Recall traces | missed retrievals must be diagnosable |
| Evidence APIs | aggregation questions need diverse coverage, not only top-k relevance |

## Component Responsibilities

### `Engram`

`src/engram/client.py` is the public facade. It owns lifecycle and exposes:

- memory CRUD
- search, deep search, critical recall, and trace recall
- evidence-set retrieval and neighboring context
- task, event, checkpoint, and memory-job APIs
- long-input ingestion and context
- graph relation and traversal APIs
- session and health APIs

### `MemoryPolicy`

`src/engram/policy.py` controls type inference, critical memory selection,
critical slots, and conflict keys. Policies enrich metadata before `MemoryStore`
writes a memory.

### `MemoryStore`

`src/engram/memory/store.py` handles embeddings, inserts, updates,
near-duplicate detection, conflict superseding, hybrid search, and listing
policy memories.

### `TaskMemoryManager`

`src/engram/task/manager.py` persists task runs, ledger events, redactions,
checkpoints, and durable memory jobs.

### `ContextBuilder`

`src/engram/task/context.py` builds bounded task context from task state, recent
events, checkpoints, typed memory search, and optional graph traversal.

### `GraphTraversal`

`src/engram/graph/traversal.py` creates and traverses typed memory relations.
`traverse_many()` supports prompt assembly from several retrieved memories.

### Provider Services

`EmbeddingService` and `LLMService` create configured providers from
`EngramSettings`. Embeddings are required. LLMs are optional and enable fact
extraction, query expansion, and evidence answering.

## Database Tables

| Table | Purpose |
|-------|---------|
| `agents` | agent namespace |
| `users` | optional user namespace |
| `agent_memory` | fact memory with embeddings, type, metadata, and source context |
| `memory_relations` | directed graph edges between memories |
| `agent_sessions` | conversation sessions and rolling summaries |
| `agent_task_runs` | long-running task runs |
| `agent_events` | raw user/assistant/tool/agent/system event ledger with optional embeddings for hybrid event recall |
| `agent_checkpoints` | compact task summaries |
| `memory_jobs` | durable queue for derivation work |

## Connect Flow

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
    %% Input
    Start{{"<b>Engram.connect()</b>"}}

    subgraph PREP ["🔵 1 · Preparation"]
        direction TB
        Emb([<b>EmbeddingService</b><br/>from_settings]) --> 
        Detect(Detect embedding dimension)
    end

    subgraph INIT ["🟡 2 · Initialization"]
        direction TB
        DBConn([<b>PostgresStorage</b><br/>connect]) --> 
        Schema(<b>init_schema</b><br/>run migrations, align dimension) -->
        Svcs(<b>Initialize Services</b><br/>MemoryStore, Tasks, etc.)
    end

    %% Flow
    Start ==> PREP
    PREP ==> INIT

    %% Styling
    style PREP fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style INIT fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;

    class Detect,Schema,Svcs process;
    class Emb,DBConn api;
    class Start source;
```

If a vector dimension change would clear existing embeddings,
`init_schema()` raises unless `ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true`.

## Memory Write Flow

```mermaid
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#e8f5e9'
---
flowchart TD
    %% Input
    Input{{"<b>engram.add()</b><br/>fact, context, metadata"}}

    subgraph POLICY ["🔵 1 · Policy Check"]
        direction TB
        Pol([<b>MemoryPolicy</b><br/>apply_metadata]) -->
        Infer(Infer type & assign conflict_key)
    end

    subgraph STORAGE ["🟢 2 · Storage & Dedup"]
        direction TB
        Add([<b>MemoryStore</b><br/>add]) -->
        Lock(Acquire duplicate-scope lock) -->
        Dedup(Check near duplicates) -->
        Insert[(Insert to agent_memory)] -->
        Supersede(Supersede older conflict rows)
    end

    Out((<b>Memory</b>))

    %% Flow
    Input ==> POLICY
    POLICY ==> STORAGE
    STORAGE ==> Out

    %% Styling
    style POLICY fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style STORAGE fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;
    classDef dbnode fill:#fff,stroke:#2e7d32,stroke-width:2px;

    class Infer,Lock,Dedup,Supersede process;
    class Pol,Add api;
    class Input source;
    class Insert dbnode;
```

## Recall Flow

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
    %% Input
    Input{{"<b>trace_recall(query)</b>"}}

    subgraph RETRIEVAL ["🔵 1 · Retrieval Pipelines"]
        direction TB
        Crit(["<b>recall_critical()</b><br/>metadata lookup"])
        Srch(["<b>deep_search() / search()</b><br/>vector + keyword + decay"])
        Hist(List superseded history)
    end

    subgraph RANKING ["🟣 2 · Fusion & Ranking"]
        direction TB
        Dedupe(Dedupe critical + search hits) -->
        Trim(Trim to prompt budget)
    end

    Out((<b>RecallTrace</b>))

    %% Flow
    Input ==> Crit & Srch & Hist
    Crit & Srch ==> Dedupe
    Hist ==> Trim
    Trim ==> Out

    %% Styling
    style RETRIEVAL fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style RANKING fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;

    class Dedupe,Trim,Hist process;
    class Crit,Srch api;
    class Input source;
```

## Evidence Flow

Aggregation questions, where the answer may be spread across several turns or
sessions, are composed from public primitives:

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
    %% Stages
    subgraph SEARCH ["🔵 1 · Search"]
        direction TB
        Deep(["<b>deep_search()</b><br/>multi-query retrieval"])
    end

    subgraph EXPAND ["🟡 2 · Expand Context"]
        direction TB
        Mem(["<b>get_memories()</b><br/>neighboring session turns"]) -->
        Ctx(["<b>get_context_block()</b><br/>budgeted prompt block"])
    end

    subgraph READ ["🟣 3 · Machine Reading"]
        direction TB
        LLM(["<b>LLM.complete()</b><br/>answer from evidence"])
    end

    %% Flow
    SEARCH ==> EXPAND
    EXPAND ==> READ

    %% Styling
    style SEARCH fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style EXPAND fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10
    style READ fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,rx:10,ry:10

    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    class Deep,Mem,Ctx,LLM api;
```

The session-diversified selection, turn-window expansion, and multi-call
evidence-ledger reader used by the LongMemEval benchmark are a reference
implementation of this flow in `scripts/longmemeval_harness.py`, built on these
same public APIs rather than baked into the library.

## Task Flow

```mermaid
---
config:
  theme: base
  look: handDrawn
  themeVariables:
    primaryColor: '#e1f5fe'
    secondaryColor: '#fffde7'
    tertiaryColor: '#e8f5e9'
---
flowchart TD
    Start{{"<b>start_task(goal)</b>"}}

    subgraph LEDGER ["🔵 1 · Event Ledger"]
        direction TB
        Turn(["<b>record_turn()</b><br/>user, assistant, tools"]) -->
        App(agent_events append) -->
        Enq(Enqueue turn_ingest)
    end

    subgraph DERIVE ["🟡 2 · Background Derivation"]
        direction TB
        Work(["<b>process_memory_jobs()</b>"]) -->
        Fact(Fact extraction via LLM)
    end

    subgraph STATE ["🟢 3 · Context Assembly"]
        direction TB
        Ckpt(["<b>create_checkpoint()</b>"]) -->
        Ctx(["<b>build_context()</b>"])
    end

    %% Flow
    Start ==> LEDGER
    LEDGER ==> DERIVE
    DERIVE ==> STATE

    %% Styling
    style LEDGER fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style DERIVE fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10
    style STATE fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;

    class App,Enq,Fact process;
    class Turn,Work,Ckpt,Ctx api;
    class Start source;
```

The raw ledger is authoritative. Derived memories and checkpoints are optimized
views used for recall and prompt assembly.

## Long-Input Flow

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
    Input{{"<b>record_long_input()</b><br/>text payload"}}

    subgraph CHUNKING ["🔵 1 · Chunking"]
        direction TB
        Src(Raw source event) -->
        Split(Chunk by headings) -->
        Art[(Artifact event<br/>char span + quote_hash)]
    end

    subgraph EXTRACTION ["🟡 2 · Fact Extraction"]
        direction TB
        LLM([<b>LLM Extractor</b>]) -->
        Mem[(Anchored memories<br/>with source metadata)] -->
        Man(Manifest checkpoint)
    end

    %% Flow
    Input ==> CHUNKING
    CHUNKING ==> EXTRACTION

    %% Styling
    style CHUNKING fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    style EXTRACTION fill:#fffde7,stroke:#fbc02d,stroke-width:2px,rx:10,ry:10

    classDef process fill:#fff,stroke:#333,stroke-width:1px,rx:8,ry:8;
    classDef api fill:#fff,stroke:#333,stroke-width:2px,rx:20,ry:20;
    classDef dbnode fill:#fff,stroke:#333,stroke-width:1px,rx:2,ry:2;
    classDef source fill:#fff,stroke:#01579b,stroke-width:2px;

    class Src,Split,Man process;
    class LLM api;
    class Art,Mem dbnode;
    class Input source;
```

`build_long_input_context()` combines recall trace, selected source chunks, and
the long-input manifest.

## Search Implementation

Hybrid search uses:

- pgvector cosine similarity over `agent_memory.embedding`
- PostgreSQL full-text search over generated `fact_tsv`
- recency/access decay
- memory importance
- optional JSONB `metadata_filter`
- optional `memory_types`
- optional local cross-encoder reranking

Superseded memories are excluded from normal search with metadata status checks.

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

Providers register through the provider registry and are created from
`EngramSettings`.

## Reliability Boundaries

Engram provides durable storage, retrieval traces, conflict metadata, and
resumable task state. Applications remain responsible for:

- tenant authorization
- PII detection
- legal citation verification
- provider retry policy
- job monitoring and alerting
- user-facing privacy workflows
- human review in high-stakes domains
