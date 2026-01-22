# Engram Library Implementation Plan

## Architecture Overview

Engram is a drop-in memory layer for AI applications. Users import it, connect to PostgreSQL, provide their embedding function, and get persistent memory with hybrid search, memory decay, and graph traversal. Production-ready with batching, timeouts, monitoring, and error recovery.

```mermaid
flowchart TB
    subgraph UserApp [User Application]
        LLM[LLM API Layer]
        EmbedFn[Embedding Function]
        SumFn[Summarization Function - Optional]
    end
    
    subgraph EngramLib [Engram Library]
        Client[Engram Client]
        EmbedService[Embedding Service<br/>Batching + Cache]
        MemStore[Memory Store<br/>+ Smart Dedup]
        Decay[Memory Decay<br/>+ Compression Tiers]
        Search[Multi-Source Search<br/>+ Knapsack Selection]
        Graph[Graph Traversal]
        Sessions[Session Manager<br/>+ Checkpoints]
        Knowledge[Knowledge State<br/>Accumulated Facts]
        Context[Context Builder<br/>Budget-Aware]
        Health[Health Checks]
        Maintenance[Background Maintenance<br/>+ Active Forgetting]
    end
    
    subgraph Storage [PostgreSQL + pgvector]
        Agents[agents]
        Users[users]
        AgentSessions[agent_sessions]
        Memory[agent_memory<br/>Multi-embedding]
        Relations[memory_relations]
        KnowledgeStore[user_knowledge_state]
        Checkpoints[session_checkpoints]
    end
    
    LLM --> Client
    EmbedFn --> EmbedService
    SumFn -.-> Sessions
    Client --> EmbedService
    Client --> MemStore
    Client --> Search
    Client --> Sessions
    Client --> Graph
    Client --> Context
    Client --> Health
    MemStore --> Memory
    MemStore --> Decay
    Search --> Memory
    Search --> Decay
    Search --> Knowledge
    Graph --> Relations
    Graph --> Memory
    Sessions --> AgentSessions
    Sessions --> Users
    Sessions --> Checkpoints
    Knowledge --> KnowledgeStore
    Context --> Search
    Context --> Knowledge
    Maintenance --> Memory
    Maintenance --> Relations
    Maintenance --> Knowledge
```

---

## Research-Driven Design Decisions

### Memory Approaches (ChatGPT vs Claude)

| Aspect | ChatGPT | Claude | Engram |
|--------|---------|--------|--------|
| History | Pre-computed summaries | On-demand retrieval | Both (configurable) |
| Format | Lightweight digest | Full context | User's choice |
| Trade-off | Speed over depth | Depth over speed | Flexible |

Engram supports both patterns:

- **Default**: Hybrid search with decay (fast, always available)
- **Optional**: Summarization pipeline for conversation consolidation

### Memory Decay (MemoryBank Research)

Based on production-proven MemoryBank formula:

- **Decay rate**: `0.995^hours_elapsed` (~88.6% after 24h, ~43.3% after 1 week)
- **Strength tracking**: Increment on recall, reset decay timer
- **Weighted scoring**: Configurable weights for relevance, recency, importance

### Graph Traversal (Graphiti Research)

Based on Graphiti's P95 300ms latency achievement:

- **Hybrid retrieval**: Vector + Graph + Keyword search
- **Hop limit**: 1-3 hops (configurable)
- **Relationship types**: Typed edges for precise reasoning

### Smart Deduplication (Fact Extraction Research)

Based on semantic deduplication with two-tier thresholds:

- **DUPLICATE_THRESHOLD = 0.90**: Exact duplicate → Skip (NOOP), increment mention_count
- **UPDATE_THRESHOLD = 0.78**: Similar → Merge (UPDATE), combine content
- **Below 0.78**: Treat as new memory (ADD)
- **Correction handling**: Automatically supersede related outdated memories

### Multi-Resolution Compression (Checkpoint Research)

Based on age-based compression tiers:

| Tier | Age | Storage | Detail Level |
|------|-----|---------|--------------|
| **HOT** | < 1 week | Full content | Full fidelity |
| **WARM** | 1-4 weeks | Content + summary | Full |
| **COOL** | 1-3 months | Summary only | Truncated |
| **FROZEN** | > 3 months | Key facts only | Minimal |

### Multi-Source Context Building (Context Builder Research)

Based on modular source architecture with greedy knapsack:

- **Multiple retrieval sources** with configurable weights
- **Early deduplication**: Same memory from multiple sources → sum scores
- **Greedy knapsack selection**: Sort by score, select within token budget
- **Budget-first design**: Never exceed context window

### Database Choice: PostgreSQL + pgvector

Based on comprehensive analysis (see [REDIS_VS_POSTGRESQL_COMPARISON.md](./Research/REDIS_VS_POSTGRESQL_COMPARISON.md)):

- **ACID guarantees** critical for memory consistency
- **Graph traversal** requires recursive CTEs (PostgreSQL excels)
- **Complex queries** needed for hybrid search + decay + filters
- **Cost-effective** at scale (5-18x cheaper than Redis)
- **Production-ready** with full persistence and tooling

---

## Target API Design

```python
from engram import Engram

# Initialize with user's embedding function (supports batch)
async def my_embed(texts: list[str]) -> list[list[float]]:
    return await openai.embeddings.create(input=texts, model="text-embedding-3-small")

memory = Engram(
    database_url="postgresql+asyncpg://user:pass@localhost/engram",
    embedding_fn=my_embed,
    embedding_dim=1536,  # Required: 1024, 1536, or 3072
    agent_name="my-assistant",
    
    # Production configuration
    pool_size=20,
    max_overflow=10,
    query_timeout_ms=5000,
    enable_fallback=True,
    batch_size=100,
    cache_size=10000,
    
    # Deduplication thresholds (from Fact Extraction research)
    duplicate_threshold=0.90,  # Skip if above
    update_threshold=0.78,     # Merge if above
)

# === BASIC MEMORY OPERATIONS (CRUD) ===

# Create - with automatic smart deduplication
memory_id, operation = await memory.add(
    "User prefers dark mode", 
    user_id="user_123", 
    memory_type="preference",  # preference|requirement|context|correction
    entities=["dark mode", "UI"],
    metadata={"source": "chat"}
)
# operation: "ADD" | "UPDATE" | "NOOP" | "CORRECTION"

# Read
results = await memory.search("user preferences", user_id="user_123", limit=5)
specific = await memory.get(memory_id)  # Get by ID
recent = await memory.list_recent(user_id="user_123", limit=20)  # Recent memories
count = await memory.count(user_id="user_123", filters={"type": "preference"})

# Update
await memory.update(memory_id, content="User strongly prefers dark mode")
await memory.reinforce(memory_id)  # Boost memory strength manually
await memory.tag(memory_id, tags=["ui", "preference", "important"])

# Delete
await memory.forget(memory_id)  # Soft delete
await memory.forget_all(user_id="user_123", filters={"type": "temporary"})  # Bulk soft delete
await memory.purge(memory_id)  # Hard delete (permanent)

# === MEMORY TYPES & CORRECTION HANDLING ===

# Add a correction - automatically supersedes related memories
correction_id, op = await memory.add(
    "User now uses Python 3.12 instead of 3.9",
    user_id="user_123",
    memory_type="correction",  # Triggers supersession logic
    entities=["Python"]
)
# Related memories about Python version are now marked as superseded

# Query by memory type
preferences = await memory.search_by_type("preference", user_id="user_123")
corrections = await memory.search_by_type("correction", user_id="user_123")

# === KNOWLEDGE STATE (Accumulated User Knowledge) ===

# Get accumulated knowledge for a user
knowledge = await memory.get_knowledge_state(user_id="user_123")
# Returns: {
#   "facts": {"hash1": "User prefers Python", ...},
#   "entities": ["Python", "TechCorp", ...],
#   "topics": ["programming", "career", ...],
#   "preferences": {"theme": "dark", ...}
# }

# Get knowledge summary (for context injection)
summary = await memory.get_knowledge_summary(user_id="user_123")
# "Facts: User prefers Python, Works at TechCorp | Entities: Python, TechCorp"

# === MULTI-SOURCE SEARCH (Context Builder Pattern) ===

results = await memory.search(
    "user preferences", 
    user_id="user_123",
    # Source weights (from Context Builder research)
    source_weights={
        "recency": 1.0,      # Recent memories
        "semantic": 0.6,     # Vector similarity
        "graph": 0.5,        # Graph-connected
        "importance": 0.7,   # High importance_score
        "keyword": 0.4,      # Text match
    },
    decay_weight=0.25,
    include_stale=False,
    timeout_ms=3000
)

# === BUDGET-AWARE CONTEXT BUILDING ===

context = await memory.build_context(
    query="What are the user's Python preferences?",
    user_id="user_123",
    token_budget=4000,
    include_knowledge_state=True,  # Start with accumulated knowledge
    source_weights={...}
)
# Returns: {
#   "memories": [...],           # Selected memories (chronological)
#   "knowledge_summary": "...",  # From knowledge_state
#   "total_tokens": 3850,
#   "sources_used": {"semantic": 5, "recency": 3, "graph": 2}
# }

# === HUMAN-MEMORY-STYLE OPERATIONS ===

# Consolidation (merge similar memories)
await memory.merge(
    memory_ids=["mem_abc", "mem_def"],
    strategy="latest"  # or "strongest", "combined"
)

# Deduplication (find and merge duplicates)
duplicates = await memory.find_duplicates(user_id="user_123", similarity_threshold=0.95)
await memory.deduplicate(user_id="user_123", auto_merge=True)

# Association (find related without graph)
related = await memory.get_related(memory_id, user_id="user_123", limit=10)

# Bulk operations
memory_ids = await memory.add_batch([
    {"content": "Memory 1", "user_id": "user_123", "memory_type": "context"},
    {"content": "Memory 2", "user_id": "user_123", "memory_type": "preference"}
])
await memory.update_batch([
    {"id": "mem_1", "metadata": {"verified": True}},
    {"id": "mem_2", "importance_score": 0.8}
])

# === SESSION CONTEXT WITH CHECKPOINTS ===

async with memory.session(user_id="user_123") as session:
    await session.add("User asked about Python")
    context = await session.get_context("What did they ask about?", limit=10)
    all_session_memories = await session.get_all()
    
    # Create checkpoint (diff-based, stores only novel info)
    checkpoint = await session.create_checkpoint()
    # checkpoint.diff contains: novel_facts, updated_facts, new_entities, new_topics

# Get session checkpoints
checkpoints = await memory.get_session_checkpoints(session_id="sess_123")

# === COMPRESSION TIERS (Age-Based) ===

# Memories are automatically compressed based on age
# HOT → WARM → COOL → FROZEN

# Force compression tier update
await memory.compress_old_memories(user_id="user_123")

# Get compression statistics
stats = await memory.get_compression_stats(user_id="user_123")
# {"hot": 50, "warm": 200, "cool": 500, "frozen": 1000}

# === GRAPH TRAVERSAL ===

related = await memory.traverse(
    start_memory_id="mem_abc123",
    relation_types=["causes", "relates_to", "contradicts", "supersedes"],
    max_hops=2,
    min_weight=0.5
)

await memory.relate(
    source_id="mem_abc123",
    target_id="mem_xyz789",
    relation_type="causes",
    weight=0.8
)

# === ACTIVE FORGETTING (LLM-Driven Cleanup) ===

# Trigger active forgetting (categorizes: KEEP, COMPRESS, FORGET)
from engram.maintenance import ActiveForgetting

forgetter = ActiveForgetting(memory, llm=my_llm)
results = await forgetter.run(user_id="user_123", max_facts=50)
# {"kept": 30, "compressed": 15, "forgotten": 5}

# === HEALTH CHECKS ===

from engram.health import HealthChecker
health = HealthChecker(memory)
status = await health.full_health_check()

# === MAINTENANCE ===

from engram.maintenance import MemoryMaintenance
maintenance = MemoryMaintenance(memory, cleanup_interval_hours=24)
```

---

## Project Structure

```
engram/
├── pyproject.toml              # Package config, dependencies
├── README.md                   # Documentation
├── docker-compose.yml          # PostgreSQL + pgvector setup
├── Makefile                    # Dev commands
├── engram/
│   ├── __init__.py             # Public API exports
│   ├── client.py               # Main Engram class
│   ├── config.py               # Configuration dataclass
│   ├── exceptions.py           # Custom exceptions
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py       # Async connection pool + monitoring
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   └── migrations/
│   │       └── 001_initial.sql # Schema with multi-embedding support
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py            # Core CRUD operations (add, get, update, delete)
│   │   ├── deduplication.py    # Two-tier smart deduplication (NEW)
│   │   ├── operations.py       # Human-memory operations (merge, deduplicate)
│   │   ├── decay.py            # Memory decay scoring (MemoryBank)
│   │   ├── compression.py      # Multi-resolution compression tiers (NEW)
│   │   └── types.py            # Memory types (preference, requirement, etc.) (NEW)
│   ├── search/
│   │   ├── __init__.py
│   │   ├── hybrid.py           # Hybrid search (RRF + decay + timeouts)
│   │   ├── sources.py          # Retrieval sources (recency, semantic, graph) (NEW)
│   │   └── fusion.py           # Multi-source fusion with weights (NEW)
│   ├── context/
│   │   ├── __init__.py
│   │   ├── builder.py          # Budget-aware context building (NEW)
│   │   └── knapsack.py         # Greedy knapsack selection (NEW)
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── state.py            # User knowledge state management (NEW)
│   │   └── entities.py         # Entity extraction and tracking (NEW)
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── batch_service.py    # Batched embedding generation + cache
│   │   └── cache.py            # LRU cache for embeddings
│   ├── graph/
│   │   ├── __init__.py
│   │   └── traversal.py        # Graph traversal queries
│   ├── session/
│   │   ├── __init__.py
│   │   ├── manager.py          # Session lifecycle
│   │   ├── checkpoints.py      # Diff-based checkpoints (NEW)
│   │   └── summarizer.py       # Optional summarization pipeline
│   ├── utils/
│   │   ├── __init__.py
│   │   └── retry.py            # Retry logic with exponential backoff
│   ├── health/
│   │   ├── __init__.py
│   │   └── checker.py          # Health check endpoints
│   ├── maintenance/
│   │   ├── __init__.py
│   │   ├── cleanup.py          # Background maintenance jobs
│   │   └── forgetting.py       # Active forgetting (LLM-driven) (NEW)
│   ├── tools/
│   │   ├── __init__.py
│   │   └── migrate_embeddings.py # Embedding model migration tool
│   └── types.py                # Type definitions
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── test_memory.py          # CRUD operations
│   ├── test_deduplication.py   # Smart dedup tests (NEW)
│   ├── test_operations.py      # merge, deduplicate, consolidate
│   ├── test_decay.py
│   ├── test_compression.py     # Compression tier tests (NEW)
│   ├── test_search.py
│   ├── test_context.py         # Context builder tests (NEW)
│   ├── test_knowledge.py       # Knowledge state tests (NEW)
│   ├── test_graph.py
│   ├── test_session.py
│   ├── test_checkpoints.py     # Checkpoint tests (NEW)
│   ├── test_batching.py
│   ├── test_retry.py
│   └── test_health.py
└── examples/
    ├── openai_chatbot.py       # OpenAI integration example
    ├── anthropic_chatbot.py    # Anthropic integration example
    ├── ollama_local.py         # Local model example
    ├── graph_reasoning.py      # Multi-hop graph example
    ├── memory_operations.py    # CRUD and human-memory operations
    ├── context_building.py     # Context builder example (NEW)
    ├── knowledge_state.py      # Knowledge state example (NEW)
    └── production_deployment.py # Production configuration example
```

---

## Implementation Details

### 1. Smart Deduplication (from Fact Extraction Research)

**Purpose**: Automatically deduplicate memories at write time using semantic similarity

**Two-Tier Threshold System**:

```
┌─────────────────────────────────────────────────────────────┐
│                  add() Deduplication Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: content, memory_type, entities                     │
│                        │                                    │
│                        ▼                                    │
│           ┌────────────────────────┐                        │
│           │ Is type="correction"?  │                        │
│           └───────────┬────────────┘                        │
│              YES │         │ NO                             │
│                  ▼         ▼                                │
│        ┌──────────────┐  ┌──────────────┐                   │
│        │ Handle       │  │ Get Embedding │                  │
│        │ Correction   │  └───────┬──────┘                   │
│        │ (supersede)  │          │                          │
│        └──────────────┘          ▼                          │
│                        ┌──────────────────┐                 │
│                        │ Search Similar   │                 │
│                        │ (top 10)         │                 │
│                        └────────┬─────────┘                 │
│                                 │                           │
│              ┌──────────────────┼──────────────────┐        │
│              ▼                  ▼                  ▼        │
│        score > 0.90       0.78 < score ≤ 0.90   score < 0.78│
│              │                  │                  │        │
│              ▼                  ▼                  ▼        │
│         ┌────────┐        ┌─────────┐       ┌──────────┐    │
│         │  NOOP  │        │ UPDATE  │       │   ADD    │    │
│         │(skip)  │        │ (merge) │       │  (new)   │    │
│         └────────┘        └─────────┘       └──────────┘    │
│              │                  │                  │        │
│              ▼                  ▼                  ▼        │
│         Increment         Merge content      Insert new     │
│         mention_count     Update embedding   memory         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Deduplication Operations**:

| Operation | Similarity | Action |
|-----------|------------|--------|
| **NOOP** | > 0.90 | Skip insert, increment mention_count, update last_seen |
| **UPDATE** | 0.78 - 0.90 | Merge facts, update embedding, union entities |
| **ADD** | < 0.78 | Insert as new memory |
| **CORRECTION** | Any (type=correction) | Supersede related memories |

**Correction Handling**:

When a memory has `type="correction"`:
1. Embed the correction
2. Search for related memories (by similarity + entity overlap)
3. Mark related memories as `status="superseded"`
4. Add correction as new active memory
5. Create `supersedes` relation in graph

**Configuration**:

```python
# Tuning thresholds
| Scenario | DUPLICATE | UPDATE | Effect |
|----------|-----------|--------|--------|
| Aggressive | 0.85 | 0.70 | Fewer memories, more merging |
| Default | 0.90 | 0.78 | Balanced |
| Conservative | 0.95 | 0.85 | More memories, less merging |
```

### 2. Memory Types (from Fact Extraction Research)

**Purpose**: Categorize memories for better retrieval and correction handling

**Types**:

| Type | Description | Example |
|------|-------------|---------|
| `preference` | User preferences and likes | "User prefers dark mode" |
| `requirement` | Project or task requirements | "API must support OAuth 2.0" |
| `context` | Background information | "User works at TechCorp" |
| `correction` | Updates to previous facts | "User now uses Python 3.12 (was 3.9)" |
| `entity` | Information about an entity | "TechCorp is a startup in SF" |

**Behavior by Type**:

- `preference`: Prioritized in context building, merged on conflict
- `requirement`: Never auto-deleted, high importance
- `context`: Standard decay, can be compressed
- `correction`: Triggers supersession of related memories
- `entity`: Linked to entity graph, lower decay rate

### 3. Multi-Resolution Compression (from Checkpoint Research)

**Purpose**: Reduce storage and improve retrieval by compressing old memories

**Compression Tiers**:

```python
TIER_CONFIG = {
    CompressionTier.HOT: {
        'max_age_days': 7,
        'store_full_content': True,
        'summary_length': None,  # Full
    },
    CompressionTier.WARM: {
        'max_age_days': 30,
        'store_full_content': True,
        'summary_length': None,  # Full
    },
    CompressionTier.COOL: {
        'max_age_days': 90,
        'store_full_content': False,  # Summary only
        'summary_length': 200,
    },
    CompressionTier.FROZEN: {
        'max_age_days': None,  # Forever
        'store_full_content': False,
        'summary_length': 100,  # Ultra-compressed
    },
}
```

**Compression Algorithm**:

```
For each memory (during maintenance):
    1. Calculate age in days
    2. Determine tier based on age
    3. If tier changed:
       - WARM: No change (full content retained)
       - COOL: Generate summary, archive full content
       - FROZEN: Truncate summary, extract key facts only
    4. Update compression_tier field
```

### 4. Knowledge State (from Checkpoint Research)

**Purpose**: Maintain accumulated knowledge per user separate from raw memories

**Schema**:

```sql
CREATE TABLE user_knowledge_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) UNIQUE,
    
    -- Accumulated knowledge
    facts JSONB DEFAULT '{}',           -- fact_hash -> fact_text
    entities TEXT[] DEFAULT '{}',       -- Known entities
    topics TEXT[] DEFAULT '{}',         -- Known topics
    preferences JSONB DEFAULT '{}',     -- Extracted preferences
    
    -- Embeddings for dedup
    fact_embeddings JSONB DEFAULT '{}', -- fact_hash -> embedding (stored as array)
    
    -- Statistics
    total_memories INT DEFAULT 0,
    active_memories INT DEFAULT 0,
    superseded_memories INT DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_knowledge_user ON user_knowledge_state(user_id);
```

**Update Triggers**:

- On memory ADD: Update facts, entities, topics
- On memory UPDATE: Update affected facts
- On memory CORRECTION: Remove superseded facts, add new
- On memory FORGET: Remove from facts (if not mentioned elsewhere)

### 5. Multi-Source Search (from Context Builder Research)

**Purpose**: Combine multiple retrieval sources with configurable weights

**Search Sources**:

| Source | Default Weight | Description |
|--------|---------------|-------------|
| `recency` | 1.0 | Most recent memories (always included) |
| `semantic` | 0.6 | Vector similarity to query |
| `graph` | 0.5 | Graph-connected memories |
| `importance` | 0.7 | High importance_score memories |
| `keyword` | 0.4 | Full-text search match |
| `entity` | 0.5 | Memories mentioning relevant entities |

**Fusion Algorithm**:

```
1. Query each source in parallel
2. Early deduplication:
   - If same memory_id found by multiple sources:
     - Sum scores
     - Union source tags
3. Sort by combined score (descending)
4. Apply decay weighting
5. Return top_k results
```

**Configuration by Use Case**:

| Use Case | Recency | Semantic | Graph | Importance | Keyword |
|----------|---------|----------|-------|------------|---------|
| **Default** | 1.0 | 0.6 | 0.5 | 0.7 | 0.4 |
| **Research** | 0.5 | 1.0 | 0.7 | 0.8 | 0.6 |
| **Coding** | 0.8 | 0.5 | 0.4 | 1.0 | 0.5 |
| **Chat** | 1.0 | 0.4 | 0.3 | 0.5 | 0.3 |

### 6. Budget-Aware Context Building (from Context Builder Research)

**Purpose**: Build optimal context within token budget using greedy knapsack

**Algorithm**:

```
┌─────────────────────────────────────────────────────────────┐
│                 Greedy Knapsack Selection                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: candidates (from multi-source search), budget      │
│                                                             │
│   1. Reserve 15% budget for knowledge_summary               │
│   2. Calculate tokens for each candidate                    │
│   3. Sort candidates by score (descending)                  │
│                                                             │
│   selected = []                                             │
│   used_tokens = knowledge_summary_tokens                    │
│                                                             │
│   For each candidate (highest score first):                 │
│       If used_tokens + candidate.tokens ≤ budget:           │
│           selected.append(candidate)                        │
│           used_tokens += candidate.tokens                   │
│       Else:                                                 │
│           Skip and continue (don't stop!)                   │
│                                                             │
│   4. Sort selected chronologically                          │
│   5. Prepend knowledge_summary                              │
│                                                             │
│   Return Context(memories, tokens, sources_used)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Skip oversized memories instead of stopping - this allows fitting more smaller high-value memories.

**Return Type**:

```python
@dataclass
class Context:
    memories: List[Dict]           # Selected memories (chronological)
    knowledge_summary: str         # From knowledge_state
    total_tokens: int              # Total tokens used
    budget: int                    # Original budget
    sources_used: Dict[str, int]   # {"semantic": 5, "recency": 3}
```

### 7. Session Checkpoints (from Checkpoint Research)

**Purpose**: Store diff-based checkpoints instead of full conversation logs

**Diff-Based Storage**:

```python
@dataclass
class MemoryDiff:
    novel_facts: List[str]              # New facts learned
    updated_facts: List[Dict[str, str]] # {"old": ..., "new": ...}
    obsolete_facts: List[str]           # Facts to remove
    new_entities: List[str]             # New entities mentioned
    new_topics: List[str]               # New topics discussed
    is_novel: bool                      # Whether diff contains new info
```

**Schema**:

```sql
CREATE TABLE session_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES agent_sessions(id),
    checkpoint_number INT NOT NULL,
    
    -- Diff-based storage (NOT full conversation)
    novel_facts TEXT[] DEFAULT '{}',
    updated_facts JSONB DEFAULT '[]',  -- [{old: ..., new: ...}]
    obsolete_facts TEXT[] DEFAULT '{}',
    new_entities TEXT[] DEFAULT '{}',
    new_topics TEXT[] DEFAULT '{}',
    
    -- Summary
    summary TEXT,
    embedding VECTOR(1536),  -- For semantic search
    
    -- Compression
    compression_tier TEXT DEFAULT 'hot',
    
    -- Usage tracking
    usage_count INT DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(session_id, checkpoint_number)
);

CREATE INDEX idx_checkpoint_session ON session_checkpoints(session_id);
CREATE INDEX idx_checkpoint_embedding ON session_checkpoints 
    USING hnsw (embedding vector_cosine_ops);
```

**Checkpoint Creation Flow**:

```
1. Get current knowledge state summary
2. Analyze session messages
3. Extract:
   - NOVEL_FACTS: New information not in knowledge state
   - UPDATED_FACTS: Corrections to existing knowledge
   - OBSOLETE_FACTS: Outdated info
   - NEW_ENTITIES: People, places, technologies
   - NEW_TOPICS: Discussion subjects
4. Create checkpoint with diff only
5. Update knowledge state with new facts
```

### 8. Active Forgetting (from Checkpoint Research)

**Purpose**: LLM-driven cleanup to keep memory relevant

**Categories**:

| Category | Criteria | Action |
|----------|----------|--------|
| **KEEP** | High-value, frequently accessed, important | Boost importance_score |
| **COMPRESS** | Similar facts, redundant | Merge into single memory |
| **FORGET** | Outdated, superseded, noise, temporary | Soft delete |

**Algorithm**:

```
Every N hours (configurable):
1. Collect low-importance memories (bottom 20%, max 50)
2. For each memory, include:
   - Content
   - Memory type
   - Access count
   - Age
   - Related memories
3. Prompt LLM:
   "Categorize these memories as KEEP, COMPRESS, or FORGET.
    Consider: relevance, recency, redundancy, importance"
4. Apply recommendations:
   - KEEP: importance_score += 0.1
   - COMPRESS: Merge similar, delete originals
   - FORGET: Set deleted_at
5. Update knowledge state accordingly
```

**Safety**:

- Only processes low-importance memories
- Respects minimum access_count threshold
- Never forgets memories with `type="requirement"`
- Logs all forgetting decisions

### 9. Core Memory Operations (CRUD + Human-Memory)

**Create Operations**:
- `add()` - Single memory with automatic smart deduplication
- `add_batch()` - Bulk insert with batch embedding (100x faster)
- `import_from()` - Import from external sources

**Read Operations**:
- `get()` - Retrieve by ID with optional decay calculation
- `search()` - Multi-source hybrid search
- `build_context()` - Budget-aware context building
- `list_recent()` - Time-ordered recent memories (no search)
- `count()` - Count with filters (for pagination)
- `exists()` - Check memory existence
- `get_related()` - Find similar memories by vector proximity
- `get_knowledge_state()` - Get accumulated user knowledge
- `get_knowledge_summary()` - Get compact knowledge summary

**Update Operations**:
- `update()` - Modify content (regenerates embedding, checks dedup)
- `update_metadata()` - Update metadata only (fast, no re-embedding)
- `reinforce()` - Manually boost memory_strength and reset decay
- `tag()` - Add/update/remove tags
- `update_batch()` - Bulk updates

**Delete Operations**:
- `forget()` - Soft delete (sets deleted_at)
- `forget_all()` - Bulk soft delete with filters
- `purge()` - Hard delete (permanent removal)
- `restore()` - Undelete soft-deleted memory

**Human-Memory Operations**:
- `merge()` - Combine similar memories (preserves strongest/latest)
- `find_duplicates()` - Detect near-duplicate memories
- `deduplicate()` - Auto-merge duplicates
- `consolidate_session()` - Summarize session into key memories

**Implementation Notes**:
- All `add()` calls go through smart deduplication
- Corrections automatically supersede related memories
- Knowledge state updated on every write operation
- Soft deletes are excluded from search by default
- `reinforce()` increments memory_strength and resets last_accessed_at

### 10. Database Schema (Enhanced)

**Multi-Embedding Support**:

- Support for 1024, 1536, and 3072 dimensions
- Constraint ensures exactly one embedding column is populated
- `embedding_dim` field tracks which dimension is used
- `embedding_model` field tracks model name for migration

**Schema**:

```sql
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Memory types enum
CREATE TYPE memory_type AS ENUM ('preference', 'requirement', 'context', 'correction', 'entity');

-- Memory status enum
CREATE TYPE memory_status AS ENUM ('active', 'superseded', 'archived');

-- Compression tier enum
CREATE TYPE compression_tier AS ENUM ('hot', 'warm', 'cool', 'frozen');

CREATE TABLE agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id),
    user_id UUID NOT NULL REFERENCES users(id),
    session_id UUID NOT NULL REFERENCES agent_sessions(id),
    
    -- Content
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    summary TEXT,  -- For compressed memories
    
    -- Memory classification (from Fact Extraction research)
    memory_type memory_type DEFAULT 'context',
    entities TEXT[] DEFAULT '{}',
    status memory_status DEFAULT 'active',
    
    -- Supersession tracking
    supersedes UUID[],        -- IDs this memory supersedes
    superseded_by UUID,       -- ID of memory that superseded this
    
    -- Multi-model embedding support
    embedding_model TEXT NOT NULL DEFAULT 'custom',
    embedding_dim INT NOT NULL CHECK (embedding_dim IN (1024, 1536, 3072)),
    embedding_1024 VECTOR(1024),
    embedding_1536 VECTOR(1536),
    embedding_3072 VECTOR(3072),
    
    -- Constraint: exactly one embedding must be set
    CONSTRAINT check_single_embedding CHECK (
        (embedding_1024 IS NOT NULL)::int +
        (embedding_1536 IS NOT NULL)::int +
        (embedding_3072 IS NOT NULL)::int = 1
    ),
    
    -- Metadata and search
    metadata JSONB DEFAULT '{}'::jsonb,
    tags TEXT[] DEFAULT '{}',
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    
    -- Decay tracking (MemoryBank)
    memory_strength INT DEFAULT 1,
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INT DEFAULT 0,
    mention_count INT DEFAULT 1,  -- From Fact Extraction research
    importance_score FLOAT DEFAULT 0.5,
    
    -- Compression (from Checkpoint research)
    compression_tier compression_tier DEFAULT 'hot',
    compressed_at TIMESTAMPTZ,
    
    -- Summarization support
    is_summary BOOLEAN DEFAULT FALSE,
    source_memory_ids UUID[] DEFAULT '{}',
    
    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- Indices for each embedding dimension
CREATE INDEX idx_memory_embedding_1024 ON agent_memory 
    USING hnsw (embedding_1024 vector_cosine_ops) 
    WHERE embedding_1024 IS NOT NULL AND deleted_at IS NULL;

CREATE INDEX idx_memory_embedding_1536 ON agent_memory 
    USING hnsw (embedding_1536 vector_cosine_ops) 
    WHERE embedding_1536 IS NOT NULL AND deleted_at IS NULL;

CREATE INDEX idx_memory_embedding_3072 ON agent_memory 
    USING hnsw (embedding_3072 vector_cosine_ops) 
    WHERE embedding_3072 IS NOT NULL AND deleted_at IS NULL;

-- Additional indices
CREATE INDEX idx_memory_user ON agent_memory(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_session ON agent_memory(session_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_type ON agent_memory(memory_type) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_status ON agent_memory(status) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_created ON agent_memory(created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_importance ON agent_memory(importance_score DESC) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_text_search ON agent_memory USING gin(text_search) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_entities ON agent_memory USING gin(entities) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_tags ON agent_memory USING gin(tags) WHERE deleted_at IS NULL;

-- Relations table (for graph traversal)
CREATE TABLE memory_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES agent_memory(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES agent_memory(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,  -- causes, relates_to, contradicts, supersedes
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(source_id, target_id, relation_type)
);

CREATE INDEX idx_relation_source ON memory_relations(source_id);
CREATE INDEX idx_relation_target ON memory_relations(target_id);
CREATE INDEX idx_relation_type ON memory_relations(relation_type);

-- Knowledge state table (from Checkpoint research)
CREATE TABLE user_knowledge_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) UNIQUE,
    agent_id UUID NOT NULL REFERENCES agents(id),
    
    -- Accumulated knowledge
    facts JSONB DEFAULT '{}',           -- fact_hash -> fact_text
    entities TEXT[] DEFAULT '{}',       -- Known entities
    topics TEXT[] DEFAULT '{}',         -- Known topics  
    preferences JSONB DEFAULT '{}',     -- Extracted preferences
    
    -- Statistics
    total_memories INT DEFAULT 0,
    active_memories INT DEFAULT 0,
    superseded_memories INT DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_knowledge_user ON user_knowledge_state(user_id);

-- Session checkpoints table (from Checkpoint research)
CREATE TABLE session_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES agent_sessions(id),
    checkpoint_number INT NOT NULL,
    
    -- Diff-based storage
    novel_facts TEXT[] DEFAULT '{}',
    updated_facts JSONB DEFAULT '[]',
    obsolete_facts TEXT[] DEFAULT '{}',
    new_entities TEXT[] DEFAULT '{}',
    new_topics TEXT[] DEFAULT '{}',
    
    -- Summary and embedding
    summary TEXT,
    embedding_1536 VECTOR(1536),  -- For semantic search
    
    -- Compression
    compression_tier compression_tier DEFAULT 'hot',
    
    -- Usage tracking
    usage_count INT DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(session_id, checkpoint_number)
);

CREATE INDEX idx_checkpoint_session ON session_checkpoints(session_id);
CREATE INDEX idx_checkpoint_embedding ON session_checkpoints 
    USING hnsw (embedding_1536 vector_cosine_ops) 
    WHERE embedding_1536 IS NOT NULL;
```

### 11. Embedding Batching Service

**Purpose**: Batch embedding API calls to improve performance (100x faster than serial)

**Key Features**:

- **Async queue**: Collects embedding requests
- **Batching**: Groups requests into batches (default: 100)
- **Timeout**: Processes batch after timeout (default: 100ms) even if not full
- **LRU cache**: Caches recent embeddings to avoid duplicate API calls
- **Automatic**: Transparent to user - works with concurrent `add()` calls

### 12. Query Timeouts & Fallback

**Purpose**: Prevent long-running queries from blocking the system

**Implementation**:

- Set `statement_timeout` per query using `SET LOCAL`
- Default timeout: 5000ms (configurable)
- On timeout: Fallback to semantic-only search (faster, no keyword component)
- Return partial results if fallback succeeds

### 13. Background Maintenance

**Purpose**: Keep memory store healthy and performant

**Tasks**:

- **Decay updates**: Reduce importance_score for stale memories
- **Compression**: Apply compression tiers to aged memories
- **Active forgetting**: LLM-driven cleanup (configurable)
- **Soft-delete cleanup**: Mark very old, unimportant memories as deleted
- **Hard delete**: Permanently remove old soft-deleted records (>90 days)
- **Orphan cleanup**: Remove relations pointing to deleted memories
- **Knowledge state sync**: Ensure knowledge state matches active memories

---

## Dependencies

```toml
[project]
name = "engram"
version = "0.1.0"
description = "AI Memory Layer for LLM Applications"
dependencies = [
    "asyncpg>=0.29.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "pgvector>=0.2.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]
monitoring = [
    "prometheus-client>=0.19.0",  # Optional metrics
]
```

---

## Docker Setup

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: engram-db
    environment:
      POSTGRES_DB: engram
      POSTGRES_USER: engram
      POSTGRES_PASSWORD: engram
    ports:
      - "5432:5432"
    volumes:
      - engram_data:/var/lib/postgresql/data
      - ./engram/db/migrations:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U engram"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  engram_data:
```

---

## Production Configuration

### Recommended Settings

**Connection Pool**:

- `pool_size`: 20 (base connections)
- `max_overflow`: 10 (burst capacity)
- `pool_timeout`: 30s (wait for connection)
- `pool_recycle`: 3600s (recycle connections)

**Search**:

- `query_timeout_ms`: 5000 (5 second timeout)
- `enable_fallback`: True (fallback to semantic-only)
- `default_limit`: 10 (results per query)

**Deduplication** (from Fact Extraction research):

- `duplicate_threshold`: 0.90 (skip if above)
- `update_threshold`: 0.78 (merge if above)

**Context Building** (from Context Builder research):

- `default_token_budget`: 4000
- `knowledge_summary_budget_pct`: 0.15 (15% for summary)
- `default_source_weights`: See table above

**Embedding**:

- `batch_size`: 100 (batch API calls)
- `batch_timeout_ms`: 100 (max wait for batch)
- `cache_size`: 10000 (LRU cache entries)

**Compression** (from Checkpoint research):

- `hot_max_age_days`: 7
- `warm_max_age_days`: 30
- `cool_max_age_days`: 90

**Maintenance**:

- `cleanup_interval_hours`: 24 (daily cleanup)
- `decay_threshold`: 0.1 (delete below this score)
- `soft_delete_grace_days`: 90 (keep soft-deleted for 90 days)
- `active_forgetting_enabled`: False (opt-in, requires LLM)
- `forgetting_interval_hours`: 168 (weekly if enabled)

---

## What's Deferred (Future Releases)

- PII detection and handling (separate implementation planned)
- Hot/Cold partitioning (performance optimization)
- Transactional outbox pattern (reliability for actions)
- Multi-tenancy (schema-per-agent)
- Streaming memory updates (real-time)
- Cross-agent memory sharing
- Prometheus metrics integration (optional dependency)
- Custom retrieval source plugins

---

## Success Criteria

1. User can `pip install engram` and have working memory in under 5 minutes
2. Full CRUD operations (create, read, update, delete) work intuitively
3. **Smart deduplication** automatically prevents duplicate memories (two-tier threshold)
4. **Memory types** enable correction handling and supersession
5. Hybrid search with decay returns relevant results within 200ms for 100k memories
6. **Multi-source retrieval** combines recency, semantic, graph, and importance signals
7. **Budget-aware context building** optimally fills token budget using knapsack
8. **Knowledge state** provides fast access to accumulated user facts
9. Graph traversal completes 2-hop queries in under 300ms (Graphiti benchmark)
10. Memory decay correctly prioritizes recent + frequently accessed memories
11. **Compression tiers** reduce storage for old memories
12. Human-memory operations (merge, deduplicate, reinforce) work naturally
13. **Session checkpoints** store diff-based snapshots efficiently
14. Sessions persist across restarts with optional summarization
15. Query timeouts prevent long-running queries from blocking system
16. Batched embeddings improve throughput by 100x for concurrent operations
17. Bulk operations (add_batch, update_batch) handle 1000+ memories efficiently
18. Health checks provide visibility into system status
19. Background maintenance keeps memory store healthy
20. **Active forgetting** (opt-in) uses LLM to intelligently prune memories
21. Clear examples for OpenAI, Anthropic, and local models

---

## Production Readiness Checklist

- [x] Multi-embedding dimension support (1024, 1536, 3072)
- [x] Query timeouts with fallback strategies
- [x] Batched embedding generation with caching
- [x] Connection pool monitoring and alerts
- [x] Retry logic for transient failures
- [x] Health check endpoints
- [x] Background memory cleanup
- [x] Embedding model migration tool
- [x] Performance monitoring and slow query logging
- [x] Error handling and recovery
- [x] Structured logging
- [x] Production configuration documentation
- [x] **Smart deduplication with two-tier thresholds** (from Fact Extraction)
- [x] **Memory types with correction supersession** (from Fact Extraction)
- [x] **User knowledge state management** (from Checkpoint)
- [x] **Multi-resolution compression tiers** (from Checkpoint)
- [x] **Session checkpoints with diff-based storage** (from Checkpoint)
- [x] **Multi-source retrieval with configurable weights** (from Context Builder)
- [x] **Budget-aware context building with greedy knapsack** (from Context Builder)
- [x] **Active forgetting (opt-in LLM-driven cleanup)** (from Checkpoint)
