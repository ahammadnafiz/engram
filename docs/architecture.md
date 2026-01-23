# Engram Architecture

> Technical design document for Engram - AI Memory Layer for LLM Applications

## Overview

Engram is an async-first, production-ready memory management library that provides persistent, searchable, intelligent memory for AI agents using PostgreSQL + pgvector.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                              │
│                     (Chatbots, Agents, RAG Systems)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENGRAM CLIENT                                  │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │ Memory Ops  │  │   Search    │  │    Graph    │  │   Session   │        │
│   │ add/update  │  │   hybrid    │  │  traverse   │  │   manage    │        │
│   │ get/forget  │  │  semantic   │  │   relate    │  │   context   │        │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│   MEMORY STORE      │  │  EMBEDDING SERVICE  │  │     LLM SERVICE         │
│                     │  │                     │  │                         │
│ • CRUD operations   │  │ • Vector generation │  │ • Fact extraction       │
│ • Search queries    │  │ • Batch embedding   │  │ • Memory operations     │
│ • Importance mgmt   │  │ • LRU caching       │  │ • Summarization         │
└─────────────────────┘  └─────────────────────┘  └─────────────────────────┘
              │                       │                       │
              │          ┌────────────┴────────────┐          │
              │          ▼                         ▼          │
              │  ┌─────────────────┐  ┌─────────────────┐     │
              │  │ EMBEDDING       │  │ LLM PROVIDER    │     │
              │  │ PROVIDER        │  │ REGISTRY        │     │
              │  │ REGISTRY        │  │                 │     │
              │  │                 │  │ • OpenAI        │     │
              │  │ • OpenAI        │  │ • Anthropic     │     │
              │  │ • Cohere        │  │ • Ollama        │     │
              │  │ • Sentence-TF   │  │ • Groq          │     │
              │  │ • Ollama        │  │ • LiteLLM       │     │
              │  │ • HuggingFace   │  └─────────────────┘     │
              │  └─────────────────┘                          │
              │                                               │
              ▼                                               │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POSTGRES STORAGE                                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     CONNECTION POOL (asyncpg)                       │   │
│   │                     min=2, max=10 connections                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POSTGRESQL + PGVECTOR                               │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐     │
│   │  agent_memory   │  │ memory_relations│  │    agent_sessions       │     │
│   │                 │  │                 │  │                         │     │
│   │ • content       │  │ • source_id     │  │ • session_id            │     │
│   │ • embedding     │  │ • target_id     │  │ • agent_id              │     │
│   │ • importance    │  │ • relation_type │  │ • metadata              │     │
│   │ • metadata      │  │ • weight        │  │ • started_at            │     │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────┘     │
│                                                                             │
│   Indexes: HNSW (vector), GIN (full-text), B-tree (lookups)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Engram Client (`client.py`)

The main entry point for all operations. Orchestrates all services.

```python
class Engram:
    """Main client - facade pattern over internal services."""
    
    _storage: PostgresStorage      # Database connection
    _memory_store: MemoryStore     # CRUD operations
    _embedding: EmbeddingService   # Vector generation
    _graph: GraphTraversal         # Relation operations
    _sessions: SessionManager      # Session lifecycle
    _health: HealthChecker         # Diagnostics
```

**Key Design Decisions:**
- Async context manager for safe resource cleanup
- Lazy initialization of services
- Single connection pool shared across services

### 2. Memory Store (`memory/store.py`)

Handles all memory CRUD operations and search.

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY STORE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  add(content, agent_id, ...)                                    │
│    │                                                            │
│    ├─► Generate embedding via EmbeddingService                  │
│    ├─► Auto-create agent/user if not exists                     │
│    └─► INSERT with ON CONFLICT handling                         │
│                                                                 │
│  search(query, agent_id, ...)                                   │
│    │                                                            │
│    ├─► Generate query embedding                                 │
│    ├─► Execute hybrid_search.sql                                │
│    │     • Vector similarity (cosine)                           │
│    │     • Keyword matching (ts_rank)                           │
│    │     • Time decay (exponential)                             │
│    │     • Importance weighting                                 │
│    └─► Return ranked SearchResults                              │
│                                                                 │
│  reinforce(memory_id, boost)                                    │
│    │                                                            │
│    └─► UPDATE importance = MIN(1.0, importance + boost)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Embedding Service (`embedding/service.py`)

Wraps embedding providers with caching and batching.

```
┌─────────────────────────────────────────────────────────────────┐
│                     EMBEDDING SERVICE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    LRU CACHE                            │    │
│  │            (default: 1000 entries)                      │    │
│  │                                                         │    │
│  │  "User likes Python" → [0.012, -0.034, 0.056, ...]      │    │
│  │  "User works in AI"  → [0.023, -0.045, 0.067, ...]      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼ cache miss                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 EMBEDDING PROVIDER                      │    │
│  │                                                         │    │
│  │  • OpenAI (text-embedding-3-small/large, ada-002)       │    │
│  │  • Sentence Transformers (all-MiniLM, etc.)             │    │
│  │  • Cohere (embed-english-v3.0)                          │    │
│  │  • Ollama (nomic-embed-text, etc.)                      │    │
│  │  • HuggingFace Inference API                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- LRU cache prevents redundant API calls
- Batch embedding for efficiency
- Auto-detection of embedding dimension

### 4. LLM Service (`llm/service.py`)

High-level LLM operations for memory intelligence.

```
┌─────────────────────────────────────────────────────────────────┐
│                       LLM SERVICE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  extract_facts(user_msg, bot_msg, history)                      │
│    │                                                            │
│    └─► Prompt engineering to extract atomic facts               │
│        "I'm Nafiz, I work at AskTuring"                         │
│        → ["User's name is Nafiz", "User works at AskTuring"]    │
│                                                                 │
│  evaluate_memory_operation(new_fact, existing_memories)         │
│    │                                                            │
│    └─► Decide: ADD, UPDATE, DELETE, or NOOP                     │
│        New: "User likes tea"                                    │
│        Existing: "User likes coffee"                            │
│        → DELETE old, ADD new (contradiction)                    │
│                                                                 │
│  summarize(text, max_length, style)                             │
│    │                                                            │
│    └─► Condense text for memory storage                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Graph Traversal (`graph/traversal.py`)

Memory relationship operations using recursive CTEs.

```
┌─────────────────────────────────────────────────────────────────┐
│                     GRAPH TRAVERSAL                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Memory Graph Example:                                          │
│                                                                 │
│     ┌──────────────┐         ┌──────────────┐                   │
│     │ User is      │ causes  │ User enrolled│                   │
│     │ learning AI  │────────►│ in ML course │                   │
│     └──────────────┘         └──────────────┘                   │
│            │                        │                           │
│            │ related_to             │ related_to                │
│            ▼                        ▼                           │
│     ┌──────────────┐         ┌──────────────┐                   │
│     │ User works   │         │ User studying│                   │
│     │ at AskTuring │         │ transformers │                   │
│     └──────────────┘         └──────────────┘                   │
│                                                                 │
│  traverse(start_id, max_depth=2, direction="outbound")          │
│    │                                                            │
│    └─► Recursive CTE to find connected memories                 │
│        Returns: [depth, content, relation_type, path]           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6. PostgreSQL Storage (`storage/postgres.py`)

Low-level database operations with connection pooling.

```
┌─────────────────────────────────────────────────────────────────┐
│                    POSTGRES STORAGE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Connection Pool (asyncpg)                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  min_size=2, max_size=10                                │    │
│  │                                                         │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        ┌─────┐         │    │
│  │  │conn1│ │conn2│ │conn3│ │conn4│  ...   │conn10│        │    │
│  │  └─────┘ └─────┘ └─────┘ └─────┘        └─────┘         │    │
│  │     ▲       ▲       ▲                                   │    │
│  │     │       │       │                                   │    │
│  │  request  request  request                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Methods:                                                       │
│  • execute(query, *args)     → Run INSERT/UPDATE/DELETE         │
│  • fetchone(query, *args)    → Single row                       │
│  • fetchall(query, *args)    → Multiple rows                    │
│  • fetchval(query, *args)    → Single value                     │
│                                                                 │
│  Schema Management:                                             │
│  • Auto-loads schema.sql on connect                             │
│  • Auto-adjusts vector dimension to match embedding provider    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

### Tables

```sql
-- Core memory storage (Two-Column System v2)
CREATE TABLE agent_memory (
    memory_id TEXT PRIMARY KEY,           -- mem_uuid
    agent_id TEXT NOT NULL,               -- Foreign key to agents
    user_id TEXT,                         -- Optional user scope
    session_id TEXT,                      -- Optional session scope
    
    -- LEGACY: Kept for backward compatibility (maps to fact)
    content TEXT NOT NULL,
    
    -- NEW: Two-column system
    fact TEXT NOT NULL,                   -- Extracted user fact (EMBEDDED)
    main_content TEXT,                    -- [USER]: msg\n[AI]: summary (NOT embedded)
    
    -- Embedding for fact column only
    embedding VECTOR(1536),               -- Auto-adjusted to match provider
    
    -- Full-text search vectors
    fact_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', fact)) STORED,
    main_content_tsv TSVECTOR GENERATED ALWAYS AS (
        CASE WHEN main_content IS NOT NULL 
        THEN to_tsvector('english', main_content) 
        ELSE NULL END
    ) STORED,
    
    -- Scoring factors
    importance FLOAT DEFAULT 0.5,         -- 0.0 to 1.0
    access_count INTEGER DEFAULT 0,       -- Usage tracking
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Memory relationships (graph)
CREATE TABLE memory_relations (
    source_memory_id TEXT REFERENCES agent_memory,
    target_memory_id TEXT REFERENCES agent_memory,
    relation_type TEXT DEFAULT 'related_to',
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    
    UNIQUE (source_memory_id, target_memory_id, relation_type)
);

-- Session tracking
CREATE TABLE agent_sessions (
    session_id TEXT PRIMARY KEY,
    agent_id TEXT REFERENCES agents,
    user_id TEXT REFERENCES users,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);
```

### Indexes

```sql
-- Vector similarity (HNSW for fast approximate search on fact embeddings)
CREATE INDEX idx_memory_embedding ON agent_memory 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search on fact
CREATE INDEX idx_memory_fact_tsv ON agent_memory USING GIN (fact_tsv);

-- Full-text search on main_content (for fallback search)
CREATE INDEX idx_memory_main_content_tsv ON agent_memory USING GIN (main_content_tsv) 
    WHERE main_content IS NOT NULL;

-- Trigram for fuzzy matching on fact
CREATE INDEX idx_memory_fact_trgm ON agent_memory USING GIN (fact gin_trgm_ops);

-- Compound indexes for filtering
CREATE INDEX idx_memory_agent_user ON agent_memory(agent_id, user_id);

-- Prevent duplicate facts per agent+user
CREATE UNIQUE INDEX idx_unique_memory_fact 
    ON agent_memory(agent_id, COALESCE(user_id, ''), fact);
```

## Hybrid Search Algorithm

The core differentiator of Engram is its hybrid search combining multiple signals:

```
┌─────────────────────────────────────────────────────────────────┐
│                      HYBRID SEARCH                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: "What does Nafiz do for work?"                          │
│                                                                 │
│  Step 1: Generate Signals                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Semantic: embedding <-> query_embedding (cosine)        │    │
│  │ Keyword:  content_tsv @@ plainto_tsquery(query)         │    │
│  │ Recency:  decay_rate ^ hours_since_access               │    │
│  │ Importance: importance column (0.0 - 1.0)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Step 2: Rank Each Signal                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ semantic_rank = ROW_NUMBER() ORDER BY cosine_sim DESC   │    │
│  │ keyword_rank  = ROW_NUMBER() ORDER BY ts_rank DESC      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Step 3: Reciprocal Rank Fusion (RRF)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                         │    │
│  │  RRF(rank) = 1 / (k + rank)    where k = 60             │    │
│  │                                                         │    │
│  │  final_score = w_semantic × RRF(semantic_rank)          │    │
│  │              + w_keyword  × RRF(keyword_rank)           │    │
│  │              × time_decay × importance                  │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Default Weights:                                               │
│  • semantic_weight = 0.5                                        │
│  • keyword_weight  = 0.3                                        │
│  • recency_weight  = 0.1                                        │
│  • importance_weight = 0.1                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SQL Implementation (`sql/hybrid_search.sql`)

```sql
-- Two-column hybrid search: searches fact, returns both fact and main_content
WITH semantic_search AS (
    SELECT memory_id, fact, main_content, importance, last_accessed_at,
           1 - (embedding <=> $1) as semantic_score,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1) as semantic_rank
    FROM agent_memory
    WHERE agent_id = $2 AND embedding IS NOT NULL
),
keyword_search AS (
    SELECT memory_id,
           ts_rank(fact_tsv, plainto_tsquery($3)) as keyword_score,
           ROW_NUMBER() OVER (ORDER BY ts_rank(fact_tsv, plainto_tsquery($3)) DESC) as keyword_rank
    FROM agent_memory
    WHERE fact_tsv @@ plainto_tsquery($3) AND agent_id = $2
),
combined AS (
    SELECT 
        s.memory_id,
        s.fact AS content,      -- Return fact as content for API compatibility
        s.main_content,         -- Full conversation context
        -- RRF fusion
        (COALESCE(1.0 / (60 + s.semantic_rank), 0) * $4 +
         COALESCE(1.0 / (60 + k.keyword_rank), 0) * $5) 
        * calculate_decay(s.last_accessed_at) 
        * s.importance as score
    FROM semantic_search s
    LEFT JOIN keyword_search k USING (memory_id)
)
SELECT * FROM combined ORDER BY score DESC LIMIT $6;
```

### Two-Column Design Benefits

| Aspect | Before (v1) | After (v2) |
|--------|-------------|------------|
| **Embedded** | Full content | Only facts |
| **Cost** | High (long texts) | Low (concise facts) |
| **Context** | Lost after extraction | Preserved in `main_content` |
| **Search** | Content only | Fact + context returned |

## Memory Decay System

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY DECAY                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Formula: decay = decay_rate ^ hours_elapsed                    │
│                                                                 │
│  Default decay_rate = 0.995                                     │
│                                                                 │
│  Example:                                                       │
│  • 1 hour ago:   0.995^1   = 0.995 (99.5% relevance)            │
│  • 1 day ago:    0.995^24  = 0.886 (88.6% relevance)            │
│  • 1 week ago:   0.995^168 = 0.430 (43.0% relevance)            │
│  • 1 month ago:  0.995^720 = 0.027 (2.7% relevance)             │
│                                                                 │
│  Counteracted by:                                               │
│  • Reinforcement: engram.reinforce(memory_id, boost)            │
│  • Access: updates last_accessed_at on retrieval                │
│                                                                 │
│  Result: Frequently used memories stay relevant                 │
│          Unused memories fade naturally                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Provider System

### Registry Pattern

```python
# providers/registry.py
class ProviderRegistry(Generic[T]):
    """Dynamic provider registration and retrieval."""
    
    _providers: dict[str, type[T]]
    _aliases: dict[str, str]
    
    def register(self, name: str, provider: type[T], aliases: list[str] = [])
    def get(self, name: str, **kwargs) -> T
    def available_providers() -> list[str]
```

### Adding a New Provider

```python
# providers/embedding/builtin.py

@embedding_registry.register("my-provider", aliases=["mp"])
class MyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "default"):
        self._client = MyClient(api_key)
        self._model = model
    
    @property
    def dimension(self) -> int:
        return 768
    
    async def embed(self, text: str) -> list[float]:
        return await self._client.embed(text)
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._client.embed_batch(texts)
```

## Configuration System

```python
# core/config.py

class EngramSettings(BaseSettings):
    """Pydantic settings with validation."""
    
    # Database
    database_url: str = "postgresql://localhost:5432/engram"
    min_pool_size: int = 2
    max_pool_size: int = 10
    
    # Embedding
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int | None = None  # Auto-detect
    
    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    
    # Search weights (must sum to 1.0)
    semantic_weight: float = 0.5
    keyword_weight: float = 0.3
    recency_weight: float = 0.1
    importance_weight: float = 0.1
    
    # Validation
    @validator('max_pool_size')
    def validate_pool_sizes(cls, v, values):
        if v < values.get('min_pool_size', 2):
            raise ValueError('max_pool_size must be >= min_pool_size')
        return v
    
    class Config:
        env_prefix = "ENGRAM_"
        env_file = ".env"
```

## Error Handling

```python
# core/exceptions.py

EngramError                    # Base exception
├── DatabaseConnectionError    # Connection failures
├── StorageError              # Database operations
├── ValidationError           # Input validation
├── EmbeddingError            # Embedding generation
├── MemoryNotFoundError       # Memory lookup
├── SessionError              # Session management
├── GraphError                # Graph operations
└── ConfigurationError        # Configuration issues
```

## Directory Structure

```
src/engram/
├── __init__.py              # Public API exports
├── client.py                # Main Engram client
│
├── core/
│   ├── config.py            # Pydantic settings
│   ├── exceptions.py        # Error hierarchy
│   └── _types.py            # Type aliases
│
├── memory/
│   ├── store.py             # MemoryStore CRUD
│   └── models.py            # Memory, SearchQuery, SearchResult
│
├── embedding/
│   └── service.py           # EmbeddingService with caching
│
├── llm/
│   └── service.py           # LLMService for fact extraction
│
├── graph/
│   ├── traversal.py         # GraphTraversal operations
│   └── models.py            # Relation, TraversalResult
│
├── session/
│   ├── manager.py           # SessionManager
│   └── models.py            # Session model
│
├── storage/
│   └── postgres.py          # PostgresStorage with pooling
│
├── providers/
│   ├── registry.py          # Generic ProviderRegistry
│   ├── embedding/
│   │   ├── protocol.py      # EmbeddingProvider ABC
│   │   ├── registry.py      # embedding_registry
│   │   └── builtin.py       # OpenAI, Cohere, etc.
│   └── llm/
│       ├── protocol.py      # LLMProvider ABC
│       ├── registry.py      # llm_registry
│       └── builtin.py       # OpenAI, Anthropic, etc.
│
├── sql/
│   ├── schema.sql           # Database schema
│   ├── hybrid_search.sql    # Hybrid search query
│   ├── semantic_search.sql  # Pure semantic search
│   └── graph_traverse.sql   # Recursive CTE traversal
│
└── health/
    └── checker.py           # HealthChecker diagnostics
```

## Performance Considerations

### Connection Pooling
- Min 2, max 10 connections by default
- Connections reused across requests
- Automatic connection health checks

### Embedding Caching
- LRU cache (1000 entries default)
- Prevents redundant API calls
- Cache key: hash of text content

### Batch Operations
- `add_batch()` for multiple memories
- Single embedding API call for batch
- Transaction-based inserts

### Index Strategy
- HNSW for vector search (fast approximate)
- GIN for full-text search
- B-tree for lookups and filtering
- Compound indexes for common patterns

## Security

- No hardcoded credentials
- Environment variable configuration
- SQL parameterization (no injection)
- Connection string from secure config
- API keys via environment or settings

---

*Engram - AI Memory Layer for LLM Applications*

