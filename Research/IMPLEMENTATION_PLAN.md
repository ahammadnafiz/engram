# Engram Library Implementation Plan

> **AI Memory Layer for LLM Applications**
> 
> Build a Python AI memory library that provides persistent, searchable memory for LLM applications using PostgreSQL with pgvector. Provider-agnostic (users bring their own embeddings), includes Docker setup, and handles core memory operations with session management, memory decay scoring, optional summarization, and graph traversal.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Research-Driven Design Decisions](#research-driven-design-decisions)
3. [Target API Design](#target-api-design)
4. [Project Structure](#project-structure)
5. [Implementation Details](#implementation-details)
6. [Dependencies](#dependencies)
7. [Docker Setup](#docker-setup)
8. [Implementation Checklist](#implementation-checklist)
9. [Success Criteria](#success-criteria)
10. [Deferred Features](#deferred-features)

---

## Architecture Overview

Engram is a drop-in memory layer for AI applications. Users import it, connect to PostgreSQL, provide their embedding function, and get persistent memory with hybrid search, memory decay, and graph traversal.

```mermaid
flowchart TB
    subgraph UserApp [User Application]
        LLM[LLM API Layer]
        EmbedFn[Embedding Function]
        SumFn[Summarization Function - Optional]
    end
    
    subgraph EngramLib [Engram Library]
        Client[Engram Client]
        MemStore[Memory Store]
        Decay[Memory Decay]
        Search[Hybrid Search + RRF]
        Graph[Graph Traversal]
        Sessions[Session Manager]
        Summarizer[Summarization Pipeline]
    end
    
    subgraph Storage [PostgreSQL + pgvector]
        Agents[agents]
        Users[users]
        AgentSessions[agent_sessions]
        Memory[agent_memory]
        Relations[memory_relations]
    end
    
    LLM --> Client
    EmbedFn --> Client
    SumFn -.-> Summarizer
    Client --> MemStore
    Client --> Search
    Client --> Sessions
    Client --> Graph
    MemStore --> Memory
    MemStore --> Decay
    Search --> Memory
    Search --> Decay
    Graph --> Relations
    Graph --> Memory
    Sessions --> AgentSessions
    Sessions --> Users
    Summarizer --> MemStore
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

**Formula:**
```
final_score = (0.6 × relevance) + (0.25 × recency) + (0.15 × importance)
```

### Graph Traversal (Graphiti Research)

Based on Graphiti's P95 300ms latency achievement:

- **Hybrid retrieval**: Vector + Graph + Keyword search
- **Hop limit**: 1-3 hops (configurable)
- **Relationship types**: Typed edges for precise reasoning

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
    cache_size=10000
)

# === BASIC MEMORY OPERATIONS (CRUD) ===

# Create
memory_id = await memory.add("User prefers dark mode", user_id="user_123", metadata={"type": "preference"})

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
    {"content": "Memory 1", "user_id": "user_123"},
    {"content": "Memory 2", "user_id": "user_123"}
])

# === SESSION CONTEXT ===
async with memory.session(user_id="user_123") as session:
    await session.add("User asked about Python")
    context = await session.get_context("What did they ask about?", limit=10)
    all_session_memories = await session.get_all()  # All memories in this session

# === MEMORY DECAY (automatic, configurable) ===
# Decay is applied automatically during search
# Access count and timestamps updated on retrieval
results = await memory.search(
    "user preferences", 
    user_id="user_123",
    decay_weight=0.25,  # How much recency matters (default: 0.25)
    include_stale=False  # Exclude memories with score < 0.1
)

# === GRAPH TRAVERSAL ===
# Find related memories through relationships
related = await memory.traverse(
    start_memory_id="mem_abc123",
    relation_types=["causes", "relates_to", "contradicts"],
    max_hops=2,
    min_weight=0.5
)

# Create relationships between memories
await memory.relate(
    source_id="mem_abc123",
    target_id="mem_xyz789",
    relation_type="causes",
    weight=0.8
)

# === OPTIONAL SUMMARIZATION ===
# User provides their own summarization function
async def my_summarize(messages: list[str]) -> str:
    return await openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize: {messages}"}]
    )

# Enable summarization for a session
async with memory.session(user_id="user_123", summarize_fn=my_summarize) as session:
    # After 10 messages, auto-summarize and store
    await session.add("Message 1")
    # ... more messages ...
    
    # Or manually trigger summarization
    await session.consolidate(last_n=10)  # Summarize last 10, store as single memory
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
│   │   ├── connection.py       # Async connection pool
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   └── migrations/
│   │       └── 001_initial.sql # Schema from architecture docs
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py            # Core CRUD operations (add, get, update, delete)
│   │   ├── operations.py       # Human-memory operations (merge, deduplicate)
│   │   ├── decay.py            # Memory decay scoring (MemoryBank)
│   │   └── search.py           # Hybrid search (RRF + decay + timeouts)
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
│   │   └── summarizer.py       # Optional summarization pipeline
│   ├── utils/
│   │   ├── __init__.py
│   │   └── retry.py            # Retry logic with exponential backoff
│   ├── health/
│   │   ├── __init__.py
│   │   └── checker.py          # Health check endpoints
│   ├── maintenance/
│   │   ├── __init__.py
│   │   └── cleanup.py          # Background maintenance jobs
│   ├── tools/
│   │   ├── __init__.py
│   │   └── migrate_embeddings.py # Embedding model migration tool
│   └── types.py                # Type definitions
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── test_memory.py          # CRUD operations
│   ├── test_operations.py      # merge, deduplicate, consolidate
│   ├── test_decay.py
│   ├── test_search.py
│   ├── test_graph.py
│   ├── test_session.py
│   ├── test_batching.py
│   ├── test_retry.py
│   └── test_health.py
└── examples/
    ├── openai_chatbot.py       # OpenAI integration example
    ├── anthropic_chatbot.py    # Anthropic integration example
    ├── ollama_local.py         # Local model example
    ├── graph_reasoning.py      # Multi-hop graph example
    ├── memory_operations.py    # CRUD and human-memory operations
    └── production_deployment.py # Production configuration example
```

---

## Implementation Details

### 1. Database Schema

Based on the Cognitive Architecture research:

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'archived'))
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id TEXT UNIQUE NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sessions table
CREATE TABLE agent_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    parent_session_id UUID REFERENCES agent_sessions(id),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '24 hours'),
    metadata JSONB DEFAULT '{}'::jsonb,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'expired', 'terminated'))
);

-- Memory table with decay tracking
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES agent_sessions(id),
    
    -- Content
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    
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
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    
    -- Decay tracking (MemoryBank)
    memory_strength INT DEFAULT 1,
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INT DEFAULT 0,
    importance_score FLOAT DEFAULT 0.5,
    
    -- Summarization support
    is_summary BOOLEAN DEFAULT FALSE,
    source_memory_ids UUID[] DEFAULT '{}',
    
    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- Memory relations for graph traversal
CREATE TABLE memory_relations (
    source_id UUID NOT NULL REFERENCES agent_memory(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES agent_memory(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    
    PRIMARY KEY (source_id, target_id, relation_type)
);

-- Indices
CREATE INDEX idx_sessions_active ON agent_sessions(agent_id, user_id, last_active_at) 
    WHERE status = 'active';

-- Indices for each embedding dimension
CREATE INDEX idx_memory_embedding_1024 ON agent_memory 
    USING hnsw (embedding_1024 vector_cosine_ops) 
    WHERE embedding_1024 IS NOT NULL;

CREATE INDEX idx_memory_embedding_1536 ON agent_memory 
    USING hnsw (embedding_1536 vector_cosine_ops) 
    WHERE embedding_1536 IS NOT NULL;

CREATE INDEX idx_memory_embedding_3072 ON agent_memory 
    USING hnsw (embedding_3072 vector_cosine_ops) 
    WHERE embedding_3072 IS NOT NULL;

CREATE INDEX idx_memory_text ON agent_memory USING GIN (text_search);
CREATE INDEX idx_memory_user ON agent_memory (agent_id, user_id, created_at DESC);
CREATE INDEX idx_memory_active ON agent_memory(agent_id) WHERE deleted_at IS NULL;

CREATE INDEX idx_relations_source ON memory_relations(source_id, relation_type) 
    WHERE deleted_at IS NULL;
CREATE INDEX idx_relations_target ON memory_relations(target_id, relation_type) 
    WHERE deleted_at IS NULL;
```

### 2. Memory Decay Scoring

```python
# engram/memory/decay.py

from datetime import datetime
from typing import Optional

class MemoryDecay:
    """MemoryBank-style memory decay implementation."""
    
    DEFAULT_DECAY_RATE = 0.995  # Per hour
    
    def __init__(self, decay_rate: float = DEFAULT_DECAY_RATE):
        self.decay_rate = decay_rate
    
    def calculate_recency_score(
        self, 
        last_accessed: datetime,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate recency score using exponential decay.
        
        Formula: recency_score = decay_rate ^ hours_elapsed
        
        Examples:
        - 0 hours: 1.0
        - 24 hours: ~0.886
        - 168 hours (1 week): ~0.433
        - 720 hours (30 days): ~0.025
        """
        current_time = current_time or datetime.utcnow()
        hours_elapsed = (current_time - last_accessed).total_seconds() / 3600
        return self.decay_rate ** hours_elapsed
    
    def calculate_memory_score(
        self,
        relevance_score: float,      # From vector similarity (0-1)
        recency_score: float,        # From decay calculation (0-1)
        importance_score: float,     # User-provided or default (0-1)
        weights: tuple[float, float, float] = (0.6, 0.25, 0.15)
    ) -> float:
        """
        Weighted memory scoring (MemoryBank formula).
        
        Default weights: relevance=0.6, recency=0.25, importance=0.15
        """
        w_rel, w_rec, w_imp = weights
        return (
            w_rel * relevance_score +
            w_rec * recency_score +
            w_imp * importance_score
        )
    
    def on_memory_access(self, memory) -> dict:
        """
        Update memory on access (MemoryBank behavior).
        
        - Increment strength (reduces future decay)
        - Reset last_accessed_at
        - Increment access_count
        """
        return {
            "memory_strength": memory.memory_strength + 1,
            "last_accessed_at": datetime.utcnow(),
            "access_count": memory.access_count + 1
        }
```

### 3. Hybrid Search with Decay

```sql
-- Hybrid search with decay scoring
WITH semantic AS (
    SELECT id, content, metadata,
           embedding_1536 <=> :vec as distance,
           1.0 - (embedding_1536 <=> :vec) as semantic_score,
           RANK() OVER (ORDER BY embedding_1536 <=> :vec) as rank_dense,
           -- Decay calculation inline
           POWER(0.995, EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 3600) as recency_score,
           importance_score
    FROM agent_memory
    WHERE agent_id = :aid 
      AND user_id = :uid 
      AND deleted_at IS NULL
    ORDER BY embedding_1536 <=> :vec
    LIMIT 50
),
keyword AS (
    SELECT id, content, metadata,
           ts_rank_cd(text_search, plainto_tsquery(:txt)) as keyword_score,
           RANK() OVER (ORDER BY ts_rank_cd(text_search, plainto_tsquery(:txt)) DESC) as rank_sparse,
           POWER(0.995, EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 3600) as recency_score,
           importance_score
    FROM agent_memory
    WHERE agent_id = :aid 
      AND user_id = :uid 
      AND text_search @@ plainto_tsquery(:txt)
      AND deleted_at IS NULL
    LIMIT 50
)
SELECT 
    COALESCE(s.id, k.id) as id,
    COALESCE(s.content, k.content) as content,
    COALESCE(s.metadata, k.metadata) as metadata,
    -- Combined score with decay
    (
        :w_rel * COALESCE(s.semantic_score, 0) +
        :w_rec * COALESCE(s.recency_score, k.recency_score) +
        :w_imp * COALESCE(s.importance_score, k.importance_score) +
        -- RRF component
        COALESCE(1.0 / (60 + s.rank_dense), 0) * 0.1 +
        COALESCE(1.0 / (60 + k.rank_sparse), 0) * 0.1
    ) as final_score
FROM semantic s
FULL OUTER JOIN keyword k ON s.id = k.id
WHERE (
    :w_rec * COALESCE(s.recency_score, k.recency_score) > :min_recency
    OR :include_stale = TRUE
)
ORDER BY final_score DESC
LIMIT :limit;
```

### 4. Graph Traversal

```python
# engram/graph/traversal.py

from typing import Optional
from sqlalchemy import text

class GraphTraversal:
    """Multi-hop graph traversal using memory_relations."""
    
    async def traverse(
        self,
        session,
        start_id: str,
        relation_types: Optional[list[str]] = None,
        max_hops: int = 2,
        min_weight: float = 0.5,
        limit: int = 20
    ) -> list[dict]:
        """
        Traverse graph from a starting memory.
        
        Uses recursive CTE for efficient multi-hop traversal.
        Graphiti achieves P95 < 300ms with similar approach.
        """
        query = text("""
            WITH RECURSIVE traversal AS (
                -- Base case: start node
                SELECT 
                    m.id,
                    m.content,
                    m.metadata,
                    0 as hop_depth,
                    ARRAY[m.id] as path,
                    1.0 as path_weight
                FROM agent_memory m
                WHERE m.id = :start_id
                  AND m.deleted_at IS NULL
                
                UNION ALL
                
                -- Recursive case: follow relations
                SELECT 
                    m.id,
                    m.content,
                    m.metadata,
                    t.hop_depth + 1,
                    t.path || m.id,
                    t.path_weight * r.weight
                FROM traversal t
                JOIN memory_relations r ON r.source_id = t.id
                JOIN agent_memory m ON m.id = r.target_id
                WHERE t.hop_depth < :max_hops
                  AND r.weight >= :min_weight
                  AND m.deleted_at IS NULL
                  AND NOT (m.id = ANY(t.path))  -- Prevent cycles
                  AND (:relation_types IS NULL OR r.relation_type = ANY(:relation_types))
            )
            SELECT DISTINCT ON (id)
                id, content, metadata, hop_depth, path, path_weight
            FROM traversal
            WHERE hop_depth > 0  -- Exclude start node
            ORDER BY id, path_weight DESC
            LIMIT :limit
        """)
        
        result = await session.execute(query, {
            "start_id": start_id,
            "max_hops": max_hops,
            "min_weight": min_weight,
            "relation_types": relation_types,
            "limit": limit
        })
        
        return [dict(row._mapping) for row in result]
```

### 5. Summarization Pipeline (Optional)

```python
# engram/session/summarizer.py

from typing import Callable, Optional
from datetime import datetime

class Summarizer:
    """Optional conversation summarization pipeline."""
    
    def __init__(
        self,
        summarize_fn: Callable[[list[str]], str],
        buffer_size: int = 10,
        auto_summarize: bool = False
    ):
        self.summarize_fn = summarize_fn
        self.buffer_size = buffer_size
        self.auto_summarize = auto_summarize
        self.buffer: list[dict] = []
    
    async def add_to_buffer(self, memory: dict) -> Optional[dict]:
        """
        Add memory to buffer. If buffer full and auto_summarize, consolidate.
        
        Returns summary memory if created, None otherwise.
        """
        self.buffer.append(memory)
        
        if self.auto_summarize and len(self.buffer) >= self.buffer_size:
            return await self.consolidate()
        
        return None
    
    async def consolidate(self, last_n: Optional[int] = None) -> dict:
        """
        Summarize buffered memories into single memory.
        
        ChatGPT-style: Only user content, lightweight format.
        """
        to_summarize = self.buffer[-last_n:] if last_n else self.buffer
        
        # Extract content (ChatGPT only summarizes user messages)
        contents = [m["content"] for m in to_summarize]
        
        # Call user-provided summarization function
        summary_text = await self.summarize_fn(contents)
        
        # Create summary memory
        summary_memory = {
            "content": summary_text,
            "metadata": {
                "is_summary": True,
                "source_count": len(to_summarize),
                "time_range": {
                    "start": to_summarize[0]["created_at"],
                    "end": to_summarize[-1]["created_at"]
                }
            },
            "source_memory_ids": [m["id"] for m in to_summarize],
            "is_summary": True,
            "created_at": datetime.utcnow()
        }
        
        # Clear buffer
        self.buffer = []
        
        return summary_memory
```

### 6. Production Enhancements

#### Multi-Embedding Support

Support for multiple embedding dimensions (1024, 1536, 3072) with automatic column selection:

```python
# engram/client.py
from typing import Literal

EmbeddingDim = Literal[1024, 1536, 3072]

class Engram:
    def __init__(
        self,
        database_url: str,
        embedding_fn: Callable[[list[str]], list[list[float]]],
        embedding_dim: EmbeddingDim,  # Required!
        **kwargs
    ):
        self.embedding_dim = embedding_dim
        self.embedding_column = f"embedding_{embedding_dim}"
        
        # Validate embedding dimension
        if embedding_dim not in [1024, 1536, 3072]:
            raise ValueError(f"Unsupported embedding_dim: {embedding_dim}")
```

#### Query Timeouts & Fallback

Prevent long-running queries from blocking the system:

```python
# engram/memory/search.py
class HybridSearch:
    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        timeout_ms: int = 5000,
        **kwargs
    ):
        """Hybrid search with timeout and fallback."""
        async with self.session.begin():
            # Set statement timeout
            await self.session.execute(
                text(f"SET LOCAL statement_timeout = '{timeout_ms}ms'")
            )
            
            try:
                # Attempt hybrid search
                return await self._hybrid_search_query(
                    query_text, query_vector, **kwargs
                )
            except asyncpg.QueryCanceledError:
                # Fallback to semantic-only (faster)
                logger.warning(f"Hybrid search timeout, falling back to semantic-only")
                return await self._semantic_only_search(query_vector, **kwargs)
```

#### Embedding Batching & Caching

Batch embedding API calls for 100x performance improvement:

```python
# engram/embedding/batch_service.py
class EmbeddingService:
    """Batched embedding generation with LRU caching."""
    
    def __init__(
        self,
        embedding_fn: Callable,
        batch_size: int = 100,
        batch_timeout_ms: int = 100,
        cache_size: int = 10000
    ):
        self.embedding_fn = embedding_fn
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._cache = lru_cache(maxsize=cache_size)(self._hash_and_cache)
        self.queue = asyncio.Queue()
        self.batch_task = asyncio.create_task(self._batch_worker())
    
    async def embed(self, text: str) -> list[float]:
        """Embed single text with automatic batching and caching."""
        # Check cache
        cache_key = self._hash_and_cache(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Queue for batching
        future = asyncio.Future()
        await self.queue.put((text, cache_key, future))
        return await future
    
    async def _batch_worker(self):
        """Background worker that batches requests."""
        while True:
            batch = []
            try:
                # Collect batch (up to batch_size or timeout)
                for _ in range(self.batch_size):
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.batch_timeout_ms / 1000
                    )
                    batch.append(item)
            except asyncio.TimeoutError:
                pass
            
            if not batch:
                continue
            
            # Generate embeddings (single API call!)
            texts = [item[0] for item in batch]
            try:
                embeddings = await self.embedding_fn(texts)
                
                # Resolve futures and cache
                for (text, cache_key, future), embedding in zip(batch, embeddings):
                    self._cache[cache_key] = embedding
                    future.set_result(embedding)
            except Exception as e:
                # Fail all futures in batch
                for _, _, future in batch:
                    future.set_exception(e)
```

#### Connection Pool Monitoring

Track connection pool health and prevent exhaustion:

```python
# engram/db/connection.py
class MonitoredEngine:
    """SQLAlchemy engine with monitoring."""
    
    def __init__(self, database_url: str, **kwargs):
        self.engine = create_async_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            **kwargs
        )
        
        # Start monitoring
        asyncio.create_task(self._monitor_pool())
    
    async def _monitor_pool(self):
        """Background task to update metrics."""
        while True:
            pool = self.engine.pool
            
            # Alert if pool exhausted
            if pool.checked_out() >= pool.size() + pool.overflow():
                logger.warning("Database pool exhausted!")
            
            await asyncio.sleep(10)
```

#### Error Recovery with Retry

Handle transient database failures automatically:

```python
# engram/utils/retry.py
def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True
):
    """Decorator for retrying async functions with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (asyncpg.ConnectionDoesNotExistError,
                        asyncpg.ConnectionFailureError) as e:
                    # Retryable errors
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/"
                            f"{max_attempts}), retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable, raise immediately
                    logger.error(f"{func.__name__} failed: {e}")
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator
```

#### Health Checks

Monitor system health for production deployments:

```python
# engram/health/checker.py
class HealthChecker:
    """Health check endpoints for monitoring."""
    
    def __init__(self, engram: Engram):
        self.engram = engram
    
    async def check_database(self) -> dict:
        """Check database connectivity and latency."""
        start = time.time()
        try:
            async with self.engram.db.session() as session:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1
            
            latency_ms = (time.time() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def full_health_check(self) -> dict:
        """Complete health assessment."""
        return {
            "database": await self.check_database(),
            "pool": await self.check_pool(),
            "search": await self.check_search_performance(),
            "timestamp": datetime.utcnow().isoformat()
        }
```

#### Background Maintenance

Keep memory store healthy and performant:

```python
# engram/maintenance/cleanup.py
class MemoryMaintenance:
    """Background maintenance for memory health."""
    
    def __init__(
        self,
        engram: Engram,
        decay_threshold: float = 0.1,
        cleanup_interval_hours: int = 24
    ):
        self.engram = engram
        self.decay_threshold = decay_threshold
        self.cleanup_interval_hours = cleanup_interval_hours
        self.task = asyncio.create_task(self._maintenance_loop())
    
    async def _run_maintenance(self):
        """Execute all maintenance tasks."""
        async with self.engram.db.session() as session:
            # 1. Decay low-importance memories
            await session.execute(text("""
                UPDATE agent_memory
                SET importance_score = importance_score * 0.95
                WHERE access_count = 0
                  AND created_at < NOW() - INTERVAL '30 days'
                  AND deleted_at IS NULL
            """))
            
            # 2. Soft delete stale memories
            deleted = await session.execute(text("""
                UPDATE agent_memory
                SET deleted_at = NOW()
                WHERE importance_score < :threshold
                  AND access_count = 0
                  AND created_at < NOW() - INTERVAL '180 days'
                  AND deleted_at IS NULL
                RETURNING id
            """), {"threshold": self.decay_threshold})
            
            # 3. Hard delete old soft-deleted records
            await session.execute(text("""
                DELETE FROM agent_memory
                WHERE deleted_at < NOW() - INTERVAL '90 days'
            """))
            
            await session.commit()
```

#### Embedding Model Migration Tool

Zero-downtime migration between embedding models:

```python
# engram/tools/migrate_embeddings.py
class EmbeddingMigrator:
    """Zero-downtime embedding model migration."""
    
    async def migrate(
        self,
        old_dim: int,
        new_dim: int,
        new_embedding_fn: Callable,
        batch_size: int = 100
    ):
        """
        Migrate embeddings from old to new model.
        
        Strategy:
        1. Add new embeddings alongside old
        2. Update indices to use new column
        3. Remove old embeddings after verification
        """
        old_col = f"embedding_{old_dim}"
        new_col = f"embedding_{new_dim}"
        
        # Count memories to migrate
        total = await self._count_memories_with_column(old_col)
        
        print(f"Migrating {total} memories from {old_col} to {new_col}")
        
        migrated = 0
        while migrated < total:
            # Fetch batch
            memories = await self._fetch_batch(old_col, batch_size)
            
            # Generate new embeddings (batched!)
            texts = [m["content"] for m in memories]
            new_embeddings = await self._batch_embed(texts, new_embedding_fn)
            
            # Store new embeddings
            for memory, embedding in zip(memories, new_embeddings):
                await self._store_embedding(
                    memory["id"], new_col, embedding, new_dim
                )
            
            migrated += len(memories)
            print(f"Progress: {migrated}/{total} ({migrated/total*100:.1f}%)")
        
        print("Migration complete!")
```

---

## Dependencies

```toml
[project]
name = "engram"
version = "0.1.0"
description = "AI Memory Layer for LLM Applications"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
keywords = ["ai", "memory", "llm", "postgresql", "pgvector"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

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

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
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

## Implementation Checklist

### Core Features

| # | Task | Description | Priority | Status |
|---|------|-------------|----------|--------|
| 1 | **setup-project** | Create project structure with pyproject.toml, README, and package scaffolding | P0 | ⬜ |
| 2 | **docker-setup** | Create docker-compose.yml with pgvector/pgvector:pg16 image | P0 | ⬜ |
| 3 | **db-schema** | Implement SQL migrations (agents, users, sessions, memory, relations) | P0 | ⬜ |
| 4 | **db-connection** | Build async connection pool with asyncpg and monitoring hooks | P0 | ⬜ |
| 5 | **memory-store** | Implement Memory CRUD operations (add, get, update, delete, restore) | P0 | ⬜ |
| 6 | **memory-operations** | Implement human-memory operations (merge, deduplicate, reinforce, tag) | P0 | ⬜ |
| 7 | **memory-decay** | Implement memory decay scoring with MemoryBank formula (0.995^hours) | P0 | ⬜ |
| 8 | **hybrid-search** | Implement hybrid search with weighted RRF + decay scoring | P0 | ⬜ |
| 9 | **graph-traversal** | Implement graph traversal queries using memory_relations (1-3 hops) | P0 | ⬜ |
| 10 | **session-manager** | Build session lifecycle management with auto-creation and expiry | P0 | ⬜ |
| 11 | **summarization** | Implement optional summarization pipeline for conversation consolidation | P1 | ⬜ |
| 12 | **client-api** | Create main Engram client class with simple public API | P0 | ⬜ |

### Production Enhancements

| # | Task | Description | Priority | Status |
|---|------|-------------|----------|--------|
| 14 | **multi-embedding-support** | Support multiple embedding dimensions (1024, 1536, 3072) with auto-detection | P0 | ⬜ |
| 15 | **query-timeouts** | Add statement timeouts and fallback strategies for long-running queries | P0 | ⬜ |
| 16 | **embedding-batching** | Implement batched embedding generation with caching layer | P0 | ⬜ |
| 17 | **connection-monitoring** | Add connection pool monitoring with metrics and alerts | P1 | ⬜ |
| 18 | **error-recovery** | Implement retry logic with exponential backoff for transient failures | P1 | ⬜ |
| 19 | **health-checks** | Database health endpoints and connectivity validation | P1 | ⬜ |
| 20 | **memory-cleanup** | Background job for memory decay, soft-delete cleanup, and archival | P2 | ⬜ |
| 21 | **migration-tooling** | Build embedding model migration tool with zero-downtime strategy | P1 | ⬜ |
| 22 | **performance-monitoring** | Add search latency tracking and slow query logging | P1 | ⬜ |

### Testing & Documentation

| # | Task | Description | Priority | Status |
|---|------|-------------|----------|--------|
| 23 | **tests** | Write tests for all modules (memory, search, decay, graph, session, batching, retry, health) | P0 | ⬜ |
| 24 | **examples** | Create integration examples for OpenAI, Anthropic, Ollama, and production deployment | P0 | ⬜ |
| 25 | **documentation** | Complete README with quickstart, API reference, and production guide | P0 | ⬜ |

---

## Success Criteria

### Functionality
1. **Quick Start**: User can `pip install engram` and have working memory in under 5 minutes
2. **Full CRUD**: All memory operations (create, read, update, delete, restore) work intuitively
3. **Human-Memory Operations**: Merge, deduplicate, reinforce, and tag operations work naturally
4. **Decay Accuracy**: Memory decay correctly prioritizes recent + frequently accessed memories
5. **Persistence**: Sessions persist across restarts with optional summarization

### Performance
6. **Search Performance**: Hybrid search with decay returns relevant results within 200ms for 100k memories
7. **Graph Performance**: Graph traversal completes 2-hop queries in under 300ms (Graphiti benchmark)
8. **Batching Efficiency**: Batched embeddings improve throughput by 100x for concurrent operations
9. **Bulk Operations**: add_batch and update_batch handle 1000+ memories efficiently

### Production Readiness
10. **Query Timeouts**: Timeouts prevent long-running queries from blocking system
11. **Error Recovery**: Transient failures are automatically retried with exponential backoff
12. **Health Monitoring**: Health checks provide visibility into system status
13. **Background Maintenance**: Automatic cleanup keeps memory store healthy
14. **Multi-Embedding Support**: Seamless support for 1024, 1536, and 3072 dimension embeddings

### Documentation
15. **Examples**: Clear integration examples for OpenAI, Anthropic, and local models
16. **Production Guide**: Comprehensive guide for production deployment and configuration

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

**Embedding**:
- `batch_size`: 100 (batch API calls)
- `batch_timeout_ms`: 100 (max wait for batch)
- `cache_size`: 10000 (LRU cache entries)

**Maintenance**:
- `cleanup_interval_hours`: 24 (daily cleanup)
- `decay_threshold`: 0.1 (delete below this score)
- `soft_delete_grace_days`: 90 (keep soft-deleted for 90 days)

### Production Example

```python
from engram import Engram
from engram.health import HealthChecker
from engram.maintenance import MemoryMaintenance

async def openai_embed_batch(texts: list[str]) -> list[list[float]]:
    """Batched embedding for performance."""
    import openai
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]

# Initialize with production settings
memory = Engram(
    database_url="postgresql+asyncpg://user:pass@localhost/engram",
    embedding_fn=openai_embed_batch,
    embedding_dim=1536,
    agent_name="production-chatbot",
    
    # Production configuration
    pool_size=20,
    max_overflow=10,
    query_timeout_ms=5000,
    enable_fallback=True,
    batch_size=100,
    cache_size=10000
)

# Start background maintenance
maintenance = MemoryMaintenance(memory, cleanup_interval_hours=24)

# Set up health checks
health = HealthChecker(memory)
health_status = await health.full_health_check()

if not health_status["database"]["status"] == "healthy":
    raise RuntimeError("Database unhealthy!")

# Normal operations
async with memory.session(user_id="user_123") as session:
    await session.add("User prefers dark mode")
    context = await session.get_context(
        "What are user's preferences?",
        limit=5,
        timeout_ms=3000
    )
```

---

## Deferred Features

These features are planned for future releases:

- **PII Detection**: Separate implementation planned for privacy handling
- **Hot/Cold Partitioning**: Performance optimization for large memory stores
- **Transactional Outbox**: Reliability pattern for action execution
- **Multi-tenancy**: Schema-per-agent for isolation
- **Streaming Updates**: Real-time memory synchronization
- **Cross-agent Sharing**: Shared memory between multiple agents
- **Prometheus Metrics**: Full Prometheus integration (optional dependency available)
- **Distributed Tracing**: OpenTelemetry integration for tracing
- **Read Replicas**: Support for read-only replicas for scaling reads
- **Sharding**: Horizontal scaling for massive memory stores

---

## References

- [Cognitive Architecture.md](./Cognitive%20Architecture.md) - Production-ready schema design
- [Converged Cognitive Architecture.md](./Converged%20Cognitive%20Architecture.md) - Architecture review and fixes
- [Memory Decay Scoring.md](./Memory%20Decay%20Scoring.md) - Decay formulas and research
- [AI Memory for Memory Decay Scoring and Graph Traversal Queries.md](./AI%20Memory%20for%20Memory%20Decay%20Scoring%20and%20Graph%20Traversal%20Queries.md) - Advanced patterns
- [claude-memory.md](./claude-memory.md) - Claude's memory architecture
- [chat-gpt-memory.md](./chat-gpt-memory.md) - ChatGPT's memory architecture

