# Engram MVP Implementation Plan

> **Goal**: Ship a working memory layer in 2-3 weeks that users can `pip install` and use immediately.

---

## MVP Philosophy

**Include**: Features that provide immediate value with minimal complexity  
**Defer**: Features that add complexity without blocking core use cases  
**Principle**: "Make it work, make it right, make it fast" - we're at "make it work"

---

## What's IN the MVP

| Feature | Why It's Essential |
|---------|-------------------|
| Basic CRUD | Core functionality - can't ship without it |
| Hybrid Search | Vector + keyword + decay = the value proposition |
| Memory Decay | Differentiator from simple vector stores |
| Embedding Batching | 100x performance gain, trivial to implement |
| Basic Graph | Enables relationship queries |
| Simple Sessions | Users expect session context |
| Health Checks | Production requirement |
| Content-Hash Dedup | Prevents exact duplicates (simple) |

## What's DEFERRED

| Feature | Why Deferred | When to Add |
|---------|--------------|-------------|
| Smart Deduplication (two-tier) | Adds complexity, basic hash dedup sufficient | Phase 2 |
| Memory Types | Can use metadata.type for now | Phase 2 |
| Correction Supersession | Advanced feature | Phase 2 |
| Knowledge State | Separate table sync is complex | Phase 2 |
| Compression Tiers | Optimization, not blocking | Phase 3 |
| Multi-Source Search | Simple hybrid is sufficient | Phase 2 |
| Budget-Aware Context | Users can slice results themselves | Phase 2 |
| Session Checkpoints | Simple session.get_all() sufficient | Phase 3 |
| Active Forgetting | Requires LLM, optional feature | Phase 3 |

---

## MVP Architecture

```mermaid
flowchart TB
    subgraph UserApp [User Application]
        EmbedFn[Embedding Function]
    end
    
    subgraph EngramMVP [Engram MVP]
        Client[Engram Client]
        EmbedService[Embedding Service<br/>Batching + Cache]
        MemStore[Memory Store]
        Search[Hybrid Search<br/>Vector + Keyword + Decay]
        Graph[Graph Traversal]
        Sessions[Session Manager]
        Health[Health Checks]
    end
    
    subgraph Storage [PostgreSQL + pgvector]
        Memory[agent_memory]
        Relations[memory_relations]
        AgentSessions[agent_sessions]
    end
    
    EmbedFn --> EmbedService
    Client --> EmbedService
    Client --> MemStore
    Client --> Search
    Client --> Graph
    Client --> Sessions
    Client --> Health
    MemStore --> Memory
    Search --> Memory
    Graph --> Relations
    Sessions --> AgentSessions
```

---

## MVP API

```python
from engram import Engram

# Initialize - SIMPLE
memory = Engram(
    database_url="postgresql+asyncpg://user:pass@localhost/engram",
    embedding_fn=my_embed,
    embedding_dim=1536,
)

# === CRUD ===

# Create
memory_id = await memory.add(
    content="User prefers dark mode",
    user_id="user_123",
    metadata={"type": "preference"}  # Use metadata for types in MVP
)

# Read
results = await memory.search("user preferences", user_id="user_123", limit=5)
specific = await memory.get(memory_id)
recent = await memory.list_recent(user_id="user_123", limit=20)

# Update
await memory.update(memory_id, content="User strongly prefers dark mode")
await memory.reinforce(memory_id)  # Boost strength

# Delete
await memory.forget(memory_id)      # Soft delete
await memory.purge(memory_id)       # Hard delete

# === SEARCH (Hybrid) ===
results = await memory.search(
    query="user preferences",
    user_id="user_123",
    limit=10,
    decay_weight=0.25,      # How much to weight recency
    include_deleted=False,  # Default
)
# Returns: [{"id", "content", "score", "created_at", "metadata", ...}]

# === GRAPH ===
await memory.relate(source_id, target_id, relation_type="causes", weight=0.8)
related = await memory.traverse(start_id, relation_types=["causes"], max_hops=2)

# === SESSIONS ===
async with memory.session(user_id="user_123") as session:
    await session.add("User asked about Python")
    memories = await session.get_all()
    context = await session.search("Python question", limit=5)

# === HEALTH ===
status = await memory.health_check()
# {"database": "ok", "latency_ms": 2.5}

# === BULK (Bonus) ===
ids = await memory.add_batch([
    {"content": "Fact 1", "user_id": "user_123"},
    {"content": "Fact 2", "user_id": "user_123"},
])
```

---

## MVP Database Schema

```sql
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id TEXT NOT NULL,  -- User's ID from their system
    agent_id UUID NOT NULL REFERENCES agents(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(external_id, agent_id)
);

-- Sessions table
CREATE TABLE agent_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    agent_id UUID NOT NULL REFERENCES agents(id),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

-- Main memory table (SIMPLIFIED for MVP)
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id),
    user_id UUID NOT NULL REFERENCES users(id),
    session_id UUID REFERENCES agent_sessions(id),
    
    -- Content
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- For simple dedup
    
    -- Embedding (single dimension for MVP, pick one)
    embedding VECTOR(1536),  -- Or parameterize: VECTOR(%s)
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Search
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    
    -- Decay tracking
    memory_strength INT DEFAULT 1,
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INT DEFAULT 0,
    importance_score FLOAT DEFAULT 0.5,
    
    -- Lifecycle
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    
    -- Simple dedup constraint
    UNIQUE(user_id, content_hash)
);

-- Indices
CREATE INDEX idx_memory_embedding ON agent_memory 
    USING hnsw (embedding vector_cosine_ops) 
    WHERE deleted_at IS NULL;

CREATE INDEX idx_memory_user ON agent_memory(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_session ON agent_memory(session_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_created ON agent_memory(created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX idx_memory_text_search ON agent_memory USING gin(text_search) WHERE deleted_at IS NULL;

-- Relations table (simple)
CREATE TABLE memory_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES agent_memory(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES agent_memory(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, target_id, relation_type)
);

CREATE INDEX idx_relation_source ON memory_relations(source_id);
CREATE INDEX idx_relation_target ON memory_relations(target_id);
```

---

## MVP Project Structure

```
engram/
├── pyproject.toml
├── README.md
├── docker-compose.yml
├── engram/
│   ├── __init__.py             # Public exports
│   ├── client.py               # Main Engram class (~200 lines)
│   ├── config.py               # Configuration (~50 lines)
│   ├── exceptions.py           # Custom exceptions (~30 lines)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py       # Connection pool (~100 lines)
│   │   ├── models.py           # SQLAlchemy models (~150 lines)
│   │   └── migrations/
│   │       └── 001_initial.sql
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py            # CRUD operations (~200 lines)
│   │   ├── search.py           # Hybrid search (~150 lines)
│   │   └── decay.py            # Decay scoring (~50 lines)
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── service.py          # Batching + cache (~150 lines)
│   ├── graph/
│   │   ├── __init__.py
│   │   └── traversal.py        # Graph queries (~100 lines)
│   ├── session/
│   │   ├── __init__.py
│   │   └── manager.py          # Session context (~100 lines)
│   └── health/
│       ├── __init__.py
│       └── checker.py          # Health check (~50 lines)
├── tests/
│   ├── conftest.py
│   ├── test_memory.py
│   ├── test_search.py
│   ├── test_graph.py
│   └── test_session.py
└── examples/
    ├── quickstart.py           # 20-line example
    └── openai_chatbot.py       # Full example
```

**Estimated Total**: ~1,300 lines of code (excluding tests)

---

## MVP Implementation Order

### Week 1: Foundation

```
Day 1-2: Project Setup
├── pyproject.toml with dependencies
├── docker-compose.yml
├── Database schema migration
├── SQLAlchemy models
└── Connection pool

Day 3-4: Core Memory Store
├── add() with content-hash dedup
├── get() by ID
├── update() with re-embedding
├── forget() / purge()
└── list_recent()

Day 5: Embedding Service
├── Batch queue with timeout
├── LRU cache
└── Integration with store
```

### Week 2: Search & Features

```
Day 1-2: Hybrid Search
├── Vector similarity search
├── Keyword (tsvector) search  
├── RRF fusion
├── Decay weighting
└── Combined scoring

Day 3: Graph Traversal
├── relate() - create relation
├── traverse() - BFS/recursive CTE
└── get_related() - direct neighbors

Day 4: Sessions
├── Session context manager
├── session.add()
├── session.search()
├── session.get_all()

Day 5: Health & Polish
├── health_check()
├── Error handling
├── Logging
└── Basic retry logic
```

### Week 3: Testing & Docs

```
Day 1-2: Tests
├── Unit tests for each module
├── Integration tests
└── Performance benchmarks

Day 3-4: Documentation
├── README with quickstart
├── API documentation
├── Examples

Day 5: Release
├── Final testing
├── PyPI package
└── GitHub release
```

---

## MVP Search Implementation

The hybrid search is the core value - here's the algorithm:

```python
async def search(
    self,
    query: str,
    user_id: str,
    limit: int = 10,
    decay_weight: float = 0.25,
) -> List[Dict]:
    """
    Hybrid search: Vector + Keyword + Decay
    
    Final score = (vector_score * 0.5) + (keyword_score * 0.25) + (decay_score * 0.25)
    """
    
    # 1. Get query embedding
    query_embedding = await self.embedding_service.embed(query)
    
    # 2. Vector search (top 3x limit for RRF)
    vector_results = await self._vector_search(
        embedding=query_embedding,
        user_id=user_id,
        limit=limit * 3
    )
    
    # 3. Keyword search (top 3x limit for RRF)
    keyword_results = await self._keyword_search(
        query=query,
        user_id=user_id,
        limit=limit * 3
    )
    
    # 4. RRF Fusion
    fused = self._rrf_fusion(vector_results, keyword_results, k=60)
    
    # 5. Apply decay scoring
    scored = []
    for memory_id, rrf_score in fused.items():
        memory = self._memory_cache[memory_id]
        decay_score = self._calculate_decay(memory)
        
        final_score = (
            rrf_score * (1 - decay_weight) +
            decay_score * decay_weight
        )
        scored.append((memory, final_score))
    
    # 6. Sort and return top limit
    scored.sort(key=lambda x: x[1], reverse=True)
    return [self._format_result(m, s) for m, s in scored[:limit]]


def _rrf_fusion(self, vector_results, keyword_results, k=60):
    """Reciprocal Rank Fusion"""
    scores = {}
    
    for rank, (memory_id, _) in enumerate(vector_results):
        scores[memory_id] = scores.get(memory_id, 0) + 1 / (k + rank + 1)
    
    for rank, (memory_id, _) in enumerate(keyword_results):
        scores[memory_id] = scores.get(memory_id, 0) + 1 / (k + rank + 1)
    
    return scores


def _calculate_decay(self, memory) -> float:
    """MemoryBank decay formula"""
    hours_elapsed = (now() - memory.last_accessed_at).total_seconds() / 3600
    base_decay = 0.995 ** hours_elapsed
    
    # Boost for strength and access count
    strength_boost = min(memory.memory_strength / 10, 1.0)
    access_boost = min(memory.access_count / 100, 0.5)
    
    return base_decay * (1 + strength_boost + access_boost)
```

---

## MVP Configuration

```python
from pydantic_settings import BaseSettings

class EngramConfig(BaseSettings):
    # Database
    database_url: str
    pool_size: int = 10
    max_overflow: int = 5
    
    # Embedding
    embedding_dim: int = 1536
    batch_size: int = 50
    batch_timeout_ms: int = 100
    cache_size: int = 5000
    
    # Search
    default_limit: int = 10
    decay_weight: float = 0.25
    
    # Timeouts
    query_timeout_ms: int = 5000
    
    class Config:
        env_prefix = "ENGRAM_"
```

---

## MVP Success Criteria

| Criteria | Target | How to Verify |
|----------|--------|---------------|
| Install & run | < 5 minutes | Time a fresh setup |
| Add 1000 memories | < 10 seconds | Benchmark add_batch |
| Search latency | < 100ms p95 | Benchmark with 10k memories |
| Graph 2-hop | < 200ms | Benchmark traverse() |
| Memory usage | < 500MB for 100k | Monitor during load test |

---

## What Users Get with MVP

```python
# 1. Simple setup (30 seconds)
from engram import Engram
import openai

async def embed(texts):
    response = await openai.embeddings.create(input=texts, model="text-embedding-3-small")
    return [e.embedding for e in response.data]

memory = Engram(
    database_url="postgresql+asyncpg://localhost/engram",
    embedding_fn=embed,
    embedding_dim=1536,
)

# 2. Add memories naturally
await memory.add("User's name is Alice", user_id="user_1")
await memory.add("Alice prefers Python over JavaScript", user_id="user_1")
await memory.add("Alice works at TechCorp", user_id="user_1")

# 3. Search with decay (recent memories ranked higher)
results = await memory.search("What does Alice like?", user_id="user_1")
# Returns: Alice prefers Python... (most relevant + recent)

# 4. Build relationships
await memory.relate(mem1_id, mem2_id, "relates_to")
related = await memory.traverse(mem1_id, max_hops=2)

# 5. Session context
async with memory.session(user_id="user_1") as session:
    await session.add("Alice asked about databases")
    context = await session.search("database question")
```

---

## After MVP: Phase 2 Features

Once MVP is stable, add in this order:

1. **Smart Deduplication** (two-tier thresholds)
2. **Memory Types** (preference, requirement, context, correction)
3. **Multi-Source Search** (configurable weights)
4. **Knowledge State** (accumulated facts per user)
5. **Budget-Aware Context Building** (greedy knapsack)

Each feature is **additive** - existing code doesn't break.

---

## Quick Start Example (for README)

```python
import asyncio
from engram import Engram

async def main():
    # Your embedding function (OpenAI, Cohere, local, etc.)
    async def embed(texts):
        # Replace with your embedding API
        return [[0.1] * 1536 for _ in texts]  # Dummy
    
    # Initialize
    memory = Engram(
        database_url="postgresql+asyncpg://engram:engram@localhost/engram",
        embedding_fn=embed,
        embedding_dim=1536,
    )
    
    # Store memories
    await memory.add("User loves Python", user_id="alice")
    await memory.add("User works on ML projects", user_id="alice")
    
    # Search with decay
    results = await memory.search("programming preferences", user_id="alice")
    for r in results:
        print(f"[{r['score']:.2f}] {r['content']}")
    
    # Cleanup
    await memory.close()

asyncio.run(main())
```

---

## Timeline Summary

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Foundation | CRUD + Embedding service working |
| 2 | Features | Search + Graph + Sessions complete |
| 3 | Polish | Tests + Docs + PyPI release |

**Total: 3 weeks to usable library**

After that, iterate based on user feedback before adding Phase 2 features.

