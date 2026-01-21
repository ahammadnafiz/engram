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

# Initialize with user's embedding function
async def my_embed(text: str) -> list[float]:
    return await openai.embeddings.create(input=text, model="text-embedding-3-small")

memory = Engram(
    database_url="postgresql+asyncpg://user:pass@localhost/engram",
    embedding_fn=my_embed,
    embedding_dim=1536,
    agent_name="my-assistant"
)

# === BASIC MEMORY OPERATIONS ===
await memory.add("User prefers dark mode", user_id="user_123", metadata={"type": "preference"})
results = await memory.search("user preferences", user_id="user_123", limit=5)

# === SESSION CONTEXT ===
async with memory.session(user_id="user_123") as session:
    await session.add("User asked about Python")
    context = await session.get_context("What did they ask about?", limit=10)

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
│   │   ├── store.py            # Memory CRUD operations
│   │   ├── decay.py            # Memory decay scoring (MemoryBank)
│   │   └── search.py           # Hybrid search (RRF + decay)
│   ├── graph/
│   │   ├── __init__.py
│   │   └── traversal.py        # Graph traversal queries
│   ├── session/
│   │   ├── __init__.py
│   │   ├── manager.py          # Session lifecycle
│   │   └── summarizer.py       # Optional summarization pipeline
│   └── types.py                # Type definitions
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── test_memory.py
│   ├── test_decay.py
│   ├── test_search.py
│   ├── test_graph.py
│   └── test_session.py
└── examples/
    ├── openai_chatbot.py       # OpenAI integration example
    ├── anthropic_chatbot.py    # Anthropic integration example
    ├── ollama_local.py         # Local model example
    └── graph_reasoning.py      # Multi-hop graph example
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
    embedding_1024 VECTOR(1024),
    embedding_1536 VECTOR(1536),
    embedding_3072 VECTOR(3072),
    
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

CREATE INDEX idx_memory_embedding_1536 ON agent_memory 
    USING hnsw (embedding_1536 vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);

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

| # | Task | Description | Status |
|---|------|-------------|--------|
| 1 | **setup-project** | Create project structure with pyproject.toml, README, and package scaffolding | ⬜ |
| 2 | **docker-setup** | Create docker-compose.yml with pgvector/pgvector:pg16 image | ⬜ |
| 3 | **db-schema** | Implement SQL migrations (agents, users, sessions, memory, relations) | ⬜ |
| 4 | **db-connection** | Build async connection pool with asyncpg and monitoring hooks | ⬜ |
| 5 | **memory-store** | Implement Memory CRUD operations (add, get, update, soft-delete) | ⬜ |
| 6 | **memory-decay** | Implement memory decay scoring with MemoryBank formula (0.995^hours) | ⬜ |
| 7 | **hybrid-search** | Implement hybrid search with weighted RRF + decay scoring | ⬜ |
| 8 | **graph-traversal** | Implement graph traversal queries using memory_relations (1-3 hops) | ⬜ |
| 9 | **session-manager** | Build session lifecycle management with auto-creation and expiry | ⬜ |
| 10 | **summarization** | Implement optional summarization pipeline for conversation consolidation | ⬜ |
| 11 | **client-api** | Create main Engram client class with simple public API | ⬜ |
| 12 | **tests** | Write tests for all modules (memory, search, decay, graph, session) | ⬜ |
| 13 | **examples** | Create integration examples for OpenAI, Anthropic, and Ollama | ⬜ |

---

## Success Criteria

1. **Quick Start**: User can `pip install engram` and have working memory in under 5 minutes
2. **Search Performance**: Hybrid search with decay returns relevant results within 200ms for 100k memories
3. **Graph Performance**: Graph traversal completes 2-hop queries in under 300ms (Graphiti benchmark)
4. **Decay Accuracy**: Memory decay correctly prioritizes recent + frequently accessed memories
5. **Persistence**: Sessions persist across restarts with optional summarization
6. **Examples**: Clear integration examples for OpenAI, Anthropic, and local models

---

## Deferred Features

These features are planned for future releases:

- **PII Detection**: Separate implementation planned for privacy handling
- **Hot/Cold Partitioning**: Performance optimization for large memory stores
- **Transactional Outbox**: Reliability pattern for action execution
- **Multi-tenancy**: Schema-per-agent for isolation
- **Streaming Updates**: Real-time memory synchronization
- **Cross-agent Sharing**: Shared memory between multiple agents

---

## References

- [Cognitive Architecture.md](./Cognitive%20Architecture.md) - Production-ready schema design
- [Converged Cognitive Architecture.md](./Converged%20Cognitive%20Architecture.md) - Architecture review and fixes
- [Memory Decay Scoring.md](./Memory%20Decay%20Scoring.md) - Decay formulas and research
- [AI Memory for Memory Decay Scoring and Graph Traversal Queries.md](./AI%20Memory%20for%20Memory%20Decay%20Scoring%20and%20Graph%20Traversal%20Queries.md) - Advanced patterns
- [claude-memory.md](./claude-memory.md) - Claude's memory architecture
- [chat-gpt-memory.md](./chat-gpt-memory.md) - ChatGPT's memory architecture

