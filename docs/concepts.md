# Core Concepts

Understanding how Engram manages AI memory.

## Memory Model

A **Memory** in Engram represents a piece of information that your AI should remember:

```python
@dataclass
class Memory:
    id: UUID                    # Unique identifier
    agent_id: str               # Which agent owns this memory
    content: str                # The actual content
    embedding: list[float]      # Vector representation
    importance: float           # 0.0-1.0, how important
    access_count: int           # Times accessed
    created_at: datetime        # When created
    last_accessed_at: datetime  # Last access time
    metadata: dict              # Flexible JSON metadata
    is_deleted: bool            # Soft delete flag
```

## Hybrid Search

Engram uses **Reciprocal Rank Fusion (RRF)** to combine multiple ranking signals:

```
final_score = w₁·semantic + w₂·keyword + w₃·decay + w₄·importance
```

### Search Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Semantic** | 0.40 | Vector similarity (cosine distance) |
| **Keyword** | 0.20 | Full-text search (BM25-like) |
| **Decay** | 0.25 | Recency + access frequency |
| **Importance** | 0.15 | Explicit importance score |

### How It Works

1. **Semantic Search**: Find memories with similar meaning using vector embeddings
2. **Keyword Search**: Find exact term matches using PostgreSQL full-text search
3. **Decay Scoring**: Recent and frequently accessed memories rank higher
4. **RRF Fusion**: Combine all signals with weighted reciprocal rank fusion

```sql
-- Simplified hybrid search
WITH semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> query_vec) as rank
    FROM agent_memory
),
keyword AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(search_vector, query)) as rank
    FROM agent_memory
)
SELECT id,
    (0.4 / (60 + semantic.rank)) +
    (0.2 / (60 + keyword.rank)) +
    (0.25 * decay_score) +
    (0.15 * importance) as score
FROM ...
```

## Memory Decay

Memories naturally decay over time, mimicking human memory:

```
decay_score = base_rate ^ hours_since_access
```

With `base_rate = 0.995`:

| Time Since Access | Decay Score |
|-------------------|-------------|
| 1 hour | 0.995 |
| 1 day | 0.887 |
| 1 week | 0.512 |
| 1 month | 0.023 |

### Reinforcement

Accessing a memory "reinforces" it:

```python
# This automatically updates last_accessed_at and access_count
memory = await engram.get(memory_id)

# Or explicitly reinforce
await engram.reinforce(memory_id)
```

## Graph Relationships

Memories can be connected to form a knowledge graph:

```python
# Create relationship
await engram.relate(
    source_id=memory_a.id,
    target_id=memory_b.id,
    relation_type="relates_to",  # or: causes, contradicts, supports, etc.
    strength=0.8
)
```

### Relationship Types

| Type | Description |
|------|-------------|
| `relates_to` | General association |
| `causes` | Causal relationship |
| `supports` | Supporting evidence |
| `contradicts` | Conflicting information |
| `is_part_of` | Hierarchical relationship |
| `follows` | Sequential relationship |

### Graph Traversal

Multi-hop traversal finds related memories:

```python
# Find all memories within 2 hops
related = await engram.traverse(
    start_id=memory.id,
    max_hops=2,
    min_strength=0.5,
    relation_types=["supports", "causes"]
)
```

This uses PostgreSQL recursive CTEs for efficient traversal:

```sql
WITH RECURSIVE graph AS (
    -- Base case
    SELECT target_id, 1 as depth, strength
    FROM memory_relations
    WHERE source_id = start_id
    
    UNION ALL
    
    -- Recursive case
    SELECT r.target_id, g.depth + 1, g.strength * r.strength
    FROM graph g
    JOIN memory_relations r ON g.target_id = r.source_id
    WHERE g.depth < max_hops
)
SELECT * FROM graph WHERE strength >= min_strength;
```

## Sessions

Sessions provide context continuity across conversations:

```python
async with engram.session(
    agent_id="assistant",
    user_id="user_123",
    ttl_hours=24
) as session:
    # Memories added here are linked to this session
    await session.add("User asked about Python")
    
    # Get context considers session history
    context = await session.get_context("What did they ask?")
```

### Session Lifecycle

1. **Create**: New session with unique ID and TTL
2. **Active**: Memories added are linked to session
3. **Expire**: After TTL, session becomes inactive
4. **Cleanup**: Expired sessions are periodically purged

## Agents

An **Agent** is a namespace for memories:

```python
# Different agents have separate memory spaces
await engram.add(content="...", agent_id="customer-support")
await engram.add(content="...", agent_id="sales-assistant")

# Search only within an agent's memories
await engram.search(query="...", agent_id="customer-support")
```

This allows multiple AI agents to share the same database while maintaining separate memory spaces.

## Architecture

Engram uses a **converged architecture** with PostgreSQL:

```
┌─────────────────────────────────────────────────────┐
│                    Engram Client                    │
├─────────────────────────────────────────────────────┤
│  Memory Store  │  Graph Traversal  │  Session Mgr   │
├─────────────────────────────────────────────────────┤
│                  PostgreSQL Storage                 │
├─────────────────────────────────────────────────────┤
│  pgvector  │  Full-Text  │  JSONB  │  Relations     │
└─────────────────────────────────────────────────────┘
```

### Why PostgreSQL?

| Feature | Benefit |
|---------|---------|
| **ACID** | Data integrity guaranteed |
| **pgvector** | Fast vector similarity search |
| **Full-text** | Built-in keyword search |
| **JSONB** | Flexible metadata |
| **Recursive CTEs** | Efficient graph traversal |
| **Single Database** | Simpler ops, lower latency |

All operations complete in a single database round-trip (~50ms typical).
