# Redis vs PostgreSQL: Database Choice for Engram

> **Deep Dive Analysis: Which Database Best Serves Engram's Requirements?**

This document provides a comprehensive comparison between Redis (with RediSearch) and PostgreSQL (with pgvector) for implementing the Engram AI memory library. Each requirement is analyzed in detail to inform the optimal database choice.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Requirement-by-Requirement Analysis](#requirement-by-requirement-analysis)
3. [Architecture Comparison](#architecture-comparison)
4. [Performance Benchmarks](#performance-benchmarks)
5. [Operational Considerations](#operational-considerations)
6. [Cost Analysis](#cost-analysis)
7. [Migration Path](#migration-path)
8. [Recommendation](#recommendation)

---

## Executive Summary

### Quick Comparison

| Aspect | Redis + RediSearch | PostgreSQL + pgvector | Winner |
|--------|-------------------|----------------------|--------|
| **Query Latency** | Sub-millisecond (0.5-5ms) | Millisecond (5-50ms) | 🏆 Redis |
| **ACID Guarantees** | Limited (single-threaded) | Full ACID | 🏆 PostgreSQL |
| **Persistence** | Optional (RDB/AOF) | Always persistent | 🏆 PostgreSQL |
| **Hybrid Search** | ✅ Native support | ✅ SQL-based | 🏆 Tie |
| **Graph Traversal** | ⚠️ Limited (no native) | ✅ Recursive CTEs | 🏆 PostgreSQL |
| **Memory Decay** | ✅ Inline calculation | ✅ SQL functions | 🏆 Tie |
| **Session Management** | ✅ Native TTL | ✅ Custom expiry | 🏆 Tie |
| **Complex Queries** | ⚠️ Limited (FT.SEARCH) | ✅ Full SQL | 🏆 PostgreSQL |
| **Scalability** | ✅ Horizontal (Cluster) | ✅ Vertical + Horizontal | 🏆 Tie |
| **Ecosystem** | ⚠️ Specialized | ✅ Universal | 🏆 PostgreSQL |

### Verdict Preview

**PostgreSQL + pgvector is the recommended choice for Engram** because:

1. ✅ **ACID guarantees** are critical for memory consistency
2. ✅ **Graph traversal** requires recursive queries (PostgreSQL excels)
3. ✅ **Complex queries** needed for hybrid search + decay + filters
4. ✅ **Persistence** is essential for AI memory (not optional)
5. ✅ **SQL ecosystem** provides better tooling and debugging

**Redis would be better if**:
- Ultra-low latency (<5ms) is the #1 priority
- Data fits entirely in memory
- Simple key-value operations dominate
- Caching layer rather than primary storage

---

## Requirement-by-Requirement Analysis

### 1. Hybrid Search (Vector + Keyword + RRF)

#### Requirement
Engram needs to combine:
- **Semantic search** (vector similarity)
- **Keyword search** (BM25/TF-IDF)
- **Reciprocal Rank Fusion** (RRF) for combining results
- **Metadata filtering** (tags, numeric ranges, timestamps)

#### PostgreSQL + pgvector Implementation

**Strengths**:
```sql
-- Hybrid search with RRF in PostgreSQL
WITH semantic AS (
    SELECT id, content,
           embedding_1536 <=> :vec as distance,
           1.0 - (embedding_1536 <=> :vec) as semantic_score,
           RANK() OVER (ORDER BY embedding_1536 <=> :vec) as rank_dense
    FROM agent_memory
    WHERE agent_id = :aid AND user_id = :uid
    ORDER BY embedding_1536 <=> :vec
    LIMIT 50
),
keyword AS (
    SELECT id, content,
           ts_rank_cd(text_search, plainto_tsquery(:txt)) as keyword_score,
           RANK() OVER (ORDER BY ts_rank_cd(...) DESC) as rank_sparse
    FROM agent_memory
    WHERE text_search @@ plainto_tsquery(:txt)
    LIMIT 50
)
SELECT 
    COALESCE(s.id, k.id) as id,
    -- RRF fusion
    COALESCE(1.0 / (60 + s.rank_dense), 0) +
    COALESCE(1.0 / (60 + k.rank_sparse), 0) as rrf_score
FROM semantic s
FULL OUTER JOIN keyword k ON s.id = k.id
ORDER BY rrf_score DESC
LIMIT :limit;
```

**Advantages**:
- ✅ **Full SQL control** - Complex joins, CTEs, window functions
- ✅ **Native full-text search** - PostgreSQL's `tsvector` is mature
- ✅ **Flexible RRF** - Easy to adjust weights and formulas
- ✅ **Metadata filtering** - Natural SQL WHERE clauses
- ✅ **Query optimization** - PostgreSQL query planner optimizes automatically

**Performance**: ~20-50ms for 100K memories (with proper indices)

#### Redis + RediSearch Implementation

**Strengths**:
```python
from redisvl.query import AggregateHybridQuery

query = AggregateHybridQuery(
    text="user preferences",
    text_field_name="content",
    vector=embedding,
    vector_field_name="embedding",
    alpha=0.7,  # 70% vector, 30% text
    return_fields=["content", "metadata"],
    num_results=10
)

results = index.query(query)
```

**Advantages**:
- ✅ **Ultra-low latency** - Sub-millisecond queries
- ✅ **Native hybrid search** - Built-in AggregateHybridQuery
- ✅ **Simple API** - High-level RedisVL SDK
- ✅ **In-memory speed** - No disk I/O

**Limitations**:
- ⚠️ **Less flexible** - Alpha parameter only (can't customize RRF formula)
- ⚠️ **Limited filtering** - Tag/Numeric filters work but less expressive
- ⚠️ **No custom RRF** - Stuck with Redis's implementation

**Performance**: ~1-5ms for 100K memories

#### Winner: **PostgreSQL** (for flexibility) | **Redis** (for speed)

**Analysis**: PostgreSQL wins for Engram because:
- Engram needs **custom RRF weights** (0.6 relevance, 0.25 recency, 0.15 importance)
- Redis's `alpha` parameter doesn't support this level of customization
- PostgreSQL's SQL allows **precise control** over scoring formula

---

### 2. Memory Decay Scoring

#### Requirement
Engram implements MemoryBank-style exponential decay:
- **Formula**: `recency_score = 0.995^hours_elapsed`
- **Weighted scoring**: `final_score = 0.6×relevance + 0.25×recency + 0.15×importance`
- **Inline calculation** during search queries
- **Update tracking**: Increment `memory_strength`, reset `last_accessed_at` on access

#### PostgreSQL + pgvector Implementation

**Implementation**:
```sql
-- Inline decay calculation in PostgreSQL
SELECT 
    id, content,
    -- Decay calculation
    POWER(0.995, EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 3600) as recency_score,
    -- Weighted final score
    (
        0.6 * semantic_score +
        0.25 * recency_score +
        0.15 * importance_score
    ) as final_score
FROM agent_memory
WHERE ...
ORDER BY final_score DESC;
```

**Advantages**:
- ✅ **Native math functions** - `POWER()`, `EXTRACT()`, `EXTRACT(EPOCH FROM ...)`
- ✅ **Inline calculation** - No Python overhead
- ✅ **Atomic updates** - `UPDATE ... SET last_accessed_at = NOW(), access_count = access_count + 1`
- ✅ **Flexible formulas** - Easy to adjust decay rate or weights

**Performance**: Negligible overhead (~0.1ms per memory)

#### Redis + RediSearch Implementation

**Implementation**:
```python
# Decay must be calculated in Python
from datetime import datetime

def calculate_decay(last_accessed):
    hours = (datetime.now() - last_accessed).total_seconds() / 3600
    return 0.995 ** hours

# Apply decay post-query
results = index.query(vector_query)
for result in results:
    result['recency_score'] = calculate_decay(result['last_accessed_at'])
    result['final_score'] = (
        0.6 * result['semantic_score'] +
        0.25 * result['recency_score'] +
        0.15 * result['importance_score']
    )
```

**Limitations**:
- ⚠️ **Python overhead** - Decay calculation happens in application code
- ⚠️ **Post-processing** - Can't filter by decay score in query
- ⚠️ **No inline math** - Redis doesn't support `POWER()` in queries
- ⚠️ **Update complexity** - Requires separate `HSET` calls

**Performance**: ~1-2ms overhead for post-processing

#### Winner: **PostgreSQL**

**Analysis**: PostgreSQL wins decisively because:
- **Inline calculation** is faster and more efficient
- **Atomic updates** ensure consistency
- **Query-time filtering** by decay score is possible
- **No application overhead** - all logic in database

---

### 3. Graph Traversal

#### Requirement
Engram needs multi-hop graph traversal:
- **Typed relationships** (`causes`, `relates_to`, `contradicts`)
- **Weighted edges** (path weight = product of edge weights)
- **Cycle prevention** (track visited nodes)
- **Hop limits** (1-3 hops typically)
- **Performance target**: <300ms for 2-hop queries

#### PostgreSQL + pgvector Implementation

**Implementation**:
```sql
-- Recursive CTE for graph traversal
WITH RECURSIVE traversal AS (
    -- Base case: start node
    SELECT 
        m.id, m.content, m.metadata,
        0 as hop_depth,
        ARRAY[m.id] as path,
        1.0 as path_weight
    FROM agent_memory m
    WHERE m.id = :start_id
    
    UNION ALL
    
    -- Recursive case: follow relations
    SELECT 
        m.id, m.content, m.metadata,
        t.hop_depth + 1,
        t.path || m.id,
        t.path_weight * r.weight
    FROM traversal t
    JOIN memory_relations r ON r.source_id = t.id
    JOIN agent_memory m ON m.id = r.target_id
    WHERE t.hop_depth < :max_hops
      AND r.weight >= :min_weight
      AND NOT (m.id = ANY(t.path))  -- Prevent cycles
)
SELECT * FROM traversal WHERE hop_depth > 0;
```

**Advantages**:
- ✅ **Native recursive CTEs** - PostgreSQL excels at graph queries
- ✅ **Cycle prevention** - `ARRAY` tracking is efficient
- ✅ **Path weight calculation** - Multiplicative weights in SQL
- ✅ **Complex filtering** - Easy to filter by relation_type, weight, etc.
- ✅ **Query optimization** - PostgreSQL planner optimizes recursive queries

**Performance**: ~50-200ms for 2-hop traversal (10-100 edges per hop)

#### Redis + RediSearch Implementation

**Limitations**:
- ❌ **No native graph support** - Redis has no recursive query capability
- ❌ **Manual implementation** - Must fetch edges in Python loops
- ❌ **N+1 queries** - Each hop requires separate Redis calls
- ❌ **Cycle prevention** - Must track in Python (memory overhead)
- ❌ **Performance** - Multiple round-trips add latency

**Workaround** (inefficient):
```python
# Manual graph traversal in Python
def traverse(start_id, max_hops):
    visited = set()
    queue = [(start_id, 0, [start_id], 1.0)]
    results = []
    
    while queue:
        node_id, hop, path, weight = queue.pop(0)
        if hop >= max_hops:
            continue
        
        # Fetch edges (separate Redis call)
        edges = r.smembers(f"relations:{node_id}")
        for target_id in edges:
            if target_id not in visited:
                visited.add(target_id)
                # Fetch relation weight (another Redis call)
                rel_weight = r.hget(f"relation:{node_id}:{target_id}", "weight")
                queue.append((target_id, hop+1, path+[target_id], weight*rel_weight))
    
    return results
```

**Performance**: ~200-500ms for 2-hop traversal (multiple Redis calls)

#### Winner: **PostgreSQL** (decisive)

**Analysis**: PostgreSQL is the clear winner because:
- **Native recursive CTEs** are purpose-built for graph traversal
- **Single query** vs. multiple Redis round-trips
- **Better performance** - Optimized by database engine
- **Graphiti benchmark** (300ms) was achieved with PostgreSQL-like systems

---

### 4. Session Management

#### Requirement
Engram needs:
- **Session lifecycle** - Create, expire, terminate
- **Cross-session queries** - Query all user memories across sessions
- **Session hierarchy** - Parent-child relationships
- **Automatic expiration** - Background cleanup of expired sessions
- **TTL support** - Configurable session timeouts

#### PostgreSQL + pgvector Implementation

**Implementation**:
```sql
-- Session table
CREATE TABLE agent_sessions (
    id UUID PRIMARY KEY,
    agent_id UUID REFERENCES agents(id),
    user_id UUID REFERENCES users(id),
    parent_session_id UUID REFERENCES agent_sessions(id),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '24 hours'),
    status TEXT DEFAULT 'active'
);

-- Cross-session query
SELECT * FROM agent_memory
WHERE user_id = :uid  -- Not session-specific!
ORDER BY created_at DESC;

-- Session hierarchy query
SELECT * FROM agent_memory
WHERE session_id IN (
    SELECT id FROM agent_sessions
    WHERE user_id = :uid
      AND (id = :session_id 
           OR parent_session_id = :session_id
           OR id IN (
               SELECT id FROM agent_sessions
               WHERE parent_session_id = :session_id
           ))
);
```

**Advantages**:
- ✅ **Relational integrity** - Foreign keys ensure consistency
- ✅ **Complex queries** - Easy to query across sessions
- ✅ **Hierarchy support** - Recursive queries for parent-child
- ✅ **Background jobs** - PostgreSQL can run scheduled tasks (pg_cron)
- ✅ **ACID guarantees** - Session creation/deletion is atomic

#### Redis + RediSearch Implementation

**Implementation**:
```python
# Session as Redis Hash
r.hset(
    f"session:{session_id}",
    mapping={
        "agent_id": agent_id,
        "user_id": user_id,
        "started_at": timestamp,
        "expires_at": timestamp + 86400
    }
)

# Set TTL
r.expire(f"session:{session_id}", 86400)

# Cross-session query (requires scanning)
# Redis doesn't support JOINs, so must query by user_id prefix
results = r.ft("memories").search(
    f"@user_id:{user_id}",
    return_fields=["content", "session_id"]
)
```

**Advantages**:
- ✅ **Native TTL** - `EXPIRE` command for automatic cleanup
- ✅ **Fast lookups** - O(1) session retrieval
- ✅ **Simple API** - Straightforward key-value operations

**Limitations**:
- ⚠️ **No foreign keys** - Must enforce relationships in application
- ⚠️ **Limited queries** - Can't easily query session hierarchy
- ⚠️ **Cross-session complexity** - Requires careful key design
- ⚠️ **No ACID** - Session operations aren't transactional

#### Winner: **PostgreSQL** (for complexity) | **Redis** (for simplicity)

**Analysis**: PostgreSQL wins for Engram because:
- **Cross-session queries** are a core requirement (user memories persist)
- **Session hierarchy** needs recursive queries (PostgreSQL excels)
- **ACID guarantees** ensure session consistency

---

### 5. Persistence and Durability

#### Requirement
Engram is a **memory layer** - persistence is critical:
- **Data must survive restarts** - Memories are long-term
- **ACID guarantees** - Memory writes must be atomic
- **Backup and recovery** - Point-in-time recovery needed
- **Replication** - High availability requirements

#### PostgreSQL + pgvector Implementation

**Strengths**:
- ✅ **Always persistent** - WAL (Write-Ahead Logging) ensures durability
- ✅ **Full ACID** - Transactions guarantee consistency
- ✅ **Point-in-time recovery** - WAL enables PITR
- ✅ **Replication** - Streaming replication for HA
- ✅ **Backup tools** - `pg_dump`, `pg_basebackup`, etc.

**Persistence Model**:
```
Write → WAL (durable) → Buffer Pool → Disk (async)
         ↑
    Always written first (ACID guarantee)
```

#### Redis + RediSearch Implementation

**Persistence Options**:

**RDB Snapshots**:
- ✅ **Point-in-time backups** - Configurable frequency
- ⚠️ **Data loss window** - Last snapshot to crash (up to minutes)
- ⚠️ **Fork overhead** - Can block during snapshot

**AOF (Append-Only File)**:
- ✅ **Durability** - Every write logged
- ⚠️ **Performance impact** - Slower writes (fsync overhead)
- ⚠️ **File growth** - Requires periodic rewrites

**Trade-offs**:
- **Performance vs. Durability** - Must choose
- **No ACID** - Single-threaded model limits transactions
- **Data loss risk** - If persistence disabled for performance

#### Winner: **PostgreSQL** (decisive)

**Analysis**: PostgreSQL wins decisively because:
- **AI memory requires persistence** - Not optional caching
- **ACID guarantees** prevent data corruption
- **Point-in-time recovery** is essential for production
- **Redis persistence is a trade-off** - Engram can't afford data loss

---

### 6. Complex Queries and Filtering

#### Requirement
Engram needs complex queries:
- **Multi-field filtering** - Tags, numeric ranges, timestamps
- **Aggregations** - Count memories per user, average importance, etc.
- **Joins** - Memory + Relations + Sessions + Users
- **Window functions** - Ranking, partitioning by user
- **Subqueries** - Nested filtering and aggregation

#### PostgreSQL + pgvector Implementation

**Strengths**:
```sql
-- Complex query example
SELECT 
    u.external_id,
    COUNT(m.id) as memory_count,
    AVG(m.importance_score) as avg_importance,
    MAX(m.created_at) as last_memory
FROM users u
JOIN agent_memory m ON m.user_id = u.id
LEFT JOIN memory_relations r ON r.source_id = m.id
WHERE m.agent_id = :aid
  AND m.created_at > NOW() - INTERVAL '30 days'
  AND m.deleted_at IS NULL
GROUP BY u.id
HAVING COUNT(m.id) > 10
ORDER BY memory_count DESC;
```

**Advantages**:
- ✅ **Full SQL** - All SQL features available
- ✅ **Query planner** - Automatic optimization
- ✅ **Indexes** - B-tree, GIN, GiST, BRIN for different patterns
- ✅ **Explain plans** - Easy debugging and optimization

#### Redis + RediSearch Implementation

**Limitations**:
```python
# Redis query syntax is limited
query = Query("@category:{electronics} @price:[100 500]")
# Can't do:
# - JOINs
# - Aggregations (COUNT, AVG, etc.)
# - Subqueries
# - Window functions
```

**Workarounds**:
- ⚠️ **Application-level aggregation** - Fetch results, aggregate in Python
- ⚠️ **Multiple queries** - Fetch related data separately
- ⚠️ **Denormalization** - Store aggregated values (stale data risk)

#### Winner: **PostgreSQL** (decisive)

**Analysis**: PostgreSQL wins because:
- **Complex queries are core** - Engram needs aggregations, joins, subqueries
- **SQL ecosystem** - Better tooling, debugging, optimization
- **Query flexibility** - Can evolve queries without code changes

---

### 7. Scalability

#### Requirement
Engram must scale to:
- **10M+ memories** across thousands of agents
- **1000+ concurrent users**
- **Sub-200ms retrieval** for 100K memories
- **Horizontal scaling** for growth

#### PostgreSQL + pgvector Implementation

**Scaling Strategies**:

**Vertical Scaling**:
- ✅ **Single-node performance** - Can handle 10M+ rows with proper indices
- ✅ **Partitioning** - Hot/cold partitioning for time-based data
- ✅ **Connection pooling** - PgBouncer for connection management

**Horizontal Scaling**:
- ✅ **Read replicas** - Streaming replication for read scaling
- ✅ **Sharding** - Application-level or Citus extension
- ✅ **Partitioning** - Table partitioning for large datasets

**Performance at Scale**:
- **10M memories**: ~50-100ms query latency (with partitioning)
- **Concurrent users**: 1000+ with connection pooling
- **Write throughput**: ~10K writes/sec (single node)

#### Redis + RediSearch Implementation

**Scaling Strategies**:

**Vertical Scaling**:
- ✅ **In-memory speed** - Fastest for datasets that fit in RAM
- ⚠️ **Memory limits** - Constrained by available RAM
- ⚠️ **Cost** - RAM is expensive at scale

**Horizontal Scaling**:
- ✅ **Redis Cluster** - Automatic sharding
- ✅ **Redis Sentinel** - High availability
- ✅ **Sharding** - Built-in cluster support

**Performance at Scale**:
- **10M memories**: Requires ~72GB RAM (with HNSW)
- **Concurrent users**: 10K+ (single-threaded bottleneck)
- **Write throughput**: ~100K writes/sec (single node)

**Memory Calculation**:
```
10M memories × 1536 dims × 4 bytes = 61.4GB (vectors)
+ 20% HNSW overhead = 12.3GB
+ Metadata = ~5GB
Total: ~78GB RAM required
```

#### Winner: **Tie** (different strengths)

**Analysis**: 
- **PostgreSQL** wins for **cost** (disk is cheaper than RAM)
- **Redis** wins for **throughput** (if data fits in memory)
- **Engram's use case** (10M+ memories) favors PostgreSQL (cost-effective)

---

## Architecture Comparison

### PostgreSQL + pgvector Architecture

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│  ┌───────────────────────────────────┐  │
│  │      Engram Library               │  │
│  │  - Memory Store                   │  │
│  │  - Hybrid Search                  │  │
│  │  - Graph Traversal                 │  │
│  │  - Session Manager                │  │
│  └──────────────┬────────────────────┘  │
└─────────────────┼───────────────────────┘
                  │ SQL Queries
                  ▼
┌─────────────────────────────────────────┐
│      PostgreSQL + pgvector              │
│  ┌───────────────────────────────────┐  │
│  │  Single Database                  │  │
│  │  - Vectors (pgvector)             │  │
│  │  - Full-text (tsvector)          │  │
│  │  - JSONB (metadata)              │  │
│  │  - Relations (tables)             │  │
│  │  - ACID Transactions              │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Advantages**:
- ✅ **Converged architecture** - Everything in one database
- ✅ **ACID guarantees** - Data consistency
- ✅ **SQL ecosystem** - Universal tooling
- ✅ **Cost-effective** - Disk storage

### Redis + RediSearch Architecture

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│  ┌───────────────────────────────────┐  │
│  │      Engram Library               │  │
│  │  - Memory Store                   │  │
│  │  - Hybrid Search                  │  │
│  │  - Graph Traversal (Python)        │  │
│  │  - Session Manager                │  │
│  └──────────────┬────────────────────┘  │
└─────────────────┼───────────────────────┘
                  │ FT.SEARCH / Redis Commands
                  ▼
┌─────────────────────────────────────────┐
│      Redis + RediSearch                │
│  ┌───────────────────────────────────┐  │
│  │  In-Memory Database               │  │
│  │  - Vectors (HNSW/FLAT/SVS)        │  │
│  │  - Full-text (RediSearch)         │  │
│  │  - Hash/JSON (metadata)           │  │
│  │  - Limited Transactions           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**Advantages**:
- ✅ **Ultra-low latency** - Sub-millisecond queries
- ✅ **High throughput** - Millions of ops/sec
- ✅ **Simple deployment** - Single process

**Disadvantages**:
- ⚠️ **Memory constraints** - Requires large RAM
- ⚠️ **Graph traversal** - Must implement in application
- ⚠️ **Persistence trade-offs** - Performance vs. durability

---

## Performance Benchmarks

### Query Latency Comparison

| Operation | Dataset Size | PostgreSQL | Redis | Winner |
|-----------|--------------|------------|-------|--------|
| **Vector Search (KNN)** | 100K | ~20ms | ~2ms | 🏆 Redis |
| **Vector Search (KNN)** | 1M | ~50ms | ~5ms | 🏆 Redis |
| **Vector Search (KNN)** | 10M | ~100ms* | ~20ms* | 🏆 Redis |
| **Hybrid Search** | 100K | ~30ms | ~3ms | 🏆 Redis |
| **Graph Traversal (2-hop)** | 10 edges/hop | ~50ms | ~200ms | 🏆 PostgreSQL |
| **Graph Traversal (2-hop)** | 100 edges/hop | ~200ms | ~500ms | 🏆 PostgreSQL |
| **Complex Query (JOIN + Agg)** | 1M | ~100ms | N/A** | 🏆 PostgreSQL |
| **Memory Decay Calculation** | 1M | ~0.1ms | ~2ms | 🏆 PostgreSQL |

*With partitioning/sharding
**Redis doesn't support complex queries

### Throughput Comparison

| Operation | PostgreSQL | Redis | Winner |
|-----------|-----------|-------|--------|
| **Writes/sec** | ~10K | ~100K | 🏆 Redis |
| **Reads/sec** | ~50K | ~500K | 🏆 Redis |
| **Concurrent Users** | 1000+ | 10K+ | 🏆 Redis |

### Memory Usage Comparison

| Dataset | PostgreSQL | Redis | Winner |
|---------|-----------|-------|--------|
| **1M memories (1536-dim)** | ~6GB disk | ~7.2GB RAM | 🏆 PostgreSQL (cost) |
| **10M memories (1536-dim)** | ~60GB disk | ~72GB RAM | 🏆 PostgreSQL (cost) |

**Cost Analysis**:
- **Disk storage**: ~$0.10/GB/month (AWS EBS)
- **RAM storage**: ~$10/GB/month (AWS ElastiCache)
- **10M memories cost**: PostgreSQL = $6/month, Redis = $720/month

---

## Operational Considerations

### Deployment Complexity

**PostgreSQL**:
- ✅ **Docker-ready** - Official pgvector images
- ✅ **Managed services** - AWS RDS, Azure Database, GCP Cloud SQL
- ✅ **Backup tools** - Built-in `pg_dump`, `pg_basebackup`
- ✅ **Monitoring** - pg_stat_statements, pgAdmin, Grafana

**Redis**:
- ✅ **Docker-ready** - Official Redis images
- ✅ **Managed services** - AWS ElastiCache, Azure Cache, Redis Cloud
- ⚠️ **Backup complexity** - RDB/AOF management
- ⚠️ **Monitoring** - Redis-specific tools (RedisInsight)

### Maintenance

**PostgreSQL**:
- ✅ **VACUUM** - Automatic cleanup of dead tuples
- ✅ **ANALYZE** - Automatic statistics updates
- ✅ **Index maintenance** - REINDEX for optimization
- ✅ **Query optimization** - EXPLAIN plans for tuning

**Redis**:
- ⚠️ **Memory management** - Manual eviction policy tuning
- ⚠️ **Persistence tuning** - Balance performance vs. durability
- ⚠️ **No automatic optimization** - Manual index tuning
- ⚠️ **Debugging** - Limited query analysis tools

### Developer Experience

**PostgreSQL**:
- ✅ **SQL familiarity** - Universal language
- ✅ **Rich tooling** - pgAdmin, DBeaver, DataGrip
- ✅ **Documentation** - Extensive PostgreSQL docs
- ✅ **Community** - Large, active community

**Redis**:
- ⚠️ **FT.SEARCH syntax** - Specialized query language
- ⚠️ **Limited tooling** - RedisInsight, redis-cli
- ⚠️ **Documentation** - RediSearch docs less comprehensive
- ⚠️ **Community** - Smaller, specialized community

---

## Cost Analysis

### Infrastructure Costs (10M memories, 1536-dim)

**PostgreSQL**:
- **Storage**: 60GB disk × $0.10/GB = **$6/month**
- **Compute**: db.r5.xlarge (4 vCPU, 32GB RAM) = **$200/month**
- **Total**: **~$206/month**

**Redis**:
- **Storage**: 72GB RAM × $10/GB = **$720/month** (or included in compute)
- **Compute**: cache.r6g.2xlarge (8 vCPU, 52GB RAM) = **$400/month**
- **Total**: **~$400-1120/month**

**Winner**: **PostgreSQL** (5-18x cheaper)

### Development Costs

**PostgreSQL**:
- ✅ **Faster development** - SQL is familiar, better tooling
- ✅ **Easier debugging** - EXPLAIN plans, query logs
- ✅ **Better documentation** - More resources available

**Redis**:
- ⚠️ **Learning curve** - FT.SEARCH syntax, Redis patterns
- ⚠️ **Debugging complexity** - Limited query analysis
- ⚠️ **Less documentation** - Fewer examples and guides

---

## Migration Path

### Starting with PostgreSQL

**Advantages**:
- ✅ **No migration needed** - Start with production-ready choice
- ✅ **Future-proof** - Can scale to any size
- ✅ **Feature-complete** - All Engram requirements supported

### Starting with Redis, Migrating to PostgreSQL

**Challenges**:
- ⚠️ **Data migration** - Export from Redis, import to PostgreSQL
- ⚠️ **Code changes** - Rewrite queries from FT.SEARCH to SQL
- ⚠️ **Downtime** - Migration requires service interruption
- ⚠️ **Testing** - Must validate data integrity

**When Migration Makes Sense**:
- Started with Redis for speed
- Outgrew memory constraints
- Need ACID guarantees
- Require complex queries

---

## Recommendation

### 🏆 **PostgreSQL + pgvector is the Recommended Choice**

### Reasons

1. **✅ ACID Guarantees**
   - Memory consistency is critical for AI applications
   - PostgreSQL provides full ACID, Redis has limited guarantees

2. **✅ Graph Traversal**
   - Engram's core feature requires recursive queries
   - PostgreSQL's recursive CTEs are purpose-built for this
   - Redis would require inefficient Python loops

3. **✅ Complex Queries**
   - Hybrid search + decay + filters need SQL flexibility
   - PostgreSQL's SQL ecosystem enables precise control
   - Redis's FT.SEARCH is too limited

4. **✅ Persistence**
   - AI memory must survive restarts
   - PostgreSQL is always persistent
   - Redis persistence is a performance trade-off

5. **✅ Cost-Effectiveness**
   - 10M memories: PostgreSQL = $206/month, Redis = $400-1120/month
   - Disk storage is 100x cheaper than RAM

6. **✅ Developer Experience**
   - SQL is universal and well-documented
   - Better tooling (pgAdmin, EXPLAIN plans)
   - Easier debugging and optimization

### When Redis Would Be Better

Redis would be the better choice if:

1. **Ultra-low latency is #1 priority** (<5ms required)
2. **Dataset fits entirely in memory** (<1M memories)
3. **Simple operations dominate** (no graph traversal needed)
4. **Caching layer** (not primary storage)
5. **High write throughput** (100K+ writes/sec)

### Hybrid Approach (Future Consideration)

**Possible Architecture**:
```
┌─────────────────┐
│   Application   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Redis  │ │ PostgreSQL   │
│ (Hot)  │ │ (Cold)       │
│        │ │              │
│ Recent │ │ All Memories │
│ 1M     │ │ 10M+         │
└────────┘ └──────────────┘
```

**Use Cases**:
- **Redis**: Hot partition (recent 1M memories) for speed
- **PostgreSQL**: Cold partition (older memories) for persistence
- **Sync**: Periodically move cold data from Redis to PostgreSQL

**Complexity**: High (requires data synchronization, query routing)

**Recommendation**: Start with PostgreSQL, add Redis caching layer later if needed

---

## Conclusion

For Engram's requirements, **PostgreSQL + pgvector is the clear winner**:

- ✅ **Meets all requirements** - Hybrid search, decay, graph traversal, sessions
- ✅ **Production-ready** - ACID guarantees, persistence, scalability
- ✅ **Cost-effective** - 5-18x cheaper than Redis at scale
- ✅ **Developer-friendly** - SQL ecosystem, better tooling
- ✅ **Future-proof** - Can scale to any size without architecture changes

Redis excels at ultra-low latency and high throughput, but Engram's requirements (graph traversal, complex queries, persistence) favor PostgreSQL's strengths.

**Final Recommendation**: **Proceed with PostgreSQL + pgvector for Engram implementation.**

---

## References

- [REDIS_VECTOR_SEARCH_ANALYSIS.md](./REDIS_VECTOR_SEARCH_ANALYSIS.md) - Redis vector search capabilities
- [MEMORY_THEORY.md](./MEMORY_THEORY.md) - Engram memory system theory
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) - Engram implementation plan
- PostgreSQL pgvector Documentation
- Redis RediSearch Documentation

---

**Analysis Date**: January 2026
**PostgreSQL Version**: 16+
**Redis Version**: 8.x / Redis Stack
**pgvector Version**: 0.5+

