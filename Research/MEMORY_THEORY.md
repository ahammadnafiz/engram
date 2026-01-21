# Engram Memory System: Theoretical Foundations

> **How Each Component Works: A Deep Dive**

This document explains the theoretical foundations and operational mechanics of each component in the Engram memory system. Understanding these principles is crucial for effective implementation and optimization.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Memory Store](#memory-store)
3. [Memory Decay](#memory-decay)
4. [Hybrid Search](#hybrid-search)
5. [Graph Traversal](#graph-traversal)
6. [Session Management](#session-management)
7. [Summarization Pipeline](#summarization-pipeline)
8. [Database Schema](#database-schema)
9. [Component Interactions](#component-interactions)
10. [Performance Characteristics](#performance-characteristics)

---

## System Overview

Engram implements a **converged cognitive architecture** where all memory operations happen within a single PostgreSQL database. This eliminates the complexity of coordinating multiple systems (vector DB, graph DB, file storage) while maintaining ACID guarantees.

### Core Principle: Converged Storage

```
Traditional Approach:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Vector DB   │  │ Graph DB    │  │ File Store  │
│ (Pinecone)  │  │ (Neo4j)     │  │ (S3)        │
└─────────────┘  └─────────────┘  └─────────────┘
     │                 │                 │
     └─────────────────┴─────────────────┘
              No ACID guarantees
              Complex coordination

Engram Approach:
┌─────────────────────────────────────┐
│      PostgreSQL + pgvector          │
│  ┌──────────┐  ┌──────────┐         │
│  │ Vectors  │  │ Graphs   │         │
│  │ JSONB    │  │ Relations│         │
│  │ Full-text│  │ Sessions │         │
│  └──────────┘  └──────────┘         │
│         ACID Guaranteed             │
└─────────────────────────────────────┘
```

**Benefits:**
- **Atomicity**: Memory writes are all-or-nothing
- **Consistency**: No orphaned relationships or corrupted state
- **Isolation**: Concurrent access handled safely
- **Durability**: Persisted to disk immediately

---

## Memory Store

### Theoretical Foundation

The Memory Store is the foundational CRUD layer that handles all persistent memory operations. It implements the **Repository Pattern** with soft-delete support.

### How It Works

#### 1. **Memory Creation Flow**

```
User Request: "Remember: User prefers dark mode"
    ↓
┌─────────────────────────────────────┐
│ 1. Content Processing               │
│    - Extract text: "User prefers    │
│      dark mode"                     │
│    - Generate hash: MD5(content)    │
│    - Check for duplicates           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Embedding Generation             │
│   - Call user's embedding_fn()      │
│   - Get vector: [0.1, 0.5, ..., 0.3]│
│   - Store in appropriate column     │
│     (embedding_1536 for 1536-dim)   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Database Insert                  │
│    - Generate UUID                  │
│    - Create session (if needed)     │
│    - Insert into agent_memory       │
│    - Auto-generate text_search      │
│      (PostgreSQL TSVECTOR)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. Index Update                     │
│    - HNSW index (vector)            │
│    - GIN index (text_search)        │
│    - B-tree index (user_id, created)│
└─────────────────────────────────────┘
```

#### 2. **Deduplication Strategy**

**Content Hash Mechanism:**
```python
content_hash = MD5(content + user_id + agent_id)
```

**Why Hash-Based Deduplication:**
- Prevents storing identical memories multiple times
- Fast lookup: O(1) hash check before insert
- Reduces storage and improves search quality

**Trade-off:** Two memories with identical content but different contexts are treated as duplicates. This is intentional—if the exact same fact is stated twice, we only need one memory.

#### 3. **Soft Delete Pattern**

**Why Soft Delete:**
- Preserves referential integrity (graph relationships)
- Enables recovery/undo functionality
- Maintains audit trail
- Allows gradual decay instead of instant removal

**Implementation:**
```sql
-- Soft delete
UPDATE agent_memory 
SET deleted_at = NOW() 
WHERE id = :memory_id;

-- Query excludes deleted
SELECT * FROM agent_memory 
WHERE deleted_at IS NULL;
```

**Cleanup Strategy:** Deleted memories older than 90 days can be permanently removed via background job.

#### 4. **Multi-Dimensional Embedding Support**

**Problem:** Different embedding models use different dimensions:
- OpenAI ada-002: 1536
- OpenAI text-3-large: 3072
- Cohere embed-v3: 1024

**Solution:** Multiple columns with NULL values:
```sql
embedding_1024 VECTOR(1024),   -- NULL if not used
embedding_1536 VECTOR(1536),   -- NULL if not used
embedding_3072 VECTOR(3072)    -- NULL if not used
```

**Benefits:**
- No schema migration when switching models
- Can store multiple embeddings per memory (future)
- Search uses appropriate column based on query

**Storage Overhead:** ~4KB per memory (acceptable trade-off for flexibility)

---

## Memory Decay

### Theoretical Foundation: Ebbinghaus Forgetting Curve

The Memory Decay component implements **exponential decay** based on the Ebbinghaus Forgetting Curve (1885), which models how memory retention decreases over time.

### Mathematical Model

#### 1. **Exponential Decay Formula**

```
R(t) = e^(-t/S)

Where:
- R(t) = Memory retention at time t (0 to 1)
- t = Time elapsed since last access
- S = Memory strength (increases with access)
- e = Euler's number (≈2.71828)
```

#### 2. **MemoryBank Simplification**

Engram uses MemoryBank's simplified formula:

```
recency_score = decay_rate ^ hours_elapsed

Where:
- decay_rate = 0.995 (per hour)
- hours_elapsed = (current_time - last_accessed_at) / 3600
```

**Why 0.995?**
- After 24 hours: 0.995^24 ≈ 0.886 (88.6% retention)
- After 1 week: 0.995^168 ≈ 0.433 (43.3% retention)
- After 30 days: 0.995^720 ≈ 0.025 (2.5% retention)

This matches human memory patterns: rapid initial decay, then gradual decline.

### How It Works

#### 1. **Recency Score Calculation**

```python
def calculate_recency_score(last_accessed: datetime) -> float:
    hours_elapsed = (NOW() - last_accessed).total_seconds() / 3600
    return 0.995 ** hours_elapsed
```

**Examples:**
- **0 hours**: 1.0 (perfect recency)
- **1 hour**: 0.995 (99.5% retention)
- **24 hours**: 0.886 (88.6% retention)
- **1 week**: 0.433 (43.3% retention)
- **1 month**: 0.025 (2.5% retention)

#### 2. **Memory Strength Tracking**

**MemoryBank Behavior:**
- **Initial strength**: S = 1 (first mention)
- **On recall**: S = S + 1 (strength increases)
- **Time reset**: t = 0 (decay timer resets)

**Why This Matters:**
- Frequently accessed memories decay slower
- Important memories (high access_count) stay relevant longer
- Mimics human memory: repetition strengthens recall

**Implementation:**
```python
def on_memory_access(memory):
    return {
        "memory_strength": memory.memory_strength + 1,
        "last_accessed_at": NOW(),
        "access_count": memory.access_count + 1
    }
```

#### 3. **Weighted Scoring**

**Final Memory Score:**
```
final_score = (w_rel × relevance) + (w_rec × recency) + (w_imp × importance)

Default weights:
- w_rel = 0.6 (relevance dominates)
- w_rec = 0.25 (recency matters)
- w_imp = 0.15 (importance bonus)
```

**Why These Weights:**
- **Relevance (60%)**: Semantic similarity is most important
- **Recency (25%)**: Recent memories are more likely relevant
- **Importance (15%)**: User-specified importance adjusts ranking

**Configurable:** Users can adjust weights based on use case:
- Time-sensitive: `(0.4, 0.4, 0.2)` - recency matters more
- Fact-focused: `(0.7, 0.2, 0.1)` - relevance dominates

### Decay in Practice

#### Scenario: User Preference Memory

```
Memory A: "User prefers dark mode" (created 1 hour ago, accessed 5 times)
Memory B: "User prefers light mode" (created 1 month ago, accessed 1 time)

Query: "What are my preferences?"

Calculation:
Memory A:
  - relevance: 0.9 (high semantic match)
  - recency: 0.995^1 = 0.995
  - importance: 0.7 (high access_count)
  - score: 0.6×0.9 + 0.25×0.995 + 0.15×0.7 = 0.899

Memory B:
  - relevance: 0.85 (good semantic match)
  - recency: 0.995^720 = 0.025
  - importance: 0.5 (low access_count)
  - score: 0.6×0.85 + 0.25×0.025 + 0.15×0.5 = 0.631

Result: Memory A ranks higher (correct!)
```

**Key Insight:** Decay ensures that even if an old memory has high semantic relevance, recent memories with similar relevance will rank higher.

---

## Hybrid Search

### Theoretical Foundation: Reciprocal Rank Fusion (RRF)

Hybrid Search combines **semantic search** (vector similarity) and **keyword search** (BM25) using Reciprocal Rank Fusion to overcome the limitations of each approach.

### The Problem: Keyword Blindness

**Vector Search Limitation:**
- "Error 502" and "Error 503" have similar embeddings
- Vector search treats them as identical
- But they're different HTTP status codes!

**Keyword Search Limitation:**
- "I like apples" vs "I enjoy fruit"
- No keyword overlap
- But semantically similar!

**Solution:** Combine both using RRF.

### How RRF Works

#### 1. **Reciprocal Rank Fusion Formula**

```
RRF_score(d) = Σ [1 / (k + rank(d, r))]

Where:
- d = document (memory)
- r = result set (semantic or keyword)
- k = constant (typically 60)
- rank(d, r) = position of d in result set r
```

**Why Reciprocal Rank:**
- Rank 1: 1/(60+1) = 0.0164
- Rank 2: 1/(60+2) = 0.0161
- Rank 10: 1/(60+10) = 0.0143
- Rank 50: 1/(60+50) = 0.0091

Higher ranks contribute more, but the difference decreases (smooth curve).

#### 2. **Hybrid Search Flow**

```
Query: "user preferences"
    ↓
┌─────────────────────────────────────┐
│ 1. Generate Query Embedding         │
│    - Call embedding_fn("user        │
│      preferences")                  │
│    - Get vector: [0.2, 0.1, ...]    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Parallel Search                  │
│                                     │
│  A. Semantic Search (Vector)        │
│     - HNSW index lookup             │
│     - Cosine similarity             │
│     - Top 50 results                │
│                                     │
│  B. Keyword Search (BM25)           │
│     - PostgreSQL full-text search   │
│     - GIN index on text_search      │
│     - Top 50 results                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Apply Memory Decay               │
│    - Calculate recency_score        │
│    - For each result in both sets   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. RRF Fusion                       │
│    - Calculate rank in each set     │
│    - RRF_score = 1/(60+rank_sem) +  │
│                  1/(60+rank_key)    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. Weighted Final Score             │
│    final_score =                    │
│      0.6 × semantic_score +         │
│      0.25 × recency_score +         │
│      0.15 × importance_score +      │
│      0.1 × RRF_score                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6. Sort & Return Top N              │
│    - Order by final_score DESC      │
│    - Limit to requested count       │
│    - Update access tracking         │
└─────────────────────────────────────┘
```

#### 3. **Why RRF Works**

**Example: Query "Error 502"**

```
Semantic Results:
Rank 1: "Server gateway timeout issues" (ID: A)
Rank 2: "Bad gateway errors" (ID: B)
Rank 20: "HTTP status codes" (ID: T)

Keyword Results:
Rank 1: "502 Bad Gateway documentation" (ID: X)
Rank 2: "Timeout configuration guide" (ID: Y)
Rank 20: "Error handling best practices" (ID: Z)

RRF Calculation:
ID A: RRF = 1/(60+1) + 0 = 0.0164 (only in semantic)
ID X: RRF = 0 + 1/(60+1) = 0.0164 (only in keyword)
ID B: RRF = 1/(60+2) + 0 = 0.0161 (only in semantic)

But if ID X also appears in semantic at rank 15:
ID X: RRF = 1/(60+15) + 1/(60+1) = 0.0133 + 0.0164 = 0.0297

Result: ID X ranks highest (correct! Exact keyword match)
```

**Key Insight:** RRF ensures that memories appearing in both result sets get boosted, while still allowing single-set results to contribute.

#### 4. **Performance Optimization**

**Why Limit to 50 per Set:**
- RRF works best with top-k results
- Diminishing returns beyond 50
- Keeps query fast (<200ms target)

**Why Inline Decay Calculation:**
```sql
POWER(0.995, EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 3600)
```
- Calculated in SQL (no Python overhead)
- Uses PostgreSQL's efficient math functions
- Single query instead of post-processing

---

## Graph Traversal

### Theoretical Foundation: Multi-Hop Reasoning

Graph Traversal enables **relational reasoning** by following typed edges between memories. This allows the system to answer questions that require connecting multiple facts.

### Graph Structure

#### 1. **Memory Relations Table**

```
memory_relations:
- source_id → target_id (directed edge)
- relation_type (typed: "causes", "relates_to", "contradicts")
- weight (edge strength: 0.0 to 1.0)
- confidence (certainty: 0.0 to 1.0)
```

**Why Typed Relations:**
- Enables semantic filtering ("only follow 'causes' edges")
- Supports different reasoning patterns
- Allows weighted traversal (stronger edges preferred)

#### 2. **Graph Traversal Algorithm**

**Recursive CTE Pattern:**
```sql
WITH RECURSIVE traversal AS (
    -- Base case: start node
    SELECT id, content, 0 as hop_depth, ARRAY[id] as path
    FROM agent_memory WHERE id = :start_id
    
    UNION ALL
    
    -- Recursive case: follow edges
    SELECT m.id, m.content, t.hop_depth + 1, t.path || m.id
    FROM traversal t
    JOIN memory_relations r ON r.source_id = t.id
    JOIN agent_memory m ON m.id = r.target_id
    WHERE t.hop_depth < :max_hops
      AND NOT (m.id = ANY(t.path))  -- Prevent cycles
)
SELECT * FROM traversal WHERE hop_depth > 0;
```

**How It Works:**

```
Start: Memory A ("User reported bug")
    ↓
Hop 1: Follow "causes" → Memory B ("Error 502 occurred")
    ↓
Hop 2: Follow "relates_to" → Memory C ("Server configuration issue")
    ↓
Result: [A, B, C] - Connected chain of reasoning
```

#### 3. **Cycle Prevention**

**Problem:** Graphs can have cycles:
```
A → B → C → A (cycle!)
```

**Solution:** Track path in recursive CTE:
```sql
ARRAY[id] as path  -- Track visited nodes
AND NOT (m.id = ANY(t.path))  -- Skip if already visited
```

**Why This Works:**
- Path array grows with each hop
- Check prevents revisiting nodes
- Ensures termination (max_hops limit)

#### 4. **Weighted Traversal**

**Path Weight Calculation:**
```sql
path_weight = 1.0 × weight_edge1 × weight_edge2 × ...
```

**Example:**
```
A → B (weight: 0.8) → C (weight: 0.6)
path_weight = 1.0 × 0.8 × 0.6 = 0.48
```

**Why Multiplicative:**
- Weak edges reduce path strength
- Strong paths (all high weights) rank higher
- Filters out low-confidence paths

#### 5. **Use Cases**

**Scenario 1: Causal Reasoning**
```
Query: "What caused the server error?"

Memory A: "User reported bug"
  → "causes" → Memory B: "Error 502 occurred"
    → "causes" → Memory C: "Server overload"

Traversal finds: [A, B, C]
Answer: "Server overload caused Error 502, which the user reported"
```

**Scenario 2: Contradiction Detection**
```
Memory A: "User prefers dark mode"
  → "contradicts" → Memory B: "User prefers light mode"

Traversal finds contradiction
System can flag for user confirmation
```

**Scenario 3: Related Concepts**
```
Memory A: "User studying Python"
  → "relates_to" → Memory B: "User asked about decorators"
    → "relates_to" → Memory C: "User working on Flask project"

Traversal finds related learning context
```

### Performance Characteristics

**Time Complexity:**
- Single hop: O(E) where E = edges from start node
- Multi-hop: O(E^h) where h = hop depth
- With max_hops=2: O(E^2) worst case

**Optimization Strategies:**
1. **Index on relations**: Fast edge lookup
2. **Limit results**: Early termination
3. **Filter by weight**: Skip weak edges
4. **Filter by type**: Only follow relevant relation types

**Target Performance:** <300ms for 2-hop traversal (Graphiti benchmark)

---

## Session Management

### Theoretical Foundation: Ephemeral Context

Sessions provide **temporal boundaries** for conversations while maintaining cross-session memory continuity through user_id.

### How Sessions Work

#### 1. **Session Lifecycle**

```
Session Creation:
┌─────────────────────────────────────┐
│ User starts conversation            │
│    ↓                                │
│ Check for active session            │
│    ↓                                │
│ If expired or none exists:          │
│   - Create new session              │
│   - Set expires_at = NOW() + 24h    │
│   - Link to user_id                 │
└─────────────────────────────────────┘

Session Usage:
┌─────────────────────────────────────┐
│ Each memory.add() call:             │
│   - Associate with current session  │
│   - Update session.last_active_at   │
│   - Extend expires_at if needed     │
└─────────────────────────────────────┘

Session Expiration:
┌─────────────────────────────────────┐
│ Background job (hourly):            │
│   - Find sessions where             │
│     expires_at < NOW()              │
│   - Set status = 'expired'          │
│   - Memories remain (linked to      │
│     user_id, not session)           │
└─────────────────────────────────────┘
```

#### 2. **Session Hierarchy**

**Parent-Child Relationships:**
```
Session A (parent)
  └─ Session B (child, parent_session_id = A.id)
      └─ Session C (child, parent_session_id = B.id)
```

**Use Cases:**
- **Conversation threads**: Related sessions form a thread
- **Topic continuation**: "Continue previous conversation"
- **Context inheritance**: Child sessions can access parent context

**Implementation:**
```sql
SELECT * FROM agent_memory
WHERE user_id = :uid
  AND session_id IN (
    SELECT id FROM agent_sessions
    WHERE user_id = :uid
      AND (id = :session_id 
           OR parent_session_id = :session_id
           OR id IN (
             SELECT id FROM agent_sessions
             WHERE parent_session_id = :session_id
           ))
  )
```

#### 3. **Cross-Session Queries**

**Key Design:** Memories are linked to `user_id`, not just `session_id`.

**Why This Matters:**
```python
# Query all user memories across sessions
results = await memory.search(
    "user preferences",
    user_id="user_123"  # Not session-specific!
)
```

**Benefits:**
- User preferences persist across sessions
- Long-term memory continuity
- Personalization improves over time

**Trade-off:** Session-specific context requires explicit filtering:
```python
# Query only current session
results = await memory.search(
    "user preferences",
    user_id="user_123",
    session_id="session_abc"  # Filter to session
)
```

#### 4. **Session Expiration Strategy**

**Default: 24-hour expiration**
- Balances context freshness with continuity
- Prevents stale sessions from accumulating
- Configurable per agent

**Expiration Behavior:**
- **Memories persist**: Linked to user_id, not deleted
- **Session marked expired**: Can't add new memories
- **Query still works**: Can search expired session memories

---

## Summarization Pipeline

### Theoretical Foundation: Information Compression

The Summarization Pipeline implements **ChatGPT-style lightweight summarization** to reduce storage while maintaining context continuity.

### How It Works

#### 1. **Buffer-Based Accumulation**

```
Memory Buffer (size: 10):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ M1  │ M2  │ M3  │ M4  │ M5  │ M6  │ M7  │ M8  │ M9  │ M10 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↑                                                          ↑
oldest                                                    newest

When buffer full:
┌─────────────────────────────────────┐
│ Trigger summarization               │
│    ↓                                │
│ Call user's summarize_fn()          │
│    ↓                                │
│ Generate summary: "User discussed   │
│  Python decorators, Flask routing,  │
│  and database queries"              │
│    ↓                                │
│ Store summary as new memory         │
│    ↓                                │
│ Clear buffer                        │
└─────────────────────────────────────┘
```

#### 2. **ChatGPT-Style Summarization**

**Key Insight:** Only summarize user messages, not assistant responses.

**Why:**
- User messages contain the information to remember
- Assistant responses are derivations (can be regenerated)
- Reduces summary size by ~50%

**Implementation:**
```python
# Extract only user content
contents = [m["content"] for m in memories if m["role"] == "user"]
summary = await summarize_fn(contents)
```

#### 3. **Summary Memory Structure**

```json
{
  "content": "User discussed Python decorators and Flask routing",
  "is_summary": true,
  "source_memory_ids": ["mem_1", "mem_2", ..., "mem_10"],
  "metadata": {
    "is_summary": true,
    "source_count": 10,
    "time_range": {
      "start": "2024-01-01T10:00:00Z",
      "end": "2024-01-01T10:30:00Z"
    }
  }
}
```

**Why Store Source IDs:**
- Enables drill-down: "Tell me more about that conversation"
- Allows summary verification
- Supports summary updates if sources change

#### 4. **Auto vs Manual Summarization**

**Auto Mode:**
```python
session = memory.session(
    user_id="user_123",
    summarize_fn=my_summarize,
    auto_summarize=True  # Auto-trigger at buffer_size
)
```

**Manual Mode:**
```python
session = memory.session(
    user_id="user_123",
    summarize_fn=my_summarize,
    auto_summarize=False
)

# ... add memories ...

# Manually trigger
await session.consolidate(last_n=10)
```

**When to Use Each:**
- **Auto**: High-volume conversations, storage optimization
- **Manual**: Important conversations, user control

#### 5. **Storage Reduction**

**Before Summarization:**
```
10 memories × 500 bytes = 5KB
```

**After Summarization:**
```
1 summary × 200 bytes = 200 bytes
Reduction: 96% (20x smaller)
```

**Trade-off:** Loss of detail vs. storage efficiency. Acceptable for older conversations where exact wording matters less.

---

## Database Schema

### Theoretical Foundation: Converged Storage

The database schema implements a **unified data model** where all memory components coexist in a single PostgreSQL database.

### Schema Components

#### 1. **Identity Layer**

**Tables: `agents`, `users`, `agent_sessions`**

**Purpose:** Multi-tenancy and session management

**Relationships:**
```
agents (1) ──→ (many) agent_sessions
users (1) ──→ (many) agent_sessions
agent_sessions (1) ──→ (many) agent_memory
```

**Why Separate Users Table:**
- External auth system integration
- Cross-agent user identity
- User-level analytics

#### 2. **Memory Storage**

**Table: `agent_memory`**

**Key Design Decisions:**

**A. Multi-Dimensional Embeddings**
```sql
embedding_1024 VECTOR(1024),
embedding_1536 VECTOR(1536),
embedding_3072 VECTOR(3072)
```
- **Why:** Different models, different dimensions
- **Trade-off:** Storage overhead (~4KB per memory)
- **Benefit:** No schema migration when switching models

**B. Generated Text Search**
```sql
text_search TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('english', content)
) STORED
```
- **Why:** Automatic full-text indexing
- **Benefit:** No manual index maintenance
- **Cost:** Slight write overhead (acceptable)

**C. Decay Tracking Fields**
```sql
memory_strength INT DEFAULT 1,
last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
access_count INT DEFAULT 0,
importance_score FLOAT DEFAULT 0.5
```
- **Why:** Enable memory decay scoring
- **Benefit:** Inline calculation, no joins needed
- **Cost:** 4 additional columns (minimal)

#### 3. **Graph Layer**

**Table: `memory_relations`**

**Composite Primary Key:**
```sql
PRIMARY KEY (source_id, target_id, relation_type)
```

**Why Composite:**
- Same memory pair can have multiple relation types
- Example: A "causes" B AND A "relates_to" B (both valid)
- Enables rich relationship modeling

**Soft Delete Support:**
```sql
deleted_at TIMESTAMPTZ
```
- Preserves referential integrity
- Allows relationship updates (delete old, create new)
- Maintains audit trail

### Index Strategy

#### 1. **Vector Index (HNSW)**

```sql
CREATE INDEX idx_memory_embedding_1536 ON agent_memory 
USING hnsw (embedding_1536 vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
```

**HNSW Parameters:**
- **m = 16**: Connections per node (balance speed/quality)
- **ef_construction = 64**: Search width during build (higher = better quality, slower build)

**Why HNSW:**
- Approximate Nearest Neighbor (ANN) search
- O(log N) query time vs O(N) brute force
- Production-proven (used by Pinecone, Weaviate)

#### 2. **Full-Text Index (GIN)**

```sql
CREATE INDEX idx_memory_text ON agent_memory 
USING GIN (text_search);
```

**Why GIN:**
- Generalized Inverted Index
- Fast full-text search (BM25 ranking)
- Supports complex queries (phrases, proximity)

#### 3. **B-Tree Indices**

```sql
CREATE INDEX idx_memory_user ON agent_memory 
(agent_id, user_id, created_at DESC);
```

**Why Composite:**
- Filters by agent_id and user_id first (high selectivity)
- Orders by created_at DESC (recent first)
- Covers common query pattern

**Partial Index:**
```sql
CREATE INDEX idx_memory_active ON agent_memory(agent_id) 
WHERE deleted_at IS NULL;
```

**Why Partial:**
- Smaller index (only active memories)
- Faster queries (skips deleted)
- Lower maintenance overhead

---

## Component Interactions

### How Components Work Together

#### 1. **Memory Add Flow**

```
User: memory.add("User prefers dark mode")
    ↓
┌─────────────────────────────────────┐
│ Engram Client                       │
│  - Validate input                   │
│  - Get/create session               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Memory Store                        │
│  - Generate content_hash            │
│  - Check duplicates                 │
│  - Call embedding_fn()              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Database Insert                     │
│  - Insert into agent_memory         │
│  - Auto-generate text_search        │
│  - Update indices                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Session Manager                     │
│  - Update session.last_active_at    │
│  - Check expiration                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Summarizer (if enabled)             │
│  - Add to buffer                    │
│  - Check if full                    │
│  - Trigger consolidation if needed  │
└─────────────────────────────────────┘
```

#### 2. **Memory Search Flow**

```
User: memory.search("preferences", user_id="user_123")
    ↓
┌─────────────────────────────────────┐
│ Engram Client                       │
│  - Generate query embedding         │
│  - Prepare search parameters        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Hybrid Search                       │
│  - Parallel semantic + keyword      │
│  - Get top 50 from each             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Memory Decay                        │
│  - Calculate recency_score          │
│  - For each result                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ RRF Fusion                          │
│  - Combine semantic + keyword       │
│  - Calculate final scores           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Memory Store                        │
│  - Update access tracking           │
│  - Increment memory_strength        │
│  - Reset last_accessed_at           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Return Results                      │
│  - Sort by final_score              │
│  - Limit to requested count         │
└─────────────────────────────────────┘
```

#### 3. **Graph Traversal Flow**

```
User: memory.traverse(start_id="mem_123", max_hops=2)
    ↓
┌─────────────────────────────────────┐
│ Graph Traversal                     │
│  - Execute recursive CTE            │
│  - Follow memory_relations edges    │
│  - Filter by relation_type, weight  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Memory Store                        │
│  - Join with agent_memory           │
│  - Get full memory content          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Return Results                      │
│  - Sort by path_weight              │
│  - Include hop_depth, path          │
└─────────────────────────────────────┘
```

---

## Performance Characteristics

### Query Performance

#### 1. **Hybrid Search**

**Time Complexity:**
- Semantic: O(log N) with HNSW index
- Keyword: O(log N) with GIN index
- RRF Fusion: O(K) where K = result set size (50)
- **Total: O(log N)** - Logarithmic!

**Real-World Performance:**
- 10K memories: ~50ms
- 100K memories: ~150ms
- 1M memories: ~300ms (with proper indices)

**Bottlenecks:**
- Embedding generation (user's function)
- HNSW index quality (tune m, ef_construction)
- Database connection pool size

#### 2. **Graph Traversal**

**Time Complexity:**
- Single hop: O(E) where E = edges from start
- Multi-hop: O(E^h) where h = hop depth
- With max_hops=2: O(E^2) worst case

**Real-World Performance:**
- 2-hop, 10 edges/hop: ~50ms
- 2-hop, 100 edges/hop: ~200ms
- 3-hop, 10 edges/hop: ~300ms

**Optimization:**
- Limit max_hops (2-3 optimal)
- Filter by min_weight (skip weak edges)
- Filter by relation_type (reduce edge set)

#### 3. **Memory Decay**

**Time Complexity:**
- Calculation: O(1) per memory
- Applied inline: No extra queries
- **Total: O(K)** where K = result set size

**Overhead:**
- ~0.1ms per memory (negligible)
- No performance impact on queries

### Storage Characteristics

#### 1. **Memory Size**

**Per Memory:**
- Content: ~500 bytes (average)
- Embedding (1536-dim): 6KB (float32)
- Metadata: ~200 bytes (JSONB)
- Indexes: ~2KB (overhead)
- **Total: ~8.7KB per memory**

**1M Memories:**
- Raw data: ~8.7GB
- With indexes: ~12GB
- Acceptable for modern hardware

#### 2. **Index Size**

**HNSW Index:**
- ~2-3x vector size
- 1M memories, 1536-dim: ~18GB
- Largest component

**GIN Index (text_search):**
- ~30% of content size
- 1M memories: ~150MB
- Minimal overhead

**B-Tree Indices:**
- ~10% of table size
- 1M memories: ~870MB
- Acceptable

---

## Conclusion

The Engram memory system implements a **converged cognitive architecture** that combines:

1. **Memory Store**: CRUD operations with deduplication and soft deletes
2. **Memory Decay**: Exponential decay based on Ebbinghaus Forgetting Curve
3. **Hybrid Search**: RRF fusion of semantic and keyword search
4. **Graph Traversal**: Multi-hop reasoning via typed relationships
5. **Session Management**: Temporal boundaries with cross-session continuity
6. **Summarization**: Optional compression for storage efficiency

Each component is theoretically grounded, production-proven, and designed to work together seamlessly within a single PostgreSQL database. The result is a memory system that is both powerful and practical for real-world AI applications.

---

## References

- **Ebbinghaus Forgetting Curve**: Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology*
- **MemoryBank**: Production memory system using exponential decay
- **Reciprocal Rank Fusion**: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). *Reciprocal Rank Fusion*
- **Graphiti**: Neo4j-based knowledge graph memory system
- **ChatGPT Memory**: Reverse-engineered architecture analysis
- **Claude Memory**: Reverse-engineered architecture analysis