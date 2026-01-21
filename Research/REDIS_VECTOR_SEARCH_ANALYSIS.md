# Redis Vector Search: Comprehensive Analysis

> **How Redis Works for Vector Search, Querying, and Indexing**

This document provides a comprehensive analysis of Redis's vector search capabilities based on extensive research of Redis documentation, RedisVL, and LangCache implementations.

---

## Table of Contents

1. [Redis Vector Search Overview](#redis-vector-search-overview)
2. [Index Types and Algorithms](#index-types-and-algorithms)
3. [Vector Storage](#vector-storage)
4. [Query Mechanisms](#query-mechanisms)
5. [Index Management](#index-management)
6. [Performance Characteristics](#performance-characteristics)
7. [Integration Patterns](#integration-patterns)
8. [Comparison with PostgreSQL pgvector](#comparison-with-postgresql-pgvector)

---

## Redis Vector Search Overview

### Core Architecture

Redis implements vector search through **RediSearch** (Redis Search module), which extends Redis with:

- **Vector Similarity Search**: K-nearest neighbor (KNN) and range-based queries
- **Full-Text Search**: BM25 and TF-IDF algorithms
- **Hybrid Search**: Combined vector + metadata filtering
- **Multiple Index Types**: FLAT, HNSW, SVS-VAMANA
- **Distance Metrics**: L2 (Euclidean), IP (Inner Product), COSINE

### Key Components

```
┌─────────────────────────────────────┐
│         Redis Server                │
│  ┌───────────────────────────────┐ │
│  │    RediSearch Module          │ │
│  │  ┌─────────────────────────┐ │ │
│  │  │  Vector Index Engine     │ │ │
│  │  │  - FLAT/HNSW/SVS-VAMANA  │ │ │
│  │  └─────────────────────────┘ │ │
│  │  ┌─────────────────────────┐ │ │
│  │  │  Query Parser           │ │ │
│  │  │  - KNN queries           │ │ │
│  │  │  - Range queries         │ │ │
│  │  │  - Hybrid queries        │ │ │
│  │  └─────────────────────────┘ │ │
│  └───────────────────────────────┘ │
│  ┌───────────────────────────────┐ │
│  │    Data Storage               │ │
│  │  - Redis Hash (HASH)         │ │
│  │  - JSON Documents (JSON)      │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
```

---

## Index Types and Algorithms

### 1. FLAT Index

**Algorithm**: Brute-force linear search

**Characteristics**:
- **Accuracy**: 100% recall (exact results)
- **Performance**: O(n) linear time complexity
- **Memory**: Minimal overhead
- **Build Time**: Fastest (no preprocessing)

**Use Cases**:
- Small datasets (< 100K vectors)
- Perfect accuracy required
- Simple setup preferred
- Query latency not critical

**Configuration**:
```python
from redis.commands.search.field import VectorField

vector_field = VectorField(
    "doc_embedding",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": 1536,
        "DISTANCE_METRIC": "COSINE"
    }
)
```

**CLI Equivalent**:
```
FT.CREATE documents
  ON HASH
  PREFIX 1 docs:
  SCHEMA doc_embedding VECTOR FLAT 6
    TYPE FLOAT32
    DIM 1536
    DISTANCE_METRIC COSINE
```

### 2. HNSW Index

**Algorithm**: Hierarchical Navigable Small World

**Characteristics**:
- **Accuracy**: 95-99% recall (tunable)
- **Performance**: O(log n) logarithmic time complexity
- **Memory**: Moderate overhead (~20% graph structure)
- **Build Time**: Moderate (graph construction)

**Use Cases**:
- Large datasets (100K - 1M+ vectors)
- Balanced speed and accuracy
- Cross-platform compatibility
- Most common production choice

**Index-Time Parameters**:
- **M**: Max outgoing edges per node (default: 16, range: 8-64)
  - Higher M = better recall, more memory, slower build
- **EF_CONSTRUCTION**: Candidates during build (default: 200, range: 100-800)
  - Higher EF_CONSTRUCTION = better graph quality, slower build

**Runtime Parameters** (set at query time):
- **EF_RUNTIME**: Candidates during search (default: 10)
  - Higher EF_RUNTIME = better recall, slower search
- **EPSILON**: Range query boundary factor (default: 0.01)
  - Controls approximation tolerance for range queries

**Configuration**:
```python
vector_field = VectorField(
    "doc_embedding",
    "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": 1536,
        "DISTANCE_METRIC": "COSINE",
        "M": 40,                    # Graph connectivity
        "EF_CONSTRUCTION": 250      # Build-time accuracy
    }
)
```

**Query with Runtime Parameters**:
```python
query = Query("(*)=>[KNN 10 @doc_embedding $vec AS score]") \
    .return_fields("title", "score") \
    .dialect(2)

params = {
    "vec": vector_bytes
}

# Higher EF_RUNTIME for better recall
results = r.ft("documents").search(
    query, 
    query_params=params,
    params={"EF_RUNTIME": 50}  # Runtime parameter
)
```

### 3. SVS-VAMANA Index

**Algorithm**: Graph-based with compression support

**Characteristics**:
- **Accuracy**: 90-95% recall (tunable)
- **Performance**: O(log n) logarithmic time complexity
- **Memory**: Low (with compression: LVQ, LeanVec)
- **Build Time**: Slower than HNSW for smaller datasets

**Use Cases**:
- Large datasets with memory constraints
- Intel hardware optimization
- High-dimensional vectors (with compression)
- Redis 8.2+ required

**Compression Options**:
- **LVQ4x4**: 8 bits/dim, fast search, large memory savings
- **LVQ8**: Faster ingestion, slower search
- **LVQ4x8**: Two-level quantization, improved recall
- **LeanVec4x8**: Fastest search and ingestion (with dimensionality reduction)
- **LeanVec8x8**: Improved recall

**Configuration**:
```python
vector_field = VectorField(
    "doc_embedding",
    "SVS-VAMANA", {
        "TYPE": "FLOAT32",
        "DIM": 1536,
        "DISTANCE_METRIC": "COSINE",
        "GRAPH_MAX_DEGREE": 40,
        "CONSTRUCTION_WINDOW_SIZE": 250,
        "COMPRESSION": "LVQ8"  # or "LeanVec4x8"
    }
)
```

**Runtime Parameters**:
- **search_window_size**: Larger window = better recall, slower (default: varies)
- **use_search_history**: ON, OFF, or AUTO (default: AUTO)
- **search_buffer_capacity**: Buffer size for search (default: varies)
- **epsilon**: Range query approximation (default: 0.01)

**CompressionAdvisor Utility**:
```python
from redisvl.utils import CompressionAdvisor

# Get recommendations
config = CompressionAdvisor.recommend(
    dims=1536,
    priority="balanced"  # "memory", "speed", or "balanced"
)

# Estimate memory savings
savings = CompressionAdvisor.estimate_memory_savings(
    compression="LeanVec4x8",
    dims=1536,
    reduce=768
)
# Returns: 81.2% memory saved
```

### Algorithm Comparison

| Algorithm | Best For | Performance | Memory Usage | Recall Quality | Build Time |
|-----------|----------|-------------|--------------|----------------|------------|
| **FLAT** | < 100K vectors | O(n) | Minimal | 100% (exact) | Fastest |
| **HNSW** | 100K - 1M+ | O(log n) | Moderate (~20%) | 95-99% (tunable) | Moderate |
| **SVS-VAMANA** | Large + memory constraints | O(log n) | Low (with compression) | 90-95% (tunable) | Slower |

---

## Vector Storage

### Storage Formats

#### 1. Hash Storage (HASH)

**Format**: Binary bytes stored in Redis Hash field

**Python Example**:
```python
import numpy as np

# Create vector
vector = np.array([0.34, 0.63, -0.54, -0.69, 0.98, 0.61], dtype=np.float32)

# Convert to bytes
vector_bytes = vector.tobytes()

# Store in Redis Hash
r.hset(
    "docs:doc1",
    mapping={
        "content": "Document text",
        "doc_embedding": vector_bytes
    }
)
```

**Retrieval**:
```python
# Get vector bytes
vector_bytes = r.hget("docs:doc1", "doc_embedding")

# Convert back to numpy array
vector = np.frombuffer(vector_bytes, dtype=np.float32)
```

#### 2. JSON Storage (JSON)

**Format**: Array stored in Redis JSON document

**Python Example**:
```python
import json

# Store vector as JSON array
r.json().set(
    "docs:doc1",
    "$",
    {
        "content": "Document text",
        "doc_embedding": [0.34, 0.63, -0.54, -0.69, 0.98, 0.61]
    }
)
```

**Retrieval**:
```python
# Get JSON document
doc = r.json().get("docs:doc1")
vector = doc["doc_embedding"]
```

### Multi-Value Vectors

Redis supports storing multiple vectors per document:

```python
# Store multiple vectors per document
r.hset(
    "docs:doc1",
    mapping={
        "content": "Document text",
        "text_embedding": text_vector_bytes,
        "image_embedding": image_vector_bytes
    }
)

# Create index with multiple vector fields
schema = (
    VectorField("text_embedding", "HNSW", {...}),
    VectorField("image_embedding", "HNSW", {...}),
    TextField("content")
)
```

---

## Query Mechanisms

### 1. Vector Similarity Search (KNN)

**Query Type**: K-Nearest Neighbor search

**RedisVL Example**:
```python
from redisvl.query import VectorQuery

query = VectorQuery(
    vector=[0.1, 0.2, 0.3, ...],
    vector_field_name="doc_embedding",
    return_fields=["title", "content", "vector_distance"],
    num_results=10,
    return_score=True
)

results = index.query(query)
```

**Raw Redis Command**:
```
FT.SEARCH documents
  "(*)=>[KNN 10 @doc_embedding $vec AS score]"
  PARAMS 2 vec <vector_bytes>
  RETURN 3 title content score
  DIALECT 2
```

**With Runtime Parameters (HNSW)**:
```python
query = VectorQuery(
    vector=embedding,
    vector_field_name="doc_embedding",
    num_results=10,
    ef_runtime=50,  # Higher = better recall, slower
    epsilon=0.01    # Range search approximation
)
```

**With Runtime Parameters (SVS-VAMANA)**:
```python
query = VectorQuery(
    vector=embedding,
    vector_field_name="doc_embedding",
    num_results=10,
    search_window_size=40,      # Larger = better recall
    use_search_history='ON',     # Use search history
    search_buffer_capacity=50   # Buffer capacity
)
```

### 2. Vector Range Query

**Query Type**: Find vectors within distance threshold

**RedisVL Example**:
```python
from redisvl.query import VectorRangeQuery

query = VectorRangeQuery(
    vector=[0.1, 0.2, 0.3, ...],
    vector_field_name="doc_embedding",
    distance_threshold=0.5,  # Maximum distance
    return_fields=["title", "content"],
    num_results=10
)

results = index.query(query)
```

**Raw Redis Command**:
```
FT.SEARCH documents
  "@doc_embedding:[VECTOR_RANGE $radius $vec]"
  PARAMS 3 radius 0.5 vec <vector_bytes>
  RETURN 2 title content
  DIALECT 2
```

### 3. Hybrid Search (Vector + Text)

**Query Type**: Combines vector similarity with full-text search

**RedisVL Example**:
```python
from redisvl.query import AggregateHybridQuery

query = AggregateHybridQuery(
    text="running shoes",
    text_field_name="brief_description",
    vector=[0.1, 0.2, 0.1],
    vector_field_name="text_embedding",
    alpha=0.7,  # 70% vector, 30% text (default)
    return_fields=["product_id", "brief_description", "price"],
    num_results=5
)

results = index.query(query)
```

**Alpha Parameter**:
- `alpha=1.0`: Pure vector search
- `alpha=0.0`: Pure text search
- `alpha=0.7`: 70% vector, 30% text (default)

**Raw Redis Command**:
```
FT.SEARCH products
  "(@brief_description:running shoes)=>[KNN 5 @text_embedding $vec AS score]"
  PARAMS 2 vec <vector_bytes>
  RETURN 3 product_id brief_description price
  DIALECT 2
```

### 4. Filtered Vector Search

**Query Type**: Vector search with metadata filters

**RedisVL Example**:
```python
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, Num

# Tag filter
category_filter = Tag("category") == "electronics"

# Numeric filter
price_filter = Num("price").between(100, 500)

# Combined filter
combined_filter = category_filter & price_filter

query = VectorQuery(
    vector=embedding,
    vector_field_name="doc_embedding",
    filter_expression=combined_filter,
    return_fields=["title", "category", "price"],
    num_results=10
)

results = index.query(query)
```

**Raw Redis Command**:
```
FT.SEARCH products
  "@category:{electronics} @price:[100 500] =>[KNN 10 @doc_embedding $vec AS score]"
  PARAMS 2 vec <vector_bytes>
  RETURN 3 title category price
  DIALECT 2
```

### 5. Multi-Vector Query

**Query Type**: Search across multiple vector fields

**RedisVL Example**:
```python
from redisvl.query import MultiVectorQuery, Vector

# Define multiple vectors
text_vector = Vector(
    vector=[0.1, 0.2, 0.1],
    field_name="text_embedding",
    dtype="float32",
    weight=0.7  # 70% weight
)

image_vector = Vector(
    vector=[0.8, 0.1],
    field_name="image_embedding",
    dtype="float32",
    weight=0.3  # 30% weight
)

query = MultiVectorQuery(
    vectors=[text_vector, image_vector],
    return_fields=["product_id", "title"],
    num_results=5
)

results = index.query(query)
```

### 6. Text-Only Search

**Query Type**: Full-text search without vectors

**RedisVL Example**:
```python
from redisvl.query import TextQuery

query = TextQuery(
    text="running shoes",
    text_field_name="brief_description",
    text_scorer="BM25STD",  # or "TFIDF"
    return_fields=["product_id", "brief_description", "price"],
    num_results=5
)

results = index.query(query)
```

---

## Index Management

### Creating an Index

#### Using RedisVL (Python)

**From YAML Schema**:
```python
from redisvl.index import SearchIndex

index = SearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis://localhost:6379",
    validate_on_load=True
)

# Create the index
index.create(overwrite=True, drop=False)
```

**From Dictionary**:
```python
schema_dict = {
    "index": {
        "name": "my-index",
        "prefix": "docs",
        "storage_type": "hash"  # or "json"
    },
    "fields": [
        {
            "name": "title",
            "type": "text"
        },
        {
            "name": "doc_embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "dims": 1536,
                "distance_metric": "cosine",
                "datatype": "float32",
                "m": 16,
                "ef_construction": 200
            }
        }
    ]
}

index = SearchIndex.from_dict(
    schema_dict,
    redis_url="redis://localhost:6379"
)

index.create(overwrite=True)
```

**From Existing Index**:
```python
index = SearchIndex.from_existing(
    name="my-index",
    redis_url="redis://localhost:6379"
)
```

#### Using Raw Redis Commands

```redis
FT.CREATE documents
  ON HASH
  PREFIX 1 docs:
  SCHEMA
    title TEXT
    content TEXT
    doc_embedding VECTOR HNSW 10
      TYPE FLOAT32
      DIM 1536
      DISTANCE_METRIC COSINE
      M 16
      EF_CONSTRUCTION 200
```

### Loading Data

**Basic Load**:
```python
data = [
    {
        "title": "Product A",
        "content": "Description",
        "doc_embedding": vector_bytes
    },
    {
        "title": "Product B",
        "content": "Description",
        "doc_embedding": vector_bytes
    }
]

keys = index.load(data)
```

**Load with Custom Keys**:
```python
keys = index.load(data, keys=["product:1", "product:2"])
```

**Load with TTL**:
```python
keys = index.load(data, ttl=3600)  # 1 hour expiration
```

**Batch Loading**:
```python
keys = index.load(data, batch_size=100)
```

### Index Operations

**Check if Index Exists**:
```python
exists = index.exists()
```

**Get Index Info**:
```python
info = index.info()
# Returns: num_docs, max_doc_id, index_definition, etc.
```

**List All Indices**:
```python
all_indices = index.listall()
```

**Delete Index**:
```python
index.delete()
```

### Schema Updates

**Add Field**:
```python
index.schema.add_fields([
    {
        "name": "new_field",
        "type": "text"
    }
])

# Recreate index (requires data reload)
index.create(overwrite=True)
```

**Remove Field**:
```python
index.schema.remove_field("old_field")
index.create(overwrite=True)
```

---

## Performance Characteristics

### Query Performance

**FLAT Index**:
- **Latency**: O(n) - scales linearly with dataset size
- **100K vectors**: ~10-50ms
- **1M vectors**: ~100-500ms
- **Accuracy**: 100% recall

**HNSW Index**:
- **Latency**: O(log n) - logarithmic scaling
- **100K vectors**: ~1-5ms (EF_RUNTIME=10)
- **1M vectors**: ~5-20ms (EF_RUNTIME=10)
- **10M vectors**: ~20-50ms (EF_RUNTIME=10)
- **Accuracy**: 95-99% recall (tunable via EF_RUNTIME)

**SVS-VAMANA Index**:
- **Latency**: O(log n) - logarithmic scaling
- **With compression**: Similar to HNSW but lower memory
- **Accuracy**: 90-95% recall (tunable)

### Memory Usage

**FLAT Index**:
- **Overhead**: Minimal (~4 bytes per vector for metadata)
- **1M vectors (1536-dim)**: ~6GB vectors + minimal index overhead

**HNSW Index**:
- **Overhead**: ~20% additional memory for graph structure
- **1M vectors (1536-dim)**: ~6GB vectors + ~1.2GB graph = ~7.2GB total

**SVS-VAMANA with Compression**:
- **LVQ8**: ~75% memory reduction (8 bits vs 32 bits per dimension)
- **LeanVec4x8**: ~81% memory reduction (with dimensionality reduction)
- **1M vectors (1536-dim)**: ~1.4GB (with LeanVec4x8)

### Build Time

**FLAT Index**:
- **1M vectors**: ~10-30 seconds
- **No preprocessing required**

**HNSW Index**:
- **1M vectors**: ~2-5 minutes (EF_CONSTRUCTION=200)
- **10M vectors**: ~20-50 minutes
- **Scales with EF_CONSTRUCTION parameter**

**SVS-VAMANA Index**:
- **1M vectors**: ~3-7 minutes (with compression)
- **Slower than HNSW for smaller datasets**
- **Faster for very large datasets**

---

## Integration Patterns

### RedisVL (High-Level Python Library)

**Architecture**:
```
┌─────────────┐
│ Application │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   RedisVL SDK   │
│  - Query Builder│
│  - Schema Mgmt  │
│  - Vectorizers  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Redis Server  │
│  (Search/Query) │
└─────────────────┘
```

**Key Classes**:
- **SearchIndex**: Synchronous index operations
- **AsyncSearchIndex**: Asynchronous index operations
- **VectorQuery**: Vector similarity queries
- **HybridQuery**: Combined vector + text queries
- **TextQuery**: Full-text search queries
- **FilterQuery**: Metadata filtering

**Example**:
```python
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

# Initialize
index = SearchIndex.from_yaml("schema.yaml")

# Query
query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    num_results=10
)

results = index.query(query)
```

### LangCache (Semantic Caching)

**Architecture**:
```
┌─────────────┐
│   Client    │
│ Application │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  LangCache API  │
│  - Embedding Gen│
│  - Vector Search│
│  - Cache Mgmt   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Redis Server  │
│  (HNSW Index)   │
└─────────────────┘
```

**Use Case**: Semantic caching for LLM responses

**Flow**:
1. Client sends prompt to LangCache
2. LangCache generates embedding automatically
3. Vector similarity search against cached entries
4. **Cache Hit**: Return cached response
5. **Cache Miss**: Return empty, client calls LLM
6. Client stores new response in LangCache

**Example**:
```python
from langcache import LangCache

client = LangCache(api_key="...", cache_id="...")

# Search cache
result = client.search(
    prompt="What is Redis?",
    search_strategies=["semantic"]
)

if result.response:
    # Cache hit
    print(result.response)
else:
    # Cache miss - call LLM
    llm_response = call_llm(result.prompt)
    client.store(
        prompt=result.prompt,
        response=llm_response
    )
```

---

## Comparison with PostgreSQL pgvector

### Architecture Differences

| Aspect | Redis + RediSearch | PostgreSQL + pgvector |
|--------|-------------------|----------------------|
| **Storage** | In-memory (optional persistence) | Disk-based (with buffer cache) |
| **Index Types** | FLAT, HNSW, SVS-VAMANA | IVFFlat, HNSW |
| **Query Language** | FT.SEARCH (RediSearch) | SQL |
| **ACID** | Limited (single-threaded) | Full ACID |
| **Persistence** | Optional (RDB/AOF) | Always persistent |
| **Latency** | Sub-millisecond | Millisecond range |
| **Scalability** | Horizontal (Redis Cluster) | Vertical + horizontal |

### Performance Comparison

**Query Latency** (1M vectors, 1536-dim):
- **Redis HNSW**: ~5-20ms
- **PostgreSQL HNSW**: ~20-50ms

**Memory Usage**:
- **Redis**: In-memory, ~7.2GB for 1M vectors
- **PostgreSQL**: Disk-based, ~6GB + buffer cache

**Build Time** (1M vectors):
- **Redis HNSW**: ~2-5 minutes
- **PostgreSQL HNSW**: ~5-10 minutes

### Use Case Recommendations

**Choose Redis when**:
- Ultra-low latency required (<10ms)
- High-throughput workloads (millions QPS)
- Caching layer for AI applications
- In-memory data fits available RAM
- Simple deployment preferred

**Choose PostgreSQL when**:
- ACID guarantees required
- Complex SQL queries needed
- Data persistence critical
- Integration with existing PostgreSQL infrastructure
- Larger datasets that don't fit in memory

---

## Key Takeaways

### Redis Vector Search Strengths

1. **Ultra-Low Latency**: Sub-millisecond query times
2. **Multiple Algorithms**: FLAT, HNSW, SVS-VAMANA with compression
3. **Hybrid Search**: Combines vector + text + metadata filtering
4. **High-Level SDKs**: RedisVL and LangCache simplify development
5. **In-Memory Performance**: Leverages Redis's in-memory architecture

### Redis Vector Search Limitations

1. **Memory Constraints**: Requires sufficient RAM
2. **Limited ACID**: Single-threaded model limits transactions
3. **Persistence Trade-offs**: Optional persistence vs. performance
4. **No SQL**: Custom query language (FT.SEARCH)
5. **Single-Threaded**: Blocking operations impact throughput

### Best Practices

1. **Choose Algorithm Wisely**:
   - FLAT: < 100K vectors, perfect accuracy
   - HNSW: 100K - 1M+, balanced performance
   - SVS-VAMANA: Large datasets, memory constraints

2. **Tune Runtime Parameters**:
   - HNSW: Increase EF_RUNTIME for better recall
   - SVS-VAMANA: Tune search_window_size

3. **Use Compression**:
   - LVQ/LeanVec for memory-constrained environments
   - CompressionAdvisor for recommendations

4. **Hybrid Search**:
   - Combine vector + text for best results
   - Tune alpha parameter (0.7 default)

5. **Monitor Performance**:
   - Track query latency
   - Monitor memory usage
   - Tune EF_RUNTIME based on recall requirements

---

## References

- Redis Search and Vector Database - Python Implementation Guide
- RedisVL Technical Implementation Guide
- RedisVL User Guide Implementation Reference
- LangCache Technical Implementation Guide
- Redis research.md - Comprehensive Redis Analysis

---

**Analysis Date**: January 2026
**Redis Version**: Redis 8.x / Redis Stack
**RediSearch Version**: 2.8+

