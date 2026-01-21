# RedisVL (Redis Vector Library) 0.12.1: Technical Implementation Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation and Setup](#installation-and-setup)
3. [Core Architecture](#core-architecture)
4. [Schema Definition](#schema-definition)
5. [Index Management](#index-management)
6. [Vector Search Algorithms](#vector-search-algorithms)
7. [Query Types](#query-types)
8. [Vectorizers](#vectorizers)
9. [Rerankers](#rerankers)
10. [Caching Systems](#caching-systems)
11. [Message History](#message-history)
12. [Semantic Routing](#semantic-routing)
13. [Advanced Features](#advanced-features)
14. [Performance Optimization](#performance-optimization)
15. [Best Practices](#best-practices)

---

## Overview

### What is RedisVL?

RedisVL (Redis Vector Library) is a powerful, dedicated Python client library for Redis that enables seamless integration and management of high-dimensional vector data. Built to support machine learning and artificial intelligence workflows, RedisVL simplifies the process of storing, searching, and analyzing vector embeddings.

### Key Features

- **Vector Similarity Search**: Efficiently find nearest neighbors using algorithms like HNSW, FLAT, and SVS-VAMANA
- **Integration with AI Frameworks**: Works seamlessly with TensorFlow, PyTorch, and Hugging Face
- **Scalable and Fast**: Leverages Redis's in-memory architecture for low-latency access
- **Multiple Query Types**: Vector, hybrid, text, filter, and multi-vector queries
- **Built-in Caching**: Semantic cache for LLM responses and embeddings cache
- **Message History Management**: Track conversation context for AI applications
- **Semantic Routing**: Route queries to appropriate handlers based on semantic similarity

### Use Cases

- Recommendation systems
- Semantic search engines
- RAG (Retrieval Augmented Generation) applications
- AI agent memory and context management
- Content similarity matching
- Anomaly detection

---

## Installation and Setup

### Prerequisites

- Python >= 3.8
- Redis instance with Search and Query capability (Redis Stack, Redis Cloud, or Redis Enterprise)

### Installation Methods

#### Standard Installation

```bash
pip install redisvl
```

#### With Optional Dependencies

```bash
# Install with all vectorizer dependencies
pip install redisvl[all]

# Install with dev dependencies
pip install redisvl[dev]

# Install with hiredis for performance
pip install redisvl[hiredis]
```

#### From Source

```bash
git clone https://github.com/redis/redis-vl-python.git
cd redisvl
pip install .

# For editable installation (developers)
pip install -e .
```

### Redis Setup Options

#### Redis Cloud (Recommended for Production)

1. Sign up at [redis.io/cloud](https://redis.io/cloud)
2. Create a database with "Search and Query" capability enabled
3. Copy connection details (host, port, password)

#### Redis Stack (Local Development)

```bash
docker run -d --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

Redis Insight GUI available at `http://localhost:8001`

#### Redis Enterprise (Self-Hosted)

Download from [redis.io/downloads](https://redis.io/downloads) or use the [Redis Enterprise Operator](https://docs.redis.com/latest/kubernetes/) for Kubernetes deployments.

### Sentinel Support

For high availability deployments:

```python
from redisvl.index import SearchIndex

# Connect via Sentinel
# Format: redis+sentinel://[username:password@]host1:port1,host2:port2/service_name[/db]
index = SearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
)

# With authentication
index = SearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis+sentinel://user:pass@sentinel1:26379,sentinel2:26379/mymaster/0"
)
```

---

## Core Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│         RedisVL Application             │
├─────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │ Schema   │  │  Index   │  │ Query │ │
│  │ Manager  │  │ Manager  │  │ Types │ │
│  └────┬─────┘  └────┬─────┘  └───┬───┘ │
│       │              │             │     │
│  ┌────▼──────────────▼─────────────▼───┐ │
│  │      SearchIndex / AsyncSearchIndex │ │
│  └──────────────┬──────────────────────┘ │
└─────────────────┼───────────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │   Redis Server  │
         │  (Search/Query) │
         └─────────────────┘
```

### Key Classes

1. **IndexSchema**: Defines index structure and field configurations
2. **SearchIndex**: Synchronous index operations
3. **AsyncSearchIndex**: Asynchronous index operations
4. **Query Classes**: VectorQuery, HybridQuery, TextQuery, FilterQuery, etc.
5. **Vectorizers**: Text-to-vector conversion (OpenAI, HuggingFace, etc.)
6. **Rerankers**: Post-search result reranking
7. **Caches**: SemanticCache, EmbeddingsCache
8. **MessageHistory**: Conversation context management
9. **SemanticRouter**: Query routing based on semantic similarity

---

## Schema Definition

### Schema Structure

A RedisVL schema consists of three components:

1. **version**: Schema specification version (currently '0.1.0')
2. **index**: Index-level settings (name, prefix, storage type, stopwords)
3. **fields**: Field definitions with types and attributes

### YAML Schema Example

```yaml
version: '0.1.0'

index:
  name: product-index
  prefix: product
  key_separator: ":"
  storage_type: json
  stopwords: []  # Disable stopwords

fields:
  - name: title
    type: text
    path: $.title
    attrs:
      weight: 1.0
      no_stem: false
      withsuffixtrie: true
  
  - name: category
    type: tag
    attrs:
      separator: ","
      case_sensitive: false
  
  - name: price
    type: numeric
    attrs:
      sortable: true
  
  - name: location
    type: geo
    attrs:
      sortable: true
  
  - name: embedding
    type: vector
    attrs:
      algorithm: hnsw
      dims: 768
      distance_metric: cosine
      datatype: float32
      m: 16
      ef_construction: 200
```

### Python Dictionary Schema

```python
from redisvl.schema import IndexSchema

schema = IndexSchema.from_dict({
    "index": {
        "name": "product-index",
        "prefix": "product",
        "key_separator": ":",
        "storage_type": "json",
        "stopwords": []  # Disable stopwords
    },
    "fields": [
        {"name": "title", "type": "text", "attrs": {"weight": 1.0}},
        {"name": "category", "type": "tag"},
        {"name": "price", "type": "numeric", "attrs": {"sortable": True}},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "dims": 768,
                "distance_metric": "cosine",
                "datatype": "float32"
            }
        }
    ]
})
```

### Field Types

#### Text Fields

Full-text search with stemming, phonetic matching, and text analysis.

```python
from redisvl.schema import TextField

TextField(
    name="description",
    attrs={
        "weight": 1.0,
        "no_stem": False,
        "withsuffixtrie": True,
        "phonetic_matcher": "dm:en"
    }
)
```

**Attributes:**
- `weight`: Field importance (default: 1.0)
- `no_stem`: Disable stemming
- `withsuffixtrie`: Optimize suffix queries
- `phonetic_matcher`: Phonetic matching (dm:en, dm:fr, etc.)
- `sortable`: Enable sorting
- `index_missing`: Index documents without this field

#### Tag Fields

Exact-match filtering and faceted search on categorical data.

```python
from redisvl.schema import TagField

TagField(
    name="status",
    attrs={
        "separator": ",",
        "case_sensitive": False,
        "sortable": True
    }
)
```

**Attributes:**
- `separator`: How to split tag values (default: ",")
- `case_sensitive`: Case sensitivity (default: False)
- `sortable`: Enable sorting
- `withsuffixtrie`: Optimize suffix queries

#### Numeric Fields

Range queries and sorting on numeric data.

```python
from redisvl.schema import NumericField

NumericField(
    name="age",
    attrs={
        "sortable": True,
        "unf": False  # Un-normalized form
    }
)
```

**Attributes:**
- `sortable`: Enable sorting
- `unf`: Disable normalization for sorting
- `index_missing`: Index documents without this field

#### Geo Fields

Location-based search with geographic coordinates.

```python
from redisvl.schema import GeoField

GeoField(
    name="location",
    attrs={
        "sortable": True
    }
)
```

**Attributes:**
- `sortable`: Enable sorting
- `index_missing`: Index documents without this field

---

## Index Management

### Creating an Index

#### From YAML File

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

#### From Dictionary

```python
from redisvl.index import SearchIndex

schema_dict = {
    "index": {"name": "my-index", "prefix": "docs"},
    "fields": [{"name": "text", "type": "text"}]
}

index = SearchIndex.from_dict(
    schema_dict,
    redis_url="redis://localhost:6379"
)

index.create(overwrite=True)
```

#### From Existing Index

```python
# Load existing index by name
index = SearchIndex.from_existing(
    name="my-index",
    redis_url="redis://localhost:6379"
)
```

### Index Operations

#### Load Data

```python
# Basic load
data = [
    {"title": "Product A", "price": 100, "embedding": vector_bytes},
    {"title": "Product B", "price": 200, "embedding": vector_bytes}
]

keys = index.load(data)

# Load with custom IDs
keys = index.load(data, id_field="product_id")

# Load with explicit keys
keys = index.load(data, keys=["product:1", "product:2"])

# Load with TTL
keys = index.load(data, ttl=3600)  # 1 hour expiration

# Load with preprocessing
def preprocess(doc):
    doc["processed"] = True
    return doc

keys = index.load(data, preprocess=preprocess)

# Batch loading
keys = index.load(data, batch_size=100)
```

#### Fetch Documents

```python
# Fetch by ID
doc = index.fetch("product:1")

# Fetch multiple (using client directly)
docs = index.client.mget(["product:1", "product:2"])
```

#### Update Documents

```python
# Update specific fields
index.set_client(index.client)
index.client.hset(
    "product:1",
    mapping={"price": 150, "status": "updated"}
)

# For JSON storage
index.client.json().set("product:1", "$.price", 150)
```

#### Delete Documents

```python
# Delete by document IDs
count = index.drop_documents(["product:1", "product:2"])

# Delete by Redis keys
count = index.drop_keys(["product:1", "product:2"])

# Clear all documents (keeps index)
count = index.clear()

# Delete index and all documents
index.delete(drop=True)
```

#### Index Information

```python
# Check if index exists
exists = index.exists()

# Get index info
info = index.info()

# List all indices
all_indices = index.listall()
```

### Async Index Operations

```python
from redisvl.index import AsyncSearchIndex

async_index = AsyncSearchIndex.from_yaml(
    "schema.yaml",
    redis_url="redis://localhost:6379"
)

# Create index
await async_index.create(overwrite=True)

# Load data
keys = await async_index.load(data)

# Query
results = await async_index.query(query)

# Fetch
doc = await async_index.fetch("product:1")

# Disconnect
await async_index.disconnect()
```

---

## Vector Search Algorithms

### Algorithm Comparison

| Algorithm | Best For | Performance | Memory Usage | Recall Quality |
|-----------|----------|-------------|--------------|----------------|
| **FLAT** | Small datasets (<100K) | 100% recall, O(n) | Minimal | 100% (exact) |
| **HNSW** | General purpose (100K-1M+) | 95-99% recall, O(log n) | Moderate | 95-99% (tunable) |
| **SVS-VAMANA** | Large datasets, memory constraints | 90-95% recall, O(log n) | Low (with compression) | 90-95% (tunable) |

### FLAT Algorithm

**Use when:**
- Dataset size < 100,000 vectors
- Exact results are mandatory
- Simple setup preferred
- Query latency not critical

**Configuration:**

```yaml
- name: embedding
  type: vector
  attrs:
    algorithm: flat
    dims: 768
    distance_metric: cosine
    datatype: float32
    block_size: 1024  # Optional: tune for batch processing
```

**Performance:**
- Search accuracy: 100% exact results
- Search speed: Linear time O(n)
- Memory usage: Minimal overhead
- Build time: Fastest (no preprocessing)

### HNSW Algorithm

**Use when:**
- Dataset size 100K - 1M+ vectors
- Need balanced speed and accuracy
- Cross-platform compatibility required
- Most common choice for production

**Configuration:**

```yaml
- name: embedding
  type: vector
  attrs:
    algorithm: hnsw
    dims: 768
    distance_metric: cosine
    datatype: float32
    # Index-time parameters
    m: 16                    # Graph connectivity (8-64)
    ef_construction: 200     # Build-time accuracy (100-800)
    # Runtime parameters (set at query time)
    # ef_runtime: 10         # Query-time accuracy (default: 10)
    # epsilon: 0.01          # Range search factor (default: 0.01)
```

**Performance:**
- Search speed: Logarithmic time O(log n)
- Memory usage: Moderate (graph overhead ~20%)
- Recall quality: 95-99% (tunable via `ef_runtime`)
- Build time: Moderate

**Tuning Guidelines:**

**Balanced Configuration (Recommended):**
```yaml
m: 16
ef_construction: 200
# Query time: ef_runtime=10
```

**High-Recall Configuration:**
```yaml
m: 32
ef_construction: 400
# Query time: ef_runtime=50
```

**High-Speed Configuration:**
```yaml
m: 16
ef_construction: 200
# Query time: ef_runtime=5
```

### SVS-VAMANA Algorithm

**Use when:**
- Dataset size > 100K vectors
- Memory usage is primary concern
- Running on Intel hardware
- Can accept 90-95% recall for memory savings

**Requirements:**
- Redis >= 8.2.0 with RediSearch >= 2.8.10
- Datatype: float16 or float32 only

**Configuration:**

```yaml
- name: embedding
  type: vector
  attrs:
    algorithm: svs-vamana
    dims: 768
    distance_metric: cosine
    datatype: float32
    # Index-time parameters
    graph_max_degree: 40
    construction_window_size: 250
    # Compression (optional)
    compression: LeanVec4x8
    reduce: 384  # Dimensionality reduction (for LeanVec only)
    training_threshold: 1000
    # Runtime parameters (set at query time)
    # search_window_size: 20
    # epsilon: 0.01
    # use_search_history: AUTO
    # search_buffer_capacity: 20
```

**Compression Types:**

1. **No Compression**: Best performance, standard memory
2. **LVQ4/LVQ8**: Good balance (2x-4x compression)
3. **LeanVec4x8/LeanVec8x8**: Maximum compression (up to 8x) with dimensionality reduction

**Memory Savings Examples (1M vectors, 768 dims):**
- No compression (float32): 3.1 GB
- LVQ4x4 compression: 1.6 GB (~48% savings)
- LeanVec4x8 + reduce to 384: 580 MB (~81% savings)

**Performance:**
- Search speed: Logarithmic time O(log n)
- Memory usage: Low (with compression)
- Recall quality: 90-95% (tunable via `search_window_size`)
- Build time: Slower than HNSW for smaller datasets

**CompressionAdvisor Utility:**

```python
from redisvl.schema import CompressionAdvisor

# Get recommendations
config = CompressionAdvisor.recommend(
    dims=1536,
    priority="balanced"  # "memory", "speed", or "balanced"
)
# Returns: compression type, reduce value, datatype, etc.

# Estimate memory savings
savings = CompressionAdvisor.estimate_memory_savings(
    compression="LeanVec4x8",
    dims=1536,
    reduce=768
)
# Returns: 81.2 (percentage saved)
```

---

## Query Types

### VectorQuery

Basic vector similarity search.

```python
from redisvl.query import VectorQuery

query = VectorQuery(
    vector=[0.1, 0.2, 0.3, ...],
    vector_field_name="embedding",
    return_fields=["title", "price", "vector_distance"],
    num_results=10,
    return_score=True,
    filter_expression=Tag("category") == "electronics"
)

results = index.query(query)
```

**Runtime Parameters (HNSW):**

```python
query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    num_results=10,
    ef_runtime=50,  # Higher = better recall, slower
    epsilon=0.01    # Range search approximation
)
```

**Runtime Parameters (SVS-VAMANA):**

```python
query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    num_results=10,
    search_window_size=40,  # Higher = better recall, slower
    use_search_history="ON",  # OFF, ON, or AUTO
    search_buffer_capacity=40,
    epsilon=0.01
)
```

**Hybrid Policy:**

```python
query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    filter_expression=Tag("status") == "active",
    hybrid_policy="BATCHES",  # or "ADHOC_BF"
    batch_size=100  # For BATCHES policy
)
```

### VectorRangeQuery

Find vectors within a distance threshold.

```python
from redisvl.query import VectorRangeQuery

query = VectorRangeQuery(
    vector=[0.1, 0.2, 0.3, ...],
    vector_field_name="embedding",
    radius=0.5,  # Maximum distance
    return_fields=["title", "price"],
    filter_expression=Num("price") < 100
)

results = index.query(query)
```

### HybridQuery

Combine vector search with text search.

```python
from redisvl.query import HybridQuery

query = HybridQuery(
    vector=[0.1, 0.2, 0.3, ...],
    vector_field_name="embedding",
    text_query="laptop computer",
    text_fields=["title", "description"],
    return_fields=["title", "price", "vector_distance"],
    num_results=10
)

results = index.query(query)
```

### TextQuery

Full-text search without vectors.

```python
from redisvl.query import TextQuery

query = TextQuery(
    text_query="laptop computer",
    text_fields=["title", "description"],
    return_fields=["title", "price"],
    num_results=10,
    filter_expression=Num("price") < 1000
)

results = index.query(query)
```

### FilterQuery

Metadata filtering only.

```python
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag, Num

query = FilterQuery(
    filter_expression=(
        Tag("category") == "electronics" &
        Num("price") >= 100 &
        Num("price") <= 500
    ),
    return_fields=["title", "price"],
    sort_by="price",
    num_results=20
)

results = index.query(query)
```

### MultiVectorQuery

Search across multiple vector fields.

```python
from redisvl.query import MultiVectorQuery
from redisvl.query.vector import Vector

query = MultiVectorQuery(
    vectors=[
        Vector(
            vector=[0.1, 0.2, ...],
            field_name="title_embedding",
            weight=0.7
        ),
        Vector(
            vector=[0.3, 0.4, ...],
            field_name="description_embedding",
            weight=0.3
        )
    ],
    return_fields=["title", "price"],
    num_results=10
)

results = index.query(query)
```

### CountQuery

Count documents matching criteria.

```python
from redisvl.query import CountQuery
from redisvl.query.filter import Tag

query = CountQuery(
    filter_expression=Tag("category") == "electronics"
)

count = index.query(query)
```

### Filter Expressions

#### Combining Filters

```python
from redisvl.query.filter import Tag, Num, Text, Geo

# AND operation
filter_expr = (
    Tag("category") == "electronics" &
    Num("price") >= 100 &
    Num("price") <= 500
)

# OR operation
filter_expr = (
    Tag("category") == "electronics" |
    Tag("category") == "computers"
)

# Complex combinations
filter_expr = (
    (Tag("category") == "electronics" | Tag("category") == "computers") &
    Num("price") >= 100 &
    Num("price") <= 1000
)
```

#### Tag Filters

```python
from redisvl.query.filter import Tag

# Equality
Tag("status") == "active"

# Inequality
Tag("status") != "inactive"

# Multiple values
Tag("tags") == ["python", "redis", "ai"]
```

#### Numeric Filters

```python
from redisvl.query.filter import Num

# Range queries
Num("price") >= 100
Num("price") <= 500
Num("age") > 18
Num("age") < 65

# Combined range
Num("price") >= 100 & Num("price") <= 500
```

#### Text Filters

```python
from redisvl.query.filter import Text

# Exact match
Text("job") == "engineer"

# LIKE with wildcards
Text("job") % "engine*"  # Suffix wildcard
Text("job") % "*engine*"  # Contains

# Fuzzy match
Text("job") % "%%engine%%"  # Levenshtein distance

# Multiple terms
Text("job") % "engineer|doctor"  # OR
Text("job") % "engineer doctor"  # AND
```

#### Geo Filters

```python
from redisvl.query.filter import Geo, GeoRadius

# Within radius
Geo("location").within_radius(
    center=(37.7749, -122.4194),  # (lat, lon)
    radius=5000,  # meters
    unit="m"
)

# Within bounding box
Geo("location").within_bounds(
    min_lat=37.0,
    min_lon=-123.0,
    max_lat=38.0,
    max_lon=-122.0
)
```

---

## Vectorizers

### Overview

Vectorizers convert text into vector embeddings for semantic search.

### HFTextVectorizer (HuggingFace)

**Use when:** Running models on your own hardware without API costs.

```python
from redisvl.utils.vectorizers import HFTextVectorizer

vectorizer = HFTextVectorizer(
    model="sentence-transformers/all-mpnet-base-v2",
    dtype="float32"
)

# Single embedding
embedding = vectorizer.embed("Hello, world!")

# Batch embeddings
embeddings = vectorizer.embed_many(
    ["Text 1", "Text 2", "Text 3"],
    batch_size=32
)

# With caching
from redisvl.extensions.cache.embeddings import EmbeddingsCache
cache = EmbeddingsCache(name="hf_cache")
vectorizer = HFTextVectorizer(
    model="sentence-transformers/all-mpnet-base-v2",
    cache=cache
)
```

### OpenAITextVectorizer

**Use when:** Need high-quality embeddings via API.

```python
from redisvl.utils.vectorizers import OpenAITextVectorizer

vectorizer = OpenAITextVectorizer(
    model="text-embedding-ada-002",
    api_config={"api_key": "your-key"}  # Or set OPENAI_API_KEY env var
)

embedding = vectorizer.embed("Hello, world!")

# Async batch processing
embeddings = await vectorizer.aembed_many(
    ["Text 1", "Text 2"],
    batch_size=100
)
```

### AzureOpenAITextVectorizer

```python
from redisvl.utils.vectorizers import AzureOpenAITextVectorizer

vectorizer = AzureOpenAITextVectorizer(
    model="text-embedding-ada-002",
    api_config={
        "azure_endpoint": "https://your-endpoint.openai.azure.com",
        "api_version": "2023-05-15",
        "api_key": "your-key"
    }
)
```

### CohereTextVectorizer

```python
from redisvl.utils.vectorizers import CohereTextVectorizer

vectorizer = CohereTextVectorizer(
    model="embed-english-v3.0",
    api_config={"api_key": "your-key"}  # Or set COHERE_API_KEY
)
```

### VertexAITextVectorizer (Google Cloud)

```python
from redisvl.utils.vectorizers import VertexAITextVectorizer

vectorizer = VertexAITextVectorizer(
    model="textembedding-gecko@003",
    api_config={
        "project_id": "your-project",
        "location": "us-central1"
    }
)
```

### BedrockTextVectorizer (AWS)

```python
from redisvl.utils.vectorizers import BedrockTextVectorizer

vectorizer = BedrockTextVectorizer(
    model="amazon.titan-embed-text-v1",
    api_config={
        "aws_access_key_id": "your-key",
        "aws_secret_access_key": "your-secret",
        "region_name": "us-east-1"
    }
)
```

### VoyageAITextVectorizer

```python
from redisvl.utils.vectorizers import VoyageAITextVectorizer

vectorizer = VoyageAITextVectorizer(
    model="voyage-large-2",
    api_config={"api_key": "your-key"}
)
```

### CustomTextVectorizer

```python
from redisvl.utils.vectorizers import CustomTextVectorizer

def my_embedding_function(text: str) -> List[float]:
    # Your custom embedding logic
    return [0.1, 0.2, 0.3, ...]

vectorizer = CustomTextVectorizer(
    embedding_function=my_embedding_function,
    dims=768
)
```

---

## Rerankers

### Overview

Rerankers improve search result relevance by reordering results based on query-document similarity.

### CohereReranker

```python
from redisvl.utils.rerank import CohereReranker

reranker = CohereReranker(
    model="rerank-english-v3.0",
    rank_by=["content"],  # Fields to rank by
    limit=5,
    return_score=True
)

# Rerank search results
reranked = reranker.rank(
    query="What is machine learning?",
    docs=[
        {"content": "Document 1 text..."},
        {"content": "Document 2 text..."},
        {"content": "Document 3 text..."}
    ]
)
```

### HFCrossEncoderReranker

```python
from redisvl.utils.rerank import HFCrossEncoderReranker

reranker = HFCrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    limit=5,
    return_score=True
)

reranked = reranker.rank(
    query="What is machine learning?",
    docs=[...]
)
```

### VoyageAIReranker

```python
from redisvl.utils.rerank import VoyageAIReranker

reranker = VoyageAIReranker(
    model="rerank-lite-1",
    api_config={"api_key": "your-key"},
    limit=5
)

reranked = reranker.rank(query="...", docs=[...])
```

### Integration Pattern

```python
# 1. Perform vector search
query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    num_results=20  # Get more results for reranking
)

results = index.query(query)

# 2. Extract documents
docs = [{"content": r["content"]} for r in results]

# 3. Rerank
reranker = CohereReranker(limit=5)
reranked = reranker.rank(
    query=user_query,
    docs=docs
)

# 4. Use reranked results
for doc in reranked:
    print(doc["content"])
```

---

## Caching Systems

### SemanticCache (LLM Response Caching)

Cache LLM responses using semantic similarity.

```python
from redisvl.extensions.cache import SemanticCache
from redisvl.utils.vectorizers import OpenAITextVectorizer

# Initialize cache
cache = SemanticCache(
    name="llm_cache",
    distance_threshold=0.1,  # Semantic similarity threshold
    ttl=3600,  # 1 hour expiration
    vectorizer=OpenAITextVectorizer(model="text-embedding-ada-002"),
    redis_url="redis://localhost:6379"
)

# Check cache before calling LLM
cached = cache.check(
    prompt="What is Redis?",
    num_results=1,
    distance_threshold=0.1
)

if cached:
    response = cached[0]["response"]
else:
    # Cache miss - call LLM
    response = await llm.generate("What is Redis?")
    
    # Store in cache
    cache.store(
        prompt="What is Redis?",
        response=response,
        metadata={"model": "gpt-4", "temperature": 0.7}
    )
```

**With Filters:**

```python
from redisvl.query.filter import Tag

cached = cache.check(
    prompt="What is Redis?",
    filter_expression=Tag("user_id") == "user123",
    num_results=1
)
```

**Async Operations:**

```python
# Async check
cached = await cache.acheck(prompt="What is Redis?")

# Async store
key = await cache.astore(
    prompt="What is Redis?",
    response="Redis is...",
    metadata={"model": "gpt-4"}
)

# Async update
await cache.aupdate(key, metadata={"hit_count": 5})

# Async drop
await cache.adrop(ids=["entry1", "entry2"])
```

### EmbeddingsCache

Cache embedding vectors with exact key matching.

```python
from redisvl.extensions.cache.embeddings import EmbeddingsCache

cache = EmbeddingsCache(
    name="embed_cache",
    ttl=86400,  # 24 hours
    redis_url="redis://localhost:6379"
)

# Store embedding
key = cache.set(
    text="What is machine learning?",
    model_name="text-embedding-ada-002",
    embedding=[0.1, 0.2, 0.3, ...],
    metadata={"source": "user_query"}
)

# Retrieve embedding
embedding_data = cache.get(
    text="What is machine learning?",
    model_name="text-embedding-ada-002"
)

if embedding_data:
    embedding = embedding_data["embedding"]
else:
    # Generate and cache
    embedding = vectorizer.embed("What is machine learning?")
    cache.set(
        text="What is machine learning?",
        model_name="text-embedding-ada-002",
        embedding=embedding
    )
```

**Batch Operations:**

```python
# Batch store
keys = cache.mset([
    {
        "text": "Text 1",
        "model_name": "text-embedding-ada-002",
        "embedding": [0.1, 0.2, ...],
        "metadata": {"source": "docs"}
    },
    {
        "text": "Text 2",
        "model_name": "text-embedding-ada-002",
        "embedding": [0.3, 0.4, ...]
    }
])

# Batch get
embeddings = cache.mget(
    texts=["Text 1", "Text 2"],
    model_name="text-embedding-ada-002"
)

# Batch exists check
exists_results = cache.mexists(
    texts=["Text 1", "Text 2"],
    model_name="text-embedding-ada-002"
)
```

---

## Message History

### MessageHistory (Simple)

Store conversation history without semantic search.

```python
from redisvl.extensions.message_history import MessageHistory

history = MessageHistory(
    name="conversation_history",
    session_tag="user123_session1",
    redis_url="redis://localhost:6379"
)

# Store messages
history.store(
    prompt="What is Redis?",
    response="Redis is an in-memory data store..."
)

# Get recent messages
recent = history.get_recent(
    top_k=5,
    as_text=True,  # Return as single string
    role="user"  # Filter by role
)

# Get all messages
all_messages = history.messages
```

### SemanticMessageHistory

Store conversation history with semantic search capabilities.

```python
from redisvl.extensions.message_history import SemanticMessageHistory
from redisvl.utils.vectorizers import OpenAITextVectorizer

history = SemanticMessageHistory(
    name="semantic_history",
    session_tag="user123_session1",
    vectorizer=OpenAITextVectorizer(),
    distance_threshold=0.3,
    redis_url="redis://localhost:6379"
)

# Store messages
history.store(
    prompt="What is Redis?",
    response="Redis is..."
)

# Get semantically relevant messages
relevant = history.get_relevant(
    prompt="Tell me about caching",
    top_k=3,
    as_text=False,
    distance_threshold=0.3,
    fall_back=True  # Fall back to recent if no matches
)

# Get recent messages
recent = history.get_recent(top_k=5, role=["user", "assistant"])
```

**Multi-User Management:**

```python
# Different sessions for different users
user1_history = SemanticMessageHistory(
    name="history",
    session_tag="user1_session1"
)

user2_history = SemanticMessageHistory(
    name="history",
    session_tag="user2_session1"
)

# Store messages in respective sessions
user1_history.store("Hello", "Hi there!")
user2_history.store("Hello", "Hi there!")

# Retrieve user-specific history
user1_messages = user1_history.get_recent(session_tag="user1_session1")
```

---

## Semantic Routing

### Overview

Route queries to appropriate handlers based on semantic similarity to predefined routes.

### Basic Usage

```python
from redisvl.extensions.router import SemanticRouter, Route
from redisvl.utils.vectorizers import OpenAITextVectorizer

# Define routes
routes = [
    Route(
        name="technical_support",
        references=[
            "How do I fix an error?",
            "Technical issue",
            "Bug report"
        ],
        distance_threshold=0.5,
        metadata={"handler": "tech_support", "priority": "high"}
    ),
    Route(
        name="sales",
        references=[
            "I want to buy",
            "Pricing information",
            "Product features"
        ],
        distance_threshold=0.5,
        metadata={"handler": "sales_team"}
    ),
    Route(
        name="general",
        references=[
            "General question",
            "Information request"
        ],
        distance_threshold=0.6
    )
]

# Initialize router
router = SemanticRouter(
    name="customer_service_router",
    routes=routes,
    vectorizer=OpenAITextVectorizer(),
    redis_url="redis://localhost:6379"
)

# Route a query
match = router.route(
    statement="I'm having trouble with my account",
    max_k=1
)

if match:
    route_name = match.name
    distance = match.distance
    # Route to appropriate handler based on route_name
```

### Multiple Matches

```python
# Get multiple route matches
matches = router.route_many(
    statement="I need help with pricing and technical support",
    max_k=2,
    distance_threshold=0.6
)

for match in matches:
    print(f"Route: {match.name}, Distance: {match.distance}")
```

### Dynamic Route Management

```python
# Add references to existing route
router.add_route_references(
    route_name="technical_support",
    references=["Account login issues", "Password reset"]
)

# Get route references
refs = router.get_route_references(route_name="technical_support")

# Delete route references
router.delete_route_references(
    route_name="technical_support",
    reference_ids=["ref1", "ref2"]
)

# Remove entire route
router.remove_route("general")

# Update route thresholds
router.update_route_thresholds({
    "technical_support": 0.4,
    "sales": 0.5
})
```

### Routing Configuration

```python
from redisvl.extensions.router import RoutingConfig, DistanceAggregationMethod

config = RoutingConfig(
    max_k=1,  # Maximum routes to return
    aggregation_method=DistanceAggregationMethod.avg  # avg, min, or sum
)

router = SemanticRouter(
    name="router",
    routes=routes,
    routing_config=config
)

# Update config
router.update_routing_config(
    RoutingConfig(max_k=2, aggregation_method=DistanceAggregationMethod.min)
)
```

### Serialization

```python
# Save router to YAML
router.to_yaml("router.yaml")

# Load router from YAML
router = SemanticRouter.from_yaml("router.yaml")

# Convert to dictionary
router_dict = router.to_dict()

# Create from dictionary
router = SemanticRouter.from_dict(router_dict)
```

---

## Advanced Features

### Batch Operations

```python
# Batch queries
queries = [
    VectorQuery(vector=v1, vector_field_name="embedding", num_results=5),
    VectorQuery(vector=v2, vector_field_name="embedding", num_results=5),
    VectorQuery(vector=v3, vector_field_name="embedding", num_results=5)
]

results = index.batch_query(queries, batch_size=10)

# Batch search (raw Redis API)
results = index.batch_search(queries, batch_size=10)
```

### Pagination

```python
query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    num_results=10
)

# Paginate results
for result_batch in index.paginate(query, page_size=10):
    for result in result_batch:
        print(result)
```

### Aggregation

```python
from redis import AggregationRequest, reducers

# Group by category and calculate average price
agg_request = AggregationRequest("*").group_by(
    "@category",
    reducers.avg("@price").alias("avg_price")
)

results = index.aggregate(agg_request)
```

### Key Expiration

```python
# Set expiration on specific keys
index.expire_keys(
    keys=["product:1", "product:2"],
    ttl=3600  # 1 hour
)

# Set expiration during load
keys = index.load(data, ttl=3600)
```

### Data Validation

```python
# Enable validation on load
index = SearchIndex.from_yaml(
    "schema.yaml",
    validate_on_load=True  # Validates data against schema
)

# Invalid data will raise SchemaValidationError
try:
    index.load([{"embedding": "invalid"}])
except SchemaValidationError as e:
    print(f"Validation failed: {e}")
```

### Schema Manipulation

```python
# Add field to schema
schema.add_field({
    "name": "new_field",
    "type": "tag"
})

# Add multiple fields
schema.add_fields([
    {"name": "field1", "type": "text"},
    {"name": "field2", "type": "numeric"}
])

# Remove field
schema.remove_field("field1")

# Convert to YAML
schema.to_yaml("updated_schema.yaml")

# Convert to dictionary
schema_dict = schema.to_dict()
```

---

## Performance Optimization

### Connection Pooling

```python
from redis import ConnectionPool
from redisvl.index import SearchIndex

# Create connection pool
pool = ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50,
    decode_responses=False
)

# Use pool with index
index = SearchIndex.from_yaml(
    "schema.yaml",
    redis_client=redis.Redis(connection_pool=pool)
)
```

### Batch Loading Optimization

```python
# Optimize batch size based on document size
# Small documents: batch_size=1000
# Large documents: batch_size=100

keys = index.load(
    data,
    batch_size=500,  # Tune based on document size
    preprocess=lambda doc: doc  # Optional preprocessing
)
```

### Query Performance Tuning

**HNSW Tuning:**

```python
# Balance speed vs accuracy
query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    num_results=10,
    ef_runtime=10,  # Lower = faster, higher = more accurate
    epsilon=0.01   # Range query approximation
)
```

**SVS-VAMANA Tuning:**

```python
query = VectorQuery(
    vector=embedding,
    vector_field_name="embedding",
    num_results=10,
    search_window_size=20,  # Primary tuning parameter
    use_search_history="AUTO",
    search_buffer_capacity=20
)
```

### Memory Optimization

**Use SVS-VAMANA with Compression:**

```python
# For large datasets with memory constraints
schema = IndexSchema.from_dict({
    "index": {"name": "large_index", "prefix": "docs"},
    "fields": [{
        "name": "embedding",
        "type": "vector",
        "attrs": {
            "algorithm": "svs-vamana",
            "dims": 1536,
            "compression": "LeanVec4x8",
            "reduce": 768,  # 50% dimensionality reduction
            "datatype": "float16"
        }
    }]
})
```

**Use Appropriate Storage Type:**

```python
# Hash: Better for flat data, smaller memory footprint
# JSON: Better for nested data, more flexible queries

# Hash storage
schema = IndexSchema.from_dict({
    "index": {"name": "index", "prefix": "docs", "storage_type": "hash"},
    "fields": [...]
})

# JSON storage
schema = IndexSchema.from_dict({
    "index": {"name": "index", "prefix": "docs", "storage_type": "json"},
    "fields": [...]
})
```

### Async Operations

```python
import asyncio
from redisvl.index import AsyncSearchIndex

async def main():
    index = AsyncSearchIndex.from_yaml(
        "schema.yaml",
        redis_url="redis://localhost:6379"
    )
    
    # Concurrent queries
    queries = [
        VectorQuery(vector=v1, vector_field_name="embedding"),
        VectorQuery(vector=v2, vector_field_name="embedding"),
        VectorQuery(vector=v3, vector_field_name="embedding")
    ]
    
    results = await asyncio.gather(*[
        index.query(q) for q in queries
    ])
    
    await index.disconnect()

asyncio.run(main())
```

---

## Best Practices

### Schema Design

1. **Choose Appropriate Field Types**
   - Use `tag` for categorical data (status, category, etc.)
   - Use `text` for full-text search
   - Use `numeric` for range queries
   - Use `geo` for location-based queries

2. **Vector Algorithm Selection**
   - FLAT: < 100K vectors, exact results needed
   - HNSW: 100K-1M+ vectors, general purpose
   - SVS-VAMANA: > 100K vectors, memory constraints

3. **Storage Type Selection**
   - Hash: Flat data structures, smaller memory
   - JSON: Nested data, complex queries, JSONPath support

### Data Loading

1. **Use Batch Loading**
   ```python
   # Good: Batch loading
   index.load(data, batch_size=500)
   
   # Bad: Individual loads
   for doc in data:
       index.load([doc])
   ```

2. **Validate on Load**
   ```python
   index = SearchIndex.from_yaml(
       "schema.yaml",
       validate_on_load=True  # Catch errors early
   )
   ```

3. **Use Preprocessing**
   ```python
   def normalize_data(doc):
       doc["text"] = doc["text"].lower().strip()
       return doc
   
   index.load(data, preprocess=normalize_data)
   ```

### Query Optimization

1. **Use Appropriate Query Types**
   - VectorQuery: Semantic similarity
   - HybridQuery: Combine semantic + keyword
   - TextQuery: Keyword-only search
   - FilterQuery: Metadata filtering

2. **Tune Runtime Parameters**
   ```python
   # Start with defaults, then tune
   query = VectorQuery(
       vector=embedding,
       vector_field_name="embedding",
       ef_runtime=10  # Increase for better recall
   )
   ```

3. **Use Filters Effectively**
   ```python
   # Apply filters to reduce search space
   query = VectorQuery(
       vector=embedding,
       vector_field_name="embedding",
       filter_expression=Tag("status") == "active"
   )
   ```

### Caching Strategies

1. **Semantic Cache for LLM Responses**
   ```python
   # Check cache before expensive LLM calls
   cached = cache.check(prompt=user_query)
   if cached:
       return cached[0]["response"]
   ```

2. **Embeddings Cache for Repeated Texts**
   ```python
   # Cache embeddings for frequently used texts
   embedding = cache.get(text=text, model_name=model)
   if not embedding:
       embedding = vectorizer.embed(text)
       cache.set(text=text, model_name=model, embedding=embedding)
   ```

### Error Handling

```python
from redisvl.exceptions import RedisVLError, SchemaValidationError

try:
    results = index.query(query)
except RedisVLError as e:
    logger.error(f"RedisVL error: {e}")
    # Fallback logic
except SchemaValidationError as e:
    logger.error(f"Schema validation failed: {e}")
    # Handle invalid data
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # General error handling
```

### Monitoring

```python
# Check index stats
stats = index.stats()

# Monitor cache hit rates
cache_hits = 0
cache_misses = 0

cached = cache.check(prompt=query)
if cached:
    cache_hits += 1
else:
    cache_misses += 1

hit_rate = cache_hits / (cache_hits + cache_misses)
```

### Production Deployment

1. **Use Connection Pooling**
   ```python
   pool = ConnectionPool(max_connections=50)
   index = SearchIndex.from_yaml("schema.yaml", redis_client=Redis(connection_pool=pool))
   ```

2. **Enable Validation in Development**
   ```python
   # Development
   index = SearchIndex.from_yaml("schema.yaml", validate_on_load=True)
   
   # Production (disable for performance)
   index = SearchIndex.from_yaml("schema.yaml", validate_on_load=False)
   ```

3. **Use Async for High Throughput**
   ```python
   # Use AsyncSearchIndex for concurrent operations
   async_index = AsyncSearchIndex.from_yaml("schema.yaml")
   ```

4. **Monitor Memory Usage**
   ```python
   # Check index size
   stats = index.stats()
   memory_usage = stats.get("vector_index_sz_mb", 0)
   ```

---

## CLI Usage

### Basic Commands

```bash
# Check version
rvl version

# Create index from schema
rvl index create -s schema.yaml

# List all indices
rvl index listall

# Get index information
rvl index info -i index_name

# Get index statistics
rvl stats -i index_name

# Delete index (keeps data)
rvl index delete -i index_name

# Destroy index (deletes data)
rvl index destroy -i index_name
```

### Connection Options

```bash
# Specify Redis connection
rvl index listall --host localhost --port 6379

# With authentication
rvl index listall --user username -a password

# With SSL
rvl index listall --ssl --user username -a password

# Using environment variable
export REDIS_URL="redis://localhost:6379"
rvl index listall
```

---

## Integration Patterns

### RAG Pipeline Integration

```python
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorizers import OpenAITextVectorizer

class RAGPipeline:
    def __init__(self):
        self.index = SearchIndex.from_yaml("schema.yaml")
        self.vectorizer = OpenAITextVectorizer()
    
    async def query(self, user_query: str):
        # 1. Generate query embedding
        query_embedding = self.vectorizer.embed(user_query)
        
        # 2. Vector search
        query = VectorQuery(
            vector=query_embedding,
            vector_field_name="embedding",
            num_results=5,
            return_fields=["content", "metadata"]
        )
        
        results = self.index.query(query)
        
        # 3. Extract context
        context = "\n\n".join([r["content"] for r in results])
        
        # 4. Generate response with LLM
        response = await llm.generate(
            f"Context: {context}\n\nQuestion: {user_query}"
        )
        
        return response
```

### Agent Memory Integration

```python
from redisvl.extensions.message_history import SemanticMessageHistory
from redisvl.extensions.cache import SemanticCache

class AgentMemory:
    def __init__(self):
        self.history = SemanticMessageHistory(
            name="agent_history",
            session_tag="agent_session"
        )
        self.cache = SemanticCache(name="agent_cache")
    
    async def process(self, user_input: str):
        # 1. Check cache
        cached = self.cache.check(prompt=user_input)
        if cached:
            return cached[0]["response"]
        
        # 2. Get relevant context
        context = self.history.get_relevant(
            prompt=user_input,
            top_k=5,
            fall_back=True
        )
        
        # 3. Generate response
        response = await llm.generate(
            f"Context: {context}\n\nUser: {user_input}"
        )
        
        # 4. Store in history and cache
        self.history.store(user_input, response)
        self.cache.store(user_input, response)
        
        return response
```

---

## Conclusion

RedisVL provides a comprehensive Python library for building vector search applications with Redis. Key takeaways:

- **Flexible Schema System**: YAML or Python dict-based schema definitions
- **Multiple Vector Algorithms**: FLAT, HNSW, and SVS-VAMANA for different use cases
- **Rich Query Types**: Vector, hybrid, text, filter, and multi-vector queries
- **Built-in Caching**: Semantic cache for LLM responses and embeddings cache
- **Message History**: Track conversation context for AI applications
- **Semantic Routing**: Route queries based on semantic similarity
- **Production Ready**: Async support, connection pooling, error handling

For more information:
- [RedisVL GitHub Repository](https://github.com/redis/redis-vl-python)
- [Redis Documentation](https://redis.io/docs/)
- [RedisVL API Reference](./api-reference.md)

---

**Document Version**: 1.0  
**RedisVL Version**: 0.12.1  
**Last Updated**: January 2026
