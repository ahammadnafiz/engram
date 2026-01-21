# Redis Search and Vector Database - Python Implementation Guide

**Comprehensive Technical Reference for Redis Query Engine and Vector Search**

*Version: 1.0*  
*Last Updated: January 2026*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Vector Search Concepts](#2-vector-search-concepts)
3. [Index Management](#3-index-management)
4. [Query Operations](#4-query-operations)
5. [Best Practices](#5-best-practices)
6. [Python API Reference](#6-python-api-reference)
7. [Advanced Topics](#7-advanced-topics)

---

## 1. Introduction

### 1.1 Overview

Redis provides a high-performance vector database that enables semantic searches over vector embeddings, combined with filtering capabilities over text, numerical, geospatial, and tag metadata. The Redis Query Engine (RQE) supports:

- **Vector Search**: K-nearest neighbor (KNN) and range-based similarity search
- **Full-Text Search**: Tokenized, stemmed, and scored text queries
- **Structured Queries**: Numeric ranges, exact tag matches, geospatial radius/shape queries
- **Hybrid Search**: Combined vector + metadata filtering
- **Aggregations**: Group, reduce, and transform query results

### 1.2 Key Features

- **Multiple Index Types**: FLAT (brute-force), HNSW (approximate), SVS-VAMANA (compressed)
- **Distance Metrics**: L2 (Euclidean), IP (Inner Product), COSINE
- **Storage Options**: Redis Hash or JSON documents
- **Compression**: LVQ and LeanVec for memory optimization
- **Scalability**: Cluster support with tunable shard ratios

---

## 2. Vector Search Concepts

### 2.1 Vector Index Types

#### 2.1.1 FLAT Index

**Use Case**: Small datasets (< 1M vectors), perfect accuracy required

**Characteristics**:
- Brute-force search algorithm
- Guaranteed 100% recall
- Best for datasets where latency is acceptable

**Python Example**:

```python
import redis
import numpy as np
from redis.commands.search.field import VectorField, TagField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Define schema with FLAT vector index
schema = (
    VectorField("doc_embedding",
        "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": 1536,
            "DISTANCE_METRIC": "COSINE"
        }),
    TagField("category"),
    NumericField("price", sortable=True)
)

# Create index
r.ft("documents").create_index(
    schema,
    definition=IndexDefinition(prefix=["docs:"], index_type=IndexType.HASH)
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

#### 2.1.2 HNSW Index

**Use Case**: Large datasets (> 1M vectors), scalability over perfect accuracy

**Characteristics**:
- Hierarchical navigable small world algorithm
- Approximate nearest neighbor (ANN) search
- Tunable accuracy vs performance tradeoff

**Parameters**:
- `M`: Max outgoing edges per node (default: 16)
- `EF_CONSTRUCTION`: Candidates during build (default: 200)
- `EF_RUNTIME`: Candidates during search (default: 10)
- `EPSILON`: Range query boundary factor (default: 0.01)

**Python Example**:

```python
from redis.commands.search.field import VectorField

# HNSW with custom parameters
vector_field = VectorField("doc_embedding",
    "HNSW", {
        "TYPE": "FLOAT64",
        "DIM": 1536,
        "DISTANCE_METRIC": "COSINE",
        "M": 40,
        "EF_CONSTRUCTION": 250,
        "EF_RUNTIME": 10
    }
)

r.ft("documents").create_index(
    (vector_field,),
    definition=IndexDefinition(prefix=["docs:"], index_type=IndexType.HASH)
)
```

**Runtime Parameter Tuning**:

```python
# Search with custom EF_RUNTIME
query = Query("(*)=>[KNN 10 @doc_embedding $vec AS score]") \
    .return_fields("title", "score") \
    .sort_by("score") \
    .dialect(2)

# Pass EF_RUNTIME as parameter
params = {
    "vec": vector_bytes,
    "EF_RUNTIME": 150
}
results = r.ft("documents").search(query, query_params=params)
```

#### 2.1.3 SVS-VAMANA Index

**Use Case**: High-performance + reduced memory + Intel hardware optimization

**Characteristics**:
- Graph-based algorithm optimized for compression
- Supports LVQ and LeanVec compression
- Available in Redis 8.2+

**Python Example**:

```python
vector_field = VectorField("doc_embedding",
    "SVS-VAMANA", {
        "TYPE": "FLOAT32",
        "DIM": 1536,
        "DISTANCE_METRIC": "COSINE",
        "GRAPH_MAX_DEGREE": 40,
        "CONSTRUCTION_WINDOW_SIZE": 250,
        "COMPRESSION": "LVQ8"
    }
)
```

### 2.2 Vector Compression

#### 2.2.1 LVQ (Locally-adaptive Vector Quantization)

**Variants**:
- `LVQ4x4`: 8 bits/dim, fast search, large memory savings
- `LVQ8`: Faster ingestion, slower search
- `LVQ4x8`: Two-level quantization, improved recall

**Python Example**:

```python
# Create compressed vector index
compressed_vector = VectorField("embedding",
    "SVS-VAMANA", {
        "TYPE": "FLOAT32",
        "DIM": 768,
        "DISTANCE_METRIC": "L2",
        "COMPRESSION": "LVQ4x8",
        "TRAINING_THRESHOLD": 10240  # 10 * DEFAULT_BLOCK_SIZE
    }
)
```

#### 2.2.2 LeanVec

**Best For**: High-dimensional vectors (reduce dimensionality + quantize)

**Variants**:
- `LeanVec4x8`: Fastest search and ingestion
- `LeanVec8x8`: Improved recall

**Python Example**:

```python
# LeanVec with dimensionality reduction
leanvec_field = VectorField("embedding",
    "SVS-VAMANA", {
        "TYPE": "FLOAT32",
        "DIM": 1536,
        "DISTANCE_METRIC": "COSINE",
        "COMPRESSION": "LeanVec4x8",
        "REDUCE": 384  # Reduce to DIM/4 for faster search
    }
)
```

### 2.3 Distance Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **L2** | $d(u,v) = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$ | Euclidean distance, general-purpose |
| **IP** | $d(u,v) = 1 - u \cdot v$ | Inner product, embeddings with magnitude |
| **COSINE** | $d(u,v) = 1 - \frac{u \cdot v}{\|u\| \|v\|}$ | Normalized similarity, text embeddings |

**Important**: Smaller distance = higher similarity. All return values between 0 and positive values.

### 2.4 Storing Vectors

#### 2.4.1 Hash Storage

**Python Example**:

```python
import numpy as np

# Create vector
vector = np.array([0.34, 0.63, -0.54, -0.69, 0.98, 0.61], dtype=np.float32)

# Convert to bytes
vector_bytes = vector.tobytes()

# Store in hash
r.hset('docs:01', mapping={
    "vector": vector_bytes,
    "category": "sports",
    "title": "Example Document"
})
```

**Binary Format**: Vectors stored as raw binary bytes. Blob size must match DIM × type size (e.g., 1536 dims × 4 bytes = 6144 bytes for FLOAT32).

#### 2.4.2 JSON Storage

**Python Example**:

```python
import json

# Store vector as JSON array
doc = {
    "doc_embedding": [0.34, 0.63, -0.54, -0.69, 0.98, 0.61],
    "category": "sports",
    "title": "Example Document"
}

r.json().set('docs:01', '$', doc)
```

**Benefits**:
- Schema flexibility
- Multi-value indexing support (arrays of vectors)
- Human-readable format

**Multi-value Vector Example**:

```python
# Store multiple vectors per document
doc = {
    "embeddings": [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ],
    "title": "Document with multiple chunks"
}

r.json().set('docs:multi', '$', doc)

# Index with multi-value JSONPath
r.ft("multi_idx").create_index(
    (VectorField("$.embeddings[*]", "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": 4,
        "DISTANCE_METRIC": "L2"
    }, as_name="embeddings"),),
    definition=IndexDefinition(prefix=["docs:"], index_type=IndexType.JSON)
)
```

---

## 3. Index Management

### 3.1 Schema Definition

#### 3.1.1 Complete Schema Example

**Python Implementation**:

```python
from redis.commands.search.field import (
    TextField, TagField, NumericField, GeoField, VectorField
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Comprehensive schema
schema = (
    # Text fields
    TextField("title", weight=5.0),
    TextField("description", no_stem=True),
    
    # Tag fields
    TagField("category", separator=",", case_sensitive=False),
    TagField("tags", sortable=True),
    
    # Numeric fields
    NumericField("price", sortable=True),
    NumericField("stock"),
    
    # Geo fields
    GeoField("store_location", sortable=True),
    
    # Vector field
    VectorField("embedding", "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": 768,
        "DISTANCE_METRIC": "COSINE",
        "M": 32,
        "EF_CONSTRUCTION": 200
    })
)

# Create index with prefix
r.ft("products").create_index(
    schema,
    definition=IndexDefinition(
        prefix=["product:"],
        index_type=IndexType.JSON,
        language="english",
        score=1.0,
        score_field="doc_score"
    )
)
```

#### 3.1.2 Field Type Reference

| Field Type | Use Case | Queryable | Sortable | Options |
|------------|----------|-----------|----------|---------|
| **TEXT** | Full-text search | Yes | Yes | `WEIGHT`, `NOSTEM`, `PHONETIC` |
| **TAG** | Exact match, categories | Yes | Yes | `SEPARATOR`, `CASESENSITIVE` |
| **NUMERIC** | Ranges, sorting | Yes | Yes | `SORTABLE`, `NOINDEX` |
| **GEO** | Radius queries | Yes | Yes | Longitude, Latitude |
| **GEOSHAPE** | Polygon queries | Yes | No | FLAT or SPHERICAL |
| **VECTOR** | Similarity search | Yes | No | Algorithm-specific params |

### 3.2 Index Operations

#### 3.2.1 Create Index

```python
def create_product_index(redis_client):
    """Create product search index with vector support"""
    try:
        # Drop existing index if it exists
        try:
            redis_client.ft("products").dropindex(delete_documents=False)
        except:
            pass
        
        # Define schema
        schema = (
            TextField("name", weight=2.0, sortable=True),
            TextField("description"),
            TagField("brand", sortable=True),
            NumericField("price", sortable=True),
            TagField("category", separator="|"),
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": 384,
                "DISTANCE_METRIC": "COSINE",
                "M": 16,
                "EF_CONSTRUCTION": 200
            })
        )
        
        # Create index
        redis_client.ft("products").create_index(
            schema,
            definition=IndexDefinition(
                prefix=["product:"],
                index_type=IndexType.JSON
            )
        )
        
        print("Index created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating index: {e}")
        return False
```

#### 3.2.2 Index Aliasing

**Use Case**: Schema updates without downtime

```python
def update_index_with_alias(redis_client):
    """Update index schema using aliases"""
    
    # Create new index version
    new_index_name = "products_v2"
    
    redis_client.ft(new_index_name).create_index(
        schema,  # Updated schema
        definition=IndexDefinition(prefix=["product:"], index_type=IndexType.JSON)
    )
    
    # Wait for indexing to complete
    info = redis_client.ft(new_index_name).info()
    while info.get('indexing', True):
        time.sleep(1)
        info = redis_client.ft(new_index_name).info()
    
    # Update alias to point to new index
    try:
        redis_client.execute_command('FT.ALIASUPDATE', 'products_alias', new_index_name)
    except:
        redis_client.execute_command('FT.ALIASADD', 'products_alias', new_index_name)
    
    # Drop old index after verification
    # redis_client.ft("products_v1").dropindex(delete_documents=False)
```

#### 3.2.3 Monitor Index Population

```python
def check_index_readiness(redis_client, index_name, expected_docs):
    """Monitor index population progress"""
    info = redis_client.ft(index_name).info()
    
    # Parse info response
    info_dict = {}
    for i in range(0, len(info), 2):
        key = info[i].decode() if isinstance(info[i], bytes) else info[i]
        value = info[i+1]
        info_dict[key] = value
    
    num_docs = int(info_dict.get('num_docs', 0))
    indexing = info_dict.get('indexing', False)
    
    print(f"Documents indexed: {num_docs}/{expected_docs}")
    print(f"Still indexing: {indexing}")
    
    return num_docs >= expected_docs and not indexing
```

### 3.3 Multiple Prefix Indexing

**Use Case**: Index different entity types under one index

```python
# Index both users and orders
schema = (
    TextField("name", sortable=True),
    NumericField("timestamp", sortable=True),
    TagField("type")  # 'user' or 'order'
)

redis_client.execute_command(
    'FT.CREATE', 'unified_index',
    'ON', 'HASH',
    'PREFIX', '2', 'user:', 'order:',
    'SCHEMA',
    'name', 'TEXT', 'SORTABLE',
    'timestamp', 'NUMERIC', 'SORTABLE',
    'type', 'TAG'
)
```

---

## 4. Query Operations

### 4.1 Vector Search Queries

#### 4.1.1 K-Nearest Neighbors (KNN)

**Basic KNN**:

```python
def knn_search(redis_client, query_vector, k=10):
    """Perform KNN vector search"""
    
    # Convert vector to bytes
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    
    # Create query
    query = Query("(*)=>[KNN $K @embedding $vector AS score]") \
        .return_fields("name", "score") \
        .sort_by("score") \
        .paging(0, k) \
        .dialect(2)
    
    # Execute search
    results = redis_client.ft("products").search(
        query,
        query_params={
            "K": k,
            "vector": query_bytes
        }
    )
    
    return results
```

**With Pre-filtering**:

```python
def filtered_knn_search(redis_client, query_vector, category, price_max, k=10):
    """KNN with metadata pre-filtering"""
    
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    
    # Filter before KNN
    query = Query(
        "(@category:{$cat} @price:[-inf $max])=>[KNN $K @embedding $vector AS score]"
    ).return_fields("name", "price", "score") \
     .sort_by("score") \
     .dialect(2)
    
    results = redis_client.ft("products").search(
        query,
        query_params={
            "cat": category,
            "max": price_max,
            "K": k,
            "vector": query_bytes
        }
    )
    
    return results
```

**With Hybrid Policy**:

```python
def knn_with_policy(redis_client, query_vector, filter_expr, k=10):
    """Control filter mode for hybrid search"""
    
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    
    # Force batches mode with custom batch size
    query = Query(
        f"({filter_expr})=>[KNN $K @embedding $vector HYBRID_POLICY BATCHES BATCH_SIZE $batch]"
    ).return_fields("name", "score") \
     .sort_by("score") \
     .dialect(2)
    
    results = redis_client.ft("products").search(
        query,
        query_params={
            "K": k,
            "vector": query_bytes,
            "batch": 50
        }
    )
    
    return results
```

#### 4.1.2 Vector Range Queries

```python
def vector_range_search(redis_client, query_vector, radius=0.5):
    """Find vectors within a distance radius"""
    
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    
    query = Query(
        "@embedding:[VECTOR_RANGE $radius $vector]=>{$YIELD_DISTANCE_AS: dist}"
    ).return_fields("name", "dist") \
     .sort_by("dist") \
     .dialect(2)
    
    results = redis_client.ft("products").search(
        query,
        query_params={
            "radius": radius,
            "vector": query_bytes
        }
    )
    
    return results
```

#### 4.1.3 Cluster Optimization

```python
def cluster_optimized_knn(redis_client, query_vector, k=100, shard_ratio=0.6):
    """KNN optimized for Redis cluster with shard ratio"""
    
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    
    query = Query(
        "(*)=>[KNN $K @embedding $vector]=>{$SHARD_K_RATIO: $ratio; $YIELD_DISTANCE_AS: score}"
    ).return_fields("name", "score") \
     .sort_by("score") \
     .paging(0, k) \
     .dialect(2)
    
    results = redis_client.ft("products").search(
        query,
        query_params={
            "K": k,
            "vector": query_bytes,
            "ratio": shard_ratio
        }
    )
    
    return results
```

### 4.2 Full-Text Search

#### 4.2.1 Basic Text Queries

```python
def text_search(redis_client, search_term):
    """Simple full-text search"""
    
    # Search across all TEXT fields
    query = Query(search_term).return_fields("name", "description")
    results = redis_client.ft("products").search(query)
    
    return results

def field_specific_text_search(redis_client, field, term):
    """Search within specific TEXT field"""
    
    query = Query(f"@{field}:{term}").return_fields("name", field)
    results = redis_client.ft("products").search(query)
    
    return results
```

#### 4.2.2 Advanced Text Queries

```python
# Prefix search
def prefix_search(redis_client, prefix):
    """Search for words starting with prefix"""
    return redis_client.ft("products").search(Query(f"@name:{prefix}*"))

# Fuzzy search (typo tolerance)
def fuzzy_search(redis_client, term, distance=1):
    """Fuzzy search with Levenshtein distance"""
    fuzzy_term = "%" * distance + term + "%" * distance
    return redis_client.ft("products").search(Query(fuzzy_term))

# Phrase search (exact match)
def phrase_search(redis_client, phrase):
    """Search for exact phrase"""
    return redis_client.ft("products").search(Query(f'@description:"{phrase}"'))
```

### 4.3 Structured Queries

#### 4.3.1 Exact Match (TAG)

```python
def tag_search(redis_client, tag_field, tag_value):
    """Exact match on TAG field"""
    query = Query(f"@{tag_field}:{{{tag_value}}}").return_fields("name", tag_field)
    return redis_client.ft("products").search(query)

# Multiple tags (OR)
def multi_tag_search(redis_client, categories):
    """Match any of multiple tags"""
    tags = " | ".join(categories)
    query = Query(f"@category:{{{tags}}}").return_fields("name", "category")
    return redis_client.ft("products").search(query)
```

#### 4.3.2 Numeric Range

```python
def numeric_range_search(redis_client, field, min_val, max_val):
    """Query numeric field within range"""
    query = Query(f"@{field}:[{min_val} {max_val}]").return_fields("name", field)
    return redis_client.ft("products").search(query)

# Open-ended range
def price_greater_than(redis_client, min_price):
    """Find products above price threshold"""
    query = Query(f"@price:[{min_price} +inf]") \
        .sort_by("price") \
        .return_fields("name", "price")
    return redis_client.ft("products").search(query)

# Using FILTER
def numeric_filter_search(redis_client, field, min_val, max_val):
    """Alternative syntax using FILTER"""
    query = Query("*").add_filter(field, min_val, max_val)
    return redis_client.ft("products").search(query)
```

#### 4.3.3 Geospatial Queries

**GEO (Radius)**:

```python
def geo_radius_search(redis_client, lon, lat, radius, unit='mi'):
    """Find locations within radius"""
    query = Query(f"@store_location:[{lon} {lat} {radius} {unit}]") \
        .return_fields("name", "store_location")
    return redis_client.ft("products").search(query)
```

**GEOSHAPE (Polygon)**:

```python
def geoshape_within_search(redis_client, polygon_wkt):
    """Find shapes within a polygon"""
    query = Query("@pickup_zone:[WITHIN $shape]") \
        .return_fields("name") \
        .dialect(2)
    
    results = redis_client.ft("products").search(
        query,
        query_params={"shape": polygon_wkt}
    )
    return results

def geoshape_contains_search(redis_client, point_wkt):
    """Find shapes containing a point"""
    query = Query("@pickup_zone:[CONTAINS $point]") \
        .return_fields("name") \
        .dialect(2)
    
    results = redis_client.ft("products").search(
        query,
        query_params={"point": point_wkt}
    )
    return results
```

### 4.4 Combined Queries

```python
def complex_combined_query(redis_client, query_vector, category, min_price, max_price, k=10):
    """Combine vector search with multiple filters"""
    
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    
    # Combine TAG, NUMERIC, and VECTOR
    query = Query(
        "(@category:{$cat} @price:[$min $max])=>[KNN $K @embedding $vector AS score]"
    ).return_fields("name", "category", "price", "score") \
     .sort_by("score") \
     .dialect(2)
    
    results = redis_client.ft("products").search(
        query,
        query_params={
            "cat": category,
            "min": min_price,
            "max": max_price,
            "K": k,
            "vector": query_bytes
        }
    )
    
    return results
```

### 4.5 Aggregation Queries

#### 4.5.1 Group and Reduce

```python
from redis.commands.search.aggregation import AggregateRequest, Asc, Desc

def aggregate_by_category(redis_client):
    """Group products by category and calculate averages"""
    
    request = AggregateRequest("*") \
        .load("price", "category") \
        .group_by("@category", 
            reducers.count().alias("count"),
            reducers.avg("@price").alias("avg_price"),
            reducers.min("@price").alias("min_price"),
            reducers.max("@price").alias("max_price")
        ) \
        .sort_by(Desc("@count"))
    
    results = redis_client.ft("products").aggregate(request)
    return results
```

#### 4.5.2 Apply Transformations

```python
def calculate_discounted_prices(redis_client, discount_rate=0.1):
    """Apply discount calculation during aggregation"""
    
    request = AggregateRequest("*") \
        .load("name", "price") \
        .apply(
            discounted=f"@price - (@price * {discount_rate})"
        ) \
        .sort_by(Desc("@discounted"))
    
    results = redis_client.ft("products").aggregate(request)
    return results
```

---

## 5. Best Practices

### 5.1 Index Design

#### 5.1.1 Field Selection

```python
# ✅ GOOD: Only index queried fields
schema = (
    TextField("title"),
    TextField("description"),
    TagField("category", sortable=True),
    NumericField("price", sortable=True)
)

# ❌ BAD: Over-indexing
schema = (
    TextField("title"),
    TextField("description"),
    TextField("internal_notes"),  # Not queried
    TextField("metadata"),  # Not queried
    TagField("category", sortable=True),
    NumericField("price", sortable=True),
    NumericField("internal_id", sortable=True)  # Not needed
)
```

#### 5.1.2 Field Types

```python
# ✅ GOOD: Use TAG for exact match
TagField("status")  # For values like "active", "inactive"

# ❌ BAD: Use TEXT for exact match (wasteful)
TextField("status")

# ✅ GOOD: Use TAG over NUMERIC for exact match only
TagField("year")  # If only matching exact years

# ✅ GOOD: Use NUMERIC for ranges/sorting
NumericField("year", sortable=True)  # If range queries needed
```

#### 5.1.3 Sortable Fields

```python
# With QPF (Query Performance Factor) / threading:
schema = (
    TextField("title", sortable=True, no_stem=True),
    TagField("category", sortable=True),
    NumericField("price", sortable=True)
)

# Without QPF:
schema = (
    TextField("title"),  # Only make sortable if SORTBY used
    TagField("category"),
    NumericField("price", sortable=True)  # Sortable needed for range queries
)
```

### 5.2 Query Optimization

#### 5.2.1 Result Pagination

```python
def paginated_search(redis_client, query_str, page=1, page_size=10):
    """Efficient result pagination"""
    
    offset = (page - 1) * page_size
    
    query = Query(query_str) \
        .paging(offset, page_size) \
        .return_fields("name", "price")
    
    results = redis_client.ft("products").search(query)
    
    return {
        "total": results.total,
        "page": page,
        "page_size": page_size,
        "results": results.docs
    }
```

#### 5.2.2 Projection (Return Only Needed Fields)

```python
# ✅ GOOD: Return only needed fields
query = Query("@category:{electronics}") \
    .return_fields("id", "name", "price")

# ❌ BAD: Return all fields (wasteful)
query = Query("@category:{electronics}")  # Returns entire document
```

#### 5.2.3 Avoid Wildcard Searches

```python
# ❌ BAD: Unrestricted wildcard
query = Query("*")  # Returns everything

# ✅ BETTER: Use specific filters
query = Query("@category:{electronics}")

# ✅ ACCEPTABLE: Wildcard with constraints
query = Query("*").add_filter("price", 100, 1000)
```

### 5.3 Vector Search Optimization

#### 5.3.1 Index Selection

```python
# Small dataset (<1M vectors)
VectorField("embedding", "FLAT", {
    "TYPE": "FLOAT32",
    "DIM": 768,
    "DISTANCE_METRIC": "COSINE"
})

# Large dataset (>1M vectors)
VectorField("embedding", "HNSW", {
    "TYPE": "FLOAT32",
    "DIM": 768,
    "DISTANCE_METRIC": "COSINE",
    "M": 16,  # Lower M = less memory, faster build
    "EF_CONSTRUCTION": 200
})

# Very large dataset + memory constrained
VectorField("embedding", "SVS-VAMANA", {
    "TYPE": "FLOAT32",
    "DIM": 768,
    "DISTANCE_METRIC": "COSINE",
    "COMPRESSION": "LVQ4x8"  # Compress for memory savings
})
```

#### 5.3.2 Batch Mode vs Ad-Hoc Brute Force

```python
# Let Redis choose (default)
query = Query("(@category:{electronics})=>[KNN 10 @embedding $vec]")

# Force batches mode (when filter is selective)
query = Query(
    "(@category:{rare_category})=>[KNN 10 @embedding $vec HYBRID_POLICY BATCHES]"
)

# Force ad-hoc brute force (when filter is highly selective)
query = Query(
    "(@price:[9000 10000])=>[KNN 10 @embedding $vec HYBRID_POLICY ADHOC_BF]"
)
```

### 5.4 Performance Monitoring

#### 5.4.1 Query Profiling

```python
def profile_query(redis_client, query_str):
    """Profile query execution"""
    
    result = redis_client.execute_command(
        'FT.PROFILE', 'products', 'SEARCH', 'QUERY', query_str
    )
    
    # Parse profile information
    profile_data = result[0]  # Profile details
    search_results = result[1]  # Actual results
    
    print("Query execution profile:")
    print(profile_data)
    
    return search_results
```

#### 5.4.2 Index Statistics

```python
def get_index_stats(redis_client, index_name):
    """Retrieve index statistics"""
    
    info = redis_client.ft(index_name).info()
    
    stats = {
        "num_docs": info.get('num_docs', 0),
        "num_terms": info.get('num_terms', 0),
        "max_doc_id": info.get('max_doc_id', 0),
        "num_records": info.get('num_records', 0),
        "percent_indexed": info.get('percent_indexed', 0),
        "indexing": info.get('indexing', False),
        "index_name": info.get('index_name', ''),
        "index_options": info.get('index_options', [])
    }
    
    return stats
```

### 5.5 Production Checklist

```python
def production_readiness_check(redis_client, index_name):
    """Verify index is production-ready"""
    
    checks = {
        "index_exists": False,
        "indexing_complete": False,
        "sufficient_docs": False,
        "query_response_time": None
    }
    
    try:
        # Check index exists
        info = redis_client.ft(index_name).info()
        checks["index_exists"] = True
        
        # Check indexing complete
        checks["indexing_complete"] = not info.get('indexing', True)
        
        # Check document count
        num_docs = int(info.get('num_docs', 0))
        checks["sufficient_docs"] = num_docs > 0
        
        # Check query performance
        import time
        start = time.time()
        redis_client.ft(index_name).search(Query("*").paging(0, 10))
        checks["query_response_time"] = time.time() - start
        
    except Exception as e:
        print(f"Health check failed: {e}")
    
    return checks
```

---

## 6. Python API Reference

### 6.1 Complete Working Example

```python
import redis
import numpy as np
from redis.commands.search.field import VectorField, TextField, TagField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

class RedisVectorDB:
    """Complete Redis Vector Database wrapper"""
    
    def __init__(self, host='localhost', port=6379, decode_responses=True):
        self.client = redis.Redis(
            host=host,
            port=port,
            decode_responses=decode_responses
        )
    
    def create_index(self, index_name, vector_dim, distance_metric='COSINE'):
        """Create product search index"""
        
        schema = (
            TextField("name", weight=2.0, sortable=True),
            TextField("description"),
            TagField("category", sortable=True),
            TagField("brand"),
            NumericField("price", sortable=True),
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": vector_dim,
                "DISTANCE_METRIC": distance_metric,
                "M": 16,
                "EF_CONSTRUCTION": 200
            })
        )
        
        try:
            self.client.ft(index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=["product:"],
                    index_type=IndexType.JSON
                )
            )
            print(f"✓ Index '{index_name}' created")
            return True
        except Exception as e:
            print(f"✗ Index creation failed: {e}")
            return False
    
    def add_document(self, doc_id, data, embedding):
        """Add document with vector embedding"""
        
        # Store data as JSON
        data['embedding'] = embedding.tolist()
        
        self.client.json().set(f"product:{doc_id}", '$', data)
        print(f"✓ Document 'product:{doc_id}' added")
    
    def knn_search(self, index_name, query_vector, k=10, filters=None):
        """Perform KNN vector search with optional filters"""
        
        query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        
        # Build query string
        if filters:
            filter_str = " ".join([f"@{k}:{{{v}}}" for k, v in filters.items()])
            query_str = f"({filter_str})=>[KNN {k} @embedding $vector AS score]"
        else:
            query_str = f"(*)=>[KNN {k} @embedding $vector AS score]"
        
        query = Query(query_str) \
            .return_fields("name", "category", "price", "score") \
            .sort_by("score") \
            .paging(0, k) \
            .dialect(2)
        
        results = self.client.ft(index_name).search(
            query,
            query_params={"vector": query_bytes}
        )
        
        return results
    
    def hybrid_search(self, index_name, text_query, query_vector, k=10):
        """Combine text and vector search"""
        
        query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        
        query = Query(
            f"({text_query})=>[KNN {k} @embedding $vector AS score]"
        ).return_fields("name", "description", "score") \
         .sort_by("score") \
         .dialect(2)
        
        results = self.client.ft(index_name).search(
            query,
            query_params={"vector": query_bytes}
        )
        
        return results
    
    def delete_index(self, index_name, delete_docs=False):
        """Drop index"""
        try:
            self.client.ft(index_name).dropindex(delete_documents=delete_docs)
            print(f"✓ Index '{index_name}' dropped")
            return True
        except Exception as e:
            print(f"✗ Index drop failed: {e}")
            return False

# Usage example
def main():
    # Initialize
    db = RedisVectorDB()
    
    # Create index
    db.create_index("products", vector_dim=384)
    
    # Add documents
    products = [
        {
            "name": "Laptop Pro",
            "description": "High-performance laptop",
            "category": "electronics",
            "brand": "TechCorp",
            "price": 1200
        },
        {
            "name": "Wireless Mouse",
            "description": "Ergonomic wireless mouse",
            "category": "accessories",
            "brand": "TechCorp",
            "price": 25
        }
    ]
    
    for i, product in enumerate(products):
        # Generate fake embedding (replace with real embeddings)
        embedding = np.random.rand(384).astype(np.float32)
        db.add_document(i, product, embedding)
    
    # Wait for indexing
    import time
    time.sleep(1)
    
    # Search
    query_vector = np.random.rand(384).astype(np.float32)
    results = db.knn_search("products", query_vector, k=10)
    
    print(f"\nFound {results.total} results:")
    for doc in results.docs:
        print(f"  - {doc.name} (score: {doc.score})")

if __name__ == "__main__":
    main()
```

### 6.2 Embedding Generation Integration

```python
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingGenerator:
    """Generate embeddings for text using Sentence Transformers"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts):
        """Generate embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        return embeddings.cpu().numpy()
    
    def encode_batch(self, texts, batch_size=32):
        """Generate embeddings in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

# Integration example
def index_documents_with_embeddings(redis_db, documents):
    """Index documents with generated embeddings"""
    
    encoder = EmbeddingGenerator()
    
    # Generate embeddings for all documents
    texts = [f"{doc['name']} {doc['description']}" for doc in documents]
    embeddings = encoder.encode_batch(texts)
    
    # Add to Redis
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        redis_db.add_document(i, doc, embedding)
    
    print(f"✓ Indexed {len(documents)} documents with embeddings")
```

### 6.3 Advanced Query Examples

```python
class AdvancedQueryBuilder:
    """Build complex Redis queries"""
    
    @staticmethod
    def range_filtered_vector_search(category, min_price, max_price, vector_bytes, k=10):
        """Vector search with category and price range filter"""
        query = Query(
            "(@category:{$cat} @price:[$min $max])=>[KNN $K @embedding $vec AS score]"
        ).return_fields("name", "category", "price", "score") \
         .sort_by("score") \
         .dialect(2)
        
        params = {
            "cat": category,
            "min": min_price,
            "max": max_price,
            "K": k,
            "vec": vector_bytes
        }
        
        return query, params
    
    @staticmethod
    def multi_field_text_vector_search(text_query, vector_bytes, k=10):
        """Combined full-text and vector search"""
        query = Query(
            f"({text_query})=>[KNN $K @embedding $vec AS score]"
        ).return_fields("name", "description", "score") \
         .sort_by("score") \
         .dialect(2)
        
        params = {
            "K": k,
            "vec": vector_bytes
        }
        
        return query, params
    
    @staticmethod
    def geo_vector_search(lon, lat, radius, vector_bytes, k=10):
        """Vector search with geospatial filter"""
        query = Query(
            f"(@location:[{lon} {lat} {radius} km])=>[KNN $K @embedding $vec AS score]"
        ).return_fields("name", "location", "score") \
         .sort_by("score") \
         .dialect(2)
        
        params = {
            "K": k,
            "vec": vector_bytes
        }
        
        return query, params
```

---

## 7. Advanced Topics

### 7.1 Scoring Functions

Redis supports multiple scoring functions for full-text search:

| Function | Description | Use Case |
|----------|-------------|----------|
| **BM25STD** (default) | Okapi BM25 variant | General purpose |
| **TFIDF** | Term frequency × inverse document frequency | Classic ranking |
| **TFIDF.DOCNORM** | TF-IDF normalized by doc length | Length-normalized ranking |
| **DISMAX** | Sum of term frequencies | Union queries |
| **DOCSCORE** | Document score only | External scoring |

```python
# Use specific scorer
query = Query("laptop").set_scorer("BM25STD")
results = redis_client.ft("products").search(query)
```

### 7.2 Sorting

```python
# Sort by numeric field
query = Query("@category:{electronics}") \
    .sort_by("price", asc=True)

# Sort by text field (if sortable)
query = Query("*") \
    .sort_by("name", asc=True)

# Sort by vector score
query = Query("(*)=>[KNN 10 @embedding $vec AS score]") \
    .sort_by("score", asc=True) \
    .dialect(2)
```

### 7.3 Query Dialects

```python
# Dialect 1 (legacy, default)
query = Query("@field:value")

# Dialect 2 (recommended, supports vector search)
query = Query("@field:value").dialect(2)

# Dialect 3 (latest features)
query = Query("@field:value").dialect(3)
```

### 7.4 Expiration and TTL

```python
# Set TTL on document
redis_client.json().set('product:temp', '$', data)
redis_client.expire('product:temp', 3600)  # 1 hour

# Automatically removed from index when expired
```

### 7.5 Error Handling

```python
def safe_search(redis_client, index_name, query_str):
    """Search with proper error handling"""
    try:
        query = Query(query_str).dialect(2)
        results = redis_client.ft(index_name).search(query)
        return results
    
    except redis.exceptions.ResponseError as e:
        if "no such index" in str(e):
            print(f"Index '{index_name}' does not exist")
        elif "Timeout" in str(e):
            print("Query timeout - consider increasing TIMEOUT config")
        else:
            print(f"Query error: {e}")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

---

## 8. Conclusion

This guide provides a comprehensive reference for implementing Redis Search and Vector Database functionality using Python. Key takeaways:

1. **Choose the right index type**: FLAT for small datasets, HNSW for large, SVS-VAMANA for memory-constrained
2. **Design schemas carefully**: Only index queryable fields, use appropriate field types
3. **Optimize queries**: Use pre-filtering, pagination, and field projection
4. **Monitor performance**: Profile queries, track index stats, test at scale
5. **Follow best practices**: Use aliases for schema updates, batch operations, handle errors

For production deployments, refer to the [Best Practices](#5-best-practices) section and conduct thorough load testing.

---

## Appendix A: Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIMEOUT` | 500ms | Query execution timeout |
| `MINPREFIX` | 2 | Minimum wildcard prefix length |
| `MAXPREFIXEXPANSIONS` | 200 | Max wildcard expansions |
| `DEFAULT_DIALECT` | 1 | Default query dialect |

## Appendix B: Command Reference

| Command | Purpose |
|---------|---------|
| `FT.CREATE` | Create index |
| `FT.SEARCH` | Search documents |
| `FT.AGGREGATE` | Aggregate results |
| `FT.INFO` | Index statistics |
| `FT.PROFILE` | Profile query |
| `FT.DROPINDEX` | Delete index |
| `FT.ALTER` | Add fields to index |
| `FT.ALIASADD` | Create alias |
| `FT.ALIASUPDATE` | Update alias |

## Appendix C: Resources

- [Redis Vector Search Documentation](https://redis.io/docs/develop/ai/search-and-query/vectors/)
- [Redis Python Client](https://redis.readthedocs.io/)
- [Redis AI Resources (GitHub)](https://github.com/redis-developer/redis-ai-resources)
- [Sentence Transformers](https://www.sbert.net/)

---

**End of Document**

