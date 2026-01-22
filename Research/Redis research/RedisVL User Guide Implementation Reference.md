# RedisVL User Guide Implementation Reference

## Table of Contents
1. [Getting Started](#getting-started)
2. [Querying with RedisVL](#querying-with-redisvl)
3. [LLM Caching](#llm-caching)
4. [Vectorizers](#vectorizers)
5. [Hash vs JSON Storage](#hash-vs-json-storage)
6. [Rerankers](#rerankers)
7. [LLM Message History](#llm-message-history)
8. [Semantic Routing](#semantic-routing)
9. [SVS-VAMANA Vector Search](#svs-vamana-vector-search)
10. [Advanced Query Types](#advanced-query-types)
11. [Caching Embeddings](#caching-embeddings)

---

## Getting Started

### Prerequisites
- `redisvl` installed in Python environment
- Running Redis Stack or Redis Cloud instance

### Define an IndexSchema

**YAML Format:**
```yaml
version: '0.1.0'

index:
  name: user_simple
  prefix: user_simple_docs

fields:
    - name: user
      type: tag
    - name: credit_score
      type: tag
    - name: job
      type: text
    - name: age
      type: numeric
    - name: user_embedding
      type: vector
      attrs:
        algorithm: flat
        dims: 3
        distance_metric: cosine
        datatype: float32
```

**Python Dictionary:**
```python
schema = {
    "index": {
        "name": "user_simple",
        "prefix": "user_simple_docs",
    },
    "fields": [
        {"name": "user", "type": "tag"},
        {"name": "credit_score", "type": "tag"},
        {"name": "job", "type": "text"},
        {"name": "age", "type": "numeric"},
        {
            "name": "user_embedding",
            "type": "vector",
            "attrs": {
                "dims": 3,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32"
            }
        }
    ]
}
```

### Sample Dataset Preparation

**Important:** Vectors must be converted to bytes for Hash storage:
```python
import numpy as np

data = [
    {
        'user': 'john',
        'age': 1,
        'job': 'engineer',
        'credit_score': 'high',
        'user_embedding': np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes()
    },
    # ... more records
]
```

### Create a SearchIndex

**With Custom Redis Connection:**
```python
from redisvl.index import SearchIndex
from redis import Redis

client = Redis.from_url("redis://localhost:6379")
index = SearchIndex.from_dict(schema, redis_client=client, validate_on_load=True)
```

**Let Index Manage Connection:**
```python
index = SearchIndex.from_dict(schema, redis_url="redis://localhost:6379", validate_on_load=True)
```

**Create the Index:**
```python
index.create(overwrite=True)
```

### Inspect with CLI

```bash
rvl index listall
rvl index info -i user_simple
rvl stats -i user_simple
```

### Load Data

```python
keys = index.load(data)
# Returns list of generated keys: ['user_simple_docs:01JY4J4Y08GFY10VMB9D4YDMZQ', ...]
```

**Custom Keys:**
```python
keys = index.load(data, keys=["custom:key:1", "custom:key:2"])
```

**Using ID Field:**
```python
# If schema has an id_field, use it for keys
schema["index"]["id_field"] = "user_id"
```

### Creating VectorQuery Objects

```python
from redisvl.query import VectorQuery

query = VectorQuery(
    vector=[0.1, 0.1, 0.5],
    vector_field_name="user_embedding",
    return_fields=["user", "age", "job", "credit_score", "vector_distance"],
    num_results=3
)

results = index.query(query)
```

**With Runtime Parameters (HNSW/SVS-VAMANA):**
```python
query = VectorQuery(
    vector=[0.1, 0.1, 0.5],
    vector_field_name="user_embedding",
    return_fields=["user", "age", "job"],
    num_results=3,
    ef_runtime=50  # HNSW: higher for better recall
    # search_window_size=40  # SVS-VAMANA: larger window for better recall
)
```

### Asynchronous Operations

```python
from redisvl.index import AsyncSearchIndex
from redis.asyncio import Redis

client = Redis.from_url("redis://localhost:6379")
index = AsyncSearchIndex.from_dict(schema, redis_client=client)

# Execute queries async
results = await index.query(query)
```

### Updating Schema

```python
# Modify schema
index.schema.remove_field("job")
index.schema.remove_field("user_embedding")
index.schema.add_fields([
    {"name": "job", "type": "tag"},
    {
        "name": "user_embedding",
        "type": "vector",
        "attrs": {
            "dims": 3,
            "distance_metric": "cosine",
            "algorithm": "hnsw",
            "datatype": "float32"
        }
    }
])

# Update index (keeps underlying data)
await index.create(overwrite=True, drop=False)
```

### Cleanup

```python
# Clear all data (keeps index)
await index.clear()

# Delete index and data
await index.delete()
```

---

## Querying with RedisVL

### Tag Filters

```python
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

# Exact match
t = Tag("credit_score") == "high"

# Negation
t = Tag("credit_score") != "high"

# Multiple values
t = Tag("credit_score") == ["high", "medium"]

# Empty case gracefully falls back to "*"
t = Tag("credit_score") == []
```

### Numeric Filters

```python
from redisvl.query.filter import Num

# Range
numeric_filter = Num("age").between(15, 35)

# Exact match
numeric_filter = Num("age") == 14

# Comparison
numeric_filter = Num("age") >= 18
numeric_filter = Num("age") < 100
```

### Timestamp Filters

```python
from redisvl.query.filter import Timestamp
from datetime import datetime

dt = datetime(2025, 3, 16, 13, 45, 39, 132589)

# Greater than
timestamp_filter = Timestamp("last_updated") > dt

# Less than
timestamp_filter = Timestamp("last_updated") < dt

# Between
timestamp_filter = Timestamp("last_updated").between(dt_1, dt_2)
```

### Text Filters

```python
from redisvl.query.filter import Text

# Exact match
text_filter = Text("job") == "doctor"

# Negation
negate_text_filter = Text("job") != "doctor"

# Wildcard
wildcard_filter = Text("job") % "doct*"

# Fuzzy match
fuzzy_match = Text("job") % "%%engine%%"

# Conditional (OR)
conditional = Text("job") % "engineer|doctor"

# Empty case
empty_case = Text("job") % ""
```

### Geographic Filters

```python
from redisvl.query.filter import Geo, GeoRadius

# Within radius
geo_filter = Geo("office_location") == GeoRadius(-122.4194, 37.7749, 10, "km")

# Not within radius
geo_filter = Geo("office_location") != GeoRadius(-122.4194, 37.7749, 10, "km")
```

### Combining Filters

**Intersection (AND):**
```python
t = Tag("credit_score") == "high"
low = Num("age") >= 18
high = Num("age") <= 100
ts = Timestamp("last_updated") > datetime(2025, 3, 16)

combined = t & low & high & ts
```

**Union (OR):**
```python
low = Num("age") < 18
high = Num("age") > 93
combined = low | high
```

**Dynamic Combination:**
```python
def make_filter(age=None, credit=None, job=None):
    flexible_filter = (
        (Num("age") > age if age else None) &
        (Tag("credit_score") == credit if credit else None) &
        (Text("job") % job if job else None)
    )
    return flexible_filter

# All parameters
combined = make_filter(age=18, credit="high", job="engineer")

# Partial parameters
combined = make_filter(age=18, credit="high")

# No filters (returns None, which becomes "*")
combined = make_filter()
```

### Non-Vector Queries

**FilterQuery:**
```python
from redisvl.query import FilterQuery

has_low_credit = Tag("credit_score") == "low"

filter_query = FilterQuery(
    return_fields=["user", "credit_score", "age", "job"],
    filter_expression=has_low_credit
)

results = index.query(filter_query)
```

**CountQuery:**
```python
from redisvl.query import CountQuery

has_low_credit = Tag("credit_score") == "low"
filter_query = CountQuery(filter_expression=has_low_credit)
count = index.query(filter_query)
print(f"{count} records match the filter")
```

**RangeQuery:**
```python
from redisvl.query import RangeQuery

range_query = RangeQuery(
    vector=[0.1, 0.1, 0.5],
    vector_field_name="user_embedding",
    return_fields=["user", "credit_score", "age", "job"],
    distance_threshold=0.2
)

results = index.query(range_query)

# Adjust threshold
range_query.set_distance_threshold(0.1)

# Add filter
is_engineer = Text("job") == "engineer"
range_query.set_filter(is_engineer)
```

### Advanced Query Modifiers

```python
query = VectorQuery(
    vector=[0.1, 0.1, 0.5],
    vector_field_name="user_embedding",
    return_fields=["user", "credit_score", "age", "job"],
    num_results=5,
    filter_expression=is_engineer
).sort_by("age", asc=False).dialect(3)

# Raw query string
str(query)
# '@job:("engineer")=>[KNN 5 @user_embedding $vector AS vector_distance] RETURN 6 user credit_score age job vector_distance SORTBY age DESC DIALECT 3 LIMIT 0 5'
```

---

## LLM Caching

### Initializing SemanticCache

```python
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

llmcache = SemanticCache(
    name="llmcache",
    redis_url="redis://localhost:6379",
    distance_threshold=0.1,
    vectorizer=HFTextVectorizer("redis/langcache-embed-v1"),
)
```

### Basic Cache Usage

```python
# Check cache
question = "What is the capital of France?"
if response := llmcache.check(prompt=question):
    print(response[0]['response'])
else:
    print("Cache miss")

# Store in cache
llmcache.store(
    prompt=question,
    response="Paris",
    metadata={"city": "Paris", "country": "france"}
)

# Check again (semantic similarity)
similar_question = "What actually is the capital of France?"
result = llmcache.check(prompt=similar_question)
print(result[0]['response'])  # "Paris"
```

### Customize Distance Threshold

```python
# Widen threshold
llmcache.set_threshold(0.5)

# Check with new threshold
result = llmcache.check(prompt="What is the capital city of the country in Europe that also has a city named Nice?")
```

### Utilize TTL

```python
# Set global TTL (5 seconds)
llmcache.set_ttl(5)

# Store entry
llmcache.store("This is a TTL test", "This is a TTL test response")

# Wait for expiration
import time
time.sleep(6)

# Check (should be empty)
result = llmcache.check("This is a TTL test")  # []

# Reset TTL to None (long-lived)
llmcache.set_ttl()
```

### Cache Access Controls, Tags & Filters

**Single Filterable Field:**
```python
private_cache = SemanticCache(
    name="private_cache",
    filterable_fields=[{"name": "user_id", "type": "tag"}]
)

private_cache.store(
    prompt="What is the phone number linked to my account?",
    response="The number on file is 123-555-0000",
    filters={"user_id": "abc"},
)

# Check with filter
from redisvl.query.filter import Tag
user_id_filter = Tag("user_id") == "abc"

response = private_cache.check(
    prompt="What is the phone number linked to my account?",
    filter_expression=user_id_filter,
    num_results=2
)
```

**Multiple Filterable Fields:**
```python
complex_cache = SemanticCache(
    name='account_data',
    filterable_fields=[
        {"name": "user_id", "type": "tag"},
        {"name": "account_type", "type": "tag"},
        {"name": "account_balance", "type": "numeric"},
        {"name": "transaction_amount", "type": "numeric"}
    ]
)

complex_cache.store(
    prompt="what is my most recent checking account transaction under $100?",
    response="Your most recent transaction was for $75",
    filters={"user_id": "abc", "account_type": "checking", "transaction_amount": 75},
)

# Check with complex filter
from redisvl.query.filter import Num
value_filter = Num("transaction_amount") > 100
account_filter = Tag("account_type") == "checking"
complex_filter = value_filter & account_filter

complex_cache.set_threshold(0.3)
response = complex_cache.check(
    prompt="what is my most recent checking account transaction?",
    filter_expression=complex_filter,
    num_results=5
)
```

### Cleanup

```python
# Clear cache (keeps index)
llmcache.clear()

# Delete cache and index
llmcache.delete()
```

---

## Vectorizers

### OpenAI

```python
from redisvl.utils.vectorize import OpenAITextVectorizer
import os

oai = OpenAITextVectorizer(
    model="text-embedding-ada-002",
    api_config={"api_key": os.getenv("OPENAI_API_KEY")},
)

# Single embedding
test = oai.embed("This is a test sentence.")
print("Vector dimensions:", len(test))

# Batch embeddings
sentences = ["That is a happy dog", "That is a happy person", "Today is a sunny day"]
embeddings = oai.embed_many(sentences)

# Async embeddings
embeddings = await oai.aembed_many(sentences)
```

### Azure OpenAI

```python
from redisvl.utils.vectorize import AzureOpenAITextVectorizer

az_oai = AzureOpenAITextVectorizer(
    model="text-embedding-ada-002",  # Your deployment name
    api_config={
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("OPENAI_API_VERSION"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT")
    },
)

test = az_oai.embed("This is a test sentence.")
```

### HuggingFace

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from redisvl.utils.vectorize import HFTextVectorizer

hf = HFTextVectorizer(model="sentence-transformers/all-mpnet-base-v2")

# Single embedding
test = hf.embed("This is a test sentence.")

# Batch embeddings (as buffer for Hash storage)
embeddings = hf.embed_many(sentences, as_buffer=True)
```

### VertexAI

```python
from redisvl.utils.vectorize import VertexAITextVectorizer

vtx = VertexAITextVectorizer(api_config={
    "project_id": os.getenv("GCP_PROJECT_ID"),
    "location": os.getenv("GCP_LOCATION"),
    "google_application_credentials": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
})

test = vtx.embed("This is a test sentence.")
```

### Cohere

```python
from redisvl.utils.vectorize import CohereTextVectorizer

co = CohereTextVectorizer(
    model="embed-english-v3.0",
    api_config={"api_key": os.getenv("COHERE_API_KEY")},
)

# Embed query
query_embedding = co.embed("This is a test sentence.", input_type='search_query')

# Embed document
doc_embedding = co.embed("This is a test sentence.", input_type='search_document')
```

### VoyageAI

```python
from redisvl.utils.vectorize import VoyageAITextVectorizer

vo = VoyageAITextVectorizer(
    model="voyage-law-2",
    api_config={"api_key": os.getenv("VOYAGE_API_KEY")},
)

# Embed query
query_embedding = vo.embed("This is a test sentence.", input_type='query')

# Embed document
doc_embedding = vo.embed("This is a test sentence.", input_type='document')
```

### Mistral AI

```python
from redisvl.utils.vectorize import MistralAITextVectorizer

mistral = MistralAITextVectorizer()

# Async embedding
test = await mistral.aembed("This is a test sentence.")
```

### Amazon Bedrock

```python
from redisvl.utils.vectorize import BedrockTextVectorizer

bedrock = BedrockTextVectorizer(
    model="amazon.titan-embed-text-v2:0"
)

# Single embedding
embedding = bedrock.embed("This is a test sentence.")

# Multiple embeddings
embeddings = bedrock.embed_many(sentences)
```

### Custom Vectorizers

```python
from redisvl.utils.vectorize import CustomTextVectorizer

def generate_embeddings(text_input, **kwargs):
    return [0.101] * 768

custom_vectorizer = CustomTextVectorizer(generate_embeddings)

# Use with SemanticCache
from redisvl.extensions.cache.llm import SemanticCache
cache = SemanticCache(name="custom_cache", vectorizer=custom_vectorizer)
```

### Selecting Float Data Type

```python
# float16
vectorizer = HFTextVectorizer(dtype="float16")
float16_bytes = vectorizer.embed('test sentence', as_buffer=True)

# float32 (default)
vectorizer = HFTextVectorizer(dtype="float32")

# float64
vectorizer_64 = HFTextVectorizer(dtype='float64')
float64_bytes = vectorizer_64.embed('test sentence', as_buffer=True)
```

---

## Hash vs JSON Storage

### Hash Storage (Default)

**Characteristics:**
- Best for performance and storage space
- Single-level dictionary structure
- Vectors must be bytes

**Schema:**
```python
hash_schema = {
    "index": {
        "name": "user-hash",
        "prefix": "user-hash-docs",
        "storage_type": "hash",  # default
    },
    "fields": [
        {"name": "user", "type": "tag"},
        {"name": "credit_score", "type": "tag"},
        {"name": "job", "type": "text"},
        {"name": "age", "type": "numeric"},
        {"name": "office_location", "type": "geo"},
        {
            "name": "user_embedding",
            "type": "vector",
            "attrs": {
                "dims": 3,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32"
            }
        }
    ],
}

# Data must have vectors as bytes
data = [
    {
        'user': 'john',
        'age': 18,
        'job': 'engineer',
        'credit_score': 'high',
        'office_location': '-122.4194,37.7749',
        'user_embedding': np.array([0.1, 0.1, 0.5], dtype=np.float32).tobytes()
    }
]
```

### JSON Storage

**Characteristics:**
- Best for ease of use and flexibility
- Native JSON data
- Vectors as float arrays

**Schema:**
```python
json_schema = {
    "index": {
        "name": "user-json",
        "prefix": "user-json-docs",
        "storage_type": "json",
    },
    "fields": [
        {"name": "user", "type": "tag"},
        {"name": "credit_score", "type": "tag"},
        {"name": "job", "type": "text"},
        {"name": "age", "type": "numeric"},
        {"name": "office_location", "type": "geo"},
        {
            "name": "user_embedding",
            "type": "vector",
            "attrs": {
                "dims": 3,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32"
            }
        }
    ],
}

# Convert vectors to float arrays
from redisvl.redis.utils import buffer_to_array

json_data = data.copy()
for d in json_data:
    d['user_embedding'] = buffer_to_array(d['user_embedding'], dtype='float32')
```

### Working with Nested JSON Data

**Full JSON Path Support:**
```python
bike_schema = {
    "index": {
        "name": "bike-json",
        "prefix": "bike-json",
        "storage_type": "json",
    },
    "fields": [
        {
            "name": "model",
            "type": "tag",
            "path": "$.metadata.model"  # JSONPath
        },
        {
            "name": "brand",
            "type": "tag",
            "path": "$.metadata.brand"
        },
        {
            "name": "price",
            "type": "numeric",
            "path": "$.metadata.price"
        },
        {
            "name": "bike_embedding",
            "type": "vector",
            "attrs": {
                "dims": 768,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32"
            }
        }
    ],
}

# Query with JSONPath in return_fields
query = VectorQuery(
    vector=vec,
    vector_field_name="bike_embedding",
    return_fields=[
        "brand",
        "name",
        "$.metadata.type"  # Full path for non-indexed fields
    ]
)
```

---

## Rerankers

### HuggingFace Cross-Encoder Reranker

```python
from redisvl.utils.rerank import HFCrossEncoderReranker

cross_encoder_reranker = HFCrossEncoderReranker("BAAI/bge-reranker-base")

query = "What is the capital of the United States?"
docs = [
    "Carson City is the capital city of the American state of Nevada...",
    "Washington, D.C. is the capital of the United States...",
    # ... more documents
]

results, scores = cross_encoder_reranker.rank(query=query, docs=docs)

for result, score in zip(results, scores):
    print(score, " -- ", result)
```

### Cohere Reranker

```python
from redisvl.utils.rerank import CohereReranker

cohere_reranker = CohereReranker(
    limit=3,
    api_config={"api_key": os.getenv("COHERE_API_KEY")}
)

results, scores = cohere_reranker.rank(query=query, docs=docs)

# With semi-structured documents
docs = [
    {
        "source": "wiki",
        "passage": "Carson City is the capital city..."
    },
    # ... more documents
]

results, scores = cohere_reranker.rank(
    query=query,
    docs=docs,
    rank_by=["passage", "source"]  # Rank by multiple fields
)
```

### VoyageAI Reranker

```python
from redisvl.utils.rerank import VoyageAIReranker

reranker = VoyageAIReranker(
    model="rerank-lite-1",
    limit=3,
    api_config={"api_key": os.getenv("VOYAGE_API_KEY")}
)

results, scores = reranker.rank(query=query, docs=docs)
```

---

## LLM Message History

### Basic Message History

```python
from redisvl.extensions.message_history import MessageHistory

chat_history = MessageHistory(name='student tutor')

# Add messages one at a time
chat_history.add_message({
    "role": "system",
    "content": "You are a helpful geography tutor..."
})

# Add multiple messages
chat_history.add_messages([
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "llm", "content": "The capital is Paris."},
    {"role": "user", "content": "And what is the capital of Spain?"},
    {"role": "llm", "content": "The capital is Madrid."},
])

# Get recent history
context = chat_history.get_recent()
for message in context:
    print(message)

# Convenience method for prompt/response pairs
prompt = "what is the size of England compared to Portugal?"
response = "England is larger in land area than Portugal..."
chat_history.store(prompt, response)
```

### Managing Multiple Users

```python
# Use session_tag to separate conversations
chat_history.add_message(
    {"role": "system", "content": "You are a helpful algebra tutor..."},
    session_tag='student two'
)

chat_history.add_messages([
    {"role": "user", "content": "What is the value of x in 2x + 3 = 7?"},
    {"role": "llm", "content": "The value of x is 2."},
], session_tag='student two')

# Get history for specific session
for message in chat_history.get_recent(session_tag='student two'):
    print(message)
```

### Semantic Message History

```python
from redisvl.extensions.message_history import SemanticMessageHistory

semantic_history = SemanticMessageHistory(name='tutor')

# Add messages from regular history
semantic_history.add_messages(chat_history.get_recent(top_k=8))

# Get semantically relevant context
prompt = "what have I learned about the size of England?"
semantic_history.set_distance_threshold(0.35)
context = semantic_history.get_relevant(prompt)

for message in context:
    print(message)

# Adjust threshold (0.0 = exact match, 1.0 = everything)
semantic_history.set_distance_threshold(0.7)
larger_context = semantic_history.get_relevant(prompt)
```

### Conversation Control

```python
# Store incorrect information
semantic_history.store(
    prompt="what is the smallest country in Europe?",
    response="Monaco is the smallest country..."  # Incorrect
)

# Get key of incorrect message
context = semantic_history.get_recent(top_k=1, raw=True)
bad_key = context[0]['entry_id']

# Drop incorrect message
semantic_history.drop(bad_key)

# Verify removal
corrected_context = semantic_history.get_recent()
```

### Cleanup

```python
chat_history.clear()
semantic_history.clear()
```

---

## Semantic Routing

### Define Routes

```python
from redisvl.extensions.router import Route

technology = Route(
    name="technology",
    references=[
        "what are the latest advancements in AI?",
        "tell me about the newest gadgets",
        "what's trending in tech?"
    ],
    metadata={"category": "tech", "priority": 1},
    distance_threshold=0.71
)

sports = Route(
    name="sports",
    references=[
        "who won the game last night?",
        "tell me about the upcoming sports events",
        "what's the latest in the world of sports?",
    ],
    metadata={"category": "sports", "priority": 2},
    distance_threshold=0.72
)
```

### Initialize SemanticRouter

```python
from redisvl.extensions.router import SemanticRouter
from redisvl.utils.vectorize import HFTextVectorizer

router = SemanticRouter(
    name="topic-router",
    vectorizer=HFTextVectorizer(),
    routes=[technology, sports, entertainment],
    redis_url="redis://localhost:6379",
    overwrite=True
)
```

### Simple Routing

```python
# Route single statement
route_match = router("Can you tell me about the latest in artificial intelligence?")
print(route_match)  # RouteMatch(name='technology', distance=0.419)

# Route with miss (no match)
route_match = router("are aliens real?")
print(route_match)  # RouteMatch(name=None, distance=None)

# Route to multiple routes
from redisvl.extensions.router.schema import DistanceAggregationMethod

route_matches = router.route_many(
    "How is AI used in basketball?",
    max_k=3,
    aggregation_method=DistanceAggregationMethod.min
)
```

### Update Routing Config

```python
from redisvl.extensions.router import RoutingConfig

router.update_routing_config(
    RoutingConfig(
        aggregation_method=DistanceAggregationMethod.min,
        max_k=3
    )
)
```

### Router Serialization

```python
# To dictionary
router_dict = router.to_dict()

# From dictionary
router2 = SemanticRouter.from_dict(router_dict, redis_url="redis://localhost:6379")

# To YAML
router.to_yaml("router.yaml", overwrite=True)

# From YAML
router3 = SemanticRouter.from_yaml("router.yaml", redis_url="redis://localhost:6379")
```

### Add Route References

```python
# Add references to existing route
router.add_route_references(
    route_name="technology",
    references=["latest AI trends", "new tech gadgets"]
)
```

### Get Route References

```python
# By route name
refs = router.get_route_references(route_name="technology")

# By reference IDs
refs = router.get_route_references(reference_ids=[refs[0]["reference_id"]])
```

### Delete Route References

```python
# By route name
deleted_count = router.delete_route_references(route_name="sports")

# By reference IDs
deleted_count = router.delete_route_references(reference_ids=[refs[0]["reference_id"]])
```

### Cleanup

```python
# Clear all routes (keeps index)
router.clear()

# Delete router and index
router.delete()
```

---

## SVS-VAMANA Vector Search

### Prerequisites
- Redis >= 8.2.0
- RediSearch >= 2.8.10
- Only supports FLOAT16 and FLOAT32 datatypes

### Quick Start with CompressionAdvisor

```python
from redisvl.utils import CompressionAdvisor

dims = 1024

# Get recommended configuration
config = CompressionAdvisor.recommend(
    dims=dims,
    priority="balanced"  # Options: "memory", "speed", "balanced"
)

print("Recommended Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Estimate memory savings
savings = CompressionAdvisor.estimate_memory_savings(
    config["compression"],
    dims,
    config.get("reduce")
)
print(f"Estimated Memory Savings: {savings}%")
```

### Creating an SVS-VAMANA Index

```python
schema = {
    "index": {
        "name": "svs_demo",
        "prefix": "doc",
    },
    "fields": [
        {"name": "content", "type": "text"},
        {"name": "category", "type": "tag"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": dims,
                **config,  # Use recommended configuration
                "distance_metric": "cosine"
            }
        }
    ]
}

index = SearchIndex.from_dict(schema, redis_url=REDIS_URL)
index.create(overwrite=True)
```

### Loading Sample Data

```python
import numpy as np
from redisvl.redis.utils import array_to_buffer

# Use reduced dimensions if LeanVec compression is applied
vector_dims = config.get("reduce", dims)

data_to_load = []
for doc in sample_documents:
    base_vector = np.random.random(vector_dims).astype(np.float32)
    
    # Convert to specified datatype
    if config["datatype"] == "float16":
        base_vector = base_vector.astype(np.float16)
    
    data_to_load.append({
        "content": doc["content"],
        "category": doc["category"],
        "embedding": array_to_buffer(base_vector, dtype=config["datatype"])
    })

index.load(data_to_load)
```

### Performing Vector Searches

```python
# Query vector must match index datatype and dimensions
vector_dims = config.get("reduce", dims)
if config["datatype"] == "float16":
    query_vector = np.random.random(vector_dims).astype(np.float16)
else:
    query_vector = np.random.random(vector_dims).astype(np.float32)

query = VectorQuery(
    vector=query_vector.tolist(),
    vector_field_name="embedding",
    return_fields=["content", "category"],
    num_results=5
)

results = index.query(query)
```

### Runtime Parameters

```python
# Basic query with default parameters
basic_query = VectorQuery(
    vector=query_vector.tolist(),
    vector_field_name="embedding",
    return_fields=["content", "category"],
    num_results=5
)

# Tuned query for higher recall
tuned_query = VectorQuery(
    vector=query_vector.tolist(),
    vector_field_name="embedding",
    return_fields=["content", "category"],
    num_results=5,
    search_window_size=40,      # Larger window for better recall
    use_search_history='ON',    # Use search history
    search_buffer_capacity=50   # Larger buffer capacity
)

# Range query with runtime parameters
from redisvl.query import VectorRangeQuery

range_query = VectorRangeQuery(
    vector=query_vector.tolist(),
    vector_field_name="embedding",
    return_fields=["content", "category"],
    distance_threshold=0.3,
    epsilon=0.05,               # Approximation factor
    search_window_size=30,      # Search window size
    use_search_history='AUTO'   # Automatic history management
)
```

### Compression Types

**LVQ (Learned Vector Quantization):**
- LVQ4: 4 bits per dimension (87.5% memory savings)
- LVQ4x4: 8 bits per dimension (75% memory savings)
- LVQ4x8: 12 bits per dimension (62.5% memory savings)
- LVQ8: 8 bits per dimension (75% memory savings)

**LeanVec (Compression + Dimensionality Reduction):**
- LeanVec4x8: 12 bits per dimension + dimensionality reduction
- LeanVec8x8: 16 bits per dimension + dimensionality reduction

### Manual Configuration

```python
manual_schema = {
    "index": {
        "name": "svs_manual",
        "prefix": "manual",
    },
    "fields": [
        {"name": "content", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 768,
                "algorithm": "svs-vamana",
                "datatype": "float32",
                "distance_metric": "cosine",
                
                # Graph construction parameters
                "graph_max_degree": 64,
                "construction_window_size": 300,
                
                # Search parameters
                "search_window_size": 40,
                
                # Compression settings
                "compression": "LVQ4x4",
                "training_threshold": 10000,
            }
        }
    ]
}
```

### Best Practices

**When to Use SVS-VAMANA:**
- Large datasets (>10K vectors) where memory efficiency matters
- High-dimensional vectors (>512 dimensions) that benefit from compression
- Applications that can tolerate slight recall trade-offs

**Parameter Tuning:**
- Start with CompressionAdvisor recommendations
- Use LeanVec for high-dimensional vectors (≥1024 dims)
- Use LVQ for lower-dimensional vectors (<1024 dims)
- `search_window_size`: Start with 20, increase to 40-100 for higher recall
- `epsilon`: Use 0.01-0.05 for range queries

---

## Advanced Query Types

### TextQuery: Full Text Search

```python
from redisvl.query import TextQuery

# Basic text search
text_query = TextQuery(
    text="running shoes",
    text_field_name="brief_description",
    return_fields=["product_id", "brief_description", "category", "price"],
    num_results=5
)

results = index.query(text_query)

# Different scoring algorithms
bm25_query = TextQuery(
    text="comfortable shoes",
    text_field_name="brief_description",
    text_scorer="BM25STD",  # or "TFIDF"
    return_fields=["product_id", "brief_description", "price"],
    num_results=3
)

# Text search with filters
from redisvl.query.filter import Tag, Num

filtered_text_query = TextQuery(
    text="shoes",
    text_field_name="brief_description",
    filter_expression=Tag("category") == "footwear",
    return_fields=["product_id", "brief_description", "category", "price"],
    num_results=5
)

# Multiple fields with weights
weighted_query = TextQuery(
    text="shoes",
    text_field_name={"brief_description": 1.0, "full_description": 0.5},
    return_fields=["product_id", "brief_description"],
    num_results=3
)

# Custom stopwords
query_with_stopwords = TextQuery(
    text="the best shoes for running",
    text_field_name="brief_description",
    stopwords="english",  # or custom list: ["for", "with"]
    return_fields=["product_id", "brief_description"],
    num_results=3
)
```

### AggregateHybridQuery: Combining Text and Vector Search

```python
from redisvl.query import AggregateHybridQuery

# Basic hybrid query
hybrid_query = AggregateHybridQuery(
    text="running shoes",
    text_field_name="brief_description",
    vector=[0.1, 0.2, 0.1],
    vector_field_name="text_embedding",
    return_fields=["product_id", "brief_description", "category", "price"],
    num_results=5
)

results = index.query(hybrid_query)

# Adjust alpha parameter (controls weight between vector and text)
# alpha=1.0: Pure vector search
# alpha=0.0: Pure text search
# alpha=0.7 (default): 70% vector, 30% text

vector_heavy_query = AggregateHybridQuery(
    text="comfortable",
    text_field_name="brief_description",
    vector=[0.15, 0.25, 0.15],
    vector_field_name="text_embedding",
    alpha=0.9,  # 90% vector, 10% text
    return_fields=["product_id", "brief_description"],
    num_results=3
)

# Hybrid query with filters
filtered_hybrid_query = AggregateHybridQuery(
    text="professional equipment",
    text_field_name="brief_description",
    vector=[0.9, 0.1, 0.05],
    vector_field_name="text_embedding",
    filter_expression=Num("price") > 100,
    return_fields=["product_id", "brief_description", "category", "price"],
    num_results=5
)

# Different text scorers
hybrid_tfidf = AggregateHybridQuery(
    text="shoes support",
    text_field_name="brief_description",
    vector=[0.12, 0.18, 0.12],
    vector_field_name="text_embedding",
    text_scorer="TFIDF",
    return_fields=["product_id", "brief_description"],
    num_results=3
)

# Note: AggregateHybridQuery does NOT support runtime parameters
# Use VectorQuery or VectorRangeQuery for runtime parameter support
```

### MultiVectorQuery: Multi-Vector Search

```python
from redisvl.query import MultiVectorQuery, Vector

# Define multiple vectors for the query
text_vector = Vector(
    vector=[0.1, 0.2, 0.1],
    field_name="text_embedding",
    dtype="float32",
    weight=0.7  # 70% weight for text embedding
)

image_vector = Vector(
    vector=[0.8, 0.1],
    field_name="image_embedding",
    dtype="float32",
    weight=0.3  # 30% weight for image embedding
)

# Create multi-vector query
multi_vector_query = MultiVectorQuery(
    vectors=[text_vector, image_vector],
    return_fields=["product_id", "brief_description", "category"],
    num_results=5
)

results = index.query(multi_vector_query)

# Adjust vector weights
text_vec = Vector(
    vector=[0.9, 0.1, 0.05],
    field_name="text_embedding",
    dtype="float32",
    weight=0.2  # 20% weight
)

image_vec = Vector(
    vector=[0.1, 0.9],
    field_name="image_embedding",
    dtype="float32",
    weight=0.8  # 80% weight
)

image_heavy_query = MultiVectorQuery(
    vectors=[text_vec, image_vec],
    return_fields=["product_id", "brief_description", "category"],
    num_results=3
)

# Multi-vector query with filters
filtered_multi_query = MultiVectorQuery(
    vectors=[text_vector, image_vector],
    filter_expression=Tag("category") == "footwear",
    return_fields=["product_id", "brief_description", "category", "price"],
    num_results=5
)
```

### Index-Level Stopwords Configuration

```python
# Disable stopwords completely
stopwords_schema = {
    "index": {
        "name": "company_index",
        "prefix": "company:",
        "storage_type": "hash",
        "stopwords": []  # STOPWORDS 0 - disable stopwords
    },
    "fields": [
        {"name": "company_name", "type": "text"},
        {"name": "description", "type": "text"}
    ]
}

# Custom stopwords
custom_stopwords_schema = {
    "index": {
        "name": "custom_stopwords_index",
        "prefix": "custom:",
        "stopwords": ["inc", "llc", "corp"]  # Custom stopwords list
    },
    "fields": [
        {"name": "name", "type": "text"}
    ]
}
```

---

## Caching Embeddings

### Initializing EmbeddingsCache

```python
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import HFTextVectorizer

# Initialize vectorizer
vectorizer = HFTextVectorizer(
    model="redis/langcache-embed-v1",
    cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME")
)

# Initialize cache
cache = EmbeddingsCache(
    name="embedcache",
    redis_url="redis://localhost:6379",
    ttl=None  # Optional TTL in seconds (None means no expiration)
)
```

### Basic Usage

```python
# Store embedding
text = "What is machine learning?"
model_name = "redis/langcache-embed-v1"
embedding = vectorizer.embed(text)
metadata = {"category": "ai", "source": "user_query"}

key = cache.set(
    text=text,
    model_name=model_name,
    embedding=embedding,
    metadata=metadata
)

# Retrieve embedding
if result := cache.get(text=text, model_name=model_name):
    print(f"Found in cache: {result['text']}")
    print(f"Model: {result['model_name']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Embedding shape: {np.array(result['embedding']).shape}")

# Check existence
exists = cache.exists(text=text, model_name=model_name)

# Remove entry
cache.drop(text=text, model_name=model_name)
```

### Key-Based Operations

```python
# Store entry
key = cache.set(
    text=text,
    model_name=model_name,
    embedding=embedding,
    metadata=metadata
)

# Check existence by key
exists_by_key = cache.exists_by_key(key)

# Retrieve by key
result_by_key = cache.get_by_key(key)

# Drop by key
cache.drop_by_key(key)
```

### Batch Operations

```python
# Prepare batch items
texts = [
    "What is machine learning?",
    "How do neural networks work?",
    "What is deep learning?"
]
embeddings = [vectorizer.embed(t) for t in texts]

batch_items = [
    {
        "text": texts[0],
        "model_name": model_name,
        "embedding": embeddings[0],
        "metadata": {"category": "ai", "type": "question"}
    },
    # ... more items
]

# Store multiple embeddings
keys = cache.mset(batch_items)

# Check if multiple exist
exist_results = cache.mexists(texts, model_name)

# Retrieve multiple
results = cache.mget(texts, model_name)

# Delete multiple
cache.mdrop(texts, model_name)

# Key-based batch operations
# cache.mget_by_keys(keys)
# cache.mexists_by_keys(keys)
# cache.mdrop_by_keys(keys)
```

### Working with TTL

```python
# Create cache with default TTL
ttl_cache = EmbeddingsCache(
    name="ttl_cache",
    redis_url="redis://localhost:6379",
    ttl=5  # 5 second TTL
)

# Store with custom TTL override
key1 = ttl_cache.set(
    text="Short-lived entry",
    model_name=model_name,
    embedding=embedding,
    ttl=1  # Override with 1 second TTL
)

# Store with default TTL
key2 = ttl_cache.set(
    text="Default TTL entry",
    model_name=model_name,
    embedding=embedding
    # Uses default 5 seconds
)
```

### Async Support

```python
async def async_cache_demo():
    # Store asynchronously
    key = await cache.aset(
        text="Async embedding",
        model_name=model_name,
        embedding=embedding,
        metadata={"async": True}
    )
    
    # Check existence
    exists = await cache.aexists_by_key(key)
    
    # Retrieve
    result = await cache.aget_by_key(key)
    
    # Remove
    await cache.adrop_by_key(key)

# Run async demo
await async_cache_demo()
```

### Integration with Vectorizers

```python
# Create cache
example_cache = EmbeddingsCache(
    name="example_cache",
    redis_url="redis://localhost:6379",
    ttl=3600  # 1 hour TTL
)

# Attach cache to vectorizer
vectorizer = HFTextVectorizer(
    model=model_name,
    cache=example_cache,
    cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME")
)

# Vectorizer will automatically use cache
embedding = vectorizer.embed(query)  # Checks cache first, computes if miss

# Skip cache if needed
embedding = vectorizer.embed(query, skip_cache=True)
```

### Cleanup

```python
# Clear all cache entries
cache.clear()
```

---

## Summary

This implementation reference covers all major features of RedisVL user guides:

1. **Getting Started**: Schema definition, index creation, data loading, basic queries
2. **Querying**: Tag, numeric, timestamp, text, geographic filters and combinations
3. **LLM Caching**: Semantic caching with thresholds, TTL, and access controls
4. **Vectorizers**: Support for OpenAI, HuggingFace, Cohere, Azure, VertexAI, Bedrock, Mistral, VoyageAI, and custom
5. **Storage**: Hash vs JSON storage options and nested JSON support
6. **Rerankers**: Cross-encoder, Cohere, and VoyageAI rerankers
7. **Message History**: Simple and semantic message history for LLM conversations
8. **Semantic Routing**: Route queries based on semantic similarity
9. **SVS-VAMANA**: Compressed vector search with memory optimization
10. **Advanced Queries**: TextQuery, AggregateHybridQuery, MultiVectorQuery
11. **Embedding Cache**: Cache embeddings to reduce computation costs

All code examples are production-ready and can be directly used in your applications.

