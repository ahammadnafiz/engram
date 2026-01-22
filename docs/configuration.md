# Configuration

Complete guide to configuring Engram.

## Environment Variables

Engram uses environment variables for configuration, with the `ENGRAM_` prefix.

### Database

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENGRAM_DATABASE_URL` | Yes | - | PostgreSQL connection string |

**Format:**
```
postgresql://user:password@host:port/database
```

**Example:**
```bash
ENGRAM_DATABASE_URL=postgresql://engram:secret@localhost:5432/engram
```

### Embeddings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | No | `openai` | `openai` or `sentence-transformers` |
| `OPENAI_API_KEY` | If using OpenAI | - | Your OpenAI API key |
| `ENGRAM_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model name |
| `ENGRAM_EMBEDDING_DIMENSION` | No | `1536` | Vector dimensions |
| `ENGRAM_EMBEDDING_BATCH_SIZE` | No | `100` | Batch size for bulk embedding |
| `ENGRAM_EMBEDDING_CACHE_SIZE` | No | `1000` | LRU cache size (0 to disable) |

**OpenAI Models:**

| Model | Dimensions | Notes |
|-------|------------|-------|
| `text-embedding-3-small` | 1536 | Fast, cheap, good quality |
| `text-embedding-3-large` | 3072 | Higher quality |
| `text-embedding-ada-002` | 1536 | Legacy model |

**Sentence Transformers Models:**

| Model | Dimensions | Notes |
|-------|------------|-------|
| `all-MiniLM-L6-v2` | 384 | Fast, lightweight |
| `all-mpnet-base-v2` | 768 | Better quality |

### Search Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_WEIGHT_SEMANTIC` | `0.40` | Semantic similarity weight |
| `ENGRAM_WEIGHT_KEYWORD` | `0.20` | Keyword match weight |
| `ENGRAM_WEIGHT_DECAY` | `0.25` | Recency/access weight |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.15` | Importance score weight |

!!! note
    Weights must sum to 1.0

### Memory Decay

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DECAY_RATE` | `0.995` | Decay rate per hour |

**Decay Examples:**

| Rate | After 1 Day | After 1 Week |
|------|-------------|--------------|
| 0.999 | 0.976 | 0.844 |
| 0.995 | 0.887 | 0.512 |
| 0.990 | 0.786 | 0.262 |

### Connection Pool

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_MIN_POOL_SIZE` | `5` | Minimum connections |
| `ENGRAM_MAX_POOL_SIZE` | `20` | Maximum connections |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `ENGRAM_LOG_SQL_QUERIES` | `false` | Log SQL queries (debug) |

## Configuration File

You can also use a `.env` file:

```bash
# .env
ENGRAM_DATABASE_URL=postgresql://engram:secret@localhost:5432/engram
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_PROVIDER=openai
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
```

Engram automatically loads `.env` from the current directory.

## Programmatic Configuration

Override settings in code:

```python
from engram import Engram
from engram.core import EngramSettings

# Custom settings
settings = EngramSettings(
    database_url="postgresql://...",
    embedding_dimension=1536,
    decay_rate=0.995,
    min_pool_size=10,
    max_pool_size=50,
)

# Use custom settings
async with Engram(settings=settings) as engram:
    ...
```

## Custom Embedding Provider

Use any embedding function:

```python
from engram import Engram

# Custom embedding function
async def my_embeddings(texts: list[str]) -> list[list[float]]:
    # Your embedding logic here
    # Must return list of vectors
    return [[0.1, 0.2, ...] for _ in texts]

# Use custom embeddings
async with Engram(
    embedding_fn=my_embeddings,
    embedding_dim=1536  # Must match your vectors
) as engram:
    ...
```

## Docker Configuration

When using Docker, configure via environment:

```yaml
# docker-compose.yml
services:
  engram:
    environment:
      - ENGRAM_DATABASE_URL=postgresql://engram:secret@postgres:5432/engram
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - EMBEDDING_PROVIDER=openai
```

Or use an env file:

```yaml
services:
  engram:
    env_file:
      - .env
```

## Production Recommendations

### Database

```bash
# Use connection pooling (PgBouncer recommended for high load)
ENGRAM_DATABASE_URL=postgresql://user:pass@pgbouncer:6432/engram

# Larger pool for production
ENGRAM_MIN_POOL_SIZE=10
ENGRAM_MAX_POOL_SIZE=50
```

### Embeddings

```bash
# Enable caching
ENGRAM_EMBEDDING_CACHE_SIZE=10000

# Batch for bulk operations
ENGRAM_EMBEDDING_BATCH_SIZE=100
```

### Search Tuning

Adjust weights based on your use case:

```bash
# More emphasis on semantic similarity
ENGRAM_WEIGHT_SEMANTIC=0.50
ENGRAM_WEIGHT_KEYWORD=0.15
ENGRAM_WEIGHT_DECAY=0.20
ENGRAM_WEIGHT_IMPORTANCE=0.15

# More emphasis on recency
ENGRAM_WEIGHT_SEMANTIC=0.35
ENGRAM_WEIGHT_KEYWORD=0.15
ENGRAM_WEIGHT_DECAY=0.35
ENGRAM_WEIGHT_IMPORTANCE=0.15
```

### Logging

```bash
# Production logging
LOG_LEVEL=WARNING
ENGRAM_LOG_SQL_QUERIES=false
```

## Validation

Engram validates configuration on startup:

```python
from engram.core import EngramSettings

# This will raise ValidationError if invalid
settings = EngramSettings()

# Check specific values
print(f"Database: {settings.database_url}")
print(f"Pool size: {settings.min_pool_size}-{settings.max_pool_size}")
```

Common validation errors:

- `database_url` is required
- Weights must sum to 1.0
- `decay_rate` must be between 0 and 1
- `embedding_dimension` must be positive
