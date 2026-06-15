# Configuration

Engram uses `pydantic-settings`. Environment variables use the `ENGRAM_`
prefix, and `.env` is read from the current working directory.

## Local Development

The repository `docker-compose.yml` defaults to this PostgreSQL URL:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
```

Use local embeddings when you do not want API-key-backed embedding calls:

```bash
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Install the matching extra:

```bash
pip install -e ".[dev,examples,sentence-transformers]"
```

## OpenAI Setup

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...
```

Install:

```bash
pip install -e ".[dev,examples,openai]"
```

The LLM provider is optional. Direct `add()`, `search()`, `trace_recall()`,
task/event storage, graph operations, and heuristic long-input extraction work
without it. `add_conversation()`, LLM query expansion, richer memory jobs, and
any LLM-backed reader over retrieved context need an LLM provider.

## Database Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DATABASE_URL` | `postgresql://localhost:5432/engram` | PostgreSQL connection URL |
| `ENGRAM_MIN_POOL_SIZE` | `5` | Minimum asyncpg pool size |
| `ENGRAM_MAX_POOL_SIZE` | `20` | Maximum asyncpg pool size |
| `ENGRAM_CONNECTION_TIMEOUT` | `30.0` | Connection timeout in seconds |
| `ENGRAM_COMMAND_TIMEOUT` | `60.0` | SQL command timeout in seconds |

Use PostgreSQL with the `vector` and `pg_trgm` extensions. The local compose
stack uses `pgvector/pgvector:pg16`.

## Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_EMBEDDING_PROVIDER` | `openai` | `openai`, `sentence-transformers`, `cohere`, `ollama`, or `huggingface` |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | Provider-specific model |
| `ENGRAM_EMBEDDING_DIMENSION` | unset | Optional explicit dimension. Usually auto-detected |
| `ENGRAM_EMBEDDING_BATCH_SIZE` | `100` | Batch size for embedding writes |
| `ENGRAM_EMBEDDING_CACHE_SIZE` | `1000` | In-memory embedding cache entries. `0` disables caching |
| `ENGRAM_EMBEDDING_MAX_INPUT_CHARS` | `30000` | Text longer than this is truncated before embedding |
| `ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE` | `false` | Permit clearing existing embeddings when the provider dimension changes |

Dimension changes are protected by default. If a database already has 1536
dimension embeddings and you connect with a 384 dimension model, Engram raises a
configuration error instead of clearing stored vectors. Use a fresh database, or
set `ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true` only when you plan to
re-embed affected memories.

## Provider Packages

| Provider | Extra | Notes |
|----------|-------|-------|
| OpenAI embeddings and LLM | `engram[openai]` | Uses `ENGRAM_OPENAI_API_KEY`; `ENGRAM_OPENAI_BASE_URL` supports compatible APIs |
| Anthropic LLM | `engram[anthropic]` | Uses `ENGRAM_ANTHROPIC_API_KEY` |
| Cohere embeddings | `engram[cohere]` | Uses `ENGRAM_COHERE_API_KEY` |
| Ollama embeddings or LLM | `engram[http]` | Uses `ENGRAM_OLLAMA_BASE_URL`, default `http://localhost:11434` |
| HuggingFace Inference embeddings | `engram[http]` | Uses `ENGRAM_HF_API_KEY` when required |
| Groq LLM | `engram[http]` | Uses `ENGRAM_GROQ_API_KEY` |
| LiteLLM LLM | `engram[litellm]` | Uses LiteLLM model naming and provider env |
| Sentence Transformers | `engram[sentence-transformers]` | Local embedding models |
| Cross-encoder reranking | `engram[rerank]` | Needed for `search(..., rerank=True)` |

`engram[all]` installs all provider extras plus example dependencies.

## LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_LLM_PROVIDER` | unset | `openai`, `anthropic`, `ollama`, `groq`, or `litellm` |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | Provider-specific model |
| `ENGRAM_OPENAI_API_KEY` | unset | OpenAI key for embeddings and LLM |
| `ENGRAM_OPENAI_BASE_URL` | unset | OpenAI-compatible endpoint |
| `ENGRAM_ANTHROPIC_API_KEY` | unset | Anthropic key |
| `ENGRAM_GROQ_API_KEY` | unset | Groq key |
| `ENGRAM_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |

## Search Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_WEIGHT_SEMANTIC` | `0.40` | Vector similarity weight |
| `ENGRAM_WEIGHT_KEYWORD` | `0.20` | Full-text weight |
| `ENGRAM_WEIGHT_DECAY` | `0.25` | Recency/access weight |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.15` | Importance weight |
| `ENGRAM_DECAY_RATE` | `0.995` | Hourly decay base. Must be greater than `0` and less than `1` |
| `ENGRAM_SEARCH_CANDIDATE_MULTIPLIER` | `5` | Overfetch factor before fusion or reranking |
| `ENGRAM_DEFAULT_SEARCH_LIMIT` | `10` | Default result limit |
| `ENGRAM_MAX_SEARCH_LIMIT` | `100` | API-level max result limit |
| `ENGRAM_NEAR_DUPLICATE_THRESHOLD` | `0.95` | Cosine threshold for near-duplicate suppression. `1.0` disables the guard |
| `ENGRAM_HNSW_EF_SEARCH` | unset | Optional per-connection `hnsw.ef_search` override |
| `ENGRAM_TEXT_SEARCH_CONFIG` | `english` | PostgreSQL text search config for generated tsvector columns |

The four search weights must sum to approximately `1.0`.

## Reranking

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used by `rerank=True` |
| `ENGRAM_RERANKER_BACKEND` | `torch` | `torch`, `onnx`, or `openvino` |

Reranking runs locally and is loaded lazily on the first reranked search.

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_LOG_LEVEL` | `INFO` | Python log level |
| `ENGRAM_LOG_SQL_QUERIES` | `false` | Log SQL statements for debugging |

## Programmatic Settings

```python
from engram import Engram, EngramSettings

settings = EngramSettings(
    database_url="postgresql://engram:engram_secret@localhost:5432/engram",
    embedding_provider="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    llm_provider=None,
)

async with Engram(settings=settings, memory_policy="coding_agent") as engram:
    health = await engram.health_check()
    print(health["status"])
```

Constructor overrides are useful in tests and small scripts:

```python
engram = Engram(
    database_url="postgresql://engram:engram_secret@localhost:5432/engram",
    memory_policy="default",
)
```

## Memory Policy

`memory_policy` is passed to `Engram`; it is not part of `EngramSettings`.

```python
from engram import Engram

async with Engram(memory_policy="default") as engram:
    ...

async with Engram(memory_policy="legal") as engram:
    ...

async with Engram(memory_policy="coding_agent") as engram:
    ...
```

Use a custom `MemoryPolicy` when a domain needs its own type rules, critical
slots, or conflict keys.

## Docker Compose

```yaml
services:
  app:
    environment:
      ENGRAM_DATABASE_URL: postgresql://engram:engram_secret@postgres:5432/engram
      ENGRAM_EMBEDDING_PROVIDER: openai
      ENGRAM_EMBEDDING_MODEL: text-embedding-3-small
      ENGRAM_LLM_PROVIDER: openai
      ENGRAM_LLM_MODEL: gpt-4o-mini
      ENGRAM_OPENAI_API_KEY: ${ENGRAM_OPENAI_API_KEY}
```

## Common Validation Errors

- `ENGRAM_MAX_POOL_SIZE` lower than `ENGRAM_MIN_POOL_SIZE`
- search weights that do not sum to `1.0`
- invalid `ENGRAM_EMBEDDING_DIMENSION`
- invalid `ENGRAM_TEXT_SEARCH_CONFIG`
- missing optional provider package
- missing API key for cloud providers
- embedding model dimension mismatch against an existing populated database
