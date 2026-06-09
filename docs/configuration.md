# Configuration

Engram uses `pydantic-settings` with the `ENGRAM_` environment prefix and reads
`.env` from the current working directory.

## Minimal Local Config

```bash
ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Minimal OpenAI Config

```bash
ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
ENGRAM_EMBEDDING_PROVIDER=openai
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
ENGRAM_LLM_PROVIDER=openai
ENGRAM_LLM_MODEL=gpt-4o-mini
ENGRAM_OPENAI_API_KEY=sk-...
```

## Database

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DATABASE_URL` | `postgresql://localhost:5432/engram` | PostgreSQL connection URL |
| `ENGRAM_MIN_POOL_SIZE` | `5` | Minimum asyncpg pool size |
| `ENGRAM_MAX_POOL_SIZE` | `20` | Maximum asyncpg pool size |
| `ENGRAM_CONNECTION_TIMEOUT` | `30.0` | Connection timeout in seconds |
| `ENGRAM_COMMAND_TIMEOUT` | `60.0` | SQL command timeout in seconds |

## Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_EMBEDDING_PROVIDER` | `openai` | `openai`, `sentence-transformers`, `cohere`, `ollama`, `huggingface` |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | Provider-specific model name |
| `ENGRAM_EMBEDDING_DIMENSION` | unset | Auto-detected when possible |
| `ENGRAM_EMBEDDING_BATCH_SIZE` | `100` | Batch size for bulk embedding |
| `ENGRAM_EMBEDDING_CACHE_SIZE` | `1000` | LRU embedding cache entries, `0` disables |

Provider keys and URLs:

| Variable | Used by |
|----------|---------|
| `ENGRAM_OPENAI_API_KEY` | OpenAI embeddings and LLM |
| `ENGRAM_OPENAI_BASE_URL` | OpenAI-compatible APIs |
| `ENGRAM_COHERE_API_KEY` | Cohere embeddings |
| `ENGRAM_HF_API_KEY` | HuggingFace Inference embeddings |
| `ENGRAM_OLLAMA_BASE_URL` | Ollama embeddings and LLM |

## LLM

LLM features are optional. Without an LLM provider, direct `add()`, `search()`,
`trace_recall()`, task/event storage, and heuristic long-input extraction still
work. LLM-backed features include `add_conversation()`, query expansion in
`deep_search()`, and richer memory job processing.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_LLM_PROVIDER` | unset | `openai`, `anthropic`, `ollama`, `groq`, `litellm` |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | Provider-specific model name |
| `ENGRAM_ANTHROPIC_API_KEY` | unset | Anthropic key |
| `ENGRAM_GROQ_API_KEY` | unset | Groq key |

## Search

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_WEIGHT_SEMANTIC` | `0.40` | Vector similarity weight |
| `ENGRAM_WEIGHT_KEYWORD` | `0.20` | Full-text weight |
| `ENGRAM_WEIGHT_DECAY` | `0.25` | Recency/access weight |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.15` | Importance weight |
| `ENGRAM_DECAY_RATE` | `0.995` | Hourly decay base |
| `ENGRAM_DEFAULT_SEARCH_LIMIT` | `10` | Default result limit |
| `ENGRAM_MAX_SEARCH_LIMIT` | `100` | API-level max search limit |
| `ENGRAM_NEAR_DUPLICATE_THRESHOLD` | `0.95` | Similarity threshold for duplicate guard |

Weights must sum to approximately `1.0`.

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_LOG_LEVEL` | `INFO` | Python log level |
| `ENGRAM_LOG_SQL_QUERIES` | `false` | Log SQL statements for debugging |

## Programmatic Settings

```python
from engram import Engram
from engram.core import EngramSettings

settings = EngramSettings(
    database_url="postgresql://engram:engram@localhost:5432/engram",
    embedding_provider="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    llm_provider=None,
)

async with Engram(settings=settings, memory_policy="coding_agent") as engram:
    ...
```

## Memory Policy

`memory_policy` is passed to `Engram`, not configured through
`EngramSettings`.

```python
async with Engram(memory_policy="default") as engram:
    ...

async with Engram(memory_policy="legal") as engram:
    ...

async with Engram(memory_policy="coding_agent") as engram:
    ...
```

Use a custom `MemoryPolicy` when you need domain-specific memory types,
critical slots, and conflict rules.

## Docker Compose

```yaml
services:
  app:
    environment:
      ENGRAM_DATABASE_URL: postgresql://engram:engram@postgres:5432/engram
      ENGRAM_EMBEDDING_PROVIDER: openai
      ENGRAM_EMBEDDING_MODEL: text-embedding-3-small
      ENGRAM_LLM_PROVIDER: openai
      ENGRAM_LLM_MODEL: gpt-4o-mini
      ENGRAM_OPENAI_API_KEY: ${ENGRAM_OPENAI_API_KEY}
```

## Validation Errors

Common startup errors:

- `max_pool_size` lower than `min_pool_size`
- search weights do not sum to `1.0`
- invalid `ENGRAM_EMBEDDING_DIMENSION`
- missing provider package, such as using OpenAI without installing `engram[openai]`
- missing API key for cloud providers

