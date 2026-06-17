# Configuration

Engram is configured using `pydantic-settings`. By default, it reads environment variables prefixed with `ENGRAM_` and automatically loads from a `.env` file in the current working directory if one exists.

> [!NOTE]
> `EngramSettings` handles connections, models, and search tuning. Behavior policies (like `memory_policy`) are passed directly to the `Engram` client at instantiation rather than via environment variables.

---

## 1. Quickstarts

### Local / Offline Development
To run entirely offline using local embeddings (no API keys required):

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```
*Requires `pip install -e ".[sentence-transformers]"`*

### Cloud Providers (OpenAI)
To use OpenAI for both embeddings and the optional LLM capabilities (required for `deep_search` and automated intent parsing via `recall`):

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...
```
*Requires `pip install -e ".[openai]"`*

---

## 2. Environment Variables

### Database Settings
Engram requires PostgreSQL equipped with the `vector` and `pg_trgm` extensions.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DATABASE_URL` | `postgresql://localhost:5432/engram` | Connection string |
| `ENGRAM_MIN_POOL_SIZE` | `5` | Minimum active asyncpg connections |
| `ENGRAM_MAX_POOL_SIZE` | `20` | Maximum asyncpg connections |
| `ENGRAM_CONNECTION_TIMEOUT`| `30.0` | Connection timeout (seconds) |
| `ENGRAM_COMMAND_TIMEOUT` | `60.0` | SQL statement timeout (seconds) |

### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_EMBEDDING_PROVIDER`| `openai` | Valid: `openai`, `sentence-transformers`, `cohere`, `ollama`, `huggingface` |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small`| Provider-specific model ID |
| `ENGRAM_EMBEDDING_DIMENSION`| `None` | Optional explicit dimension. Usually auto-detected from provider. |
| `ENGRAM_EMBEDDING_BATCH_SIZE`| `100` | Vector batch size for ingestion |
| `ENGRAM_EMBEDDING_CACHE_SIZE`| `1000` | In-memory cache entries. `0` disables caching. |
| `ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE` | `false` | **Crucial:** Set to `true` if changing dimension against existing DB. |

> [!WARNING]
> Changing the embedding model against an already-populated database will raise an error on startup to prevent catastrophic data corruption. To override this and allow Engram to alter the pgvector column (which drops existing vectors), you must set `ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true`.

### LLM Settings
The LLM is technically optional for CRUD memory, but strictly required for high-level cognitive operations like `recall()`, `deep_search()`, and intelligent intent parsing.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_LLM_PROVIDER` | `None` | Valid: `openai`, `anthropic`, `ollama`, `groq`, `litellm` |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | Provider-specific model ID |
| `ENGRAM_OPENAI_API_KEY` | `None` | Required if provider is `openai` |
| `ENGRAM_ANTHROPIC_API_KEY` | `None` | Required if provider is `anthropic` |

### Search & Retrieval Settings
These weights dictate the ranking algorithm for `hybrid` search mode. They must sum to roughly `1.0`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_WEIGHT_SEMANTIC` | `0.40` | Importance of vector similarity |
| `ENGRAM_WEIGHT_KEYWORD` | `0.20` | Importance of exact keyword matches |
| `ENGRAM_WEIGHT_DECAY` | `0.25` | Importance of recency/frequency |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.15` | Importance of memory priority |
| `ENGRAM_DECAY_RATE` | `0.995` | Base decay per hour (0.0 to 1.0) |
| `ENGRAM_DEFAULT_SEARCH_LIMIT`| `10` | Default results returned |
| `ENGRAM_NEAR_DUPLICATE_THRESHOLD`| `0.95` | Cosine threshold to block near-duplicate memory ingestion |

### Reranking Settings
If you execute `search(..., rerank=True)`, Engram uses a local Cross-Encoder to re-score the fetched documents. This lazy-loads the model into memory.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace model ID |
| `ENGRAM_RERANKER_BACKEND`| `torch` | Valid: `torch`, `onnx`, `openvino` |

---

## 3. Programmatic Configuration

While `.env` files are standard, you can fully configure Engram programmatically using the `EngramSettings` object.

```python
from engram import Engram, EngramSettings

# 1. Define infrastructure and provider settings
settings = EngramSettings(
    database_url="postgresql://admin:secret@pg:5432/db",
    embedding_provider="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    llm_provider=None,  # Running in pure-storage mode
)

# 2. Instantiate Client (injecting behavioral policies)
async with Engram(
    settings=settings, 
    memory_policy="coding_agent"  # Determines how facts are categorized
) as engram:
    
    health = await engram.health_check()
    print(health["status"])
```

> [!TIP]
> **Behavior vs Infrastructure**: `EngramSettings` defines *how* Engram connects to the database and models (infrastructure). The `memory_policy` argument defines *what* Engram does with the memories it encounters (behavior).

---

## 4. Required Provider Packages

Because Engram supports a vast ecosystem of model providers, dependencies are split into optional `extras` to keep the footprint small.

| Backend Need | Installation Command |
|--------------|----------------------|
| OpenAI (LLM / Embed) | `pip install engram[openai]` |
| Anthropic (LLM) | `pip install engram[anthropic]` |
| Cohere (Embed) | `pip install engram[cohere]` |
| Local Sentence Transformers | `pip install engram[sentence-transformers]` |
| HTTP/Ollama/Groq | `pip install engram[http]` |
| Reranking Support | `pip install engram[rerank]` |
| **All Providers** | `pip install engram[all]` |
