<p align="center">
  <img src="assets/engram-banner.png" alt="Engram - AI Memory Layer" width="600">
</p>

<p align="center">
  <strong>Production-Ready Memory Infrastructure for AI Applications</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#examples">Examples</a>
</p>

---

Engram is a memory management library that provides persistent, searchable memory for AI agents using PostgreSQL and pgvector. It handles the complexity of hybrid search, memory decay, and graph relationships so you can focus on building your application.

## Features

- **Hybrid Search** — Combines vector similarity, keyword matching (BM25), time decay, and importance scoring using Reciprocal Rank Fusion
- **Two-Column Memory System** — Embed only facts for search, store full conversation context separately (cost-effective)
- **Memory Decay** — Exponential decay prioritizes recent and frequently accessed memories
- **Graph Traversal** — Multi-hop reasoning through typed memory relationships using recursive CTEs
- **Session Management** — Track conversation context with automatic TTL expiration
- **Pluggable Providers** — Support for OpenAI, Anthropic, Cohere, Ollama, Sentence Transformers, and custom providers
- **Production-Ready** — ACID guarantees, connection pooling, automatic vector dimension scaling

## Installation

```bash
pip install engram
```

With specific providers:

```bash
pip install engram[openai]                # OpenAI embeddings and LLM
pip install engram[anthropic]             # Anthropic Claude
pip install engram[sentence-transformers] # Local embeddings (free)
pip install engram[all]                   # All providers
```

## Quick Start

### 1. Start the Database

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
docker compose up -d
```

### 2. Configure Environment

```bash
# .env
ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
ENGRAM_EMBEDDING_PROVIDER=openai
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
ENGRAM_OPENAI_API_KEY=sk-...
```

### 3. Use Engram

```python
import asyncio
from engram import Engram

async def main():
    async with Engram() as engram:
        # Store a memory
        memory = await engram.add(
            content="User prefers dark mode",
            agent_id="assistant",
            user_id="user_123",
        )
        
        # Store with conversation context (two-column system)
        memory = await engram.add(
            content="User is learning Python",           # Fact (embedded)
            main_content="[USER]: I'm learning Python\n[AI]: Great choice!",  # Context (not embedded)
            agent_id="assistant",
            user_id="user_123",
        )
        
        # Search memories
        results = await engram.search(
            query="user preferences",
            agent_id="assistant",
            user_id="user_123",
        )
        
        for r in results:
            print(f"[{r.score:.2f}] {r.memory.content}")

asyncio.run(main())
```

## Architecture

Engram uses a converged architecture where all operations run in PostgreSQL:

```
┌─────────────────────────────────────────────────────────────────────┐
│                            ENGRAM                                   │
├─────────────────────────────────────────────────────────────────────┤
│   Embedding Service          │          LLM Service                 │
│   (OpenAI, Local, etc.)      │          (OpenAI, Anthropic, etc.)   │
├──────────────────────────────┴──────────────────────────────────────┤
│                       PostgreSQL + pgvector                         │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│   │  Vectors   │  │  Full-text │  │   Graph    │  │   JSONB    │    │
│   │   (HNSW)   │  │   (GIN)    │  │ Relations  │  │  Metadata  │    │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Two-Column Memory System

| Column | Embedded | Purpose |
|--------|----------|---------|
| `fact` | Yes | Concise user facts for semantic search |
| `main_content` | No | Full conversation context (cost-effective storage) |

### Hybrid Search

Search combines multiple signals using Reciprocal Rank Fusion:

```
score = 0.40 × semantic_similarity
      + 0.20 × keyword_score
      + 0.25 × time_decay
      + 0.15 × importance
```

## Provider Support

### Embedding Providers

| Provider | Installation | Example Models |
|----------|--------------|----------------|
| `openai` | `pip install openai` | `text-embedding-3-small`, `text-embedding-3-large` |
| `sentence-transformers` | `pip install sentence-transformers` | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` |
| `cohere` | `pip install cohere` | `embed-english-v3.0`, `embed-multilingual-v3.0` |
| `ollama` | Ollama server | `nomic-embed-text`, `mxbai-embed-large` |
| `huggingface` | `pip install httpx` | Any model via Inference API |

### LLM Providers

| Provider | Installation | Example Models |
|----------|--------------|----------------|
| `openai` | `pip install openai` | `gpt-4o-mini`, `gpt-4o` |
| `anthropic` | `pip install anthropic` | `claude-3-haiku`, `claude-3-sonnet` |
| `ollama` | Ollama server | `llama3.2`, `mistral` |
| `groq` | `pip install httpx` | `llama-3.1-8b-instant`, `mixtral-8x7b` |
| `litellm` | `pip install litellm` | 100+ models via unified API |

### Custom Providers

```python
from engram import embedding_registry, EmbeddingProvider

@embedding_registry.register("custom")
class CustomEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "default"):
        self._model = model
        self._dimension = 768
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model(self) -> str:
        return self._model
    
    async def embed(self, text: str) -> list[float]:
        # Implementation
        return [0.0] * self._dimension
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]
```

## Examples

### Memory Chatbot

```bash
python examples/chatbot.py
```

### Programmatic Usage

```python
from engram import Engram

async def example():
    async with Engram() as engram:
        # Add memories
        m1 = await engram.add(
            content="User's name is Alice",
            agent_id="bot",
            main_content="[USER]: I'm Alice\n[AI]: Nice to meet you!",
        )
        m2 = await engram.add(
            content="Alice works in finance",
            agent_id="bot",
        )
        
        # Create graph relations
        await engram.relate(m1.memory_id, m2.memory_id, "related_to", weight=0.8)
        
        # Traverse the graph
        related = await engram.traverse(
            start_memory_id=m1.memory_id,
            max_depth=2,
            direction="outbound",
        )
        
        # Reinforce important memories
        await engram.reinforce(m1.memory_id, importance_boost=0.2)
        
        # Search with hybrid ranking
        results = await engram.search("Alice", agent_id="bot", limit=5)
        for r in results:
            print(f"Fact: {r.memory.fact}")
            print(f"Context: {r.memory.main_content}")
```

## Configuration

All settings are configured via environment variables with the `ENGRAM_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DATABASE_URL` | `postgresql://localhost:5432/engram` | PostgreSQL connection string |
| `ENGRAM_EMBEDDING_PROVIDER` | `openai` | Embedding provider |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `ENGRAM_LLM_PROVIDER` | — | LLM provider (optional) |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | LLM model |
| `ENGRAM_WEIGHT_SEMANTIC` | `0.40` | Semantic search weight |
| `ENGRAM_WEIGHT_KEYWORD` | `0.20` | Keyword search weight |
| `ENGRAM_WEIGHT_DECAY` | `0.25` | Time decay weight |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.15` | Importance weight |
| `ENGRAM_DECAY_RATE` | `0.995` | Decay rate per hour |

## Docker Commands

```bash
# Start database
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Reset (delete all data)
docker compose down -v
```

## Documentation

- [Quickstart Guide](./docs/quickstart.md) — Get started in 5 minutes
- [Core Concepts](./docs/concepts.md) — Memory model, hybrid search, decay
- [API Reference](./docs/api.md) — Complete API documentation
- [Architecture](./docs/architecture.md) — System design and schema
- [Configuration](./docs/configuration.md) — All configuration options

## Requirements

- Python 3.10+
- PostgreSQL 16+ with pgvector
- Docker and Docker Compose (for database)

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for the AI community</sub>
</p>
