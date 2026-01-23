# Engram

> **AI Memory Layer for LLM Applications**

Engram is a production-ready memory library that gives your AI applications persistent, searchable memory using PostgreSQL. **Fully pluggable provider system** — use any embedding model or LLM provider out of the box.

## ✨ Features

- 🧠 **Hybrid Search** — Combines semantic (vector) + keyword (BM25) search using RRF fusion
- 💰 **Two-Column System** — Embed only facts, store full context separately (cost-effective)
- ⏱️ **Memory Decay** — Exponential decay prioritizes recent and frequently accessed memories
- 🔗 **Graph Traversal** — Multi-hop reasoning through typed memory relationships
- 📝 **Session Management** — Cross-session continuity with automatic expiration
- 🔌 **Pluggable Providers** — Any embedding or LLM provider (OpenAI, Anthropic, Cohere, Ollama, local)
- 🚀 **Production-Ready** — ACID guarantees, connection pooling, auto-scaling vector dimensions

## 🚀 Quick Start

### 1. Start PostgreSQL

```bash
# Clone and start
git clone https://github.com/ahammadnafiz/engram.git
cd engram
docker compose up -d
```

### 2. Install Engram

```bash
pip install engram

# Or with specific providers
pip install engram[openai]           # OpenAI embeddings + LLM
pip install engram[anthropic]        # Anthropic Claude
pip install engram[sentence-transformers]  # Local embeddings (free!)
pip install engram[all]              # Everything
```

### 3. Configure Provider

```bash
# .env file
ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram

# Choose your embedding provider
ENGRAM_EMBEDDING_PROVIDER=openai              # or: sentence-transformers, cohere, ollama
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small # or: all-MiniLM-L6-v2, embed-english-v3.0

# Optional: LLM provider for fact extraction
ENGRAM_LLM_PROVIDER=openai                    # or: anthropic, ollama, groq
ENGRAM_LLM_MODEL=gpt-4o-mini                  # or: claude-3-haiku, llama3.2

# API Keys (as needed)
ENGRAM_OPENAI_API_KEY=sk-...
ENGRAM_ANTHROPIC_API_KEY=sk-ant-...
ENGRAM_COHERE_API_KEY=...
```

### 4. Use It

```python
import asyncio
from engram import Engram

async def main():
    async with Engram() as engram:
        # Add a memory (fact only)
        memory = await engram.add(
            content="User prefers dark mode",  # Fact (embedded)
            agent_id="my-assistant",
            user_id="user_123",
        )
        
        # Add with conversation context (two-column system)
        memory = await engram.add(
            content="User is learning Python",  # Fact (embedded)
            agent_id="my-assistant",
            user_id="user_123",
            main_content="[USER]: I'm learning Python\n[AI]: Great choice!",  # Context (NOT embedded)
        )
        
        # Search memories (hybrid: vector + keyword + decay + importance)
        results = await engram.search(
            query="user preferences",
            agent_id="my-assistant",
            user_id="user_123",
        )
        
        for r in results:
            print(f"[{r.score:.2f}] {r.memory.content}")
            if r.memory.main_content:
                print(f"    Context: {r.memory.main_content[:50]}...")

asyncio.run(main())
```

## 🔌 Provider System

Engram's pluggable architecture lets you use **any provider** without code changes:

### Embedding Providers

| Provider | Install | Model Examples |
|----------|---------|----------------|
| `openai` | `pip install openai` | `text-embedding-3-small`, `text-embedding-3-large` |
| `sentence-transformers` | `pip install sentence-transformers` | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` |
| `cohere` | `pip install cohere` | `embed-english-v3.0`, `embed-multilingual-v3.0` |
| `ollama` | Ollama server | `nomic-embed-text`, `mxbai-embed-large` |
| `huggingface` | `pip install httpx` | Any HF model via Inference API |

### LLM Providers

| Provider | Install | Model Examples |
|----------|---------|----------------|
| `openai` | `pip install openai` | `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` |
| `anthropic` | `pip install anthropic` | `claude-3-haiku`, `claude-3-sonnet`, `claude-3-opus` |
| `ollama` | Ollama server | `llama3.2`, `mistral`, `codellama` |
| `groq` | `pip install httpx` | `llama-3.1-8b-instant`, `mixtral-8x7b` |
| `litellm` | `pip install litellm` | 100+ models via unified API |

### Register Custom Providers

```python
from engram import embedding_registry, EmbeddingProvider

@embedding_registry.register("my-provider")
class MyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "my-model"):
        self._model = model
        self._dimension = 768
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model(self) -> str:
        return self._model
    
    async def embed(self, text: str) -> list[float]:
        # Your implementation
        return [0.0] * self._dimension
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

# Now use it
# ENGRAM_EMBEDDING_PROVIDER=my-provider
```

## 🏗️ Architecture

Engram uses a **converged cognitive architecture** — everything runs in PostgreSQL:

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENGRAM                                  │
├─────────────────────────────────────────────────────────────────┤
│  Embedding Provider     │     LLM Provider                      │
│  (OpenAI, Local, etc)   │     (OpenAI, Anthropic, etc)          │
├─────────────────────────┴───────────────────────────────────────┤
│                      PostgreSQL + pgvector                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Vectors  │  │ Full-text│  │  Graph   │  │  JSONB   │         │
│  │ (HNSW)   │  │  (GIN)   │  │Relations │  │ Metadata │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Two-Column Memory System

| Column | Embedded? | Purpose |
|--------|-----------|---------|
| `fact` | ✅ Yes | Concise user facts for semantic search |
| `main_content` | ❌ No | Full conversation context (cost-effective) |

### Hybrid Search Formula

```
score = 0.40 × semantic_similarity (on fact)
      + 0.20 × keyword_rrf_score (on fact)
      + 0.25 × time_decay
      + 0.15 × importance
```

## 📖 Examples

### Personal Chatbot with Memory

```bash
# Run the example chatbot
python examples/chatbot.py

# Use different providers via CLI
python examples/chatbot.py --embedding sentence-transformers --llm ollama
python examples/chatbot.py --embedding cohere --llm anthropic
```

### Programmatic Usage

```python
from engram import Engram, EmbeddingService, LLMService

async def advanced_example():
    async with Engram() as engram:
        # Add memories with two-column system
        m1 = await engram.add(
            content="User's name is Alice",  # Fact (embedded)
            agent_id="bot",
            main_content="[USER]: I'm Alice\n[AI]: Nice to meet you!",  # Context
        )
        m2 = await engram.add(
            content="Alice likes Python",
            agent_id="bot",
            main_content="[USER]: I love Python\n[AI]: Great language!",
        )
        
        # Create relations (graph)
        await engram.relate(m1.memory_id, m2.memory_id, "related_to", weight=0.8)
        
        # Traverse the graph
        related = await engram.traverse(
            start_memory_id=m1.memory_id,
            max_depth=2,
            direction="outbound",
        )
        
        # Reinforce useful memories
        await engram.reinforce(m1.memory_id, importance_boost=0.2)
        
        # Search returns both fact and main_content
        results = await engram.search("Alice", agent_id="bot", limit=5)
        for r in results:
            print(f"Fact: {r.memory.fact}")
            print(f"Context: {r.memory.main_content}")
```

### Direct Provider Usage

```python
from engram import EmbeddingService, LLMService

# Create embedding service with any provider
embeddings = EmbeddingService.from_provider(
    "sentence-transformers",
    model="all-MiniLM-L6-v2",
)

vector = await embeddings.embed("Hello, world!")
print(f"Dimension: {embeddings.dimension}")  # 384

# Create LLM service
llm = LLMService.from_provider(
    "openai",
    api_key="sk-...",
    model="gpt-4o-mini",
)

response = await llm.complete("What is 2 + 2?")
facts = await llm.extract_facts("I'm a software engineer at Google", "")
```

## ⚙️ Configuration

All settings via environment variables (prefix: `ENGRAM_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DATABASE_URL` | `postgresql://localhost:5432/engram` | PostgreSQL connection |
| `ENGRAM_EMBEDDING_PROVIDER` | `openai` | Embedding provider name |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `ENGRAM_LLM_PROVIDER` | `None` | LLM provider (optional) |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | LLM model |
| `ENGRAM_WEIGHT_SEMANTIC` | `0.40` | Semantic search weight |
| `ENGRAM_WEIGHT_KEYWORD` | `0.20` | Keyword search weight |
| `ENGRAM_WEIGHT_DECAY` | `0.25` | Time decay weight |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.15` | Importance weight |
| `ENGRAM_DECAY_RATE` | `0.995` | Decay rate per hour |

## 🐳 Docker Commands

```bash
# Start PostgreSQL
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Reset (delete all data)
docker compose down -v
```

## 📚 Documentation

- **[Quickstart Guide](./docs/quickstart.md)** — Get started in 5 minutes
- **[Concepts](./docs/concepts.md)** — Understand memories, search, and graphs
- **[Configuration](./docs/configuration.md)** — All configuration options
- **[Examples](./examples/)** — Chatbot, demos, and integrations

## Requirements

- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 16+ with pgvector (provided by Docker)

## License

MIT

---

**Built with ❤️ for the AI community**
