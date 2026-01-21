# Engram

> **AI Memory Layer for LLM Applications**

Engram is a drop-in memory library that gives your AI applications persistent, searchable memory using PostgreSQL. Provider-agnostic, production-ready, and inspired by ChatGPT and Claude's memory systems.

## Features

- 🧠 **Hybrid Search** - Combines semantic (vector) and keyword (BM25) search using RRF
- ⏱️ **Memory Decay** - Exponential decay prioritizes recent and frequently accessed memories
- 🔗 **Graph Traversal** - Multi-hop reasoning through typed memory relationships
- 📝 **Session Management** - Cross-session continuity with automatic expiration
- 🎯 **Provider-Agnostic** - Works with any embedding model (OpenAI, Anthropic, Cohere, local)
- 🚀 **Production-Ready** - ACID guarantees, soft deletes, connection pooling

## Quick Start

### Installation

```bash
pip install engram
```

### Setup Database

```bash
docker-compose up -d
```

### Basic Usage

```python
from engram import Engram

# Initialize with your embedding function
async def my_embed(text: str) -> list[float]:
    return await openai.embeddings.create(
        input=text, 
        model="text-embedding-3-small"
    ).data[0].embedding

memory = Engram(
    database_url="postgresql+asyncpg://user:pass@localhost/engram",
    embedding_fn=my_embed,
    embedding_dim=1536,
    agent_name="my-assistant"
)

# Add memory
await memory.add(
    "User prefers dark mode",
    user_id="user_123",
    metadata={"type": "preference"}
)

# Search memories
results = await memory.search(
    "user preferences",
    user_id="user_123",
    limit=5
)

# Session context
async with memory.session(user_id="user_123") as session:
    await session.add("User asked about Python")
    context = await session.get_context("What did they ask?", limit=10)
```

## Architecture

Engram uses a **converged cognitive architecture** - everything runs in PostgreSQL:

- **Vectors** (pgvector) - Semantic search
- **Full-text** (PostgreSQL) - Keyword search  
- **Graphs** (relations table) - Multi-hop reasoning
- **JSONB** - Flexible metadata

All with ACID guarantees in a single database.

## Documentation

- **[Implementation Plan](./IMPLEMENTATION_PLAN.md)** - Complete implementation guide
- **[Memory Theory](./MEMORY_THEORY.md)** - Deep dive into how each component works
- **[Examples](./examples/)** - Integration examples for OpenAI, Anthropic, Ollama

## Requirements

- Python 3.10+
- PostgreSQL 16+ with pgvector extension
- Your own embedding function (any provider)

## License

MIT

---

**Built with ❤️ for the AI community**

