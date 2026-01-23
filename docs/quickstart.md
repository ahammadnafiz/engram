# Quickstart Guide

Get Engram running in under 5 minutes.

## Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for database)
- **Embedding provider** — Choose one:
    - OpenAI API key (cloud, paid)
    - Sentence Transformers (local, free)
    - Cohere, Ollama, HuggingFace, etc.

## Step 1: Install Engram

=== "pip"

    ```bash
    pip install engram
    ```

=== "With OpenAI"

    ```bash
    pip install engram[openai]
    ```

=== "With Local Embeddings (Free)"

    ```bash
    pip install engram[sentence-transformers]
    ```

=== "From Source"

    ```bash
    git clone https://github.com/ahammadnafiz/engram.git
    cd engram
    pip install -e ".[all]"
    ```

## Step 2: Start the Database

```bash
# Start PostgreSQL with pgvector
docker compose up -d
```

Verify it's running:

```bash
docker compose ps
# Should show: engram-postgres ... Up (healthy)
```

## Step 3: Configure Your Provider

Create a `.env` file in your project root:

=== "OpenAI (Cloud)"

    ```bash
    # .env
    ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
    ENGRAM_EMBEDDING_PROVIDER=openai
    ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
    ENGRAM_OPENAI_API_KEY=sk-your-key-here
    ```

=== "Sentence Transformers (Local, Free)"

    ```bash
    # .env
    ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
    ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
    ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
    # No API key needed!
    ```

=== "Ollama (Local)"

    ```bash
    # .env
    ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
    ENGRAM_EMBEDDING_PROVIDER=ollama
    ENGRAM_EMBEDDING_MODEL=nomic-embed-text
    ENGRAM_OLLAMA_BASE_URL=http://localhost:11434
    ```

=== "Cohere (Cloud)"

    ```bash
    # .env
    ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
    ENGRAM_EMBEDDING_PROVIDER=cohere
    ENGRAM_EMBEDDING_MODEL=embed-english-v3.0
    ENGRAM_COHERE_API_KEY=your-cohere-key
    ```

## Step 4: Use Engram

### Basic Usage

```python
import asyncio
from engram import Engram

async def main():
    # Connect to Engram (reads config from .env)
    async with Engram() as engram:
        
        # Add a memory (fact only)
        memory = await engram.add(
            content="User prefers dark mode and Python",  # Fact (embedded)
            agent_id="my-assistant",
            user_id="user_123",
            metadata={"type": "preference"}
        )
        print(f"Created memory: {memory.memory_id}")
        
        # Add with conversation context (two-column system)
        memory = await engram.add(
            content="User is learning machine learning",  # Fact (embedded)
            agent_id="my-assistant",
            user_id="user_123",
            main_content="[USER]: I'm taking an ML course\n[AI]: That's great!",  # Context (NOT embedded)
        )
        
        # Search memories (hybrid: vector + keyword + decay + importance)
        results = await engram.search(
            query="What does the user prefer?",
            agent_id="my-assistant",
            user_id="user_123",
            limit=5
        )
        
        for r in results:
            print(f"  [{r.score:.2f}] {r.memory.content}")
            if r.memory.main_content:
                print(f"      Context: {r.memory.main_content[:50]}...")

asyncio.run(main())
```

### Reinforce Useful Memories

When a memory helps generate a good response, reinforce it:

```python
async with Engram() as engram:
    # Search for relevant memories
    results = await engram.search(
        query="user preferences",
        agent_id="my-assistant",
    )
    
    # Use the memory... then reinforce it
    if results:
        await engram.reinforce(
            results[0].memory.memory_id,
            importance_boost=0.1  # Boost importance by 0.1
        )
```

### With Session Context

Sessions group memories within a conversation:

```python
async with Engram() as engram:
    # Start a session
    async with engram.session(
        agent_id="my-assistant",
        user_id="user_123"
    ) as session:
        # Add memories with session context
        await engram.add(
            content="User asked about Python async",
            agent_id="my-assistant",
            user_id="user_123",
            session_id=session.session_id,
        )
        
        await engram.add(
            content="User is building a web scraper",
            agent_id="my-assistant", 
            user_id="user_123",
            session_id=session.session_id,
        )
    
    # Search across all sessions
    results = await engram.search(
        query="What are they working on?",
        agent_id="my-assistant",
        user_id="user_123",
        limit=10
    )
    
    print("Relevant context:")
    for r in results:
        print(f"  - {r.memory.content}")
```

### Create Memory Relationships (Graph)

Build a knowledge graph:

```python
async with Engram() as engram:
    # Add related memories
    python_mem = await engram.add(
        content="Python is a programming language",
        agent_id="tutor"
    )
    
    async_mem = await engram.add(
        content="asyncio enables async programming in Python",
        agent_id="tutor"
    )
    
    # Create a relationship
    await engram.relate(
        source_memory_id=async_mem.memory_id,
        target_memory_id=python_mem.memory_id,
        relation_type="part_of",
        weight=0.9
    )
    
    # Traverse the graph
    related = await engram.traverse(
        start_memory_id=python_mem.memory_id,
        max_depth=2,
        min_weight=0.5,
        direction="any",  # outbound, inbound, or any
    )
    
    print(f"Found {len(related)} related memories:")
    for r in related:
        print(f"  [depth={r.depth}] {r.content}")
```

## Step 5: Verify It's Working

Check your setup:

```python
async with Engram() as engram:
    health = await engram.health_check()
    print(f"Status: {health['status']}")
    print(f"Database: {health['database']}")
    print(f"Embedding dimension: {health.get('embedding_dimension', 'N/A')}")
```

Or from the command line:

```bash
docker compose ps
docker compose logs engram-postgres
```

## Configuration Options

All settings via environment variables (prefix: `ENGRAM_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DATABASE_URL` | `postgresql://localhost:5432/engram` | PostgreSQL connection |
| `ENGRAM_EMBEDDING_PROVIDER` | `openai` | Provider: `openai`, `sentence-transformers`, `cohere`, `ollama`, `huggingface` |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | Model name (provider-specific) |
| `ENGRAM_LLM_PROVIDER` | `None` | Optional LLM for fact extraction |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `ENGRAM_OPENAI_API_KEY` | - | OpenAI API key |
| `ENGRAM_ANTHROPIC_API_KEY` | - | Anthropic API key |
| `ENGRAM_COHERE_API_KEY` | - | Cohere API key |
| `ENGRAM_WEIGHT_SEMANTIC` | `0.40` | Semantic search weight |
| `ENGRAM_WEIGHT_KEYWORD` | `0.20` | Keyword search weight |
| `ENGRAM_WEIGHT_DECAY` | `0.25` | Time decay weight |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.15` | Importance weight |
| `ENGRAM_DECAY_RATE` | `0.995` | Memory decay rate per hour |

See [Configuration](./configuration.md) for all options.

## Docker Commands Reference

```bash
# Start database
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop containers
docker compose down

# Reset database (delete all data)
docker compose down -v

# Open PostgreSQL shell
docker exec -it engram-postgres psql -U engram -d engram
```

## Next Steps

- 📖 [Core Concepts](./concepts.md) — Understand memories, search, and graphs
- 🔧 [Configuration](./configuration.md) — All configuration options
- 🤖 [Chatbot Example](../examples/chatbot.py) — Full chatbot with memory
- 📚 [API Reference](./api/index.md) — Complete API documentation

## Troubleshooting

### Database connection failed

1. Check if Docker is running: `docker ps`
2. Check container health: `docker compose ps`
3. View logs: `docker compose logs engram-postgres`
4. Verify `.env` has correct `ENGRAM_DATABASE_URL`

### Embeddings not working

1. **OpenAI**: Verify API key is set: `echo $ENGRAM_OPENAI_API_KEY`
2. **Sentence Transformers**: Install: `pip install sentence-transformers`
3. **Ollama**: Ensure Ollama server is running: `ollama serve`

### Dimension mismatch error

If you change embedding providers, the vector dimensions may differ. Reset the database:

```bash
docker compose down -v
docker compose up -d
```

### Import errors

Make sure you installed the correct extras:

```bash
pip install engram[openai]              # For OpenAI
pip install engram[sentence-transformers]  # For local embeddings
pip install engram[all]                 # Everything
```

### Need help?

- [GitHub Issues](https://github.com/ahammadnafiz/engram/issues)
- [Discussions](https://github.com/ahammadnafiz/engram/discussions)
