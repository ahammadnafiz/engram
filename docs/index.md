# Engram

**AI Memory Library for LLM Applications**

Engram is a production-ready memory management library for AI agents using 
PostgreSQL + pgvector for converged storage with hybrid search.

<div class="grid cards" markdown>

- :material-memory: **Hybrid Search**
  
    Combines vector similarity, keyword matching, time decay, and importance scoring using RRF fusion.

- :material-graph: **Graph Relations**
  
    Multi-hop traversal using recursive CTEs for associative reasoning and knowledge graphs.

- :material-chat: **Session Management**
  
    Track conversation context with async context managers and automatic TTL expiration.

- :material-rocket-launch: **Production Ready**
  
    Async-first, fully typed, connection pooling, and comprehensive error handling.

</div>

## Quick Start

```bash
# 1. Clone and start database
git clone https://github.com/ahammadnafiz/engram.git
cd engram
./scripts/docker-setup.sh

# 2. Install
pip install engram

# 3. Set your API key
export OPENAI_API_KEY=sk-your-key-here
```

```python
import asyncio
from engram import Engram

async def main():
    async with Engram() as engram:
        # Add a memory
        memory = await engram.add(
            content="User prefers dark mode in applications",
            agent_id="my_agent",
            importance=0.8,
        )
        
        # Search memories
        results = await engram.search(
            query="user interface preferences",
            agent_id="my_agent",
        )
        
        for result in results:
            print(f"{result.score:.2f}: {result.memory.content}")

asyncio.run(main())
```

## Installation

=== "Basic"

    ```bash
    pip install engram
    ```

=== "With OpenAI"

    ```bash
    pip install engram[openai]
    ```

=== "With Local Embeddings"

    ```bash
    pip install engram[sentence-transformers]
    ```

=== "All Features"

    ```bash
    pip install engram[all]
    ```

## Learn More

<div class="grid cards" markdown>

- :material-play-circle: **[Quickstart](quickstart.md)**
  
    Get Engram running in 5 minutes.

- :material-brain: **[Core Concepts](concepts.md)**
  
    Understand how hybrid search, decay, and graphs work.

- :material-cog: **[Configuration](configuration.md)**
  
    All configuration options explained.

- :material-api: **[API Reference](api/client.md)**
  
    Complete API documentation.

</div>
