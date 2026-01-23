# Engram

**AI Memory Library for LLM Applications**

Engram is a production-ready memory management library for AI agents using 
PostgreSQL + pgvector for converged storage with hybrid search.

<div class="grid cards" markdown>

- :material-memory: **Hybrid Search**
  
    Combines vector similarity, keyword matching, time decay, and importance scoring using RRF fusion.

- :material-currency-usd: **Two-Column System**
  
    Embed only facts (cheap), store full conversation context separately. Cost-effective memory.

- :material-graph: **Graph Relations**
  
    Multi-hop traversal using recursive CTEs for associative reasoning and knowledge graphs.

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
        # Add a memory (two-column system)
        memory = await engram.add(
            content="User prefers dark mode",  # Fact (embedded for search)
            agent_id="my_agent",
            main_content="[USER]: I like dark themes\n[AI]: Noted!",  # Context (not embedded)
        )
        
        # Search memories (hybrid: vector + keyword + decay + importance)
        results = await engram.search(
            query="user interface preferences",
            agent_id="my_agent",
        )
        
        for r in results:
            print(f"[{r.score:.2f}] {r.memory.content}")
            if r.memory.main_content:
                print(f"    Context: {r.memory.main_content}")

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

- :material-api: **[API Reference](api.md)**
  
    Complete API documentation.

- :material-database-arrow-up: **[Migration Guide](migration.md)**
  
    Database migration instructions for upgrades.

- :material-console: **[Command Reference](commands.md)**
  
    Docker, database, and chatbot commands.

</div>
