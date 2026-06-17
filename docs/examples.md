# Examples

The Engram repository ships with three core example scripts designed for local development, API exploration, and demonstrating best practices. 

> [!WARNING]  
> These scripts are intended for learning and experimentation. Do not run them directly as production services without adding your own auth, rate-limiting, and error-handling wrappers.

---

## Environment Setup

Before running the examples, you need a running PostgreSQL instance equipped with the `pgvector` and `pg_trgm` extensions. The easiest way is using the provided Docker compose file:

```bash
docker compose up -d postgres
```

### Option A: Local / Offline Mode
Run the examples entirely offline using `sentence-transformers` (no API keys):
```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2

pip install -e ".[dev,examples,sentence-transformers]"
```

### Option B: Cloud Provider (OpenAI)
To enable LLM-backed features (like the Chatbot and Intelligent Recall):
```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_EMBEDDING_DIMENSION=1536
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...

pip install -e ".[dev,examples,openai]"
```

---

## 1. `examples/basic_usage.py`

**Goal:** A comprehensive, broad walkthrough of the entire Engram API surface.  
**Command:** `python examples/basic_usage.py`

This script executes a series of isolated demonstrations. Use this as your primary reference when you need to see exactly how a specific API is called.

| Domain | Demonstrated APIs |
|--------|-------------------|
| **Lifecycle** | `connect()`, `close()`, Async Context Manager (`async with`) |
| **Fact CRUD** | `add()`, `add_batch()`, `update()`, `revise()`, `get_current()`, `get_lineage()`, `forget()`, `purge()` |
| **Search** | `search()`, `deep_search()`, `recall_critical()`, `trace_recall()` |
| **Context** | `get_context_block()`, `get_memories()` |
| **Graph** | `relate()`, `traverse()`, `traverse_many()`, `render_graph_context()` |
| **Task State** | `start_task()`, `record_event()`, `record_turn()`, `create_checkpoint()`, `build_context()` |
| **Background** | `process_memory_jobs()` |

---

## 2. `examples/long_input_usage.py`

**Goal:** Demonstrates the pattern for processing massive documents securely.  
**Command:** `python examples/long_input_usage.py`

This script ingests a large block of text, chunks it using heading and token boundaries, creates source-anchored `Artifact` events, extracts searchable facts, and then answers questions by tracing back to the exact source chunks.

> [!TIP]
> This pattern is highly recommended for **Legal, Medical, and Financial domains**, where you must be able to prove exactly which paragraph of a 100-page document an LLM used to make a decision.

**Key APIs Showcased:**
- `record_long_input()`
- `build_long_input_context()`

---

## 3. `examples/chatbot.py`

**Goal:** A fully functional, terminal-based AI assistant backed by Engram memory.  
**Command:** `python examples/chatbot.py` *(Requires OpenAI setup)*

This is a real chat loop. It builds prompt context before every LLM call, records every turn to the event ledger, and queues background memory jobs so that the agent dynamically "remembers" facts you told it earlier in the conversation.

### Chatbot Recall Modes

You can control how the chatbot pulls memory by exporting `ENGRAM_CHATBOT_RECALL_MODE`.

| Mode | Behavior | Use Case |
|------|----------|----------|
| `operator` *(Default)* | Routes the turn through `engram.recall()`. The LLM classifies user intent (current vs history vs event) and maps temporal filters before retrieving evidence. | Production default. Balances intelligence and latency. |
| `fast` | Single embedding-backed lookup + deterministic critical pin lookup. No pre-LLM routing. | Ultra-low latency requirements. |
| `deep` | High-recall evaluation. Injects bounded recent-memory safety nets, hard constraints, and `deep_search()` RRF fusion. | Complex reasoning or open-ended aggregation questions. |
| `debug` | Same as `deep`, but includes `trace_recall()` metadata directly in the terminal output. | Debugging why a specific memory was (or wasn't) injected. |

### Memory Processing Modes

| Mode | Behavior |
|------|----------|
| `inline` *(Default)* | Blocks the chat loop while background facts are derived. Slower, but facts are instantly available on the next turn. |
| `deferred` | `fast` mode only. Queues the memory job, but requires you to run `run_memory_worker()` in a separate terminal process to actually extract the facts. |

### Built-in Chat Commands

Inside the terminal, you can type special commands to interact directly with the memory layer:

| Command | Action |
|---------|--------|
| `/remember <fact>` | Force-store a semantic fact bypassing background derivation. |
| `/revise <id> <fact>` | Force-update a memory, creating a new lineage revision. |
| `/history [limit]` | View the chronological timeline of your memory updates. |
| `/memories` | List all recent stored facts for this session. |
| `/trace <query>` | Run the observability trace to see what would be recalled. |
| `/context <query>` | Render the exact prompt block the LLM would see. |
| `/forget <id>` | Soft-delete a memory. |

---

## Minimal Integration Pattern

If you are building your own application and want to bypass the heavy examples, this is the absolute minimum code required to implement a robust memory loop:

```python
from engram import Engram

async def handle_message(task_id: str, message: str) -> str:
    # 1. Connect
    async with Engram(memory_policy="default") as engram:
        
        # 2. Build context from existing memory/checkpoints
        context = await engram.build_context(task_id, query=message)
        
        # 3. Call your standard LLM pipeline
        response = await call_your_llm(system_prompt=context.text, user_input=message)
        
        # 4. Commit the turn to the immutable ledger
        await engram.record_turn(task_id, message, response)
        
        # 5. Kick off derivation (extract facts from the turn)
        await engram.process_memory_jobs(limit=10)
        
        return response
```

> [!IMPORTANT]
> **Scaling for Production:** In a real-world web application, you should **remove** Step 5 (`process_memory_jobs()`) from the user's request path. Instead, run Engram's `run_memory_worker()` in a completely separate background process (like Celery or a continuous Docker service) to process the queued memory jobs asynchronously without blocking the user's API response.
