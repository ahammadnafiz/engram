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
**Command:** `python examples/chatbot.py` *(defaults to local `all-MiniLM-L6-v2` embeddings + Google `gemini-3.1-flash-lite`; set `GEMINI_API_KEY`)*

This is the exact pipeline proven in `benchmark/`, run live. Each turn does three things:

1. **Ingest** — the turn is stored verbatim as a date-anchored episodic memory via `add_batch()`. On-device embeddings only: no LLM extraction at write time, no cost, no supersession.
2. **Retrieve** — 4 surfaces over everything stored before the turn: hybrid `search()` (vector + full-text, cross-encoder reranked), `recall(compose_answer=False)` for structured current/previous evidence, `get_lineage()` for superseded predecessors, and `traverse_many()` for graph relations.
3. **Generate** — one composer LLM call answers from the assembled evidence block.

> **Why not `add_conversation()` at ingest?** We evaluated the hybrid (`add_batch` + `add_conversation` per turn) and removed it. With the raw turns co-located in the same memory space, `add_conversation`'s extractor finds each fact already present verbatim in the floor row it came from, so the decision step judges it "semantically identical" and NOOPs it — the lineage layer barely fires while roughly doubling LLM cost per turn. The floor + composer answers temporal and overwrite questions correctly on its own. `add_conversation()` remains a first-class API; it just must be the **sole writer** to a memory space, never mixed with `add_batch()` (see its API note).

### How memory changes are tracked

Because ingest never supersedes, evolving facts (e.g. a budget revised from $5k to $7k) are stored as separate date-anchored turns. The composer reconstructs the current value and its history by reasoning over those dated rows at read time. Structured lineage chains (`get_lineage()` / `recall("...before...")`) only form when you drive them explicitly with `/remember` + `/revise`.

### Built-in Chat Commands

Inside the terminal, you can type special commands to interact directly with the memory layer:

| Command | Action |
|---------|--------|
| `/remember <fact>` | Store a durable fact immediately (`add()`). |
| `/revise <memory_id> <fact>` | Create a new active revision, superseding the old one. |
| `/lineage <memory_id>` | Show the current head and full revision history. |
| `/history [active\|limit\|memory_id]` | Show the memory add/update timeline. |
| `/memories` | List recent stored memories. |
| `/search <query>` | Hybrid search over stored memories. |
| `/recall <question>` | Ask memory: current / historical / event / lineage answer. |
| `/evidence <query>` | Show the 4-surface evidence block for a query. |
| `/forget <memory_id>` | Delete one memory. |
| `/clear` | Purge this chatbot user's memories. |

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
