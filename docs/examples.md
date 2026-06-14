# Examples

The repository includes three example scripts. They are meant for local
development and API exploration, not production templates.

## Setup

```bash
pip install -e ".[dev,examples,sentence-transformers]"
docker compose up -d postgres

export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

For OpenAI-backed fact extraction and chat responses:

```bash
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_EMBEDDING_DIMENSION=1536
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...
```

## `examples/basic_usage.py`

Run:

```bash
python examples/basic_usage.py
```

This is the broad API walkthrough. It covers:

| Capability | APIs |
|------------|------|
| lifecycle | `connect()`, `close()`, async context manager |
| memory CRUD | `add()`, `add_batch()`, `get()`, `update()`, `reinforce()`, `forget()`, `purge()` |
| search and recall | `search()`, `deep_search()`, `recall_critical()`, `trace_recall()` |
| context blocks | `get_context_block()`, `get_memories()` |
| evidence retrieval | `search_evidence_set()`, `get_neighboring_context_block()`, `answer_from_evidence()` |
| graph | `relate()`, `traverse()`, `traverse_many()`, `render_graph_context()` |
| sessions | `session()` |
| task memory | `start_task()`, `record_event()`, `record_turn()`, `list_events()`, `create_checkpoint()`, `build_context()`, `process_memory_jobs()`, task status APIs |
| long input | `record_long_input()`, `build_long_input_context()` |
| LLM extraction | `add_conversation()` when an LLM provider is configured |
| health | `health_check()` |

Use this example when you want to see the API surface in one place.

## `examples/long_input_usage.py`

Run:

```bash
python examples/long_input_usage.py
```

This example records a large source input, chunks it, creates anchored memories,
and builds source-aware answer context.

Important APIs:

- `start_task()`
- `record_long_input()`
- `build_long_input_context()`
- `trace_recall()`
- `search_evidence_set()`
- `get_neighboring_context_block()`
- `answer_from_evidence()`
- `build_context()`

The pattern is useful for legal documents, specifications, research packets, and
large task briefs. For production exact-document answers, store external source
metadata such as page numbers or document IDs in the metadata you pass to
`record_long_input()`.

## `examples/chatbot.py`

Configure OpenAI embeddings and chat first:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_EMBEDDING_DIMENSION=1536
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...
```

Run:

```bash
python examples/chatbot.py
```

The chatbot is a real OpenAI-backed chat loop using Engram memory. It fails fast
without `ENGRAM_OPENAI_API_KEY`, builds Engram context before every model call,
records every turn, and processes memory jobs so later replies can recall
durable facts from the conversation.

By default the chatbot processes memory jobs inline after each reply. For lower
chat latency, defer extraction and run a separate worker:

```bash
export ENGRAM_CHATBOT_MEMORY_JOBS=deferred
```

For broad memory-evaluation questions, the chatbot also includes a bounded
recent-memory safety net plus smaller hard-constraint and query-specific
attention blocks in the prompt. Tune the safety net with:

```bash
export ENGRAM_CHATBOT_BROAD_MEMORY_LIMIT=60
export ENGRAM_CHATBOT_BROAD_MEMORY_CHARS=3600
```

| Capability | API |
|------------|-----|
| real chat response | `engram.llm.complete_full()` |
| prompt memory | `get_context_block()`, `trace_recall()`, `deep_search()`, `list_recent()`, hard-constraint and query-specific attention blocks, `build_context()` |
| persistent facts | `add()`, `forget()`, `purge()`, `list_recent()` |
| search and reinforcement | `search()`, `reinforce()` |
| durable conversation ledger | `record_turn()` |
| background derivation | `process_memory_jobs()`, `run_memory_worker()` |
| resumable task/session state | `session()`, `start_task()`, `get_task()`, `list_tasks()` |

Chat commands:

| Command | Behavior |
|---------|----------|
| `/remember <fact>` | store a durable fact immediately |
| `/memories` | show recent stored facts |
| `/search <query>` | run hybrid search and reinforce hits |
| `/trace <query>` | run `trace_recall()` |
| `/context <query>` | render the memory and task context used for prompting |
| `/task` | show the resumable task/session backing this chat |
| `/forget <memory_id>` | delete one memory |
| `/clear` | purge memories for the configured agent/user |
| `/help` | show command help |
| `/quit` | exit |

## How The Chatbot Builds Context

The chatbot combines:

1. system prompt
2. `get_context_block()` memory block
3. `trace_recall()` recall trace
4. `deep_search()` high-recall memory block
5. bounded `list_recent()` memory safety net
6. hard-constraint attention block for avoidances, cancellations, owners, and thresholds
7. query-specific attention block for food, scheduling, update, and launch-rule questions
8. `build_context()` task memory block
9. recent in-process sliding window
10. current user message

That mix keeps the prompt useful for short sessions, long sessions, interrupted
work, and resumed tasks.

## Minimal Pattern To Copy

```python
from engram import Engram


async def handle_message(task_id: str, message: str) -> str:
    async with Engram(memory_policy="default") as engram:
        context = await engram.build_context(task_id, query=message)
        response = await call_your_llm(context.text, message)
        await engram.record_turn(task_id, message, response)
        await engram.process_memory_jobs(limit=10)
        return response
```

For production, split this into two processes:

- API/chat process: build context, call the model, record turns
- worker process: run `run_memory_worker()` and handle memory derivation
