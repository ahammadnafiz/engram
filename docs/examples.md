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
| memory CRUD | `add()`, `add_batch()`, `get()`, `update()`, `revise()`, `get_current()`, `get_lineage()`, `explain_memory()`, `get_history()`, `reinforce()`, `forget()`, `purge()` |
| search and recall | `search()`, `deep_search()`, `recall_critical()`, `trace_recall()` |
| context blocks | `get_context_block()`, `get_memories()` |
| evidence reading | composed from `deep_search()` + `get_memories()` + `engram.llm` |
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
- `deep_search()`
- `get_context_block()`
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
records every turn, and queues memory jobs so later replies can recall durable
facts from the conversation.

By default the chatbot uses operator recall with inline memory processing:

```bash
export ENGRAM_CHATBOT_RECALL_MODE=operator
export ENGRAM_CHATBOT_MEMORY_JOBS=inline
export ENGRAM_CHATBOT_RERANK=auto
```

`operator` routes the turn through `engram.recall(..., compose_answer=False)` to
get intent and source-backed evidence, then uses the regular OpenAI chat path to
write the final answer from recall evidence, active memory context, and the
memory history timeline. After the reply, inline memory processing stores new
facts so they are available to the next turn.

For lower latency, switch to `fast` mode:

```bash
export ENGRAM_CHATBOT_RECALL_MODE=fast
export ENGRAM_CHATBOT_MEMORY_JOBS=deferred
```

`fast` does one embedding-backed context lookup plus deterministic critical
memory recall before the OpenAI chat call. With deferred jobs, run
`run_memory_worker()` or call `process_memory_jobs()` from a separate process to
ingest new facts.

`ENGRAM_CHATBOT_RERANK=auto` keeps reranking off in `fast` mode and enables it
in `deep` and `debug` mode. Set `ENGRAM_CHATBOT_RERANK=true` to force reranking
for every mode, or `false` to disable it everywhere.

For high-recall evaluation questions, enable the broader prompt context:

```bash
export ENGRAM_CHATBOT_RECALL_MODE=deep
```

For recall debugging, include `trace_recall()` metadata in the prompt and stored
turn metadata:

```bash
export ENGRAM_CHATBOT_RECALL_MODE=debug
```

`deep` and `debug` include a bounded recent-memory safety net plus smaller
hard-constraint and query-specific attention blocks in the prompt. They also
rerank retrieved candidates when `ENGRAM_CHATBOT_RERANK=auto`. Tune the safety
net with:

```bash
export ENGRAM_CHATBOT_BROAD_MEMORY_LIMIT=60
export ENGRAM_CHATBOT_BROAD_MEMORY_CHARS=3600
```

| Capability | API |
|------------|-----|
| real chat response | `engram.llm.complete_full()` |
| operator recall evidence | `recall(..., compose_answer=False)` |
| fast prompt memory | `recall_critical()`, `get_context_block()` |
| deep/debug prompt memory | `deep_search()`, `list_recent()`, hard-constraint and query-specific attention blocks, `build_context()` |
| recall debugging | `trace_recall()` |
| persistent facts | `add()`, `revise()`, `get_history()`, `forget()`, `purge()`, `list_recent()` |
| search and reinforcement | `search()`, `reinforce()` |
| durable conversation ledger | `record_turn()` |
| background derivation | `process_memory_jobs()`, `run_memory_worker()` |
| resumable task/session state | `session()`, `start_task()`, `get_task()`, `list_tasks()` |

Chat commands:

| Command | Behavior |
|---------|----------|
| `/remember <fact>` | store a durable fact immediately |
| `/revise <memory_id> <fact>` | create a new active revision |
| `/lineage <memory_id>` | show current head and revision history |
| `/history [active\|limit\|memory_id]` | show memory add/update timeline |
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

In `fast` mode the chatbot combines:

1. system prompt
2. `recall_critical()` deterministic critical memories
3. `get_context_block()` memory block
4. recent in-process sliding window
5. current user message

In `deep` mode it also adds `deep_search()`, bounded `list_recent()` safety-net
memory, hard-constraint attention, query-specific attention, and `build_context()`
task memory. In `debug` mode it adds `trace_recall()` on top of `deep`.

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
