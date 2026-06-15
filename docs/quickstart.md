# Quickstart

This walks you through a local Postgres setup and the main things Engram does:
store a fact, search for it, pin a critical one, and trace why recall did or
didn't include it. It should take about ten minutes.

## Prerequisites

- Python 3.10 or newer
- Docker and Docker Compose
- One embedding provider

If you don't have an embedding API key, use `sentence-transformers`. It runs
locally and needs nothing else.

## Install

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
pip install -e ".[dev,examples,sentence-transformers]"
```

If you'd rather use OpenAI for embeddings and the LLM features:

```bash
pip install -e ".[dev,examples,openai]"
```

## Start Postgres

```bash
docker compose up -d postgres
docker compose ps postgres
```

The compose defaults line up with `.env.example`:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
```

The first time you call `connect()`, Engram sets up everything it needs: the
`vector` and `pg_trgm` extensions, the tables and indexes, the text-search
config, and a vector column sized to your embedding model. You never run a
migration by hand for normal use.

## Configure embeddings

For the local option:

```bash
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

For OpenAI instead:

```bash
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_EMBEDDING_DIMENSION=1536
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...
```

## Store and search a fact

Save this as `quickstart_memory.py`:

```python
import asyncio

from engram import Engram


async def main() -> None:
    async with Engram(memory_policy="default") as engram:
        memory = await engram.add(
            "User prefers dark mode",
            "assistant",
            user_id="user_123",
            main_content="[USER]: I always switch apps to dark mode.\n[AI]: Noted.",
        )

        results = await engram.search(
            "interface preferences",
            "assistant",
            user_id="user_123",
            limit=5,
        )

        print(memory.memory_id)
        for result in results:
            print(f"{result.score:.3f}: {result.memory.content}")


asyncio.run(main())
```

Run it:

```bash
python quickstart_memory.py
```

The search query never mentions "dark mode," but the fact still comes back. That
is the vector half of hybrid search doing its job. `main_content` holds the raw
exchange for context and is not embedded; only the fact is.

## Pin and trace a critical fact

Some facts should never drop out of the prompt just because a query ranked other
things higher. A `MemoryPolicy` decides which ones. The `coding_agent` policy
treats repo constraints as critical and gives them a stable conflict slot, so a
later correction supersedes the old value instead of piling up next to it.

```python
import asyncio

from engram import Engram


async def main() -> None:
    async with Engram(memory_policy="coding_agent") as engram:
        await engram.add(
            "Repo constraint: never revert user changes without explicit approval",
            "codex",
            user_id="nafiz",
        )

        trace = await engram.trace_recall(
            "resume repository work",
            "codex",
            user_id="nafiz",
            expected_terms=["never revert"],
            max_tokens=800,
        )

        print(trace.context)
        print("missing:", trace.missing_expected_terms)
        print("kept:", trace.kept_memory_ids)


asyncio.run(main())
```

Here's what you should see:

- the memory is typed `constraint`
- its metadata carries `critical`, `critical_slot`, and `conflict_key`
- `trace_recall()` puts it ahead of the ordinary search-ranked memories, and
  `missing_expected_terms` comes back empty

## Store facts from a conversation

`add_conversation()` needs an LLM provider. It reads a user/assistant exchange,
pulls out the durable facts, checks them against what you already have, and only
writes the ones that add something.

```python
memories = await engram.add_conversation(
    user_message="I'm Sarah. I am allergic to shellfish.",
    assistant_response="I will avoid shellfish recommendations.",
    agent_id="assistant",
    user_id="sarah",
)

for memory in memories:
    print(memory.memory_type, memory.content, memory.metadata)
```

## Track resumable work

Task memory is for work that outlives one reply: the task record, the raw events,
checkpoints, and the background jobs that turn events into memories.

```python
task = await engram.start_task(
    "Make the project OSS publish ready",
    "codex",
    user_id="nafiz",
)

await engram.record_turn(
    task.task_run_id,
    user_message="Clean Ruff and update docs.",
    assistant_response="Ruff passes and I am updating the docs now.",
    tool_calls=[{"name": "ruff", "result": "All checks passed"}],
)

await engram.process_memory_jobs(limit=10)

context = await engram.build_context(
    task.task_run_id,
    query="what remains before publishing?",
    max_tokens=200000,
)

print(context.text)
```

`process_memory_jobs()` runs the queue inline, which is handy in a script. In
production, run `run_memory_worker()` in its own process so ingestion doesn't
block the request.

## Ingest a long document

Reach for long input when the exact source matters: contracts, specs, large
prompts, or a task packet you'll answer questions about later.

```python
contract_text = """
# Termination
Either party may terminate with 30 days written notice.

# Audit Logs
The vendor shall retain audit logs for 180 days.
"""

task = await engram.start_task(
    "Review vendor contract",
    "legal-agent",
    user_id="user_123",
)

report = await engram.record_long_input(
    task.task_run_id,
    text=contract_text,
    title="Vendor agreement",
    max_chunk_tokens=700,
)

context = await engram.build_long_input_context(
    task.task_run_id,
    query="What are the termination notice requirements?",
    expected_terms=["termination", "notice"],
    max_tokens=4000,
)

print(report.manifest)
print(context.text)
print(context.trace["missing_expected_terms"])
```

Each chunk keeps its character span and a content hash, so an answer can point
back to the exact text it came from.

## Run the examples

```bash
python examples/basic_usage.py
python examples/long_input_usage.py
python examples/chatbot.py
```

`examples/chatbot.py` is a real OpenAI-backed chatbot. It pulls Engram memory
before each model call, records the turn, and queues memory jobs by default. Set
`ENGRAM_CHATBOT_RECALL_MODE=deep` for broad high-recall retrieval, or `debug` to
see `trace_recall()` in the prompt and turn metadata. With
`ENGRAM_CHATBOT_RERANK=auto`, the deep and debug modes rerank candidates while
fast mode stays low-latency.

## Check health

```python
async with Engram() as engram:
    health = await engram.health_check()
    print(health["status"])
    print(health["components"])
```

## Clean up

Stop the containers but keep the data:

```bash
docker compose down
```

Drop the local database volume too:

```bash
docker compose down -v
```
