# Quickstart

This guide starts a local PostgreSQL + pgvector database and runs the core
memory flows against the current API.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- One embedding provider

For a local setup with no embedding API key, use `sentence-transformers`.

## Install

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
pip install -e ".[dev,examples,sentence-transformers]"
```

For OpenAI-backed embeddings and LLM features:

```bash
pip install -e ".[dev,examples,openai]"
```

## Start Postgres

```bash
docker compose up -d postgres
docker compose ps postgres
```

The compose defaults match `.env.example`:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
```

Engram initializes the schema on `connect()`, including `vector`, `pg_trgm`,
tables, indexes, migrations, text-search config, and vector dimension alignment.

## Configure Local Embeddings

```bash
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

OpenAI alternative:

```bash
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_EMBEDDING_DIMENSION=1536
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...
```

## Store And Search A Fact

Create `quickstart_memory.py`:

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

## Trace Critical Memory

Policies can type and pin critical facts. The `coding_agent` policy marks repo
constraints as critical and gives them deterministic conflict slots.

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

Expected behavior:

- the memory is typed as `constraint`
- metadata includes `critical`, `critical_slot`, and `conflict_key`
- `trace_recall()` includes it before ordinary search-ranked memories

## Store Conversation Facts

`add_conversation()` needs an LLM provider. It extracts facts from a user and
assistant exchange, compares them with existing memories, and stores only useful
updates.

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

## Use Task Memory

Task memory stores resumable agent work: the task record, raw events,
checkpoints, and background jobs.

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

For production, run `run_memory_worker()` in a separate worker process instead
of processing jobs inline.

## Use Long Input

Use long input for legal documents, specs, large prompts, or task packets where
the exact source matters.

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

## Run Included Examples

```bash
python examples/basic_usage.py
python examples/long_input_usage.py
python examples/chatbot.py
```

`examples/chatbot.py` is a real OpenAI-backed chatbot. It retrieves Engram
memory before each model call, records the turn, processes memory jobs, and
offers small operational commands for search, trace, context, task state, and
manual memory cleanup.

## Verify Health

```python
async with Engram() as engram:
    health = await engram.health_check()
    print(health["status"])
    print(health["components"])
```

## Clean Up

Stop containers without deleting data:

```bash
docker compose down
```

Delete the local database volume:

```bash
docker compose down -v
```
