# Quickstart

This guide gets a local Engram database running and exercises the current memory
architecture.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- One embedding provider:
  - `sentence-transformers` for local/free development
  - OpenAI, Cohere, HuggingFace, or Ollama for other deployments
- Optional LLM provider for fact extraction, deep search, and job processing

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

## Start Postgres + pgvector

```bash
docker compose up -d
docker compose ps
```

The default compose stack creates:

```bash
ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
```

## Configure

Local embeddings:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

OpenAI embeddings and LLM:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-...
```

## Store And Trace Critical Memory

```python
import asyncio
from engram import Engram

async def main():
    async with Engram(memory_policy="coding_agent") as engram:
        await engram.add(
            "Repo constraint: never revert user changes without explicit approval",
            agent_id="codex",
            user_id="nafiz",
        )

        trace = await engram.trace_recall(
            "resume repository work",
            agent_id="codex",
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

- The repo constraint is typed as `constraint`.
- It is marked critical by policy metadata.
- `trace_recall()` includes it even if vector ranking would not.

## Store Conversation Facts

With an LLM provider configured:

```python
async with Engram(memory_policy="default") as engram:
    memories = await engram.add_conversation(
        user_message="I'm Sarah. I am allergic to shellfish.",
        assistant_response="I will remember that and avoid shellfish suggestions.",
        agent_id="assistant",
        user_id="sarah",
    )

    for memory in memories:
        print(memory.memory_type, memory.content, memory.metadata)
```

`add_conversation()` extracts facts, compares them with existing memories, stores
the raw exchange in `main_content`, and applies conflict metadata.

## Long-Running Task Memory

```python
async with Engram(memory_policy="coding_agent") as engram:
    task = await engram.start_task(
        "Make the project OSS publish ready",
        agent_id="codex",
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

Use `run_memory_worker()` instead of inline `process_memory_jobs()` when you
want a separate background worker.

## Long Input

Use this for large prompts, legal documents, specs, or multi-thousand-token
instructions.

```python
async with Engram(memory_policy="legal") as engram:
    task = await engram.start_task(
        "Review vendor contract",
        agent_id="legal-agent",
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

## Run The Examples

```bash
python examples/basic_usage.py
python examples/long_input_usage.py
python examples/chatbot.py
```

`examples/chatbot.py` demonstrates task memory, trace recall, graph context,
manual `/worker` processing, and the full Engram API.

## Verify Health

```python
async with Engram() as engram:
    health = await engram.health_check()
    print(health["status"])
    print(health["components"])
```

## Clean Up

```bash
docker compose down
```

To remove all local database data:

```bash
docker compose down -v
```

