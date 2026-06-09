# Engram

**Persistent memory infrastructure for long-running AI agents.**

Engram is an alpha-stage memory library for LLM applications. It stores typed,
searchable memory in PostgreSQL + pgvector, adds deterministic recall for
critical facts, preserves raw task/event history, and provides traceable context
assembly for long-running agents.

!!! warning "Alpha status"
    The public API and schema are still evolving. Use migrations carefully,
    back up production data, and read the production guide before deploying.

## What Changed

The current architecture includes:

- **Two-column memory**: embed concise facts in `fact`; preserve full context in
  `main_content` without embedding it.
- **Typed memory**: profile, project, task, preference, constraint, decision,
  tool result, semantic, episodic, and procedural memories.
- **Memory policies**: automatic typing, critical slots, and conflict keys via
  `MemoryPolicy`, with `default`, `legal`, and `coding_agent` presets.
- **Deterministic critical recall**: important facts are retrieved by metadata,
  not only by vector rank.
- **Conflict resolution**: corrected facts supersede older facts with the same
  conflict key.
- **Recall observability**: `trace_recall()` reports stored, ranked, kept,
  trimmed, missing, and superseded memories.
- **Long-running task memory**: durable task runs, append-only event ledger,
  checkpoints, and background memory jobs.
- **Long-input support**: source-anchored chunking for large prompts, legal docs,
  specs, and multi-day task context.

## Quick Start

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
docker compose up -d
pip install -e ".[dev,examples,sentence-transformers]"
```

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

```python
import asyncio
from engram import Engram

async def main():
    async with Engram(memory_policy="coding_agent") as engram:
        memory = await engram.add(
            "Repo constraint: never revert user changes without approval",
            agent_id="codex",
            user_id="nafiz",
        )

        trace = await engram.trace_recall(
            "continue the repository work",
            agent_id="codex",
            user_id="nafiz",
            expected_terms=["never revert"],
        )

        print(memory.memory_type)
        print(trace.context)
        print(trace.missing_expected_terms)

asyncio.run(main())
```

## Long-Running Task Example

```python
async with Engram(memory_policy="coding_agent") as engram:
    task = await engram.start_task(
        "Ship persistent memory for Codex",
        agent_id="codex",
        user_id="nafiz",
    )

    await engram.record_turn(
        task.task_run_id,
        user_message="Implement deterministic recall and conflict resolution.",
        assistant_response="I will add policy metadata, traces, and tests.",
        tool_calls=[{"name": "pytest", "result": "190 passed"}],
    )

    await engram.process_memory_jobs(limit=10)

    context = await engram.build_context(
        task.task_run_id,
        query="resume the work",
        max_tokens=200000,
    )
    print(context.text)
```

## Learn More

<div class="grid cards" markdown>

- :material-play-circle: **[Quickstart](quickstart.md)**

    Install, configure, and run your first memory flow.

- :material-memory: **[Core Concepts](concepts.md)**

    Understand memory types, policies, conflict resolution, search, and graph recall.

- :material-timeline-clock: **[Long-Running Memory](long-running-memory.md)**

    Deep dive into task memory, events, checkpoints, long input, and recall traces.

- :material-api: **[API Reference](api.md)**

    Public methods, models, and examples.

- :material-database-arrow-up: **[Migration Guide](migration.md)**

    Schema changes and upgrade order.

- :material-shield-check: **[Production Guide](production-guide.md)**

    Practical deployment, observability, and safety guidance.

</div>

## Examples

- `examples/basic_usage.py`: broad API walkthrough.
- `examples/chatbot.py`: persistent memory chatbot using task memory.
- `examples/long_input_usage.py`: source-anchored long-input ingestion and context.

