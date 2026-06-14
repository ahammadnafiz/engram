# Engram

Engram is an async Python memory layer for LLM applications and long-running
agents. It stores searchable facts, raw task history, source-anchored long
inputs, and graph relations in PostgreSQL + pgvector.

!!! warning "Alpha status"
    Engram is `0.3.0a1`. The codebase has unit and integration coverage, but
    public APIs and the schema may still change before a stable release. Back
    up data before migrations.

## What Engram Gives You

| Need | Engram surface |
|------|----------------|
| Store durable facts | `add()`, `add_batch()`, `add_conversation()` |
| Retrieve memories | `search()`, `deep_search()`, `get_context_block()` |
| Debug missed recall | `trace_recall()` |
| Pin critical facts | `MemoryPolicy`, `recall_critical()` |
| Handle corrections | `conflict_key`, active/superseded metadata |
| Resume long work | tasks, events, checkpoints, memory jobs |
| Work with large documents | `record_long_input()`, `build_long_input_context()` |
| Expand related context | graph relations and traversal |
| Build evidence sets | `search_evidence_set()`, neighboring context, `answer_from_evidence()` |

## Quick Start

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
pip install -e ".[dev,examples,sentence-transformers]"
docker compose up -d postgres

export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

```python
import asyncio

from engram import Engram


async def main() -> None:
    async with Engram(memory_policy="coding_agent") as engram:
        memory = await engram.add(
            "Repo constraint: never revert user changes without approval",
            "codex",
            user_id="nafiz",
        )

        trace = await engram.trace_recall(
            "continue the repository work",
            "codex",
            user_id="nafiz",
            expected_terms=["never revert"],
        )

        print(memory.memory_type)
        print(trace.context)


asyncio.run(main())
```

## Architecture In One Page

Engram has two connected planes:

| Plane | Tables | Optimized for |
|-------|--------|---------------|
| Fact memory | `agent_memory`, `memory_relations` | search, type filters, conflict resolution, graph recall |
| Task memory | `agent_task_runs`, `agent_events`, `agent_checkpoints`, `memory_jobs` | resumability, audit history, background derivation |

`Engram.connect()` creates or migrates the schema for normal library use. It
also aligns the vector column with the configured embedding dimension, with a
safety guard that blocks destructive dimension changes unless explicitly
enabled.

## Where To Go Next

<div class="grid cards" markdown>

- :material-play-circle: **[Quickstart](quickstart.md)**

    Install, configure, and run the first memory flow.

- :material-memory: **[Core Concepts](concepts.md)**

    Understand fact memory, policies, conflict slots, search, graph recall, and task memory.

- :material-timeline-clock: **[Task Memory](task-memory.md)**

    Work with task runs, events, checkpoints, background jobs, and long inputs.

- :material-api: **[API Reference](api-reference.md)**

    Current public methods, signatures, models, and examples from the codebase.

- :material-cog: **[Configuration](configuration.md)**

    Environment variables, provider extras, search tuning, reranking, and safety flags.

- :material-shield-check: **[Production Guide](production-guide.md)**

    Deployment shape, privacy boundaries, observability, and failure modes.

</div>

## Included Examples

| File | Purpose |
|------|---------|
| `examples/basic_usage.py` | broad API walkthrough |
| `examples/chatbot.py` | real OpenAI-backed chatbot with Engram recall, turn recording, memory jobs, and cleanup commands |
| `examples/long_input_usage.py` | source-anchored long-input ingestion and context |

Read [Examples](examples.md) for what each script exercises.
