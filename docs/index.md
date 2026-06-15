<section class="engram-hero" markdown="1">

# Engram

Persistent memory for AI agents that have to remember across sessions.

Engram keeps an agent's memory in PostgreSQL: the facts it has learned, the raw
history of what happened, large documents split into traceable pieces, and the
links between them. What you get back is recall you can inspect. You can rank
it, see why a memory surfaced, and pick a task back up days later.

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[API reference](api-reference.md){ .md-button }
[Architecture](architecture.md){ .md-button }

</section>

<div class="engram-signal-grid" markdown="1">

<div markdown="1">
**Search that shows its work**

Retrieval blends vector similarity, keyword matching, recency, and importance.
You see each score, not one number you have to take on faith.
</div>

<div markdown="1">
**Memory that survives restarts**

Task runs, event logs, checkpoints, and background jobs hold the state of work
that outlives a single conversation.
</div>

<div markdown="1">
**Recall you can debug**

When an agent forgets something it should have known, `trace_recall()` tells you
whether the fact was stored, ranked, trimmed by the token budget, or quietly
superseded.
</div>

</div>

!!! warning "Alpha"
    Engram is at `0.3.0a2`. It has unit and integration tests behind it, but the
    public API and the database schema can still change before 1.0. Back up your
    data before you run a migration.

## Guides

<div class="grid cards" markdown="1">

- :material-rocket-launch-outline: **[Quickstart](quickstart.md)**

    Install Engram, point it at Postgres, and store and recall your first memory.

- :material-lightbulb-on-outline: **[Core concepts](concepts.md)**

    How facts, memory types, conflict handling, search, and the graph fit together.

- :material-history: **[Task memory](task-memory.md)**

    Runs, events, checkpoints, and background jobs for work that spans sessions.

- :material-api: **[API reference](api-reference.md)**

    Every public method, with signatures and examples taken straight from the code.

- :material-tune: **[Configuration](configuration.md)**

    Environment variables, provider extras, search tuning, and the safety flags.

- :material-shield-check-outline: **[Production guide](production-guide.md)**

    Deployment shape, privacy boundaries, observability, and the failure modes to plan for.

</div>

## Install and run

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

This stores one repo constraint, then asks Engram to build the recall block for
the next turn. `trace_recall()` hands back the context it would inject plus the
bookkeeping: which memories it kept, which it dropped, and whether your expected
term ("never revert") actually made it in.

## How it works

Engram keeps two kinds of memory side by side.

| Plane | Tables | Built for |
|-------|--------|-----------|
| Fact memory | `agent_memory`, `memory_relations` | search, type filters, conflict resolution, graph recall |
| Task memory | `agent_task_runs`, `agent_events`, `agent_checkpoints`, `memory_jobs` | resuming work, audit history, background processing |

`Engram.connect()` creates the schema, or migrates an existing one, and sizes
the vector column to match your embedding model. If a config change would shrink
or wipe stored embeddings, it stops and tells you instead of destroying data.

## Common tasks

| You want to | Use |
|-------------|-----|
| Store durable facts | `add()`, `add_batch()`, `add_conversation()` |
| Retrieve memories | `search()`, `deep_search()`, `get_context_block()` |
| Figure out why recall missed | `trace_recall()` |
| Keep critical facts from getting buried | `MemoryPolicy`, `recall_critical()` |
| Handle corrections and contradictions | `conflict_key`, active/superseded status |
| Resume long-running work | tasks, events, checkpoints, memory jobs |
| Work with large documents | `record_long_input()`, `build_long_input_context()` |
| Follow related context | graph relations and traversal |

## What's in the box

| File | What it shows |
|------|---------------|
| `examples/basic_usage.py` | a tour of most of the API |
| `examples/chatbot.py` | a real OpenAI-backed chatbot with fast, deep, and debug recall modes |
| `examples/long_input_usage.py` | ingesting a long document and answering from anchored chunks |

See [Examples](examples.md) for what each script covers.
