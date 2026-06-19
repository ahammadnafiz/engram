<section class="engram-hero" markdown="1">

# Engram

**Persistent memory for AI agents that have to remember across sessions.**

Engram keeps an agent's memory in PostgreSQL: the facts it has learned, the raw
history of what happened, large documents split into traceable pieces, and the
links between them. What you get back is a cognitive `recall()` operator that routes intents, retrieves exact citations, and traces its own logic.

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[Architecture](architecture.md){ .md-button }
[API Reference](api-reference.md){ .md-button }

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

> [!WARNING]  
> **Beta**: Engram is at `0.3.0b1`. It is rigorously tested, but the public API and the database schema are subject to change before 1.0. Always back up your data before you run a migration.

## Documentation Guides

<div class="grid cards" markdown="1">

- :material-rocket-launch-outline: **[Quickstart](quickstart.md)**

    Install Engram, boot Postgres, and build an end-to-end memory loop in 10 minutes.

- :material-lightbulb-on-outline: **[Core Concepts](concepts.md)**

    How facts, conflict resolution, semantic search, and the intelligent `recall` operator fit together.

- :material-history: **[Task Memory](task-memory.md)**

    Resumability, the immutable event ledger, checkpoints, and exact-citation chunking.

- :material-api: **[API Reference](api-reference.md)**

    The complete public API, including type signatures and code examples.

- :material-tune: **[Configuration](configuration.md)**

    Environment variables, `EngramSettings`, search weight tuning, and provider extras.

- :material-shield-check-outline: **[Production Guide](production-guide.md)**

    Process isolation, privacy boundaries, observability, and failure modes.

- :material-chart-bar: **[Benchmarks](benchmarks.md)**

    79.6% on BEAM 1M (ICLR 2026), 89.8% on LongMemEval-S (ICLR 2025), 85.7% on LoCoMo-10 (ACL 2024). All three runs use `add_batch()` (no LLM at ingest) and are reproducible with the scripts in `benchmark/`.

</div>

## Install & Run

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
pip install -e ".[dev,examples,sentence-transformers]"

# Start the database
docker compose up -d postgres

# Configure for local embeddings
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

```python
import asyncio
from engram import Engram

async def main() -> None:
    # 1. Connect and automatically apply schema migrations
    async with Engram(memory_policy="coding_agent") as engram:
        
        # 2. Store a durable, critical fact
        memory = await engram.add(
            "Repo constraint: never revert user changes without approval",
            agent_id="codex",
            user_id="nafiz",
        )

        # 3. Recall a source-backed answer, routed by the question's intent
        answer = await engram.recall(
            "what are my repository constraints?",
            agent_id="codex",
            user_id="nafiz",
        )

        print(memory.memory_type)   # -> "constraint"
        print(answer.answer_text)   # -> grounded answer built from stored memory

if __name__ == "__main__":
    asyncio.run(main())
```

This snippet stores a strict repo constraint, then asks Engram to answer a question about it. `recall()` classifies the question's intent, retrieves the matching memories, and composes a source-backed answer in `answer.answer_text`. It requires a configured LLM (set `ENGRAM_LLM_PROVIDER`); for LLM-free retrieval use `search()` instead.

## How It Works

Engram keeps two kinds of memory side by side.

| Plane | Tables | Primary Purpose |
|-------|--------|-----------------|
| **Fact Memory** | `agent_memory`, `memory_relations` | Vector search, type filters, conflict resolution, and graph traversal. |
| **Task Memory** | `agent_task_runs`, `agent_events`, `agent_checkpoints` | Resuming workflows, exact audit history, and background semantic extraction. |

When `Engram.connect()` is called, it automatically creates the database schema, ensures the `pgvector` and `pg_trgm` extensions exist, and sizes the vector columns to match your chosen embedding model.

## Common Operations

| Objective | Recommended API |
|-----------|-----------------|
| Store new facts | `add()`, `add_batch()`, `add_conversation()` |
| Intelligently fetch context | `recall()` |
| Query the raw event ledger | `search_events()` |
| Analyze why recall failed | `trace_recall()` |
| Extract facts asynchronously | `run_memory_worker()`, `process_memory_jobs()` |
| Ingest & cite a 50-page PDF | `record_long_input()`, `build_long_input_context()` |

## Benchmark results

Engram is evaluated on three standard long-term memory benchmarks. All runs use on-device embeddings (`all-MiniLM-L6-v2`, free, no API cost at ingest) and `add_batch()` — raw episodic turns stored verbatim, with all reasoning deferred to query time via `search()` + `recall()` + `get_lineage()`. Same model is used for both composer and judge (`claude-sonnet-4-6`), which is a known leniency bias worth disclosing. These are floor numbers: `add_conversation()` (full LLM extraction at ingest) is expected to score higher.

![Engram benchmark results](assets/engram-benchmark.svg)

| Benchmark | Questions | Accuracy | Composer |
|---|---|---|---|
| [LongMemEval-S](benchmarks.md#longmemeval-s-898) (ICLR 2025) | 500 | **89.8%** | claude-sonnet-4-6 |
| [LoCoMo-10](benchmarks.md#locomo-10-857) (ACL 2024) | 1,540 | **85.7%** | claude-sonnet-4-6 |
| [BEAM 1M](benchmarks.md#beam-1m-796) (ICLR 2026) | 700 | **79.6%** | claude-sonnet-4-6 |

All three benchmark scripts are in `benchmark/` and can be run against your own database. See [Benchmarks](benchmarks.md) for full per-type breakdowns, honest caveats, ablation table, and reproduce commands.

## Included Examples

Engram ships with working reference implementations in the `examples/` directory.

| File | What it demonstrates |
|------|----------------------|
| `examples/basic_usage.py` | A comprehensive tour of almost the entire API surface. |
| `examples/chatbot.py` | A real OpenAI-backed terminal chatbot with background event extraction. |
| `examples/long_input_usage.py` | Securely ingesting a massive document and answering questions from anchored source chunks. |
