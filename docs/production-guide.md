# Production Guide

Engram is currently alpha-stage software. This guide describes how to run it
responsibly in serious applications while the API and schema continue to evolve.

## Deployment Checklist

- Use PostgreSQL 14+ with `pgvector`.
- Back up the database before upgrades.
- Run migrations in order.
- Scope every operation by `agent_id` and, for multi-tenant apps, `user_id`.
- Choose a memory policy appropriate to the domain.
- Run a memory worker if using task memory.
- Monitor failed `memory_jobs`.
- Inspect `trace_recall()` for missed-retrieval incidents.
- Use source chunks for legal, compliance, financial, or exact-document answers.

## Recommended Runtime Shape

```text
API / Agent process
  - calls build_context or trace_recall
  - calls LLM
  - records turns/events

Memory worker process
  - runs process_memory_jobs or run_memory_worker
  - derives facts/checkpoints

PostgreSQL
  - pgvector
  - backups
  - monitoring
```

## Basic Application Loop

```python
async def handle_turn(engram, task_id, user_message):
    context = await engram.build_context(
        task_id,
        query=user_message,
        max_tokens=120000,
        include_graph=True,
    )

    response = await call_your_llm(context.text, user_message)

    await engram.record_turn(
        task_id,
        user_message=user_message,
        assistant_response=response,
        enqueue_processing=True,
    )

    return response
```

Worker:

```python
async with Engram(memory_policy="coding_agent") as engram:
    await engram.run_memory_worker(batch_size=20, interval_seconds=1.0)
```

For small apps, you can call `process_memory_jobs(limit=10)` inline after a turn.

## Critical Facts

Do not rely on vector search alone for:

- allergies
- identity
- user preferences
- repo constraints
- task requirements
- legal citation rules
- decisions and corrections
- tool results that affect safety or rollout decisions

Use policy-backed memory and `trace_recall()`:

```python
trace = await engram.trace_recall(
    query=user_message,
    agent_id=agent_id,
    user_id=user_id,
    expected_terms=["allergy", "rollback owner"],
    max_tokens=1500,
)

if trace.missing_expected_terms:
    logger.warning("Memory context missing expected terms: %s", trace.model_dump())
```

## Long Prompts And Documents

For prompts above a few thousand tokens, store them as long input:

```python
await engram.record_long_input(task_id, text=document, title="MSA v4")
context = await engram.build_long_input_context(
    task_id,
    query="termination notice",
    expected_terms=["termination", "notice"],
)
```

Production rules for legal/exact-document apps:

- Require source chunk IDs in answers.
- Prefer source chunks over distilled facts.
- Fail closed when expected terms are missing.
- Store document IDs, page numbers, OCR coordinates, or external citation data
  in chunk metadata.

## Database Operations

Use connection pooling:

```bash
ENGRAM_MIN_POOL_SIZE=10
ENGRAM_MAX_POOL_SIZE=50
```

For high-traffic deployments, put PgBouncer in front of PostgreSQL and tune the
application pool lower.

Backups:

```bash
pg_dump "$ENGRAM_DATABASE_URL" > engram_backup.sql
```

Vacuum/analyze tables with high churn:

```sql
VACUUM ANALYZE agent_memory;
VACUUM ANALYZE agent_events;
VACUUM ANALYZE memory_jobs;
```

## Security And Privacy

- Treat `agent_events.content`, `main_content`, and memory metadata as sensitive.
- Use `forget()` and `purge()` for memory deletion.
- Use `redact_event()` for raw event redaction.
- Do not send sensitive data to cloud embedding/LLM providers unless your policy
  allows it.
- Enforce authorization outside Engram before passing `agent_id` and `user_id`.

## Observability

Recommended logs/metrics:

| Signal | Why |
|--------|-----|
| search latency | prompt assembly performance |
| embedding latency/errors | provider health |
| LLM extraction errors | memory freshness |
| `memory_jobs` pending count | worker backlog |
| `memory_jobs` failed count | derivation failures |
| trace missing terms | recall quality incidents |
| superseded count | correction churn |

Example job backlog query:

```sql
SELECT status, COUNT(*)
FROM memory_jobs
GROUP BY status;
```

Failed jobs:

```sql
SELECT job_id, attempts, error, updated_at
FROM memory_jobs
WHERE status = 'failed'
ORDER BY updated_at DESC
LIMIT 20;
```

## FastAPI Sketch

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from engram import Engram

engram: Engram | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engram
    engram = Engram(memory_policy="default")
    await engram.connect()
    try:
        yield
    finally:
        await engram.close()

app = FastAPI(lifespan=lifespan)

@app.post("/tasks/{task_id}/turn")
async def turn(task_id: str, body: dict):
    assert engram is not None
    context = await engram.build_context(task_id, query=body["message"])
    response = await call_your_llm(context.text, body["message"])
    await engram.record_turn(task_id, body["message"], response)
    return {"response": response}
```

## Failure Modes

| Failure | Mitigation |
|---------|------------|
| Embedding provider outage | fail request or queue retry at app layer |
| LLM extraction failure | raw event remains; job records failure |
| Missed critical fact | inspect `trace_recall()` and policy slot metadata |
| Old fact recalled | check `conflict_key`, `status`, `superseded_by` |
| Large prompt drifts | use `record_long_input()` and source chunks |
| User data leak risk | always filter by `user_id`, enforce app auth |

## Release Readiness

Before publishing or deploying:

```bash
ruff check src tests examples
ruff format --check src tests examples
pytest tests/unit -q
python -m build
python -m twine check dist/*
```

