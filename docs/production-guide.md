# Production Guide

Engram is alpha software. This guide describes how to run it responsibly while
the API and schema continue to evolve.

## Deployment Checklist

- Use PostgreSQL with `vector` and `pg_trgm` extensions. The local stack uses
  `pgvector/pgvector:pg16`.
- Back up the database before upgrades.
- Run migrations in order for existing databases.
- Scope every operation by `agent_id`, and by `user_id` for multi-user apps.
- Choose a memory policy for the domain.
- Run a memory worker if you use task memory.
- Monitor pending and failed `memory_jobs`.
- Inspect `trace_recall()` for missed-retrieval incidents.
- Use source chunks for legal, compliance, financial, or exact-document answers.
- Treat `agent_events.content`, `main_content`, and metadata as sensitive.

## Runtime Shape

```text
API / agent process
  - builds context with trace_recall(), get_context_block(), or build_context()
  - calls the application LLM
  - records turns and events

Memory worker process
  - runs process_memory_jobs() or run_memory_worker()
  - derives facts and checkpoints

PostgreSQL
  - pgvector + pg_trgm
  - backups
  - monitoring
```

## Basic Turn Loop

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
from engram import Engram

async with Engram(memory_policy="coding_agent") as engram:
    await engram.run_memory_worker(batch_size=20, interval_seconds=1.0)
```

Small applications can call `process_memory_jobs(limit=10)` inline after a turn,
but this makes the request wait on memory derivation.

## Critical Facts

Do not rely on vector search alone for:

- allergies and health facts
- identity and profile facts
- user preferences
- repository constraints
- task requirements
- legal citation rules
- decisions and corrections
- tool results that affect safety or rollout decisions

Use policy-backed memory and inspect `trace_recall()`.

```python
trace = await engram.trace_recall(
    query=user_message,
    agent_id=agent_id,
    user_id=user_id,
    expected_terms=["allergy", "rollback owner"],
    max_tokens=1500,
)

if trace.missing_expected_terms:
    logger.warning("memory_context_missing_terms=%s", trace.model_dump())
```

## Long Prompts And Documents

Use `record_long_input()` for prompts above a few thousand tokens or for source
documents where exact quotes matter.

```python
await engram.record_long_input(
    task_id,
    text=document,
    title="MSA v4",
    metadata={"document_id": "msa-v4"},
)

context = await engram.build_long_input_context(
    task_id,
    query="termination notice",
    expected_terms=["termination", "notice"],
)
```

Production rules for exact-document apps:

- require source chunk IDs in answers
- prefer source chunks over distilled facts
- fail closed when expected terms are missing
- store document IDs, page numbers, OCR coordinates, or external citation data
  in metadata when the source parser knows them

## Database Operations

Connection pool settings:

```bash
export ENGRAM_MIN_POOL_SIZE=10
export ENGRAM_MAX_POOL_SIZE=50
```

For high-traffic deployments, put PgBouncer in front of PostgreSQL and tune the
application pool lower.

Back up before migrations and provider-dimension changes:

```bash
pg_dump "$ENGRAM_DATABASE_URL" > engram_backup.sql
```

Vacuum tables with high churn:

```sql
VACUUM ANALYZE agent_memory;
VACUUM ANALYZE agent_events;
VACUUM ANALYZE memory_jobs;
```

## Embedding Dimension Changes

Engram auto-detects the embedding dimension during `connect()` and aligns the
`agent_memory.embedding` column. If existing embeddings would be cleared,
Engram raises unless:

```bash
export ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true
```

Use that flag only with a re-embedding plan. Safer options are:

- keep the same embedding provider and model
- use a fresh database for tests or experiments
- migrate data and rebuild embeddings deliberately

## Security And Privacy

- Enforce authorization before passing `agent_id` and `user_id` to Engram.
- Treat raw events, `main_content`, and metadata as sensitive.
- Use `forget()` and `purge()` for fact memory deletion.
- Use `redact_event()` for raw event redaction.
- Delete or supersede derived memories linked to redacted source events when
  strict privacy deletion is required.
- Do not send sensitive data to cloud embedding or LLM providers unless your
  product policy allows it.

## Observability

Recommended metrics:

| Signal | Why |
|--------|-----|
| search latency | prompt assembly performance |
| embedding latency and errors | provider health |
| LLM extraction errors | memory freshness |
| `memory_jobs` pending count | worker backlog |
| `memory_jobs` failed count | derivation failures |
| trace missing terms | recall quality incidents |
| superseded count | correction churn |
| database pool usage | saturation and timeout risk |

Job backlog:

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
| Embedding provider outage | fail the request or queue retry at the app layer |
| LLM extraction failure | raw event remains; job records failure |
| Missed critical fact | inspect `trace_recall()` and policy slot metadata |
| Old fact recalled | check `conflict_key`, `status`, and `superseded_by` |
| Large prompt drifts | use `record_long_input()` and source chunks |
| User data leak risk | always filter by `user_id` and enforce app auth |
| Test targets production data | use `ENGRAM_TEST_DATABASE_URL` or an isolated database |

## Release Readiness

Before publishing or deploying:

```bash
ruff check src tests examples
ruff format --check src tests examples
pytest tests/unit -q
pytest tests/integration -q --run-integration
python -m build
python -m twine check dist/*
mkdocs build --strict
```
