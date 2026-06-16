# Migrations

This guide covers database schema changes in the current alpha line.

!!! warning
    Back up the database before running migrations. Engram is alpha software,
    and schema evolution is still active.

## Fresh Database

For normal library use, `Engram.connect()` initializes the schema:

```python
from engram import Engram

async with Engram() as engram:
    health = await engram.health_check()
    print(health["status"])
```

For manual setup:

```bash
docker compose up -d postgres
docker compose exec -T postgres psql -U engram -d engram \
  < src/engram/sql/schema.sql
```

`schema.sql` creates required extensions, all tables, indexes, generated
tsvector columns, and default vector shape before `connect()` aligns the vector
dimension with the configured embedding provider.

## Existing Database

Run migrations in numeric order:

| Migration | Purpose |
|-----------|---------|
| `001_add_fact_columns.sql` | adds `fact`, `main_content`, generated tsvector columns, and related indexes |
| `002_add_session_summary.sql` | adds rolling summaries to `agent_sessions` |
| `003_add_memory_type.sql` | adds `memory_type` and a type index |
| `004_add_task_memory.sql` | adds task runs, event ledger, checkpoints, and memory jobs |
| `005_widen_memory_type_constraint.sql` | allows all current policy memory types |
| `006_add_memory_lineage.sql` | adds first-class memory revisions and lineage metadata |
| `007_add_event_search.sql` | adds nullable event embeddings for hybrid event recall without startup backfill |

## Back Up

```bash
pg_dump "$ENGRAM_DATABASE_URL" > backup_before_engram_migration.sql
```

Docker:

```bash
docker compose exec -T postgres pg_dump -U engram -d engram \
  > backup_before_engram_migration.sql
```

## Run Migrations

With `psql`:

```bash
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/001_add_fact_columns.sql
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/002_add_session_summary.sql
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/003_add_memory_type.sql
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/004_add_task_memory.sql
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/005_widen_memory_type_constraint.sql
```

With Docker:

```bash
for file in src/engram/sql/migrations/*.sql; do
  docker compose exec -T postgres psql -U engram -d engram < "$file"
done
```

## Verify

Columns:

```sql
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'agent_memory'
  AND column_name IN ('fact', 'main_content', 'memory_type');
```

Task tables:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_name IN (
  'agent_task_runs',
  'agent_events',
  'agent_checkpoints',
  'memory_jobs'
);
```

Expected core tables:

- `agents`
- `users`
- `agent_memory`
- `memory_relations`
- `agent_sessions`
- `agent_task_runs`
- `agent_events`
- `agent_checkpoints`
- `memory_jobs`

## What Changed

### Two-Column Memory

`agent_memory.content` remains for compatibility. New code treats it as the
same user-facing fact as `fact`.

| Column | Meaning |
|--------|---------|
| `fact` | concise fact, embedded |
| `main_content` | source context, not embedded |
| `memory_type` | semantic, profile, task, constraint, and other current types |
| `metadata` | policy metadata, conflict keys, source anchors |

### Session Summaries

`agent_sessions` includes:

- `summary`
- `summary_updated_at`

`add_conversation(..., update_summary=True)` can update the summary when an LLM
provider is configured and a `session_id` is supplied.

### Task Memory

Task memory adds:

| Table | Meaning |
|-------|---------|
| `agent_task_runs` | durable task run state |
| `agent_events` | append-only user/assistant/tool/artifact ledger |
| `agent_checkpoints` | compact resumable state |
| `memory_jobs` | durable background derivation queue |

## Embedding Dimension Changes

`connect()` checks the provider dimension against the vector column. If existing
embeddings would be cleared, Engram raises a `ConfigurationError` by default.

Only set this when you intentionally plan to clear and rebuild embeddings:

```bash
export ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true
```

Use a fresh test database when switching between OpenAI 1536-dimension embeddings
and local 384-dimension sentence-transformer embeddings.

## Rollback

The safest rollback is restore from backup:

```bash
psql "$ENGRAM_DATABASE_URL" < backup_before_engram_migration.sql
```

Manual rollback is not recommended. Task/event data, generated columns, vector
shape changes, and policy metadata can make partial rollback unsafe.
