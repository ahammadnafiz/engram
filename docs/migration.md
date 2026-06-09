# Database Migration Guide

This guide covers the schema changes for the current alpha line.

!!! warning
    Back up the database before running migrations. The current package is
    alpha-stage and schema evolution is still active.

## Migration Order

Run migrations in numeric order:

| Migration | Purpose |
|-----------|---------|
| `001_add_fact_columns.sql` | Adds two-column memory: `fact`, `main_content`, generated TSV columns, indexes |
| `002_add_session_summary.sql` | Adds rolling summaries to `agent_sessions` |
| `003_add_memory_type.sql` | Adds `memory_type` and type index |
| `004_add_task_memory.sql` | Adds task runs, event ledger, checkpoints, memory jobs |

Fresh installs can run `src/engram/sql/schema.sql` instead of individual
migrations.

## Backup

```bash
pg_dump "$ENGRAM_DATABASE_URL" > backup_before_engram_migration.sql
```

Docker example:

```bash
docker exec engram-postgres pg_dump -U engram -d engram \
  > backup_before_engram_migration.sql
```

## Run Migrations

```bash
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/001_add_fact_columns.sql
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/002_add_session_summary.sql
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/003_add_memory_type.sql
psql "$ENGRAM_DATABASE_URL" -f src/engram/sql/migrations/004_add_task_memory.sql
```

Docker:

```bash
for file in src/engram/sql/migrations/*.sql; do
  docker exec -i engram-postgres psql -U engram -d engram < "$file"
done
```

## Verify

```sql
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'agent_memory'
  AND column_name IN ('fact', 'main_content', 'memory_type');

SELECT table_name
FROM information_schema.tables
WHERE table_name IN (
  'agent_tasks',
  'agent_events',
  'task_checkpoints',
  'memory_jobs'
);
```

Expected tables:

- `agents`
- `users`
- `agent_memory`
- `memory_relations`
- `agent_sessions`
- `agent_tasks`
- `agent_events`
- `task_checkpoints`
- `memory_jobs`

## What Changed

### Two-column memory

`agent_memory.content` remains for compatibility. New code treats `content` as
the same user-facing fact as `fact`.

| Column | Meaning |
|--------|---------|
| `fact` | concise fact, embedded |
| `main_content` | source context, not embedded |
| `memory_type` | semantic, profile, task, constraint, etc. |
| `metadata` | policy metadata, conflict keys, source anchors |

### Session summaries

`agent_sessions` now has:

- `summary`
- `summary_updated_at`

`add_conversation(..., update_summary=True)` can roll this forward.

### Task memory

`004_add_task_memory.sql` adds:

| Table | Meaning |
|-------|---------|
| `agent_tasks` | durable task run state |
| `agent_events` | append-only user/assistant/tool/artifact ledger |
| `task_checkpoints` | compact resumable state |
| `memory_jobs` | durable background derivation queue |

## Fresh Install

```bash
docker compose up -d
docker exec -i engram-postgres psql -U engram -d engram \
  < src/engram/sql/schema.sql
```

`PostgresStorage.connect()` also initializes schema for normal library use.

## Rollback Strategy

The safest rollback is restore from backup:

```bash
psql "$ENGRAM_DATABASE_URL" < backup_before_engram_migration.sql
```

Manual rollback is not recommended because task/event data and policy metadata
may have been written after migration.

## Operational Notes

- `search()` hides memories with `metadata.status = "superseded"`.
- `trace_recall()` can report superseded memories for debugging.
- If switching embedding dimensions, use a fresh database or rebuild the vector
  column and all embeddings.
- Keep migration files in source control and run them exactly once per database.

