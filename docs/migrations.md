# Database Migrations

This guide covers how to initialize, upgrade, and manage the PostgreSQL database schema for Engram.

> [!WARNING]
> Engram is currently in alpha and the schema is subject to change. **Always back up your database before running migrations against an existing populated database.**

---

## 1. Fresh Database Setup

If you are starting a new project, Engram can initialize the database schema automatically. When you call `Engram.connect()`, the client executes the `schema.sql` file if the tables do not exist.

```python
from engram import Engram

async with Engram() as engram:
    health = await engram.health_check()
    print(health["status"])
```

### Manual Initialization
If you prefer to initialize the schema manually via CLI:

```bash
docker compose up -d postgres
docker compose exec -T postgres psql -U engram -d engram < src/engram/sql/schema.sql
```

> [!NOTE]
> `schema.sql` creates all required extensions (`vector`, `pg_trgm`), tables, indexes, and generated `tsvector` columns.

---

## 2. Upgrading an Existing Database

If you are upgrading an older installation of Engram, you must apply migrations sequentially.

### The Migration Ledger

| Migration Script | Purpose |
|------------------|---------|
| `001_add_fact_columns.sql` | Adds the explicit `fact` and `main_content` columns, along with related indexes and `tsvector` generated columns. |
| `002_add_session_summary.sql` | Adds rolling summaries to the `agent_sessions` table. |
| `003_add_memory_type.sql` | Adds the `memory_type` column and a type index. |
| `004_add_task_memory.sql` | Creates the tables required for Task Memory: `agent_task_runs`, `agent_events`, `agent_checkpoints`, and `memory_jobs`. |
| `005_widen_memory_type_constraint.sql` | Widens the database constraint to allow all current policy memory types. |
| `006_add_memory_lineage.sql` | Adds `status`, `lineage_id`, `revision`, and `superseded_by` columns to support immutable memory revisions and conflict resolution. |
| `007_add_event_search.sql` | Adds nullable `event_embedding` to `agent_events` to support hybrid event recall on the task ledger. |

### Step 1: Back Up Your Database

Using standard `pg_dump`:
```bash
pg_dump "$ENGRAM_DATABASE_URL" > backup_before_engram_migration.sql
```

Using Docker:
```bash
docker compose exec -T postgres pg_dump -U engram -d engram > backup_before_engram_migration.sql
```

### Step 2: Apply Migrations

You can apply them one by one, or via a simple loop:

```bash
for file in src/engram/sql/migrations/*.sql; do
  docker compose exec -T postgres psql -U engram -d engram < "$file"
done
```

---

## 3. Schema Verifications

After running the migrations, you can verify the tables exist with the following SQL queries.

**Verify `agent_memory` columns:**
```sql
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'agent_memory'
  AND column_name IN ('fact', 'main_content', 'memory_type', 'lineage_id', 'status');
```

**Verify Task Memory tables:**
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

---

## 4. Troubleshooting

### Embedding Dimension Errors

By default, when `connect()` is called, Engram checks the dimension of your configured embedding provider against the `embedding` column in PostgreSQL. 

> [!CAUTION]
> If the dimension has changed (e.g., switching from OpenAI's `1536` to local `384`), altering the pgvector column type will **permanently delete all existing embeddings** in the database.

To protect your data, Engram will raise a `ConfigurationError` and refuse to boot if it detects this. If you are *intentionally* switching models and are prepared to re-embed all your data (or are just running tests), you must explicitly allow the destructive schema change:

```bash
export ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true
```

### Rollbacks

Because Engram relies heavily on generated columns, vector shapes, and immutable ledgers, manual database downgrades are **not recommended or supported**.

If a migration fails, the safest rollback strategy is dropping the database and restoring from the backup you took in Step 1:

```bash
psql "$ENGRAM_DATABASE_URL" < backup_before_engram_migration.sql
```
