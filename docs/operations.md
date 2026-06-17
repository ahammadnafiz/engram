# Operational Reference

This guide provides a cheat sheet of useful commands for local development, database introspection, and maintenance operations.

> [!NOTE]
> For configuration environment variables, see [Configuration](configuration.md). For schema upgrades, see [Migrations](migrations.md). For running the chatbot or other demo scripts, see [Examples](examples.md).

---

## 1. Docker & Infrastructure

Manage the local PostgreSQL instance equipped with `pgvector` and `pg_trgm`:

| Action | Command |
|--------|---------|
| Start Database | `docker compose up -d postgres` |
| Stop Database | `docker compose down` |
| **Wipe All Data** | `docker compose down -v` |
| Tail Logs | `docker compose logs -f postgres` |

> [!TIP]
> The repository includes a helper script for common workflows: `./scripts/docker-setup.sh`. It automatically provisions `.env` files and handles port binding (e.g., `./scripts/docker-setup.sh --reset`).

---

## 2. Database Introspection

These commands allow you to inspect the database schema and running state directly via `psql`.

**Interactive Shell:**
```bash
docker compose exec postgres psql -U engram -d engram
```

**List all tables:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c "\dt"
```

**Describe table schema:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c "\d agent_memory"
```

---

## 3. Useful SQL Queries

You can execute these directly against the running Docker container to debug what your agent is storing.

### Fact Memory

**Count total stored facts:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c "SELECT COUNT(*) FROM agent_memory;"
```

**View the 10 most recently stored facts:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT memory_id, memory_type, fact, importance, created_at
   FROM agent_memory
   ORDER BY created_at DESC
   LIMIT 10;"
```

**View superseded (historical) facts:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT fact, superseded_by_memory_id, lineage_id, revision
   FROM agent_memory
   WHERE status = 'superseded'
   ORDER BY created_at DESC
   LIMIT 20;"
```

### Task Ledger

**View active tasks:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT task_run_id, agent_id, user_id, status, goal, updated_at
   FROM agent_task_runs
   WHERE deleted_at IS NULL
   ORDER BY updated_at DESC
   LIMIT 20;"
```

**View the background derivation backlog:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT status, COUNT(*) FROM memory_jobs GROUP BY status;"
```

**Debug failed background jobs:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT job_id, attempts, left(error, 160) AS error, updated_at
   FROM memory_jobs
   WHERE status = 'failed'
   ORDER BY updated_at DESC
   LIMIT 20;"
```

---

## 4. Running the Memory Worker

If you are using deferred memory jobs (which is recommended for production), you must run the background worker to parse the event ledger into semantic facts.

**Run inline (batch mode):**
```python
from engram import Engram

async def run_batch():
    async with Engram() as engram:
        jobs = await engram.process_memory_jobs(limit=10)
        print(f"Processed {len(jobs)} jobs.")
```

**Run continuously (daemon mode):**
```python
from engram import Engram

async def run_daemon():
    async with Engram(memory_policy="coding_agent") as engram:
        await engram.run_memory_worker(batch_size=20, interval_seconds=1.0)
```

---

## 5. Cleanup & Data Wiping

> [!CAUTION]
> These commands are destructive. Ensure you are executing them against a local or test database.

**Delete all facts for a specific agent:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "DELETE FROM agent_memory WHERE agent_id = 'assistant';"
```

**Truncate all tables (wipe data, keep schema):**
```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "TRUNCATE agent_memory, memory_relations, agent_sessions,
            agent_task_runs, agent_events, agent_checkpoints, memory_jobs
   CASCADE;"
```

**Re-index and clean up dead tuples:**
```bash
docker compose exec -T postgres psql -U engram -d engram -c "VACUUM ANALYZE agent_memory;"
```

---

## 6. Testing & CI Checks

If you are contributing to Engram or running it locally, use these commands to verify repository health.

**Run the Python test suite:**
```bash
pytest tests/unit -q
pytest tests/integration -q --run-integration
```

**Run code quality checks:**
```bash
ruff check src tests examples
```

**Build documentation locally:**
```bash
mkdocs build --strict
```
