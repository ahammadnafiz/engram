# Operations

This page collects local development, database, migration, worker, and
verification commands.

## Docker

Start only PostgreSQL:

```bash
docker compose up -d postgres
docker compose ps postgres
```

Start the full default compose set:

```bash
docker compose up -d
```

Stop containers without deleting data:

```bash
docker compose down
```

Delete the local database volume:

```bash
docker compose down -v
```

Follow logs:

```bash
docker compose logs -f postgres
```

The helper script wraps common compose flows:

```bash
./scripts/docker-setup.sh
./scripts/docker-setup.sh --status
./scripts/docker-setup.sh --logs
./scripts/docker-setup.sh --down
./scripts/docker-setup.sh --reset
```

The helper creates `.env` on the first run. On later runs it preserves existing
values, including `ENGRAM_OPENAI_API_KEY`, embedding settings, and other local
secrets. It only appends missing Docker defaults. Use
`./scripts/docker-setup.sh --port 5433` when you intentionally want to change
the local PostgreSQL port.

## Environment

Local compose defaults:

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

OpenAI:

```bash
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_EMBEDDING_DIMENSION=1536
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-your-key
```

## Database Shell

Interactive shell:

```bash
docker compose exec postgres psql -U engram -d engram
```

Single command:

```bash
docker compose exec -T postgres psql -U engram -d engram -c "SELECT version();"
```

List tables:

```bash
docker compose exec -T postgres psql -U engram -d engram -c "\dt"
```

Describe core tables:

```bash
docker compose exec -T postgres psql -U engram -d engram -c "\d agent_memory"
docker compose exec -T postgres psql -U engram -d engram -c "\d agent_task_runs"
docker compose exec -T postgres psql -U engram -d engram -c "\d agent_events"
docker compose exec -T postgres psql -U engram -d engram -c "\d agent_checkpoints"
docker compose exec -T postgres psql -U engram -d engram -c "\d memory_jobs"
```

## Schema Initialization

Normal library use initializes schema during `Engram.connect()`.

```python
from engram import Engram

async with Engram() as engram:
    health = await engram.health_check()
    print(health["status"])
```

For manual database setup, run the schema directly:

```bash
docker compose exec -T postgres psql -U engram -d engram \
  < src/engram/sql/schema.sql
```

Verify required extensions:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');"
```

## Memory Queries

Count memories:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT COUNT(*) FROM agent_memory;"
```

Recent memories:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT memory_id, memory_type, fact, importance, created_at
   FROM agent_memory
   ORDER BY created_at DESC
   LIMIT 10;"
```

Context usage:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT COUNT(*) AS total,
          COUNT(main_content) AS with_context,
          COUNT(*) - COUNT(main_content) AS without_context
   FROM agent_memory;"
```

Active critical memories:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT memory_type, fact, metadata->>'critical_slot' AS slot
   FROM agent_memory
   WHERE metadata->>'critical' = 'true'
     AND status <> 'superseded'
     AND COALESCE(metadata->>'status', 'active') <> 'superseded'
   ORDER BY created_at DESC
   LIMIT 20;"
```

Superseded memories:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT fact, superseded_by_memory_id, lineage_id, revision
   FROM agent_memory
   WHERE status = 'superseded'
   ORDER BY created_at DESC
   LIMIT 20;"
```

## Task And Worker Queries

Active tasks:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT task_run_id, agent_id, user_id, status, goal, updated_at
   FROM agent_task_runs
   WHERE deleted_at IS NULL
   ORDER BY updated_at DESC
   LIMIT 20;"
```

Recent events:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT event_id, task_run_id, role, event_type, left(content, 80) AS content
   FROM agent_events
   WHERE deleted_at IS NULL
   ORDER BY created_at DESC
   LIMIT 20;"
```

Job backlog:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT status, COUNT(*) FROM memory_jobs GROUP BY status;"
```

Failed jobs:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT job_id, attempts, left(error, 160) AS error, updated_at
   FROM memory_jobs
   WHERE status = 'failed'
   ORDER BY updated_at DESC
   LIMIT 20;"
```

## Run The Memory Worker

One inline batch:

```python
jobs = await engram.process_memory_jobs(limit=10)
print([job.status for job in jobs])
```

Long-running worker:

```python
from engram import Engram

async with Engram(memory_policy="coding_agent") as engram:
    processed = await engram.run_memory_worker(
        batch_size=20,
        interval_seconds=1.0,
    )
    print(processed)
```

For application deployments, run the worker as a separate process or service.

## Migrations

Fresh databases can use `schema.sql`. Existing databases should run migrations
in numeric order:

```bash
for file in src/engram/sql/migrations/*.sql; do
  docker compose exec -T postgres psql -U engram -d engram < "$file"
done
```

Verify migration state:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT column_name
   FROM information_schema.columns
   WHERE table_name = 'agent_memory'
     AND column_name IN ('fact', 'main_content', 'memory_type');"
```

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "SELECT table_name
   FROM information_schema.tables
   WHERE table_name IN (
     'agent_task_runs',
     'agent_events',
     'agent_checkpoints',
     'memory_jobs'
   );"
```

## Cleanup

Delete memories for one agent:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "DELETE FROM agent_memory WHERE agent_id = 'assistant';"
```

Delete fact, graph, session, and task data while keeping schema:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "TRUNCATE agent_memory, memory_relations, agent_sessions,
            agent_task_runs, agent_events, agent_checkpoints, memory_jobs
   CASCADE;"
```

Vacuum:

```bash
docker compose exec -T postgres psql -U engram -d engram -c \
  "VACUUM ANALYZE agent_memory;"
```

## Examples

```bash
python examples/basic_usage.py
python examples/long_input_usage.py
python examples/chatbot.py
```

For the real OpenAI-backed chatbot, the default is operator recall plus inline
memory-job processing:

```bash
export ENGRAM_CHATBOT_RECALL_MODE=operator
export ENGRAM_CHATBOT_MEMORY_JOBS=inline
export ENGRAM_CHATBOT_RERANK=auto
python examples/chatbot.py
```

`operator` first calls the LLM recall router with
`compose_answer=False`. `recall()` classifies the turn and retrieves structured
memory evidence; the chatbot then uses its regular OpenAI chat prompt to produce
the final answer from recall evidence, active memory context, and the memory
history timeline. This keeps broad history questions such as "what changed?"
from depending on the narrow recall composer. Set
`ENGRAM_CHATBOT_MEMORY_JOBS=deferred` to process jobs with `/jobs`,
`process_memory_jobs()`, or `run_memory_worker()`.

For lower latency and cost, set `ENGRAM_CHATBOT_RECALL_MODE=fast`. `fast`
performs one embedding-backed memory context lookup and critical-memory recall
before the OpenAI chat call.

For broad recall evaluation, use `ENGRAM_CHATBOT_RECALL_MODE=deep`. For
retrieval debugging, use `ENGRAM_CHATBOT_RECALL_MODE=debug`, which includes
`trace_recall()` output in the prompt and turn metadata.

`ENGRAM_CHATBOT_RERANK=auto` keeps reranking off in `fast` mode and enables it
in `deep` and `debug`. Set it to `true` to force reranking for every mode, or
`false` to disable it everywhere.

`deep` and `debug` also add a bounded recent-memory safety net for broad recall
questions. Tune `ENGRAM_CHATBOT_BROAD_MEMORY_LIMIT` and
`ENGRAM_CHATBOT_BROAD_MEMORY_CHARS` if your prompt budget is tight.

Chatbot commands:

| Command | Behavior |
|---------|----------|
| `/remember <fact>` | store a durable fact immediately |
| `/revise <memory_id> <fact>` | create a new active revision |
| `/lineage <memory_id>` | show the current head and revision history |
| `/history [active\|limit\|memory_id]` | show memory add/update timeline |
| `/memories` | list recent chatbot memories |
| `/jobs` | process queued memory extraction jobs |
| `/search <query>` | run hybrid search and reinforce hits |
| `/recall <question>` | ask current, historical, event, or lineage memory directly |
| `/trace <query>` | inspect recall trace |
| `/context <query>` | render memory and task context used for prompting |
| `/task` | show current resumable task/session |
| `/forget <memory_id>` | delete one memory |
| `/clear` | purge memories for the configured agent/user |
| `/help` | show help |
| `/quit` | exit |

## Verification Commands

```bash
python -c "from engram import Engram; print('OK')"
ruff check src tests examples
pytest tests/unit -q
pytest tests/integration -q --run-integration
mkdocs build --strict
```

Health check:

```python
import asyncio

from engram import Engram


async def check() -> None:
    async with Engram() as engram:
        health = await engram.health_check()
        print(health)


asyncio.run(check())
```
