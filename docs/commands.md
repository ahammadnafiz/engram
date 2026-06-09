# Engram Command Reference

Quick reference for Docker, database, and chatbot commands.

---

## Docker Commands

### Container Management

```bash
# Start Engram (auto-detect port)
./scripts/docker-setup.sh

# Stop containers
./scripts/docker-setup.sh --down

# Reset database (delete all data)
./scripts/docker-setup.sh --reset

# Fresh start (reset + start)
./scripts/docker-setup.sh --reset && ./scripts/docker-setup.sh

# Check status
./scripts/docker-setup.sh --status

# View logs
./scripts/docker-setup.sh --logs
```

### Direct Docker Compose

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Fresh start (removes volumes) - requires schema init after!
docker compose down -v && docker compose up -d

# View logs
docker compose logs -f
```

### Fresh Start (Complete)

After `docker compose down -v`, the database is empty. Run the schema to initialize:

```bash
# Fresh start with schema initialization
docker compose down -v && docker compose up -d && \
  sleep 2 && \
  docker exec -i engram-postgres psql -U engram -d engram < src/engram/sql/schema.sql
```

Or step by step:

```bash
# 1. Reset database
docker compose down -v && docker compose up -d

# 2. Wait for postgres to start
sleep 2

# 3. Initialize schema (includes fact + main_content columns)
docker exec -i engram-postgres psql -U engram -d engram < src/engram/sql/schema.sql

# 4. Verify
docker exec -i engram-postgres psql -U engram -d engram -c "\d agent_memory"
```

---

## Database Commands

### Connect to Database

```bash
# Interactive psql shell
docker exec -it engram-postgres psql -U engram -d engram

# Run single command
docker exec -i engram-postgres psql -U engram -d engram -c "YOUR_SQL_HERE"
```

### Schema & Structure

```bash
# List all tables
docker exec -i engram-postgres psql -U engram -d engram -c "\dt"

# View core table structures
docker exec -i engram-postgres psql -U engram -d engram -c "\d agent_memory"
docker exec -i engram-postgres psql -U engram -d engram -c "\d agent_task_runs"
docker exec -i engram-postgres psql -U engram -d engram -c "\d agent_events"
docker exec -i engram-postgres psql -U engram -d engram -c "\d agent_checkpoints"
docker exec -i engram-postgres psql -U engram -d engram -c "\d memory_jobs"

# View all indexes
docker exec -i engram-postgres psql -U engram -d engram -c "\di"
```

### Memory Queries

```bash
# Count all memories
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT COUNT(*) FROM agent_memory;"

# View recent memories (fact + main_content)
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT memory_id, fact, main_content, importance, created_at 
   FROM agent_memory ORDER BY created_at DESC LIMIT 10;"

# View memories with context stats
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT COUNT(*) as total, 
          COUNT(main_content) as with_context,
          COUNT(*) - COUNT(main_content) as without_context
   FROM agent_memory;"

# Pretty formatted view (expanded)
docker exec -i engram-postgres psql -U engram -d engram -c "\x" -c \
  "SELECT * FROM agent_memory LIMIT 3;"

# Search memories by content
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT fact, main_content FROM agent_memory 
   WHERE fact ILIKE '%nafiz%' LIMIT 5;"

# View memories by importance
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT memory_type, fact, importance, access_count, metadata->>'status' AS status
   FROM agent_memory ORDER BY importance DESC LIMIT 10;"

# View active critical memories
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT memory_type, fact, metadata->>'critical_slot' AS slot
   FROM agent_memory
   WHERE metadata->>'critical' = 'true'
     AND COALESCE(metadata->>'status', 'active') <> 'superseded'
   ORDER BY created_at DESC LIMIT 20;"

# View superseded memories
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT fact, metadata->>'superseded_by' AS superseded_by
   FROM agent_memory
   WHERE metadata->>'status' = 'superseded'
   ORDER BY created_at DESC LIMIT 20;"
```

### Agent & User Queries

```bash
# List agents
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT * FROM agents;"

# List users
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT * FROM users;"

# Memories per agent
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT agent_id, COUNT(*) as memory_count 
   FROM agent_memory GROUP BY agent_id;"
```

### Relations & Graph

```bash
# View memory relations
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT source_memory_id, target_memory_id, relation_type, weight 
   FROM memory_relations LIMIT 10;"

# Count relations
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT COUNT(*) FROM memory_relations;"
```

### Sessions

```bash
# View active sessions
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT * FROM agent_sessions WHERE ended_at IS NULL;"

# All sessions
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT session_id, agent_id, started_at, ended_at, summary_updated_at
   FROM agent_sessions;"
```

### Long-Running Tasks

```bash
# Active tasks
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT task_run_id, agent_id, user_id, status, goal, updated_at
   FROM agent_task_runs
   WHERE deleted_at IS NULL
   ORDER BY updated_at DESC LIMIT 20;"

# Recent events
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT event_id, task_run_id, role, event_type, left(content, 80) AS content
   FROM agent_events
   WHERE deleted_at IS NULL
   ORDER BY created_at DESC LIMIT 20;"

# Latest checkpoints
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT checkpoint_id, task_run_id, left(summary, 120) AS summary, created_at
   FROM agent_checkpoints
   ORDER BY created_at DESC LIMIT 20;"

# Memory job backlog
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT status, COUNT(*) FROM memory_jobs GROUP BY status;"

# Failed memory jobs
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT job_id, attempts, left(error, 160) AS error, updated_at
   FROM memory_jobs
   WHERE status = 'failed'
   ORDER BY updated_at DESC LIMIT 20;"
```

### Cleanup Commands

```bash
# Delete all memories for an agent
docker exec -i engram-postgres psql -U engram -d engram -c \
  "DELETE FROM agent_memory WHERE agent_id = 'assistant';"

# Delete all data (keep schema)
docker exec -i engram-postgres psql -U engram -d engram -c \
  "TRUNCATE agent_memory, memory_relations, agent_sessions CASCADE;"

# Vacuum (reclaim space)
docker exec -i engram-postgres psql -U engram -d engram -c \
  "VACUUM ANALYZE agent_memory;"
```

---

## Schema & Migration Commands

### When to Use What

| Scenario | Command |
|----------|---------|
| Fresh database (after `down -v`) | Run `schema.sql` |
| Existing database, upgrade current alpha schema | Run migrations `001` through `005` |
| Error: "column fact does not exist" | Run `001_add_fact_columns.sql` |
| Error: "column memory_type does not exist" | Run `003_add_memory_type.sql` |
| Error: "relation agent_task_runs does not exist" | Run `004_add_task_memory.sql` |

### Fresh Install (schema.sql)

Use after `docker compose down -v` or new database:

```bash
# Initialize schema with current alpha system
docker exec -i engram-postgres psql -U engram -d engram \
  < src/engram/sql/schema.sql

# Verify columns exist
docker exec -i engram-postgres psql -U engram -d engram -c "\d agent_memory"
```

### Upgrade Existing Database (migration)

Use when you have existing data:

```bash
# Run migrations in order
for file in src/engram/sql/migrations/*.sql; do
  docker exec -i engram-postgres psql -U engram -d engram < "$file"
done

# Verify migration
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT COUNT(*) as total,
          COUNT(fact) as with_fact,
          COUNT(main_content) as with_main_content,
          COUNT(memory_type) as with_memory_type
   FROM agent_memory;"
```

### Troubleshooting

```bash
# Error: "column fact does not exist"
# Solution: Run schema or migration
docker exec -i engram-postgres psql -U engram -d engram < src/engram/sql/schema.sql

# Check if current alpha columns/tables exist
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT column_name FROM information_schema.columns 
   WHERE table_name = 'agent_memory'
     AND column_name IN ('fact', 'main_content', 'memory_type');"

docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_name IN ('agent_task_runs', 'agent_events', 'agent_checkpoints', 'memory_jobs');"
```

---

## Chatbot Commands

### Running the Chatbot

```bash
# Activate environment and run
conda activate engram
python examples/chatbot.py
```

### In-Chat Commands

| Command | Description |
|---------|-------------|
| `/memories` | List recent memories |
| `/search <query>` | Hybrid search |
| `/graph` | Show memory relations |
| `/task` | Show active long-running task context |
| `/worker` | Process queued memory jobs |
| `/forget` | Clear all memories |
| `/help` | Show help |
| `/quit` | Exit |

---

## Python Quick Commands

```bash
# Test imports
python -c "from engram import Engram; print('OK')"

# Run unit tests
python -m pytest tests/unit/ -v

# Run specific test
python -m pytest tests/unit/test_memory_store_comprehensive.py -v

# Check health
python -c "
import asyncio
from engram import Engram

async def check():
    async with Engram() as e:
        h = await e.health_check()
        print(h)

asyncio.run(check())
"
```

---

## Environment Variables

```bash
# Required for local compose
export ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram

# Embedding provider
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional LLM provider
export ENGRAM_LLM_PROVIDER=openai
export ENGRAM_LLM_MODEL=gpt-4o-mini
export ENGRAM_OPENAI_API_KEY=sk-your-key

# Optional OpenAI embeddings instead of local embeddings
export ENGRAM_EMBEDDING_PROVIDER=openai
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
```

---

## Useful Aliases

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Engram shortcuts
alias engram-start="cd /path/to/engram && ./scripts/docker-setup.sh"
alias engram-stop="cd /path/to/engram && ./scripts/docker-setup.sh --down"
alias engram-reset="cd /path/to/engram && ./scripts/docker-setup.sh --reset"
alias engram-db="docker exec -it engram-postgres psql -U engram -d engram"
alias engram-chat="cd /path/to/engram && python examples/chatbot.py"

# Quick memory check
alias engram-memories="docker exec -i engram-postgres psql -U engram -d engram -c 'SELECT fact, importance FROM agent_memory ORDER BY created_at DESC LIMIT 10;'"
```
