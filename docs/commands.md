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

# View agent_memory table structure
docker exec -i engram-postgres psql -U engram -d engram -c "\d agent_memory"

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
  "SELECT fact, importance, access_count 
   FROM agent_memory ORDER BY importance DESC LIMIT 10;"
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
  "SELECT session_id, agent_id, started_at, ended_at FROM agent_sessions;"
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
| Existing database, upgrade to two-column | Run migration |
| Error: "column fact does not exist" | Run `schema.sql` or migration |

### Fresh Install (schema.sql)

Use after `docker compose down -v` or new database:

```bash
# Initialize schema with two-column system
docker exec -i engram-postgres psql -U engram -d engram \
  < src/engram/sql/schema.sql

# Verify columns exist
docker exec -i engram-postgres psql -U engram -d engram -c "\d agent_memory"
```

### Upgrade Existing Database (migration)

Use when you have existing data and need to add fact/main_content columns:

```bash
# Run migration (preserves existing data)
docker exec -i engram-postgres psql -U engram -d engram \
  < src/engram/sql/migrations/001_add_fact_columns.sql

# Verify migration
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT COUNT(*) as total,
          COUNT(fact) as with_fact,
          COUNT(main_content) as with_main_content
   FROM agent_memory;"
```

### Troubleshooting

```bash
# Error: "column fact does not exist"
# Solution: Run schema or migration
docker exec -i engram-postgres psql -U engram -d engram < src/engram/sql/schema.sql

# Check if columns exist
docker exec -i engram-postgres psql -U engram -d engram -c \
  "SELECT column_name FROM information_schema.columns 
   WHERE table_name = 'agent_memory' AND column_name IN ('fact', 'main_content');"
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
# Required
export OPENAI_API_KEY=sk-your-key

# Optional overrides
export ENGRAM_DATABASE_URL=postgresql://engram:engram@localhost:5432/engram
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

