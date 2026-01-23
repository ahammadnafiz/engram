# Database Migration Guide

This guide covers migrating existing Engram databases to the two-column memory system.

## Overview

**Version**: 001 - Two-Column Memory System

The migration adds two new columns to the `agent_memory` table:

| Column | Purpose | Embedded? |
|--------|---------|-----------|
| `fact` | Extracted user facts for search | Yes |
| `main_content` | Conversation context `[USER]: ...\n[AI]: ...` | No |

This design is **cost-effective**: only facts are embedded, while full conversation context is preserved without additional embedding costs.

---

## Prerequisites

1. **Backup your database** before running any migration
2. PostgreSQL 14+ with pgvector extension
3. Database credentials with ALTER TABLE permissions

---

## Migration Steps

### Step 1: Backup (Required)

```bash
# Full backup
pg_dump -U engram -d engram > backup_before_migration.sql

# Or just the agent_memory table
pg_dump -U engram -d engram -t agent_memory > agent_memory_backup.sql
```

### Step 2: Run Migration

```bash
# Using psql directly
psql -U engram -d engram -f src/engram/sql/migrations/001_add_fact_columns.sql

# Or with host/port specified
psql -h localhost -p 5432 -U engram -d engram -f src/engram/sql/migrations/001_add_fact_columns.sql

# Or if using Docker
docker exec -i engram-postgres psql -U engram -d engram < src/engram/sql/migrations/001_add_fact_columns.sql
```

### Step 3: Verify Migration

```sql
-- Connect to database
psql -U engram -d engram

-- Check columns were added
\d agent_memory

-- Verify data migration
SELECT 
    COUNT(*) AS total_memories,
    COUNT(fact) AS memories_with_fact,
    COUNT(main_content) AS memories_with_main_content,
    COUNT(*) FILTER (WHERE fact = content) AS fact_equals_content
FROM agent_memory;
```

Expected output after migration:
- `memories_with_fact` should equal `total_memories`
- `fact_equals_content` should equal `total_memories` (existing content migrated to fact)
- `memories_with_main_content` will be 0 for old data (new memories will have it)

---

## What the Migration Does

1. **Adds new columns**:
   - `fact TEXT NOT NULL` - stores the user fact (embedded)
   - `main_content TEXT` - stores conversation context (not embedded)

2. **Migrates existing data**:
   - Copies `content` → `fact` for all existing rows

3. **Updates indexes**:
   - Drops old `content_tsv` index
   - Creates `fact_tsv` for keyword search on facts
   - Creates `main_content_tsv` for fallback search

4. **Updates unique constraint**:
   - Changes from `(agent_id, user_id, content)` to `(agent_id, user_id, fact)`

---

## Rollback (If Needed)

If you need to rollback the migration:

```sql
-- Restore from backup
psql -U engram -d engram < backup_before_migration.sql

-- Or manually revert (data loss for main_content)
BEGIN;

-- Remove new columns
ALTER TABLE agent_memory DROP COLUMN IF EXISTS fact;
ALTER TABLE agent_memory DROP COLUMN IF EXISTS main_content;
ALTER TABLE agent_memory DROP COLUMN IF EXISTS fact_tsv;
ALTER TABLE agent_memory DROP COLUMN IF EXISTS main_content_tsv;

-- Recreate old index
DROP INDEX IF EXISTS idx_unique_memory_fact;
CREATE UNIQUE INDEX idx_unique_memory_content 
    ON agent_memory(agent_id, COALESCE(user_id, ''), content);

-- Recreate content_tsv
ALTER TABLE agent_memory ADD COLUMN content_tsv TSVECTOR 
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX idx_memory_content_tsv ON agent_memory USING GIN (content_tsv);

COMMIT;
```

---

## Fresh Installation

For new installations, no migration is needed. The schema already includes the two-column system:

```bash
# Initialize fresh database
psql -U engram -d engram -f src/engram/sql/schema.sql
```

---

## Troubleshooting

### "Column already exists"

The migration uses `IF NOT EXISTS` and `IF EXISTS` clauses, so running it multiple times is safe.

### "Permission denied"

Ensure your database user has ALTER TABLE permissions:

```sql
GRANT ALL ON TABLE agent_memory TO engram;
```

### "fact cannot be null"

If you have NULL content rows (shouldn't happen), fix them first:

```sql
-- Find problematic rows
SELECT memory_id FROM agent_memory WHERE content IS NULL;

-- Delete or fix them before migration
DELETE FROM agent_memory WHERE content IS NULL;
```

---

## After Migration

Once migrated, the chatbot will automatically:

1. Store facts in `fact` column (embedded for search)
2. Store conversation context in `main_content` (not embedded)
3. Return both columns in search results for richer LLM context

Example of new memory structure:

```
┌─────────────────────────────────────────────────────────────┐
│ fact:         "User's name is Nafiz"          ← EMBEDDED    │
│ main_content: "[USER]: I'm Nafiz, I work..."  ← NOT embedded│
│               "[AI]: Nice to meet you!"                     │
└─────────────────────────────────────────────────────────────┘
```

