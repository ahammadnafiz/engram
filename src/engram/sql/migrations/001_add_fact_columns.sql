-- ============================================================================
-- Migration: Add fact and main_content columns
-- Run this ONCE on existing databases
-- Version: 001
-- Description: Implements two-column memory system
--   - fact: Extracted user facts (embedded for search)
--   - main_content: Full conversation context (not embedded)
-- ============================================================================

BEGIN;

-- Step 1: Add new columns (if not exist)
ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS fact TEXT;
ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS main_content TEXT;

-- Step 2: Migrate existing content to fact column
-- Only update rows where fact is NULL (not yet migrated)
UPDATE agent_memory SET fact = content WHERE fact IS NULL;

-- Step 3: Make fact NOT NULL (after migration)
ALTER TABLE agent_memory ALTER COLUMN fact SET NOT NULL;

-- Step 4: Drop old content_tsv column and add fact_tsv
-- Note: PostgreSQL will error if column doesn't exist, so we use dynamic SQL
DO $$
BEGIN
    -- Drop old content_tsv if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agent_memory' AND column_name = 'content_tsv'
    ) THEN
        ALTER TABLE agent_memory DROP COLUMN content_tsv;
    END IF;
    
    -- Add fact_tsv if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agent_memory' AND column_name = 'fact_tsv'
    ) THEN
        ALTER TABLE agent_memory ADD COLUMN fact_tsv TSVECTOR 
            GENERATED ALWAYS AS (to_tsvector('english', fact)) STORED;
    END IF;
    
    -- Add main_content_tsv if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agent_memory' AND column_name = 'main_content_tsv'
    ) THEN
        ALTER TABLE agent_memory ADD COLUMN main_content_tsv TSVECTOR 
            GENERATED ALWAYS AS (
                CASE WHEN main_content IS NOT NULL 
                THEN to_tsvector('english', main_content) 
                ELSE NULL END
            ) STORED;
    END IF;
END $$;

-- Step 5: Update unique constraint
-- Drop old content-based constraint if it exists
DROP INDEX IF EXISTS idx_unique_memory_content;

-- Create new fact-based constraint if it doesn't exist
-- (md5(fact) keeps entries under the btree row-size limit for long facts)
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_memory_fact
    ON agent_memory(agent_id, COALESCE(user_id, ''), md5(fact));

-- Step 6: Update text search indexes
DROP INDEX IF EXISTS idx_memory_content_tsv;
DROP INDEX IF EXISTS idx_memory_content_trgm;

CREATE INDEX IF NOT EXISTS idx_memory_fact_tsv ON agent_memory USING GIN (fact_tsv);
CREATE INDEX IF NOT EXISTS idx_memory_main_content_tsv ON agent_memory USING GIN (main_content_tsv) 
    WHERE main_content IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_fact_trgm ON agent_memory USING GIN (fact gin_trgm_ops);

COMMIT;

-- ============================================================================
-- Verification Query (run manually to verify migration)
-- ============================================================================
-- SELECT 
--     COUNT(*) AS total_memories,
--     COUNT(fact) AS memories_with_fact,
--     COUNT(main_content) AS memories_with_main_content,
--     COUNT(*) FILTER (WHERE fact = content) AS fact_equals_content
-- FROM agent_memory;

