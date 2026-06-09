-- ============================================================================
-- Migration: Add cognitive memory taxonomy
-- Run this ONCE on existing databases (also applied automatically by
-- init_schema() on connect).
-- Version: 003
-- Description: Tags each memory with a type:
--   semantic   - durable facts ("who the user is")
--   episodic   - dated events ("what happened")
--   procedural - behavioral rules ("how to act")
-- ============================================================================

BEGIN;

ALTER TABLE agent_memory
    ADD COLUMN IF NOT EXISTS memory_type TEXT NOT NULL DEFAULT 'semantic';

-- Add the value check if it is not already present.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'agent_memory_memory_type_check'
    ) THEN
        ALTER TABLE agent_memory
            ADD CONSTRAINT agent_memory_memory_type_check
            CHECK (memory_type IN ('semantic', 'episodic', 'procedural'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_memory_agent_type
    ON agent_memory(agent_id, memory_type);

COMMIT;
