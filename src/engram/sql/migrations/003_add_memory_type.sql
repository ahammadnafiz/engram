-- ============================================================================
-- Migration: Add cognitive memory taxonomy
-- Run this ONCE on existing databases (also applied automatically by
-- init_schema() on connect).
-- Version: 003
-- Description: Tags each memory with a type:
--   semantic   - durable facts ("who the user is")
--   episodic   - dated events ("what happened")
--   procedural - behavioral rules ("how to act")
--   profile, project, task, preference, constraint, decision, tool_result
-- ============================================================================

BEGIN;

ALTER TABLE agent_memory
    ADD COLUMN IF NOT EXISTS memory_type TEXT NOT NULL DEFAULT 'semantic';

ALTER TABLE agent_memory
    DROP CONSTRAINT IF EXISTS agent_memory_memory_type_check;

ALTER TABLE agent_memory
    ADD CONSTRAINT agent_memory_memory_type_check
    CHECK (
        memory_type IN (
            'semantic', 'episodic', 'procedural',
            'profile', 'project', 'task', 'preference',
            'constraint', 'decision', 'tool_result'
        )
    );

CREATE INDEX IF NOT EXISTS idx_memory_agent_type
    ON agent_memory(agent_id, memory_type);

COMMIT;
