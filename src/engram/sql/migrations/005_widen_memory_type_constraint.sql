-- ============================================================================
-- Migration: Widen memory_type check constraint for policy-driven taxonomy
-- Version: 005
-- ============================================================================

BEGIN;

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

COMMIT;
