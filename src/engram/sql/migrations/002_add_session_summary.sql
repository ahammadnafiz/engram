-- ============================================================================
-- Migration: Add rolling conversation summary to sessions
-- Run this ONCE on existing databases (also applied automatically by
-- init_schema() on connect).
-- Version: 002
-- Description: Per-session rolling summary, iteratively updated by
--   add_conversation() and fed back into fact extraction as context.
-- ============================================================================

BEGIN;

ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS summary TEXT;
ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS summary_updated_at TIMESTAMPTZ;

COMMIT;
