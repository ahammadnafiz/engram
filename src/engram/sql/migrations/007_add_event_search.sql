-- ============================================================================
-- Migration: Add scalable hybrid search support to the event ledger
-- Run this ONCE on existing databases (also applied automatically by
-- init_schema() on connect).
-- Version: 007
-- Description: Adds a nullable event_embedding column for semantic event
--   recall. It intentionally avoids a generated tsvector column and avoids
--   backfilling embeddings inside the migration; both would rewrite/scan large
--   event ledgers during application startup. Search indexes are created
--   separately by init_schema() with CREATE INDEX CONCURRENTLY.
-- ============================================================================

BEGIN;

ALTER TABLE agent_events ADD COLUMN IF NOT EXISTS event_embedding VECTOR(1536);

-- Do not drop old pre-release content_tsv columns or indexes here. Startup
-- replaces the index online with CREATE/DROP INDEX CONCURRENTLY, and leaving an
-- unused generated column in place is safer than taking an ACCESS EXCLUSIVE
-- table lock during application startup.

COMMIT;
