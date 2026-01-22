-- =============================================================================
-- Engram Database Initialization
-- This script runs automatically on first PostgreSQL container start
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Confirm extensions are installed
DO $$
BEGIN
    RAISE NOTICE 'pgvector extension version: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
    RAISE NOTICE 'pg_trgm extension installed successfully';
END $$;
