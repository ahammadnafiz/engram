-- ============================================================================
-- Engram Database Schema
-- PostgreSQL + pgvector schema for AI memory storage
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- Agents Table
-- Stores AI agent configurations and metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Users Table
-- Stores user information for multi-user agent interactions
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Agent Sessions Table
-- Tracks conversation sessions between agents and users
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    
    -- Indices
    CONSTRAINT fk_agent FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_agent ON agent_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON agent_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON agent_sessions(started_at DESC);

-- ============================================================================
-- Agent Memory Table
-- Core memory storage with vector embeddings and full-text search
-- Two-column system: fact (embedded) + main_content (context, not embedded)
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_memory (
    memory_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    session_id TEXT REFERENCES agent_sessions(session_id) ON DELETE SET NULL,
    
    -- LEGACY: Kept for backward compatibility (maps to fact)
    content TEXT NOT NULL,
    
    -- NEW: Two-column memory system
    fact TEXT NOT NULL,              -- Extracted user fact (EMBEDDED for search)
    main_content TEXT,               -- [USER]: msg\n[AI]: summary (NOT embedded, context only)
    
    -- Embedding for fact column only
    embedding VECTOR(1536),          -- Auto-adjusted by Engram to match provider
    
    -- Full-text search vectors
    fact_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', fact)) STORED,
    main_content_tsv TSVECTOR GENERATED ALWAYS AS (
        CASE WHEN main_content IS NOT NULL 
        THEN to_tsvector('english', main_content) 
        ELSE NULL END
    ) STORED,
    
    -- Scoring factors
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    access_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Vector similarity search index (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON agent_memory 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_memory_fact_tsv ON agent_memory USING GIN (fact_tsv);
CREATE INDEX IF NOT EXISTS idx_memory_main_content_tsv ON agent_memory USING GIN (main_content_tsv) 
    WHERE main_content IS NOT NULL;

-- Trigram index for fuzzy text matching on fact
CREATE INDEX IF NOT EXISTS idx_memory_fact_trgm ON agent_memory 
    USING GIN (fact gin_trgm_ops);

-- Compound indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_memory_agent ON agent_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_agent_user ON agent_memory(agent_id, user_id);
CREATE INDEX IF NOT EXISTS idx_memory_agent_session ON agent_memory(agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_memory_created ON agent_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_last_accessed ON agent_memory(last_accessed_at DESC);

-- JSONB index for metadata queries
CREATE INDEX IF NOT EXISTS idx_memory_metadata ON agent_memory USING GIN (metadata);

-- Prevent duplicate facts per agent+user
-- This ensures the same fact isn't stored twice
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_memory_fact 
    ON agent_memory(agent_id, COALESCE(user_id, ''), fact);

-- ============================================================================
-- Memory Relations Table
-- Graph structure for memory associations
-- ============================================================================
CREATE TABLE IF NOT EXISTS memory_relations (
    id SERIAL PRIMARY KEY,
    source_memory_id TEXT NOT NULL REFERENCES agent_memory(memory_id) ON DELETE CASCADE,
    target_memory_id TEXT NOT NULL REFERENCES agent_memory(memory_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL DEFAULT 'related_to',
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0 AND weight <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prevent duplicate relations
    UNIQUE (source_memory_id, target_memory_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON memory_relations(relation_type);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to update last_accessed_at on memory access
CREATE OR REPLACE FUNCTION update_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_accessed_at = NOW();
    NEW.access_count = OLD.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for timestamp updates
DROP TRIGGER IF EXISTS trigger_agents_updated ON agents;
CREATE TRIGGER trigger_agents_updated
    BEFORE UPDATE ON agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS trigger_users_updated ON users;
CREATE TRIGGER trigger_users_updated
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- NOTE: Memory access tracking (update_memory_access) is handled in application code
-- via the reinforce() method, not via triggers, to avoid overhead on every SELECT.

-- ============================================================================
-- Decay Calculation Function
-- Implements MemoryBank-style decay: decay_rate ^ hours_elapsed
-- ============================================================================
CREATE OR REPLACE FUNCTION calculate_decay(
    last_accessed TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 0.995
)
RETURNS FLOAT AS $$
DECLARE
    hours_elapsed FLOAT;
BEGIN
    hours_elapsed := EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 3600.0;
    RETURN POWER(decay_rate, hours_elapsed);
END;
$$ LANGUAGE plpgsql STABLE;  -- STABLE not IMMUTABLE: uses NOW() which changes per statement
