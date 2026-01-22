-- =============================================================================
-- Engram Seed Data (Optional)
-- Creates a default agent for quick testing
-- =============================================================================

-- Insert a default agent for testing
INSERT INTO agents (agent_id, name, description, config)
VALUES (
    'default',
    'Default Agent',
    'Default Engram agent for testing and development',
    '{"version": "1.0", "features": ["memory", "search", "graph"]}'::jsonb
)
ON CONFLICT (agent_id) DO NOTHING;

-- Verification
DO $$
BEGIN
    RAISE NOTICE 'Default agent created (or already exists)';
END $$;
