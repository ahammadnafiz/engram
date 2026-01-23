-- ============================================================================
-- Semantic Search Query
-- Pure vector similarity search without keyword matching
-- Two-column system: searches on fact embeddings, returns both fact and main_content
-- ============================================================================

-- Parameters:
-- $1: query_embedding (VECTOR) - The query vector
-- $2: agent_id (TEXT) - Filter by agent
-- $3: user_id (TEXT) - Filter by user (optional, NULL for all)
-- $4: limit_count (INTEGER) - Number of results
-- $5: decay_rate (FLOAT) - Decay rate per hour

-- Note: Weights are adjusted from hybrid search (0.40 semantic + 0.20 keyword + 0.25 decay + 0.15 importance)
-- Since no keyword matching, semantic weight absorbs keyword weight: 0.40 + 0.20 = 0.60

SELECT 
    memory_id,
    fact AS content,  -- Return fact as content for backward API compatibility
    fact,
    main_content,
    importance,
    metadata,
    created_at,
    last_accessed_at,
    -- Semantic similarity (0-1 range, clamped)
    GREATEST(0, 1 - (embedding <=> $1::vector)) AS semantic_score,
    -- Time decay (0-1 range)
    calculate_decay(last_accessed_at, $5) AS decay_score,
    -- Combined score: semantic(0.60) + decay(0.25) + importance(0.15) = 1.0
    (
        0.60 * GREATEST(0, 1 - (embedding <=> $1::vector)) +
        0.25 * calculate_decay(last_accessed_at, $5) +
        0.15 * importance
    ) AS score
FROM agent_memory
WHERE agent_id = $2
    AND ($3::text IS NULL OR user_id = $3)
    AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $4;
