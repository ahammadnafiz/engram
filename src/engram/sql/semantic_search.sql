-- ============================================================================
-- Semantic Search Query (Optimized v2.1)
-- Pure vector similarity search without keyword matching
-- Faster than hybrid search when keyword matching not needed
-- Two-column system: searches on fact embeddings, returns both fact and main_content
-- ============================================================================

-- Parameters:
-- $1: query_embedding (VECTOR) - The query vector
-- $2: agent_id (TEXT) - Filter by agent
-- $3: user_id (TEXT) - Filter by user (optional, NULL for all)
-- $4: limit_count (INTEGER) - Number of results
-- $5: decay_rate (FLOAT) - Decay rate per hour (default: 0.995)
-- $6: metadata_filter (JSONB) - Optional metadata containment filter (NULL for none)
-- $7: memory_types (TEXT[]) - Optional memory type filter (NULL for all types)
-- $8: min_score (FLOAT) - Minimum final score (applied BEFORE the final LIMIT)
-- $9: include_superseded (BOOLEAN) - When true, historical (superseded) revisions
--      are included; default false restricts to active facts only.

-- Note: Weights adjusted from hybrid search (0.40 semantic + 0.20 keyword + 0.25 decay + 0.15 importance)
-- Since no keyword matching, semantic weight absorbs keyword weight: 0.40 + 0.20 = 0.60

-- The inner query overfetches in index (distance) order; the outer query
-- computes the final score, filters by min_score, and re-sorts by score so
-- decay/importance can reorder within the candidate pool. Without the
-- overfetch, high-similarity but low-score rows would consume the LIMIT and
-- hide qualifying matches.

SELECT *
FROM (
    SELECT
        memory_id,
        agent_id,
        user_id,
        session_id,
        memory_type,
        fact AS content,  -- API compatibility
        fact,
        main_content,
        importance,
        metadata,
        created_at,
        last_accessed_at,
        access_count,
        lineage_id,
        revision,
        status,
        valid_from,
        valid_to,
        superseded_by_memory_id,
        superseded_at,
        -- Semantic similarity (GREATEST ensures 0-1 range even if embeddings not perfectly normalized)
        GREATEST(0, 1 - (embedding <=> $1::vector)) AS semantic_score,
        -- Time decay
        calculate_decay(last_accessed_at, $5) AS decay_score,
        -- Final score: semantic-focused
        (
            0.60 * GREATEST(0, 1 - (embedding <=> $1::vector)) +    -- semantic + absorbed keyword (0.60)
            0.25 * calculate_decay(last_accessed_at, $5) +          -- decay (0.25)
            0.15 * importance                                       -- importance (0.15)
        ) AS score
    FROM agent_memory
    WHERE agent_id = $2
        AND ($3::text IS NULL OR user_id = $3)
        AND ($6::jsonb IS NULL OR metadata @> $6::jsonb)
        AND ($7::text[] IS NULL OR memory_type = ANY($7))
        AND (
            $9::boolean
            OR (
                status <> 'superseded'
                AND COALESCE(metadata->>'status', 'active') <> 'superseded'
            )
        )
        AND embedding IS NOT NULL
        AND GREATEST(0, 1 - (embedding <=> $1::vector)) > 0.1  -- Early filter: min 10% similarity
    ORDER BY embedding <=> $1::vector  -- Index-optimized candidate ordering
    LIMIT GREATEST($4 * 3, 30)
) candidates
WHERE score >= $8
ORDER BY score DESC
LIMIT $4;
