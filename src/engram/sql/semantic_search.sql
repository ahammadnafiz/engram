-- ============================================================================
-- Semantic Search Query (Optimized v2.2)
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
-- $10: weight_semantic_combined (FLOAT) - semantic weight + absorbed keyword
--      weight (default 0.40 + 0.20 = 0.60). Threaded from settings so the
--      configured search weights apply to semantic mode, not just hybrid.
-- $11: weight_decay (FLOAT) - decay weight (default 0.25)
-- $12: weight_importance (FLOAT) - importance weight (default 0.15)

-- Note: there is no keyword branch in semantic mode, so the semantic term
-- absorbs the keyword weight ($10 = weight_semantic + weight_keyword). At
-- default settings this reproduces the previous 0.60 / 0.25 / 0.15 split.

-- The inner query overfetches in index (distance) order. The middle layer
-- computes semantic_score and decay_score once each and filters by the 10%
-- minimum similarity. The outer layer applies min_score and re-sorts so
-- decay/importance can reorder within the candidate pool.

-- Memory-type-aware scoring: durable types (constraint, preference, profile,
-- decision) shift the non-semantic budget toward importance over decay so they
-- don't fade behind fresher but less relevant episodic noise. Ephemeral types
-- (episodic, tool_result) shift toward decay. The fractions always sum to 1.0
-- relative to (1 - $10), so the total score is bounded [0, 1] regardless of the
-- configured semantic weight.

SELECT *
FROM (
    SELECT
        *,
        (
            $10 * semantic_score
            + CASE memory_type
                WHEN 'constraint'  THEN (1.0 - $10) * (0.25 * decay_score + 0.75 * importance)
                WHEN 'preference'  THEN (1.0 - $10) * (0.25 * decay_score + 0.75 * importance)
                WHEN 'profile'     THEN (1.0 - $10) * (0.25 * decay_score + 0.75 * importance)
                WHEN 'decision'    THEN (1.0 - $10) * (0.25 * decay_score + 0.75 * importance)
                WHEN 'episodic'    THEN (1.0 - $10) * (0.75 * decay_score + 0.25 * importance)
                WHEN 'tool_result' THEN (1.0 - $10) * (0.65 * decay_score + 0.35 * importance)
                ELSE ($11 * decay_score + $12 * importance)
              END
        ) AS score
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
            -- Compute similarity and decay once; outer layers reference these aliases.
            GREATEST(0, 1 - (embedding <=> $1::vector)) AS semantic_score,
            calculate_decay(last_accessed_at, $5) AS decay_score
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
        ORDER BY embedding <=> $1::vector  -- Index-optimized candidate ordering
        LIMIT GREATEST($4 * 3, 30)
    ) pre
    WHERE pre.semantic_score > 0.1
) scored
WHERE score >= $8
ORDER BY score DESC
LIMIT $4;
