-- ============================================================================
-- Hybrid Search Query (Optimized v2.0)
-- Combines vector similarity, keyword matching (via RRF), time decay, and importance
-- Uses RRF (Reciprocal Rank Fusion) for combining semantic and keyword rankings
-- Two-column system: searches on fact column, returns both fact and main_content
-- ============================================================================

-- Parameters:
-- $1: query_embedding (VECTOR) - The query vector
-- $2: query_text (TEXT) - The query text for keyword search
-- $3: agent_id (TEXT) - Filter by agent
-- $4: user_id (TEXT) - Filter by user (optional, NULL for all)
-- $5: limit_count (INTEGER) - Number of results
-- $6: weight_semantic (FLOAT) - Weight for semantic score (default: 0.40)
-- $7: weight_keyword (FLOAT) - Weight for RRF keyword boost (default: 0.20)
-- $8: weight_decay (FLOAT) - Weight for decay score (default: 0.25)
-- $9: weight_importance (FLOAT) - Weight for importance score (default: 0.15)
-- $10: decay_rate (FLOAT) - Decay rate per hour (default: 0.995)
-- $11: metadata_filter (JSONB) - Optional metadata containment filter (NULL for none)
-- $12: memory_types (TEXT[]) - Optional memory type filter (NULL for all types)

WITH 
-- Semantic search on fact embeddings (2x overfetch for RRF, minimum 20)
semantic_search AS (
    SELECT
        memory_id,
        user_id,
        session_id,
        memory_type,
        fact,
        main_content,
        importance,
        metadata,
        created_at,
        last_accessed_at,
        access_count,
        -- Cosine similarity (GREATEST ensures 0-1 range even if embeddings not perfectly normalized)
        GREATEST(0, 1 - (embedding <=> $1::vector)) AS semantic_score,
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS semantic_rank
    FROM agent_memory
    WHERE agent_id = $3
        AND ($4::text IS NULL OR user_id = $4)
        AND ($11::jsonb IS NULL OR metadata @> $11::jsonb)
        AND ($12::text[] IS NULL OR memory_type = ANY($12))
        AND COALESCE(metadata->>'status', 'active') <> 'superseded'
        AND embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT GREATEST($5 * 2, 20)
),

-- Keyword search on fact_tsv (2x overfetch for RRF)
-- ts_rank with normalization flag 32 is faster than ts_rank_cd
keyword_search AS (
    SELECT 
        memory_id,
        ts_rank(fact_tsv, query, 32) AS keyword_score_raw,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fact_tsv, query, 32) DESC) AS keyword_rank
    FROM agent_memory,
         plainto_tsquery('english', $2) AS query
    WHERE agent_id = $3
        AND ($4::text IS NULL OR user_id = $4)
        AND ($11::jsonb IS NULL OR metadata @> $11::jsonb)
        AND ($12::text[] IS NULL OR memory_type = ANY($12))
        AND COALESCE(metadata->>'status', 'active') <> 'superseded'
        AND fact_tsv @@ query
    ORDER BY keyword_score_raw DESC
    LIMIT GREATEST($5 * 2, 20)
),

-- Optimized combination: LEFT JOIN + UNION ALL (faster than FULL OUTER JOIN)
combined AS (
    -- Semantic results with keyword boost
    SELECT
        s.memory_id,
        s.user_id,
        s.session_id,
        s.memory_type,
        s.fact,
        s.main_content,
        s.importance,
        s.metadata,
        s.created_at,
        s.last_accessed_at,
        s.access_count,
        s.semantic_score,
        s.semantic_rank,
        COALESCE(k.keyword_rank, 999999) AS keyword_rank,
        -- Pre-calculate RRF scores (k=60 standard)
        (1.0 / (60.0 + s.semantic_rank)) AS rrf_semantic,
        CASE WHEN k.keyword_rank IS NOT NULL 
            THEN (1.0 / (60.0 + k.keyword_rank))
            ELSE 0.0 
        END AS rrf_keyword
    FROM semantic_search s
    LEFT JOIN keyword_search k USING (memory_id)
    
    UNION ALL
    
    -- Keyword-only results (not in semantic search)
    SELECT
        k.memory_id,
        m.user_id,
        m.session_id,
        m.memory_type,
        m.fact,
        m.main_content,
        m.importance,
        m.metadata,
        m.created_at,
        m.last_accessed_at,
        m.access_count,
        0.0 AS semantic_score,
        999999 AS semantic_rank,
        k.keyword_rank,
        0.0 AS rrf_semantic,
        (1.0 / (60.0 + k.keyword_rank)) AS rrf_keyword
    FROM keyword_search k
    LEFT JOIN semantic_search s USING (memory_id)
    JOIN agent_memory m ON k.memory_id = m.memory_id
    WHERE s.memory_id IS NULL
),

-- Single-pass final scoring
final_scored AS (
    SELECT
        memory_id,
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
        semantic_score,
        (rrf_semantic + rrf_keyword) * 30.0 AS keyword_score,  -- Normalized RRF
        calculate_decay(last_accessed_at, $10) AS decay_score,
        -- Final score: weighted combination
        (
            $6 * semantic_score +                           -- semantic weight (0.40)
            $7 * (rrf_semantic + rrf_keyword) * 30.0 +      -- keyword weight (0.20)
            $8 * calculate_decay(last_accessed_at, $10) +   -- decay weight (0.25)
            $9 * importance                                 -- importance weight (0.15)
        ) AS combined_score
    FROM combined
    WHERE 
        -- Early filtering: minimum semantic OR keyword match
        (semantic_score > 0.1 OR rrf_keyword > 0.0)
)

-- Final results: hybrid search output
SELECT
    memory_id,
    user_id,
    session_id,
    memory_type,
    content,
    fact,
    main_content,
    importance,
    metadata,
    created_at,
    last_accessed_at,
    access_count,
    semantic_score,
    keyword_score,
    decay_score,
    combined_score AS score
FROM final_scored
WHERE combined_score > 0
ORDER BY combined_score DESC
LIMIT $5;
