-- ============================================================================
-- Hybrid Search Query
-- Combines vector similarity, keyword matching (via RRF), time decay, and importance
-- Uses RRF (Reciprocal Rank Fusion) for combining semantic and keyword rankings
-- ============================================================================

-- Parameters:
-- $1: query_embedding (VECTOR) - The query vector
-- $2: query_text (TEXT) - The query text for keyword search
-- $3: agent_id (TEXT) - Filter by agent
-- $4: user_id (TEXT) - Filter by user (optional, NULL for all)
-- $5: limit_count (INTEGER) - Number of results
-- $6: weight_semantic (FLOAT) - Weight for semantic score (direct similarity)
-- $7: weight_keyword (FLOAT) - Weight for RRF keyword boost
-- $8: weight_decay (FLOAT) - Weight for decay score
-- $9: weight_importance (FLOAT) - Weight for importance score
-- $10: decay_rate (FLOAT) - Decay rate per hour

WITH 
-- Vector similarity search with ranking
semantic_search AS (
    SELECT 
        memory_id,
        content,
        importance,
        metadata,
        created_at,
        last_accessed_at,
        -- Cosine similarity: 1 - cosine_distance (range roughly -1 to 1, usually 0 to 1 for similar content)
        GREATEST(0, 1 - (embedding <=> $1::vector)) AS semantic_score,
        -- Rank for RRF
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS semantic_rank
    FROM agent_memory
    WHERE agent_id = $3
        AND ($4::text IS NULL OR user_id = $4)
        AND embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT $5 * 3  -- Overfetch for RRF fusion
),

-- Keyword/BM25 search with ranking
keyword_search AS (
    SELECT 
        memory_id,
        ts_rank_cd(content_tsv, plainto_tsquery('english', $2)) AS keyword_score_raw,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $2)) DESC
        ) AS keyword_rank
    FROM agent_memory
    WHERE agent_id = $3
        AND ($4::text IS NULL OR user_id = $4)
        AND content_tsv @@ plainto_tsquery('english', $2)
    ORDER BY keyword_score_raw DESC
    LIMIT $5 * 3  -- Overfetch for RRF fusion
),

-- Combine results using RRF (Reciprocal Rank Fusion)
-- RRF formula: 1/(k + rank) where k=60 is standard
combined AS (
    SELECT 
        COALESCE(s.memory_id, k.memory_id) AS memory_id,
        COALESCE(s.content, k_mem.content) AS content,
        COALESCE(s.importance, k_mem.importance) AS importance,
        COALESCE(s.metadata, k_mem.metadata) AS metadata,
        COALESCE(s.created_at, k_mem.created_at) AS created_at,
        COALESCE(s.last_accessed_at, k_mem.last_accessed_at) AS last_accessed_at,
        
        -- Direct semantic similarity score (0-1 range)
        COALESCE(s.semantic_score, 0) AS semantic_score,
        
        -- RRF scores for ranking fusion (k=60)
        CASE WHEN s.semantic_rank IS NOT NULL 
            THEN 1.0 / (60 + s.semantic_rank) ELSE 0 END AS rrf_semantic,
        CASE WHEN k.keyword_rank IS NOT NULL 
            THEN 1.0 / (60 + k.keyword_rank) ELSE 0 END AS rrf_keyword
            
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.memory_id = k.memory_id
    -- Join to get content/metadata for keyword-only results
    LEFT JOIN agent_memory k_mem ON k.memory_id = k_mem.memory_id AND s.memory_id IS NULL
),

-- Calculate final scores with decay
scored AS (
    SELECT 
        memory_id,
        content,
        importance,
        metadata,
        created_at,
        last_accessed_at,
        semantic_score,
        -- Combined RRF score (sum of both rankings, scaled to ~0-1)
        -- Max RRF sum ≈ 2 * 1/61 ≈ 0.033, so multiply by 30 to normalize
        (rrf_semantic + rrf_keyword) * 30.0 AS rrf_combined,
        rrf_semantic,
        rrf_keyword,
        -- Time decay score (0-1)
        calculate_decay(last_accessed_at, $10) AS decay_score
    FROM combined
),

-- Apply final weighting formula from MVP plan:
-- weight_semantic * semantic_score + weight_keyword * rrf_combined + weight_decay * decay + weight_importance * importance
final_scored AS (
    SELECT 
        memory_id,
        content,
        importance,
        metadata,
        created_at,
        last_accessed_at,
        semantic_score,
        rrf_combined AS keyword_score,  -- The normalized RRF fusion score
        decay_score,
        -- Final combined score per MVP formula
        (
            $6 * semantic_score +           -- weight_semantic * semantic similarity
            $7 * rrf_combined +              -- weight_keyword * RRF fusion boost
            $8 * decay_score +               -- weight_decay * time decay
            $9 * importance                  -- weight_importance * importance
        ) AS combined_score
    FROM scored
)

SELECT 
    memory_id,
    content,
    importance,
    metadata,
    created_at,
    last_accessed_at,
    semantic_score,
    keyword_score,
    decay_score,
    combined_score AS score
FROM final_scored
WHERE combined_score > 0  -- Filter out zero-score results
ORDER BY combined_score DESC
LIMIT $5;
