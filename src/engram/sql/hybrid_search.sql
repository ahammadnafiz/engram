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
-- $13: min_score (FLOAT) - Minimum combined score (applied BEFORE the final
--      LIMIT so qualifying matches beyond the first page are not lost)
-- $14: text_search_config (TEXT) - Text search configuration name (must match
--      the configuration of the generated fact_tsv column)
-- $15: candidate_multiplier (INTEGER) - Overfetch factor per branch before
--      rank fusion (default: 5). Higher improves recall completeness.
-- $16: include_superseded (BOOLEAN) - When true, historical (superseded)
--      revisions are included; default false restricts to active facts only.

WITH
-- Semantic search on fact embeddings (overfetch for RRF, minimum 20)
semantic_search AS (
    SELECT
        memory_id,
        agent_id,
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
        lineage_id,
        revision,
        status,
        valid_from,
        valid_to,
        superseded_by_memory_id,
        superseded_at,
        -- Cosine similarity (GREATEST ensures 0-1 range even if embeddings not perfectly normalized)
        GREATEST(0, 1 - (embedding <=> $1::vector)) AS semantic_score,
        ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS semantic_rank
    FROM agent_memory
    WHERE agent_id = $3
        AND ($4::text IS NULL OR user_id = $4)
        AND ($11::jsonb IS NULL OR metadata @> $11::jsonb)
        AND ($12::text[] IS NULL OR memory_type = ANY($12))
        AND (
            $16::boolean
            OR (
                status <> 'superseded'
                AND COALESCE(metadata->>'status', 'active') <> 'superseded'
            )
        )
        AND embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT GREATEST($5::int * $15::int, 20)
),

-- Keyword search on fact_tsv (overfetch for RRF)
-- ts_rank with normalization flag 32 is faster than ts_rank_cd
keyword_search AS (
    SELECT
        memory_id,
        agent_id,
        ts_rank(fact_tsv, query, 32) AS keyword_score_raw,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fact_tsv, query, 32) DESC) AS keyword_rank
    FROM agent_memory,
         plainto_tsquery($14::regconfig, $2) AS query
    WHERE agent_id = $3
        AND ($4::text IS NULL OR user_id = $4)
        AND ($11::jsonb IS NULL OR metadata @> $11::jsonb)
        AND ($12::text[] IS NULL OR memory_type = ANY($12))
        AND (
            $16::boolean
            OR (
                status <> 'superseded'
                AND COALESCE(metadata->>'status', 'active') <> 'superseded'
            )
        )
        AND fact_tsv @@ query
    ORDER BY keyword_score_raw DESC
    LIMIT GREATEST($5::int * $15::int, 20)
),

-- Optimized combination: LEFT JOIN + UNION ALL (faster than FULL OUTER JOIN)
combined AS (
    -- Semantic results with keyword boost
    SELECT
        s.memory_id,
        s.agent_id,
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
        s.lineage_id,
        s.revision,
        s.status,
        s.valid_from,
        s.valid_to,
        s.superseded_by_memory_id,
        s.superseded_at,
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
        m.agent_id,
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
        m.lineage_id,
        m.revision,
        m.status,
        m.valid_from,
        m.valid_to,
        m.superseded_by_memory_id,
        m.superseded_at,
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

-- Two-pass final scoring: pre_score computes decay and keyword once each so the
-- combined_score formula can reference them without re-evaluation. Memory-type-
-- aware weights shift the non-semantic budget (1 - $6 - $7 = 0.40 at defaults)
-- between decay and importance per type. Durable types (constraint, preference,
-- profile, decision) weight importance over decay. Ephemeral types (episodic,
-- tool_result) weight decay over importance. Fractions always sum to 1.0
-- relative to (1 - $6 - $7), so the total score stays bounded [0, 1].
final_scored AS (
    SELECT
        *,
        (
            $6 * semantic_score
            + $7 * keyword_score
            + CASE memory_type
                WHEN 'constraint'  THEN (1.0 - $6 - $7) * (0.25 * decay_score + 0.75 * importance)
                WHEN 'preference'  THEN (1.0 - $6 - $7) * (0.25 * decay_score + 0.75 * importance)
                WHEN 'profile'     THEN (1.0 - $6 - $7) * (0.25 * decay_score + 0.75 * importance)
                WHEN 'decision'    THEN (1.0 - $6 - $7) * (0.25 * decay_score + 0.75 * importance)
                WHEN 'episodic'    THEN (1.0 - $6 - $7) * (0.75 * decay_score + 0.25 * importance)
                WHEN 'tool_result' THEN (1.0 - $6 - $7) * (0.65 * decay_score + 0.35 * importance)
                ELSE ($8 * decay_score + $9 * importance)
              END
        ) AS combined_score
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
            semantic_score,
            -- Rescale RRF into the same 0-1 range as semantic_score so the
            -- configured weights ($6..$9) combine on a common scale.
            -- RRF per branch peaks at 1/(60+1) ≈ 0.0164 (k=60, rank 1); both
            -- branches together ≈ 0.0328. Multiplying by 30 maps that to ≈ 0.98,
            -- i.e. a rank-1-in-both match scores ~1.0 on the keyword term, matching
            -- a perfect 1.0 cosine on the semantic term. The factor is the inverse
            -- of the peak two-branch RRF (≈ 1/0.0328); it is not arbitrary tuning.
            (rrf_semantic + rrf_keyword) * 30.0 AS keyword_score,
            calculate_decay(last_accessed_at, $10) AS decay_score
        FROM combined
        WHERE
            -- Early filtering: minimum semantic OR keyword match
            (semantic_score > 0.1 OR rrf_keyword > 0.0)
    ) pre_score
)

-- Final results: hybrid search output
SELECT
    memory_id,
    agent_id,
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
    lineage_id,
    revision,
    status,
    valid_from,
    valid_to,
    superseded_by_memory_id,
    superseded_at,
    semantic_score,
    keyword_score,
    decay_score,
    combined_score AS score
FROM final_scored
WHERE combined_score > 0
    AND combined_score >= $13
ORDER BY combined_score DESC
LIMIT $5;
