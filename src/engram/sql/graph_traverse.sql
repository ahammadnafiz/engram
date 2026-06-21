-- ============================================================================
-- Graph Traversal Query (Optimized v2.1)
-- Multi-hop memory relationship traversal using recursive CTE
-- Explores connected memories through explicit relationships
-- Two-column system: returns both fact and main_content for each node
-- ============================================================================

-- Parameters:
-- $1: start_memory_id (TEXT) - Starting memory ID
-- $2: max_depth (INTEGER) - Maximum traversal depth (recommend 2-3)
-- $3: relation_types (TEXT[]) - Filter by relation types (NULL for all)
--     Common types: 'related_to', 'causes', 'supports', 'contradicts', 'part_of'
-- $4: direction (TEXT) - 'outbound', 'inbound', or 'any'
-- $5: min_weight (FLOAT) - Minimum relation weight threshold (0.0-1.0)
-- $6: limit_count (INTEGER) - Maximum results
-- $7: query_embedding (VECTOR) - Query vector for relevance scoring.
--     Pass NULL to disable query-aware scoring (falls back to v2.0 weights).

-- Embeddings are NOT carried through the recursive CTE to avoid propagating
-- large vectors (384-1536 floats) through every recursion level. Instead, a
-- final JOIN fetches embeddings once per result node for score computation.

-- Score formula:
--   With query_embedding:  path_weight * 0.5 + node_similarity * 0.3 + importance * 0.2
--   Without query_embedding: path_weight * 0.7 + importance * 0.3
-- The query-aware formula penalises topically unrelated nodes that happen to
-- be connected through high-weight edges, keeping expansions on-topic.

WITH RECURSIVE traversal AS (
    -- Base case: starting node (no embedding — fetched at scoring time)
    SELECT
        m.memory_id,
        m.fact AS content,
        m.fact,
        m.main_content,
        m.importance,
        m.metadata,
        m.created_at,
        m.last_accessed_at,
        m.access_count,
        0 AS depth,
        ARRAY[m.memory_id] AS path,
        NULL::text AS relation_type,
        1.0::float AS path_weight
    FROM agent_memory m
    WHERE m.memory_id = $1

    UNION ALL

    -- Recursive case: follow relations (no embedding in traversal state)
    SELECT
        m.memory_id,
        m.fact AS content,
        m.fact,
        m.main_content,
        m.importance,
        m.metadata,
        m.created_at,
        m.last_accessed_at,
        m.access_count,
        t.depth + 1,
        t.path || m.memory_id,
        r.relation_type,
        t.path_weight * r.weight AS path_weight
    FROM traversal t
    JOIN memory_relations r ON (
        -- Optimized direction handling using boolean expressions
        ($4 = 'outbound' AND r.source_memory_id = t.memory_id) OR
        ($4 = 'inbound'  AND r.target_memory_id = t.memory_id) OR
        ($4 = 'any' AND (r.source_memory_id = t.memory_id OR r.target_memory_id = t.memory_id))
    )
    JOIN agent_memory m ON (
        m.memory_id = CASE
            WHEN $4 = 'outbound' THEN r.target_memory_id
            WHEN $4 = 'inbound'  THEN r.source_memory_id
            WHEN r.source_memory_id = t.memory_id THEN r.target_memory_id
            ELSE r.source_memory_id
        END
    )
    WHERE t.depth < $2
        AND NOT (m.memory_id = ANY(t.path))  -- Prevent cycles
        AND r.weight >= $5
        AND ($3::text[] IS NULL OR r.relation_type = ANY($3))
)

-- Score is computed in the inner SELECT (wrapping in an outer SELECT lets
-- ORDER BY reference the alias without re-evaluating the expression).
SELECT *
FROM (
    SELECT
        t.memory_id,
        t.content,
        t.fact,
        t.main_content,
        t.importance,
        t.metadata,
        t.created_at,
        t.last_accessed_at,
        t.access_count,
        t.depth,
        t.path,
        t.relation_type,
        t.path_weight,
        CASE
            WHEN $7::vector IS NOT NULL AND m.embedding IS NOT NULL
                THEN t.path_weight * 0.5
                     + GREATEST(0, 1 - (m.embedding <=> $7::vector)) * 0.3
                     + t.importance * 0.2
            ELSE t.path_weight * 0.7 + t.importance * 0.3
        END AS score
    FROM traversal t
    JOIN agent_memory m ON t.memory_id = m.memory_id
    WHERE t.depth > 0  -- Exclude starting node
) scored
ORDER BY score DESC, depth ASC
LIMIT $6;
