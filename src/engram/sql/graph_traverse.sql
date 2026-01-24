-- ============================================================================
-- Graph Traversal Query (Optimized v2.0)
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

WITH RECURSIVE traversal AS (
    -- Base case: starting node
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
    
    -- Recursive case: follow relations
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
        ($4 = 'inbound' AND r.target_memory_id = t.memory_id) OR
        ($4 = 'any' AND (r.source_memory_id = t.memory_id OR r.target_memory_id = t.memory_id))
    )
    JOIN agent_memory m ON (
        m.memory_id = CASE 
            WHEN $4 = 'outbound' THEN r.target_memory_id
            WHEN $4 = 'inbound' THEN r.source_memory_id
            WHEN r.source_memory_id = t.memory_id THEN r.target_memory_id
            ELSE r.source_memory_id
        END
    )
    WHERE t.depth < $2
        AND NOT (m.memory_id = ANY(t.path))  -- Prevent cycles
        AND r.weight >= $5
        AND ($3::text[] IS NULL OR r.relation_type = ANY($3))
)

-- Final results: graph traversal output
SELECT 
    memory_id,
    content,
    fact,
    main_content,
    importance,
    metadata,
    created_at,
    last_accessed_at,
    access_count,
    depth,
    path,
    relation_type,
    path_weight,
    -- Score: path weight (70%) + importance (30%)
    (path_weight * 0.7 + importance * 0.3) AS score
FROM traversal
WHERE depth > 0  -- Exclude starting node
ORDER BY 
    depth ASC,           -- Closer nodes first
    path_weight DESC,    -- Higher weight paths first
    importance DESC      -- Higher importance as tiebreaker
LIMIT $6;
