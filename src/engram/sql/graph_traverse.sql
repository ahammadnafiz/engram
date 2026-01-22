-- ============================================================================
-- Graph Traversal Query
-- Multi-hop traversal using recursive CTE
-- ============================================================================

-- Parameters:
-- $1: start_memory_id (TEXT) - Starting memory ID
-- $2: max_depth (INTEGER) - Maximum traversal depth
-- $3: relation_types (TEXT[]) - Filter by relation types (NULL for all)
-- $4: direction (TEXT) - 'outbound', 'inbound', or 'any'
-- $5: min_weight (FLOAT) - Minimum relation weight threshold
-- $6: limit_count (INTEGER) - Maximum results per depth level

WITH RECURSIVE traversal AS (
    -- Base case: starting node
    SELECT 
        m.memory_id,
        m.content,
        m.importance,
        m.metadata,
        m.created_at,
        m.last_accessed_at,
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
        m.content,
        m.importance,
        m.metadata,
        m.created_at,
        m.last_accessed_at,
        t.depth + 1,
        t.path || m.memory_id,
        r.relation_type,
        t.path_weight * r.weight AS path_weight
    FROM traversal t
    JOIN memory_relations r ON (
        -- Handle direction
        CASE $4
            WHEN 'outbound' THEN r.source_memory_id = t.memory_id
            WHEN 'inbound' THEN r.target_memory_id = t.memory_id
            ELSE r.source_memory_id = t.memory_id OR r.target_memory_id = t.memory_id
        END
    )
    JOIN agent_memory m ON (
        CASE $4
            WHEN 'outbound' THEN m.memory_id = r.target_memory_id
            WHEN 'inbound' THEN m.memory_id = r.source_memory_id
            ELSE m.memory_id = CASE 
                WHEN r.source_memory_id = t.memory_id THEN r.target_memory_id
                ELSE r.source_memory_id
            END
        END
    )
    WHERE t.depth < $2
        AND NOT m.memory_id = ANY(t.path)  -- Prevent cycles
        AND r.weight >= $5
        AND ($3::text[] IS NULL OR r.relation_type = ANY($3))
)

SELECT 
    memory_id,
    content,
    importance,
    metadata,
    created_at,
    last_accessed_at,
    depth,
    path,
    relation_type,
    path_weight,
    -- Score combining path weight and importance
    (path_weight * 0.7 + importance * 0.3) AS score
FROM traversal
WHERE depth > 0  -- Exclude starting node
ORDER BY depth, path_weight DESC
LIMIT $6;
