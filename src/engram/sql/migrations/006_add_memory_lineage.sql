-- ============================================================================
-- Migration: Add first-class memory lineage/current-head state
-- Version: 006
-- Description:
--   - Keep agent_memory as the active read model.
--   - Preserve corrected facts as superseded revisions.
--   - Track current lineage heads without scanning JSON metadata.
-- ============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS memory_lineages (
    lineage_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    conflict_key TEXT,
    current_memory_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_lineage_conflict_key
    ON memory_lineages(agent_id, COALESCE(user_id, ''), conflict_key)
    WHERE conflict_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_lineage_current
    ON memory_lineages(current_memory_id)
    WHERE current_memory_id IS NOT NULL;

ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS lineage_id TEXT;
ALTER TABLE agent_memory
    ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 1;
ALTER TABLE agent_memory
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE agent_memory
    ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ;
ALTER TABLE agent_memory
    ADD COLUMN IF NOT EXISTS superseded_by_memory_id TEXT;
ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ;

ALTER TABLE agent_memory DROP CONSTRAINT IF EXISTS agent_memory_lineage_id_fkey;

UPDATE agent_memory
SET status = CASE
        WHEN metadata->>'status' = 'superseded' THEN 'superseded'
        ELSE 'active'
    END
WHERE status IS NULL OR status NOT IN ('active', 'superseded');

UPDATE agent_memory m
SET superseded_by_memory_id = metadata->>'superseded_by'
WHERE superseded_by_memory_id IS NULL
    AND metadata ? 'superseded_by'
    AND EXISTS (
        SELECT 1
        FROM agent_memory winner
        WHERE winner.memory_id = m.metadata->>'superseded_by'
    );

UPDATE agent_memory
SET superseded_at = NOW()
WHERE superseded_at IS NULL AND status = 'superseded';

UPDATE agent_memory
SET valid_from = created_at
WHERE valid_from IS NULL;

UPDATE agent_memory
SET valid_to = superseded_at
WHERE valid_to IS NULL AND status = 'superseded';

UPDATE agent_memory
SET lineage_id = COALESCE(
        metadata->>'lineage_id',
        CASE
            WHEN metadata ? 'conflict_key' THEN
                'lin_' || md5(
                    agent_id || chr(31) || COALESCE(user_id, '') ||
                    chr(31) || (metadata->>'conflict_key')
                )
            ELSE memory_id
        END
    )
WHERE lineage_id IS NULL;

WITH ranked AS (
    SELECT
        memory_id,
        ROW_NUMBER() OVER (
            PARTITION BY lineage_id
            ORDER BY created_at ASC, memory_id ASC
        ) AS version_number
    FROM agent_memory
    WHERE lineage_id IS NOT NULL
)
UPDATE agent_memory m
SET revision = ranked.version_number
FROM ranked
WHERE m.memory_id = ranked.memory_id
    AND (m.revision IS NULL OR m.revision = 1);

WITH heads AS (
    SELECT DISTINCT ON (lineage_id)
        lineage_id,
        agent_id,
        user_id,
        metadata->>'conflict_key' AS conflict_key,
        memory_id AS current_memory_id,
        created_at
    FROM agent_memory
    WHERE lineage_id IS NOT NULL
    ORDER BY
        lineage_id,
        (status = 'active') DESC,
        revision DESC,
        created_at DESC,
        memory_id DESC
)
INSERT INTO memory_lineages (
    lineage_id, agent_id, user_id, conflict_key, current_memory_id, created_at
)
SELECT lineage_id, agent_id, user_id, conflict_key, current_memory_id, created_at
FROM heads
ON CONFLICT (lineage_id) DO UPDATE
SET current_memory_id = COALESCE(
        memory_lineages.current_memory_id,
        EXCLUDED.current_memory_id
    ),
    updated_at = NOW();

UPDATE agent_memory
SET metadata = metadata
    || jsonb_build_object(
        'status', status,
        'lineage_id', lineage_id,
        'revision', revision
    );

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'agent_memory_revision_check'
    ) THEN
        ALTER TABLE agent_memory
            ADD CONSTRAINT agent_memory_revision_check CHECK (revision >= 1);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'agent_memory_status_check'
    ) THEN
        ALTER TABLE agent_memory
            ADD CONSTRAINT agent_memory_status_check
            CHECK (status IN ('active', 'superseded'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'agent_memory_superseded_by_memory_id_fkey'
    ) THEN
        ALTER TABLE agent_memory
            ADD CONSTRAINT agent_memory_superseded_by_memory_id_fkey
            FOREIGN KEY (superseded_by_memory_id)
            REFERENCES agent_memory(memory_id)
            ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_memory_lineage_revision
    ON agent_memory(lineage_id, revision DESC)
    WHERE lineage_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_status
    ON agent_memory(agent_id, COALESCE(user_id, ''), status);

DROP INDEX IF EXISTS idx_unique_memory_fact;
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_memory_fact
    ON agent_memory(agent_id, COALESCE(user_id, ''), md5(fact))
    WHERE status = 'active';

INSERT INTO memory_relations (
    source_memory_id, target_memory_id, relation_type, weight, metadata
)
SELECT
    superseded_by_memory_id,
    memory_id,
    'supersedes',
    1.0,
    jsonb_build_object('source', 'migration_006')
FROM agent_memory
WHERE superseded_by_memory_id IS NOT NULL
ON CONFLICT (source_memory_id, target_memory_id, relation_type) DO NOTHING;

COMMIT;
