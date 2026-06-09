-- Add durable long-running task memory records.

CREATE TABLE IF NOT EXISTS agent_task_runs (
    task_run_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    session_id TEXT REFERENCES agent_sessions(session_id) ON DELETE SET NULL,
    goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'failed', 'cancelled')),
    outcome TEXT,
    metadata JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_task_runs_agent ON agent_task_runs(agent_id);
CREATE INDEX IF NOT EXISTS idx_task_runs_user ON agent_task_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_task_runs_session ON agent_task_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_task_runs_status ON agent_task_runs(status);
CREATE INDEX IF NOT EXISTS idx_task_runs_started ON agent_task_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_task_runs_deleted ON agent_task_runs(deleted_at)
    WHERE deleted_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS agent_events (
    event_id TEXT PRIMARY KEY,
    task_run_id TEXT REFERENCES agent_task_runs(task_run_id) ON DELETE CASCADE,
    session_id TEXT REFERENCES agent_sessions(session_id) ON DELETE SET NULL,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'agent', 'tool', 'system')),
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'user_message',
            'assistant_message',
            'tool_call',
            'tool_result',
            'agent_action',
            'decision',
            'observation',
            'artifact',
            'error',
            'system_note'
        )
    ),
    content TEXT NOT NULL DEFAULT '',
    payload JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    redacted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_events_task_created
    ON agent_events(task_run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_session_created
    ON agent_events(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_agent_created
    ON agent_events(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type_created
    ON agent_events(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_deleted ON agent_events(deleted_at)
    WHERE deleted_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS agent_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    task_run_id TEXT NOT NULL REFERENCES agent_task_runs(task_run_id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    user_id TEXT REFERENCES users(user_id) ON DELETE SET NULL,
    summary TEXT NOT NULL,
    completed_steps JSONB DEFAULT '[]',
    pending_steps JSONB DEFAULT '[]',
    decisions JSONB DEFAULT '[]',
    blockers JSONB DEFAULT '[]',
    artifacts JSONB DEFAULT '[]',
    source_event_ids JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_task_created
    ON agent_checkpoints(task_run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_checkpoints_agent_created
    ON agent_checkpoints(agent_id, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL CHECK (job_type IN ('turn_ingest', 'checkpoint')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    payload JSONB DEFAULT '{}',
    error TEXT,
    locked_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_jobs_status_created
    ON memory_jobs(status, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_memory_jobs_locked
    ON memory_jobs(locked_until)
    WHERE status = 'processing';
