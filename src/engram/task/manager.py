"""Task memory persistence for long-running agents."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from engram.core.exceptions import EngramError, StorageError
from engram.core.serialization import json_dumps
from engram.task.models import (
    AgentEvent,
    EventCreate,
    MemoryJob,
    TaskCheckpoint,
    TaskRun,
    TaskRunStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from engram.core._types import AgentId, Metadata, SessionId, UserId
    from engram.storage.postgres import PostgresStorage


_INSERT_EVENT_SQL = """
INSERT INTO agent_events (
    event_id, task_run_id, session_id, agent_id, user_id,
    role, event_type, content, payload, metadata, created_at,
    deleted_at, redacted_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
"""

_INSERT_JOB_SQL = """
INSERT INTO memory_jobs (
    job_id, job_type, status, attempts, payload, error,
    locked_until, created_at, updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
"""


class TaskNotFoundError(EngramError):
    """Raised when a task run cannot be found."""


class EventNotFoundError(EngramError):
    """Raised when an event cannot be found."""


class MemoryJobNotFoundError(EngramError):
    """Raised when a memory job cannot be found."""


class TaskMemoryManager:
    """Storage operations for task runs, event ledger, checkpoints, and jobs."""

    def __init__(self, storage: PostgresStorage) -> None:
        self._storage = storage

    async def _ensure_agent_exists(self, agent_id: AgentId) -> None:
        await self._storage.execute(
            """
            INSERT INTO agents (agent_id, name)
            VALUES ($1, $2)
            ON CONFLICT (agent_id) DO NOTHING
            """,
            agent_id,
            agent_id,
        )

    async def _ensure_user_exists(self, user_id: UserId) -> None:
        await self._storage.execute(
            """
            INSERT INTO users (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
            """,
            user_id,
        )

    async def start_task(
        self,
        *,
        goal: str,
        agent_id: AgentId,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        metadata: Metadata | None = None,
    ) -> TaskRun:
        """Create a task run."""
        await self._ensure_agent_exists(agent_id)
        if user_id:
            await self._ensure_user_exists(user_id)

        task = TaskRun(
            goal=goal,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        try:
            await self._storage.execute(
                """
                INSERT INTO agent_task_runs (
                    task_run_id, agent_id, user_id, session_id, goal, status,
                    outcome, metadata, started_at, ended_at, updated_at, deleted_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                task.task_run_id,
                task.agent_id,
                task.user_id,
                task.session_id,
                task.goal,
                task.status,
                task.outcome,
                json_dumps(task.metadata),
                task.started_at,
                task.ended_at,
                task.updated_at,
                task.deleted_at,
            )
            return task
        except Exception as e:
            raise StorageError(f"Failed to start task: {e}") from e

    async def get_task(
        self, task_run_id: str, *, include_deleted: bool = False
    ) -> TaskRun:
        condition = "" if include_deleted else "AND deleted_at IS NULL"
        row = await self._storage.fetchone(
            f"""
            SELECT task_run_id, agent_id, user_id, session_id, goal, status,
                   outcome, metadata, started_at, ended_at, updated_at, deleted_at
            FROM agent_task_runs
            WHERE task_run_id = $1 {condition}
            """,
            task_run_id,
        )
        if row is None:
            raise TaskNotFoundError(f"Task not found: {task_run_id}")
        return self._row_to_task(row)

    async def list_tasks(
        self,
        *,
        agent_id: AgentId | None = None,
        user_id: UserId | None = None,
        status: TaskRunStatus | list[TaskRunStatus] | None = None,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> list[TaskRun]:
        """List task runs for resuming or inspecting long-running work."""
        conditions = []
        params: list[Any] = []
        idx = 1
        if agent_id:
            conditions.append(f"agent_id = ${idx}")
            params.append(agent_id)
            idx += 1
        if user_id:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if status:
            statuses = [status] if isinstance(status, str) else status
            conditions.append(f"status = ANY(${idx}::text[])")
            params.append(statuses)
            idx += 1
        if not include_deleted:
            conditions.append("deleted_at IS NULL")
        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)
        rows = await self._storage.fetchall(
            f"""
            SELECT task_run_id, agent_id, user_id, session_id, goal, status,
                   outcome, metadata, started_at, ended_at, updated_at, deleted_at
            FROM agent_task_runs
            WHERE {where}
            ORDER BY updated_at DESC, started_at DESC
            LIMIT ${idx}
            """,
            *params,
        )
        return [self._row_to_task(row) for row in rows]

    async def set_task_status(
        self,
        task_run_id: str,
        status: TaskRunStatus,
        *,
        outcome: str | None = None,
    ) -> TaskRun:
        """Change a task's status.

        Terminal statuses (completed/failed/cancelled) are final: setting the
        same terminal status again is an idempotent no-op, but transitioning
        out of a terminal status raises EngramError.
        """
        ended_expr = (
            "NOW()" if status in {"completed", "failed", "cancelled"} else "ended_at"
        )
        row = await self._storage.fetchone(
            f"""
            UPDATE agent_task_runs
            SET status = $2,
                outcome = COALESCE($3, outcome),
                ended_at = {ended_expr},
                updated_at = NOW()
            WHERE task_run_id = $1 AND deleted_at IS NULL
                AND (
                    status NOT IN ('completed', 'failed', 'cancelled')
                    OR status = $2
                )
            RETURNING task_run_id, agent_id, user_id, session_id, goal, status,
                      outcome, metadata, started_at, ended_at, updated_at, deleted_at
            """,
            task_run_id,
            status,
            outcome,
        )
        if row is None:
            # Distinguish missing task from an invalid transition
            task = await self.get_task(task_run_id)  # raises TaskNotFoundError
            raise EngramError(
                f"Task {task_run_id} is terminal ({task.status}); "
                f"cannot transition to {status}",
                task_run_id=task_run_id,
                current_status=task.status,
                requested_status=status,
            )
        return self._row_to_task(row)

    async def soft_delete_task(self, task_run_id: str) -> TaskRun:
        row = await self._storage.fetchone(
            """
            UPDATE agent_task_runs
            SET deleted_at = COALESCE(deleted_at, NOW()), updated_at = NOW()
            WHERE task_run_id = $1
            RETURNING task_run_id, agent_id, user_id, session_id, goal, status,
                      outcome, metadata, started_at, ended_at, updated_at, deleted_at
            """,
            task_run_id,
        )
        if row is None:
            raise TaskNotFoundError(f"Task not found: {task_run_id}")
        await self._storage.execute(
            """
            UPDATE agent_events
            SET deleted_at = COALESCE(deleted_at, NOW())
            WHERE task_run_id = $1
            """,
            task_run_id,
        )
        return self._row_to_task(row)

    async def record_event(self, create: EventCreate) -> AgentEvent:
        """Append an immutable event to the ledger."""
        await self._ensure_agent_exists(create.agent_id)
        if create.user_id:
            await self._ensure_user_exists(create.user_id)

        event = AgentEvent(
            task_run_id=create.task_run_id,
            session_id=create.session_id,
            agent_id=create.agent_id,
            user_id=create.user_id,
            role=create.role,
            event_type=create.event_type,
            content=create.content,
            payload=create.payload,
            metadata=create.metadata,
        )
        try:
            await self._storage.execute(*self._event_insert_args(event))
            return event
        except Exception as e:
            raise StorageError(f"Failed to record event: {e}") from e

    def _event_insert_args(self, event: AgentEvent) -> tuple[Any, ...]:
        return (
            _INSERT_EVENT_SQL,
            event.event_id,
            event.task_run_id,
            event.session_id,
            event.agent_id,
            event.user_id,
            event.role,
            event.event_type,
            event.content,
            json_dumps(event.payload),
            json_dumps(event.metadata),
            event.created_at,
            event.deleted_at,
            event.redacted_at,
        )

    async def record_events(
        self,
        creates: list[EventCreate],
        *,
        job_type: str | None = None,
        job_payload: Callable[[list[AgentEvent]], dict[str, Any]] | None = None,
    ) -> tuple[list[AgentEvent], MemoryJob | None]:
        """Record several events — and optionally a derivation job — atomically.

        Used by record_turn(): a crash between writing the turn's events and
        enqueueing its ingestion job would otherwise leave a turn that is
        recorded but never processed (or vice versa).

        Args:
            creates: Events to record, in order.
            job_type: Optional memory job type to enqueue with the events.
            job_payload: Builds the job payload from the created events
                (their event_ids are assigned before insert).

        Returns:
            (events, job) — job is None when job_type wasn't given.
        """
        if not creates:
            return [], None

        for agent_id in {c.agent_id for c in creates}:
            await self._ensure_agent_exists(agent_id)
        for user_id in {c.user_id for c in creates if c.user_id}:
            await self._ensure_user_exists(user_id)

        events = [
            AgentEvent(
                task_run_id=create.task_run_id,
                session_id=create.session_id,
                agent_id=create.agent_id,
                user_id=create.user_id,
                role=create.role,
                event_type=create.event_type,
                content=create.content,
                payload=create.payload,
                metadata=create.metadata,
            )
            for create in creates
        ]

        job: MemoryJob | None = None
        if job_type is not None:
            payload = job_payload(events) if job_payload is not None else {}
            job = MemoryJob(job_type=job_type, payload=payload)  # type: ignore[arg-type]

        try:
            async with self._storage.transaction() as conn:
                for event in events:
                    await conn.execute(*self._event_insert_args(event))
                if job is not None:
                    await conn.execute(
                        _INSERT_JOB_SQL,
                        job.job_id,
                        job.job_type,
                        job.status,
                        job.attempts,
                        json_dumps(job.payload),
                        job.error,
                        job.locked_until,
                        job.created_at,
                        job.updated_at,
                    )
            return events, job
        except Exception as e:
            raise StorageError(f"Failed to record events atomically: {e}") from e

    async def list_events(
        self,
        *,
        task_run_id: str | None = None,
        session_id: SessionId | None = None,
        agent_id: AgentId | None = None,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> list[AgentEvent]:
        """List recent events in chronological order."""
        conditions = []
        params: list[Any] = []
        idx = 1
        if task_run_id:
            conditions.append(f"task_run_id = ${idx}")
            params.append(task_run_id)
            idx += 1
        if session_id:
            conditions.append(f"session_id = ${idx}")
            params.append(session_id)
            idx += 1
        if agent_id:
            conditions.append(f"agent_id = ${idx}")
            params.append(agent_id)
            idx += 1
        if not include_deleted:
            conditions.append("deleted_at IS NULL")
        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)
        rows = await self._storage.fetchall(
            f"""
            WITH recent AS (
                SELECT event_id, task_run_id, session_id, agent_id, user_id,
                       role, event_type, content, payload, metadata, created_at,
                       deleted_at, redacted_at
                FROM agent_events
                WHERE {where}
                ORDER BY created_at DESC, event_id DESC
                LIMIT ${idx}
            )
            SELECT * FROM recent ORDER BY created_at ASC, event_id ASC
            """,
            *params,
        )
        return [self._row_to_event(row) for row in rows]

    async def redact_event(self, event_id: str) -> AgentEvent:
        row = await self._storage.fetchone(
            """
            UPDATE agent_events
            SET content = '[REDACTED]',
                payload = '{}'::jsonb,
                metadata = metadata || '{"redacted": true}'::jsonb,
                redacted_at = COALESCE(redacted_at, NOW())
            WHERE event_id = $1 AND deleted_at IS NULL
            RETURNING event_id, task_run_id, session_id, agent_id, user_id,
                      role, event_type, content, payload, metadata, created_at,
                      deleted_at, redacted_at
            """,
            event_id,
        )
        if row is None:
            raise EventNotFoundError(f"Event not found: {event_id}")
        return self._row_to_event(row)

    async def create_checkpoint(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        try:
            await self._storage.execute(
                """
                INSERT INTO agent_checkpoints (
                    checkpoint_id, task_run_id, agent_id, user_id, summary,
                    completed_steps, pending_steps, decisions, blockers,
                    artifacts, source_event_ids, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                checkpoint.checkpoint_id,
                checkpoint.task_run_id,
                checkpoint.agent_id,
                checkpoint.user_id,
                checkpoint.summary,
                json_dumps(checkpoint.completed_steps),
                json_dumps(checkpoint.pending_steps),
                json_dumps(checkpoint.decisions),
                json_dumps(checkpoint.blockers),
                json_dumps(checkpoint.artifacts),
                json_dumps(checkpoint.source_event_ids),
                json_dumps(checkpoint.metadata),
                checkpoint.created_at,
            )
            return checkpoint
        except Exception as e:
            raise StorageError(f"Failed to create checkpoint: {e}") from e

    async def list_checkpoints(
        self,
        task_run_id: str,
        *,
        limit: int = 3,
    ) -> list[TaskCheckpoint]:
        rows = await self._storage.fetchall(
            """
            SELECT checkpoint_id, task_run_id, agent_id, user_id, summary,
                   completed_steps, pending_steps, decisions, blockers,
                   artifacts, source_event_ids, metadata, created_at
            FROM agent_checkpoints
            WHERE task_run_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            task_run_id,
            limit,
        )
        return [self._row_to_checkpoint(row) for row in rows]

    async def latest_checkpoint(self, task_run_id: str) -> TaskCheckpoint | None:
        checkpoints = await self.list_checkpoints(task_run_id, limit=1)
        return checkpoints[0] if checkpoints else None

    async def enqueue_job(
        self,
        job_type: str,
        payload: dict[str, Any],
    ) -> MemoryJob:
        job = MemoryJob(job_type=job_type, payload=payload)  # type: ignore[arg-type]
        await self._storage.execute(
            _INSERT_JOB_SQL,
            job.job_id,
            job.job_type,
            job.status,
            job.attempts,
            json_dumps(job.payload),
            job.error,
            job.locked_until,
            job.created_at,
            job.updated_at,
        )
        return job

    async def claim_jobs(
        self, *, limit: int = 10, lock_seconds: int = 300, max_attempts: int = 5
    ) -> list[MemoryJob]:
        """Claim pending (or lock-expired) jobs for processing.

        Jobs that already burned ``max_attempts`` are dead-lettered as
        ``failed`` instead of being reclaimed forever — a job whose payload
        crashes the worker before fail_job() runs would otherwise loop
        indefinitely.
        """
        # Dead-letter exhausted jobs before claiming.
        await self._storage.execute(
            """
            UPDATE memory_jobs
            SET status = 'failed',
                error = 'exceeded max attempts (' || attempts || ')',
                locked_until = NULL,
                updated_at = NOW()
            WHERE attempts >= $1
              AND (status = 'pending'
                   OR (status = 'processing' AND locked_until < NOW()))
            """,
            max_attempts,
        )

        rows = await self._storage.fetchall(
            """
            WITH candidates AS (
                SELECT job_id
                FROM memory_jobs
                WHERE (status = 'pending'
                       OR (status = 'processing' AND locked_until < NOW()))
                  AND attempts < $3
                ORDER BY created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE memory_jobs j
            SET status = 'processing',
                attempts = attempts + 1,
                locked_until = NOW() + ($2 * INTERVAL '1 second'),
                updated_at = NOW()
            FROM candidates
            WHERE j.job_id = candidates.job_id
            RETURNING j.job_id, j.job_type, j.status, j.attempts, j.payload,
                      j.error, j.locked_until, j.created_at, j.updated_at
            """,
            limit,
            lock_seconds,
            max_attempts,
        )
        return [self._row_to_job(row) for row in rows]

    async def complete_job(self, job_id: str) -> MemoryJob:
        row = await self._storage.fetchone(
            """
            UPDATE memory_jobs
            SET status = 'completed', error = NULL, locked_until = NULL, updated_at = NOW()
            WHERE job_id = $1
            RETURNING job_id, job_type, status, attempts, payload, error,
                      locked_until, created_at, updated_at
            """,
            job_id,
        )
        if row is None:
            raise MemoryJobNotFoundError(f"Memory job not found: {job_id}")
        return self._row_to_job(row)

    async def fail_job(self, job_id: str, error: str) -> MemoryJob:
        row = await self._storage.fetchone(
            """
            UPDATE memory_jobs
            SET status = 'failed', error = $2, locked_until = NULL, updated_at = NOW()
            WHERE job_id = $1
            RETURNING job_id, job_type, status, attempts, payload, error,
                      locked_until, created_at, updated_at
            """,
            job_id,
            error[:2000],
        )
        if row is None:
            raise MemoryJobNotFoundError(f"Memory job not found: {job_id}")
        return self._row_to_job(row)

    def _json(self, value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _row_to_task(self, row: Any) -> TaskRun:
        return TaskRun(
            task_run_id=row["task_run_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            goal=row["goal"],
            status=row["status"],
            outcome=row["outcome"],
            metadata=self._json(row["metadata"], {}),
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            updated_at=row["updated_at"],
            deleted_at=row["deleted_at"],
        )

    def _row_to_event(self, row: Any) -> AgentEvent:
        return AgentEvent(
            event_id=row["event_id"],
            task_run_id=row["task_run_id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            role=row["role"],
            event_type=row["event_type"],
            content=row["content"] or "",
            payload=self._json(row["payload"], {}),
            metadata=self._json(row["metadata"], {}),
            created_at=row["created_at"],
            deleted_at=row["deleted_at"],
            redacted_at=row["redacted_at"],
        )

    def _row_to_checkpoint(self, row: Any) -> TaskCheckpoint:
        return TaskCheckpoint(
            checkpoint_id=row["checkpoint_id"],
            task_run_id=row["task_run_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            summary=row["summary"],
            completed_steps=self._json(row["completed_steps"], []),
            pending_steps=self._json(row["pending_steps"], []),
            decisions=self._json(row["decisions"], []),
            blockers=self._json(row["blockers"], []),
            artifacts=self._json(row["artifacts"], []),
            source_event_ids=self._json(row["source_event_ids"], []),
            metadata=self._json(row["metadata"], {}),
            created_at=row["created_at"],
        )

    def _row_to_job(self, row: Any) -> MemoryJob:
        return MemoryJob(
            job_id=row["job_id"],
            job_type=row["job_type"],
            status=row["status"],
            attempts=row["attempts"],
            payload=self._json(row["payload"], {}),
            error=row["error"],
            locked_until=row["locked_until"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
