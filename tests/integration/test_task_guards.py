"""Integration tests for TaskMemoryManager durability guards.

Run with: pytest tests/integration/test_task_guards.py -v --run-integration
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
async def task_env():
    """Connected TaskMemoryManager on a unique agent."""
    from conftest import configure_integration_environment
    from engram.core.config import EngramSettings
    from engram.storage.postgres import PostgresStorage
    from engram.task.manager import TaskMemoryManager

    database_url = configure_integration_environment()

    settings = EngramSettings(database_url=database_url)
    storage = PostgresStorage(settings)
    await storage.connect()
    await storage.init_schema()

    manager = TaskMemoryManager(storage)
    agent_id = f"task_guard_{uuid.uuid4().hex[:12]}"
    job_ids: list[str] = []

    yield manager, storage, agent_id, job_ids

    if job_ids:
        await storage.execute(
            "DELETE FROM memory_jobs WHERE job_id = ANY($1::text[])", job_ids
        )
    await storage.execute("DELETE FROM agents WHERE agent_id = $1", agent_id)
    await storage.close()


class TestRecordEventsAtomicity:
    """Turn events and the ingestion job must commit in one transaction."""

    @pytest.mark.asyncio
    async def test_events_and_job_committed_together(self, task_env) -> None:
        from engram.task.models import EventCreate

        manager, storage, agent_id, job_ids = task_env

        creates = [
            EventCreate(
                agent_id=agent_id,
                role="user",
                event_type="user_message",
                content="hello",
            ),
            EventCreate(
                agent_id=agent_id,
                role="assistant",
                event_type="assistant_message",
                content="hi there",
            ),
        ]
        events, job = await manager.record_events(
            creates,
            job_type="turn_ingest",
            job_payload=lambda evts: {"event_ids": [e.event_id for e in evts]},
        )
        assert job is not None
        job_ids.append(job.job_id)

        stored = await manager.list_events(agent_id=agent_id)
        assert {e.event_id for e in stored} == {e.event_id for e in events}
        row = await storage.fetchone(
            "SELECT payload FROM memory_jobs WHERE job_id = $1", job.job_id
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_mid_batch_failure_rolls_back_everything(self, task_env) -> None:
        """If any insert in the batch fails, neither events nor the job may
        persist — otherwise a turn exists that will never be ingested."""
        from unittest.mock import patch

        from engram.core.exceptions import StorageError
        from engram.task.models import EventCreate

        manager, storage, agent_id, _job_ids = task_env

        creates = [
            EventCreate(
                agent_id=agent_id,
                role="user",
                event_type="user_message",
                content="first",
            ),
            EventCreate(
                agent_id=agent_id,
                role="assistant",
                event_type="assistant_message",
                content="second",
            ),
        ]
        # Same generated id for both events -> PK violation on the second
        # insert, inside the transaction.
        with (
            patch("engram.task.models._id", return_value=f"evt_dup_{agent_id}"),
            pytest.raises(StorageError),
        ):
            await manager.record_events(
                creates,
                job_type="turn_ingest",
                job_payload=lambda _evts: {},
            )

        stored = await manager.list_events(agent_id=agent_id)
        assert stored == []
        job_count = await storage.fetchval(
            "SELECT COUNT(*) FROM memory_jobs WHERE job_id = $1",
            f"evt_dup_{agent_id}",
        )
        assert job_count == 0


class TestTaskStatusTransitions:
    """Terminal task statuses must not silently reopen."""

    @pytest.mark.asyncio
    async def test_completed_task_cannot_reactivate(self, task_env) -> None:
        from engram.core.exceptions import EngramError

        manager, _storage, agent_id, _job_ids = task_env
        task = await manager.start_task(goal="ship it", agent_id=agent_id)
        await manager.set_task_status(task.task_run_id, "completed", outcome="done")

        with pytest.raises(EngramError, match="terminal"):
            await manager.set_task_status(task.task_run_id, "active")

    @pytest.mark.asyncio
    async def test_same_terminal_status_is_idempotent(self, task_env) -> None:
        manager, _storage, agent_id, _job_ids = task_env
        task = await manager.start_task(goal="ship it", agent_id=agent_id)
        await manager.set_task_status(task.task_run_id, "completed")

        again = await manager.set_task_status(task.task_run_id, "completed")
        assert again.status == "completed"

    @pytest.mark.asyncio
    async def test_pause_and_resume_still_allowed(self, task_env) -> None:
        manager, _storage, agent_id, _job_ids = task_env
        task = await manager.start_task(goal="long work", agent_id=agent_id)

        paused = await manager.set_task_status(task.task_run_id, "paused")
        assert paused.status == "paused"
        resumed = await manager.set_task_status(task.task_run_id, "active")
        assert resumed.status == "active"

    @pytest.mark.asyncio
    async def test_missing_task_still_raises_not_found(self, task_env) -> None:
        from engram.task.manager import TaskNotFoundError

        manager, _storage, _agent_id, _job_ids = task_env

        with pytest.raises(TaskNotFoundError):
            await manager.set_task_status("task_missing", "completed")


class TestEventOrdering:
    @pytest.mark.asyncio
    async def test_same_timestamp_events_order_stably(self, task_env) -> None:
        """Events sharing a timestamp must order deterministically."""
        from engram.task.models import EventCreate

        manager, storage, agent_id, _job_ids = task_env

        events, _ = await manager.record_events(
            [
                EventCreate(
                    agent_id=agent_id,
                    role="user",
                    event_type="user_message",
                    content=f"message {i}",
                )
                for i in range(5)
            ]
        )
        # Force identical timestamps
        await storage.execute(
            "UPDATE agent_events SET created_at = NOW() WHERE agent_id = $1",
            agent_id,
        )

        first = await manager.list_events(agent_id=agent_id)
        second = await manager.list_events(agent_id=agent_id)

        assert [e.event_id for e in first] == [e.event_id for e in second]
        assert [e.event_id for e in first] == sorted(e.event_id for e in events)


class TestSummaryCompareAndSet:
    """Concurrent summary updates must not silently overwrite each other."""

    @pytest.mark.asyncio
    async def test_cas_succeeds_on_expected_timestamp(self, task_env) -> None:
        from engram.session.manager import SessionManager
        from engram.session.models import SessionCreate

        _manager, storage, agent_id, _job_ids = task_env
        sessions = SessionManager(storage)
        sess = await sessions.create(SessionCreate(agent_id=agent_id))

        # First write: no summary yet -> expected None
        updated = await sessions.try_update_summary(
            sess.session_id, "v1", expected_updated_at=None
        )
        assert updated is not None
        assert updated.summary == "v1"

    @pytest.mark.asyncio
    async def test_cas_returns_none_when_concurrent_writer_won(self, task_env) -> None:
        from engram.session.manager import SessionManager
        from engram.session.models import SessionCreate

        _manager, storage, agent_id, _job_ids = task_env
        sessions = SessionManager(storage)
        sess = await sessions.create(SessionCreate(agent_id=agent_id))

        first = await sessions.try_update_summary(
            sess.session_id, "writer A", expected_updated_at=None
        )
        assert first is not None

        # A second writer based on the stale (None) snapshot must lose
        stale = await sessions.try_update_summary(
            sess.session_id, "writer B", expected_updated_at=None
        )
        assert stale is None

        current = await sessions.get(sess.session_id)
        assert current.summary == "writer A"

    @pytest.mark.asyncio
    async def test_cas_missing_session_raises(self, task_env) -> None:
        from engram.core.exceptions import SessionNotFoundError
        from engram.session.manager import SessionManager

        _manager, storage, _agent_id, _job_ids = task_env
        sessions = SessionManager(storage)

        with pytest.raises(SessionNotFoundError):
            await sessions.try_update_summary(
                "sess_does_not_exist", "x", expected_updated_at=None
            )


class TestJobRetryCap:
    """Jobs whose workers keep crashing must dead-letter, not retry forever."""

    @pytest.mark.asyncio
    async def test_exhausted_job_is_failed_not_reclaimed(self, task_env) -> None:
        manager, storage, _agent_id, job_ids = task_env

        job = await manager.enqueue_job("turn_ingest", {"k": "v"})
        job_ids.append(job.job_id)
        # Simulate a job that crashed the worker max_attempts times: stuck in
        # 'processing' with an expired lock and attempts at the cap.
        await storage.execute(
            """
            UPDATE memory_jobs
            SET status = 'processing', attempts = 5,
                locked_until = NOW() - INTERVAL '1 hour'
            WHERE job_id = $1
            """,
            job.job_id,
        )

        claimed = await manager.claim_jobs(limit=10, max_attempts=5)

        assert all(c.job_id != job.job_id for c in claimed)
        row = await storage.fetchone(
            "SELECT status, error FROM memory_jobs WHERE job_id = $1", job.job_id
        )
        assert row["status"] == "failed"
        assert "attempts" in (row["error"] or "")

    @pytest.mark.asyncio
    async def test_job_under_cap_is_reclaimed(self, task_env) -> None:
        manager, storage, _agent_id, job_ids = task_env

        job = await manager.enqueue_job("turn_ingest", {"k": "v"})
        job_ids.append(job.job_id)
        await storage.execute(
            """
            UPDATE memory_jobs
            SET status = 'processing', attempts = 2,
                locked_until = NOW() - INTERVAL '1 hour'
            WHERE job_id = $1
            """,
            job.job_id,
        )

        claimed = await manager.claim_jobs(limit=10, max_attempts=5)

        mine = [c for c in claimed if c.job_id == job.job_id]
        assert len(mine) == 1
        assert mine[0].attempts == 3
        assert mine[0].status == "processing"


class TestSearchEvents:
    """Full-text event recall over the real agent_events index."""

    async def _seed(self, manager, agent_id):
        from engram.task.models import EventCreate

        contents = [
            ("user", "user_message", "What did I ask about the chatbot memory jobs?"),
            (
                "assistant",
                "assistant_message",
                "We discussed making memory jobs automatic.",
            ),
            ("user", "user_message", "Remind me of my budget for the trip to Berlin."),
        ]
        events = []
        for role, etype, content in contents:
            events.append(
                await manager.record_event(
                    EventCreate(
                        agent_id=agent_id,
                        role=role,
                        event_type=etype,
                        content=content,
                    )
                )
            )
        return events

    @pytest.mark.asyncio
    async def test_keyword_match_ranks_relevant_events(self, task_env) -> None:
        manager, _storage, agent_id, _job_ids = task_env
        await self._seed(manager, agent_id)

        hits = await manager.search_events("chatbot memory", agent_id=agent_id)

        assert hits, "expected at least one keyword match"
        assert all("chatbot" in h.content or "memory" in h.content for h in hits)
        # The 'Berlin budget' event shares no terms and must not match.
        assert all("Berlin" not in h.content for h in hits)

    @pytest.mark.asyncio
    async def test_role_filter(self, task_env) -> None:
        manager, _storage, agent_id, _job_ids = task_env
        await self._seed(manager, agent_id)

        hits = await manager.search_events(
            "memory", agent_id=agent_id, roles=["assistant"]
        )

        assert hits
        assert all(h.role == "assistant" for h in hits)

    @pytest.mark.asyncio
    async def test_temporal_filter_excludes_out_of_range(self, task_env) -> None:
        import datetime as _dt

        manager, _storage, agent_id, _job_ids = task_env
        await self._seed(manager, agent_id)

        now = _dt.datetime.now(_dt.timezone.utc)
        past = now - _dt.timedelta(days=1)
        future = now + _dt.timedelta(days=1)

        assert await manager.search_events("chatbot", agent_id=agent_id, since=past)
        assert not await manager.search_events(
            "chatbot", agent_id=agent_id, since=future
        )
        assert not await manager.search_events("chatbot", agent_id=agent_id, until=past)

    @pytest.mark.asyncio
    async def test_stopword_only_query_returns_empty(self, task_env) -> None:
        manager, _storage, agent_id, _job_ids = task_env
        await self._seed(manager, agent_id)

        # 'the and of' yields an empty tsquery -> no matches, and no error.
        assert await manager.search_events("the and of", agent_id=agent_id) == []

    @pytest.mark.asyncio
    async def test_deleted_events_excluded_unless_requested(self, task_env) -> None:
        manager, storage, agent_id, _job_ids = task_env
        events = await self._seed(manager, agent_id)
        await storage.execute(
            "UPDATE agent_events SET deleted_at = NOW() WHERE event_id = $1",
            events[0].event_id,
        )

        default_hits = await manager.search_events("chatbot", agent_id=agent_id)
        assert events[0].event_id not in {h.event_id for h in default_hits}

        with_deleted = await manager.search_events(
            "chatbot", agent_id=agent_id, include_deleted=True
        )
        assert events[0].event_id in {h.event_id for h in with_deleted}

    @pytest.mark.asyncio
    async def test_empty_query_raises_before_db(self, task_env) -> None:
        manager, _storage, agent_id, _job_ids = task_env

        with pytest.raises(ValueError, match="must not be empty"):
            await manager.search_events("   ", agent_id=agent_id)
