"""Unit tests for long-running task memory models and manager."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from engram.task import ContextBuildOptions, EventCreate, TaskMemoryManager
from engram.task.context import ContextBuilder
from engram.task.models import AgentEvent, TaskCheckpoint, TaskRun


class TestTaskMemoryManager:
    @pytest.mark.asyncio
    async def test_start_task_ensures_agent_and_user_then_inserts(self) -> None:
        storage = AsyncMock()
        storage.execute = AsyncMock(return_value="OK")
        manager = TaskMemoryManager(storage)

        task = await manager.start_task(
            goal="Build durable memory",
            agent_id="agent",
            user_id="user",
            metadata={"priority": "high"},
        )

        assert task.goal == "Build durable memory"
        assert task.agent_id == "agent"
        assert task.user_id == "user"
        assert task.metadata == {"priority": "high"}
        assert storage.execute.await_count == 3

    @pytest.mark.asyncio
    async def test_record_event_creates_ledger_record(self) -> None:
        storage = AsyncMock()
        storage.execute = AsyncMock(return_value="OK")
        manager = TaskMemoryManager(storage)

        event = await manager.record_event(
            EventCreate(
                task_run_id="task_1",
                agent_id="agent",
                user_id="user",
                role="user",
                event_type="user_message",
                content="Remember this",
                payload={"turn": 1},
            )
        )

        assert event.task_run_id == "task_1"
        assert event.content == "Remember this"
        assert event.payload == {"turn": 1}
        assert storage.execute.await_count == 3

    @pytest.mark.asyncio
    async def test_list_tasks_filters_resume_candidates(self) -> None:
        storage = AsyncMock()
        storage.fetchall = AsyncMock(return_value=[])
        manager = TaskMemoryManager(storage)

        tasks = await manager.list_tasks(
            agent_id="agent",
            user_id="user",
            status=["active", "paused"],
            limit=25,
        )

        assert tasks == []
        query, *params = storage.fetchall.call_args.args
        assert "agent_id = $1" in query
        assert "user_id = $2" in query
        assert "status = ANY($3::text[])" in query
        assert "deleted_at IS NULL" in query
        assert params == ["agent", "user", ["active", "paused"], 25]

    @pytest.mark.asyncio
    async def test_list_tasks_converts_rows(self) -> None:
        now = datetime.now(timezone.utc)
        storage = AsyncMock()
        storage.fetchall = AsyncMock(
            return_value=[
                {
                    "task_run_id": "task_1",
                    "agent_id": "agent",
                    "user_id": "user",
                    "session_id": None,
                    "goal": "Resume me",
                    "status": "paused",
                    "outcome": None,
                    "metadata": {"source": "test"},
                    "started_at": now,
                    "ended_at": None,
                    "updated_at": now,
                    "deleted_at": None,
                }
            ]
        )
        manager = TaskMemoryManager(storage)

        tasks = await manager.list_tasks(agent_id="agent", status="paused")

        assert tasks[0].task_run_id == "task_1"
        assert tasks[0].status == "paused"
        assert tasks[0].metadata == {"source": "test"}


class TestContextBuilder:
    @pytest.mark.asyncio
    async def test_builds_budgeted_sections_from_task_sources(self) -> None:
        from engram.memory.models import Memory, SearchResult

        manager = AsyncMock()
        manager.get_task = AsyncMock(
            return_value=TaskRun(
                task_run_id="task_1",
                agent_id="agent",
                user_id="user",
                goal="Ship agent memory",
            )
        )
        manager.list_events = AsyncMock(
            return_value=[
                AgentEvent(
                    task_run_id="task_1",
                    agent_id="agent",
                    user_id="user",
                    role="user",
                    event_type="user_message",
                    content="Need persistent task memory",
                ),
                AgentEvent(
                    task_run_id="task_1",
                    agent_id="agent",
                    user_id="user",
                    role="assistant",
                    event_type="decision",
                    content="Use a raw event ledger plus checkpoints",
                ),
            ]
        )
        manager.list_checkpoints = AsyncMock(
            return_value=[
                TaskCheckpoint(
                    task_run_id="task_1",
                    agent_id="agent",
                    user_id="user",
                    summary="Current state summary",
                    decisions=["Keep facts and raw events"],
                )
            ]
        )
        search = AsyncMock(
            return_value=[
                SearchResult(
                    memory=Memory(
                        memory_id="mem_1",
                        agent_id="agent",
                        user_id="user",
                        content="User wants memory for 200k context tasks",
                    ),
                    score=0.9,
                )
            ]
        )

        result = await ContextBuilder(manager, search).build(
            task_run_id="task_1",
            agent_id="agent",
            user_id="user",
            options=ContextBuildOptions(max_tokens=500, include_graph=False),
        )

        assert "## Task" in result.text
        assert "Ship agent memory" in result.text
        assert "Use a raw event ledger plus checkpoints" in result.text
        assert "User wants memory for 200k context tasks" in result.text
        assert result.metadata["events"] == 2


def _event_row(
    *,
    event_id: str = "evt_1",
    role: str = "user",
    event_type: str = "user_message",
    content: str = "What did I ask about the chatbot?",
) -> dict[str, object]:
    """Build a fake agent_events row for manager row-mapping tests."""
    return {
        "event_id": event_id,
        "task_run_id": "task_1",
        "session_id": "sess_1",
        "agent_id": "agent",
        "user_id": "user",
        "role": role,
        "event_type": event_type,
        "content": content,
        "payload": {},
        "metadata": {},
        "created_at": datetime(2026, 6, 14, tzinfo=timezone.utc),
        "deleted_at": None,
        "redacted_at": None,
    }


class TestSearchEvents:
    def _manager(self, rows: list[dict[str, object]]) -> TaskMemoryManager:
        storage = AsyncMock()
        storage._settings.text_search_config = "english"
        storage.fetchall = AsyncMock(return_value=rows)
        return TaskMemoryManager(storage)

    @pytest.mark.asyncio
    async def test_builds_keyword_query_with_all_filters(self) -> None:
        manager = self._manager([_event_row()])
        since = datetime(2026, 6, 14, tzinfo=timezone.utc)
        until = datetime(2026, 6, 15, tzinfo=timezone.utc)

        await manager.search_events(
            "chatbot memory",
            agent_id="agent",
            task_run_id="task_1",
            session_id="sess_1",
            user_id="user",
            event_types=["user_message"],
            roles=["user"],
            since=since,
            until=until,
            limit=7,
            mode="keyword",
        )

        sql, *params = manager._storage.fetchall.await_args.args  # type: ignore[attr-defined]
        assert "to_tsvector('english', content)" in sql
        assert "@@ plainto_tsquery('english', $1)" in sql
        assert "ts_rank(to_tsvector('english', content)" in sql
        # Every filter contributes a WHERE clause.
        assert "agent_id = $2" in sql
        assert "task_run_id = $3" in sql
        assert "session_id = $4" in sql
        assert "user_id = $5" in sql
        assert "event_type = ANY($6::text[])" in sql
        assert "role = ANY($7::text[])" in sql
        assert "created_at >= $8" in sql
        assert "created_at <= $9" in sql
        assert "deleted_at IS NULL" in sql
        assert params == [
            "chatbot memory",
            "agent",
            "task_1",
            "sess_1",
            "user",
            ["user_message"],
            ["user"],
            since,
            until,
            7,
        ]

    @pytest.mark.asyncio
    async def test_minimal_query_only_matches_and_excludes_deleted(self) -> None:
        manager = self._manager([])

        await manager.search_events("budget")

        sql, *params = manager._storage.fetchall.await_args.args  # type: ignore[attr-defined]
        assert "deleted_at IS NULL" in sql
        # No optional filters added their WHERE clauses.
        assert "agent_id = $" not in sql
        assert "created_at >=" not in sql
        assert "ANY(" not in sql
        assert params == ["budget", 50]

    @pytest.mark.asyncio
    async def test_hybrid_query_uses_embedding_and_keyword_branches(self) -> None:
        manager = self._manager([_event_row()])

        await manager.search_events(
            "chatbot memory",
            agent_id="agent",
            limit=7,
            query_embedding=[0.1, 0.2, 0.3],
        )

        sql, *params = manager._storage.fetchall.await_args.args  # type: ignore[attr-defined]
        assert "WITH semantic_search AS" in sql
        assert "keyword_search AS" in sql
        assert "event_embedding <=> $2::vector" in sql
        assert "FULL OUTER JOIN keyword_search" in sql
        assert "agent_id = $3" in sql
        assert params == ["chatbot memory", "[0.1, 0.2, 0.3]", "agent", 7]

    @pytest.mark.asyncio
    async def test_hybrid_without_embedding_falls_back_to_keyword(self) -> None:
        manager = self._manager([])

        await manager.search_events("budget", mode="hybrid")

        sql, *params = manager._storage.fetchall.await_args.args  # type: ignore[attr-defined]
        assert "WITH semantic_search AS" not in sql
        assert "event_embedding <=>" not in sql
        assert "@@ plainto_tsquery('english', $1)" in sql
        assert params == ["budget", 50]

    @pytest.mark.asyncio
    async def test_include_deleted_drops_deleted_filter(self) -> None:
        manager = self._manager([])

        await manager.search_events("budget", include_deleted=True)

        sql, *_ = manager._storage.fetchall.await_args.args  # type: ignore[attr-defined]
        assert "deleted_at IS NULL" not in sql

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_query", ["", "   ", "\n\t"])
    async def test_empty_query_raises(self, bad_query: str) -> None:
        manager = self._manager([])

        with pytest.raises(ValueError, match="must not be empty"):
            await manager.search_events(bad_query)

        manager._storage.fetchall.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_maps_rows_to_events(self) -> None:
        manager = self._manager(
            [_event_row(event_id="evt_a"), _event_row(event_id="evt_b")]
        )

        events = await manager.search_events("chatbot")

        assert [e.event_id for e in events] == ["evt_a", "evt_b"]
        assert all(isinstance(e, AgentEvent) for e in events)
