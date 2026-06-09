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
