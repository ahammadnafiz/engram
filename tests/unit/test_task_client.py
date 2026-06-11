"""Unit tests for Engram long-running task memory APIs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.client import Engram
from engram.memory.models import Memory, RecallTrace, SearchResult
from engram.task.models import AgentEvent, MemoryJob, TaskCheckpoint, TaskRun


def make_engram() -> Engram:
    eg = Engram()
    eg._connected = True
    eg._task_memory = AsyncMock()
    eg._memory_store = AsyncMock()
    eg._sessions = AsyncMock()
    eg._graph = MagicMock()
    eg._llm = None
    return eg


def task() -> TaskRun:
    return TaskRun(
        task_run_id="task_1",
        agent_id="agent",
        user_id="user",
        session_id="session",
        goal="Build persistent agent memory",
    )


class TestTaskClientTurnFlow:
    @pytest.mark.asyncio
    async def test_list_and_pause_cancel_tasks_delegate_to_manager(self) -> None:
        eg = make_engram()
        eg._task_memory.list_tasks = AsyncMock(return_value=[task()])
        eg._task_memory.set_task_status = AsyncMock(return_value=task())

        tasks = await eg.list_tasks(
            agent_id="agent",
            user_id="user",
            status=["active", "paused"],
            limit=10,
        )
        await eg.pause_task("task_1", outcome="Waiting for tools")
        await eg.cancel_task("task_1", outcome="No longer needed")

        assert tasks[0].task_run_id == "task_1"
        eg._task_memory.list_tasks.assert_awaited_once_with(
            agent_id="agent",
            user_id="user",
            status=["active", "paused"],
            limit=10,
            include_deleted=False,
        )
        eg._task_memory.set_task_status.assert_any_await(
            "task_1",
            "paused",
            outcome="Waiting for tools",
        )
        eg._task_memory.set_task_status.assert_any_await(
            "task_1",
            "cancelled",
            outcome="No longer needed",
        )

    @pytest.mark.asyncio
    async def test_record_turn_records_events_and_enqueues_ingestion(self) -> None:
        eg = make_engram()

        async def record_events(creates, *, job_type=None, job_payload=None):
            events = [
                AgentEvent(
                    event_id=f"evt_{i + 1}",
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
                for i, create in enumerate(creates)
            ]
            job = None
            if job_type is not None:
                job = MemoryJob(
                    job_type=job_type,
                    payload=job_payload(events) if job_payload else {},
                )
            return events, job

        eg._task_memory.get_task = AsyncMock(return_value=task())
        eg._task_memory.record_events = AsyncMock(side_effect=record_events)

        events = await eg.record_turn(
            "task_1",
            "Please remember that I use Postgres",
            "Stored.",
            tool_calls=[{"name": "search", "query": "postgres"}],
            artifacts=[{"path": "notes.md"}],
        )

        assert [event.event_type for event in events] == [
            "user_message",
            "assistant_message",
            "tool_call",
            "artifact",
        ]
        # Events and the ingestion job go through one atomic call
        call = eg._task_memory.record_events.call_args
        assert call.kwargs["job_type"] == "turn_ingest"
        payload = call.kwargs["job_payload"](events)
        assert payload["task_run_id"] == "task_1"
        assert payload["user_event_id"] == "evt_1"
        assert payload["assistant_event_id"] == "evt_2"
        assert payload["event_ids"] == ["evt_1", "evt_2", "evt_3", "evt_4"]

    @pytest.mark.asyncio
    async def test_process_memory_jobs_without_llm_creates_checkpoint(self) -> None:
        eg = make_engram()
        eg._task_memory.claim_jobs = AsyncMock(
            return_value=[
                MemoryJob(
                    job_id="job_1",
                    job_type="turn_ingest",
                    status="processing",
                    attempts=1,
                    payload={
                        "task_run_id": "task_1",
                        "user_message": "Remember project constraints",
                        "assistant_response": "I will keep them in memory.",
                        "event_ids": ["evt_1", "evt_2"],
                    },
                )
            ]
        )
        eg._task_memory.get_task = AsyncMock(return_value=task())
        eg._task_memory.latest_checkpoint = AsyncMock(return_value=None)
        eg._task_memory.create_checkpoint = AsyncMock(
            side_effect=lambda checkpoint: checkpoint
        )
        eg._task_memory.complete_job = AsyncMock(
            return_value=MemoryJob(
                job_id="job_1",
                job_type="turn_ingest",
                status="completed",
                attempts=1,
            )
        )

        jobs = await eg.process_memory_jobs(limit=1)

        assert jobs[0].status == "completed"
        checkpoint = eg._task_memory.create_checkpoint.call_args.args[0]
        assert checkpoint.task_run_id == "task_1"
        assert checkpoint.source_event_ids == ["evt_1", "evt_2"]
        assert "Remember project constraints" in checkpoint.summary

    @pytest.mark.asyncio
    async def test_run_memory_worker_loops_until_max_iterations(self) -> None:
        eg = make_engram()
        eg.process_memory_jobs = AsyncMock(
            side_effect=[
                [MemoryJob(job_type="turn_ingest", status="completed")],
                [],
            ]
        )

        count = await eg.run_memory_worker(
            batch_size=2,
            interval_seconds=0,
            max_iterations=2,
        )

        assert count == 1
        assert eg.process_memory_jobs.await_count == 2


class TestTaskClientContext:
    @pytest.mark.asyncio
    async def test_build_context_uses_public_retrieval_and_task_sources(self) -> None:
        eg = make_engram()
        eg._task_memory.get_task = AsyncMock(return_value=task())
        eg._task_memory.list_events = AsyncMock(
            return_value=[
                AgentEvent(
                    task_run_id="task_1",
                    agent_id="agent",
                    user_id="user",
                    role="assistant",
                    event_type="assistant_message",
                    content="Implemented event ledger.",
                )
            ]
        )
        eg._task_memory.list_checkpoints = AsyncMock(
            return_value=[
                TaskCheckpoint(
                    task_run_id="task_1",
                    agent_id="agent",
                    user_id="user",
                    summary="Task has a ledger and checkpoint path.",
                )
            ]
        )
        eg._memory_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    memory=Memory(
                        memory_id="mem_1",
                        agent_id="agent",
                        user_id="user",
                        content="User needs robust 200k context assembly",
                    ),
                    score=0.9,
                )
            ]
        )

        result = await eg.build_context(
            "task_1",
            query="memory architecture",
            include_graph=False,
            max_tokens=500,
        )

        assert "## Task" in result.text
        assert "Implemented event ledger." in result.text
        assert "User needs robust 200k context assembly" in result.text


class TestLongInputClient:
    @pytest.mark.asyncio
    async def test_record_long_input_chunks_anchors_memories_and_manifest(self) -> None:
        eg = make_engram()
        eg._task_memory.get_task = AsyncMock(return_value=task())
        event_count = 0

        async def record_event(create) -> AgentEvent:
            nonlocal event_count
            event_count += 1
            return AgentEvent(
                event_id=f"evt_{event_count}",
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

        eg._task_memory.record_event = AsyncMock(side_effect=record_event)
        eg._task_memory.create_checkpoint = AsyncMock(
            side_effect=lambda checkpoint: checkpoint
        )
        eg.add = AsyncMock(
            side_effect=[
                Memory(
                    memory_id="mem_1",
                    agent_id="agent",
                    user_id="user",
                    content="The vendor shall maintain audit logs.",
                    memory_type="constraint",
                ),
                Memory(
                    memory_id="mem_2",
                    agent_id="agent",
                    user_id="user",
                    content="The agent must answer with citations.",
                    memory_type="task",
                ),
            ]
        )

        text = """# Legal Requirements
The vendor shall maintain audit logs for seven years.

# Agent Instructions
The agent must answer with citations. Next Wednesday is the review deadline.
"""

        report = await eg.record_long_input(
            "task_1",
            text,
            title="Vendor review",
            max_chunk_tokens=40,
            extract_with_llm=False,
        )

        assert report.source_event_id == "evt_1"
        assert len(report.chunks) >= 2
        assert report.memory_ids == ["mem_1", "mem_2"]
        assert report.manifest["chunks"] == len(report.chunks)
        assert report.manifest["memory_count"] == 2
        assert report.trace["time_notes"]
        checkpoint = eg._task_memory.create_checkpoint.call_args.args[0]
        assert checkpoint.metadata["long_input_manifest"]["title"] == "Vendor review"
        first_add = eg.add.call_args_list[0].kwargs
        assert first_add["metadata"]["source_event_id"] == "evt_1"
        assert first_add["metadata"]["chunk_id"].startswith("chunk_")
        assert first_add["metadata"]["quote_hash"]

    @pytest.mark.asyncio
    async def test_build_long_input_context_combines_recall_source_chunks_and_manifest(
        self,
    ) -> None:
        eg = make_engram()
        eg._task_memory.get_task = AsyncMock(return_value=task())
        eg.trace_recall = AsyncMock(
            return_value=RecallTrace(
                query="audit logs",
                agent_id="agent",
                user_id="user",
                critical_memory_ids=["mem_1"],
                kept_memory_ids=["mem_1"],
                context="## Memory Recall\n- [constraint] The vendor shall maintain audit logs.",
            )
        )
        eg._task_memory.list_events = AsyncMock(
            return_value=[
                AgentEvent(
                    event_id="evt_chunk",
                    task_run_id="task_1",
                    agent_id="agent",
                    user_id="user",
                    role="user",
                    event_type="artifact",
                    content="The vendor shall maintain audit logs for seven years.",
                    payload={
                        "kind": "long_input_chunk",
                        "chunk": {
                            "chunk_id": "chunk_0001",
                            "kind": "legal_clause",
                            "heading": "Legal Requirements",
                            "char_start": 0,
                            "char_end": 57,
                            "quote_hash": "abc123456789",
                        },
                    },
                    metadata={"long_input_chunk": True},
                )
            ]
        )
        eg._task_memory.list_checkpoints = AsyncMock(
            return_value=[
                TaskCheckpoint(
                    task_run_id="task_1",
                    agent_id="agent",
                    user_id="user",
                    summary="Long input recorded",
                    metadata={
                        "long_input_manifest": {
                            "title": "Vendor review",
                            "chunks": 1,
                            "memory_count": 1,
                            "time_notes": [],
                        }
                    },
                )
            ]
        )

        result = await eg.build_long_input_context(
            "task_1",
            query="audit logs",
            expected_terms=["audit logs", "seven years"],
        )

        assert "## Memory Recall" in result.text
        assert "## Source Chunks" in result.text
        assert "chunk_id=chunk_0001" in result.text
        assert "## Long Input Manifest" in result.text
        assert result.trace["missing_expected_terms"] == []
