"""Models for long-running task memory.

This module defines the task/event/checkpoint/job records used by the
long-running agent memory layer. These records are additive to the existing
fact-oriented ``agent_memory`` table.
"""
# ruff: noqa: TC001

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from engram.core._types import AgentId, Metadata, SessionId, UserId

TaskRunStatus = Literal["active", "paused", "completed", "failed", "cancelled"]
EventRole = Literal["user", "assistant", "agent", "tool", "system"]
EventType = Literal[
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "agent_action",
    "decision",
    "observation",
    "artifact",
    "error",
    "system_note",
]
MemoryJobStatus = Literal["pending", "processing", "completed", "failed"]
MemoryJobType = Literal["turn_ingest", "checkpoint"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class TaskRun(BaseModel):
    """A long-running unit of agent work."""

    task_run_id: str = Field(default_factory=lambda: _id("task"))
    agent_id: AgentId
    user_id: UserId | None = None
    session_id: SessionId | None = None
    goal: str = Field(..., min_length=1, max_length=100000)
    status: TaskRunStatus = "active"
    outcome: str | None = Field(default=None, max_length=200000)
    metadata: Metadata = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    updated_at: datetime = Field(default_factory=_utcnow)
    deleted_at: datetime | None = None

    model_config = {"frozen": False, "extra": "forbid"}


class AgentEvent(BaseModel):
    """Immutable event in a task/session ledger."""

    event_id: str = Field(default_factory=lambda: _id("evt"))
    task_run_id: str | None = None
    session_id: SessionId | None = None
    agent_id: AgentId
    user_id: UserId | None = None
    role: EventRole
    event_type: EventType
    content: str = Field(default="", max_length=500000)
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: Metadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    deleted_at: datetime | None = None
    redacted_at: datetime | None = None

    model_config = {"frozen": False, "extra": "forbid"}


class EventCreate(BaseModel):
    """Input model for recording an event."""

    agent_id: AgentId
    role: EventRole
    event_type: EventType
    content: str = Field(default="", max_length=500000)
    task_run_id: str | None = None
    session_id: SessionId | None = None
    user_id: UserId | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}


class TaskCheckpoint(BaseModel):
    """Compact state snapshot for a long-running task."""

    checkpoint_id: str = Field(default_factory=lambda: _id("chk"))
    task_run_id: str
    agent_id: AgentId
    user_id: UserId | None = None
    summary: str = Field(..., min_length=1, max_length=200000)
    completed_steps: list[str] = Field(default_factory=list)
    pending_steps: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    model_config = {"frozen": False, "extra": "forbid"}


class MemoryJob(BaseModel):
    """Durable background job for memory derivation."""

    job_id: str = Field(default_factory=lambda: _id("job"))
    job_type: MemoryJobType
    status: MemoryJobStatus = "pending"
    attempts: int = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    locked_until: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"frozen": False, "extra": "forbid"}


class ContextBuildOptions(BaseModel):
    """Options for assembling a task context block."""

    query: str = Field(default="", max_length=10000)
    max_tokens: int = Field(default=200000, ge=100, le=300000)
    recent_event_limit: int = Field(default=40, ge=1, le=500)
    memory_limit: int = Field(default=25, ge=1, le=200)
    checkpoint_limit: int = Field(default=3, ge=1, le=20)
    include_graph: bool = True

    model_config = {"frozen": True, "extra": "forbid"}


class ContextBuildResult(BaseModel):
    """Rendered context and section token accounting."""

    text: str
    sections: dict[str, str] = Field(default_factory=dict)
    token_estimate: int = 0
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}


class LongInputChunk(BaseModel):
    """A source-anchored chunk extracted from a long input prompt/document."""

    chunk_id: str
    index: int = Field(ge=0)
    kind: str = Field(default="background_context", max_length=100)
    heading: str | None = Field(default=None, max_length=1000)
    text: str = Field(..., min_length=1, max_length=200000)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    token_estimate: int = Field(default=0, ge=0)
    quote_hash: str = Field(..., min_length=8, max_length=128)
    source_event_id: str | None = None
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}


class LongInputIngestionReport(BaseModel):
    """Result of recording and distilling one long input."""

    task_run_id: str
    source_event_id: str
    chunks: list[LongInputChunk] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    checkpoint_id: str | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}


class LongInputContextResult(BaseModel):
    """Prompt-ready context for answering against long-input state."""

    text: str
    trace: dict[str, Any] = Field(default_factory=dict)
    token_estimate: int = 0
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}
