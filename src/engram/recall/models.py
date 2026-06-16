"""Models for the memory recall operator."""
# ruff: noqa: TC001, TC003

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from engram.core._types import MemoryId
from engram.memory.models import Memory
from engram.task.models import AgentEvent

RecallIntent = Literal["current", "historical", "event", "lineage", "chat"]


class RecallSource(BaseModel):
    """Provenance for one piece of evidence behind a recall answer."""

    memory_id: MemoryId | None = None
    event_id: str | None = None
    session_id: str | None = None
    created_at: datetime | None = None
    status: str | None = None
    source: str | None = None  # metadata 'source' if present

    model_config = {"frozen": True}


class RecallAnswer(BaseModel):
    """Source-backed answer from the memory operator.

    Composes the right recall surface(s) for a question into one auditable
    result: the prose answer plus the structured facts behind it (current
    value, previous values, when it changed, sources, and any conflict).
    """

    answer_text: str
    intent: RecallIntent
    current: Memory | None = None
    previous: list[Memory] = Field(default_factory=list)
    when_changed: datetime | None = None
    sources: list[RecallSource] = Field(default_factory=list)
    conflict_note: str | None = None
    evidence: list[Memory] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}
