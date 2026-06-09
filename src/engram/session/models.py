"""Session models for Engram.

This module defines models for agent sessions.
"""
# ruff: noqa: TC001

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from engram.core._types import AgentId, Metadata, SessionId, UserId


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"sess_{uuid4().hex}"


def _utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Session(BaseModel):
    """Represents an agent-user conversation session.

    Sessions track conversation context and can be used to scope
    memory operations to specific interactions.

    Attributes:
        session_id: Unique identifier for the session.
        agent_id: ID of the agent in this session.
        user_id: Optional ID of the user in this session.
        started_at: When the session started.
        ended_at: When the session ended (None if active).
        summary: Rolling conversation summary (None until first update).
        summary_updated_at: When the summary was last updated.
        metadata: Additional session metadata.

    Example:
        session = Session(
            agent_id="agent_123",
            user_id="user_456",
            metadata={"channel": "web"}
        )
    """

    session_id: SessionId = Field(default_factory=generate_session_id)
    agent_id: AgentId
    user_id: UserId | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    summary: str | None = None
    summary_updated_at: datetime | None = None
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": False, "extra": "forbid"}

    @property
    def is_active(self) -> bool:
        """Check if the session is still active."""
        return self.ended_at is None

    @property
    def duration_seconds(self) -> float | None:
        """Get session duration in seconds, or None if still active."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


class SessionCreate(BaseModel):
    """Input model for creating a new session.

    Attributes:
        agent_id: ID of the agent.
        user_id: Optional user ID.
        metadata: Optional session metadata.
    """

    agent_id: AgentId
    user_id: UserId | None = None
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}
