"""Memory models for Engram.

This module defines the Pydantic models for memories and search results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from engram.core._types import (
    AgentId,
    MemoryId,
    Metadata,
    SessionId,
    UserId,
    Vector,
)


def generate_memory_id() -> str:
    """Generate a unique memory ID."""
    return f"mem_{uuid4().hex}"


def _utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime.
    
    This is the recommended replacement for the deprecated datetime.utcnow().
    """
    return datetime.now(timezone.utc)


class Memory(BaseModel):
    """Represents a single memory unit in Engram.

    A memory stores content along with its vector embedding, importance score,
    and associated metadata. Memories are linked to agents and optionally to
    specific users and sessions.

    Attributes:
        memory_id: Unique identifier for the memory.
        agent_id: ID of the agent this memory belongs to.
        user_id: Optional ID of the user associated with this memory.
        session_id: Optional ID of the session where this memory was created.
        content: The actual text content of the memory.
        embedding: Vector embedding of the content (None if not yet computed).
        importance: Importance score from 0.0 to 1.0 (default 0.5).
        access_count: Number of times this memory has been accessed.
        created_at: Timestamp when the memory was created.
        last_accessed_at: Timestamp of last access.
        metadata: Additional key-value metadata.

    Example:
        memory = Memory(
            agent_id="agent_123",
            content="User prefers dark mode in applications",
            importance=0.8,
            metadata={"category": "preference"}
        )
    """

    memory_id: MemoryId = Field(default_factory=generate_memory_id)
    agent_id: AgentId
    user_id: UserId | None = None
    session_id: SessionId | None = None

    content: str = Field(..., min_length=1, max_length=100000)
    embedding: Vector | None = None

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)

    created_at: datetime = Field(default_factory=_utcnow)
    last_accessed_at: datetime = Field(default_factory=_utcnow)

    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": False, "extra": "forbid"}

    def __hash__(self) -> int:
        """Hash based on memory_id for set operations."""
        return hash(self.memory_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on memory_id."""
        if isinstance(other, Memory):
            return self.memory_id == other.memory_id
        return False


class MemoryCreate(BaseModel):
    """Input model for creating a new memory.

    This model is used when adding memories via the API. It contains
    only the fields that can be set by the caller.

    Attributes:
        content: The text content of the memory.
        agent_id: ID of the agent this memory belongs to.
        user_id: Optional user ID.
        session_id: Optional session ID.
        metadata: Additional metadata.

    Note:
        Importance starts at 0.5 for all memories. Use reinforce()
        to boost importance when a memory proves useful.

    Example:
        create_input = MemoryCreate(
            content="User mentioned they work in finance",
            agent_id="agent_123",
            metadata={"source": "conversation"}
        )
    """

    content: str = Field(..., min_length=1, max_length=100000)
    agent_id: AgentId
    user_id: UserId | None = None
    session_id: SessionId | None = None
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}


class MemoryUpdate(BaseModel):
    """Input model for updating an existing memory.

    All fields are optional - only provided fields will be updated.

    Attributes:
        content: New content (will trigger re-embedding).
        importance: New importance score.
        metadata: Metadata to merge (not replace).
    """

    content: str | None = Field(default=None, min_length=1, max_length=100000)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: Metadata | None = None

    model_config = {"frozen": True, "extra": "forbid"}


class SearchResult(BaseModel):
    """A memory with search relevance scores.

    Returned by search operations, this includes the memory along with
    various relevance scores used for ranking.

    Attributes:
        memory: The memory object.
        score: Combined relevance score (0.0 to 1.0).
        semantic_score: Semantic similarity score from vector search.
        keyword_score: Keyword matching score from full-text search.
        decay_score: Time decay score based on last access.
    """

    memory: Memory
    score: float = Field(ge=0.0)
    semantic_score: float = Field(default=0.0, ge=0.0)
    keyword_score: float = Field(default=0.0, ge=0.0)
    decay_score: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}


class SearchQuery(BaseModel):
    """Input model for search operations.

    Defines the parameters for searching memories.

    Attributes:
        query: The search query text.
        agent_id: Filter by agent ID.
        user_id: Optional filter by user ID.
        limit: Maximum number of results.
        mode: Search mode (hybrid, semantic, or keyword).
        min_score: Minimum score threshold.
        metadata_filter: Filter by metadata key-value pairs.
    """

    query: str = Field(..., min_length=1, max_length=10000)
    agent_id: AgentId
    user_id: UserId | None = None
    limit: int = Field(default=10, ge=1, le=100)
    mode: str = Field(default="hybrid")  # hybrid, semantic, keyword
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_filter: dict[str, Any] | None = None

    model_config = {"frozen": True, "extra": "forbid"}
