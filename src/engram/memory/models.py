"""Memory models for Engram.

This module defines the Pydantic models for memories and search results.
"""
# ruff: noqa: TC001, TC003

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from engram.core._types import (
    AgentId,
    MemoryId,
    MemoryType,
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

    Two-column memory system:
        - fact: The extracted user fact (embedded for search)
        - main_content: Full conversation context (not embedded)

    Attributes:
        memory_id: Unique identifier for the memory.
        agent_id: ID of the agent this memory belongs to.
        user_id: Optional ID of the user associated with this memory.
        session_id: Optional ID of the session where this memory was created.
        content: Backward-compatible field (maps to fact).
        fact: The extracted user fact (embedded for hybrid search).
        main_content: Optional conversation context [USER]: msg\\n[AI]: summary.
        embedding: Vector embedding of the fact (None if not yet computed).
        importance: Importance score from 0.0 to 1.0 (default 0.5).
        access_count: Number of times this memory has been accessed.
        created_at: Timestamp when the memory was created.
        last_accessed_at: Timestamp of last access.
        metadata: Additional key-value metadata.

    Example:
        memory = Memory(
            agent_id="agent_123",
            content="User prefers dark mode",
            fact="User prefers dark mode",
            main_content="[USER]: I like dark mode\\n[AI]: Got it!",
            importance=0.8,
            metadata={"category": "preference"}
        )
    """

    memory_id: MemoryId = Field(default_factory=generate_memory_id)
    agent_id: AgentId
    user_id: UserId | None = None
    session_id: SessionId | None = None

    # Backward-compatible content field (maps to fact)
    content: str = Field(..., min_length=1, max_length=100000)

    # Two-column memory system
    fact: str | None = Field(default=None, min_length=1, max_length=100000)
    main_content: str | None = Field(default=None, max_length=200000)

    # Cognitive taxonomy: semantic (facts) | episodic (events) | procedural (rules)
    memory_type: MemoryType = "semantic"

    embedding: Vector | None = None

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)

    created_at: datetime = Field(default_factory=_utcnow)
    last_accessed_at: datetime = Field(default_factory=_utcnow)

    metadata: Metadata = Field(default_factory=dict)

    # Version lineage. ``agent_memory`` remains the fast read model; these
    # fields make corrections auditable without scanning JSON metadata.
    lineage_id: str | None = None
    revision: int = Field(default=1, ge=1)
    status: str = "active"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    superseded_by_memory_id: MemoryId | None = None
    superseded_at: datetime | None = None

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

    Two-column memory system:
        - content: The fact to store (will be embedded for search)
        - main_content: Optional conversation context (not embedded)

    Attributes:
        content: The user fact to store (embedded for hybrid search).
        main_content: Optional conversation context [USER]: msg\\n[AI]: summary.
        agent_id: ID of the agent this memory belongs to.
        user_id: Optional user ID.
        session_id: Optional session ID.
        metadata: Additional metadata.

    Note:
        Importance starts at 0.5 for all memories. Use reinforce()
        to boost importance when a memory proves useful.

    Example:
        create_input = MemoryCreate(
            content="User works in finance",
            main_content="[USER]: I work in finance\\n[AI]: Interesting field!",
            agent_id="agent_123",
            metadata={"source": "conversation"}
        )
    """

    content: str = Field(..., min_length=1, max_length=100000)
    main_content: str | None = Field(default=None, max_length=200000)
    agent_id: AgentId
    user_id: UserId | None = None
    session_id: SessionId | None = None
    memory_type: MemoryType = "semantic"
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


class MemoryLineage(BaseModel):
    """Version history for one memory lineage.

    The first-class lineage keeps the current fact fast to retrieve while
    preserving older corrected facts for audit/debugging.
    """

    lineage_id: str
    current_memory_id: MemoryId | None = None
    memories: list[Memory] = Field(default_factory=list)

    model_config = {"frozen": True}


class MemoryExplanation(BaseModel):
    """Debug view for why a memory is current or superseded."""

    memory: Memory
    current: Memory
    lineage: MemoryLineage
    supersedes: list[Memory] = Field(default_factory=list)
    superseded_by: Memory | None = None

    model_config = {"frozen": True}


class MemoryHistoryEvent(BaseModel):
    """One user-facing memory timeline event.

    History events are derived from the durable memory rows and lineage state:
    additions/revisions use the new row's ``created_at`` timestamp, while
    supersession events use the old row's ``superseded_at`` timestamp.
    """

    event_type: Literal["added", "revised", "superseded"]
    occurred_at: datetime
    memory: Memory
    current_memory_id: MemoryId | None = None
    previous_memory_id: MemoryId | None = None
    superseded_by_memory_id: MemoryId | None = None
    reason: str | None = None
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True}


@dataclass(frozen=True)
class FactDecision:
    """The outcome of one extracted fact in add_conversation().

    Makes every decision visible -- including NOOPs and unapplied corrections --
    so a caller can tell "correctly skipped a duplicate" apart from "silently
    dropped a real update". ``applied`` is True only when a memory row was
    actually written or revised.
    """

    fact: str
    operation: str  # "ADD" | "UPDATE" | "DELETE" | "NOOP"
    applied: bool
    reason: str = ""
    memory_id: MemoryId | None = None  # written/affected row, when applied
    target_id: MemoryId | None = None  # superseded row, for UPDATE/DELETE


@dataclass
class ConversationResult:
    """Return value of add_conversation().

    Backward-compatible: iterating, ``len()``-ing, or truth-testing this yields
    the WRITTEN memories (ADD + UPDATE/DELETE), exactly like the old
    ``list[Memory]`` return. The new ``decisions`` field exposes the per-fact
    operation + reason for EVERY extracted fact, so an intended update that
    resolved to NOOP is visible here instead of being silently absent from the
    returned list.
    """

    memories: list[Memory] = field(default_factory=list)
    decisions: list[FactDecision] = field(default_factory=list)

    def __iter__(self) -> Iterator[Memory]:
        return iter(self.memories)

    def __len__(self) -> int:
        return len(self.memories)

    def __bool__(self) -> bool:
        return bool(self.memories)

    def __getitem__(self, index: int) -> Memory:
        return self.memories[index]


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


class RecallTrace(BaseModel):
    """Observability record for one retrieval/context assembly operation.

    The trace answers the production debugging questions that matter for
    memory: was the fact stored, was it pinned as critical, did search rank it,
    was it kept in the final prompt budget, was it trimmed, or was it hidden by
    conflict resolution as superseded.
    """

    query: str
    agent_id: AgentId
    user_id: UserId | None = None
    critical_memory_ids: list[MemoryId] = Field(default_factory=list)
    search_memory_ids: list[MemoryId] = Field(default_factory=list)
    ranked_memory_ids: list[MemoryId] = Field(default_factory=list)
    kept_memory_ids: list[MemoryId] = Field(default_factory=list)
    trimmed_memory_ids: list[MemoryId] = Field(default_factory=list)
    superseded_memory_ids: list[MemoryId] = Field(default_factory=list)
    missing_expected_terms: list[str] = Field(default_factory=list)
    context: str = ""
    notes: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)

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
    limit: int = Field(default=10, ge=1, le=1000)
    mode: str = Field(default="hybrid")  # hybrid, semantic, keyword
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_filter: dict[str, Any] | None = None
    memory_types: list[MemoryType] | None = None
    # When True, superseded (historical) revisions are included in results,
    # each labeled by its ``status``. Default False keeps normal recall to
    # active facts only.
    include_superseded: bool = False

    model_config = {"frozen": True, "extra": "forbid"}
