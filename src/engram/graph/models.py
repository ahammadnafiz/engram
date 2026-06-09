"""Graph models for Engram.

This module defines models for memory relations and graph traversal results.
"""
# ruff: noqa: TC001

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from engram.core._types import MemoryId, Metadata, RelationType


def _utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class MemoryRelation(BaseModel):
    """Represents a relation between two memories.

    Relations form a directed graph where memories can be connected
    with typed edges and weights.

    Attributes:
        source_memory_id: The source memory of the relation.
        target_memory_id: The target memory of the relation.
        relation_type: Type of relation (e.g., "related_to", "causes").
        weight: Strength of the relation (0.0 to 1.0).
        metadata: Additional relation metadata.
        created_at: When the relation was created.

    Example:
        relation = MemoryRelation(
            source_memory_id="mem_abc123",
            target_memory_id="mem_def456",
            relation_type="causes",
            weight=0.8,
        )
    """

    source_memory_id: MemoryId
    target_memory_id: MemoryId
    relation_type: RelationType = "related_to"
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Metadata = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)

    model_config = {"frozen": True, "extra": "forbid"}


class RelationCreate(BaseModel):
    """Input model for creating a relation.

    Attributes:
        source_memory_id: The source memory ID.
        target_memory_id: The target memory ID.
        relation_type: Type of relation.
        weight: Relation weight (default 1.0).
        metadata: Optional metadata.
    """

    source_memory_id: MemoryId
    target_memory_id: MemoryId
    relation_type: RelationType = "related_to"
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Metadata = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}


class TraversalResult(BaseModel):
    """Result from a graph traversal operation.

    Contains the memory along with traversal-specific information
    like depth, path, and accumulated weights.

    Attributes:
        memory_id: The memory ID at this node.
        content: Memory content (alias for fact, backward compatible).
        fact: The extracted user fact (same as content).
        main_content: Optional conversation context (not embedded).
        importance: Memory importance score.
        metadata: Memory metadata.
        access_count: Number of times this memory was accessed.
        depth: Distance from the starting node.
        path: List of memory IDs from start to this node.
        relation_type: The relation type that led to this node.
        path_weight: Accumulated weight along the path.
        score: Combined relevance score.
    """

    memory_id: MemoryId
    content: str
    fact: str | None = None  # New: explicit fact field
    main_content: str | None = None  # New: conversation context
    importance: float
    metadata: Metadata
    created_at: datetime
    last_accessed_at: datetime
    access_count: int = 0  # New: access tracking
    depth: int
    path: list[MemoryId]
    relation_type: str | None
    path_weight: float
    score: float

    model_config = {"frozen": True}


class TraversalQuery(BaseModel):
    """Input parameters for graph traversal.

    Attributes:
        start_memory_id: The memory to start traversal from.
        max_depth: Maximum traversal depth (default 3).
        relation_types: Optional list of relation types to follow.
        direction: Direction of traversal (outbound, inbound, any).
        min_weight: Minimum relation weight to follow (default 0.0).
        limit: Maximum results per depth level.
    """

    start_memory_id: MemoryId
    max_depth: int = Field(default=3, ge=1, le=10)
    relation_types: list[RelationType] | None = None
    direction: str = Field(default="outbound")  # outbound, inbound, any
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=50, ge=1, le=200)

    model_config = {"frozen": True, "extra": "forbid"}
