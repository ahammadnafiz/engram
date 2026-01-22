"""Type definitions for Engram.

This module contains all type aliases and type definitions used throughout
the Engram library for consistent typing and better IDE support.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

# ============================================================================
# ID Types
# ============================================================================
AgentId: TypeAlias = str
UserId: TypeAlias = str
SessionId: TypeAlias = str
MemoryId: TypeAlias = str

# ============================================================================
# Vector Types
# ============================================================================
Vector: TypeAlias = list[float]
# VectorDimension is now just int - any dimension is supported via provider system
VectorDimension: TypeAlias = int

# ============================================================================
# Relation Types
# ============================================================================
RelationType: TypeAlias = Literal[
    "related_to",
    "causes",
    "caused_by",
    "contradicts",
    "supports",
    "precedes",
    "follows",
    "part_of",
    "contains",
    "references",
    "mentioned_with",
]

# ============================================================================
# Search Types
# ============================================================================
SearchMode: TypeAlias = Literal["hybrid", "semantic", "keyword"]

# ============================================================================
# Metadata Types
# ============================================================================
Metadata: TypeAlias = dict[str, Any]

# ============================================================================
# Graph Traversal Types
# ============================================================================
TraversalDirection: TypeAlias = Literal["outbound", "inbound", "any"]
