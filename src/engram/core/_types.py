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
    "supersedes",
    "revision_of",
    "derived_from",
    "reasserts",
]

# ============================================================================
# Memory Type (cognitive taxonomy)
# ============================================================================
# semantic    - generic durable fact
# episodic    - dated narrative event ("what happened")
# procedural  - behavioral rule / how the agent should act
# profile     - identity, location, health, relationships, durable user facts
# project     - project/product facts, owners, codenames, metrics
# task        - requirements and state for a specific unit of work
# preference  - stable user preferences and communication style
# constraint  - hard rules, repo constraints, safety limits, deadlines
# decision    - explicit decisions and corrections
# tool_result - tool outputs, measurements, test results, external observations
MemoryType: TypeAlias = Literal[
    "semantic",
    "episodic",
    "procedural",
    "profile",
    "project",
    "task",
    "preference",
    "constraint",
    "decision",
    "tool_result",
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
