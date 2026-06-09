"""Core module for Engram.

This module provides the foundational components:
- Configuration management
- Exception hierarchy
- Type definitions
"""

from engram.core._types import (
    AgentId,
    MemoryId,
    MemoryType,
    Metadata,
    RelationType,
    SearchMode,
    SessionId,
    TraversalDirection,
    UserId,
    Vector,
    VectorDimension,
)
from engram.core.config import (
    EngramSettings,
    get_settings,
)
from engram.core.exceptions import (
    ConfigurationError,
    ConnectionError,
    ConnectionPoolExhaustedError,
    CyclicRelationError,
    DuplicateMemoryError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingProviderError,
    EngramError,
    GraphError,
    LLMError,
    LLMProviderError,
    MemoryNotFoundError,
    QueryError,
    RelationNotFoundError,
    SessionClosedError,
    SessionError,
    SessionNotFoundError,
    StorageError,
    ValidationError,
)

__all__ = [
    # Types
    "AgentId",
    # Exceptions
    "ConfigurationError",
    "ConnectionError",
    "ConnectionPoolExhaustedError",
    "CyclicRelationError",
    "DuplicateMemoryError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingProviderError",
    "EngramError",
    # Config
    "EngramSettings",
    "GraphError",
    "LLMError",
    "LLMProviderError",
    "MemoryId",
    "MemoryNotFoundError",
    "MemoryType",
    "Metadata",
    "QueryError",
    "RelationNotFoundError",
    "RelationType",
    "SearchMode",
    "SessionClosedError",
    "SessionError",
    "SessionId",
    "SessionNotFoundError",
    "StorageError",
    "TraversalDirection",
    "UserId",
    "ValidationError",
    "Vector",
    "VectorDimension",
    "get_settings",
]
