"""Core module for Engram.

This module provides the foundational components:
- Configuration management
- Exception hierarchy
- Type definitions
"""

from engram.core._types import (
    AgentId,
    MemoryId,
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
    DatabaseSettings,
    EmbeddingSettings,
    EngramSettings,
    SearchSettings,
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
    "MemoryId",
    "Metadata",
    "RelationType",
    "SearchMode",
    "SessionId",
    "TraversalDirection",
    "UserId",
    "Vector",
    "VectorDimension",
    # Config
    "DatabaseSettings",
    "EmbeddingSettings",
    "EngramSettings",
    "SearchSettings",
    "get_settings",
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
    "GraphError",
    "LLMError",
    "LLMProviderError",
    "MemoryNotFoundError",
    "QueryError",
    "RelationNotFoundError",
    "SessionClosedError",
    "SessionError",
    "SessionNotFoundError",
    "StorageError",
    "ValidationError",
]
