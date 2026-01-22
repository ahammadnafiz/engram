"""Exception hierarchy for Engram.

This module defines all custom exceptions used throughout the Engram library.
All exceptions inherit from EngramError for easy catching of library-specific errors.
"""

from __future__ import annotations

from typing import Any


class EngramError(Exception):
    """Base exception for all Engram errors.

    All Engram-specific exceptions inherit from this class, allowing users
    to catch all library errors with a single except clause.

    Example:
        try:
            await engram.search("query")
        except EngramError as e:
            logger.error(f"Engram operation failed: {e}")
    """

    def __init__(self, message: str, **context: Any) -> None:
        """Initialize the exception with a message and optional context.

        Args:
            message: Human-readable error description.
            **context: Additional context key-value pairs for debugging.
        """
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        """Return string representation with context if available."""
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx})"
        return self.message


# ============================================================================
# Connection Errors
# ============================================================================


class DatabaseConnectionError(EngramError):
    """Raised when database connection fails.

    This error is raised when:
    - Initial connection to PostgreSQL fails
    - Connection pool exhaustion occurs
    - Connection times out during operation
    
    Note: Named DatabaseConnectionError to avoid shadowing Python's built-in
    ConnectionError, which is a subclass of OSError.
    """

    pass


# Alias for backwards compatibility (but prefer DatabaseConnectionError)
ConnectionError = DatabaseConnectionError  # noqa: A001


class ConnectionPoolExhaustedError(DatabaseConnectionError):
    """Raised when connection pool is exhausted.

    This typically indicates that:
    - Too many concurrent operations are in progress
    - Connections are not being released properly
    - Pool size is too small for the workload
    """

    pass


# ============================================================================
# Storage Errors
# ============================================================================


class StorageError(EngramError):
    """Base class for storage-related errors.

    This is the parent class for all errors related to data persistence
    and retrieval operations.
    """

    pass


class MemoryNotFoundError(StorageError):
    """Raised when a requested memory does not exist.

    This error is raised when:
    - Attempting to get a memory by ID that doesn't exist
    - Attempting to update a non-existent memory
    - Attempting to delete a non-existent memory
    """

    def __init__(self, memory_id: str) -> None:
        """Initialize with the missing memory ID.

        Args:
            memory_id: The ID of the memory that was not found.
        """
        super().__init__(f"Memory not found: {memory_id}", memory_id=memory_id)
        self.memory_id = memory_id


class DuplicateMemoryError(StorageError):
    """Raised when attempting to create a duplicate memory.

    This error may be raised when:
    - Memory with the same ID already exists
    - Deduplication detects semantically identical content
    """

    pass


class QueryError(StorageError):
    """Raised when a database query fails.

    This error wraps underlying database errors and provides
    additional context about the failed operation.
    """

    pass


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(EngramError):
    """Raised when input validation fails.

    This error is raised when:
    - Required fields are missing
    - Field values are out of valid range
    - Data format is incorrect
    """

    pass


class ConfigurationError(ValidationError):
    """Raised when configuration is invalid.

    This error is raised during initialization when:
    - Required configuration is missing
    - Configuration values are invalid
    - Incompatible configuration options are used
    """

    pass


# ============================================================================
# Embedding Errors
# ============================================================================


class EmbeddingError(EngramError):
    """Base class for embedding-related errors."""

    pass


class EmbeddingProviderError(EmbeddingError):
    """Raised when embedding provider fails.

    This error is raised when:
    - OpenAI API call fails
    - Sentence Transformers model fails
    - Rate limiting or quota exceeded
    """

    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match expected size.

    This typically indicates:
    - Wrong model configuration
    - Model changed between operations
    - Corrupted embeddings in storage
    """

    def __init__(self, expected: int, actual: int) -> None:
        """Initialize with expected and actual dimensions.

        Args:
            expected: Expected embedding dimension.
            actual: Actual embedding dimension received.
        """
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, got {actual}",
            expected=expected,
            actual=actual,
        )
        self.expected = expected
        self.actual = actual


# ============================================================================
# Session Errors
# ============================================================================


class SessionError(EngramError):
    """Base class for session-related errors."""

    pass


class SessionNotFoundError(SessionError):
    """Raised when a session does not exist."""

    def __init__(self, session_id: str) -> None:
        """Initialize with the missing session ID.

        Args:
            session_id: The ID of the session that was not found.
        """
        super().__init__(f"Session not found: {session_id}", session_id=session_id)
        self.session_id = session_id


class SessionClosedError(SessionError):
    """Raised when attempting to use a closed session."""

    pass


# ============================================================================
# Graph Errors
# ============================================================================


class GraphError(EngramError):
    """Base class for graph-related errors."""

    pass


class CyclicRelationError(GraphError):
    """Raised when a relation would create a cycle.

    Note: This is only raised when cycle detection is enabled,
    as some use cases may intentionally allow cycles.
    """

    pass


class RelationNotFoundError(GraphError):
    """Raised when a relation does not exist."""

    pass
