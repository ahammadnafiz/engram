"""Engram client - Main entry point for the AI memory library.

This module provides the Engram class, the main async client for
interacting with the AI memory system.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from engram.core._types import (
    AgentId,
    MemoryId,
    Metadata,
    RelationType,
    SessionId,
    UserId,
)
from engram.core.config import EngramSettings, get_settings
from engram.core.exceptions import EngramError
from engram.embedding.service import EmbeddingService
from engram.graph.models import TraversalQuery, TraversalResult
from engram.graph.traversal import GraphTraversal
from engram.health.checker import HealthChecker
from engram.memory.models import (
    Memory,
    MemoryCreate,
    MemoryUpdate,
    SearchQuery,
    SearchResult,
)
from engram.memory.store import MemoryStore
from engram.session.manager import SessionManager
from engram.session.models import Session
from engram.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


class Engram:
    """Main async client for the Engram AI memory library.

    Engram provides a unified interface for AI memory management including:
    - Adding and retrieving memories
    - Hybrid search (vector + keyword + decay + importance)
    - Graph relations and traversal
    - Session management
    - Health checking

    Example:
        # Using async context manager (recommended)
        async with Engram() as engram:
            # Add a memory
            memory = await engram.add(
                content="User prefers dark mode",
                agent_id="my_agent",
                importance=0.8,
            )
            
            # Search memories
            results = await engram.search(
                query="user preferences",
                agent_id="my_agent",
            )
            
            # Create relations
            await engram.relate(
                source_id=memory.memory_id,
                target_id=other_memory_id,
                relation_type="related_to",
            )

        # Or manual lifecycle management
        engram = Engram()
        await engram.connect()
        try:
            # ... use engram ...
        finally:
            await engram.close()
    """

    def __init__(
        self,
        settings: EngramSettings | None = None,
        *,
        database_url: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        """Initialize the Engram client.

        Args:
            settings: Full settings object. If None, loads from environment.
            database_url: Override database URL from settings.
            openai_api_key: Override OpenAI API key from settings.
        """
        self._settings = settings or get_settings()

        # Apply overrides
        if database_url:
            self._settings = self._settings.model_copy(
                update={"database_url": database_url}
            )
        if openai_api_key:
            self._settings = self._settings.model_copy(
                update={"openai_api_key": openai_api_key}
            )

        # Initialize components (lazy, connected on connect())
        self._storage: PostgresStorage | None = None
        self._embedding: EmbeddingService | None = None
        self._memory_store: MemoryStore | None = None
        self._graph: GraphTraversal | None = None
        self._sessions: SessionManager | None = None
        self._health: HealthChecker | None = None

        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to the database and initialize services.

        This method must be called before using any other methods
        unless using the async context manager.

        Raises:
            ConnectionError: If connection fails.
            ConfigurationError: If configuration is invalid.
        """
        if self._connected:
            logger.warning("Already connected")
            return

        logger.info("Connecting to Engram")

        # Initialize embedding service FIRST to get dimension
        self._embedding = EmbeddingService.from_settings(self._settings)
        embedding_dimension = self._embedding.dimension
        logger.info(f"Embedding dimension detected: {embedding_dimension}")

        # Initialize storage
        self._storage = PostgresStorage(self._settings)
        await self._storage.connect()

        # Initialize schema with auto-detected embedding dimension
        await self._storage.init_schema(embedding_dimension=embedding_dimension)

        # Initialize higher-level services
        self._memory_store = MemoryStore(self._storage, self._embedding)
        self._graph = GraphTraversal(self._storage)
        self._sessions = SessionManager(self._storage)
        self._health = HealthChecker(self._storage, self._embedding)

        self._connected = True
        logger.info("Connected to Engram successfully")

    async def close(self) -> None:
        """Close connections and cleanup resources.

        This method should be called when done using the client
        unless using the async context manager.
        """
        if not self._connected:
            return

        logger.info("Closing Engram connection")

        if self._storage:
            await self._storage.close()

        self._storage = None
        self._embedding = None
        self._memory_store = None
        self._graph = None
        self._sessions = None
        self._health = None
        self._connected = False

        logger.info("Engram connection closed")

    async def __aenter__(self) -> Engram:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_connected(self) -> None:
        """Ensure the client is connected."""
        if not self._connected:
            raise EngramError("Not connected. Call connect() first or use async with.")

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def add(
        self,
        content: str,
        agent_id: AgentId,
        *,
        main_content: str | None = None,
        user_id: UserId | None = None,
        session_id: SessionId | None = None,
        metadata: Metadata | None = None,
    ) -> Memory:
        """Add a new memory.

        Two-column memory system:
        - content: The fact to store (embedded for hybrid search)
        - main_content: Optional conversation context (not embedded)

        All memories start with importance=0.5. Use reinforce() to boost
        importance when a memory proves useful.

        Args:
            content: The user fact to store (embedded for search).
            agent_id: ID of the agent this memory belongs to.
            main_content: Optional conversation context [USER]: msg\\n[AI]: summary.
            user_id: Optional user ID.
            session_id: Optional session ID.
            metadata: Additional key-value metadata.

        Returns:
            The created memory.

        Example:
            memory = await engram.add(
                content="User works in finance",
                main_content="[USER]: I work in finance\\n[AI]: Interesting field!",
                agent_id="my_agent",
                metadata={"source": "conversation"}
            )
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.add(
            MemoryCreate(
                content=content,
                main_content=main_content,
                agent_id=agent_id,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
        )

    async def add_batch(
        self,
        memories: list[dict[str, Any]],
    ) -> list[Memory]:
        """Add multiple memories in a batch.

        More efficient than calling add() multiple times due to
        batch embedding. Only content (fact) is embedded, not main_content.

        Args:
            memories: List of memory dictionaries with keys:
                - content (required): The user fact (embedded for search)
                - agent_id (required): Agent ID
                - main_content (optional): Conversation context (not embedded)
                - user_id (optional): User ID
                - session_id (optional): Session ID
                - metadata (optional): Additional metadata

        Returns:
            List of created memories.

        Example:
            memories = await engram.add_batch([
                {"content": "Fact 1", "agent_id": "agent_1"},
                {"content": "Fact 2", "agent_id": "agent_1", 
                 "main_content": "[USER]: I work...\\n[AI]: Got it!"},
            ])
        """
        self._ensure_connected()
        assert self._memory_store is not None

        creates = [
            MemoryCreate(
                content=m["content"],
                main_content=m.get("main_content"),
                agent_id=m["agent_id"],
                user_id=m.get("user_id"),
                session_id=m.get("session_id"),
                metadata=m.get("metadata", {}),
            )
            for m in memories
        ]

        return await self._memory_store.add_batch(creates)

    async def get(self, memory_id: MemoryId) -> Memory:
        """Get a memory by ID.

        Also updates the access timestamp and count.

        Args:
            memory_id: The memory ID to retrieve.

        Returns:
            The memory.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.get(memory_id)

    async def update(
        self,
        memory_id: MemoryId,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: Metadata | None = None,
    ) -> Memory:
        """Update an existing memory.

        Args:
            memory_id: The memory to update.
            content: New content (triggers re-embedding).
            importance: New importance score.
            metadata: Metadata to merge (not replace).

        Returns:
            The updated memory.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.update(
            memory_id,
            MemoryUpdate(
                content=content,
                importance=importance,
                metadata=metadata,
            ),
        )

    async def reinforce(
        self,
        memory_id: MemoryId,
        importance_boost: float = 0.1,
    ) -> Memory:
        """Reinforce a memory by boosting its importance.

        Args:
            memory_id: The memory to reinforce.
            importance_boost: Amount to increase importance (capped at 1.0).

        Returns:
            The reinforced memory.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.reinforce(memory_id, importance_boost)

    async def forget(self, memory_id: MemoryId) -> bool:
        """Delete a single memory.

        Args:
            memory_id: The memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.forget(memory_id)

    async def purge(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
    ) -> int:
        """Delete all memories for an agent.

        Args:
            agent_id: The agent whose memories to delete.
            user_id: Optional user to filter by.

        Returns:
            Number of memories deleted.
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.purge(agent_id, user_id)

    async def list_recent(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """List recent memories for an agent.

        Args:
            agent_id: The agent ID to filter by.
            user_id: Optional user ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of memories ordered by creation time (newest first).
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.list_recent(agent_id, user_id, limit)

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search memories using hybrid search.

        Combines vector similarity, keyword matching, time decay,
        and importance scoring.

        Args:
            query: The search query text.
            agent_id: Filter by agent ID.
            user_id: Optional filter by user ID.
            limit: Maximum number of results.
            min_score: Minimum score threshold.

        Returns:
            List of search results with scores.

        Example:
            results = await engram.search(
                query="user preferences for UI",
                agent_id="my_agent",
                limit=5,
            )
            for result in results:
                print(f"{result.score:.2f}: {result.memory.content}")
        """
        self._ensure_connected()
        assert self._memory_store is not None

        return await self._memory_store.search(
            SearchQuery(
                query=query,
                agent_id=agent_id,
                user_id=user_id,
                limit=limit,
                min_score=min_score,
            )
        )

    # =========================================================================
    # Graph Operations
    # =========================================================================

    async def relate(
        self,
        source_id: MemoryId,
        target_id: MemoryId,
        relation_type: RelationType = "related_to",
        weight: float = 1.0,
        metadata: Metadata | None = None,
    ) -> None:
        """Create a relation between two memories.

        Args:
            source_id: Source memory ID.
            target_id: Target memory ID.
            relation_type: Type of relation.
            weight: Relation weight (0.0 to 1.0).
            metadata: Optional relation metadata.

        Raises:
            MemoryNotFoundError: If either memory doesn't exist.
        """
        self._ensure_connected()
        assert self._graph is not None

        await self._graph.relate(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            metadata=metadata,
        )

    async def traverse(
        self,
        start_memory_id: MemoryId,
        max_depth: int = 3,
        direction: str = "outbound",
        relation_types: list[RelationType] | None = None,
        min_weight: float = 0.0,
        limit: int = 50,
    ) -> list[TraversalResult]:
        """Traverse the memory graph from a starting point.

        Uses recursive CTEs for efficient multi-hop traversal.

        Args:
            start_memory_id: The memory to start from.
            max_depth: Maximum traversal depth.
            direction: Direction (outbound, inbound, any).
            relation_types: Optional filter by relation types.
            min_weight: Minimum relation weight to follow.
            limit: Maximum results.

        Returns:
            List of traversal results with depth and path info.

        Example:
            results = await engram.traverse(
                start_memory_id="mem_abc123",
                max_depth=2,
                direction="outbound",
            )
            for r in results:
                print(f"Depth {r.depth}: {r.content}")
        """
        self._ensure_connected()
        assert self._graph is not None

        return await self._graph.traverse(
            TraversalQuery(
                start_memory_id=start_memory_id,
                max_depth=max_depth,
                direction=direction,
                relation_types=relation_types,
                min_weight=min_weight,
                limit=limit,
            )
        )

    # =========================================================================
    # Session Operations
    # =========================================================================

    @asynccontextmanager
    async def session(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
        metadata: Metadata | None = None,
    ) -> AsyncIterator[Session]:
        """Create a session context manager.

        The session is automatically ended when the context exits.

        Args:
            agent_id: The agent ID.
            user_id: Optional user ID.
            metadata: Optional session metadata.

        Yields:
            The active session.

        Example:
            async with engram.session(agent_id="my_agent") as sess:
                memory = await engram.add(
                    content="In-session memory",
                    agent_id="my_agent",
                    session_id=sess.session_id,
                )
        """
        self._ensure_connected()
        assert self._sessions is not None

        async with self._sessions.session(agent_id, user_id, metadata) as sess:
            yield sess

    # =========================================================================
    # Health Operations
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Perform a comprehensive health check.

        Returns:
            Dictionary with health status and component details.

        Example:
            status = await engram.health_check()
            if status["status"] == "healthy":
                print("All systems operational")
        """
        self._ensure_connected()
        assert self._health is not None

        return await self._health.check()
