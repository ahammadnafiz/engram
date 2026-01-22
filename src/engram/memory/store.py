"""Memory store for Engram.

This module provides the MemoryStore class for CRUD operations on memories.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from engram.core._types import AgentId, MemoryId, Metadata, UserId
from engram.core.exceptions import MemoryNotFoundError, QueryError, StorageError
from engram.memory.models import (
    Memory,
    MemoryCreate,
    MemoryUpdate,
    SearchQuery,
    SearchResult,
)

if TYPE_CHECKING:
    from engram.embedding.service import EmbeddingService
    from engram.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


class MemoryStore:
    """Storage operations for memories.

    This class provides CRUD operations for memories including add, get,
    update, delete, search, and batch operations. It uses the PostgresStorage
    backend and EmbeddingService for vector operations.

    Example:
        store = MemoryStore(storage, embedding_service)
        
        # Add a memory
        memory = await store.add(MemoryCreate(
            content="User prefers dark mode",
            agent_id="agent_123",
        ))
        
        # Search memories
        results = await store.search(SearchQuery(
            query="user preferences",
            agent_id="agent_123",
        ))
    """

    def __init__(
        self,
        storage: PostgresStorage,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize the memory store.

        Args:
            storage: PostgreSQL storage backend.
            embedding_service: Service for computing embeddings.
        """
        self._storage = storage
        self._embedding = embedding_service

    async def _ensure_agent_exists(self, agent_id: AgentId) -> None:
        """Ensure agent exists, creating if necessary."""
        await self._storage.execute(
            """
            INSERT INTO agents (agent_id, name)
            VALUES ($1, $2)
            ON CONFLICT (agent_id) DO NOTHING
            """,
            agent_id,
            agent_id,  # Use agent_id as default name
        )

    async def _ensure_user_exists(self, user_id: UserId) -> None:
        """Ensure user exists, creating if necessary."""
        await self._storage.execute(
            """
            INSERT INTO users (user_id)
            VALUES ($1)
            ON CONFLICT (user_id) DO NOTHING
            """,
            user_id,
        )

    async def add(self, create: MemoryCreate) -> Memory:
        """Add a new memory.

        Args:
            create: Memory creation input.

        Returns:
            The created memory with generated ID and embedding.

        Raises:
            StorageError: If database operation fails.
            EmbeddingError: If embedding computation fails.
        """
        # Generate embedding for content
        embedding = await self._embedding.embed(create.content)

        memory = Memory(
            agent_id=create.agent_id,
            user_id=create.user_id,
            session_id=create.session_id,
            content=create.content,
            embedding=embedding,
            importance=0.5,  # Default importance, use reinforce() to boost
            metadata=create.metadata,
        )

        try:
            # Auto-create agent and user if they don't exist
            await self._ensure_agent_exists(create.agent_id)
            if create.user_id:
                await self._ensure_user_exists(create.user_id)

            # Use RETURNING to detect if insert actually happened
            row = await self._storage.fetchone(
                """
                INSERT INTO agent_memory (
                    memory_id, agent_id, user_id, session_id,
                    content, embedding, importance, metadata,
                    created_at, last_accessed_at, access_count
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (agent_id, COALESCE(user_id, ''), content) DO UPDATE
                    SET memory_id = agent_memory.memory_id
                RETURNING memory_id, 
                    (xmax = 0) AS was_inserted
                """,
                memory.memory_id,
                memory.agent_id,
                memory.user_id,
                memory.session_id,
                memory.content,
                json.dumps(memory.embedding),  # Store as JSON for vector type
                memory.importance,
                json.dumps(memory.metadata),
                memory.created_at,
                memory.last_accessed_at,
                memory.access_count,
            )
            
            if row and not row["was_inserted"]:
                # Memory with same content already exists
                logger.debug(f"Memory with same content already exists, returning existing: {row['memory_id']}")
                # Return the existing memory
                existing = await self.get(row["memory_id"])
                if existing:
                    return existing
            
            logger.debug(f"Added memory {memory.memory_id}")
            return memory

        except Exception as e:
            raise StorageError(f"Failed to add memory: {e}") from e

    async def add_batch(
        self,
        creates: list[MemoryCreate],
    ) -> list[Memory]:
        """Add multiple memories in a batch.

        Uses batch embedding for efficiency when adding many memories.

        Args:
            creates: List of memory creation inputs.

        Returns:
            List of created memories.

        Raises:
            StorageError: If database operation fails.
        """
        if not creates:
            return []

        # Ensure all agents and users exist first
        agent_ids = set(c.agent_id for c in creates)
        user_ids = set(c.user_id for c in creates if c.user_id)
        
        for agent_id in agent_ids:
            await self._ensure_agent_exists(agent_id)
        for user_id in user_ids:
            await self._ensure_user_exists(user_id)

        # Batch embed all content
        contents = [c.content for c in creates]
        embeddings = await self._embedding.embed_batch(contents)

        memories: list[Memory] = []
        for create, embedding in zip(creates, embeddings, strict=True):
            memory = Memory(
                agent_id=create.agent_id,
                user_id=create.user_id,
                session_id=create.session_id,
                content=create.content,
                embedding=embedding,
                importance=0.5,  # Default importance, use reinforce() to boost
                metadata=create.metadata,
            )
            memories.append(memory)

        # Batch insert (skip duplicates)
        try:
            await self._storage.executemany(
                """
                INSERT INTO agent_memory (
                    memory_id, agent_id, user_id, session_id,
                    content, embedding, importance, metadata,
                    created_at, last_accessed_at, access_count
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (agent_id, COALESCE(user_id, ''), content) DO NOTHING
                """,
                [
                    (
                        m.memory_id,
                        m.agent_id,
                        m.user_id,
                        m.session_id,
                        m.content,
                        json.dumps(m.embedding),
                        m.importance,
                        json.dumps(m.metadata),
                        m.created_at,
                        m.last_accessed_at,
                        m.access_count,
                    )
                    for m in memories
                ],
            )
            logger.debug(f"Added {len(memories)} memories in batch")
            return memories

        except Exception as e:
            raise StorageError(f"Failed to add memories in batch: {e}") from e

    async def get(self, memory_id: MemoryId) -> Memory:
        """Get a memory by ID.

        Also updates the last_accessed_at timestamp and access_count.

        Args:
            memory_id: The memory ID to retrieve.

        Returns:
            The memory object.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
        """
        row = await self._storage.fetchone(
            """
            UPDATE agent_memory
            SET last_accessed_at = NOW(), access_count = access_count + 1
            WHERE memory_id = $1
            RETURNING 
                memory_id, agent_id, user_id, session_id,
                content, embedding, importance, access_count,
                created_at, last_accessed_at, metadata
            """,
            memory_id,
        )

        if row is None:
            raise MemoryNotFoundError(memory_id)

        return self._row_to_memory(row)

    async def get_without_access_update(self, memory_id: MemoryId) -> Memory:
        """Get a memory by ID without updating access timestamp.

        Args:
            memory_id: The memory ID to retrieve.

        Returns:
            The memory object.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
        """
        row = await self._storage.fetchone(
            """
            SELECT 
                memory_id, agent_id, user_id, session_id,
                content, embedding, importance, access_count,
                created_at, last_accessed_at, metadata
            FROM agent_memory
            WHERE memory_id = $1
            """,
            memory_id,
        )

        if row is None:
            raise MemoryNotFoundError(memory_id)

        return self._row_to_memory(row)

    async def update(
        self,
        memory_id: MemoryId,
        update: MemoryUpdate,
    ) -> Memory:
        """Update an existing memory.

        Args:
            memory_id: The memory to update.
            update: Fields to update.

        Returns:
            The updated memory.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
            StorageError: If update fails.
        """
        # Get current memory
        current = await self.get_without_access_update(memory_id)

        # Build update
        new_content = update.content if update.content else current.content
        new_importance = (
            update.importance if update.importance is not None else current.importance
        )
        new_metadata = {**current.metadata, **(update.metadata or {})}

        # Re-embed if content changed
        new_embedding = current.embedding
        if update.content and update.content != current.content:
            new_embedding = await self._embedding.embed(update.content)

        try:
            row = await self._storage.fetchone(
                """
                UPDATE agent_memory
                SET 
                    content = $2,
                    embedding = $3,
                    importance = $4,
                    metadata = $5,
                    last_accessed_at = NOW()
                WHERE memory_id = $1
                RETURNING 
                    memory_id, agent_id, user_id, session_id,
                    content, embedding, importance, access_count,
                    created_at, last_accessed_at, metadata
                """,
                memory_id,
                new_content,
                json.dumps(new_embedding),
                new_importance,
                json.dumps(new_metadata),
            )

            if row is None:
                raise MemoryNotFoundError(memory_id)

            return self._row_to_memory(row)

        except MemoryNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to update memory: {e}") from e

    async def reinforce(
        self,
        memory_id: MemoryId,
        importance_boost: float = 0.1,
    ) -> Memory:
        """Reinforce a memory by boosting its importance.

        Args:
            memory_id: The memory to reinforce.
            importance_boost: Amount to increase importance (must be >= 0, capped at 1.0).

        Returns:
            The reinforced memory.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
            ValidationError: If importance_boost is negative.
        """
        from engram.core.exceptions import ValidationError
        
        if importance_boost < 0:
            raise ValidationError(
                f"importance_boost must be non-negative, got {importance_boost}",
                importance_boost=importance_boost,
            )
        
        row = await self._storage.fetchone(
            """
            UPDATE agent_memory
            SET 
                importance = GREATEST(0.0, LEAST(importance + $2, 1.0)),
                last_accessed_at = NOW(),
                access_count = access_count + 1
            WHERE memory_id = $1
            RETURNING 
                memory_id, agent_id, user_id, session_id,
                content, embedding, importance, access_count,
                created_at, last_accessed_at, metadata
            """,
            memory_id,
            importance_boost,
        )

        if row is None:
            raise MemoryNotFoundError(memory_id)

        return self._row_to_memory(row)

    async def forget(self, memory_id: MemoryId) -> bool:
        """Delete a single memory.

        Args:
            memory_id: The memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        result = await self._storage.execute(
            "DELETE FROM agent_memory WHERE memory_id = $1",
            memory_id,
        )
        deleted = result == "DELETE 1"
        if deleted:
            logger.debug(f"Deleted memory {memory_id}")
        return deleted

    async def purge(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
    ) -> int:
        """Delete all memories for an agent (and optionally user).

        Args:
            agent_id: The agent whose memories to delete.
            user_id: Optional user to filter by.

        Returns:
            Number of memories deleted.
        """
        if user_id:
            result = await self._storage.execute(
                "DELETE FROM agent_memory WHERE agent_id = $1 AND user_id = $2",
                agent_id,
                user_id,
            )
        else:
            result = await self._storage.execute(
                "DELETE FROM agent_memory WHERE agent_id = $1",
                agent_id,
            )

        # Parse "DELETE N" result safely
        count = 0
        if result:
            try:
                parts = result.split()
                if len(parts) >= 2 and parts[0] == "DELETE":
                    count = int(parts[1])
            except (ValueError, IndexError) as e:
                logger.warning(f"Unexpected DELETE result format: {result!r} - {e}")
        
        logger.info(f"Purged {count} memories for agent {agent_id}")
        return count

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
        if user_id:
            rows = await self._storage.fetchall(
                """
                SELECT 
                    memory_id, agent_id, user_id, session_id,
                    content, embedding, importance, access_count,
                    created_at, last_accessed_at, metadata
                FROM agent_memory
                WHERE agent_id = $1 AND user_id = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                agent_id,
                user_id,
                limit,
            )
        else:
            rows = await self._storage.fetchall(
                """
                SELECT 
                    memory_id, agent_id, user_id, session_id,
                    content, embedding, importance, access_count,
                    created_at, last_accessed_at, metadata
                FROM agent_memory
                WHERE agent_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                agent_id,
                limit,
            )

        return [self._row_to_memory(row) for row in rows]

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search memories using specified search mode.

        Supports three modes:
        - hybrid: Combines vector similarity, keyword matching, time decay, and importance
        - semantic: Pure vector similarity search
        - keyword: Text-based keyword search (BM25-style)

        Args:
            query: Search query parameters including mode.

        Returns:
            List of search results with scores.

        Raises:
            QueryError: If search fails.
            ValidationError: If search mode is invalid.
        """
        mode = query.mode.lower()
        valid_modes = {"hybrid", "semantic", "keyword"}
        if mode not in valid_modes:
            from engram.core.exceptions import ValidationError
            raise ValidationError(
                f"Invalid search mode: {mode!r}. Must be one of {valid_modes}",
                mode=mode,
                valid_modes=list(valid_modes),
            )

        # Route to appropriate search method
        if mode == "semantic":
            return await self.semantic_search(
                query=query.query,
                agent_id=query.agent_id,
                user_id=query.user_id,
                limit=query.limit,
            )
        elif mode == "keyword":
            return await self._keyword_search(query)
        
        # Default: hybrid search
        return await self._hybrid_search(query)

    async def _hybrid_search(self, query: SearchQuery) -> list[SearchResult]:
        """Internal hybrid search implementation."""
        try:
            # Get query embedding
            query_embedding = await self._embedding.embed(query.query)

            # Load and execute hybrid search SQL
            sql = self._storage.load_sql("hybrid_search.sql")

            # Get settings for weights
            from engram.core.config import get_settings

            settings = get_settings()

            rows = await self._storage.fetchall(
                sql,
                json.dumps(query_embedding),  # $1
                query.query,  # $2
                query.agent_id,  # $3
                query.user_id,  # $4
                query.limit,  # $5
                settings.weight_semantic,  # $6
                settings.weight_keyword,  # $7
                settings.weight_decay,  # $8
                settings.weight_importance,  # $9
                settings.decay_rate,  # $10
            )

            results: list[SearchResult] = []
            for row in rows:
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=query.agent_id,
                    user_id=row.get("user_id"),
                    session_id=row.get("session_id"),
                    content=row["content"],
                    importance=row["importance"],
                    access_count=row.get("access_count", 0),
                    metadata=json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"],
                    created_at=row["created_at"],
                    last_accessed_at=row["last_accessed_at"],
                )

                result = SearchResult(
                    memory=memory,
                    score=float(row["score"]),
                    semantic_score=float(row.get("semantic_score", 0)),
                    keyword_score=float(row.get("keyword_score", 0)),
                    decay_score=float(row.get("decay_score", 0)),
                )

                if result.score >= query.min_score:
                    results.append(result)

            return results

        except Exception as e:
            raise QueryError(f"Search failed: {e}") from e

    async def _keyword_search(self, query: SearchQuery) -> list[SearchResult]:
        """Keyword-only search using full-text search."""
        try:
            from engram.core.config import get_settings
            settings = get_settings()

            # Use keyword-focused search with text ranking
            sql = """
            WITH keyword_matches AS (
                SELECT 
                    m.memory_id,
                    m.agent_id,
                    m.user_id,
                    m.session_id,
                    m.content,
                    m.importance,
                    m.access_count,
                    m.metadata,
                    m.created_at,
                    m.last_accessed_at,
                    ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', $1)) AS keyword_rank,
                    POWER($5, EXTRACT(EPOCH FROM (NOW() - m.created_at)) / 86400.0) AS decay_score
                FROM agent_memory m
                WHERE m.agent_id = $2
                    AND ($3::TEXT IS NULL OR m.user_id = $3)
                    AND to_tsvector('english', m.content) @@ plainto_tsquery('english', $1)
            )
            SELECT *,
                (keyword_rank * 0.7 + decay_score * 0.2 + importance * 0.1) AS score
            FROM keyword_matches
            ORDER BY score DESC
            LIMIT $4
            """

            rows = await self._storage.fetchall(
                sql,
                query.query,  # $1
                query.agent_id,  # $2
                query.user_id,  # $3
                query.limit,  # $4
                settings.decay_rate,  # $5
            )

            results: list[SearchResult] = []
            for row in rows:
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=query.agent_id,
                    user_id=row.get("user_id"),
                    session_id=row.get("session_id"),
                    content=row["content"],
                    importance=row["importance"],
                    access_count=row.get("access_count", 0),
                    metadata=json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"],
                    created_at=row["created_at"],
                    last_accessed_at=row["last_accessed_at"],
                )

                result = SearchResult(
                    memory=memory,
                    score=float(row["score"]),
                    semantic_score=0.0,
                    keyword_score=float(row.get("keyword_rank", 0)),
                    decay_score=float(row.get("decay_score", 0)),
                )

                if result.score >= query.min_score:
                    results.append(result)

            return results

        except Exception as e:
            raise QueryError(f"Keyword search failed: {e}") from e

    async def semantic_search(
        self,
        query: str,
        agent_id: AgentId,
        user_id: UserId | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Pure semantic search using vector similarity.

        Args:
            query: Search query text.
            agent_id: Filter by agent.
            user_id: Optional filter by user.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        try:
            query_embedding = await self._embedding.embed(query)

            from engram.core.config import get_settings

            settings = get_settings()

            sql = self._storage.load_sql("semantic_search.sql")
            rows = await self._storage.fetchall(
                sql,
                json.dumps(query_embedding),
                agent_id,
                user_id,
                limit,
                settings.decay_rate,
            )

            results: list[SearchResult] = []
            for row in rows:
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=agent_id,
                    content=row["content"],
                    importance=row["importance"],
                    metadata=json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"],
                    created_at=row["created_at"],
                    last_accessed_at=row["last_accessed_at"],
                )

                results.append(
                    SearchResult(
                        memory=memory,
                        score=float(row["score"]),
                        semantic_score=float(row["semantic_score"]),
                        decay_score=float(row["decay_score"]),
                    )
                )

            return results

        except Exception as e:
            raise QueryError(f"Semantic search failed: {e}") from e

    def _row_to_memory(self, row: Any) -> Memory:
        """Convert a database row to a Memory object."""
        embedding = row["embedding"]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Memory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=row["content"],
            embedding=embedding,
            importance=row["importance"],
            access_count=row["access_count"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            metadata=metadata or {},
        )
