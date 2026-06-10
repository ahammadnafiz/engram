"""Memory store for Engram.

This module provides the MemoryStore class for CRUD operations on memories.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import asyncpg

from engram.core.config import EngramSettings, get_settings
from engram.core.exceptions import (
    DuplicateMemoryError,
    MemoryNotFoundError,
    QueryError,
    StorageError,
)
from engram.core.serialization import json_dumps
from engram.memory.models import (
    Memory,
    MemoryCreate,
    MemoryUpdate,
    SearchQuery,
    SearchResult,
)

if TYPE_CHECKING:
    from engram.core._types import AgentId, MemoryId, UserId
    from engram.embedding.service import EmbeddingService
    from engram.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)

# Insert one memory; on an exact-fact conflict return the existing row's id.
# (xmax = 0) distinguishes a real insert from a conflict-update.
_INSERT_MEMORY_SQL = """
INSERT INTO agent_memory (
    memory_id, agent_id, user_id, session_id,
    content, fact, main_content, memory_type, embedding, importance, metadata,
    created_at, last_accessed_at, access_count
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (agent_id, COALESCE(user_id, ''), fact) DO UPDATE
    SET memory_id = agent_memory.memory_id
RETURNING memory_id,
    (xmax = 0) AS was_inserted
"""

# Mark older active memories with the same conflict key superseded.
_SUPERSEDE_CONFLICTS_SQL = """
UPDATE agent_memory
SET metadata =
    metadata
    || jsonb_build_object(
        'status', 'superseded',
        'superseded_by', $4::text,
        'superseded_at', NOW()::text
    )
WHERE agent_id = $1
    AND ($2::text IS NULL OR user_id = $2)
    AND memory_id <> $4
    AND metadata->>'conflict_key' = $3
    AND COALESCE(metadata->>'status', 'active') <> 'superseded'
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (0.0 when either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


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
        settings: EngramSettings | None = None,
    ) -> None:
        """Initialize the memory store.

        Args:
            storage: PostgreSQL storage backend.
            embedding_service: Service for computing embeddings.
            settings: Engram settings (search weights, decay rate). Falls back
                to the global cached settings when not provided.
        """
        self._storage = storage
        self._embedding = embedding_service
        self._settings = settings or get_settings()

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

    async def _find_near_duplicate(
        self,
        agent_id: AgentId,
        user_id: UserId | None,
        embedding: list[float],
        threshold: float,
    ) -> str | None:
        """Return the nearest existing memory's id if its cosine similarity is
        at or above ``threshold``, else None.

        This catches near-duplicates that the exact-text unique index misses
        (e.g. same fact with different punctuation or wording).

        Scope matches the unique index exactly (agent + COALESCE(user_id, '')),
        and superseded memories never block a re-insert: a corrected fact that
        gets re-asserted must become active again, not resolve to the hidden
        superseded row.
        """
        row = await self._storage.fetchone(
            """
            SELECT memory_id, 1 - (embedding <=> $1::vector) AS score
            FROM agent_memory
            WHERE agent_id = $2
                AND COALESCE(user_id, '') = COALESCE($3::text, '')
                AND COALESCE(metadata->>'status', 'active') <> 'superseded'
                AND embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT 1
            """,
            json.dumps(embedding),
            agent_id,
            user_id,
        )
        if row is None:
            return None
        score = row.get("score")
        if score is not None and score >= threshold:
            return str(row["memory_id"])
        return None

    async def add(self, create: MemoryCreate) -> Memory:
        """Add a new memory.

        Two-column memory system:
        - content/fact: The user fact (embedded for hybrid search)
        - main_content: Optional conversation context (not embedded)

        Args:
            create: Memory creation input.

        Returns:
            The created memory with generated ID and embedding.

        Raises:
            StorageError: If database operation fails.
            EmbeddingError: If embedding computation fails.
        """
        # Fact is the content - this is what gets embedded
        fact_text = create.content

        # Generate embedding for FACT only (cost-effective)
        embedding = await self._embedding.embed(fact_text)

        # Near-duplicate guard: if a vector-near-identical memory already exists
        # (e.g. same fact with different punctuation/wording), return it instead
        # of inserting a twin. Setting near_duplicate_threshold=1.0 disables this.
        threshold = self._settings.near_duplicate_threshold
        if threshold < 1.0:
            dup_id = await self._find_near_duplicate(
                create.agent_id, create.user_id, embedding, threshold
            )
            if dup_id is not None:
                logger.debug(
                    f"Near-duplicate (>= {threshold}) of {dup_id}; skipping insert"
                )
                # The resolved memory still wins its conflict slot
                conflict_key = create.metadata.get("conflict_key")
                if conflict_key:
                    await self.supersede_conflicts(
                        agent_id=create.agent_id,
                        user_id=create.user_id,
                        conflict_key=str(conflict_key),
                        winner_memory_id=dup_id,
                    )
                return await self.get_without_access_update(dup_id)

        memory = Memory(
            agent_id=create.agent_id,
            user_id=create.user_id,
            session_id=create.session_id,
            content=fact_text,  # Backward compatibility
            fact=fact_text,  # New: embedded for search
            main_content=create.main_content,  # New: context (not embedded)
            memory_type=create.memory_type,
            embedding=embedding,
            importance=0.5,  # Default importance, use reinforce() to boost
            metadata=create.metadata,
        )

        try:
            # Auto-create agent and user if they don't exist
            await self._ensure_agent_exists(create.agent_id)
            if create.user_id:
                await self._ensure_user_exists(create.user_id)

            conflict_key = memory.metadata.get("conflict_key")

            # Insert and conflict-supersede atomically: a crash between the
            # two must not leave two active memories in the same slot.
            async with self._storage.transaction() as conn:
                row = await conn.fetchrow(
                    _INSERT_MEMORY_SQL,
                    memory.memory_id,
                    memory.agent_id,
                    memory.user_id,
                    memory.session_id,
                    fact_text,  # content (backward compat)
                    fact_text,  # fact (new)
                    create.main_content,  # main_content (new)
                    memory.memory_type,  # memory_type (new)
                    json.dumps(memory.embedding),  # Store as JSON for vector type
                    memory.importance,
                    json_dumps(memory.metadata),
                    memory.created_at,
                    memory.last_accessed_at,
                    memory.access_count,
                )
                if row and conflict_key:
                    await conn.execute(
                        _SUPERSEDE_CONFLICTS_SQL,
                        memory.agent_id,
                        memory.user_id,
                        str(conflict_key),
                        row["memory_id"],
                    )

            if row and not row["was_inserted"]:
                # Memory with same fact already exists
                logger.debug(
                    f"Memory with same fact already exists, returning existing: {row['memory_id']}"
                )
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
        Only embeds facts (content), not main_content.

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
        agent_ids = {c.agent_id for c in creates}
        user_ids = {c.user_id for c in creates if c.user_id}

        for agent_id in agent_ids:
            await self._ensure_agent_exists(agent_id)
        for user_id in user_ids:
            await self._ensure_user_exists(user_id)

        # Batch embed all facts (content is the fact)
        facts = [c.content for c in creates]
        embeddings = await self._embedding.embed_batch(facts)

        memories: list[Memory] = []
        for create, embedding in zip(creates, embeddings, strict=True):
            fact_text = create.content
            memory = Memory(
                agent_id=create.agent_id,
                user_id=create.user_id,
                session_id=create.session_id,
                content=fact_text,  # Backward compat
                fact=fact_text,  # New: embedded for search
                main_content=create.main_content,  # New: context (not embedded)
                memory_type=create.memory_type,
                embedding=embedding,
                importance=0.5,  # Default importance, use reinforce() to boost
                metadata=create.metadata,
            )
            memories.append(memory)

        # Near-duplicate guard against existing memories (1.0 disables). Exact
        # text duplicates are still handled by the ON CONFLICT clause below;
        # this catches vector-near-identical facts the unique index misses.
        # Near-dups of existing memories resolve to the existing memory (same
        # semantics as add()); vector-near-identical facts *within* the batch
        # collapse into the first occurrence.
        threshold = self._settings.near_duplicate_threshold
        plan: list[Memory | str] = []  # Memory -> insert, str -> existing id
        if threshold < 1.0:
            inserting: list[Memory] = []
            for m in memories:
                assert m.embedding is not None
                dup_id = await self._find_near_duplicate(
                    m.agent_id, m.user_id, m.embedding, threshold
                )
                if dup_id is not None:
                    logger.debug(f"Batch near-duplicate of {dup_id}; resolving")
                    plan.append(dup_id)
                    continue
                in_batch_dup = any(
                    k.embedding is not None
                    and k.agent_id == m.agent_id
                    and (k.user_id or "") == (m.user_id or "")
                    and _cosine_similarity(m.embedding, k.embedding) >= threshold
                    for k in inserting
                )
                if in_batch_dup:
                    logger.debug("In-batch near-duplicate; skipping")
                    continue
                inserting.append(m)
                plan.append(m)
        else:
            plan = list(memories)

        if not plan:
            return []

        # Insert per-row with RETURNING so exact-fact conflicts resolve to the
        # real existing memory instead of a phantom id, all in one transaction.
        try:
            results: list[Memory] = []
            async with self._storage.transaction() as conn:
                for item in plan:
                    if isinstance(item, str):
                        existing_row = await conn.fetchrow(
                            """
                            SELECT
                                memory_id, agent_id, user_id, session_id,
                                content, fact, main_content, memory_type, embedding,
                                importance, access_count,
                                created_at, last_accessed_at, metadata
                            FROM agent_memory
                            WHERE memory_id = $1
                            """,
                            item,
                        )
                        if existing_row is not None:
                            results.append(self._row_to_memory(existing_row))
                        continue
                    m = item
                    row = await conn.fetchrow(
                        _INSERT_MEMORY_SQL,
                        m.memory_id,
                        m.agent_id,
                        m.user_id,
                        m.session_id,
                        m.content,
                        m.fact,
                        m.main_content,
                        m.memory_type,
                        json.dumps(m.embedding),
                        m.importance,
                        json_dumps(m.metadata),
                        m.created_at,
                        m.last_accessed_at,
                        m.access_count,
                    )
                    resolved_id = row["memory_id"]
                    conflict_key = m.metadata.get("conflict_key")
                    if conflict_key:
                        await conn.execute(
                            _SUPERSEDE_CONFLICTS_SQL,
                            m.agent_id,
                            m.user_id,
                            str(conflict_key),
                            resolved_id,
                        )
                    if row["was_inserted"]:
                        results.append(m)
                    else:
                        existing_row = await conn.fetchrow(
                            """
                            SELECT
                                memory_id, agent_id, user_id, session_id,
                                content, fact, main_content, memory_type, embedding,
                                importance, access_count,
                                created_at, last_accessed_at, metadata
                            FROM agent_memory
                            WHERE memory_id = $1
                            """,
                            resolved_id,
                        )
                        results.append(self._row_to_memory(existing_row))
            logger.debug(f"Added {len(results)} memories in batch")
            return results

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
                content, fact, main_content, memory_type, embedding, importance, access_count,
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
                content, fact, main_content, memory_type, embedding, importance, access_count,
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

        If content changes, fact is also updated and re-embedded.
        main_content is preserved unless explicitly set.

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

        # Build update - content/fact are the same
        new_content = update.content if update.content else current.content
        new_importance = (
            update.importance if update.importance is not None else current.importance
        )
        new_metadata = {**current.metadata, **(update.metadata or {})}
        new_memory_type = str(new_metadata.get("memory_type") or current.memory_type)

        # Re-embed if fact (content) changed
        new_embedding = current.embedding
        if update.content and update.content != current.content:
            new_embedding = await self._embedding.embed(update.content)
            previous_versions = list(current.metadata.get("previous_versions", []))
            previous_versions.insert(
                0,
                {
                    "content": current.content,
                    "memory_type": current.memory_type,
                    "replaced_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            new_metadata["previous_versions"] = previous_versions[:10]
            new_metadata["version"] = int(current.metadata.get("version", 1)) + 1
            new_metadata["status"] = "active"
            new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            # Update both content and fact to stay in sync
            row = await self._storage.fetchone(
                """
                UPDATE agent_memory
                SET
                    content = $2,
                    fact = $2,
                    embedding = $3,
                    importance = $4,
                    metadata = $5,
                    memory_type = $6,
                    last_accessed_at = NOW()
                WHERE memory_id = $1
                RETURNING
                    memory_id, agent_id, user_id, session_id,
                    content, fact, main_content, memory_type, embedding, importance, access_count,
                    created_at, last_accessed_at, metadata
                """,
                memory_id,
                new_content,
                json.dumps(new_embedding) if new_embedding is not None else None,
                new_importance,
                json_dumps(new_metadata),
                new_memory_type,
            )

            if row is None:
                raise MemoryNotFoundError(memory_id)

            return self._row_to_memory(row)

        except MemoryNotFoundError:
            raise
        except asyncpg.UniqueViolationError as e:
            # The new content equals another memory's fact for the same
            # agent/user. Surface a typed error so callers (e.g. the
            # add_conversation merge path) can treat it as "already stored".
            raise DuplicateMemoryError(
                "Updated content collides with an existing memory's fact",
                memory_id=memory_id,
            ) from e
        except Exception as e:
            raise StorageError(f"Failed to update memory: {e}") from e

    async def supersede_conflicts(
        self,
        *,
        agent_id: AgentId,
        user_id: UserId | None,
        conflict_key: str,
        winner_memory_id: MemoryId,
    ) -> int:
        """Mark older active memories with the same conflict key superseded."""
        try:
            result = await self._storage.execute(
                _SUPERSEDE_CONFLICTS_SQL,
                agent_id,
                user_id,
                conflict_key,
                winner_memory_id,
            )
            if isinstance(result, str) and result.startswith("UPDATE "):
                return int(result.split()[-1])
            return 0
        except Exception as e:
            raise StorageError(f"Failed to supersede conflicts: {e}") from e

    async def list_policy_memories(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
        *,
        limit: int = 100,
        critical_only: bool = True,
        include_superseded: bool = False,
        memory_types: list[str] | None = None,
    ) -> list[Memory]:
        """List policy-governed memories without vector ranking.

        This powers deterministic recall for critical facts and observability
        for superseded/conflicted memories.
        """
        try:
            rows = await self._storage.fetchall(
                """
                SELECT
                    memory_id, agent_id, user_id, session_id,
                    content, fact, main_content, memory_type, embedding, importance, access_count,
                    created_at, last_accessed_at, metadata
                FROM agent_memory
                WHERE agent_id = $1
                    AND ($2::text IS NULL OR user_id = $2)
                    AND ($5::text[] IS NULL OR memory_type = ANY($5))
                    AND (
                        $4::boolean = false
                        OR COALESCE((metadata->>'critical')::boolean, false) = true
                    )
                    AND (
                        $6::boolean = true
                        OR COALESCE(metadata->>'status', 'active') <> 'superseded'
                    )
                ORDER BY
                    COALESCE((metadata->>'critical')::boolean, false) DESC,
                    CASE memory_type
                        WHEN 'profile' THEN 1
                        WHEN 'constraint' THEN 2
                        WHEN 'preference' THEN 3
                        WHEN 'project' THEN 4
                        WHEN 'task' THEN 5
                        WHEN 'decision' THEN 6
                        WHEN 'tool_result' THEN 7
                        ELSE 8
                    END,
                    created_at DESC
                LIMIT $3
                """,
                agent_id,
                user_id,
                limit,
                critical_only,
                list(memory_types) if memory_types else None,
                include_superseded,
            )
            return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            raise QueryError(f"Policy memory listing failed: {e}") from e

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
                content, fact, main_content, memory_type, embedding, importance, access_count,
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
                    content, fact, main_content, memory_type, embedding, importance, access_count,
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
                    content, fact, main_content, memory_type, embedding, importance, access_count,
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
                metadata_filter=query.metadata_filter,
                memory_types=query.memory_types,
                min_score=query.min_score,
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

            # Use injected settings for weights
            settings = self._settings

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
                json_dumps(query.metadata_filter)
                if query.metadata_filter
                else None,  # $11
                list(query.memory_types) if query.memory_types else None,  # $12
                query.min_score,  # $13
                settings.text_search_config,  # $14
            )

            results: list[SearchResult] = []
            for row in rows:
                # Handle both fact column (new) and content column (backward compat)
                fact_text = row.get("fact") or row["content"]
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=query.agent_id,
                    user_id=row.get("user_id"),
                    session_id=row.get("session_id"),
                    content=fact_text,  # Backward compat
                    fact=fact_text,
                    main_content=row.get("main_content"),
                    memory_type=row.get("memory_type", "semantic"),
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
        """Keyword-only search using full-text search on fact column."""
        try:
            settings = self._settings

            # Use keyword-focused search with text ranking on fact_tsv.
            # Decay matches hybrid/semantic: calculate_decay on
            # last_accessed_at per hour (not per-day on created_at).
            # min_score filters BEFORE the LIMIT so qualifying matches
            # beyond the first page are not lost.
            sql = """
            WITH keyword_matches AS (
                SELECT
                    m.memory_id,
                    m.agent_id,
                    m.user_id,
                    m.session_id,
                    m.memory_type,
                    m.fact,
                    m.main_content,
                    m.importance,
                    m.access_count,
                    m.metadata,
                    m.created_at,
                    m.last_accessed_at,
                    ts_rank(fact_tsv, plainto_tsquery($9::regconfig, $1)) AS keyword_rank,
                    calculate_decay(m.last_accessed_at, $5) AS decay_score
                FROM agent_memory m
                WHERE m.agent_id = $2
                    AND ($3::TEXT IS NULL OR m.user_id = $3)
                    AND ($6::jsonb IS NULL OR m.metadata @> $6::jsonb)
                    AND ($7::text[] IS NULL OR m.memory_type = ANY($7))
                    AND COALESCE(m.metadata->>'status', 'active') <> 'superseded'
                    AND fact_tsv @@ plainto_tsquery($9::regconfig, $1)
            )
            SELECT * FROM (
                SELECT *,
                    (keyword_rank * 0.7 + decay_score * 0.2 + importance * 0.1) AS score
                FROM keyword_matches
            ) scored
            WHERE score >= $8
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
                json_dumps(query.metadata_filter)
                if query.metadata_filter
                else None,  # $6
                list(query.memory_types) if query.memory_types else None,  # $7
                query.min_score,  # $8
                settings.text_search_config,  # $9
            )

            results: list[SearchResult] = []
            for row in rows:
                fact_text = row.get("fact") or row.get("content", "")
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=query.agent_id,
                    user_id=row.get("user_id"),
                    session_id=row.get("session_id"),
                    content=fact_text,  # Backward compat
                    fact=fact_text,
                    main_content=row.get("main_content"),
                    memory_type=row.get("memory_type", "semantic"),
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
        metadata_filter: dict[str, Any] | None = None,
        memory_types: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Pure semantic search using vector similarity.

        Args:
            query: Search query text.
            agent_id: Filter by agent.
            user_id: Optional filter by user.
            limit: Maximum results.
            metadata_filter: Optional metadata containment filter.
            memory_types: Optional list of memory types to restrict to.
            min_score: Minimum final score (filtered before the limit).

        Returns:
            List of search results.
        """
        try:
            query_embedding = await self._embedding.embed(query)

            settings = self._settings

            sql = self._storage.load_sql("semantic_search.sql")
            rows = await self._storage.fetchall(
                sql,
                json.dumps(query_embedding),
                agent_id,
                user_id,
                limit,
                settings.decay_rate,
                json_dumps(metadata_filter) if metadata_filter else None,
                list(memory_types) if memory_types else None,
                min_score,
            )

            results: list[SearchResult] = []
            for row in rows:
                # Handle both fact column (new) and content column (backward compat)
                fact_text = row.get("fact") or row["content"]
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=agent_id,
                    user_id=row.get("user_id"),
                    session_id=row.get("session_id"),
                    content=fact_text,  # Backward compat
                    fact=fact_text,
                    main_content=row.get("main_content"),
                    memory_type=row.get("memory_type", "semantic"),
                    importance=row["importance"],
                    access_count=row.get("access_count", 0),
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
        """Convert a database row to a Memory object.

        Handles both old schema (content only) and new schema (fact + main_content).
        """
        embedding = row["embedding"]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Handle both old and new schema
        # Prefer fact column if available, fallback to content
        fact_text = row.get("fact") or row["content"]
        main_content = row.get("main_content")

        return Memory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=fact_text,  # Backward compat - content = fact
            fact=fact_text,
            main_content=main_content,
            memory_type=row.get("memory_type", "semantic"),
            embedding=embedding,
            importance=row["importance"],
            access_count=row["access_count"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            metadata=metadata or {},
        )
