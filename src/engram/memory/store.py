"""Memory store for Engram.

This module provides the MemoryStore class for CRUD operations on memories.
"""

from __future__ import annotations

import hashlib
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
    MemoryExplanation,
    MemoryHistoryEvent,
    MemoryLineage,
    MemoryUpdate,
    SearchQuery,
    SearchResult,
)

if TYPE_CHECKING:
    from engram.core._types import AgentId, MemoryId, MemoryType, UserId
    from engram.embedding.service import EmbeddingService
    from engram.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)

_MEMORY_SELECT_COLUMNS = """
    memory_id, agent_id, user_id, session_id,
    content, fact, main_content, memory_type, embedding,
    importance, access_count,
    created_at, last_accessed_at,
    lineage_id, revision, status, valid_from, valid_to,
    superseded_by_memory_id, superseded_at, metadata
"""

_MEMORY_SELECT_COLUMNS_M = """
    m.memory_id, m.agent_id, m.user_id, m.session_id,
    m.content, m.fact, m.main_content, m.memory_type, m.embedding,
    m.importance, m.access_count,
    m.created_at, m.last_accessed_at,
    m.lineage_id, m.revision, m.status, m.valid_from, m.valid_to,
    m.superseded_by_memory_id, m.superseded_at, m.metadata
"""

# Insert one active memory; on an exact active-fact conflict return the existing
# row's id. Superseded historical rows are outside the partial unique index, so
# reasserting a corrected old fact creates a new active revision instead of
# reviving the hidden row.
_INSERT_MEMORY_SQL = """
INSERT INTO agent_memory (
    memory_id, agent_id, user_id, session_id,
    content, fact, main_content, memory_type, embedding, importance, metadata,
    lineage_id, revision, status, valid_from, valid_to,
    superseded_by_memory_id, superseded_at,
    created_at, last_accessed_at, access_count
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
    $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
)
ON CONFLICT (agent_id, COALESCE(user_id, ''), md5(fact))
    WHERE status = 'active'
DO UPDATE
    SET metadata = agent_memory.metadata || jsonb_build_object('status', 'active'),
        last_accessed_at = NOW(),
        access_count = agent_memory.access_count + 1
RETURNING memory_id,
    (xmax = 0) AS was_inserted
"""

# Mark older active memories in a lineage/conflict slot superseded.
_SUPERSEDE_CONFLICTS_SQL = """
UPDATE agent_memory
SET status = 'superseded',
    valid_to = NOW(),
    superseded_by_memory_id = $4,
    superseded_at = NOW(),
    metadata =
    metadata
    || jsonb_build_object(
        'status', 'superseded',
        'superseded_by', $4::text,
        'superseded_at', NOW()::text,
        'lineage_id', $5::text,
        'revision', revision
    )
WHERE agent_id = $1
    AND COALESCE(user_id, '') = COALESCE($2::text, '')
    AND memory_id <> $4
    AND (
        lineage_id = $5
        OR ($3::text IS NOT NULL AND metadata->>'conflict_key' = $3)
    )
    AND status <> 'superseded'
RETURNING memory_id
"""

_SUPERSEDE_LINEAGE_SQL = """
UPDATE agent_memory
SET status = 'superseded',
    valid_to = NOW(),
    superseded_by_memory_id = $2,
    superseded_at = NOW(),
    metadata =
    metadata
    || jsonb_build_object(
        'status', 'superseded',
        'superseded_by', $2::text,
        'superseded_at', NOW()::text,
        'lineage_id', $1::text,
        'revision', revision
    )
WHERE lineage_id = $1
    AND memory_id <> $2
    AND status <> 'superseded'
RETURNING memory_id
"""

_INSERT_SUPERSEDES_RELATION_SQL = """
INSERT INTO memory_relations (
    source_memory_id, target_memory_id, relation_type, weight, metadata
)
VALUES (
    $1, $2, 'supersedes', 1.0, $3::jsonb
)
ON CONFLICT (source_memory_id, target_memory_id, relation_type) DO NOTHING
"""

# Nearest active memory by cosine distance, scoped to match the unique index.
# Superseded rows are excluded so a re-asserted corrected fact becomes active
# again instead of resolving to the hidden superseded row.
_NEAR_DUPLICATE_SQL = """
SELECT memory_id, 1 - (embedding <=> $1::vector) AS score
FROM agent_memory
WHERE agent_id = $2
    AND COALESCE(user_id, '') = COALESCE($3::text, '')
    AND status <> 'superseded'
    AND COALESCE(metadata->>'status', 'active') <> 'superseded'
    AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT 1
"""

# Transaction-scoped lock keyed on (agent_id, user_id). Held until commit, it
# serializes the near-duplicate check + insert against other writers in the
# same scope so two concurrent adds of vector-near-identical but textually
# different facts cannot both pass the guard and both insert a twin.
_SCOPE_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext($1))"

_SELECT_MEMORY_BY_ID_SQL = (
    """
SELECT
"""
    + _MEMORY_SELECT_COLUMNS
    + """
FROM agent_memory
WHERE memory_id = $1
"""
)


def _scope_lock_key(agent_id: str, user_id: str | None) -> str:
    """Advisory-lock key for an (agent, user) write scope."""
    return f"{agent_id}\x1f{user_id or ''}"


def _metadata_conflict_key(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("conflict_key")
    if value is None:
        return None
    key = str(value).strip()
    return key or None


def _lineage_id_for(
    agent_id: str,
    user_id: str | None,
    conflict_key: str | None,
    fallback_memory_id: str,
) -> str:
    if not conflict_key:
        return fallback_memory_id
    digest = hashlib.md5(
        f"{agent_id}\x1f{user_id or ''}\x1f{conflict_key}".encode()
    ).hexdigest()
    return f"lin_{digest}"


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    get = getattr(row, "get", None)
    if get is not None:
        return get(key, default)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


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

    async def _ensure_lineage_row(
        self,
        conn: Any,
        *,
        lineage_id: str,
        memory: Memory,
        conflict_key: str | None,
        current_memory_id: str | None = None,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO memory_lineages (
                lineage_id, agent_id, user_id, conflict_key,
                current_memory_id, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (lineage_id) DO UPDATE
            SET current_memory_id = COALESCE(
                    EXCLUDED.current_memory_id,
                    memory_lineages.current_memory_id
                ),
                updated_at = NOW()
            """,
            lineage_id,
            memory.agent_id,
            memory.user_id,
            conflict_key,
            current_memory_id,
            json_dumps({"source": "engram"}),
        )

    async def _next_lineage_revision(
        self,
        conn: Any,
        *,
        memory: Memory,
        lineage_id: str,
        conflict_key: str | None,
        lock_existing: bool,
    ) -> int:
        if lock_existing:
            await self._ensure_lineage_row(
                conn,
                lineage_id=lineage_id,
                memory=memory,
                conflict_key=conflict_key,
            )
            lineage_row = await conn.fetchrow(
                """
                SELECT current_memory_id
                FROM memory_lineages
                WHERE lineage_id = $1
                FOR UPDATE
                """,
                lineage_id,
            )
            current_id = _row_get(lineage_row, "current_memory_id")
            if current_id:
                current = await conn.fetchrow(
                    "SELECT revision FROM agent_memory WHERE memory_id = $1",
                    current_id,
                )
                revision = _row_get(current, "revision")
                if revision is not None:
                    return int(revision) + 1

            newest = await conn.fetchrow(
                """
                SELECT revision
                FROM agent_memory
                WHERE lineage_id = $1
                ORDER BY revision DESC, created_at DESC
                LIMIT 1
                """,
                lineage_id,
            )
            revision = _row_get(newest, "revision")
            if revision is not None:
                return int(revision) + 1
        return 1

    async def _insert_supersedes_edges(
        self,
        conn: Any,
        *,
        winner_memory_id: str,
        superseded_ids: list[str],
        lineage_id: str,
    ) -> None:
        for old_id in superseded_ids:
            await conn.execute(
                _INSERT_SUPERSEDES_RELATION_SQL,
                winner_memory_id,
                old_id,
                json_dumps({"lineage_id": lineage_id}),
            )

    async def _supersede_lineage_members(
        self,
        conn: Any,
        *,
        memory: Memory,
        conflict_key: str | None,
        winner_memory_id: str,
        lineage_id: str,
    ) -> list[str]:
        if conflict_key:
            rows = await conn.fetch(
                _SUPERSEDE_CONFLICTS_SQL,
                memory.agent_id,
                memory.user_id,
                conflict_key,
                winner_memory_id,
                lineage_id,
            )
        else:
            rows = await conn.fetch(
                _SUPERSEDE_LINEAGE_SQL,
                lineage_id,
                winner_memory_id,
            )
        return [str(row["memory_id"]) for row in rows]

    async def _insert_memory_with_lineage(
        self,
        conn: Any,
        memory: Memory,
        *,
        force_lineage_id: str | None = None,
    ) -> tuple[str, bool]:
        conflict_key = _metadata_conflict_key(memory.metadata)
        lineage_id = force_lineage_id or str(
            memory.metadata.get("lineage_id")
            or _lineage_id_for(
                memory.agent_id, memory.user_id, conflict_key, memory.memory_id
            )
        )
        lock_existing = conflict_key is not None or force_lineage_id is not None
        revision = await self._next_lineage_revision(
            conn,
            memory=memory,
            lineage_id=lineage_id,
            conflict_key=conflict_key,
            lock_existing=lock_existing,
        )

        memory.lineage_id = lineage_id
        memory.revision = revision
        memory.status = "active"
        memory.valid_from = memory.created_at
        memory.valid_to = None
        memory.superseded_by_memory_id = None
        memory.superseded_at = None
        memory.metadata = {
            **memory.metadata,
            "status": "active",
            "lineage_id": lineage_id,
            "revision": revision,
        }

        row = await conn.fetchrow(
            _INSERT_MEMORY_SQL,
            memory.memory_id,
            memory.agent_id,
            memory.user_id,
            memory.session_id,
            memory.content,
            memory.fact,
            memory.main_content,
            memory.memory_type,
            json.dumps(memory.embedding),
            memory.importance,
            json_dumps(memory.metadata),
            memory.lineage_id,
            memory.revision,
            memory.status,
            memory.valid_from,
            memory.valid_to,
            memory.superseded_by_memory_id,
            memory.superseded_at,
            memory.created_at,
            memory.last_accessed_at,
            memory.access_count,
        )
        if row is None:
            raise StorageError("Memory insert returned no row")

        resolved_id = str(row["memory_id"])
        was_inserted = bool(row["was_inserted"])
        if not was_inserted:
            return resolved_id, False

        superseded_ids = (
            await self._supersede_lineage_members(
                conn,
                memory=memory,
                conflict_key=conflict_key,
                winner_memory_id=resolved_id,
                lineage_id=lineage_id,
            )
            if lock_existing
            else []
        )
        await self._insert_supersedes_edges(
            conn,
            winner_memory_id=resolved_id,
            superseded_ids=superseded_ids,
            lineage_id=lineage_id,
        )
        await self._ensure_lineage_row(
            conn,
            lineage_id=lineage_id,
            memory=memory,
            conflict_key=conflict_key,
            current_memory_id=resolved_id,
        )
        return resolved_id, True

    async def _find_near_duplicate(
        self,
        conn: Any,
        agent_id: AgentId,
        user_id: UserId | None,
        embedding: list[float],
        threshold: float,
    ) -> str | None:
        """Return the nearest existing memory's id if its cosine similarity is
        at or above ``threshold``, else None.

        This catches near-duplicates that the exact-text unique index misses
        (e.g. same fact with different punctuation or wording).

        Runs on the caller's transaction connection (``conn``) so the lookup
        and the subsequent insert are part of the same transaction; paired with
        the per-scope advisory lock the caller holds, this closes the
        check-then-insert race where two concurrent writers both see no
        duplicate and both insert a twin.

        Scope matches the unique index exactly (agent + COALESCE(user_id, '')),
        and superseded memories never block a re-insert: a corrected fact that
        gets re-asserted must become active again, not resolve to the hidden
        superseded row.
        """
        row = await conn.fetchrow(
            _NEAR_DUPLICATE_SQL,
            json.dumps(embedding),
            agent_id,
            user_id,
        )
        if row is None:
            return None
        score = _row_get(row, "score")
        if score is not None and score >= threshold:
            return str(row["memory_id"])
        return None

    async def _find_superseded_exact_lineage(
        self,
        conn: Any,
        memory: Memory,
    ) -> str | None:
        """Return a historical exact fact's lineage for reassertion."""
        row = await conn.fetchrow(
            """
            SELECT lineage_id
            FROM agent_memory
            WHERE agent_id = $1
                AND COALESCE(user_id, '') = COALESCE($2::text, '')
                AND fact = $3
                AND lineage_id IS NOT NULL
                AND (
                    status = 'superseded'
                    OR COALESCE(metadata->>'status', 'active') = 'superseded'
                )
            ORDER BY revision DESC, created_at DESC
            LIMIT 1
            """,
            memory.agent_id,
            memory.user_id,
            memory.fact,
        )
        lineage_id = _row_get(row, "lineage_id")
        return str(lineage_id) if lineage_id else None

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

        threshold = self._settings.near_duplicate_threshold

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
            metadata=dict(create.metadata),
        )

        try:
            # Auto-create agent and user if they don't exist
            await self._ensure_agent_exists(create.agent_id)
            if create.user_id:
                await self._ensure_user_exists(create.user_id)

            conflict_key = _metadata_conflict_key(memory.metadata)

            # Near-dup guard, insert, and conflict-supersede all run in one
            # transaction. The per-scope advisory lock serializes the guard
            # against concurrent writers in the same (agent, user) scope, so a
            # crash can't leave two active memories in one conflict slot and two
            # concurrent adds of near-identical facts can't both insert a twin.
            async with self._storage.transaction() as conn:
                await conn.execute(
                    _SCOPE_LOCK_SQL,
                    _scope_lock_key(create.agent_id, create.user_id),
                )
                # Near-duplicate guard: if a vector-near-identical memory
                # already exists (e.g. same fact with different
                # punctuation/wording), resolve to it instead of inserting a
                # twin. near_duplicate_threshold=1.0 disables this.
                # Conflict-keyed memories represent mutable slots; do not let
                # vector similarity absorb corrections before they become
                # lineage revisions. Exact active duplicates still dedupe via
                # the database unique index.
                if threshold < 1.0 and conflict_key is None:
                    dup_id = await self._find_near_duplicate(
                        conn, create.agent_id, create.user_id, embedding, threshold
                    )
                    if dup_id is not None:
                        logger.debug(
                            f"Near-duplicate (>= {threshold}) of {dup_id}; "
                            "skipping insert"
                        )
                        # The resolved memory still wins its conflict slot.
                        existing_row = await conn.fetchrow(
                            _SELECT_MEMORY_BY_ID_SQL, dup_id
                        )
                        if existing_row is not None:
                            return self._row_to_memory(existing_row)

                force_lineage_id = None
                if conflict_key is None:
                    force_lineage_id = await self._find_superseded_exact_lineage(
                        conn, memory
                    )

                resolved_id, was_inserted = await self._insert_memory_with_lineage(
                    conn, memory, force_lineage_id=force_lineage_id
                )

            if not was_inserted:
                # Memory with same fact already exists
                logger.debug(
                    f"Memory with same fact already exists, returning existing: {resolved_id}"
                )
                # Return the existing memory
                existing = await self.get(resolved_id)
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
                metadata=dict(create.metadata),
            )
            memories.append(memory)

        # Near-duplicate guard against existing memories (1.0 disables). Exact
        # text duplicates are still handled by the ON CONFLICT clause below;
        # this catches vector-near-identical facts the unique index misses.
        # Near-dups of existing memories resolve to the existing memory (same
        # semantics as add()); vector-near-identical facts *within* the batch
        # collapse into the first occurrence.
        #
        # Planning, insertion, and conflict-supersede all run in one
        # transaction guarded by per-scope advisory locks, so the guard sees a
        # stable view: a concurrent writer in the same (agent, user) scope can't
        # slip a twin in between the near-dup check and the insert.
        threshold = self._settings.near_duplicate_threshold
        try:
            results: list[Memory] = []
            async with self._storage.transaction() as conn:
                # Lock distinct scopes in a stable order so two batches that
                # share scopes can't deadlock on each other.
                scope_keys = sorted(
                    {_scope_lock_key(m.agent_id, m.user_id) for m in memories}
                )
                for key in scope_keys:
                    await conn.execute(_SCOPE_LOCK_SQL, key)

                plan: list[Memory | str] = []  # Memory -> insert, str -> existing
                if threshold < 1.0:
                    inserting: list[Memory] = []
                    for m in memories:
                        assert m.embedding is not None
                        if _metadata_conflict_key(m.metadata) is None:
                            dup_id = await self._find_near_duplicate(
                                conn, m.agent_id, m.user_id, m.embedding, threshold
                            )
                            if dup_id is not None:
                                logger.debug(
                                    f"Batch near-duplicate of {dup_id}; resolving"
                                )
                                plan.append(dup_id)
                                continue
                            in_batch_dup = any(
                                k.embedding is not None
                                and k.agent_id == m.agent_id
                                and (k.user_id or "") == (m.user_id or "")
                                and _metadata_conflict_key(k.metadata) is None
                                and _cosine_similarity(m.embedding, k.embedding)
                                >= threshold
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

                # Insert per-row with RETURNING so exact-fact conflicts resolve
                # to the real existing memory instead of a phantom id.
                for item in plan:
                    if isinstance(item, str):
                        existing_row = await conn.fetchrow(
                            _SELECT_MEMORY_BY_ID_SQL, item
                        )
                        if existing_row is not None:
                            results.append(self._row_to_memory(existing_row))
                        continue
                    m = item
                    force_lineage_id = None
                    if _metadata_conflict_key(m.metadata) is None:
                        force_lineage_id = await self._find_superseded_exact_lineage(
                            conn, m
                        )
                    resolved_id, was_inserted = await self._insert_memory_with_lineage(
                        conn, m, force_lineage_id=force_lineage_id
                    )
                    if was_inserted:
                        results.append(m)
                    else:
                        existing_row = await conn.fetchrow(
                            _SELECT_MEMORY_BY_ID_SQL, resolved_id
                        )
                        results.append(self._row_to_memory(existing_row))
            logger.debug(f"Added {len(results)} memories in batch")
            return results

        except Exception as e:
            raise StorageError(f"Failed to add memories in batch: {e}") from e

    async def get(self, memory_id: MemoryId, *, track_access: bool = True) -> Memory:
        """Get a memory by ID.

        By default this is a read-write: it bumps last_accessed_at and
        access_count, which feed time-decay ranking. Pass track_access=False
        for a pure read (no write) — useful for read-replica routing or
        read-heavy paths where access tracking would cause write contention.

        Args:
            memory_id: The memory ID to retrieve.
            track_access: When True (default) update access metadata; when False
                perform a plain read.

        Returns:
            The memory object.

        Raises:
            MemoryNotFoundError: If memory doesn't exist.
        """
        if not track_access:
            return await self.get_without_access_update(memory_id)

        row = await self._storage.fetchone(
            f"""
            UPDATE agent_memory
            SET last_accessed_at = NOW(), access_count = access_count + 1
            WHERE memory_id = $1
            RETURNING
                {_MEMORY_SELECT_COLUMNS}
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
            f"""
            SELECT
                {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory
            WHERE memory_id = $1
            """,
            memory_id,
        )

        if row is None:
            raise MemoryNotFoundError(memory_id)

        return self._row_to_memory(row)

    async def get_current(self, memory_id: MemoryId) -> Memory:
        """Return the active head for a memory's lineage."""
        seed = await self.get_without_access_update(memory_id)
        lineage_id = seed.lineage_id
        if lineage_id is None:
            return seed

        row = await self._storage.fetchone(
            f"""
            SELECT
                {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory
            WHERE lineage_id = $1
                AND status = 'active'
                AND COALESCE(metadata->>'status', 'active') <> 'superseded'
            ORDER BY revision DESC, created_at DESC
            LIMIT 1
            """,
            lineage_id,
        )
        if row is None:
            return seed
        return self._row_to_memory(row)

    async def get_lineage(self, memory_id: MemoryId) -> MemoryLineage:
        """Return all revisions for a memory, newest first."""
        seed = await self.get_without_access_update(memory_id)
        lineage_id = seed.lineage_id or seed.memory_id
        rows = await self._storage.fetchall(
            f"""
            SELECT
                {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory
            WHERE lineage_id = $1 OR memory_id = $2
            ORDER BY revision DESC, created_at DESC
            """,
            lineage_id,
            seed.memory_id,
        )
        memories = [self._row_to_memory(row) for row in rows]
        current_memory_id = next(
            (memory.memory_id for memory in memories if memory.status == "active"),
            None,
        )
        return MemoryLineage(
            lineage_id=lineage_id,
            current_memory_id=current_memory_id,
            memories=memories,
        )

    async def explain_memory(self, memory_id: MemoryId) -> MemoryExplanation:
        """Return the lineage and direct supersede edges for a memory."""
        memory = await self.get_without_access_update(memory_id)
        current = await self.get_current(memory_id)
        lineage = await self.get_lineage(memory_id)
        supersedes_rows = await self._storage.fetchall(
            f"""
            SELECT
                {_MEMORY_SELECT_COLUMNS_M}
            FROM memory_relations r
            JOIN agent_memory m ON m.memory_id = r.target_memory_id
            WHERE r.source_memory_id = $1
                AND r.relation_type = 'supersedes'
            ORDER BY m.revision DESC, m.created_at DESC
            """,
            memory_id,
        )
        superseded_by = None
        if memory.superseded_by_memory_id:
            try:
                superseded_by = await self.get_without_access_update(
                    memory.superseded_by_memory_id
                )
            except MemoryNotFoundError:
                superseded_by = None
        return MemoryExplanation(
            memory=memory,
            current=current,
            lineage=lineage,
            supersedes=[self._row_to_memory(row) for row in supersedes_rows],
            superseded_by=superseded_by,
        )

    async def revise(
        self,
        memory_id: MemoryId,
        update: MemoryUpdate,
        *,
        reason: str | None = None,
    ) -> Memory:
        """Create a new active revision and supersede the previous head."""
        current = await self.get_current(memory_id)
        new_content = update.content if update.content else current.content
        new_importance = (
            update.importance if update.importance is not None else current.importance
        )
        new_metadata = {**current.metadata, **(update.metadata or {})}
        lineage_id = current.lineage_id or current.memory_id
        new_metadata["lineage_id"] = lineage_id
        new_metadata["previous_memory_id"] = current.memory_id
        new_metadata["status"] = "active"
        if reason:
            new_metadata["revision_reason"] = reason

        new_embedding = current.embedding
        if new_content != current.content or new_embedding is None:
            new_embedding = await self._embedding.embed(new_content)

        memory = Memory(
            agent_id=current.agent_id,
            user_id=current.user_id,
            session_id=current.session_id,
            content=new_content,
            fact=new_content,
            main_content=current.main_content,
            memory_type=str(new_metadata.get("memory_type") or current.memory_type),
            embedding=new_embedding,
            importance=new_importance,
            metadata=new_metadata,
        )

        try:
            async with self._storage.transaction() as conn:
                await conn.execute(
                    _SCOPE_LOCK_SQL,
                    _scope_lock_key(current.agent_id, current.user_id),
                )
                resolved_id, was_inserted = await self._insert_memory_with_lineage(
                    conn,
                    memory,
                    force_lineage_id=lineage_id,
                )

            if not was_inserted:
                return await self.get(resolved_id)
            return memory
        except asyncpg.UniqueViolationError as e:
            raise DuplicateMemoryError(
                "Revised content collides with an existing active memory",
                memory_id=memory_id,
            ) from e
        except Exception as e:
            raise StorageError(f"Failed to revise memory: {e}") from e

    async def list_memories(
        self,
        agent_id: AgentId,
        *,
        user_id: UserId | None = None,
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        memory_types: list[MemoryType] | None = None,
        limit: int = 200,
    ) -> list[Memory]:
        """List active memories by filter, without relevance ranking.

        Unlike search(), this is a plain filtered read: no query, no scores,
        no access-count update. Results are ordered by created_at ascending.

        Args:
            agent_id: Agent to scope the read to.
            user_id: Optional user filter.
            session_id: Optional session filter.
            metadata_filter: Optional JSONB containment filter.
            memory_types: Optional list of memory types to restrict to.
            limit: Maximum number of memories returned.

        Returns:
            Matching memories, oldest first.
        """
        rows = await self._storage.fetchall(
            f"""
            SELECT
                {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory
            WHERE agent_id = $1
                AND ($2::text IS NULL OR user_id = $2)
                AND ($3::jsonb IS NULL OR metadata @> $3::jsonb)
                AND ($4::text[] IS NULL OR memory_type = ANY($4))
                AND ($6::text IS NULL OR session_id = $6)
                AND status <> 'superseded'
                AND COALESCE(metadata->>'status', 'active') <> 'superseded'
            ORDER BY created_at
            LIMIT $5
            """,
            agent_id,
            user_id,
            json_dumps(metadata_filter) if metadata_filter else None,
            memory_types,
            limit,
            session_id,
        )
        return [self._row_to_memory(row) for row in rows]

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
                f"""
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
                    {_MEMORY_SELECT_COLUMNS}
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
            lineage_id = _lineage_id_for(
                str(agent_id), user_id, conflict_key, winner_memory_id
            )
            async with self._storage.transaction() as conn:
                await conn.execute(_SCOPE_LOCK_SQL, _scope_lock_key(agent_id, user_id))
                rows = await conn.fetch(
                    _SUPERSEDE_CONFLICTS_SQL,
                    agent_id,
                    user_id,
                    conflict_key,
                    winner_memory_id,
                    lineage_id,
                )
                superseded_ids = [str(row["memory_id"]) for row in rows]
                await self._insert_supersedes_edges(
                    conn,
                    winner_memory_id=winner_memory_id,
                    superseded_ids=superseded_ids,
                    lineage_id=lineage_id,
                )
                return len(superseded_ids)
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
        memory_types: list[MemoryType] | None = None,
    ) -> list[Memory]:
        """List policy-governed memories without vector ranking.

        This powers deterministic recall for critical facts and observability
        for superseded/conflicted memories.
        """
        try:
            rows = await self._storage.fetchall(
                f"""
                SELECT
                    {_MEMORY_SELECT_COLUMNS}
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
                        OR status <> 'superseded'
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
            f"""
            UPDATE agent_memory
            SET
                importance = GREATEST(0.0, LEAST(importance + $2, 1.0)),
                last_accessed_at = NOW(),
                access_count = access_count + 1
            WHERE memory_id = $1
            RETURNING
                {_MEMORY_SELECT_COLUMNS}
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
                f"""
                SELECT
                    {_MEMORY_SELECT_COLUMNS}
                FROM agent_memory
                WHERE agent_id = $1 AND user_id = $2
                    AND status <> 'superseded'
                    AND COALESCE(metadata->>'status', 'active') <> 'superseded'
                ORDER BY created_at DESC
                LIMIT $3
                """,
                agent_id,
                user_id,
                limit,
            )
        else:
            rows = await self._storage.fetchall(
                f"""
                SELECT
                    {_MEMORY_SELECT_COLUMNS}
                FROM agent_memory
                WHERE agent_id = $1
                    AND status <> 'superseded'
                    AND COALESCE(metadata->>'status', 'active') <> 'superseded'
                ORDER BY created_at DESC
                LIMIT $2
                """,
                agent_id,
                limit,
            )

        return [self._row_to_memory(row) for row in rows]

    async def get_history(
        self,
        agent_id: AgentId,
        user_id: UserId | None = None,
        *,
        limit: int = 50,
        include_superseded: bool = True,
        memory_types: list[MemoryType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[MemoryHistoryEvent]:
        """Return a user-facing memory timeline for an agent/user.

        The timeline includes:
        - ``added`` for first revisions
        - ``revised`` for later active/current revisions
        - ``superseded`` for historical facts replaced by a newer revision
        """
        if limit <= 0:
            return []

        try:
            rows = await self._storage.fetchall(
                f"""
                WITH scoped AS (
                    SELECT
                        {_MEMORY_SELECT_COLUMNS}
                    FROM agent_memory
                    WHERE agent_id = $1
                        AND ($2::text IS NULL OR user_id = $2)
                        AND ($5::text[] IS NULL OR memory_type = ANY($5))
                ),
                events AS (
                    SELECT
                        CASE
                            WHEN s.revision > 1 OR s.metadata ? 'previous_memory_id'
                                THEN 'revised'
                            ELSE 'added'
                        END AS event_type,
                        CASE
                            WHEN s.revision > 1 OR s.metadata ? 'previous_memory_id'
                                THEN 1
                            ELSE 2
                        END AS event_rank,
                        s.created_at AS occurred_at,
                        s.metadata->>'previous_memory_id' AS event_previous_memory_id,
                        s.superseded_by_memory_id AS event_superseded_by_memory_id,
                        s.metadata->>'revision_reason' AS event_reason,
                        l.current_memory_id AS event_current_memory_id,
                        s.*
                    FROM scoped s
                    LEFT JOIN memory_lineages l
                        ON l.lineage_id = s.lineage_id
                    WHERE (
                        $4::boolean = true
                        OR (
                            s.status <> 'superseded'
                            AND COALESCE(s.metadata->>'status', 'active') <> 'superseded'
                        )
                    )

                    UNION ALL

                    SELECT
                        'superseded' AS event_type,
                        0 AS event_rank,
                        s.superseded_at AS occurred_at,
                        NULL::text AS event_previous_memory_id,
                        s.superseded_by_memory_id AS event_superseded_by_memory_id,
                        s.metadata->>'revision_reason' AS event_reason,
                        l.current_memory_id AS event_current_memory_id,
                        s.*
                    FROM scoped s
                    LEFT JOIN memory_lineages l
                        ON l.lineage_id = s.lineage_id
                    WHERE $4::boolean = true
                        AND s.superseded_at IS NOT NULL
                )
                SELECT *
                FROM events
                WHERE ($6::timestamptz IS NULL OR occurred_at >= $6)
                    AND ($7::timestamptz IS NULL OR occurred_at <= $7)
                ORDER BY occurred_at DESC, event_rank ASC, revision DESC, memory_id DESC
                LIMIT $3
                """,
                agent_id,
                user_id,
                limit,
                include_superseded,
                list(memory_types) if memory_types else None,
                since,
                until,
            )
            return [self._row_to_history_event(row) for row in rows]
        except Exception as e:
            raise QueryError(f"Memory history listing failed: {e}") from e

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
                settings.search_candidate_multiplier,  # $15
            )

            results: list[SearchResult] = []
            for row in rows:
                # Handle both fact column (new) and content column (backward compat)
                fact_text = _row_get(row, "fact") or row["content"]
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=query.agent_id,
                    user_id=_row_get(row, "user_id"),
                    session_id=_row_get(row, "session_id"),
                    content=fact_text,  # Backward compat
                    fact=fact_text,
                    main_content=_row_get(row, "main_content"),
                    memory_type=_row_get(row, "memory_type", "semantic"),
                    importance=row["importance"],
                    access_count=_row_get(row, "access_count", 0),
                    metadata=json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"],
                    created_at=row["created_at"],
                    last_accessed_at=row["last_accessed_at"],
                    lineage_id=_row_get(row, "lineage_id"),
                    revision=int(_row_get(row, "revision", 1)),
                    status=str(_row_get(row, "status", "active")),
                    valid_from=_row_get(row, "valid_from"),
                    valid_to=_row_get(row, "valid_to"),
                    superseded_by_memory_id=_row_get(row, "superseded_by_memory_id"),
                    superseded_at=_row_get(row, "superseded_at"),
                )

                result = SearchResult(
                    memory=memory,
                    score=float(row["score"]),
                    semantic_score=float(_row_get(row, "semantic_score", 0)),
                    keyword_score=float(_row_get(row, "keyword_score", 0)),
                    decay_score=float(_row_get(row, "decay_score", 0)),
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
                    m.lineage_id,
                    m.revision,
                    m.status,
                    m.valid_from,
                    m.valid_to,
                    m.superseded_by_memory_id,
                    m.superseded_at,
                    ts_rank(fact_tsv, plainto_tsquery($9::regconfig, $1)) AS keyword_rank,
                    calculate_decay(m.last_accessed_at, $5) AS decay_score
                FROM agent_memory m
                WHERE m.agent_id = $2
                    AND ($3::TEXT IS NULL OR m.user_id = $3)
                    AND ($6::jsonb IS NULL OR m.metadata @> $6::jsonb)
                    AND ($7::text[] IS NULL OR m.memory_type = ANY($7))
                    AND m.status <> 'superseded'
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
                fact_text = _row_get(row, "fact") or _row_get(row, "content", "")
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=query.agent_id,
                    user_id=_row_get(row, "user_id"),
                    session_id=_row_get(row, "session_id"),
                    content=fact_text,  # Backward compat
                    fact=fact_text,
                    main_content=_row_get(row, "main_content"),
                    memory_type=_row_get(row, "memory_type", "semantic"),
                    importance=row["importance"],
                    access_count=_row_get(row, "access_count", 0),
                    metadata=json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"],
                    created_at=row["created_at"],
                    last_accessed_at=row["last_accessed_at"],
                    lineage_id=_row_get(row, "lineage_id"),
                    revision=int(_row_get(row, "revision", 1)),
                    status=str(_row_get(row, "status", "active")),
                    valid_from=_row_get(row, "valid_from"),
                    valid_to=_row_get(row, "valid_to"),
                    superseded_by_memory_id=_row_get(row, "superseded_by_memory_id"),
                    superseded_at=_row_get(row, "superseded_at"),
                )

                result = SearchResult(
                    memory=memory,
                    score=float(row["score"]),
                    semantic_score=0.0,
                    keyword_score=float(_row_get(row, "keyword_rank", 0)),
                    decay_score=float(_row_get(row, "decay_score", 0)),
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
        memory_types: list[MemoryType] | None = None,
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
                fact_text = _row_get(row, "fact") or row["content"]
                memory = Memory(
                    memory_id=row["memory_id"],
                    agent_id=agent_id,
                    user_id=_row_get(row, "user_id"),
                    session_id=_row_get(row, "session_id"),
                    content=fact_text,  # Backward compat
                    fact=fact_text,
                    main_content=_row_get(row, "main_content"),
                    memory_type=_row_get(row, "memory_type", "semantic"),
                    importance=row["importance"],
                    access_count=_row_get(row, "access_count", 0),
                    metadata=json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"],
                    created_at=row["created_at"],
                    last_accessed_at=row["last_accessed_at"],
                    lineage_id=_row_get(row, "lineage_id"),
                    revision=int(_row_get(row, "revision", 1)),
                    status=str(_row_get(row, "status", "active")),
                    valid_from=_row_get(row, "valid_from"),
                    valid_to=_row_get(row, "valid_to"),
                    superseded_by_memory_id=_row_get(row, "superseded_by_memory_id"),
                    superseded_at=_row_get(row, "superseded_at"),
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
        fact_text = _row_get(row, "fact") or row["content"]
        main_content = _row_get(row, "main_content")

        return Memory(
            memory_id=row["memory_id"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=fact_text,  # Backward compat - content = fact
            fact=fact_text,
            main_content=main_content,
            memory_type=_row_get(row, "memory_type", "semantic"),
            embedding=embedding,
            importance=row["importance"],
            access_count=row["access_count"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            metadata=metadata or {},
            lineage_id=_row_get(row, "lineage_id"),
            revision=int(_row_get(row, "revision", 1)),
            status=str(_row_get(row, "status", "active")),
            valid_from=_row_get(row, "valid_from"),
            valid_to=_row_get(row, "valid_to"),
            superseded_by_memory_id=_row_get(row, "superseded_by_memory_id"),
            superseded_at=_row_get(row, "superseded_at"),
        )

    def _row_to_history_event(self, row: Any) -> MemoryHistoryEvent:
        memory = self._row_to_memory(row)
        current_memory_id = _row_get(row, "event_current_memory_id")
        if current_memory_id is None and memory.status == "active":
            current_memory_id = memory.memory_id

        return MemoryHistoryEvent(
            event_type=str(row["event_type"]),  # type: ignore[arg-type]
            occurred_at=row["occurred_at"],
            memory=memory,
            current_memory_id=current_memory_id,
            previous_memory_id=_row_get(row, "event_previous_memory_id"),
            superseded_by_memory_id=_row_get(row, "event_superseded_by_memory_id"),
            reason=_row_get(row, "event_reason"),
            metadata={
                "lineage_id": memory.lineage_id,
                "revision": memory.revision,
                "status": memory.status,
            },
        )
