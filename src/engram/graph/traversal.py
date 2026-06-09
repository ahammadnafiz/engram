"""Graph traversal for Engram.

This module provides graph traversal operations using recursive CTEs
for multi-hop reasoning across memory relations.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from engram.core.exceptions import (
    GraphError,
    MemoryNotFoundError,
)
from engram.graph.models import (
    MemoryRelation,
    RelationCreate,
    TraversalQuery,
    TraversalResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from engram.core._types import MemoryId, Metadata, RelationType
    from engram.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = {"outbound", "inbound", "any"}


class GraphTraversal:
    """Graph traversal operations for memory relations.

    This class provides operations for creating relations between memories
    and traversing the memory graph using recursive CTEs for efficient
    multi-hop queries.

    Example:
        graph = GraphTraversal(storage)

        # Create a relation
        await graph.relate(
            source_id="mem_abc",
            target_id="mem_def",
            relation_type="causes",
            weight=0.8,
        )

        # Traverse from a memory
        results = await graph.traverse(TraversalQuery(
            start_memory_id="mem_abc",
            max_depth=3,
            direction="outbound",
        ))
    """

    def __init__(self, storage: PostgresStorage) -> None:
        """Initialize graph traversal.

        Args:
            storage: PostgreSQL storage backend.
        """
        self._storage = storage

    def _validate_direction(self, direction: str) -> str:
        normalized = direction.strip().lower()
        if normalized not in VALID_DIRECTIONS:
            raise GraphError(
                "Invalid graph traversal direction",
                direction=direction,
                valid_directions=sorted(VALID_DIRECTIONS),
            )
        return normalized

    def _normalize_relation_types(
        self,
        relation_types: Iterable[RelationType] | None,
    ) -> list[RelationType] | None:
        if relation_types is None:
            return None
        normalized = list(dict.fromkeys(relation_types))
        return normalized or None

    def _json(self, value: object, default: Metadata) -> Metadata:
        if value is None:
            return default
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                logger.debug("Ignoring malformed relation metadata JSON")
                return default
            return parsed if isinstance(parsed, dict) else default
        return value if isinstance(value, dict) else default

    def _row_value(self, row: object, key: str, default: object = None) -> object:
        try:
            return row[key]  # type: ignore[index]
        except (KeyError, TypeError):
            return default

    def _row_to_relation(self, row: object) -> MemoryRelation:
        return MemoryRelation(
            source_memory_id=self._row_value(row, "source_memory_id"),
            target_memory_id=self._row_value(row, "target_memory_id"),
            relation_type=self._row_value(row, "relation_type") or "related_to",
            weight=self._row_value(row, "weight") or 1.0,
            metadata=self._json(self._row_value(row, "metadata"), {}),
            created_at=self._row_value(row, "created_at"),
        )

    def _row_to_traversal_result(self, row: object) -> TraversalResult:
        return TraversalResult(
            memory_id=self._row_value(row, "memory_id"),
            content=self._row_value(row, "content") or "",
            fact=self._row_value(row, "fact"),
            main_content=self._row_value(row, "main_content"),
            importance=self._row_value(row, "importance") or 0.0,
            metadata=self._json(self._row_value(row, "metadata"), {}),
            created_at=self._row_value(row, "created_at"),
            last_accessed_at=self._row_value(row, "last_accessed_at"),
            access_count=self._row_value(row, "access_count", 0) or 0,
            depth=self._row_value(row, "depth") or 0,
            path=self._row_value(row, "path", []) or [],
            relation_type=self._row_value(row, "relation_type"),
            path_weight=self._row_value(row, "path_weight") or 0.0,
            score=self._row_value(row, "score") or 0.0,
        )

    async def relate(
        self,
        source_id: MemoryId,
        target_id: MemoryId,
        relation_type: RelationType = "related_to",
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> MemoryRelation:
        """Create a relation between two memories.

        Args:
            source_id: Source memory ID.
            target_id: Target memory ID.
            relation_type: Type of relation.
            weight: Relation weight (0.0 to 1.0).
            metadata: Optional relation metadata.

        Returns:
            The created relation.

        Raises:
            MemoryNotFoundError: If either memory doesn't exist.
            GraphError: If relation creation fails.
        """
        if source_id == target_id:
            raise GraphError("Cannot create a graph relation from a memory to itself")

        # Verify both memories exist
        for mem_id in [source_id, target_id]:
            exists = await self._storage.fetchval(
                "SELECT EXISTS(SELECT 1 FROM agent_memory WHERE memory_id = $1)",
                mem_id,
            )
            if not exists:
                raise MemoryNotFoundError(mem_id)

        try:
            await self._storage.execute(
                """
                INSERT INTO memory_relations (
                    source_memory_id, target_memory_id,
                    relation_type, weight, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (source_memory_id, target_memory_id, relation_type)
                DO UPDATE SET weight = $4, metadata = $5
                """,
                source_id,
                target_id,
                relation_type,
                weight,
                json.dumps(metadata or {}),
            )

            relation = MemoryRelation(
                source_memory_id=source_id,
                target_memory_id=target_id,
                relation_type=relation_type,
                weight=weight,
                metadata=metadata or {},
            )

            logger.debug(
                f"Created relation: {source_id} --[{relation_type}]--> {target_id}"
            )
            return relation

        except Exception as e:
            raise GraphError(f"Failed to create relation: {e}") from e

    async def relate_batch(
        self,
        relations: list[RelationCreate],
        verify_memories: bool = True,
    ) -> list[MemoryRelation]:
        """Create multiple relations in a batch.

        Args:
            relations: List of relations to create.
            verify_memories: If True, verify all memory IDs exist before creating.
                Set to False for performance if you're certain memories exist.

        Returns:
            List of created relations.

        Raises:
            GraphError: If any memory ID doesn't exist (when verify_memories=True).
        """
        if not relations:
            return []

        try:
            for relation in relations:
                if relation.source_memory_id == relation.target_memory_id:
                    raise GraphError(
                        "Cannot create a graph relation from a memory to itself",
                        memory_id=relation.source_memory_id,
                    )

            # Verify all memory IDs exist if requested
            if verify_memories:
                # Collect all unique memory IDs
                memory_ids = set()
                for r in relations:
                    memory_ids.add(r.source_memory_id)
                    memory_ids.add(r.target_memory_id)

                # Check existence in batch
                existing = await self._storage.fetchval(
                    """
                    SELECT array_agg(memory_id)
                    FROM agent_memory
                    WHERE memory_id = ANY($1::text[])
                    """,
                    list(memory_ids),
                )

                existing_set = set(existing) if existing else set()
                missing = memory_ids - existing_set

                if missing:
                    raise GraphError(
                        f"Cannot create relations: {len(missing)} memory IDs not found",
                        missing_ids=list(missing),
                    )

            await self._storage.executemany(
                """
                INSERT INTO memory_relations (
                    source_memory_id, target_memory_id,
                    relation_type, weight, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (source_memory_id, target_memory_id, relation_type)
                DO UPDATE SET weight = $4, metadata = $5
                """,
                [
                    (
                        r.source_memory_id,
                        r.target_memory_id,
                        r.relation_type,
                        r.weight,
                        json.dumps(r.metadata),
                    )
                    for r in relations
                ],
            )

            return [
                MemoryRelation(
                    source_memory_id=r.source_memory_id,
                    target_memory_id=r.target_memory_id,
                    relation_type=r.relation_type,
                    weight=r.weight,
                    metadata=r.metadata,
                )
                for r in relations
            ]

        except GraphError:
            raise
        except Exception as e:
            raise GraphError(f"Failed to create relations batch: {e}") from e

    async def get_relations(
        self,
        memory_id: MemoryId,
        direction: str = "outbound",
        relation_types: list[RelationType] | None = None,
    ) -> list[MemoryRelation]:
        """Get relations for a memory.

        Args:
            memory_id: The memory ID to get relations for.
            direction: Direction (outbound, inbound, any).
            relation_types: Optional filter by relation types.

        Returns:
            List of relations.
        """
        direction = self._validate_direction(direction)
        relation_types = self._normalize_relation_types(relation_types)

        if direction == "outbound":
            condition = "source_memory_id = $1"
        elif direction == "inbound":
            condition = "target_memory_id = $1"
        else:
            condition = "(source_memory_id = $1 OR target_memory_id = $1)"

        query = f"""
            SELECT
                source_memory_id, target_memory_id,
                relation_type, weight, metadata, created_at
            FROM memory_relations
            WHERE {condition}
        """

        try:
            if relation_types:
                query += " AND relation_type = ANY($2::text[])"
                rows = await self._storage.fetchall(query, memory_id, relation_types)
            else:
                rows = await self._storage.fetchall(query, memory_id)

            return [self._row_to_relation(row) for row in rows]
        except GraphError:
            raise
        except Exception as e:
            raise GraphError(f"Failed to get graph relations: {e}") from e

    async def traverse(self, query: TraversalQuery) -> list[TraversalResult]:
        """Traverse the memory graph from a starting point.

        Uses recursive CTEs for efficient multi-hop traversal.

        Args:
            query: Traversal parameters.

        Returns:
            List of traversal results ordered by depth and score.

        Raises:
            MemoryNotFoundError: If start memory doesn't exist.
            GraphError: If traversal fails.
        """
        # Verify start memory exists
        exists = await self._storage.fetchval(
            "SELECT EXISTS(SELECT 1 FROM agent_memory WHERE memory_id = $1)",
            query.start_memory_id,
        )
        if not exists:
            raise MemoryNotFoundError(query.start_memory_id)

        try:
            sql = self._storage.load_sql("graph_traverse.sql")
            direction = self._validate_direction(query.direction)
            relation_types = self._normalize_relation_types(query.relation_types)

            rows = await self._storage.fetchall(
                sql,
                query.start_memory_id,  # $1
                query.max_depth,  # $2
                relation_types,  # $3
                direction,  # $4
                query.min_weight,  # $5
                query.limit,  # $6
            )

            return [self._row_to_traversal_result(row) for row in rows]

        except MemoryNotFoundError:
            raise
        except GraphError:
            raise
        except Exception as e:
            raise GraphError(f"Graph traversal failed: {e}") from e

    async def traverse_many(
        self,
        start_memory_ids: Iterable[MemoryId],
        *,
        max_depth: int = 2,
        direction: str = "any",
        relation_types: list[RelationType] | None = None,
        min_weight: float = 0.0,
        limit_per_seed: int = 25,
        total_limit: int = 100,
        skip_missing: bool = True,
    ) -> list[TraversalResult]:
        """Traverse from multiple seed memories and return deduplicated results.

        This is the preferred graph expansion primitive for prompt assembly:
        retrieval usually returns several relevant memories, and expanding all
        of them produces a more robust context graph than choosing one seed.
        """
        seen_seeds = list(dict.fromkeys(start_memory_ids))
        if not seen_seeds:
            return []

        by_memory: dict[MemoryId, TraversalResult] = {}
        for memory_id in seen_seeds:
            try:
                results = await self.traverse(
                    TraversalQuery(
                        start_memory_id=memory_id,
                        max_depth=max_depth,
                        direction=direction,
                        relation_types=relation_types,
                        min_weight=min_weight,
                        limit=limit_per_seed,
                    )
                )
            except MemoryNotFoundError:
                if skip_missing:
                    continue
                raise

            for result in results:
                current = by_memory.get(result.memory_id)
                if current is None or self._rank_key(result) > self._rank_key(current):
                    by_memory[result.memory_id] = result

        return sorted(
            by_memory.values(),
            key=lambda result: (
                result.depth,
                -result.score,
                -result.path_weight,
                result.memory_id,
            ),
        )[:total_limit]

    def render_context(
        self,
        results: list[TraversalResult],
        *,
        max_tokens: int | None = None,
        token_counter: Callable[[str], int] | None = None,
        include_paths: bool = False,
        header: str = "## Related memory graph",
    ) -> str:
        """Render traversal results into a deterministic prompt context block."""
        if not results:
            return ""

        count = token_counter or (lambda text: max(1, len(text) // 4))
        lines = [header] if header else []
        used = count("\n".join(lines)) if lines else 0

        for result in results:
            relation = result.relation_type or "related_to"
            line = (
                f"- depth={result.depth} relation={relation} "
                f"score={result.score:.3f}: {result.content}"
            )
            if include_paths:
                line += f" path={' -> '.join(result.path)}"
            line_cost = count(line)
            if max_tokens is not None and used + line_cost > max_tokens:
                break
            lines.append(line)
            used += line_cost

        return "\n".join(lines).strip()

    def _rank_key(self, result: TraversalResult) -> tuple[float, float, int]:
        return (result.score, result.path_weight, -result.depth)

    async def find_path(
        self,
        source_id: MemoryId,
        target_id: MemoryId,
        max_depth: int = 5,
    ) -> list[MemoryId] | None:
        """Find the shortest path between two memories.

        Args:
            source_id: Starting memory ID.
            target_id: Target memory ID.
            max_depth: Maximum path length.

        Returns:
            List of memory IDs forming the path, or None if no path exists.
        """
        try:
            # Use traversal to find path
            results = await self.traverse(
                TraversalQuery(
                    start_memory_id=source_id,
                    max_depth=max_depth,
                    direction="any",
                    limit=200,  # Max allowed by TraversalQuery validation
                )
            )

            # Find the target in results
            for result in results:
                if result.memory_id == target_id:
                    return result.path

            return None

        except MemoryNotFoundError:
            # Source memory doesn't exist - return None (no path possible)
            return None
        except GraphError:
            # Re-raise graph errors for proper handling
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise GraphError(f"Path finding failed: {e}") from e

    async def get_neighbors(
        self,
        memory_id: MemoryId,
        depth: int = 1,
        direction: str = "any",
    ) -> list[TraversalResult]:
        """Get neighboring memories within a certain depth.

        A convenience method for shallow graph exploration.

        Args:
            memory_id: The center memory.
            depth: How many hops to explore.
            direction: Direction of relations to follow.

        Returns:
            List of neighboring memories with traversal info.
        """
        return await self.traverse(
            TraversalQuery(
                start_memory_id=memory_id,
                max_depth=depth,
                direction=direction,
            )
        )
