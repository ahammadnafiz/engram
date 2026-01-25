"""Graph traversal for Engram.

This module provides graph traversal operations using recursive CTEs
for multi-hop reasoning across memory relations.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from engram.core._types import MemoryId, RelationType
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
    from engram.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)


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

        if relation_types:
            query += " AND relation_type = ANY($2)"
            rows = await self._storage.fetchall(query, memory_id, relation_types)
        else:
            rows = await self._storage.fetchall(query, memory_id)

        return [
            MemoryRelation(
                source_memory_id=row["source_memory_id"],
                target_memory_id=row["target_memory_id"],
                relation_type=row["relation_type"],
                weight=row["weight"],
                metadata=json.loads(row["metadata"])
                if isinstance(row["metadata"], str)
                else row["metadata"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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

            rows = await self._storage.fetchall(
                sql,
                query.start_memory_id,  # $1
                query.max_depth,  # $2
                query.relation_types,  # $3
                query.direction,  # $4
                query.min_weight,  # $5
                query.limit,  # $6
            )

            results: list[TraversalResult] = []
            for row in rows:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                results.append(
                    TraversalResult(
                        memory_id=row["memory_id"],
                        content=row["content"],
                        fact=row.get("fact"),
                        main_content=row.get("main_content"),
                        importance=row["importance"],
                        metadata=metadata or {},
                        created_at=row["created_at"],
                        last_accessed_at=row["last_accessed_at"],
                        access_count=row.get("access_count", 0),
                        depth=row["depth"],
                        path=row["path"],
                        relation_type=row["relation_type"],
                        path_weight=row["path_weight"],
                        score=row["score"],
                    )
                )

            return results

        except MemoryNotFoundError:
            raise
        except Exception as e:
            raise GraphError(f"Graph traversal failed: {e}") from e

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
