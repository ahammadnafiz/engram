"""Unit tests for graph traversal and relations."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


class RecordLike:
    """Minimal asyncpg.Record-like object without dict.get()."""

    def __init__(self, values: dict) -> None:
        self._values = values

    def __getitem__(self, key: str):
        if key not in self._values:
            raise KeyError(key)
        return self._values[key]


class TestGraphTraversalRelate:
    """Tests for GraphTraversal.relate() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.fetchone = AsyncMock(
            return_value={
                "memory_id": "mem_123",
                "content": "test",
            }
        )
        storage.fetchval = AsyncMock(return_value=True)  # Memory exists
        storage.execute = AsyncMock(return_value="INSERT 1")
        return storage

    @pytest.mark.asyncio
    async def test_relate_creates_relation(self, mock_storage: MagicMock) -> None:
        """Test creating a relation between memories."""
        from engram.graph.traversal import GraphTraversal

        traversal = GraphTraversal(storage=mock_storage)

        relation = await traversal.relate(
            source_id="mem_1",
            target_id="mem_2",
            relation_type="related_to",
            weight=0.8,
        )

        assert relation.source_memory_id == "mem_1"
        assert relation.target_memory_id == "mem_2"
        assert relation.relation_type == "related_to"
        assert relation.weight == 0.8

    @pytest.mark.asyncio
    async def test_relate_validates_memory_exists(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that relate validates source memory exists."""
        from engram.core.exceptions import MemoryNotFoundError
        from engram.graph.traversal import GraphTraversal

        # Source memory doesn't exist - fetchval returns False
        mock_storage.fetchval = AsyncMock(return_value=False)

        traversal = GraphTraversal(storage=mock_storage)

        with pytest.raises(MemoryNotFoundError):
            await traversal.relate("nonexistent", "mem_2")

    @pytest.mark.asyncio
    async def test_relate_rejects_self_relation(self, mock_storage: MagicMock) -> None:
        """Test that self-relations are rejected before touching storage."""
        from engram.core.exceptions import GraphError
        from engram.graph.traversal import GraphTraversal

        traversal = GraphTraversal(storage=mock_storage)

        with pytest.raises(GraphError):
            await traversal.relate("mem_1", "mem_1")
        mock_storage.fetchval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_relate_with_metadata(self, mock_storage: MagicMock) -> None:
        """Test creating relation with metadata."""
        from engram.graph.traversal import GraphTraversal

        traversal = GraphTraversal(storage=mock_storage)

        relation = await traversal.relate(
            source_id="mem_1",
            target_id="mem_2",
            metadata={"reason": "user mentioned both"},
        )

        assert relation.metadata == {"reason": "user mentioned both"}


class TestGraphTraversalRelateBatch:
    """Tests for GraphTraversal.relate_batch() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.executemany = AsyncMock()
        storage.fetchval = AsyncMock(return_value=["mem_1", "mem_2", "mem_3"])
        return storage

    @pytest.mark.asyncio
    async def test_relate_batch_creates_relations(
        self, mock_storage: MagicMock
    ) -> None:
        """Test creating multiple relations in batch."""
        from engram.graph.models import RelationCreate
        from engram.graph.traversal import GraphTraversal

        traversal = GraphTraversal(storage=mock_storage)

        relations = [
            RelationCreate(source_memory_id="mem_1", target_memory_id="mem_2"),
            RelationCreate(source_memory_id="mem_2", target_memory_id="mem_3"),
        ]

        results = await traversal.relate_batch(relations)

        assert len(results) == 2
        mock_storage.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_relate_batch_empty_list(self, mock_storage: MagicMock) -> None:
        """Test relate_batch with empty list."""
        from engram.graph.traversal import GraphTraversal

        traversal = GraphTraversal(storage=mock_storage)

        results = await traversal.relate_batch([])

        assert results == []
        mock_storage.executemany.assert_not_called()

    @pytest.mark.asyncio
    async def test_relate_batch_verifies_memories(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that relate_batch verifies memories exist when enabled."""
        from engram.core.exceptions import GraphError
        from engram.graph.models import RelationCreate
        from engram.graph.traversal import GraphTraversal

        # Return only some of the requested IDs (simulating missing ones)
        mock_storage.fetchval = AsyncMock(return_value=["mem_1", "mem_2"])

        traversal = GraphTraversal(storage=mock_storage)

        relations = [
            RelationCreate(source_memory_id="mem_1", target_memory_id="mem_2"),
            RelationCreate(
                source_memory_id="mem_3", target_memory_id="mem_4"
            ),  # Missing
        ]

        with pytest.raises(GraphError) as exc_info:
            await traversal.relate_batch(relations, verify_memories=True)

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_relate_batch_skips_verification(
        self, mock_storage: MagicMock
    ) -> None:
        """Test that verification can be skipped for performance."""
        from engram.graph.models import RelationCreate
        from engram.graph.traversal import GraphTraversal

        traversal = GraphTraversal(storage=mock_storage)

        relations = [
            RelationCreate(source_memory_id="mem_1", target_memory_id="mem_2"),
        ]

        results = await traversal.relate_batch(relations, verify_memories=False)

        assert len(results) == 1
        # Should NOT call fetchval for verification
        mock_storage.fetchval.assert_not_called()

    @pytest.mark.asyncio
    async def test_relate_batch_rejects_self_relation(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Test that batch relation creation rejects self-relations."""
        from engram.core.exceptions import GraphError
        from engram.graph.models import RelationCreate
        from engram.graph.traversal import GraphTraversal

        traversal = GraphTraversal(storage=mock_storage)

        with pytest.raises(GraphError):
            await traversal.relate_batch(
                [RelationCreate(source_memory_id="mem_1", target_memory_id="mem_1")]
            )
        mock_storage.executemany.assert_not_awaited()


class TestGraphTraversalTraverse:
    """Tests for GraphTraversal.traverse() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.load_sql = MagicMock(return_value="WITH RECURSIVE ...")
        storage.fetchval = AsyncMock(return_value=True)  # Memory exists
        return storage

    @pytest.mark.asyncio
    async def test_traverse_returns_results(self, mock_storage: MagicMock) -> None:
        """Test graph traversal returns results."""
        from engram.graph.models import TraversalQuery
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(
            return_value=[
                {
                    "memory_id": "mem_2",
                    "content": "Connected memory",
                    "importance": 0.7,
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc),
                    "last_accessed_at": datetime.now(timezone.utc),
                    "depth": 1,
                    "path": ["mem_1", "mem_2"],
                    "relation_type": "related_to",
                    "path_weight": 0.8,
                    "score": 0.85,
                }
            ]
        )

        traversal = GraphTraversal(storage=mock_storage)

        query = TraversalQuery(start_memory_id="mem_1", max_depth=2)
        results = await traversal.traverse(query)

        assert len(results) == 1
        assert results[0].memory_id == "mem_2"
        assert results[0].depth == 1

    @pytest.mark.asyncio
    async def test_traverse_start_memory_not_found(
        self, mock_storage: MagicMock
    ) -> None:
        """Test traversal from non-existent memory raises error."""
        from engram.core.exceptions import MemoryNotFoundError
        from engram.graph.models import TraversalQuery
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchval = AsyncMock(return_value=False)  # Memory doesn't exist

        traversal = GraphTraversal(storage=mock_storage)

        query = TraversalQuery(start_memory_id="nonexistent")

        with pytest.raises(MemoryNotFoundError):
            await traversal.traverse(query)

    @pytest.mark.asyncio
    async def test_traverse_empty_results(self, mock_storage: MagicMock) -> None:
        """Test traversal with no connections."""
        from engram.graph.models import TraversalQuery
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(return_value=[])

        traversal = GraphTraversal(storage=mock_storage)

        query = TraversalQuery(start_memory_id="isolated")
        results = await traversal.traverse(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_traverse_handles_record_like_rows(
        self, mock_storage: MagicMock
    ) -> None:
        """Test traversal row conversion without relying on dict.get()."""
        from engram.graph.models import TraversalQuery
        from engram.graph.traversal import GraphTraversal

        now = datetime.now(timezone.utc)
        mock_storage.fetchall = AsyncMock(
            return_value=[
                RecordLike(
                    {
                        "memory_id": "mem_2",
                        "content": "Connected memory",
                        "importance": 0.7,
                        "metadata": {"source": "test"},
                        "created_at": now,
                        "last_accessed_at": now,
                        "depth": 1,
                        "path": ["mem_1", "mem_2"],
                        "relation_type": "supports",
                        "path_weight": 0.8,
                        "score": 0.85,
                    }
                )
            ]
        )
        traversal = GraphTraversal(storage=mock_storage)

        results = await traversal.traverse(TraversalQuery(start_memory_id="mem_1"))

        assert results[0].memory_id == "mem_2"
        assert results[0].fact is None
        assert results[0].access_count == 0
        assert results[0].metadata == {"source": "test"}

    @pytest.mark.asyncio
    async def test_traverse_rejects_invalid_direction(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Test invalid directions fail before executing traversal SQL."""
        from engram.core.exceptions import GraphError
        from engram.graph.models import TraversalQuery
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(return_value=[])
        traversal = GraphTraversal(storage=mock_storage)

        with pytest.raises(GraphError) as exc_info:
            await traversal.traverse(
                TraversalQuery(start_memory_id="mem_1", direction="sideways")
            )

        assert exc_info.value.context["direction"] == "sideways"
        mock_storage.fetchall.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_traverse_many_dedupes_by_best_score(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Test multi-seed traversal dedupes and keeps the strongest path."""
        from engram.graph.traversal import GraphTraversal

        now = datetime.now(timezone.utc)
        mock_storage.fetchall = AsyncMock(
            side_effect=[
                [
                    {
                        "memory_id": "mem_shared",
                        "content": "Shared",
                        "importance": 0.5,
                        "metadata": "{}",
                        "created_at": now,
                        "last_accessed_at": now,
                        "depth": 2,
                        "path": ["seed_1", "mid", "mem_shared"],
                        "relation_type": "related_to",
                        "path_weight": 0.4,
                        "score": 0.5,
                    }
                ],
                [
                    {
                        "memory_id": "mem_shared",
                        "content": "Shared better",
                        "importance": 0.5,
                        "metadata": "{}",
                        "created_at": now,
                        "last_accessed_at": now,
                        "depth": 1,
                        "path": ["seed_2", "mem_shared"],
                        "relation_type": "supports",
                        "path_weight": 0.9,
                        "score": 0.95,
                    }
                ],
            ]
        )
        traversal = GraphTraversal(storage=mock_storage)

        results = await traversal.traverse_many(["seed_1", "seed_2", "seed_1"])

        assert len(results) == 1
        assert results[0].content == "Shared better"
        assert mock_storage.fetchall.await_count == 2

    def test_render_context_respects_budget(self) -> None:
        """Test prompt rendering returns deterministic budgeted graph context."""
        from engram.graph.models import TraversalResult
        from engram.graph.traversal import GraphTraversal

        now = datetime.now(timezone.utc)
        traversal = GraphTraversal(storage=MagicMock())
        rendered = traversal.render_context(
            [
                TraversalResult(
                    memory_id="mem_1",
                    content="Short relation",
                    importance=0.5,
                    metadata={},
                    created_at=now,
                    last_accessed_at=now,
                    depth=1,
                    path=["seed", "mem_1"],
                    relation_type="supports",
                    path_weight=0.9,
                    score=0.8,
                ),
                TraversalResult(
                    memory_id="mem_2",
                    content="This second line should be too expensive",
                    importance=0.5,
                    metadata={},
                    created_at=now,
                    last_accessed_at=now,
                    depth=1,
                    path=["seed", "mem_2"],
                    relation_type="supports",
                    path_weight=0.9,
                    score=0.7,
                ),
            ],
            max_tokens=15,
            token_counter=lambda text: len(text.split()),
        )

        assert "## Related memory graph" in rendered
        assert "Short relation" in rendered
        assert "too expensive" not in rendered


class TestGraphTraversalFindPath:
    """Tests for GraphTraversal.find_path() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.load_sql = MagicMock(return_value="WITH RECURSIVE ...")
        storage.fetchval = AsyncMock(return_value=True)  # Memory exists
        return storage

    @pytest.mark.asyncio
    async def test_find_path_returns_path(self, mock_storage: MagicMock) -> None:
        """Test finding path between connected memories."""
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(
            return_value=[
                {
                    "memory_id": "mem_3",
                    "content": "Target",
                    "importance": 0.5,
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc),
                    "last_accessed_at": datetime.now(timezone.utc),
                    "depth": 2,
                    "path": ["mem_1", "mem_2", "mem_3"],
                    "relation_type": "related_to",
                    "path_weight": 0.7,
                    "score": 0.8,
                }
            ]
        )

        traversal = GraphTraversal(storage=mock_storage)

        path = await traversal.find_path("mem_1", "mem_3")

        assert path == ["mem_1", "mem_2", "mem_3"]

    @pytest.mark.asyncio
    async def test_find_path_no_connection(self, mock_storage: MagicMock) -> None:
        """Test finding path when no connection exists."""
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(return_value=[])

        traversal = GraphTraversal(storage=mock_storage)

        path = await traversal.find_path("mem_1", "disconnected")

        assert path is None

    @pytest.mark.asyncio
    async def test_find_path_source_not_found(self, mock_storage: MagicMock) -> None:
        """Test find_path when source doesn't exist returns None."""
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchval = AsyncMock(return_value=False)  # Memory doesn't exist

        traversal = GraphTraversal(storage=mock_storage)

        # Should return None (not raise) for missing source
        path = await traversal.find_path("nonexistent", "mem_2")

        assert path is None


class TestGraphTraversalGetNeighbors:
    """Tests for GraphTraversal.get_neighbors() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.load_sql = MagicMock(return_value="WITH RECURSIVE ...")
        storage.fetchval = AsyncMock(return_value=True)  # Memory exists
        return storage

    @pytest.mark.asyncio
    async def test_get_neighbors_default_depth(self, mock_storage: MagicMock) -> None:
        """Test getting immediate neighbors (depth=1)."""
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(
            return_value=[
                {
                    "memory_id": "neighbor_1",
                    "content": "Neighbor",
                    "importance": 0.6,
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc),
                    "last_accessed_at": datetime.now(timezone.utc),
                    "depth": 1,
                    "path": ["mem_1", "neighbor_1"],
                    "relation_type": "related_to",
                    "path_weight": 0.9,
                    "score": 0.85,
                }
            ]
        )

        traversal = GraphTraversal(storage=mock_storage)

        neighbors = await traversal.get_neighbors("mem_1")

        assert len(neighbors) == 1
        assert neighbors[0].memory_id == "neighbor_1"
        assert neighbors[0].depth == 1


class TestGraphTraversalGetRelations:
    """Tests for GraphTraversal.get_relations() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        return storage

    @pytest.mark.asyncio
    async def test_get_relations_outbound(self, mock_storage: MagicMock) -> None:
        """Test getting outbound relations."""
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(
            return_value=[
                {
                    "source_memory_id": "mem_1",
                    "target_memory_id": "mem_2",
                    "relation_type": "causes",
                    "weight": 0.9,
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc),
                }
            ]
        )

        traversal = GraphTraversal(storage=mock_storage)

        relations = await traversal.get_relations("mem_1", direction="outbound")

        assert len(relations) == 1
        assert relations[0].target_memory_id == "mem_2"

    @pytest.mark.asyncio
    async def test_get_relations_inbound(self, mock_storage: MagicMock) -> None:
        """Test getting inbound relations."""
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(
            return_value=[
                {
                    "source_memory_id": "mem_0",
                    "target_memory_id": "mem_1",
                    "relation_type": "causes",  # Valid relation type
                    "weight": 0.7,
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc),
                }
            ]
        )

        traversal = GraphTraversal(storage=mock_storage)

        relations = await traversal.get_relations("mem_1", direction="inbound")

        assert len(relations) == 1
        assert relations[0].source_memory_id == "mem_0"

    @pytest.mark.asyncio
    async def test_get_relations_filter_by_type(self, mock_storage: MagicMock) -> None:
        """Test filtering relations by type."""
        from engram.graph.traversal import GraphTraversal

        mock_storage.fetchall = AsyncMock(return_value=[])

        traversal = GraphTraversal(storage=mock_storage)

        await traversal.get_relations(
            "mem_1",
            relation_types=["causes", "supports"],  # Valid relation types
        )

        assert "relation_type = ANY" in mock_storage.fetchall.call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_relations_rejects_invalid_direction(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Test relation reads validate direction."""
        from engram.core.exceptions import GraphError
        from engram.graph.traversal import GraphTraversal

        traversal = GraphTraversal(storage=mock_storage)

        with pytest.raises(GraphError):
            await traversal.get_relations("mem_1", direction="sideways")


class TestMemoryRelationModel:
    """Tests for MemoryRelation model."""

    def test_relation_defaults(self) -> None:
        """Test relation default values."""
        from engram.graph.models import MemoryRelation

        relation = MemoryRelation(
            source_memory_id="mem_1",
            target_memory_id="mem_2",
        )

        assert relation.relation_type == "related_to"
        assert relation.weight == 1.0
        assert relation.metadata == {}

    def test_relation_weight_bounds(self) -> None:
        """Test relation weight validation."""
        from pydantic import ValidationError

        from engram.graph.models import MemoryRelation

        # Valid weights
        MemoryRelation(source_memory_id="a", target_memory_id="b", weight=0.0)
        MemoryRelation(source_memory_id="a", target_memory_id="b", weight=1.0)

        # Invalid weights
        with pytest.raises(ValidationError):
            MemoryRelation(source_memory_id="a", target_memory_id="b", weight=-0.1)

        with pytest.raises(ValidationError):
            MemoryRelation(source_memory_id="a", target_memory_id="b", weight=1.1)

    def test_relation_immutable(self) -> None:
        """Test that relations are immutable."""
        from pydantic import ValidationError

        from engram.graph.models import MemoryRelation

        relation = MemoryRelation(
            source_memory_id="mem_1",
            target_memory_id="mem_2",
        )

        with pytest.raises(ValidationError):
            relation.weight = 0.5  # type: ignore


class TestTraversalQueryModel:
    """Tests for TraversalQuery model."""

    def test_traversal_query_defaults(self) -> None:
        """Test traversal query default values."""
        from engram.graph.models import TraversalQuery

        query = TraversalQuery(start_memory_id="mem_1")

        assert query.max_depth == 3
        assert query.direction == "outbound"
        assert query.min_weight == 0.0
        assert query.limit == 50

    def test_traversal_query_depth_bounds(self) -> None:
        """Test max_depth validation."""
        from pydantic import ValidationError

        from engram.graph.models import TraversalQuery

        # Valid depths
        TraversalQuery(start_memory_id="m", max_depth=1)
        TraversalQuery(start_memory_id="m", max_depth=10)

        # Invalid depths
        with pytest.raises(ValidationError):
            TraversalQuery(start_memory_id="m", max_depth=0)

        with pytest.raises(ValidationError):
            TraversalQuery(start_memory_id="m", max_depth=11)

    def test_traversal_query_limit_bounds(self) -> None:
        """Test limit validation."""
        from pydantic import ValidationError

        from engram.graph.models import TraversalQuery

        # Valid limits
        TraversalQuery(start_memory_id="m", limit=1)
        TraversalQuery(start_memory_id="m", limit=200)

        # Invalid limits
        with pytest.raises(ValidationError):
            TraversalQuery(start_memory_id="m", limit=0)

        with pytest.raises(ValidationError):
            TraversalQuery(start_memory_id="m", limit=201)
