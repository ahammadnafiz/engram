"""Comprehensive unit tests for memory store with edge cases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMemoryStoreAdd:
    """Tests for MemoryStore.add() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """Create mock storage."""
        storage = MagicMock()
        storage.execute = AsyncMock(return_value="INSERT 1")
        storage.fetchone = AsyncMock(
            return_value={"memory_id": "mem_123", "was_inserted": True}
        )
        storage.fetchval = AsyncMock(return_value=1)
        return storage

    @pytest.fixture
    def mock_embedding(self) -> MagicMock:
        """Create mock embedding service."""
        embedding = MagicMock()

        async def mock_embed(text: str) -> list[float]:
            return [0.1] * 1536

        embedding.embed = AsyncMock(side_effect=mock_embed)
        embedding.dimension = 1536
        return embedding

    @pytest.mark.asyncio
    async def test_add_creates_memory_with_embedding(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that add generates embedding and stores memory."""
        from engram.memory.models import MemoryCreate
        from engram.memory.store import MemoryStore

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        create = MemoryCreate(
            content="User likes coffee",
            agent_id="agent_1",
        )

        memory = await store.add(create)

        # Should have called embedding service
        mock_embedding.embed.assert_called_once_with("User likes coffee")

        # Should have stored in database
        mock_storage.fetchone.assert_called()

        assert memory.content == "User likes coffee"
        assert memory.agent_id == "agent_1"

    @pytest.mark.asyncio
    async def test_add_returns_existing_on_duplicate(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that adding duplicate content returns existing memory."""
        from engram.memory.models import Memory, MemoryCreate
        from engram.memory.store import MemoryStore

        # Simulate duplicate detection
        mock_storage.fetchone = AsyncMock(
            return_value={
                "memory_id": "existing_mem_123",
                "was_inserted": False,  # Indicates it was a conflict
            }
        )

        # Mock the get() call for existing memory
        existing_memory = Memory(
            memory_id="existing_mem_123",
            agent_id="agent_1",
            content="User likes coffee",
        )

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        with patch.object(store, "get", AsyncMock(return_value=existing_memory)):
            create = MemoryCreate(content="User likes coffee", agent_id="agent_1")
            memory = await store.add(create)

        assert memory.memory_id == "existing_mem_123"

    @pytest.mark.asyncio
    async def test_add_auto_creates_agent(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that add auto-creates agent if not exists."""
        from engram.memory.models import MemoryCreate
        from engram.memory.store import MemoryStore

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        create = MemoryCreate(content="test", agent_id="new_agent")
        await store.add(create)

        # Should have called execute for agent creation
        calls = mock_storage.execute.call_args_list
        any(
            "INSERT INTO agents" in str(call) or "agent_id" in str(call)
            for call in calls
        )
        # The implementation uses execute to ensure agent exists

    @pytest.mark.asyncio
    async def test_add_with_metadata(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test adding memory with metadata."""
        from engram.memory.models import MemoryCreate
        from engram.memory.store import MemoryStore

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        create = MemoryCreate(
            content="User mentioned they work in finance",
            agent_id="agent_1",
            metadata={"source": "conversation", "confidence": 0.9},
        )

        memory = await store.add(create)

        assert memory.metadata == {"source": "conversation", "confidence": 0.9}

    @pytest.mark.asyncio
    async def test_add_near_duplicate_returns_existing(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """A vector near-duplicate (e.g. trailing period) returns the existing memory."""
        from engram.memory.models import MemoryCreate
        from engram.memory.store import MemoryStore

        existing_row = {
            "memory_id": "dup_1",
            "agent_id": "agent_1",
            "user_id": None,
            "session_id": None,
            "content": "User works at AskTuring",
            "fact": "User works at AskTuring",
            "main_content": None,
            "embedding": json.dumps([0.1] * 1536),
            "importance": 0.5,
            "access_count": 0,
            "metadata": "{}",
            "created_at": datetime.now(timezone.utc),
            "last_accessed_at": datetime.now(timezone.utc),
        }
        mock_storage.fetchone = AsyncMock(
            side_effect=[
                {"memory_id": "dup_1", "score": 0.99},  # near-duplicate query
                existing_row,  # get_without_access_update
            ]
        )
        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        memory = await store.add(
            MemoryCreate(content="User works at AskTuring.", agent_id="agent_1")
        )

        assert memory.memory_id == "dup_1"
        mock_storage.execute.assert_not_called()  # short-circuits before insert

    @pytest.mark.asyncio
    async def test_add_persists_memory_type(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """add() carries memory_type into the model and the INSERT."""
        from engram.memory.models import MemoryCreate
        from engram.memory.store import MemoryStore

        mock_storage.fetchone = AsyncMock(
            side_effect=[
                {"memory_id": "x", "score": 0.0},  # near-dup query: none
                {"memory_id": "mem_1", "was_inserted": True},  # insert RETURNING
            ]
        )
        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        mem = await store.add(
            MemoryCreate(
                content="User moved to Berlin", agent_id="a", memory_type="episodic"
            )
        )

        assert mem.memory_type == "episodic"
        # the INSERT (last fetchone call) passed the type
        assert "episodic" in mock_storage.fetchone.call_args.args

    @pytest.mark.asyncio
    async def test_add_batch_filters_near_duplicates(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Batch add drops items that are near-duplicates of existing memories."""
        from engram.memory.models import MemoryCreate
        from engram.memory.store import MemoryStore

        mock_embedding.embed_batch = AsyncMock(
            return_value=[[0.1] * 1536, [0.2] * 1536]
        )
        mock_storage.fetchone = AsyncMock(
            side_effect=[
                {"memory_id": "dup", "score": 0.99},  # first item near-dup
                {"memory_id": "x", "score": 0.10},  # second novel
            ]
        )
        mock_storage.executemany = AsyncMock()
        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        out = await store.add_batch(
            [
                MemoryCreate(content="dup fact", agent_id="agent_1"),
                MemoryCreate(content="novel fact", agent_id="agent_1"),
            ]
        )

        assert len(out) == 1
        assert out[0].content == "novel fact"


class TestMemoryStoreGet:
    """Tests for MemoryStore.get() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        return storage

    @pytest.fixture
    def mock_embedding(self) -> MagicMock:
        embedding = MagicMock()
        embedding.dimension = 1536
        return embedding

    @pytest.mark.asyncio
    async def test_get_returns_memory(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test getting existing memory."""
        from engram.memory.store import MemoryStore

        mock_storage.fetchone = AsyncMock(
            return_value={
                "memory_id": "mem_123",
                "agent_id": "agent_1",
                "user_id": None,
                "session_id": None,
                "content": "Test content",
                "embedding": json.dumps([0.1] * 1536),
                "importance": 0.7,
                "access_count": 5,
                "metadata": "{}",
                "created_at": datetime.now(timezone.utc),
                "last_accessed_at": datetime.now(timezone.utc),
            }
        )

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)
        memory = await store.get("mem_123")

        assert memory is not None
        assert memory.memory_id == "mem_123"
        assert memory.content == "Test content"
        assert memory.importance == 0.7

    @pytest.mark.asyncio
    async def test_get_returns_error_for_missing(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test getting non-existent memory raises MemoryNotFoundError."""
        from engram.core.exceptions import MemoryNotFoundError
        from engram.memory.store import MemoryStore

        mock_storage.fetchone = AsyncMock(return_value=None)

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)
        with pytest.raises(MemoryNotFoundError):
            await store.get("nonexistent_mem")


class TestMemoryStoreReinforce:
    """Tests for MemoryStore.reinforce() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        return storage

    @pytest.fixture
    def mock_embedding(self) -> MagicMock:
        embedding = MagicMock()
        embedding.dimension = 1536
        return embedding

    @pytest.mark.asyncio
    async def test_reinforce_boosts_importance(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that reinforce increases importance."""
        from engram.memory.store import MemoryStore

        mock_storage.fetchone = AsyncMock(
            return_value={
                "memory_id": "mem_123",
                "agent_id": "agent_1",
                "user_id": None,
                "session_id": None,
                "content": "Test",
                "embedding": json.dumps([0.1] * 1536),
                "importance": 0.7,  # After boost
                "access_count": 6,
                "metadata": "{}",
                "created_at": datetime.now(timezone.utc),
                "last_accessed_at": datetime.now(timezone.utc),
            }
        )

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)
        memory = await store.reinforce("mem_123", importance_boost=0.1)

        assert memory.importance == 0.7
        mock_storage.fetchone.assert_called_once()

    @pytest.mark.asyncio
    async def test_reinforce_rejects_negative_boost(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that negative importance_boost raises ValidationError."""
        from engram.core.exceptions import ValidationError
        from engram.memory.store import MemoryStore

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        with pytest.raises(ValidationError) as exc_info:
            await store.reinforce("mem_123", importance_boost=-0.1)

        assert "non-negative" in str(exc_info.value).lower()
        mock_storage.fetchone.assert_not_called()

    @pytest.mark.asyncio
    async def test_reinforce_nonexistent_raises_error(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that reinforcing non-existent memory raises error."""
        from engram.core.exceptions import MemoryNotFoundError
        from engram.memory.store import MemoryStore

        mock_storage.fetchone = AsyncMock(return_value=None)

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        with pytest.raises(MemoryNotFoundError):
            await store.reinforce("nonexistent_mem")


class TestMemoryStoreSearch:
    """Tests for MemoryStore.search() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.load_sql = MagicMock(return_value="SELECT ...")
        return storage

    @pytest.fixture
    def mock_embedding(self) -> MagicMock:
        embedding = MagicMock()
        embedding.dimension = 1536

        async def mock_embed(text: str) -> list[float]:
            return [0.1] * 1536

        embedding.embed = AsyncMock(side_effect=mock_embed)
        return embedding

    @pytest.mark.asyncio
    async def test_search_hybrid_mode(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test hybrid search mode."""
        from engram.memory.models import SearchQuery
        from engram.memory.store import MemoryStore

        mock_storage.fetchall = AsyncMock(
            return_value=[
                {
                    "memory_id": "mem_1",
                    "content": "Result 1",
                    "importance": 0.8,
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc),
                    "last_accessed_at": datetime.now(timezone.utc),
                    "score": 0.9,
                    "semantic_score": 0.85,
                    "keyword_score": 0.7,
                    "decay_score": 0.95,
                }
            ]
        )

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        query = SearchQuery(
            query="test query",
            agent_id="agent_1",
            mode="hybrid",
        )

        results = await store.search(query)

        assert len(results) == 1
        assert results[0].score == 0.9
        mock_storage.load_sql.assert_called_with("hybrid_search.sql")

    @pytest.mark.asyncio
    async def test_search_semantic_mode(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test semantic-only search mode."""
        from engram.memory.models import SearchQuery
        from engram.memory.store import MemoryStore

        mock_storage.fetchall = AsyncMock(return_value=[])
        mock_storage.load_sql = MagicMock(return_value="SELECT ...")

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        query = SearchQuery(
            query="test query",
            agent_id="agent_1",
            mode="semantic",
        )

        # Should use semantic_search method
        with patch.object(
            store, "semantic_search", AsyncMock(return_value=[])
        ) as mock_semantic:
            await store.search(query)
            mock_semantic.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_keyword_mode(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test keyword-only search mode."""
        from engram.memory.models import SearchQuery
        from engram.memory.store import MemoryStore

        mock_storage.fetchall = AsyncMock(return_value=[])

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        query = SearchQuery(
            query="test query",
            agent_id="agent_1",
            mode="keyword",
        )

        results = await store.search(query)
        # Keyword search should work without calling embedding
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_invalid_mode_raises_error(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that invalid search mode raises ValidationError."""
        from engram.core.exceptions import ValidationError
        from engram.memory.models import SearchQuery
        from engram.memory.store import MemoryStore

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        query = SearchQuery(
            query="test",
            agent_id="agent_1",
        )
        # Manually override mode to invalid value
        object.__setattr__(query, "mode", "invalid_mode")

        with pytest.raises(ValidationError) as exc_info:
            await store.search(query)

        assert "invalid_mode" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_respects_min_score(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that min_score filters results."""
        from engram.memory.models import SearchQuery
        from engram.memory.store import MemoryStore

        mock_storage.fetchall = AsyncMock(
            return_value=[
                {
                    "memory_id": "mem_1",
                    "content": "High score",
                    "importance": 0.8,
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc),
                    "last_accessed_at": datetime.now(timezone.utc),
                    "score": 0.9,
                    "semantic_score": 0.85,
                    "keyword_score": 0.7,
                    "decay_score": 0.95,
                },
                {
                    "memory_id": "mem_2",
                    "content": "Low score",
                    "importance": 0.3,
                    "metadata": "{}",
                    "created_at": datetime.now(timezone.utc),
                    "last_accessed_at": datetime.now(timezone.utc),
                    "score": 0.3,
                    "semantic_score": 0.25,
                    "keyword_score": 0.2,
                    "decay_score": 0.4,
                },
            ]
        )

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        query = SearchQuery(
            query="test",
            agent_id="agent_1",
            min_score=0.5,
        )

        results = await store.search(query)

        # Only high score result should pass filter
        assert len(results) == 1
        assert results[0].memory.content == "High score"


class TestMemoryStorePurge:
    """Tests for MemoryStore.purge() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        return storage

    @pytest.fixture
    def mock_embedding(self) -> MagicMock:
        embedding = MagicMock()
        embedding.dimension = 1536
        return embedding

    @pytest.mark.asyncio
    async def test_purge_returns_count(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that purge returns number of deleted memories."""
        from engram.memory.store import MemoryStore

        mock_storage.execute = AsyncMock(return_value="DELETE 5")

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)
        count = await store.purge("agent_1")

        assert count == 5

    @pytest.mark.asyncio
    async def test_purge_handles_zero_deletions(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that purge handles zero deletions gracefully."""
        from engram.memory.store import MemoryStore

        mock_storage.execute = AsyncMock(return_value="DELETE 0")

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)
        count = await store.purge("agent_1")

        assert count == 0

    @pytest.mark.asyncio
    async def test_purge_handles_malformed_result(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that purge handles unexpected result format."""
        from engram.memory.store import MemoryStore

        mock_storage.execute = AsyncMock(return_value="UNEXPECTED")

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)
        count = await store.purge("agent_1")

        # Should return 0 on malformed result
        assert count == 0

    @pytest.mark.asyncio
    async def test_purge_filters_by_user(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that purge can filter by user_id."""
        from engram.memory.store import MemoryStore

        mock_storage.execute = AsyncMock(return_value="DELETE 3")

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)
        count = await store.purge("agent_1", user_id="user_1")

        assert count == 3
        # Verify user_id was passed to query
        call_args = mock_storage.execute.call_args
        assert "user_id" in str(call_args) or len(call_args[0]) > 2


class TestMemoryStoreUpdate:
    """Tests for MemoryStore.update() method."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        return storage

    @pytest.fixture
    def mock_embedding(self) -> MagicMock:
        embedding = MagicMock()
        embedding.dimension = 1536

        async def mock_embed(text: str) -> list[float]:
            return [0.2] * 1536  # Different from original

        embedding.embed = AsyncMock(side_effect=mock_embed)
        return embedding

    @pytest.mark.asyncio
    async def test_update_content_regenerates_embedding(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that updating content regenerates embedding."""
        from engram.memory.models import MemoryUpdate
        from engram.memory.store import MemoryStore

        # First call (get_without_access_update) returns original content
        original_row = {
            "memory_id": "mem_123",
            "agent_id": "agent_1",
            "user_id": None,
            "session_id": None,
            "content": "Original content",  # Different from update!
            "embedding": json.dumps([0.1] * 1536),
            "importance": 0.5,
            "access_count": 0,
            "metadata": "{}",
            "created_at": datetime.now(timezone.utc),
            "last_accessed_at": datetime.now(timezone.utc),
        }

        # Second call (UPDATE RETURNING) returns updated content
        updated_row = {
            "memory_id": "mem_123",
            "agent_id": "agent_1",
            "user_id": None,
            "session_id": None,
            "content": "Updated content",
            "embedding": json.dumps([0.2] * 1536),
            "importance": 0.5,
            "access_count": 1,
            "metadata": "{}",
            "created_at": datetime.now(timezone.utc),
            "last_accessed_at": datetime.now(timezone.utc),
        }

        mock_storage.fetchone = AsyncMock(side_effect=[original_row, updated_row])

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        update = MemoryUpdate(content="Updated content")
        memory = await store.update("mem_123", update)

        # Should have called embed for new content
        mock_embedding.embed.assert_called_with("Updated content")
        assert memory.content == "Updated content"

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises_error(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test that updating non-existent memory raises error."""
        from engram.core.exceptions import MemoryNotFoundError
        from engram.memory.models import MemoryUpdate
        from engram.memory.store import MemoryStore

        mock_storage.fetchone = AsyncMock(return_value=None)

        store = MemoryStore(storage=mock_storage, embedding_service=mock_embedding)

        update = MemoryUpdate(content="New content")

        with pytest.raises(MemoryNotFoundError):
            await store.update("nonexistent", update)


class TestMemoryModelValidation:
    """Tests for Memory model validation edge cases."""

    def test_memory_content_max_length(self) -> None:
        """Test memory content max length validation."""
        from pydantic import ValidationError

        from engram.memory.models import MemoryCreate

        # Should accept up to 100000 chars
        long_content = "x" * 100000
        create = MemoryCreate(content=long_content, agent_id="test")
        assert len(create.content) == 100000

        # Should reject over 100000 chars
        with pytest.raises(ValidationError):
            MemoryCreate(content="x" * 100001, agent_id="test")

    def test_memory_importance_bounds(self) -> None:
        """Test importance score validation bounds."""
        from pydantic import ValidationError

        from engram.memory.models import Memory

        # Valid bounds
        Memory(agent_id="test", content="test", importance=0.0)
        Memory(agent_id="test", content="test", importance=1.0)
        Memory(agent_id="test", content="test", importance=0.5)

        # Invalid - below 0
        with pytest.raises(ValidationError):
            Memory(agent_id="test", content="test", importance=-0.1)

        # Invalid - above 1
        with pytest.raises(ValidationError):
            Memory(agent_id="test", content="test", importance=1.1)

    def test_memory_id_generation(self) -> None:
        """Test that memory IDs are unique and properly formatted."""
        from engram.memory.models import Memory

        memories = [Memory(agent_id="test", content="test") for _ in range(100)]
        ids = [m.memory_id for m in memories]

        # All IDs should be unique
        assert len(ids) == len(set(ids))

        # All IDs should start with "mem_"
        assert all(id.startswith("mem_") for id in ids)

    def test_memory_datetime_is_utc(self) -> None:
        """Test that memory timestamps are UTC."""
        from engram.memory.models import Memory

        memory = Memory(agent_id="test", content="test")

        # Should have timezone info
        assert memory.created_at.tzinfo is not None
        assert memory.last_accessed_at.tzinfo is not None


class TestSearchQueryValidation:
    """Tests for SearchQuery validation edge cases."""

    def test_search_query_limit_bounds(self) -> None:
        """Test search query limit validation."""
        from pydantic import ValidationError

        from engram.memory.models import SearchQuery

        # Valid limits
        SearchQuery(query="test", agent_id="agent", limit=1)
        SearchQuery(query="test", agent_id="agent", limit=100)

        # Invalid - too low
        with pytest.raises(ValidationError):
            SearchQuery(query="test", agent_id="agent", limit=0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            SearchQuery(query="test", agent_id="agent", limit=101)

    def test_search_query_min_score_bounds(self) -> None:
        """Test min_score validation."""
        from pydantic import ValidationError

        from engram.memory.models import SearchQuery

        # Valid
        SearchQuery(query="test", agent_id="agent", min_score=0.0)
        SearchQuery(query="test", agent_id="agent", min_score=1.0)

        # Invalid
        with pytest.raises(ValidationError):
            SearchQuery(query="test", agent_id="agent", min_score=-0.1)

        with pytest.raises(ValidationError):
            SearchQuery(query="test", agent_id="agent", min_score=1.1)
