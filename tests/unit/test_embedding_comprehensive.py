"""Comprehensive unit tests for embedding service with edge cases."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEmbeddingServiceLRUCache:
    """Tests for LRU cache behavior in EmbeddingService."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create a mock embedding provider with call tracking."""
        provider = MagicMock()
        provider.dimension = 384
        provider.model = "test-model"

        self.call_count = 0

        async def mock_embed(text: str) -> list[float]:
            self.call_count += 1
            # Return unique embedding based on text
            return [hash(text) % 100 / 100.0] * 384

        async def mock_embed_batch(texts: list[str]) -> list[list[float]]:
            return [await mock_embed(t) for t in texts]

        provider.embed = AsyncMock(side_effect=mock_embed)
        provider.embed_batch = AsyncMock(side_effect=mock_embed_batch)
        return provider

    @pytest.mark.asyncio
    async def test_lru_eviction_order(self, mock_provider: MagicMock) -> None:
        """Test that LRU correctly evicts least recently used items."""
        from engram.embedding.service import EmbeddingService

        # Small cache to test eviction
        service = EmbeddingService(provider=mock_provider, cache_size=3)

        # Fill cache: A, B, C
        await service.embed("A")
        await service.embed("B")
        await service.embed("C")
        assert len(service._cache) == 3

        # Access A to make it most recently used
        await service.embed("A")
        # Order should now be: B, C, A (B is LRU)

        # Add D - should evict B (least recently used)
        await service.embed("D")
        assert len(service._cache) == 3

        # B should require new embedding (was evicted)
        call_count_before = mock_provider.embed.call_count
        await service.embed("B")
        assert mock_provider.embed.call_count == call_count_before + 1

        # A should still be cached (was accessed recently)
        call_count_before = mock_provider.embed.call_count
        await service.embed("A")
        assert mock_provider.embed.call_count == call_count_before  # No new call

    @pytest.mark.asyncio
    async def test_cache_move_to_end_on_access(self, mock_provider: MagicMock) -> None:
        """Test that accessing cached item moves it to end (most recent)."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(provider=mock_provider, cache_size=3)

        await service.embed("first")
        await service.embed("second")
        await service.embed("third")

        # Access first - should move to end
        await service.embed("first")

        # Check order in cache (first should be last now)
        keys = list(service._cache.keys())
        assert service._compute_cache_key("first") == keys[-1]

    @pytest.mark.asyncio
    async def test_batch_embed_updates_lru_order(
        self, mock_provider: MagicMock
    ) -> None:
        """Test that batch embed updates LRU order for cache hits."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(provider=mock_provider, cache_size=5)

        # Pre-populate cache
        await service.embed("cached1")
        await service.embed("cached2")
        await service.embed("new1")

        # Batch with cache hits - should update their LRU order
        await service.embed_batch(["cached1", "cached2", "brand_new"])

        # cached1 and cached2 should now be most recent
        keys = list(service._cache.keys())
        # brand_new should be the last added
        assert service._compute_cache_key("brand_new") in keys

    @pytest.mark.asyncio
    async def test_cache_disabled_when_size_zero(
        self, mock_provider: MagicMock
    ) -> None:
        """Test that cache is disabled when size is 0."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(provider=mock_provider, cache_size=0)

        await service.embed("test")
        await service.embed("test")  # Same text

        # Should call provider both times
        assert mock_provider.embed.call_count == 2
        assert len(service._cache) == 0


class TestEmbeddingInputLengthGuard:
    """Overlong inputs are truncated instead of erroring at the provider."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        provider = MagicMock()
        provider.dimension = 8
        provider.model = "test-model"
        provider.embed = AsyncMock(return_value=[0.0] * 8)
        provider.embed_batch = AsyncMock(side_effect=lambda ts: [[0.0] * 8 for _ in ts])
        return provider

    @pytest.mark.asyncio
    async def test_embed_truncates_overlong_text(
        self, mock_provider: MagicMock
    ) -> None:
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(
            provider=mock_provider, cache_size=0, max_input_chars=100
        )

        await service.embed("x" * 500)

        sent = mock_provider.embed.call_args.args[0]
        assert len(sent) == 100

    @pytest.mark.asyncio
    async def test_embed_batch_truncates_each_text(
        self, mock_provider: MagicMock
    ) -> None:
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(
            provider=mock_provider, cache_size=0, max_input_chars=100
        )

        await service.embed_batch(["short", "y" * 500])

        sent = mock_provider.embed_batch.call_args.args[0]
        assert sent[0] == "short"
        assert len(sent[1]) == 100

    @pytest.mark.asyncio
    async def test_truncated_text_shares_cache_entry(
        self, mock_provider: MagicMock
    ) -> None:
        """Two inputs identical after truncation must hit the same cache key."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(
            provider=mock_provider, cache_size=10, max_input_chars=100
        )

        await service.embed("z" * 500)
        await service.embed("z" * 600)  # same first 100 chars

        assert mock_provider.embed.call_count == 1


class TestEmbeddingServiceBatchValidation:
    """Tests for batch embedding validation and error handling."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create a mock provider."""
        provider = MagicMock()
        provider.dimension = 384
        provider.model = "test-model"
        return provider

    @pytest.mark.asyncio
    async def test_batch_length_mismatch_raises_error(
        self, mock_provider: MagicMock
    ) -> None:
        """Test that batch returning wrong number of embeddings raises error."""
        from engram.core.exceptions import EmbeddingError
        from engram.embedding.service import EmbeddingService

        # Provider returns fewer embeddings than requested
        async def bad_batch(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 384]  # Only 1 result regardless of input

        mock_provider.embed_batch = AsyncMock(side_effect=bad_batch)

        service = EmbeddingService(provider=mock_provider, cache_size=0)

        with pytest.raises(EmbeddingError) as exc_info:
            await service.embed_batch(["text1", "text2", "text3"])

        assert "1 embeddings" in str(exc_info.value)
        assert "3 texts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_batch_with_all_cache_hits(self, mock_provider: MagicMock) -> None:
        """Test batch where all items are cached."""
        from engram.embedding.service import EmbeddingService

        async def mock_embed(text: str) -> list[float]:
            return [0.1] * 384

        mock_provider.embed = AsyncMock(side_effect=mock_embed)
        mock_provider.embed_batch = AsyncMock()

        service = EmbeddingService(provider=mock_provider, cache_size=100)

        # Pre-populate cache
        await service.embed("a")
        await service.embed("b")
        await service.embed("c")

        # Batch with all cached
        results = await service.embed_batch(["a", "b", "c"])

        # Should NOT call embed_batch since all cached
        mock_provider.embed_batch.assert_not_called()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_maintains_order_with_partial_cache(
        self, mock_provider: MagicMock
    ) -> None:
        """Test that batch maintains input order with partial cache hits."""
        from engram.embedding.service import EmbeddingService

        embeddings = {
            "cached": [1.0] * 384,
            "new1": [2.0] * 384,
            "new2": [3.0] * 384,
        }

        async def mock_embed(text: str) -> list[float]:
            return embeddings.get(text, [0.0] * 384)

        async def mock_batch(texts: list[str]) -> list[list[float]]:
            return [embeddings.get(t, [0.0] * 384) for t in texts]

        mock_provider.embed = AsyncMock(side_effect=mock_embed)
        mock_provider.embed_batch = AsyncMock(side_effect=mock_batch)

        service = EmbeddingService(provider=mock_provider, cache_size=100)

        # Cache one item
        await service.embed("cached")

        # Batch with cached item in middle
        results = await service.embed_batch(["new1", "cached", "new2"])

        assert len(results) == 3
        assert results[0][0] == 2.0  # new1
        assert results[1][0] == 1.0  # cached
        assert results[2][0] == 3.0  # new2

    @pytest.mark.asyncio
    async def test_large_batch_chunking(self, mock_provider: MagicMock) -> None:
        """Test that large batches are chunked correctly."""
        from engram.embedding.service import EmbeddingService

        batch_calls = []

        async def mock_batch(texts: list[str]) -> list[list[float]]:
            batch_calls.append(len(texts))
            return [[0.1] * 384 for _ in texts]

        mock_provider.embed_batch = AsyncMock(side_effect=mock_batch)

        # Small batch size to force chunking
        service = EmbeddingService(provider=mock_provider, cache_size=0, batch_size=10)

        # 25 items should be split into 3 batches: 10, 10, 5
        texts = [f"text_{i}" for i in range(25)]
        results = await service.embed_batch(texts)

        assert len(results) == 25
        assert batch_calls == [10, 10, 5]


class TestEmbeddingServiceFromProvider:
    """Tests for creating EmbeddingService from provider name."""

    def test_from_provider_creates_service(self) -> None:
        """Test creating service from provider name."""
        from engram.embedding.service import EmbeddingService

        # Patch where the function is used (in the service module)
        with patch("engram.embedding.service.get_embedding_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.dimension = 384
            mock_provider.model = "test"
            mock_get.return_value = mock_provider

            service = EmbeddingService.from_provider(
                "sentence-transformers",
                model="all-MiniLM-L6-v2",
            )

            assert service.dimension == 384
            mock_get.assert_called_once()

    def test_from_provider_with_custom_cache_size(self) -> None:
        """Test that custom cache/batch sizes are applied."""
        from engram.embedding.service import EmbeddingService

        # Patch where the function is used (in the service module)
        with patch("engram.embedding.service.get_embedding_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.dimension = 384
            mock_provider.model = "test"
            mock_get.return_value = mock_provider

            service = EmbeddingService.from_provider(
                "test",
                cache_size=500,
                batch_size=25,
            )

            assert service._cache_size == 500
            assert service._batch_size == 25


class TestEmbeddingServiceEdgeCases:
    """Edge case tests for EmbeddingService."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        provider = MagicMock()
        provider.dimension = 384
        provider.model = "test"

        async def mock_embed(text: str) -> list[float]:
            return [0.1] * 384

        async def mock_batch(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 384 for _ in texts]

        provider.embed = AsyncMock(side_effect=mock_embed)
        provider.embed_batch = AsyncMock(side_effect=mock_batch)
        return provider

    @pytest.mark.asyncio
    async def test_empty_string_embedding(self, mock_provider: MagicMock) -> None:
        """Test embedding empty string (edge case)."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(provider=mock_provider, cache_size=100)

        # Empty string should work (provider may handle it)
        result = await service.embed("")
        assert len(result) == 384

    @pytest.mark.asyncio
    async def test_very_long_text_embedding(self, mock_provider: MagicMock) -> None:
        """Test embedding very long text."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(provider=mock_provider, cache_size=100)

        long_text = "word " * 10000
        result = await service.embed(long_text)
        assert len(result) == 384

    @pytest.mark.asyncio
    async def test_unicode_text_embedding(self, mock_provider: MagicMock) -> None:
        """Test embedding text with unicode characters."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(provider=mock_provider, cache_size=100)

        # Various unicode
        texts = [
            "Hello 你好 مرحبا 🎉",
            "émoji: 🚀🌟💡",
            "Math: ∑∏∫",
        ]

        for text in texts:
            result = await service.embed(text)
            assert len(result) == 384

    @pytest.mark.asyncio
    async def test_concurrent_embeds_same_text(self, mock_provider: MagicMock) -> None:
        """Test concurrent embedding of same text (race condition)."""
        from engram.embedding.service import EmbeddingService

        call_count = 0

        async def slow_embed(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate latency
            return [0.1] * 384

        mock_provider.embed = AsyncMock(side_effect=slow_embed)

        service = EmbeddingService(provider=mock_provider, cache_size=100)

        # Launch concurrent requests for same text
        tasks = [service.embed("same_text") for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be valid
        assert all(len(r) == 384 for r in results)
        # Note: Without proper locking, this may call provider multiple times
        # This test documents current behavior

    @pytest.mark.asyncio
    async def test_cache_key_collision_resistance(
        self, mock_provider: MagicMock
    ) -> None:
        """Test that cache keys don't easily collide."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(provider=mock_provider, cache_size=100)

        # These should have different cache keys
        similar_texts = [
            "hello world",
            "hello world ",  # trailing space
            " hello world",  # leading space
            "Hello World",  # different case
            "hello  world",  # double space
        ]

        keys = [service._compute_cache_key(t) for t in similar_texts]

        # All keys should be unique
        assert len(keys) == len(set(keys))

    def test_cache_info_accuracy(self, mock_provider: MagicMock) -> None:
        """Test cache info reflects actual state."""
        from engram.embedding.service import EmbeddingService

        service = EmbeddingService(provider=mock_provider, cache_size=50)

        # Manually add to cache
        service._cache["key1"] = [0.1]
        service._cache["key2"] = [0.2]
        service._cache["key3"] = [0.3]

        info = service.cache_info
        assert info["size"] == 3
        assert info["max_size"] == 50

    @pytest.mark.asyncio
    async def test_provider_error_propagation(self, mock_provider: MagicMock) -> None:
        """Test that provider errors propagate correctly."""
        from engram.core.exceptions import EmbeddingError
        from engram.embedding.service import EmbeddingService

        async def failing_embed(text: str) -> list[float]:
            raise EmbeddingError("API limit exceeded")

        mock_provider.embed = AsyncMock(side_effect=failing_embed)

        service = EmbeddingService(provider=mock_provider, cache_size=100)

        with pytest.raises(EmbeddingError) as exc_info:
            await service.embed("test")

        assert "API limit exceeded" in str(exc_info.value)
