"""Unit tests for health checker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestHealthChecker:
    """Tests for HealthChecker."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "database": "connected",
                "pool_size": 10,
            }
        )
        storage.fetchval = AsyncMock(return_value=1)
        return storage

    @pytest.fixture
    def mock_embedding(self) -> MagicMock:
        embedding = MagicMock()
        embedding.dimension = 1536
        embedding.model = "text-embedding-3-small"
        embedding.cache_info = {"size": 10, "max_size": 1000}

        async def mock_embed(text: str) -> list[float]:
            return [0.1] * 1536

        embedding.embed = AsyncMock(side_effect=mock_embed)
        return embedding

    @pytest.mark.asyncio
    async def test_check_returns_healthy(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test health check returns healthy status."""
        from engram.health.checker import HealthChecker

        checker = HealthChecker(storage=mock_storage, embedding_service=mock_embedding)

        result = await checker.check()

        assert result["status"] == "healthy"
        assert "components" in result
        assert "database" in result["components"]
        assert "embedding" in result["components"]

    @pytest.mark.asyncio
    async def test_check_includes_version(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test health check includes version."""
        from engram.health.checker import HealthChecker

        checker = HealthChecker(storage=mock_storage, embedding_service=mock_embedding)

        result = await checker.check()

        assert "version" in result
        assert isinstance(result["version"], str)

    @pytest.mark.asyncio
    async def test_check_includes_timestamp(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test health check includes timestamp."""
        from engram.health.checker import HealthChecker

        checker = HealthChecker(storage=mock_storage, embedding_service=mock_embedding)

        result = await checker.check()

        assert "timestamp" in result
        assert "T" in result["timestamp"]  # ISO format

    @pytest.mark.asyncio
    async def test_check_includes_duration(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test health check includes duration."""
        from engram.health.checker import HealthChecker

        checker = HealthChecker(storage=mock_storage, embedding_service=mock_embedding)

        result = await checker.check()

        assert "check_duration_ms" in result
        assert isinstance(result["check_duration_ms"], (int, float))
        assert result["check_duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_check_database_failure(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test health check when database fails."""
        from engram.health.checker import HealthChecker

        mock_storage.health_check = AsyncMock(
            return_value={
                "status": "unhealthy",
                "error": "Connection refused",
            }
        )

        checker = HealthChecker(storage=mock_storage, embedding_service=mock_embedding)

        result = await checker.check()

        assert result["status"] == "unhealthy"
        assert result["components"]["database"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_check_embedding_failure(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test health check when embedding service fails."""
        from engram.health.checker import HealthChecker

        mock_embedding.embed = AsyncMock(side_effect=Exception("API error"))

        checker = HealthChecker(storage=mock_storage, embedding_service=mock_embedding)

        result = await checker.check()

        assert result["status"] == "unhealthy"
        assert result["components"]["embedding"]["status"] == "unhealthy"
        assert "error" in result["components"]["embedding"]

    @pytest.mark.asyncio
    async def test_check_skip_embedding_test(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test health check can skip embedding API call."""
        from engram.health.checker import HealthChecker

        checker = HealthChecker(storage=mock_storage, embedding_service=mock_embedding)

        result = await checker.check(skip_embedding_test=True)

        assert result["status"] == "healthy"
        assert result["components"]["embedding"]["test_skipped"] is True
        # Should NOT call embed
        mock_embedding.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_no_embedding_service(self, mock_storage: MagicMock) -> None:
        """Test health check without embedding service."""
        from engram.health.checker import HealthChecker

        checker = HealthChecker(storage=mock_storage, embedding_service=None)

        result = await checker.check()

        assert result["status"] == "healthy"
        assert result["components"]["embedding"] is None

    @pytest.mark.asyncio
    async def test_check_includes_system_info(
        self, mock_storage: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Test health check includes system info."""
        from engram.health.checker import HealthChecker

        checker = HealthChecker(storage=mock_storage, embedding_service=mock_embedding)

        result = await checker.check()

        assert "system" in result
        assert "python_version" in result["system"]
        assert "platform" in result["system"]


class TestHealthCheckerQuick:
    """Tests for quick health check."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.fetchval = AsyncMock(return_value=1)
        return storage

    @pytest.mark.asyncio
    async def test_check_quick_returns_true(self, mock_storage: MagicMock) -> None:
        """Test quick health check returns True when healthy."""
        from engram.health.checker import HealthChecker

        checker = HealthChecker(storage=mock_storage)

        result = await checker.check_quick()

        assert result is True
        mock_storage.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_quick_returns_false_on_error(
        self, mock_storage: MagicMock
    ) -> None:
        """Test quick health check returns False on error."""
        from engram.health.checker import HealthChecker

        mock_storage.fetchval = AsyncMock(side_effect=Exception("Connection failed"))

        checker = HealthChecker(storage=mock_storage)

        result = await checker.check_quick()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_quick_returns_false_wrong_result(
        self, mock_storage: MagicMock
    ) -> None:
        """Test quick health check returns False on wrong result."""
        from engram.health.checker import HealthChecker

        mock_storage.fetchval = AsyncMock(return_value=0)  # Wrong result

        checker = HealthChecker(storage=mock_storage)

        result = await checker.check_quick()

        assert result is False
