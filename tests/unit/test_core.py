"""Unit tests for core configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    from engram.core._types import (
        AgentId,
        MemoryId,
        Metadata,
        Vector,
    )


class TestEngramSettings:
    """Tests for EngramSettings configuration."""

    def test_settings_are_valid(self) -> None:
        """Test that settings load and have valid structure."""
        from engram.core.config import EngramSettings

        settings = EngramSettings()

        # Check that essential fields are present and valid
        assert settings.database_url.startswith("postgresql://")
        assert settings.embedding_provider in [
            "openai",
            "sentence-transformers",
            "cohere",
            "ollama",
            "huggingface",
        ]
        assert isinstance(settings.embedding_dimension, int)
        assert settings.embedding_dimension > 0
        assert 0 <= settings.weight_semantic <= 1
        assert settings.min_pool_size <= settings.max_pool_size

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test settings loaded from environment variables."""
        from engram.core.config import EngramSettings

        monkeypatch.setenv("ENGRAM_DATABASE_URL", "postgresql://custom:5432/test")
        monkeypatch.setenv("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
        monkeypatch.setenv("ENGRAM_EMBEDDING_DIMENSION", "384")

        settings = EngramSettings()
        assert settings.database_url == "postgresql://custom:5432/test"
        assert settings.embedding_provider == "sentence-transformers"
        assert settings.embedding_dimension == 384

    def test_weight_validation(self) -> None:
        """Test that search weights must sum to 1.0."""
        from engram.core.config import EngramSettings

        # Valid weights (sum to 1.0)
        settings = EngramSettings(
            weight_semantic=0.4,
            weight_keyword=0.2,
            weight_decay=0.25,
            weight_importance=0.15,
        )
        assert settings.weight_semantic == 0.4

    def test_pool_size_validation(self) -> None:
        """Test that max_pool_size >= min_pool_size."""
        from engram.core.config import EngramSettings

        # Valid: max >= min
        settings = EngramSettings(min_pool_size=5, max_pool_size=20)
        assert settings.min_pool_size == 5
        assert settings.max_pool_size == 20


class TestExceptions:
    """Tests for exception hierarchy."""

    def test_engram_error_base(self) -> None:
        """Test EngramError base exception."""
        from engram.core.exceptions import EngramError

        error = EngramError("Test error", key="value")
        assert str(error) == "Test error (key='value')"
        assert error.context == {"key": "value"}

    def test_memory_not_found(self) -> None:
        """Test MemoryNotFoundError."""
        from engram.core.exceptions import MemoryNotFoundError

        error = MemoryNotFoundError("mem_123")
        assert error.memory_id == "mem_123"
        assert "mem_123" in str(error)

    def test_exception_inheritance(self) -> None:
        """Test exception hierarchy."""
        from engram.core.exceptions import (
            ConnectionError,
            EngramError,
            MemoryNotFoundError,
            StorageError,
        )

        assert issubclass(StorageError, EngramError)
        assert issubclass(MemoryNotFoundError, StorageError)
        assert issubclass(ConnectionError, EngramError)


class TestTypes:
    """Tests for type definitions."""

    def test_type_aliases_exist(self) -> None:
        """Test that all type aliases are defined."""

        # Type aliases should be usable
        agent: AgentId = "agent_123"
        memory: MemoryId = "mem_456"
        vector: Vector = [0.1, 0.2, 0.3]
        metadata: Metadata = {"key": "value"}

        assert isinstance(agent, str)
        assert isinstance(memory, str)
        assert isinstance(vector, list)
        assert isinstance(metadata, dict)
