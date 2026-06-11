"""Comprehensive unit tests for configuration with edge cases."""

from __future__ import annotations

import pytest


class TestEngramSettingsValidation:
    """Tests for EngramSettings validation."""

    def test_weight_sum_validation_pass(self) -> None:
        """Test that valid weight sum passes validation."""
        from engram.core.config import EngramSettings

        # Exactly 1.0
        settings = EngramSettings(
            weight_semantic=0.40,
            weight_keyword=0.20,
            weight_decay=0.25,
            weight_importance=0.15,
        )
        assert settings.weight_semantic == 0.40

    def test_weight_sum_validation_fail(self) -> None:
        """Test that invalid weight sum fails validation."""
        from pydantic import ValidationError

        from engram.core.config import EngramSettings

        with pytest.raises(ValidationError) as exc_info:
            EngramSettings(
                weight_semantic=0.5,
                weight_keyword=0.3,
                weight_decay=0.3,
                weight_importance=0.2,  # Sum = 1.3
            )

        assert "weights must sum to 1.0" in str(exc_info.value).lower()

    def test_weight_sum_allows_tolerance(self) -> None:
        """Test that weight sum allows small tolerance (0.99-1.01)."""
        from engram.core.config import EngramSettings

        # Just under 1.0 (within tolerance)
        settings = EngramSettings(
            weight_semantic=0.395,
            weight_keyword=0.20,
            weight_decay=0.25,
            weight_importance=0.15,  # Sum = 0.995
        )
        assert settings is not None

    def test_pool_size_validation_pass(self) -> None:
        """Test that valid pool sizes pass."""
        from engram.core.config import EngramSettings

        settings = EngramSettings(
            min_pool_size=5,
            max_pool_size=20,
        )
        assert settings.min_pool_size == 5
        assert settings.max_pool_size == 20

    def test_pool_size_validation_fail(self) -> None:
        """Test that max < min fails validation."""
        from pydantic import ValidationError

        from engram.core.config import EngramSettings

        with pytest.raises(ValidationError) as exc_info:
            EngramSettings(
                min_pool_size=20,
                max_pool_size=5,  # Less than min
            )

        assert "max_pool_size" in str(exc_info.value).lower()

    def test_pool_size_equal_allowed(self) -> None:
        """Test that max == min is allowed."""
        from engram.core.config import EngramSettings

        settings = EngramSettings(
            min_pool_size=10,
            max_pool_size=10,
        )
        assert settings.min_pool_size == settings.max_pool_size

    def test_weight_sum_validated_when_other_weights_default(self) -> None:
        """Setting only one weight must still trigger the sum check.

        Field validators skip default values, so the cross-field check must
        run as a model validator to catch e.g. weight_semantic=0.9 with the
        other weights left at their defaults (sum 1.5).
        """
        from pydantic import ValidationError

        from engram.core.config import EngramSettings

        with pytest.raises(ValidationError) as exc_info:
            EngramSettings(weight_semantic=0.9)

        assert "weights must sum to 1.0" in str(exc_info.value).lower()

    def test_text_search_config_rejects_injection(self) -> None:
        """The config name is interpolated into DDL; restrict it hard."""
        from pydantic import ValidationError

        from engram.core.config import EngramSettings

        with pytest.raises(ValidationError):
            EngramSettings(text_search_config="english'; DROP TABLE agents; --")

    def test_text_search_config_accepts_valid_names(self) -> None:
        from engram.core.config import EngramSettings

        assert EngramSettings(text_search_config="german").text_search_config == (
            "german"
        )
        assert EngramSettings().text_search_config == "english"

    def test_pool_size_validated_when_max_defaults(self) -> None:
        """min_pool_size above the default max_pool_size must fail."""
        from pydantic import ValidationError

        from engram.core.config import EngramSettings

        with pytest.raises(ValidationError) as exc_info:
            EngramSettings(min_pool_size=50)  # default max is 20

        assert "max_pool_size" in str(exc_info.value).lower()


class TestEmbeddingDimensionCoercion:
    """Tests for embedding dimension coercion."""

    def test_dimension_from_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that string dimension is coerced to int."""
        from engram.core.config import EngramSettings

        monkeypatch.setenv("ENGRAM_EMBEDDING_DIMENSION", "384")

        settings = EngramSettings()
        assert settings.embedding_dimension == 384
        assert isinstance(settings.embedding_dimension, int)

    def test_dimension_none_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that empty string becomes None."""
        from engram.core.config import EngramSettings

        monkeypatch.setenv("ENGRAM_EMBEDDING_DIMENSION", "")

        EngramSettings()
        # Should use default or None
        # Default is 1536 for OpenAI

    def test_dimension_invalid_string_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that invalid string raises error."""
        from pydantic import ValidationError

        from engram.core.config import EngramSettings

        monkeypatch.setenv("ENGRAM_EMBEDDING_DIMENSION", "not_a_number")

        with pytest.raises(ValidationError) as exc_info:
            EngramSettings()

        assert "embedding_dimension" in str(exc_info.value).lower()


class TestProviderKwargs:
    """Tests for get_*_provider_kwargs methods."""

    def test_embedding_kwargs_openai(self) -> None:
        """Test embedding kwargs for OpenAI provider."""
        from engram.core.config import EngramSettings

        settings = EngramSettings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key="sk-test-key",
        )

        kwargs = settings.get_embedding_provider_kwargs()

        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["api_key"] == "sk-test-key"

    def test_embedding_kwargs_sentence_transformers(self) -> None:
        """Test embedding kwargs for Sentence Transformers."""
        from engram.core.config import EngramSettings

        settings = EngramSettings(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
        )

        kwargs = settings.get_embedding_provider_kwargs()

        assert kwargs["model"] == "all-MiniLM-L6-v2"
        assert kwargs["dimension"] == 384
        # Should NOT have api_key

    def test_embedding_kwargs_cohere(self) -> None:
        """Test embedding kwargs for Cohere."""
        from engram.core.config import EngramSettings

        settings = EngramSettings(
            embedding_provider="cohere",
            embedding_model="embed-english-v3.0",
            cohere_api_key="cohere-test-key",
        )

        kwargs = settings.get_embedding_provider_kwargs()

        assert kwargs["api_key"] == "cohere-test-key"

    def test_llm_kwargs_openai(self) -> None:
        """Test LLM kwargs for OpenAI."""
        from engram.core.config import EngramSettings

        settings = EngramSettings(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            openai_api_key="sk-test",
            openai_base_url="https://custom.api.com",
        )

        kwargs = settings.get_llm_provider_kwargs()

        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["base_url"] == "https://custom.api.com"

    def test_llm_kwargs_anthropic(self) -> None:
        """Test LLM kwargs for Anthropic."""
        from engram.core.config import EngramSettings

        settings = EngramSettings(
            llm_provider="anthropic",
            llm_model="claude-3-sonnet-20240229",
            anthropic_api_key="anthropic-key",
        )

        kwargs = settings.get_llm_provider_kwargs()

        assert kwargs["api_key"] == "anthropic-key"

    def test_llm_kwargs_ollama(self) -> None:
        """Test LLM kwargs for Ollama."""
        from engram.core.config import EngramSettings

        settings = EngramSettings(
            llm_provider="ollama",
            llm_model="llama3",
            ollama_base_url="http://localhost:11434",
        )

        kwargs = settings.get_llm_provider_kwargs()

        assert kwargs["model"] == "llama3"
        assert kwargs["base_url"] == "http://localhost:11434"


class TestSettingsCache:
    """Tests for settings caching."""

    def test_get_settings_cached(self) -> None:
        """Test that get_settings returns cached instance."""
        from engram.core.config import clear_settings_cache, get_settings

        clear_settings_cache()

        s1 = get_settings()
        s2 = get_settings()

        assert s1 is s2

    def test_clear_cache_creates_new_instance(self) -> None:
        """Test that clearing cache creates new instance."""
        from engram.core.config import clear_settings_cache, get_settings

        s1 = get_settings()
        clear_settings_cache()
        s2 = get_settings()

        # Should be different instances (though may have same values)
        assert s1 is not s2


class TestSettingsFromEnv:
    """Tests for loading settings from environment."""

    def test_database_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading database URL from env."""
        from engram.core.config import EngramSettings

        monkeypatch.setenv("ENGRAM_DATABASE_URL", "postgresql://user:pass@host:5432/db")

        settings = EngramSettings()

        assert settings.database_url == "postgresql://user:pass@host:5432/db"

    def test_all_api_keys_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading all API keys from env."""
        from engram.core.config import EngramSettings

        monkeypatch.setenv("ENGRAM_OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("ENGRAM_ANTHROPIC_API_KEY", "sk-anthropic")
        monkeypatch.setenv("ENGRAM_COHERE_API_KEY", "cohere-key")
        monkeypatch.setenv("ENGRAM_GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("ENGRAM_HF_API_KEY", "hf-key")

        settings = EngramSettings()

        assert settings.openai_api_key == "sk-openai"
        assert settings.anthropic_api_key == "sk-anthropic"
        assert settings.cohere_api_key == "cohere-key"
        assert settings.groq_api_key == "groq-key"
        assert settings.hf_api_key == "hf-key"

    def test_boolean_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading boolean settings from env."""
        from engram.core.config import EngramSettings

        monkeypatch.setenv("ENGRAM_LOG_SQL_QUERIES", "true")

        settings = EngramSettings()

        assert settings.log_sql_queries is True
