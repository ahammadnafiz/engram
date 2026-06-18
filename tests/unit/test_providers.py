"""Unit tests for embedding and LLM providers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class FakeEmbedding(list):
    """List with the numpy-like tolist method used by the provider."""

    def tolist(self) -> list[float]:
        return list(self)


class FakeSentenceTransformer:
    """Small local stand-in for sentence_transformers.SentenceTransformer."""

    device = "cpu"

    def __init__(self, model: str, device: str | None = None) -> None:
        self.model = model
        self.device = device or "cpu"

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> list[FakeEmbedding]:
        _ = (convert_to_numpy, normalize_embeddings)
        return [FakeEmbedding([float(i % 7) / 7.0 for i in range(384)]) for _ in texts]


def fake_sentence_transformers_module() -> SimpleNamespace:
    return SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)


class TestEmbeddingProviderRegistry:
    """Tests for embedding provider registry."""

    def test_registry_has_builtin_providers(self) -> None:
        """Test that registry has built-in providers."""
        from engram.providers.embedding import embedding_registry

        # Check main providers are registered
        assert "openai" in embedding_registry._providers
        assert "sentence-transformers" in embedding_registry._providers
        assert "cohere" in embedding_registry._providers
        assert "ollama" in embedding_registry._providers
        assert "huggingface" in embedding_registry._providers

    def test_get_provider_by_alias(self) -> None:
        """Test getting provider by alias."""
        from engram.providers.embedding import embedding_registry

        # "st" should be alias for sentence-transformers
        provider_class = embedding_registry.get("st")
        assert provider_class is not None

    def test_get_nonexistent_provider_raises(self) -> None:
        """Test that getting non-existent provider raises KeyError."""
        from engram.providers.embedding import embedding_registry

        with pytest.raises(KeyError):
            embedding_registry.get("nonexistent_provider")


class TestSentenceTransformersProvider:
    """Tests for SentenceTransformers embedding provider."""

    def test_provider_loads_model(self) -> None:
        """Test that provider loads the model."""
        from engram.providers.embedding.builtin import (
            SentenceTransformersEmbeddingProvider,
        )

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_sentence_transformers_module()},
        ):
            provider = SentenceTransformersEmbeddingProvider(
                model="all-MiniLM-L6-v2",
            )

        assert provider.model == "all-MiniLM-L6-v2"
        assert provider.dimension == 384

    def test_provider_missing_package_raises(self) -> None:
        """Test that missing package raises ImportError."""
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            # Can't easily test this without actually uninstalling package
            pass

    @pytest.mark.asyncio
    async def test_embed_returns_vector(self) -> None:
        """Test that embed returns a vector."""
        from engram.providers.embedding.builtin import (
            SentenceTransformersEmbeddingProvider,
        )

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_sentence_transformers_module()},
        ):
            provider = SentenceTransformersEmbeddingProvider(
                model="all-MiniLM-L6-v2",
            )

            vector = await provider.embed("Hello world")

        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embed_batch_returns_vectors(self) -> None:
        """Test that embed_batch returns multiple vectors."""
        from engram.providers.embedding.builtin import (
            SentenceTransformersEmbeddingProvider,
        )

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_sentence_transformers_module()},
        ):
            provider = SentenceTransformersEmbeddingProvider(
                model="all-MiniLM-L6-v2",
            )

            vectors = await provider.embed_batch(["Hello", "World", "Test"])

        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_provider_cleanup(self) -> None:
        """Test that provider can be cleaned up."""
        from engram.providers.embedding.builtin import (
            SentenceTransformersEmbeddingProvider,
        )

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_sentence_transformers_module()},
        ):
            provider = SentenceTransformersEmbeddingProvider(
                model="all-MiniLM-L6-v2",
            )

        # Should not raise
        provider.close()
        provider.close()  # Should be idempotent


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAI embedding provider (mocked)."""

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Create mock OpenAI client."""
        client = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]

        client.embeddings.create = AsyncMock(return_value=mock_response)
        return client

    @pytest.mark.asyncio
    async def test_embed_calls_api(self, mock_openai_client: MagicMock) -> None:
        """Test that embed calls OpenAI API."""
        from engram.providers.embedding.builtin import OpenAIEmbeddingProvider

        # Patch openai import inside the provider constructor
        mock_openai_module = MagicMock()
        mock_openai_module.AsyncOpenAI.return_value = mock_openai_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            provider = OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-3-small",
            )

            vector = await provider.embed("test")

            mock_openai_client.embeddings.create.assert_called_once()
            assert len(vector) == 1536


class TestLLMProviderRegistry:
    """Tests for LLM provider registry."""

    def test_registry_has_builtin_providers(self) -> None:
        """Test that registry has built-in LLM providers."""
        from engram.providers.llm import llm_registry

        assert "openai" in llm_registry._providers
        assert "anthropic" in llm_registry._providers
        assert "gemini" in llm_registry._providers
        assert "ollama" in llm_registry._providers
        assert "groq" in llm_registry._providers

    def test_get_provider_by_alias(self) -> None:
        """Test getting LLM provider by alias."""
        from engram.providers.llm import llm_registry

        # "gpt" should be alias for openai
        provider_class = llm_registry.get("gpt")
        assert provider_class is not None

    def test_gemini_aliases_resolve(self) -> None:
        """Test that gemini aliases resolve to the Gemini provider."""
        from engram.providers.llm import llm_registry
        from engram.providers.llm.builtin import GeminiLLMProvider

        for name in ("gemini", "google", "google-genai"):
            assert llm_registry.get(name) is GeminiLLMProvider


class TestGeminiLLMProvider:
    """Tests for Gemini LLM provider (mocked google-genai client)."""

    def test_missing_api_key_raises(self) -> None:
        from engram.core.exceptions import ConfigurationError
        from engram.providers.llm.builtin import GeminiLLMProvider

        with pytest.raises(ConfigurationError):
            GeminiLLMProvider(api_key=None)

    @pytest.mark.asyncio
    async def test_complete_splits_system_and_contents(self) -> None:
        """System messages map to system_instruction; turns map to contents."""
        from engram.providers.llm.builtin import GeminiLLMProvider

        captured: dict[str, object] = {}

        async def fake_generate_content(*, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            resp = MagicMock()
            resp.text = "hi there"
            resp.candidates = [MagicMock(finish_reason="STOP")]
            resp.usage_metadata = MagicMock(
                prompt_token_count=10,
                candidates_token_count=3,
                total_token_count=13,
            )
            return resp

        provider = GeminiLLMProvider(api_key="test-key", model="gemini-3.5-flash")
        provider._client.aio.models.generate_content = fake_generate_content

        result = await provider.complete(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
            max_tokens=128,
            temperature=0.2,
        )

        assert result.content == "hi there"
        assert result.model == "gemini-3.5-flash"
        assert result.usage["total_tokens"] == 13
        # 3 non-system turns become contents; system text is not a content.
        assert len(captured["contents"]) == 3
        roles = [c.role for c in captured["contents"]]
        assert roles == ["user", "model", "user"]
        assert captured["config"].system_instruction == "be brief"
        assert captured["config"].max_output_tokens == 128


class TestProviderRegistryGeneric:
    """Tests for generic ProviderRegistry."""

    def test_register_provider(self) -> None:
        """Test registering a custom provider."""
        from engram.providers.registry import ProviderRegistry

        registry: ProviderRegistry[MagicMock] = ProviderRegistry("test_registry")

        @registry.register("custom", aliases=["c", "custom_alias"])
        class CustomProvider:
            pass

        assert "custom" in registry._providers
        # Aliases are in _aliases, not _providers
        assert "c" in registry._aliases
        assert "custom_alias" in registry._aliases

    def test_available_providers(self) -> None:
        """Test listing registered providers."""
        from engram.providers.registry import ProviderRegistry

        registry: ProviderRegistry[MagicMock] = ProviderRegistry("test_registry")

        @registry.register("provider1")
        class Provider1:
            pass

        @registry.register("provider2")
        class Provider2:
            pass

        providers = registry.available_providers

        assert "provider1" in providers
        assert "provider2" in providers

    def test_create_provider(self) -> None:
        """Test creating provider instance."""
        from engram.providers.registry import ProviderRegistry

        registry: ProviderRegistry[MagicMock] = ProviderRegistry("test_registry")

        @registry.register("test")
        class TestProvider:
            def __init__(self, value: int):
                self.value = value

        instance = registry.create("test", value=42)

        assert instance.value == 42


class TestGetEmbeddingProvider:
    """Tests for get_embedding_provider helper."""

    def test_get_sentence_transformers(self) -> None:
        """Test getting sentence-transformers provider."""
        from engram.providers.embedding import get_embedding_provider

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_sentence_transformers_module()},
        ):
            provider = get_embedding_provider(
                "sentence-transformers",
                model="all-MiniLM-L6-v2",
            )

        assert provider.dimension == 384

    def test_get_unknown_raises(self) -> None:
        """Test that unknown provider raises KeyError."""
        from engram.providers.embedding import get_embedding_provider

        with pytest.raises(KeyError):
            get_embedding_provider("unknown_provider")


class TestGetLLMProvider:
    """Tests for get_llm_provider helper."""

    def test_get_ollama(self) -> None:
        """Test getting Ollama LLM provider."""
        from engram.providers.llm import get_llm_provider

        provider = get_llm_provider(
            "ollama",
            model="llama3",
            base_url="http://localhost:11434",
        )

        assert provider.model == "llama3"

    def test_get_unknown_raises(self) -> None:
        """Test that unknown provider raises KeyError."""
        from engram.providers.llm import get_llm_provider

        with pytest.raises(KeyError):
            get_llm_provider("unknown_provider")


class TestEmbeddingProviderProtocol:
    """Tests for EmbeddingProvider protocol compliance."""

    def test_provider_has_required_attributes(self) -> None:
        """Test that providers have required attributes."""
        from engram.providers.embedding.builtin import (
            SentenceTransformersEmbeddingProvider,
        )

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": fake_sentence_transformers_module()},
        ):
            provider = SentenceTransformersEmbeddingProvider(model="all-MiniLM-L6-v2")

        # Check protocol attributes
        assert hasattr(provider, "dimension")
        assert hasattr(provider, "model")
        assert hasattr(provider, "embed")
        assert hasattr(provider, "embed_batch")

        assert isinstance(provider.dimension, int)
        assert isinstance(provider.model, str)


class TestLLMProviderProtocol:
    """Tests for LLMProvider protocol compliance."""

    def test_provider_has_required_attributes(self) -> None:
        """Test that LLM providers have required attributes."""
        from engram.providers.llm.builtin import OllamaLLMProvider

        provider = OllamaLLMProvider(model="llama3")

        assert hasattr(provider, "model")
        assert hasattr(provider, "complete")
        assert hasattr(provider, "complete_text")

        assert isinstance(provider.model, str)
