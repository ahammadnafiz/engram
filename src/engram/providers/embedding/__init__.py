"""Embedding provider system for Engram.

This module provides a pluggable embedding provider architecture.

Built-in providers:
- openai: OpenAI text-embedding-3-small/large
- sentence-transformers: Local Sentence Transformers models
- cohere: Cohere embed models
- ollama: Ollama local embeddings

Example:
    # Use built-in provider
    from engram.providers import get_embedding_provider

    provider = get_embedding_provider(
        "openai",
        api_key="sk-...",
        model="text-embedding-3-small",
    )

    # Register custom provider
    from engram.providers import embedding_registry, EmbeddingProvider

    @embedding_registry.register("my-provider")
    class MyProvider(EmbeddingProvider):
        def __init__(self, **kwargs):
            self._dimension = 768

        @property
        def dimension(self) -> int:
            return self._dimension

        async def embed(self, text: str) -> list[float]:
            return [0.0] * self._dimension

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self._dimension for _ in texts]
"""

# Import built-in providers to register them
from engram.providers.embedding import builtin  # noqa: F401
from engram.providers.embedding.protocol import EmbeddingProvider
from engram.providers.embedding.registry import (
    embedding_registry,
    get_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "embedding_registry",
    "get_embedding_provider",
]
