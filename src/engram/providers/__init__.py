"""Provider system for Engram.

This module provides a pluggable provider architecture for:
- Embedding providers (OpenAI, Sentence Transformers, Cohere, local, etc.)
- LLM providers (OpenAI, Anthropic, Ollama, local, etc.)

Example:
    # Register a custom embedding provider
    from engram.providers import embedding_registry

    @embedding_registry.register("my-provider")
    class MyEmbeddingProvider:
        async def embed(self, text: str) -> list[float]:
            ...

    # Use it
    engram = Engram(embedding_provider="my-provider")
"""

from engram.providers.embedding import (
    EmbeddingProvider,
    embedding_registry,
    get_embedding_provider,
)
from engram.providers.llm import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    get_llm_provider,
    llm_registry,
)
from engram.providers.registry import ProviderRegistry

__all__ = [
    # Embedding
    "EmbeddingProvider",
    "LLMMessage",
    # LLM
    "LLMProvider",
    "LLMResponse",
    # Registry
    "ProviderRegistry",
    "embedding_registry",
    "get_embedding_provider",
    "get_llm_provider",
    "llm_registry",
]
