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

from engram.providers.registry import ProviderRegistry
from engram.providers.embedding import (
    EmbeddingProvider,
    embedding_registry,
    get_embedding_provider,
)
from engram.providers.llm import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    llm_registry,
    get_llm_provider,
)

__all__ = [
    # Registry
    "ProviderRegistry",
    # Embedding
    "EmbeddingProvider",
    "embedding_registry",
    "get_embedding_provider",
    # LLM
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "llm_registry",
    "get_llm_provider",
]

