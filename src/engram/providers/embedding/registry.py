"""Embedding provider registry for Engram.

This module provides the global embedding provider registry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from engram.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from engram.providers.embedding.protocol import EmbeddingProvider

logger = logging.getLogger(__name__)

# Global embedding provider registry
embedding_registry: ProviderRegistry[EmbeddingProvider] = ProviderRegistry("embedding")


def get_embedding_provider(
    provider_name: str,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Create an embedding provider instance.

    This is the main factory function for creating embedding providers.

    Args:
        provider_name: Name of the provider (e.g., "openai", "sentence-transformers").
        **kwargs: Provider-specific configuration:

            For "openai":
                - api_key: OpenAI API key (required)
                - model: Model name (default: "text-embedding-3-small")
                - dimension: Vector dimension (default: 1536)
                - base_url: Custom API base URL (optional)

            For "sentence-transformers":
                - model: Model name (default: "all-MiniLM-L6-v2")
                - device: Device to use (default: auto-detect)

            For "cohere":
                - api_key: Cohere API key (required)
                - model: Model name (default: "embed-english-v3.0")

            For "ollama":
                - model: Model name (required)
                - base_url: Ollama server URL (default: "http://localhost:11434")

    Returns:
        An initialized embedding provider.

    Raises:
        KeyError: If provider is not registered.
        ConfigurationError: If required configuration is missing.

    Example:
        # OpenAI embeddings
        provider = get_embedding_provider(
            "openai",
            api_key="sk-...",
            model="text-embedding-3-small",
        )

        # Local embeddings
        provider = get_embedding_provider(
            "sentence-transformers",
            model="all-MiniLM-L6-v2",
        )

        # Custom provider
        provider = get_embedding_provider("my-custom-provider", **my_config)
    """
    logger.info(f"Creating embedding provider: {provider_name}")
    return embedding_registry.create(provider_name, **kwargs)


def list_embedding_providers() -> list[str]:
    """List all registered embedding providers.

    Returns:
        List of provider names.
    """
    return embedding_registry.available_providers
