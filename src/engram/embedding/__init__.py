"""Embedding module for Engram.

This module provides embedding services and providers for generating
vector representations of text.

The provider system supports pluggable backends. See engram.providers.embedding
for the full provider registry and built-in providers.
"""

from engram.embedding.service import EmbeddingService

# Re-export from providers for backward compatibility
from engram.providers.embedding import (
    EmbeddingProvider,
    embedding_registry,
    get_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingService",
    "embedding_registry",
    "get_embedding_provider",
]
