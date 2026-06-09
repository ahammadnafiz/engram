"""Embedding service for Engram.

This module provides the main embedding service with caching and batching.
It uses the provider registry system for flexible provider selection.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from engram.core.config import EngramSettings, get_settings
from engram.core.exceptions import ConfigurationError, EmbeddingError
from engram.providers.embedding import EmbeddingProvider, get_embedding_provider

if TYPE_CHECKING:
    from engram.core._types import Vector

logger = logging.getLogger(__name__)


class EmbeddingService:
    """High-level embedding service with caching and batching.

    This service wraps embedding providers and adds:
    - LRU caching for repeated content
    - Automatic batching for efficiency
    - Provider abstraction via the registry system

    Example:
        # Create from settings (uses ENGRAM_EMBEDDING_PROVIDER env var)
        service = EmbeddingService.from_settings()

        # Or with explicit provider name
        service = EmbeddingService.from_provider("openai", api_key="sk-...")

        # Or with explicit provider instance
        service = EmbeddingService(provider=my_provider)

        # Embed text
        vector = await service.embed("Hello, world!")

        # Batch embed
        vectors = await service.embed_batch(["Hello", "World"])
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_size: int = 1000,
        batch_size: int = 100,
    ) -> None:
        """Initialize the embedding service.

        Args:
            provider: The embedding provider instance to use.
            cache_size: LRU cache size for embeddings (0 to disable).
            batch_size: Maximum batch size for batch operations.
        """
        self._provider = provider
        self._batch_size = batch_size
        self._cache_size = cache_size
        # Use OrderedDict for proper LRU cache behavior
        self._cache: OrderedDict[str, Vector] = OrderedDict()

        logger.info(
            f"Initialized EmbeddingService with {provider.__class__.__name__} "
            f"(model={provider.model}, dim={provider.dimension}), "
            f"cache_size={cache_size}, batch_size={batch_size}"
        )

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._provider.dimension

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._provider.model

    @property
    def provider(self) -> EmbeddingProvider:
        """Get the underlying provider."""
        return self._provider

    @classmethod
    def from_provider(
        cls,
        provider_name: str,
        cache_size: int = 1000,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create an EmbeddingService with a specific provider.

        Args:
            provider_name: Name of the embedding provider.
            cache_size: LRU cache size for embeddings.
            batch_size: Maximum batch size.
            **kwargs: Provider-specific configuration.

        Returns:
            Configured EmbeddingService.

        Example:
            # OpenAI
            service = EmbeddingService.from_provider(
                "openai",
                api_key="sk-...",
                model="text-embedding-3-small",
            )

            # Local
            service = EmbeddingService.from_provider(
                "sentence-transformers",
                model="all-MiniLM-L6-v2",
            )
        """
        provider = get_embedding_provider(provider_name, **kwargs)
        return cls(provider=provider, cache_size=cache_size, batch_size=batch_size)

    @classmethod
    def from_settings(
        cls,
        settings: EngramSettings | None = None,
    ) -> EmbeddingService:
        """Create an EmbeddingService from settings.

        Uses the provider registry to create the appropriate provider
        based on the ENGRAM_EMBEDDING_PROVIDER setting.

        Args:
            settings: Engram settings. If None, loads from environment.

        Returns:
            Configured EmbeddingService.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        settings = settings or get_settings()

        provider_name = settings.embedding_provider
        provider_kwargs = settings.get_embedding_provider_kwargs()

        try:
            provider = get_embedding_provider(provider_name, **provider_kwargs)
        except KeyError as e:
            raise ConfigurationError(str(e)) from e
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create embedding provider '{provider_name}': {e}"
            ) from e

        return cls(
            provider=provider,
            cache_size=settings.embedding_cache_size,
            batch_size=settings.embedding_batch_size,
        )

    def _compute_cache_key(self, text: str) -> str:
        """Compute cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def embed(self, text: str) -> Vector:
        """Generate embedding for text with LRU caching.

        Args:
            text: The text to embed.

        Returns:
            A vector representing the embedding.

        Raises:
            EmbeddingError: If embedding fails.
        """
        cache_key = self._compute_cache_key(text) if self._cache_size > 0 else ""

        # Check cache (with LRU move-to-end on access)
        if self._cache_size > 0 and cache_key in self._cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            # Move to end for LRU behavior
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Generate embedding
        vector = await self._provider.embed(text)

        # Store in cache with LRU eviction
        if self._cache_size > 0:
            # Evict least recently used (first item) if cache is full
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = vector

        return vector

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        """Generate embeddings for multiple texts with batching and LRU caching.

        This method:
        1. Checks cache for already-embedded texts (with LRU update)
        2. Batches remaining texts into chunks
        3. Caches new results

        Args:
            texts: List of texts to embed.

        Returns:
            List of vectors in the same order as input texts.
            Guaranteed to have the same length as input.

        Raises:
            EmbeddingError: If embedding fails or results are incomplete.
        """
        if not texts:
            return []

        n_texts = len(texts)
        results: list[Vector | None] = [None] * n_texts
        texts_to_embed: list[tuple[int, str]] = []

        # Check cache for each text (with LRU update on hit)
        if self._cache_size > 0:
            for i, text in enumerate(texts):
                cache_key = self._compute_cache_key(text)
                if cache_key in self._cache:
                    # Move to end for LRU behavior
                    self._cache.move_to_end(cache_key)
                    results[i] = self._cache[cache_key]
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))

        # Batch embed remaining texts
        if texts_to_embed:
            # Process in batches
            for batch_start in range(0, len(texts_to_embed), self._batch_size):
                batch = texts_to_embed[batch_start : batch_start + self._batch_size]
                batch_texts = [text for _, text in batch]

                batch_vectors = await self._provider.embed_batch(batch_texts)

                # Validate batch results length
                if len(batch_vectors) != len(batch_texts):
                    raise EmbeddingError(
                        f"Provider returned {len(batch_vectors)} embeddings "
                        f"for {len(batch_texts)} texts",
                        expected=len(batch_texts),
                        actual=len(batch_vectors),
                    )

                # Store results and update cache with LRU eviction
                for (idx, text), vector in zip(batch, batch_vectors, strict=True):
                    results[idx] = vector

                    if self._cache_size > 0:
                        cache_key = self._compute_cache_key(text)
                        # Evict LRU if full
                        if len(self._cache) >= self._cache_size:
                            self._cache.popitem(last=False)
                        self._cache[cache_key] = vector

        # Validate all results are filled
        missing_indices = [i for i, v in enumerate(results) if v is None]
        if missing_indices:
            raise EmbeddingError(
                f"Failed to generate embeddings for {len(missing_indices)} texts",
                missing_indices=missing_indices,
            )

        # Type assertion: all elements are now Vector (not None)
        return results  # type: ignore[return-value]

    @property
    def cache_info(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache size and max size.
        """
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
        }
