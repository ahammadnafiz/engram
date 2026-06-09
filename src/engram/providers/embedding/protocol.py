"""Embedding provider protocol for Engram.

This module defines the abstract base class for all embedding providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement this interface. The provider
    is responsible for converting text to vector embeddings.

    Attributes:
        dimension: The dimension of the embedding vectors.
        model: The model name being used.

    Example:
        class MyEmbeddingProvider(EmbeddingProvider):
            def __init__(self, model: str = "my-model"):
                self._model = model
                self._dimension = 768

            @property
            def dimension(self) -> int:
                return self._dimension

            @property
            def model(self) -> str:
                return self._model

            async def embed(self, text: str) -> list[float]:
                # Your implementation
                return [0.0] * self._dimension

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [await self.embed(t) for t in texts]
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            The number of dimensions in the embedding vectors.
        """
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Get the model name.

        Returns:
            The model identifier being used.
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingProviderError: If embedding generation fails.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        This should be more efficient than calling embed() multiple times
        when the provider supports batch operations.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input texts.

        Raises:
            EmbeddingProviderError: If embedding generation fails.
        """
        ...

    def get_info(self) -> dict[str, Any]:
        """Get provider information.

        Returns:
            Dictionary with provider details (model, dimension, etc.)
        """
        return {
            "provider": self.__class__.__name__,
            "model": self.model,
            "dimension": self.dimension,
        }
