"""Embedding service protocols for Engram.

This module defines the protocol (interface) for embedding providers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from engram.core._types import Vector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Embedding providers must implement both single and batch embedding
    methods. This allows for different backends like OpenAI, Sentence
    Transformers, or custom models.

    Example:
        class MyEmbeddingProvider:
            async def embed(self, text: str) -> Vector:
                # Your implementation
                pass

            async def embed_batch(self, texts: list[str]) -> list[Vector]:
                # Your batch implementation
                pass
    """

    async def embed(self, text: str) -> Vector:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A vector (list of floats) representing the embedding.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        """Generate embeddings for multiple texts.

        This should be more efficient than calling embed() multiple times.

        Args:
            texts: List of texts to embed.

        Returns:
            List of vectors in the same order as input texts.
        """
        ...

    @property
    def dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            The number of dimensions in the embedding vectors.
        """
        ...
