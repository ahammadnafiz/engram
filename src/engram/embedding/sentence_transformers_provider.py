"""Sentence Transformers embedding provider for Engram.

This module provides an embedding provider using local Sentence Transformers models.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING

from engram.core._types import Vector
from engram.core.exceptions import EmbeddingProviderError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformersProvider:
    """Embedding provider using local Sentence Transformers.

    This provider runs embedding models locally, which can be faster
    and more cost-effective for high-volume use cases.

    Example:
        provider = SentenceTransformersProvider(
            model_name="all-MiniLM-L6-v2",
        )
        
        vector = await provider.embed("Hello, world!")
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        """Initialize the Sentence Transformers provider.

        Args:
            model_name: Name of the model to use.
            device: Device to run on ("cpu", "cuda", etc.). Auto-detected if None.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install engram[sentence-transformers]"
            ) from e

        logger.info(f"Loading Sentence Transformers model: {model_name}")
        self._model: SentenceTransformer = SentenceTransformer(
            model_name,
            device=device,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info(
            f"Loaded model {model_name} with dimension {self._dimension} "
            f"on device {self._model.device}"
        )

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension  # type: ignore[return-value]

    def _encode_sync(self, texts: list[str]) -> list[Vector]:
        """Synchronous encoding (runs in thread pool)."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]

    async def embed(self, text: str) -> Vector:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A vector representing the embedding.

        Raises:
            EmbeddingProviderError: If encoding fails.
        """
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                partial(self._encode_sync, [text]),
            )
            return results[0]

        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to generate embedding: {e}",
                model=str(self._model),
            ) from e

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of vectors in the same order as input texts.

        Raises:
            EmbeddingProviderError: If encoding fails.
        """
        if not texts:
            return []

        try:
            import asyncio

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(self._encode_sync, texts),
            )

        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to generate batch embeddings: {e}",
                model=str(self._model),
                batch_size=len(texts),
            ) from e
