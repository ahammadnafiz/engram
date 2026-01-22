"""OpenAI embedding provider for Engram.

This module provides an embedding provider using OpenAI's API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from engram.core._types import Vector, VectorDimension
from engram.core.exceptions import EmbeddingProviderError

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Embedding provider using OpenAI's API.

    This provider uses OpenAI's text embedding models for generating
    vector representations of text.

    Example:
        provider = OpenAIProvider(
            api_key="sk-...",
            model="text-embedding-3-small",
            dimension=1536,
        )
        
        vector = await provider.embed("Hello, world!")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: VectorDimension = 1536,
        base_url: str | None = None,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model name (e.g., "text-embedding-3-small").
            dimension: Embedding dimension.
            base_url: Optional custom API base URL.
        """
        try:
            import openai as openai_module

            self._openai = openai_module
        except ImportError as e:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install engram[openai]"
            ) from e

        self._client = self._openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model
        self._dimension = dimension

        logger.info(f"Initialized OpenAI provider with model {model}")

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> Vector:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A vector representing the embedding.

        Raises:
            EmbeddingProviderError: If API call fails.
        """
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
                dimensions=self._dimension,
            )
            return list(response.data[0].embedding)

        except self._openai.APIError as e:
            raise EmbeddingProviderError(
                f"OpenAI API error: {e}",
                model=self._model,
            ) from e
        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to generate embedding: {e}",
                model=self._model,
            ) from e

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        """Generate embeddings for multiple texts.

        OpenAI's API supports batch embedding, which is more efficient
        than individual calls.

        Args:
            texts: List of texts to embed.

        Returns:
            List of vectors in the same order as input texts.

        Raises:
            EmbeddingProviderError: If API call fails.
        """
        if not texts:
            return []

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimension,
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [list(item.embedding) for item in sorted_data]

        except self._openai.APIError as e:
            raise EmbeddingProviderError(
                f"OpenAI API error: {e}",
                model=self._model,
                batch_size=len(texts),
            ) from e
        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to generate batch embeddings: {e}",
                model=self._model,
                batch_size=len(texts),
            ) from e
