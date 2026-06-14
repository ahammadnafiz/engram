"""Cross-encoder reranking for search results.

A cross-encoder reads the query and each candidate fact together and scores
their relevance directly, which is more accurate than the bi-encoder cosine
similarity used for candidate generation. The standard retrieval pattern is:
overfetch candidates with the fast hybrid search, then let the cross-encoder
pick the final top-k ordering.

The model runs locally on CPU via sentence-transformers and is loaded lazily
on first use. sentence-transformers is an optional dependency; installs
without it can still use every other Engram feature.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from engram.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from engram.memory.models import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-orders search results with a local cross-encoder model.

    Example:
        reranker = CrossEncoderReranker()
        results = await reranker.rerank("user's dog name", candidates, top_k=10)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        backend: str = "torch",
    ) -> None:
        self._model_name = model_name
        self._backend = backend
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self) -> Any:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ConfigurationError(
                "Reranking requires sentence-transformers. "
                "Install it with: pip install 'engram[rerank]'"
            ) from e
        logger.info(
            f"Loading cross-encoder reranker: {self._model_name} "
            f"(backend={self._backend})"
        )
        try:
            return CrossEncoder(self._model_name, backend=self._backend)
        except (ImportError, ModuleNotFoundError) as e:
            raise ConfigurationError(
                f"Reranker backend {self._backend!r} is missing dependencies. "
                "For 'onnx', install: pip install 'engram[rerank-onnx]'"
            ) from e

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Return the ``top_k`` results re-ordered by cross-encoder relevance.

        The original hybrid scores on each result are preserved; only the
        ordering (and the cutoff) comes from the cross-encoder.
        """
        if len(results) <= 1:
            return results[:top_k]

        model = await self._ensure_model()
        pairs = [(query, result.memory.content) for result in results]
        scores = await asyncio.to_thread(model.predict, pairs)
        order = sorted(range(len(results)), key=lambda i: -float(scores[i]))
        return [results[i] for i in order[:top_k]]
