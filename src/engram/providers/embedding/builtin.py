"""Built-in embedding providers for Engram.

This module registers the built-in embedding providers:
- openai: OpenAI text-embedding models
- sentence-transformers: Local Sentence Transformers
- cohere: Cohere embed models
- ollama: Ollama local embeddings
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

from engram.providers.embedding.protocol import EmbeddingProvider
from engram.providers.embedding.registry import embedding_registry
from engram.core.exceptions import EmbeddingProviderError, ConfigurationError

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =============================================================================
# OpenAI Provider
# =============================================================================

@embedding_registry.register("openai", aliases=["openai-embedding"])
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API.
    
    Supports text-embedding-3-small, text-embedding-3-large, and legacy ada models.
    
    Args:
        api_key: OpenAI API key (required).
        model: Model name (default: "text-embedding-3-small").
        dimension: Vector dimension (default: model's native dimension).
        base_url: Custom API base URL (optional, for Azure or proxies).
        
    Example:
        provider = OpenAIEmbeddingProvider(
            api_key="sk-...",
            model="text-embedding-3-small",
            dimension=1536,
        )
    """
    
    # Default dimensions for OpenAI models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimension: int | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key is required. Pass api_key parameter or set "
                "ENGRAM_OPENAI_API_KEY environment variable."
            )
        
        try:
            import openai as openai_module
            self._openai = openai_module
        except ImportError as e:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            ) from e
        
        # Only pass base_url if specified (None would override defaults)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client: AsyncOpenAI = self._openai.AsyncOpenAI(**client_kwargs)
        self._model = model
        self._dimension = dimension or self.MODEL_DIMENSIONS.get(model, 1536)
        
        logger.info(f"Initialized OpenAI embedding provider: {model} ({self._dimension}d)")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model(self) -> str:
        return self._model
    
    async def embed(self, text: str) -> list[float]:
        try:
            # Only pass dimensions for models that support it
            kwargs: dict[str, Any] = {"model": self._model, "input": text}
            if "text-embedding-3" in self._model:
                kwargs["dimensions"] = self._dimension
            
            response = await self._client.embeddings.create(**kwargs)
            return list(response.data[0].embedding)
        except self._openai.APIConnectionError as e:
            raise EmbeddingProviderError(f"OpenAI connection error: {e}", model=self._model) from e
        except self._openai.APIError as e:
            raise EmbeddingProviderError(f"OpenAI API error: {e}", model=self._model) from e
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to embed: {e}", model=self._model) from e
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        try:
            # Only pass dimensions for models that support it
            kwargs: dict[str, Any] = {"model": self._model, "input": texts}
            if "text-embedding-3" in self._model:
                kwargs["dimensions"] = self._dimension
            
            response = await self._client.embeddings.create(**kwargs)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [list(item.embedding) for item in sorted_data]
        except self._openai.APIConnectionError as e:
            raise EmbeddingProviderError(f"OpenAI connection error: {e}", model=self._model) from e
        except self._openai.APIError as e:
            raise EmbeddingProviderError(f"OpenAI API error: {e}", model=self._model) from e
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to embed batch: {e}", model=self._model) from e


# =============================================================================
# Sentence Transformers Provider
# =============================================================================

@embedding_registry.register("sentence-transformers", aliases=["st", "local", "sbert"])
class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using local Sentence Transformers models.
    
    Runs entirely locally - no API calls, no cost.
    
    Args:
        model: Model name (default: "all-MiniLM-L6-v2").
        device: Device to use ("cpu", "cuda", "mps", etc.). Auto-detected if None.
        
    Example:
        provider = SentenceTransformersEmbeddingProvider(
            model="all-MiniLM-L6-v2",
        )
    """
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            ) from e
        
        logger.info(f"Loading Sentence Transformers model: {model}")
        self._st_model: SentenceTransformer = SentenceTransformer(model, device=device)
        self._model = model
        self._dimension = self._st_model.get_sentence_embedding_dimension()
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        
        logger.info(f"Loaded {model} ({self._dimension}d) on {self._st_model.device}")
    
    @property
    def dimension(self) -> int:
        return self._dimension  # type: ignore[return-value]
    
    @property
    def model(self) -> str:
        return self._model
    
    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._st_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]
    
    async def embed(self, text: str) -> list[float]:
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                self._executor,
                partial(self._encode_sync, [text]),
            )
            return results[0]
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to embed: {e}", model=self._model) from e
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(self._encode_sync, texts),
            )
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to embed batch: {e}", model=self._model) from e

    def close(self) -> None:
        """Clean up the ThreadPoolExecutor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self) -> None:
        """Ensure executor is cleaned up on garbage collection."""
        self.close()


# =============================================================================
# Cohere Provider
# =============================================================================

@embedding_registry.register("cohere")
class CohereEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Cohere's API.
    
    Args:
        api_key: Cohere API key (required).
        model: Model name (default: "embed-english-v3.0").
        input_type: Input type for embeddings (default: "search_document").
        
    Example:
        provider = CohereEmbeddingProvider(
            api_key="...",
            model="embed-english-v3.0",
        )
    """
    
    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-english-light-v2.0": 1024,
        "embed-multilingual-v2.0": 768,
    }
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "Cohere API key is required. Pass api_key parameter or set "
                "ENGRAM_COHERE_API_KEY environment variable."
            )
        
        try:
            import cohere
            self._cohere = cohere
        except ImportError as e:
            raise ImportError(
                "Cohere package not installed. Install with: pip install cohere"
            ) from e
        
        self._client = self._cohere.AsyncClient(api_key=api_key)
        self._model = model
        self._input_type = input_type
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1024)
        
        logger.info(f"Initialized Cohere embedding provider: {model} ({self._dimension}d)")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model(self) -> str:
        return self._model
    
    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.embed(
                texts=[text],
                model=self._model,
                input_type=self._input_type,
            )
            return list(response.embeddings[0])
        except Exception as e:
            raise EmbeddingProviderError(f"Cohere API error: {e}", model=self._model) from e
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        try:
            response = await self._client.embed(
                texts=texts,
                model=self._model,
                input_type=self._input_type,
            )
            return [list(emb) for emb in response.embeddings]
        except Exception as e:
            raise EmbeddingProviderError(f"Cohere API error: {e}", model=self._model) from e


# =============================================================================
# Ollama Provider
# =============================================================================

@embedding_registry.register("ollama", aliases=["ollama-embedding"])
class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Ollama's local server.
    
    Ollama runs LLMs and embedding models locally.
    
    Args:
        model: Model name (required, e.g., "nomic-embed-text", "mxbai-embed-large").
        base_url: Ollama server URL (default: "http://localhost:11434").
        dimension: Vector dimension (auto-detected if not specified).
        
    Example:
        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
        )
    """
    
    # Known model dimensions (others will be auto-detected)
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import httpx
            self._httpx = httpx
        except ImportError as e:
            raise ImportError(
                "httpx package not installed. Install with: pip install httpx"
            ) from e
        
        self._client = self._httpx.AsyncClient(base_url=base_url, timeout=60.0)
        self._model = model
        self._base_url = base_url
        self._dimension = dimension or self.MODEL_DIMENSIONS.get(model)
        self._dimension_detected = False
        
        logger.info(f"Initialized Ollama embedding provider: {model} at {base_url}")
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise ConfigurationError(
                f"Dimension not known for model {self._model}. "
                "Either specify dimension explicitly or call embed() once to auto-detect."
            )
        return self._dimension
    
    @property
    def model(self) -> str:
        return self._model
    
    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["embedding"]
            
            # Auto-detect dimension on first call
            if self._dimension is None:
                self._dimension = len(embedding)
                logger.info(f"Auto-detected dimension for {self._model}: {self._dimension}")
            
            return embedding
        except self._httpx.HTTPError as e:
            raise EmbeddingProviderError(f"Ollama API error: {e}", model=self._model) from e
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to embed: {e}", model=self._model) from e
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # Ollama doesn't support batch embedding natively
        # Use asyncio.gather for parallel execution
        import asyncio
        return list(await asyncio.gather(*[self.embed(text) for text in texts]))

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# =============================================================================
# HuggingFace Inference API Provider
# =============================================================================

@embedding_registry.register("huggingface", aliases=["hf", "huggingface-inference"])
class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using HuggingFace Inference API.
    
    Uses HuggingFace's hosted inference API for embedding models.
    
    Args:
        api_key: HuggingFace API token (required).
        model: Model name (default: "sentence-transformers/all-MiniLM-L6-v2").
        dimension: Vector dimension (auto-detected if not specified).
        
    Example:
        provider = HuggingFaceEmbeddingProvider(
            api_key="hf_...",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
    """
    
    MODEL_DIMENSIONS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "HuggingFace API key is required. Pass api_key parameter or set "
                "ENGRAM_HF_API_KEY environment variable."
            )
        
        try:
            import httpx
            self._httpx = httpx
        except ImportError as e:
            raise ImportError(
                "httpx package not installed. Install with: pip install httpx"
            ) from e
        
        self._api_key = api_key
        self._model = model
        self._api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        self._client = self._httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._dimension = dimension or self.MODEL_DIMENSIONS.get(model)
        
        logger.info(f"Initialized HuggingFace embedding provider: {model}")
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise ConfigurationError(
                f"Dimension not known for model {self._model}. "
                "Specify dimension explicitly or call embed() once to auto-detect."
            )
        return self._dimension
    
    @property
    def model(self) -> str:
        return self._model
    
    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.post(
                self._api_url,
                json={"inputs": text, "options": {"wait_for_model": True}},
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle nested response format
            embedding = data[0] if isinstance(data[0], list) else data
            
            # Auto-detect dimension
            if self._dimension is None:
                self._dimension = len(embedding)
                logger.info(f"Auto-detected dimension for {self._model}: {self._dimension}")
            
            return embedding
        except self._httpx.HTTPError as e:
            raise EmbeddingProviderError(f"HuggingFace API error: {e}", model=self._model) from e
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to embed: {e}", model=self._model) from e
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        try:
            response = await self._client.post(
                self._api_url,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response - handle nested format [[embedding], [embedding], ...]
            embeddings: list[list[float]] = []
            for item in data:
                if isinstance(item, list) and item and isinstance(item[0], list):
                    # Nested format: [[embedding]]
                    embeddings.append(item[0])
                elif isinstance(item, list):
                    # Direct format: [embedding]
                    embeddings.append(item)
                else:
                    raise EmbeddingProviderError(
                        f"Unexpected response format from HuggingFace: {type(item)}",
                        model=self._model,
                    )
            
            # Validate we got the right number of embeddings
            if len(embeddings) != len(texts):
                raise EmbeddingProviderError(
                    f"HuggingFace returned {len(embeddings)} embeddings for {len(texts)} texts",
                    model=self._model,
                    expected=len(texts),
                    actual=len(embeddings),
                )
            
            # Auto-detect dimension from first embedding if not set
            if self._dimension is None and embeddings:
                self._dimension = len(embeddings[0])
                logger.info(f"Auto-detected dimension for {self._model}: {self._dimension}")
            
            return embeddings
        except self._httpx.HTTPError as e:
            raise EmbeddingProviderError(f"HuggingFace API error: {e}", model=self._model) from e
        except EmbeddingProviderError:
            raise
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to embed batch: {e}", model=self._model) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

