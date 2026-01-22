"""Engram - AI Memory Library for LLM Applications.

Engram provides production-ready memory management for AI agents using
PostgreSQL + pgvector for converged storage with hybrid search.

Provider System:
    Engram uses a pluggable provider architecture for embeddings and LLMs.
    
    Built-in embedding providers:
    - openai: OpenAI text-embedding models
    - sentence-transformers: Local Sentence Transformers
    - cohere: Cohere embed models
    - ollama: Ollama local embeddings
    - huggingface: HuggingFace Inference API
    
    Built-in LLM providers:
    - openai: OpenAI GPT models
    - anthropic: Anthropic Claude models
    - ollama: Ollama local LLMs
    - groq: Groq fast inference
    - litellm: Universal LLM interface

Example:
    import asyncio
    from engram import Engram

    async def main():
        async with Engram() as engram:
            # Add a memory
            memory = await engram.add(
                content="User prefers dark mode",
                agent_id="my_agent",
            )
            
            # Reinforce when memory is useful
            await engram.reinforce(memory.memory_id)
            
            # Search memories
            results = await engram.search(
                query="user interface preferences",
                agent_id="my_agent",
            )
            
            for result in results:
                print(f"{result.score:.2f}: {result.memory.content}")

    asyncio.run(main())
"""

from engram._version import __version__
from engram.client import Engram
from engram.core import (
    # Configuration
    EngramSettings,
    get_settings,
    # Exceptions
    ConfigurationError,
    ConnectionError,
    ConnectionPoolExhaustedError,
    EngramError,
    MemoryNotFoundError,
    QueryError,
    SessionNotFoundError,
    StorageError,
    ValidationError,
    # Types
    AgentId,
    MemoryId,
    Metadata,
    RelationType,
    SearchMode,
    SessionId,
    TraversalDirection,
    UserId,
    Vector,
)
from engram.embedding import EmbeddingService
from engram.graph import MemoryRelation, TraversalResult
from engram.memory import Memory, MemoryCreate, SearchQuery, SearchResult
from engram.session import Session

# Provider system
from engram.providers import (
    EmbeddingProvider,
    LLMProvider,
    LLMMessage,
    LLMResponse,
    embedding_registry,
    llm_registry,
    get_embedding_provider,
    get_llm_provider,
)
from engram.llm import LLMService

__all__ = [
    # Main client
    "Engram",
    # Version
    "__version__",
    # Configuration
    "EngramSettings",
    "get_settings",
    # Models
    "Memory",
    "MemoryCreate",
    "MemoryRelation",
    "SearchQuery",
    "SearchResult",
    "Session",
    "TraversalResult",
    # Types
    "AgentId",
    "MemoryId",
    "Metadata",
    "RelationType",
    "SearchMode",
    "SessionId",
    "TraversalDirection",
    "UserId",
    "Vector",
    # Embedding & LLM Services
    "EmbeddingService",
    "LLMService",
    # Provider System
    "EmbeddingProvider",
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "embedding_registry",
    "llm_registry",
    "get_embedding_provider",
    "get_llm_provider",
    # Exceptions
    "ConfigurationError",
    "ConnectionError",
    "ConnectionPoolExhaustedError",
    "EngramError",
    "MemoryNotFoundError",
    "QueryError",
    "SessionNotFoundError",
    "StorageError",
    "ValidationError",
]
