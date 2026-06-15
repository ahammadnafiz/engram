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
    # Types
    AgentId,
    # Exceptions
    ConfigurationError,
    ConnectionError,
    ConnectionPoolExhaustedError,
    EngramError,
    # Configuration
    EngramSettings,
    MemoryId,
    MemoryNotFoundError,
    MemoryType,
    Metadata,
    QueryError,
    RelationType,
    SearchMode,
    SessionId,
    SessionNotFoundError,
    StorageError,
    TraversalDirection,
    UserId,
    ValidationError,
    Vector,
    get_settings,
)
from engram.embedding import EmbeddingService
from engram.graph import MemoryRelation, TraversalResult
from engram.llm import LLMService
from engram.memory import (
    Memory,
    MemoryCreate,
    MemoryExplanation,
    MemoryHistoryEvent,
    MemoryLineage,
    RecallTrace,
    SearchQuery,
    SearchResult,
)
from engram.policy import (
    CODING_AGENT_MEMORY_POLICY,
    DEFAULT_MEMORY_POLICY,
    LEGAL_MEMORY_POLICY,
    MemoryPolicy,
    SlotRule,
    TypeRule,
    get_memory_policy,
)

# Provider system
from engram.providers import (
    EmbeddingProvider,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    embedding_registry,
    get_embedding_provider,
    get_llm_provider,
    llm_registry,
)
from engram.session import Session
from engram.task import (
    AgentEvent,
    ContextBuildOptions,
    ContextBuildResult,
    EventCreate,
    LongInputChunk,
    LongInputContextResult,
    LongInputIngestionReport,
    MemoryJob,
    TaskCheckpoint,
    TaskRun,
)

__all__ = [
    "CODING_AGENT_MEMORY_POLICY",
    "DEFAULT_MEMORY_POLICY",
    "LEGAL_MEMORY_POLICY",
    "AgentEvent",
    # Types
    "AgentId",
    # Exceptions
    "ConfigurationError",
    "ConnectionError",
    "ConnectionPoolExhaustedError",
    "ContextBuildOptions",
    "ContextBuildResult",
    # Provider System
    "EmbeddingProvider",
    # Embedding & LLM Services
    "EmbeddingService",
    # Main client
    "Engram",
    "EngramError",
    # Configuration
    "EngramSettings",
    "EventCreate",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMService",
    "LongInputChunk",
    "LongInputContextResult",
    "LongInputIngestionReport",
    # Models
    "Memory",
    "MemoryCreate",
    "MemoryExplanation",
    "MemoryHistoryEvent",
    "MemoryId",
    "MemoryJob",
    "MemoryLineage",
    "MemoryNotFoundError",
    "MemoryPolicy",
    "MemoryRelation",
    "MemoryType",
    "Metadata",
    "QueryError",
    "RecallTrace",
    "RelationType",
    "SearchMode",
    "SearchQuery",
    "SearchResult",
    "Session",
    "SessionId",
    "SessionNotFoundError",
    "SlotRule",
    "StorageError",
    "TaskCheckpoint",
    "TaskRun",
    "TraversalDirection",
    "TraversalResult",
    "TypeRule",
    "UserId",
    "ValidationError",
    "Vector",
    # Version
    "__version__",
    "embedding_registry",
    "get_embedding_provider",
    "get_llm_provider",
    "get_memory_policy",
    "get_settings",
    "llm_registry",
]
