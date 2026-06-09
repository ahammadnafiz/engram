"""Pytest configuration and fixtures for Engram tests."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Set test environment variables before importing engram
os.environ.setdefault("ENGRAM_DATABASE_URL", "postgresql://localhost:5432/engram_test")
os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("ENGRAM_EMBEDDING_DIMENSION", "384")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Create a mock embedding provider for testing without API calls."""
    provider = MagicMock()
    provider.dimension = 1536

    # Return consistent fake embeddings
    async def mock_embed(text: str) -> list[float]:
        # Create a deterministic embedding based on text hash
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Create 1536 dimensions from hash
        embedding = []
        for i in range(1536):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        return embedding

    async def mock_embed_batch(texts: list[str]) -> list[list[float]]:
        return [await mock_embed(text) for text in texts]

    provider.embed = AsyncMock(side_effect=mock_embed)
    provider.embed_batch = AsyncMock(side_effect=mock_embed_batch)

    return provider


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create a mock storage for unit testing without database."""
    storage = MagicMock()
    storage.is_connected = True

    # Mock basic operations
    storage.connect = AsyncMock()
    storage.close = AsyncMock()
    storage.execute = AsyncMock(return_value="INSERT 1")
    storage.executemany = AsyncMock()
    storage.fetchone = AsyncMock(return_value=None)
    storage.fetchall = AsyncMock(return_value=[])
    storage.fetchval = AsyncMock(return_value=1)
    storage.init_schema = AsyncMock()
    storage.health_check = AsyncMock(
        return_value={"status": "healthy", "database": "connected"}
    )

    # Mock SQL loading
    storage.load_sql = MagicMock(return_value="SELECT 1")

    return storage


@pytest.fixture
def sample_memory_data() -> dict:
    """Sample memory data for testing."""
    return {
        "content": "User prefers dark mode in applications",
        "agent_id": "test_agent",
        "user_id": "test_user",
        "importance": 0.8,
        "metadata": {"source": "test", "category": "preference"},
    }


@pytest.fixture
def sample_memories_batch() -> list[dict]:
    """Sample batch of memories for testing."""
    return [
        {
            "content": "User works in finance industry",
            "agent_id": "test_agent",
            "importance": 0.7,
        },
        {
            "content": "User prefers morning meetings",
            "agent_id": "test_agent",
            "importance": 0.5,
        },
        {
            "content": "User is interested in AI and machine learning",
            "agent_id": "test_agent",
            "importance": 0.9,
        },
    ]


@pytest.fixture
def mock_memory_row() -> dict:
    """Create a mock database row for a memory."""
    return {
        "memory_id": "mem_test123",
        "agent_id": "test_agent",
        "user_id": "test_user",
        "session_id": None,
        "content": "Test memory content",
        "embedding": json.dumps([0.1] * 384),
        "importance": 0.5,
        "access_count": 0,
        "metadata": "{}",
        "created_at": datetime.now(timezone.utc),
        "last_accessed_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def mock_search_rows() -> list[dict]:
    """Create mock search result rows."""
    now = datetime.now(timezone.utc)
    return [
        {
            "memory_id": "mem_1",
            "content": "First result",
            "importance": 0.8,
            "metadata": "{}",
            "created_at": now,
            "last_accessed_at": now,
            "score": 0.95,
            "semantic_score": 0.9,
            "keyword_score": 0.8,
            "decay_score": 0.99,
        },
        {
            "memory_id": "mem_2",
            "content": "Second result",
            "importance": 0.6,
            "metadata": "{}",
            "created_at": now,
            "last_accessed_at": now,
            "score": 0.85,
            "semantic_score": 0.8,
            "keyword_score": 0.7,
            "decay_score": 0.95,
        },
    ]


@pytest.fixture
def mock_session_row() -> dict:
    """Create a mock database row for a session."""
    return {
        "session_id": "sess_test123",
        "agent_id": "test_agent",
        "user_id": "test_user",
        "started_at": datetime.now(timezone.utc),
        "ended_at": None,
        "metadata": "{}",
    }


@pytest.fixture
def mock_relation_row() -> dict:
    """Create a mock database row for a relation."""
    return {
        "source_memory_id": "mem_source",
        "target_memory_id": "mem_target",
        "relation_type": "related_to",
        "weight": 0.8,
        "metadata": "{}",
        "created_at": datetime.now(timezone.utc),
    }


# Integration test markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration tests unless explicitly requested."""
    if not config.getoption("--run-integration", default=False):
        skip_integration = pytest.mark.skip(reason="need --run-integration option")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
