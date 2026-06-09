"""Integration tests for full Engram workflow.

These tests require a running PostgreSQL database with pgvector.
Run with: pytest tests/integration -v --run-integration
"""

from __future__ import annotations

import contextlib
import os
import uuid

import pytest

# Skip all tests if not running integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
async def engram_client():
    """Create Engram client for integration tests."""
    from pathlib import Path

    from dotenv import load_dotenv

    # Load .env file to get actual database credentials (override existing)
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path, override=True)

    # Use sentence-transformers for local testing (no API key needed)
    os.environ["ENGRAM_EMBEDDING_PROVIDER"] = "sentence-transformers"
    os.environ["ENGRAM_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    os.environ["ENGRAM_EMBEDDING_DIMENSION"] = "384"

    # Clear cached settings to pick up new env vars
    from engram.core.config import clear_settings_cache

    clear_settings_cache()

    from engram import Engram

    client = Engram()
    await client.connect()
    yield client
    await client.close()


@pytest.fixture
async def clean_agent(engram_client):
    """Create a unique agent and clean up after test."""
    agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"

    yield agent_id

    # Cleanup
    with contextlib.suppress(Exception):
        await engram_client.purge(agent_id=agent_id)


class TestMemoryLifecycle:
    """Test complete memory lifecycle."""

    @pytest.mark.asyncio
    async def test_add_and_get_memory(self, engram_client, clean_agent) -> None:
        """Test adding and retrieving a memory."""
        memory = await engram_client.add(
            content="User prefers dark mode",
            agent_id=clean_agent,
        )

        assert memory.memory_id is not None
        assert memory.content == "User prefers dark mode"

        # Get it back
        retrieved = await engram_client.get(memory.memory_id)

        assert retrieved is not None
        assert retrieved.memory_id == memory.memory_id
        assert retrieved.content == memory.content

    @pytest.mark.asyncio
    async def test_update_memory(self, engram_client, clean_agent) -> None:
        """Test updating a memory."""
        # Add
        memory = await engram_client.add(
            content="User likes coffee",
            agent_id=clean_agent,
        )

        # Update
        updated = await engram_client.update(
            memory.memory_id,
            content="User prefers tea over coffee",
        )

        assert updated.content == "User prefers tea over coffee"
        assert updated.memory_id == memory.memory_id

    @pytest.mark.asyncio
    async def test_reinforce_memory(self, engram_client, clean_agent) -> None:
        """Test reinforcing a memory."""
        memory = await engram_client.add(
            content="Important fact",
            agent_id=clean_agent,
        )

        original_importance = memory.importance

        reinforced = await engram_client.reinforce(memory.memory_id, 0.2)

        assert reinforced.importance > original_importance

    @pytest.mark.asyncio
    async def test_forget_memory(self, engram_client, clean_agent) -> None:
        """Test forgetting a memory."""
        from engram.core.exceptions import MemoryNotFoundError

        memory = await engram_client.add(
            content="Temporary fact",
            agent_id=clean_agent,
        )

        deleted = await engram_client.forget(memory.memory_id)

        assert deleted is True

        # Should raise MemoryNotFoundError
        with pytest.raises(MemoryNotFoundError):
            await engram_client.get(memory.memory_id)


class TestSearchFunctionality:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search(self, engram_client, clean_agent) -> None:
        """Test hybrid search combines semantic and keyword."""
        # Add memories
        await engram_client.add(
            content="User works in software engineering",
            agent_id=clean_agent,
        )
        await engram_client.add(
            content="User enjoys hiking on weekends",
            agent_id=clean_agent,
        )
        await engram_client.add(
            content="User has a dog named Max",
            agent_id=clean_agent,
        )

        # Search
        results = await engram_client.search(
            query="What does the user do for work?",
            agent_id=clean_agent,
            limit=5,
        )

        assert len(results) > 0
        # Software engineering memory should be most relevant
        assert "software" in results[0].memory.content.lower()

    @pytest.mark.asyncio
    async def test_search_with_min_score(self, engram_client, clean_agent) -> None:
        """Test search respects min_score threshold."""
        await engram_client.add(
            content="Random unrelated content about weather",
            agent_id=clean_agent,
        )

        # Search for something unrelated with high threshold
        results = await engram_client.search(
            query="programming languages",
            agent_id=clean_agent,
            min_score=0.9,  # Very high threshold
        )

        # Should filter out low-relevance results
        assert all(r.score >= 0.9 for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, engram_client, clean_agent) -> None:
        """Test search respects limit parameter."""
        # Add multiple memories
        for i in range(10):
            await engram_client.add(
                content=f"Memory number {i} about testing",
                agent_id=clean_agent,
            )

        results = await engram_client.search(
            query="testing",
            agent_id=clean_agent,
            limit=3,
        )

        assert len(results) <= 3


class TestGraphOperations:
    """Test graph traversal operations."""

    @pytest.mark.asyncio
    async def test_create_relation(self, engram_client, clean_agent) -> None:
        """Test creating a relation between memories."""
        mem1 = await engram_client.add(
            content="User has goal to learn Python",
            agent_id=clean_agent,
        )
        mem2 = await engram_client.add(
            content="User started online Python course",
            agent_id=clean_agent,
        )

        # relate() returns None but creates the relation
        await engram_client.relate(
            source_id=mem1.memory_id,
            target_id=mem2.memory_id,
            relation_type="causes",  # Valid relation type
        )

        # Verify relation was created by traversing
        results = await engram_client.traverse(
            start_memory_id=mem1.memory_id,
            max_depth=1,
        )

        # Should find mem2 through the relation
        found_ids = [r.memory_id for r in results]
        assert mem2.memory_id in found_ids

    @pytest.mark.asyncio
    async def test_traverse_graph(self, engram_client, clean_agent) -> None:
        """Test graph traversal."""
        # Create chain of memories
        mem1 = await engram_client.add(content="Start", agent_id=clean_agent)
        mem2 = await engram_client.add(content="Middle", agent_id=clean_agent)
        mem3 = await engram_client.add(content="End", agent_id=clean_agent)

        await engram_client.relate(mem1.memory_id, mem2.memory_id)
        await engram_client.relate(mem2.memory_id, mem3.memory_id)

        # Traverse from start (correct param name is start_memory_id)
        results = await engram_client.traverse(
            start_memory_id=mem1.memory_id,
            max_depth=3,
        )

        # Should find connected memories
        found_ids = [r.memory_id for r in results]
        assert mem2.memory_id in found_ids or mem3.memory_id in found_ids


class TestSessionManagement:
    """Test session management."""

    @pytest.mark.asyncio
    async def test_session_context_manager(self, engram_client, clean_agent) -> None:
        """Test session as context manager."""
        # First add a memory to ensure agent exists in DB
        await engram_client.add(
            content="Pre-session memory to create agent",
            agent_id=clean_agent,
        )

        async with engram_client.session(agent_id=clean_agent) as session:
            assert session.is_active
            assert session.agent_id == clean_agent

            # Add memory in session
            memory = await engram_client.add(
                content="Session memory",
                agent_id=clean_agent,
                session_id=session.session_id,
            )

            assert memory.session_id == session.session_id


class TestBatchOperations:
    """Test batch operations."""

    @pytest.mark.asyncio
    async def test_add_batch(self, engram_client, clean_agent) -> None:
        """Test adding multiple memories in batch."""
        # add_batch expects list of dicts, not MemoryCreate objects
        creates = [
            {"content": f"Batch memory {i}", "agent_id": clean_agent} for i in range(5)
        ]

        memories = await engram_client.add_batch(creates)

        assert len(memories) == 5
        assert all(m.content.startswith("Batch memory") for m in memories)

    @pytest.mark.asyncio
    async def test_purge_agent(self, engram_client, clean_agent) -> None:
        """Test purging all memories for an agent."""
        # Add memories
        for i in range(3):
            await engram_client.add(
                content=f"To be purged {i}",
                agent_id=clean_agent,
            )

        # Purge
        count = await engram_client.purge(agent_id=clean_agent)

        assert count >= 3

        # Verify empty
        memories = await engram_client.list_recent(agent_id=clean_agent)
        assert len(memories) == 0


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check(self, engram_client) -> None:
        """Test health check returns status."""
        health = await engram_client.health_check()

        assert "status" in health
        assert health["status"] == "healthy"
        assert "components" in health
        assert "database" in health["components"]

    @pytest.mark.asyncio
    async def test_health_check_components(self, engram_client) -> None:
        """Test health check returns expected components."""
        health = await engram_client.health_check()

        assert health["status"] == "healthy"
        assert "components" in health
        # Check embedding component is present
        assert "embedding" in health["components"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, engram_client) -> None:
        """Test getting non-existent memory raises MemoryNotFoundError."""
        from engram.core.exceptions import MemoryNotFoundError

        with pytest.raises(MemoryNotFoundError):
            await engram_client.get("nonexistent_memory_id")

    @pytest.mark.asyncio
    async def test_forget_nonexistent_memory(self, engram_client) -> None:
        """Test forgetting non-existent memory returns False."""
        result = await engram_client.forget("nonexistent_memory_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_content_rejected(self, engram_client, clean_agent) -> None:
        """Test that empty content is rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await engram_client.add(content="", agent_id=clean_agent)

    @pytest.mark.asyncio
    async def test_unicode_content(self, engram_client, clean_agent) -> None:
        """Test handling of unicode content."""
        memory = await engram_client.add(
            content="User speaks 日本語, العربية, and emoji 🎉",
            agent_id=clean_agent,
        )

        assert "日本語" in memory.content
        assert "🎉" in memory.content

        # Should be searchable
        results = await engram_client.search(
            query="日本語",
            agent_id=clean_agent,
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_large_content(self, engram_client, clean_agent) -> None:
        """Test handling of large content."""
        large_content = "x" * 50000  # 50KB

        memory = await engram_client.add(
            content=large_content,
            agent_id=clean_agent,
        )

        assert len(memory.content) == 50000

    @pytest.mark.asyncio
    async def test_special_characters_in_agent_id(self, engram_client) -> None:
        """Test special characters in agent ID."""
        agent_id = "agent-with_special.chars:123"

        try:
            memory = await engram_client.add(
                content="Test content",
                agent_id=agent_id,
            )
            assert memory.agent_id == agent_id
        finally:
            await engram_client.purge(agent_id=agent_id)
