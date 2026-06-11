"""Integration tests for MemoryStore correctness guards.

These run against a real PostgreSQL with pgvector (no embedding model is
loaded; a controllable fake provider is used so vector similarity can be
set exactly per test).

Run with: pytest tests/integration/test_store_guards.py -v --run-integration
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DIMENSION = 384  # match the dimension used by the rest of the integration suite


class FakeEmbedding:
    """Embedding service double with controllable per-text vectors."""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.dimension = DIMENSION
        self.model = "fake"

    def set_vector(self, text: str, axis: int) -> None:
        """Pin a text to a unit vector along the given axis."""
        vec = [0.0] * DIMENSION
        vec[axis] = 1.0
        self.vectors[text] = vec

    async def embed(self, text: str) -> list[float]:
        if text in self.vectors:
            return self.vectors[text]
        # Deterministic pseudo-random unit-ish vector (very low mutual cosine)
        digest = hashlib.sha256(text.encode()).digest()
        raw = [(digest[i % len(digest)] - 128) / 128.0 for i in range(DIMENSION)]
        norm = sum(x * x for x in raw) ** 0.5
        return [x / norm for x in raw]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture
async def store_env():
    """Connected MemoryStore + storage + fake embedding on a unique agent."""
    from dotenv import load_dotenv

    from engram.core.config import EngramSettings
    from engram.memory.store import MemoryStore
    from engram.storage.postgres import PostgresStorage

    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path, override=True)

    settings = EngramSettings(database_url=os.environ["ENGRAM_DATABASE_URL"])
    storage = PostgresStorage(settings)
    await storage.connect()
    await storage.init_schema(embedding_dimension=DIMENSION)

    embedding = FakeEmbedding()
    store = MemoryStore(storage, embedding, settings)
    agent_id = f"guard_test_{uuid.uuid4().hex[:12]}"

    yield store, storage, embedding, agent_id

    await store.purge(agent_id)
    await storage.close()


class TestPoolVectorSettings:
    """Connections must enable iterative HNSW scans for filtered recall."""

    @pytest.mark.asyncio
    async def test_iterative_scan_enabled_on_pool_connections(self, store_env) -> None:
        _store, storage, _embedding, _agent_id = store_env

        # pgvector >= 0.8 in the dev container; the setup callback must have
        # applied strict_order on every pooled connection.
        value = await storage.fetchval("SHOW hnsw.iterative_scan")
        assert value == "strict_order"


class TestNearDuplicateGuard:
    """The vector near-dup guard must respect status and user scope."""

    @pytest.mark.asyncio
    async def test_superseded_memory_does_not_block_reinsert(self, store_env) -> None:
        """Re-asserting a fact after it was superseded must create a new
        active memory, not return the hidden superseded one."""
        from engram.memory.models import MemoryCreate

        store, storage, embedding, agent_id = store_env

        old_text = "User lives in Dhaka"
        new_text = "User lives in Dhaka."  # rewording, same vector
        embedding.set_vector(old_text, axis=0)
        embedding.set_vector(new_text, axis=0)

        old = await store.add(MemoryCreate(content=old_text, agent_id=agent_id))
        # Simulate conflict resolution hiding the old memory
        await storage.execute(
            """
            UPDATE agent_memory
            SET metadata = metadata || '{"status": "superseded"}'::jsonb
            WHERE memory_id = $1
            """,
            old.memory_id,
        )

        new = await store.add(MemoryCreate(content=new_text, agent_id=agent_id))

        assert new.memory_id != old.memory_id
        assert new.metadata.get("status") != "superseded"

    @pytest.mark.asyncio
    async def test_agent_scoped_add_not_deduped_against_user_memory(
        self, store_env
    ) -> None:
        """An agent-scoped add (user_id=None) must not be absorbed by a
        vector-similar memory belonging to a specific user."""
        from engram.memory.models import MemoryCreate

        store, _storage, embedding, agent_id = store_env

        user_text = "User likes green tea"
        agent_text = "Green tea preference noted for all users"
        embedding.set_vector(user_text, axis=1)
        embedding.set_vector(agent_text, axis=1)

        user_mem = await store.add(
            MemoryCreate(content=user_text, agent_id=agent_id, user_id="alice")
        )
        agent_mem = await store.add(
            MemoryCreate(content=agent_text, agent_id=agent_id, user_id=None)
        )

        assert agent_mem.memory_id != user_mem.memory_id
        assert agent_mem.user_id is None


class TestAddBatchIds:
    """add_batch must never return memory IDs that don't exist in the DB."""

    @pytest.mark.asyncio
    async def test_exact_duplicate_in_batch_returns_existing_id(
        self, store_env
    ) -> None:
        from engram.memory.models import MemoryCreate

        store, _storage, _embedding, agent_id = store_env

        original = await store.add(
            MemoryCreate(content="User works at AskTuring", agent_id=agent_id)
        )
        batch = await store.add_batch(
            [MemoryCreate(content="User works at AskTuring", agent_id=agent_id)]
        )

        assert len(batch) == 1
        assert batch[0].memory_id == original.memory_id
        # The returned ID must be resolvable
        fetched = await store.get_without_access_update(batch[0].memory_id)
        assert fetched.content == "User works at AskTuring"

    @pytest.mark.asyncio
    async def test_all_batch_ids_exist_in_database(self, store_env) -> None:
        from engram.memory.models import MemoryCreate

        store, storage, _embedding, agent_id = store_env

        batch = await store.add_batch(
            [
                MemoryCreate(content="User has a cat named Luna", agent_id=agent_id),
                MemoryCreate(content="User has a cat named Luna", agent_id=agent_id),
                MemoryCreate(content="User plays chess", agent_id=agent_id),
            ]
        )

        for memory in batch:
            exists = await storage.fetchval(
                "SELECT EXISTS(SELECT 1 FROM agent_memory WHERE memory_id = $1)",
                memory.memory_id,
            )
            assert exists, f"add_batch returned phantom id {memory.memory_id}"

    @pytest.mark.asyncio
    async def test_in_batch_near_duplicates_deduped(self, store_env) -> None:
        """Two reworded-but-vector-identical facts in one batch -> one insert."""
        from engram.memory.models import MemoryCreate

        store, _storage, embedding, agent_id = store_env

        a = "User's bank is BRAC Bank"
        b = "User banks at BRAC Bank"
        embedding.set_vector(a, axis=2)
        embedding.set_vector(b, axis=2)

        batch = await store.add_batch(
            [
                MemoryCreate(content=a, agent_id=agent_id),
                MemoryCreate(content=b, agent_id=agent_id),
            ]
        )

        assert len(batch) == 1
        recent = await store.list_recent(agent_id, limit=10)
        assert len(recent) == 1


class TestUpdateEdgeCases:
    """update() edge cases around the unique fact index and NULL embeddings."""

    @pytest.mark.asyncio
    async def test_update_to_existing_fact_raises_duplicate_error(
        self, store_env
    ) -> None:
        """Updating a memory's content to another memory's exact fact must
        surface as DuplicateMemoryError, not an opaque StorageError."""
        from engram.core.exceptions import DuplicateMemoryError
        from engram.memory.models import MemoryCreate, MemoryUpdate

        store, _storage, _embedding, agent_id = store_env

        await store.add(MemoryCreate(content="User plays piano", agent_id=agent_id))
        other = await store.add(
            MemoryCreate(content="User plays violin", agent_id=agent_id)
        )

        with pytest.raises(DuplicateMemoryError):
            await store.update(
                other.memory_id, MemoryUpdate(content="User plays piano")
            )

    @pytest.mark.asyncio
    async def test_update_importance_with_null_embedding(self, store_env) -> None:
        """A memory whose embedding was cleared (e.g. by a dimension change)
        must still be updatable without writing the string 'null' into the
        vector column."""
        from engram.memory.models import MemoryCreate, MemoryUpdate

        store, storage, _embedding, agent_id = store_env

        memory = await store.add(
            MemoryCreate(content="User collects stamps", agent_id=agent_id)
        )
        await storage.execute(
            "UPDATE agent_memory SET embedding = NULL WHERE memory_id = $1",
            memory.memory_id,
        )

        updated = await store.update(memory.memory_id, MemoryUpdate(importance=0.9))

        assert updated.importance == 0.9
        assert updated.embedding is None


class TestSearchScoring:
    """min_score must filter before LIMIT; keyword decay matches the others."""

    @pytest.fixture
    async def open_store(self, store_env):
        """Store with the near-duplicate guard disabled, for dense vectors."""
        from engram.core.config import EngramSettings
        from engram.memory.store import MemoryStore

        _store, storage, embedding, agent_id = store_env
        settings = EngramSettings(
            database_url=os.environ["ENGRAM_DATABASE_URL"],
            near_duplicate_threshold=1.0,
        )
        return MemoryStore(storage, embedding, settings), storage, embedding, agent_id

    @pytest.mark.asyncio
    async def test_semantic_min_score_filters_before_limit(self, open_store) -> None:
        """High-similarity but stale (low-score) rows must not consume the
        LIMIT and hide fresh qualifying matches."""
        from engram.memory.models import MemoryCreate

        store, storage, embedding, agent_id = open_store

        # Two stale memories: identical to the query vector (closest by
        # distance) but last accessed a month ago -> decay ~0 -> low score.
        stale_texts = ["stale fact alpha", "stale fact beta"]
        for text in stale_texts:
            embedding.set_vector(text, axis=5)
            m = await store.add(MemoryCreate(content=text, agent_id=agent_id))
            await storage.execute(
                "UPDATE agent_memory SET last_accessed_at = NOW() - INTERVAL '30 days' "
                "WHERE memory_id = $1",
                m.memory_id,
            )

        # Three fresh memories at cosine 0.8 to the query -> high score.
        fresh_vector = [0.0] * DIMENSION
        fresh_vector[5] = 0.8
        fresh_vector[6] = 0.6
        for text in ["fresh fact one", "fresh fact two", "fresh fact three"]:
            embedding.vectors[text] = fresh_vector
            await store.add(MemoryCreate(content=text, agent_id=agent_id))

        embedding.set_vector("the query", axis=5)
        results = await store.semantic_search(
            "the query", agent_id, limit=3, min_score=0.7
        )

        # Stale: 0.60*1.0 + 0.25*~0 + 0.15*0.5 = ~0.68 < 0.7 -> excluded.
        # Fresh: 0.60*0.8 + 0.25*1.0 + 0.15*0.5 = ~0.81 >= 0.7 -> all three
        # must be returned even though the stale rows are closer by distance.
        contents = {r.memory.content for r in results}
        assert contents == {"fresh fact one", "fresh fact two", "fresh fact three"}

    @pytest.mark.asyncio
    async def test_keyword_decay_uses_last_access_not_age(self, open_store) -> None:
        """A frequently used old memory must not be punished for its age."""
        from engram.memory.models import MemoryCreate, SearchQuery

        store, storage, _embedding, agent_id = open_store

        m = await store.add(
            MemoryCreate(content="User deploys with Kubernetes", agent_id=agent_id)
        )
        # Created 100 days ago, but accessed just now.
        await storage.execute(
            "UPDATE agent_memory SET created_at = NOW() - INTERVAL '100 days', "
            "last_accessed_at = NOW() WHERE memory_id = $1",
            m.memory_id,
        )

        results = await store.search(
            SearchQuery(query="Kubernetes", agent_id=agent_id, mode="keyword")
        )

        assert len(results) == 1
        # Old per-day created_at decay would be ~0.6; per-hour
        # last_accessed_at decay is ~1.0.
        assert results[0].decay_score > 0.95


class TestMetadataSerialization:
    @pytest.mark.asyncio
    async def test_datetime_metadata_value_does_not_crash(self, store_env) -> None:
        """Non-JSON-native metadata values (datetime, UUID) must be coerced,
        not crash deep inside storage with an opaque StorageError."""
        from datetime import datetime, timezone

        from engram.memory.models import MemoryCreate

        store, _storage, _embedding, agent_id = store_env

        when = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
        memory = await store.add(
            MemoryCreate(
                content="User has a dentist appointment",
                agent_id=agent_id,
                metadata={"appointment_at": when},
            )
        )

        fetched = await store.get_without_access_update(memory.memory_id)
        assert "2026-06-10" in str(fetched.metadata["appointment_at"])


class TestTextSearchConfig:
    """The generated tsvector columns must follow the configured language."""

    @pytest.mark.asyncio
    async def test_rebuild_to_simple_and_back(self, store_env) -> None:
        from engram.memory.models import MemoryCreate, SearchQuery

        store, storage, _embedding, agent_id = store_env

        try:
            await storage._ensure_text_search_config("simple")
            expr = await storage.fetchval(
                """
                SELECT pg_get_expr(d.adbin, d.adrelid)
                FROM pg_attrdef d
                JOIN pg_attribute a
                    ON a.attrelid = d.adrelid AND a.attnum = d.adnum
                WHERE d.adrelid = 'agent_memory'::regclass
                    AND a.attname = 'fact_tsv'
                """
            )
            assert "'simple'::regconfig" in expr

            # Keyword search must work against the rebuilt column when the
            # query-side config matches.
            from engram.core.config import EngramSettings
            from engram.memory.store import MemoryStore

            simple_store = MemoryStore(
                storage,
                store._embedding,
                EngramSettings(
                    database_url=os.environ["ENGRAM_DATABASE_URL"],
                    text_search_config="simple",
                ),
            )
            await simple_store.add(
                MemoryCreate(content="User deploys via Terraform", agent_id=agent_id)
            )
            results = await simple_store.search(
                SearchQuery(query="Terraform", agent_id=agent_id, mode="keyword")
            )
            assert len(results) == 1
        finally:
            await storage._ensure_text_search_config("english")

    @pytest.mark.asyncio
    async def test_invalid_config_rejected_before_ddl(self, store_env) -> None:
        from engram.core.exceptions import ConfigurationError

        _store, storage, _embedding, _agent_id = store_env

        with pytest.raises(ConfigurationError):
            await storage._ensure_text_search_config("bad'; DROP TABLE agents;--")


class TestSupersedeAtomicity:
    """Conflict-key supersede must happen inside the add itself."""

    @pytest.mark.asyncio
    async def test_store_add_supersedes_same_conflict_key(self, store_env) -> None:
        from engram.memory.models import MemoryCreate

        store, _storage, _embedding, agent_id = store_env
        key = f"{agent_id}:*:profile:current_city"

        first = await store.add(
            MemoryCreate(
                content="User lives in Dhaka city",
                agent_id=agent_id,
                metadata={"conflict_key": key, "status": "active"},
            )
        )
        second = await store.add(
            MemoryCreate(
                content="User lives in New York City",
                agent_id=agent_id,
                metadata={"conflict_key": key, "status": "active"},
            )
        )

        old = await store.get_without_access_update(first.memory_id)
        new = await store.get_without_access_update(second.memory_id)
        assert old.metadata.get("status") == "superseded"
        assert old.metadata.get("superseded_by") == second.memory_id
        assert new.metadata.get("status") == "active"
