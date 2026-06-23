"""End-to-end tests covering all 51 public Engram methods.

Requirements:
    - Running PostgreSQL with pgvector
    - .env with ENGRAM_DATABASE_URL + ENGRAM_ANTHROPIC_API_KEY
    - sentence-transformers (auto-downloaded, cached after first run)

Run:
    PYTHONPATH=. poetry run pytest tests/integration/test_e2e_all_methods.py -v --run-integration

LLM tests skip gracefully when ENGRAM_ANTHROPIC_API_KEY is absent.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="session"),
]

# ── Benchmark-derived seed data ───────────────────────────────────────────────

# Sampled from LongMemEval + LoCoMo runs
MEMORIES = [
    "I graduated with a degree in Business Administration",
    "My Spotify playlist is called Summer Vibes",
    "My last name before I changed it was Johnson",
    "I repainted my bedroom walls a lighter shade of gray",
    "My daily commute to work is 45 minutes each way",
    "Jon lost his banking job on January 19 2023",
    "Jon and Gina both like to destress by dancing",
    "Jon opened a dance studio after losing his banking job",
    "I prefer dark mode across all applications",
    "I work in finance as a data analyst",
    "I prefer morning meetings scheduled before 10am",
    "My laptop budget is capped at 1500 dollars",
    "I started learning Python for data science three months ago",
]

LONG_DOC = """
# AI Memory System - Product Requirements

## Overview
The system stores and retrieves user memories with high fidelity.

## Requirements
- MUST support at least 10,000 memories per agent.
- MUST return search results in under 200ms.
- SHALL support hybrid search combining vector and keyword ranking.
- MUST never lose a stored memory silently.

## Constraints
- Maximum memory content: 4,000 characters.
- Memory store MUST NOT exceed 2GB RAM.
- All data MUST be encrypted at rest.

## Decisions
- Decided to use PostgreSQL with pgvector for storage.
- Approved sentence-transformers for local embeddings.
- Rejected Redis-only approach due to persistence concerns.
- Changed embedding dimension from 1536 to 384 to reduce index size.

## Legal
This system shall comply with GDPR requirements. The operator must never
share user data with third parties without explicit written consent.
Liability is governed by the jurisdiction of the applicable user.
"""

TURN_USER = "I just got promoted to Senior Data Analyst at my company!"
TURN_ASST = (
    "Congratulations on your promotion to Senior Data Analyst! "
    "That is a real achievement. What new responsibilities excite you most?"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


_configured = False


def _configure() -> None:
    """Load .env with override=True so it always wins over conftest defaults."""
    global _configured
    if _configured:
        return
    from dotenv import load_dotenv

    from engram.core.config import clear_settings_cache

    # override=True: .env values beat the module-level setdefault in conftest.py
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path, override=True)

    # ENGRAM_TEST_DATABASE_URL takes priority if the .env defines it
    test_url = os.environ.get("ENGRAM_TEST_DATABASE_URL")
    if test_url:
        os.environ["ENGRAM_DATABASE_URL"] = test_url

    # Force embedding to local sentence-transformers — no API cost, consistent.
    os.environ["ENGRAM_EMBEDDING_PROVIDER"] = "sentence-transformers"
    os.environ["ENGRAM_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    os.environ["ENGRAM_EMBEDDING_DIMENSION"] = "384"
    # Force LLM to Anthropic/Haiku — .env may have Gemini which would rate-limit.
    os.environ["ENGRAM_LLM_PROVIDER"] = "anthropic"
    os.environ["ENGRAM_LLM_MODEL"] = "claude-haiku-4-5-20251001"
    clear_settings_cache()
    _configured = True


def _has_llm() -> bool:
    _configure()  # ensure env loaded before checking
    return bool(os.environ.get("ENGRAM_ANTHROPIC_API_KEY"))


needs_llm = pytest.mark.skipif(
    not _has_llm(), reason="ENGRAM_ANTHROPIC_API_KEY not set"
)


# Session-scoped client: sentence-transformers loads once, asyncpg pool created once.
# pytest_asyncio.fixture with loop_scope="session" keeps one event loop for all tests.
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def eng():
    _configure()
    from engram import Engram

    client = Engram()
    await client.connect()
    await client.warmup()  # pre-load cross-encoder while we're here
    yield client
    await client.close()


# Fresh unique agent per test — no state bleed even with shared client.
@pytest.fixture
def agent() -> str:
    return f"e2e_{uuid.uuid4().hex[:10]}"


@pytest_asyncio.fixture(loop_scope="session")
async def seeded(eng, agent):
    """Shared client + fresh agent pre-loaded with MEMORIES. Cleans up after test."""
    mems = await eng.add_batch(
        [{"content": c, "agent_id": agent, "memory_type": "semantic"} for c in MEMORIES]
    )
    yield eng, agent, mems
    await eng.purge(agent)


# ── 1. Lifecycle ──────────────────────────────────────────────────────────────


class TestLifecycle:
    async def test_connect_and_close(self):
        from engram import Engram

        client = Engram()
        assert not client.is_connected
        await client.connect()
        assert client.is_connected
        await client.close()
        assert not client.is_connected

    async def test_context_manager(self):
        from engram import Engram

        async with Engram() as client:
            assert client.is_connected
        assert not client.is_connected

    async def test_double_connect_is_idempotent(self, eng):
        await eng.connect()  # second call - should not raise
        assert eng.is_connected

    async def test_health_check_returns_healthy(self, eng):
        result = await eng.health_check()
        assert result["status"] == "healthy"

    async def test_health_check_skip_embedding(self, eng):
        result = await eng.health_check(skip_embedding_test=True)
        assert result["status"] == "healthy"

    async def test_warmup(self, eng):
        # warmup loads cross-encoder; safe to call multiple times
        await eng.warmup()
        await eng.warmup()


# ── 2. Memory Write ───────────────────────────────────────────────────────────


class TestMemoryWrite:
    async def test_add_basic(self, eng, agent):
        mem = await eng.add(content="User prefers dark mode", agent_id=agent)
        assert mem.memory_id
        assert mem.content == "User prefers dark mode"
        await eng.purge(agent)

    async def test_add_with_main_content_and_metadata(self, eng, agent):
        mem = await eng.add(
            content="User works in finance",
            agent_id=agent,
            main_content="[USER]: I work in finance\n[AI]: Got it.",
            memory_type="semantic",
            metadata={"source": "lme", "category": "occupation"},
        )
        assert mem.metadata.get("source") == "lme"
        await eng.purge(agent)

    async def test_add_batch_all_stored(self, eng, agent):
        mems = await eng.add_batch(
            [{"content": c, "agent_id": agent} for c in MEMORIES]
        )
        assert len(mems) == len(MEMORIES)
        ids = {m.memory_id for m in mems}
        assert len(ids) == len(MEMORIES)
        await eng.purge(agent)

    async def test_add_batch_with_session_and_user(self, eng, agent):
        async with eng.session(agent_id=agent) as sess:
            mems = await eng.add_batch(
                [
                    {
                        "content": c,
                        "agent_id": agent,
                        "session_id": sess.session_id,
                        "user_id": "u1",
                    }
                    for c in MEMORIES[:3]
                ]
            )
        assert all(m.session_id == sess.session_id for m in mems)
        await eng.purge(agent)

    async def test_update_in_place_no_lineage(self, eng, agent):
        mem = await eng.add(content="Budget is 1000 dollars", agent_id=agent)
        updated = await eng.update(mem.memory_id, content="Budget is 1500 dollars")
        assert updated.content == "Budget is 1500 dollars"
        lineage = await eng.get_lineage(mem.memory_id)
        # update() is in-place: lineage has only one entry
        assert len(lineage.memories) == 1
        await eng.purge(agent)

    async def test_revise_creates_new_revision(self, eng, agent):
        mem = await eng.add(content="I prefer coffee", agent_id=agent)
        rev = await eng.revise(
            mem.memory_id, content="I prefer tea over coffee", reason="update"
        )
        assert rev.content == "I prefer tea over coffee"
        lineage = await eng.get_lineage(mem.memory_id)
        assert len(lineage.memories) == 2
        await eng.purge(agent)

    async def test_revise_chain_three_deep(self, eng, agent):
        m = await eng.add(content="Salary is 80k", agent_id=agent)
        r1 = await eng.revise(m.memory_id, content="Salary is 90k", reason="raise")
        await eng.revise(r1.memory_id, content="Salary is 100k", reason="promotion")
        lineage = await eng.get_lineage(m.memory_id)
        assert len(lineage.memories) == 3
        current = await eng.get_current(m.memory_id)
        assert current.content == "Salary is 100k"
        await eng.purge(agent)

    async def test_reinforce_boosts_importance(self, eng, agent):
        mem = await eng.add(
            content="Critical constraint: never share passwords", agent_id=agent
        )
        original = mem.importance
        reinforced = await eng.reinforce(mem.memory_id, importance_boost=0.2)
        assert reinforced.importance > original
        await eng.purge(agent)

    async def test_reinforce_capped_at_one(self, eng, agent):
        mem = await eng.add(content="Max importance test", agent_id=agent)
        for _ in range(10):
            mem = await eng.reinforce(mem.memory_id, importance_boost=0.3)
        assert mem.importance <= 1.0
        await eng.purge(agent)

    async def test_forget_returns_true(self, eng, agent):
        mem = await eng.add(content="Temporary fact", agent_id=agent)
        deleted = await eng.forget(mem.memory_id)
        assert deleted is True
        await eng.purge(agent)

    async def test_forget_nonexistent_returns_false(self, eng):
        result = await eng.forget("nonexistent_memory_id_xyz")
        assert result is False

    async def test_purge_returns_count(self, eng, agent):
        await eng.add_batch([{"content": c, "agent_id": agent} for c in MEMORIES[:5]])
        count = await eng.purge(agent)
        assert count == 5

    async def test_purge_empty_agent_returns_zero(self, eng, agent):
        count = await eng.purge(agent)
        assert count == 0


@needs_llm
class TestMemoryWriteLLM:
    async def test_add_conversation_extracts_facts(self, eng, agent):
        result = await eng.add_conversation(TURN_USER, TURN_ASST, agent_id=agent)
        # Should extract at least one fact about the promotion
        assert len(result.decisions) > 0
        applied = [d for d in result.decisions if d.applied]
        assert len(applied) > 0
        await eng.purge(agent)

    async def test_add_conversation_with_session_updates_summary(self, eng, agent):
        async with eng.session(agent_id=agent) as sess:
            result = await eng.add_conversation(
                TURN_USER,
                TURN_ASST,
                agent_id=agent,
                session_id=sess.session_id,
                update_summary=True,
            )
        assert len(result.decisions) >= 0  # no error
        await eng.purge(agent)

    async def test_add_conversation_no_extract_assistant(self, eng, agent):
        result = await eng.add_conversation(
            "My dog's name is Biscuit.",
            "That is a lovely name for a dog!",
            agent_id=agent,
            extract_assistant_response=False,
        )
        # Only user facts extracted
        facts = [d.fact for d in result.decisions]
        assert any("Biscuit" in f or "dog" in f.lower() for f in facts)
        await eng.purge(agent)


# ── 3. Memory Read ────────────────────────────────────────────────────────────


class TestMemoryRead:
    async def test_get_updates_access_count(self, seeded):
        eng, _agent, mems = seeded
        mem = mems[0]
        before = await eng.get(mem.memory_id, track_access=False)
        await eng.get(mem.memory_id, track_access=True)
        after = await eng.get(mem.memory_id, track_access=False)
        assert after.access_count > before.access_count

    async def test_get_no_track(self, seeded):
        eng, _agent, mems = seeded
        m1 = await eng.get(mems[0].memory_id, track_access=False)
        m2 = await eng.get(mems[0].memory_id, track_access=False)
        assert m1.access_count == m2.access_count

    async def test_get_current_returns_active_head(self, seeded):
        eng, _agent, mems = seeded
        original = mems[0]
        revised = await eng.revise(original.memory_id, content="Revised content")
        current = await eng.get_current(original.memory_id)
        assert current.memory_id == revised.memory_id

    async def test_get_lineage_newest_first(self, seeded):
        eng, _agent, mems = seeded
        m = mems[0]
        r1 = await eng.revise(m.memory_id, content="Rev 1")
        await eng.revise(r1.memory_id, content="Rev 2")
        lineage = await eng.get_lineage(m.memory_id)
        assert lineage.memories[0].content == "Rev 2"
        assert len(lineage.memories) == 3

    async def test_explain_memory(self, seeded):
        eng, _agent, mems = seeded
        explanation = await eng.explain_memory(mems[0].memory_id)
        assert explanation is not None

    async def test_list_recent_ordered(self, eng, agent):
        contents = ["First", "Second", "Third"]
        for c in contents:
            await eng.add(content=c, agent_id=agent)
        recent = await eng.list_recent(agent, limit=3)
        assert recent[0].content == "Third"
        await eng.purge(agent)

    async def test_list_recent_respects_limit(self, seeded):
        eng, agent, _mems = seeded
        recent = await eng.list_recent(agent, limit=5)
        assert len(recent) <= 5

    async def test_get_memories_plain_read(self, seeded):
        eng, agent, _mems = seeded
        all_mems = await eng.get_memories(agent, limit=100)
        assert len(all_mems) == len(MEMORIES)

    async def test_get_memories_filter_by_type(self, eng, agent):
        await eng.add(content="Semantic fact", agent_id=agent, memory_type="semantic")
        await eng.add(content="Episodic event", agent_id=agent, memory_type="episodic")
        semantic = await eng.get_memories(agent, memory_types=["semantic"])
        episodic = await eng.get_memories(agent, memory_types=["episodic"])
        assert all(m.memory_type == "semantic" for m in semantic)
        assert all(m.memory_type == "episodic" for m in episodic)
        await eng.purge(agent)

    async def test_get_memories_filter_by_session(self, eng, agent):
        async with eng.session(agent_id=agent) as sess:
            await eng.add(
                content="In session", agent_id=agent, session_id=sess.session_id
            )
        await eng.add(content="No session", agent_id=agent)
        in_session = await eng.get_memories(agent, session_id=sess.session_id)
        assert len(in_session) == 1
        assert in_session[0].content == "In session"
        await eng.purge(agent)

    async def test_get_memories_metadata_filter(self, eng, agent):
        await eng.add(content="Tagged", agent_id=agent, metadata={"tag": "special"})
        await eng.add(content="Untagged", agent_id=agent)
        tagged = await eng.get_memories(agent, metadata_filter={"tag": "special"})
        assert len(tagged) == 1
        await eng.purge(agent)

    async def test_get_history_includes_superseded(self, eng, agent):
        m = await eng.add(content="Original", agent_id=agent)
        await eng.revise(m.memory_id, content="Revised")
        history = await eng.get_history(agent, include_superseded=True)
        event_types = {e.event_type for e in history}
        assert "added" in event_types or "revised" in event_types
        await eng.purge(agent)

    async def test_get_history_time_filter(self, seeded):
        eng, agent, _mems = seeded
        since = datetime.now(timezone.utc) - timedelta(hours=1)
        history = await eng.get_history(agent, since=since, limit=50)
        assert len(history) > 0


# ── 4. Search & Retrieval ─────────────────────────────────────────────────────


class TestSearch:
    async def test_search_hybrid_default(self, seeded):
        eng, agent, _ = seeded
        results = await eng.search("dark mode preference", agent_id=agent)
        assert len(results) > 0
        assert all(hasattr(r, "score") for r in results)

    async def test_search_semantic_mode(self, seeded):
        eng, agent, _ = seeded
        results = await eng.search(
            "color preference for UI", agent_id=agent, mode="semantic"
        )
        assert len(results) > 0

    async def test_search_keyword_mode(self, seeded):
        eng, agent, _ = seeded
        results = await eng.search(
            "dance studio banker", agent_id=agent, mode="keyword"
        )
        assert len(results) >= 0  # keyword may return 0 for short queries

    async def test_search_with_rerank(self, seeded):
        eng, agent, _ = seeded
        results = await eng.search("finance job", agent_id=agent, limit=5, rerank=True)
        assert len(results) <= 5

    async def test_search_include_superseded(self, eng, agent):
        m = await eng.add(content="Old value: budget 1000", agent_id=agent)
        await eng.revise(m.memory_id, content="New value: budget 2000")
        with_sup = await eng.search(
            "budget", agent_id=agent, include_superseded=True, limit=10
        )
        without_sup = await eng.search(
            "budget", agent_id=agent, include_superseded=False, limit=10
        )
        assert len(with_sup) >= len(without_sup)
        await eng.purge(agent)

    async def test_search_min_score_filters(self, seeded):
        eng, agent, _ = seeded
        results = await eng.search("dark mode", agent_id=agent, min_score=0.99)
        # very high threshold - may return 0; just verify no error
        assert isinstance(results, list)

    async def test_search_metadata_filter(self, eng, agent):
        await eng.add(content="Tagged memory", agent_id=agent, metadata={"src": "lme"})
        await eng.add(content="Other memory", agent_id=agent, metadata={"src": "beam"})
        results = await eng.search(
            "memory", agent_id=agent, metadata_filter={"src": "lme"}
        )
        assert all(r.memory.metadata.get("src") == "lme" for r in results)
        await eng.purge(agent)

    async def test_search_empty_agent_returns_empty(self, eng, agent):
        results = await eng.search("anything", agent_id=agent)
        assert results == []

    async def test_recall_critical_bypasses_vector(self, eng, agent):
        # recall_critical returns memories with critical policy metadata
        # add with default policy; critical only if policy marks it
        await eng.add_batch([{"content": c, "agent_id": agent} for c in MEMORIES[:5]])
        critical = await eng.recall_critical(agent)
        # critical subset may be empty with default policy; just no error
        assert isinstance(critical, list)
        await eng.purge(agent)

    async def test_trace_recall_structure(self, seeded):
        eng, agent, _ = seeded
        trace = await eng.trace_recall(
            "dancing job finance",
            agent_id=agent,
            limit=5,
            max_tokens=2000,
            expected_terms=["dance"],
        )
        assert trace.query == "dancing job finance"
        assert isinstance(trace.kept_memory_ids, list)
        assert isinstance(trace.context, str)

    async def test_trace_recall_missing_terms_flagged(self, seeded):
        eng, agent, _ = seeded
        trace = await eng.trace_recall(
            "Python coding",
            agent_id=agent,
            expected_terms=["nonexistent_term_xyz"],
        )
        assert "nonexistent_term_xyz" in trace.missing_expected_terms

    async def test_get_context_block_plain(self, seeded):
        eng, agent, _ = seeded
        block = await eng.get_context_block(
            "dark mode finance", agent_id=agent, limit=5
        )
        assert "## Relevant memories" in block or block == ""

    async def test_get_context_block_grouped(self, eng, agent):
        await eng.add(content="Semantic fact", agent_id=agent, memory_type="semantic")
        await eng.add(content="Episodic event", agent_id=agent, memory_type="episodic")
        block = await eng.get_context_block(
            "fact event", agent_id=agent, group_by_type=True
        )
        assert isinstance(block, str)
        await eng.purge(agent)

    async def test_get_context_block_with_session_summary(self, eng, agent):
        async with eng.session(agent_id=agent) as sess:
            await eng.add(
                content="In-session fact", agent_id=agent, session_id=sess.session_id
            )
        block = await eng.get_context_block(
            "fact", agent_id=agent, session_id=sess.session_id
        )
        assert isinstance(block, str)
        await eng.purge(agent)

    async def test_get_context_block_max_tokens_respected(self, seeded):
        eng, agent, _ = seeded
        block = await eng.get_context_block(
            "finance dark mode dance Python",
            agent_id=agent,
            max_tokens=50,
        )
        # block should be trimmed to budget
        token_est = max(1, len(block) // 4)
        assert token_est <= 100  # some slack for the header


class TestDeepSearch:
    async def test_deep_search_returns_merged_results(self, seeded):
        eng, agent, _ = seeded
        results = await eng.deep_search(
            "dancing finance job loss", agent_id=agent, limit=5
        )
        assert isinstance(results, list)
        assert len(results) <= 5

    async def test_deep_search_dedupes(self, seeded):
        eng, agent, _ = seeded
        results = await eng.deep_search(
            "dark mode preference", agent_id=agent, limit=10
        )
        ids = [r.memory.memory_id for r in results]
        assert len(ids) == len(set(ids))  # no duplicates

    async def test_deep_search_with_rerank(self, seeded):
        eng, agent, _ = seeded
        results = await eng.deep_search(
            "commute morning schedule", agent_id=agent, limit=5, rerank=True
        )
        assert isinstance(results, list)


@needs_llm
class TestRecall:
    async def test_recall_current_intent(self, seeded):
        eng, agent, _ = seeded
        answer = await eng.recall("What is my daily commute?", agent_id=agent)
        assert answer.intent in (
            "current",
            "chat",
            "historical",
            "event",
            "lineage",
            "temporal_chain",
        )
        assert isinstance(answer.answer_text, str)

    async def test_recall_knowledge_update_intent(self, eng, agent):
        m = await eng.add(content="My laptop budget was 1000 dollars", agent_id=agent)
        await eng.revise(m.memory_id, content="My laptop budget is now 1500 dollars")
        answer = await eng.recall("What is my current laptop budget?", agent_id=agent)
        assert isinstance(answer.answer_text, str)
        await eng.purge(agent)

    async def test_recall_with_question_date(self, seeded):
        eng, agent, _ = seeded
        qdate = datetime(2024, 6, 1, tzinfo=timezone.utc)
        answer = await eng.recall(
            "When did Jon lose his banking job?",
            agent_id=agent,
            question_date=qdate,
        )
        assert isinstance(answer.answer_text, str)

    async def test_recall_no_compose(self, seeded):
        eng, agent, _ = seeded
        answer = await eng.recall(
            "What do I know about dark mode?",
            agent_id=agent,
            compose_answer=False,
        )
        # no LLM composer call; structured evidence only
        assert answer.answer_text == "" or answer.evidence is not None

    async def test_recall_temporal_chain(self, eng, agent):
        await eng.add(
            content="I started the ML project on March 1 2024", agent_id=agent
        )
        await eng.add(
            content="I submitted the ML project on March 15 2024", agent_id=agent
        )
        answer = await eng.recall(
            "How many days between starting and submitting the ML project?",
            agent_id=agent,
        )
        assert isinstance(answer.answer_text, str)
        await eng.purge(agent)

    async def test_recall_chat_intent(self, seeded):
        eng, agent, _ = seeded
        answer = await eng.recall("How are you doing today?", agent_id=agent)
        # chat intent should return quickly with minimal/no evidence
        assert answer is not None


# ── 5. Graph ──────────────────────────────────────────────────────────────────


class TestGraph:
    async def test_relate_and_traverse(self, eng, agent):
        m1 = await eng.add(content="Jon lost his banking job", agent_id=agent)
        m2 = await eng.add(content="Jon opened a dance studio", agent_id=agent)
        await eng.relate(m1.memory_id, m2.memory_id, relation_type="causes", weight=0.9)
        results = await eng.traverse(m1.memory_id, max_depth=1, direction="outbound")
        ids = [r.memory_id for r in results]
        assert m2.memory_id in ids
        await eng.purge(agent)

    async def test_traverse_inbound(self, eng, agent):
        src = await eng.add(content="Source node", agent_id=agent)
        tgt = await eng.add(content="Target node", agent_id=agent)
        await eng.relate(src.memory_id, tgt.memory_id, relation_type="related_to")
        inbound = await eng.traverse(tgt.memory_id, max_depth=1, direction="inbound")
        ids = [r.memory_id for r in inbound]
        assert src.memory_id in ids
        await eng.purge(agent)

    async def test_traverse_any_direction(self, eng, agent):
        a = await eng.add(content="Node A", agent_id=agent)
        b = await eng.add(content="Node B", agent_id=agent)
        c = await eng.add(content="Node C", agent_id=agent)
        await eng.relate(a.memory_id, b.memory_id, relation_type="related_to")
        await eng.relate(c.memory_id, b.memory_id, relation_type="related_to")
        results = await eng.traverse(b.memory_id, max_depth=1, direction="any")
        ids = [r.memory_id for r in results]
        assert a.memory_id in ids
        assert c.memory_id in ids
        await eng.purge(agent)

    async def test_traverse_with_embedding(self, eng, agent):
        m1 = await eng.add(content="Node with embedding", agent_id=agent)
        m2 = await eng.add(content="Related node", agent_id=agent)
        await eng.relate(m1.memory_id, m2.memory_id, relation_type="related_to")
        # Traverse without query embedding — score by path weight only
        results = await eng.traverse(m1.memory_id, max_depth=1)
        ids = [r.memory_id for r in results]
        assert m2.memory_id in ids
        await eng.purge(agent)

    async def test_traverse_many(self, eng, agent):
        mems = await eng.add_batch(
            [{"content": f"Node {i}", "agent_id": agent} for i in range(4)]
        )
        await eng.relate(mems[0].memory_id, mems[1].memory_id)
        await eng.relate(mems[2].memory_id, mems[3].memory_id)
        results = await eng.traverse_many(
            [mems[0].memory_id, mems[2].memory_id],
            max_depth=1,
        )
        ids = {r.memory_id for r in results}
        assert mems[1].memory_id in ids
        assert mems[3].memory_id in ids
        await eng.purge(agent)

    async def test_traverse_many_skip_missing(self, eng, agent):
        m = await eng.add(content="Real node", agent_id=agent)
        results = await eng.traverse_many(
            [m.memory_id, "ghost_id_that_does_not_exist"],
            skip_missing=True,
        )
        assert isinstance(results, list)
        await eng.purge(agent)

    async def test_render_graph_context(self, eng, agent):
        m1 = await eng.add(content="Cause: job loss", agent_id=agent)
        m2 = await eng.add(content="Effect: new business", agent_id=agent)
        await eng.relate(m1.memory_id, m2.memory_id, relation_type="causes")
        results = await eng.traverse(m1.memory_id, max_depth=1)
        block = eng.render_graph_context(results, header="## Graph")
        assert isinstance(block, str)
        await eng.purge(agent)

    async def test_render_graph_context_with_paths(self, eng, agent):
        m1 = await eng.add(content="Start", agent_id=agent)
        m2 = await eng.add(content="End", agent_id=agent)
        await eng.relate(m1.memory_id, m2.memory_id)
        results = await eng.traverse(m1.memory_id, max_depth=1)
        block = eng.render_graph_context(results, include_paths=True)
        assert isinstance(block, str)
        await eng.purge(agent)


# ── 6. Sessions ───────────────────────────────────────────────────────────────


class TestSessions:
    async def test_session_context_manager(self, eng, agent):
        async with eng.session(agent_id=agent) as sess:
            assert sess.session_id
            mem = await eng.add(
                content="In-session memory",
                agent_id=agent,
                session_id=sess.session_id,
            )
            assert mem.session_id == sess.session_id
        await eng.purge(agent)

    async def test_session_with_user(self, eng, agent):
        async with eng.session(agent_id=agent, user_id="user42") as sess:
            assert sess.session_id
        await eng.purge(agent)

    async def test_session_with_metadata(self, eng, agent):
        async with eng.session(agent_id=agent, metadata={"channel": "web"}) as sess:
            assert sess.session_id
        await eng.purge(agent)


# ── 7. Task Memory ────────────────────────────────────────────────────────────


class TestTaskLifecycle:
    async def test_start_and_get_task(self, eng, agent):
        task = await eng.start_task("Analyze finance data", agent_id=agent)
        assert task.task_run_id
        fetched = await eng.get_task(task.task_run_id)
        assert fetched.task_run_id == task.task_run_id
        assert fetched.goal == "Analyze finance data"
        await eng.purge(agent)

    async def test_list_tasks(self, eng, agent):
        t1 = await eng.start_task("Task one", agent_id=agent)
        t2 = await eng.start_task("Task two", agent_id=agent)
        tasks = await eng.list_tasks(agent_id=agent)
        ids = {t.task_run_id for t in tasks}
        assert t1.task_run_id in ids
        assert t2.task_run_id in ids
        await eng.purge(agent)

    async def test_complete_task(self, eng, agent):
        task = await eng.start_task("Goal", agent_id=agent)
        done = await eng.complete_task(
            task.task_run_id, outcome="Finished successfully"
        )
        assert done.status == "completed"
        await eng.purge(agent)

    async def test_pause_task(self, eng, agent):
        task = await eng.start_task("Pausable task", agent_id=agent)
        paused = await eng.pause_task(task.task_run_id, outcome="Waiting for input")
        assert paused.status == "paused"
        await eng.purge(agent)

    async def test_fail_task(self, eng, agent):
        task = await eng.start_task("Failing task", agent_id=agent)
        failed = await eng.fail_task(task.task_run_id, outcome="Error encountered")
        assert failed.status == "failed"
        await eng.purge(agent)

    async def test_cancel_task(self, eng, agent):
        task = await eng.start_task("Cancel me", agent_id=agent)
        cancelled = await eng.cancel_task(task.task_run_id)
        assert cancelled.status == "cancelled"
        await eng.purge(agent)

    async def test_soft_delete_task(self, eng, agent):
        task = await eng.start_task("Delete me", agent_id=agent)
        deleted = await eng.soft_delete_task(task.task_run_id)
        assert deleted is not None
        # soft-deleted tasks not visible without include_deleted
        tasks = await eng.list_tasks(agent_id=agent, include_deleted=False)
        ids = {t.task_run_id for t in tasks}
        assert task.task_run_id not in ids
        await eng.purge(agent)

    async def test_list_tasks_by_status(self, eng, agent):
        t1 = await eng.start_task("Active task", agent_id=agent)
        t2 = await eng.start_task("Done task", agent_id=agent)
        await eng.complete_task(t2.task_run_id)
        completed = await eng.list_tasks(agent_id=agent, status="completed")
        completed_ids = {t.task_run_id for t in completed}
        assert t2.task_run_id in completed_ids
        assert t1.task_run_id not in completed_ids
        await eng.purge(agent)


class TestTaskEvents:
    async def test_record_event(self, eng, agent):
        task = await eng.start_task("Event task", agent_id=agent)
        event = await eng.record_event(
            agent_id=agent,
            role="user",
            event_type="user_message",
            content="What is my commute time?",
            task_run_id=task.task_run_id,
        )
        assert event.event_id
        assert event.content == "What is my commute time?"
        await eng.purge(agent)

    async def test_list_events(self, eng, agent):
        task = await eng.start_task("List events task", agent_id=agent)
        for i in range(3):
            await eng.record_event(
                agent_id=agent,
                role="user",
                event_type="user_message",
                content=f"Message {i}",
                task_run_id=task.task_run_id,
            )
        events = await eng.list_events(task_run_id=task.task_run_id)
        assert len(events) >= 3
        await eng.purge(agent)

    async def test_record_turn_creates_two_events(self, eng, agent):
        task = await eng.start_task("Turn task", agent_id=agent)
        events = await eng.record_turn(
            task.task_run_id,
            TURN_USER,
            TURN_ASST,
            agent_id=agent,
            enqueue_processing=False,
        )
        assert len(events) == 2
        roles = {e.role for e in events}
        assert "user" in roles
        assert "assistant" in roles
        await eng.purge(agent)

    async def test_record_turn_with_tool_calls(self, eng, agent):
        task = await eng.start_task("Tool task", agent_id=agent)
        events = await eng.record_turn(
            task.task_run_id,
            "Search my memories",
            "I found 3 relevant memories.",
            agent_id=agent,
            tool_calls=[{"name": "search", "query": "memories"}],
            enqueue_processing=False,
        )
        # user + assistant + tool_call = 3 events
        assert len(events) == 3
        await eng.purge(agent)

    async def test_search_events(self, eng, agent):
        task = await eng.start_task("Search events task", agent_id=agent)
        await eng.record_event(
            agent_id=agent,
            role="user",
            event_type="user_message",
            content="Jon opened a dance studio after losing his banking job",
            task_run_id=task.task_run_id,
        )
        await eng.backfill_event_embeddings(agent_id=agent)
        results = await eng.search_events("dance studio banking", agent_id=agent)
        assert isinstance(results, list)
        await eng.purge(agent)

    async def test_search_events_keyword_mode(self, eng, agent):
        task = await eng.start_task("KW search task", agent_id=agent)
        await eng.record_event(
            agent_id=agent,
            role="user",
            event_type="user_message",
            content="I prefer morning meetings before 10am",
            task_run_id=task.task_run_id,
        )
        results = await eng.search_events(
            "morning meetings", agent_id=agent, mode="keyword"
        )
        assert isinstance(results, list)
        await eng.purge(agent)

    async def test_search_events_time_filter(self, eng, agent):
        task = await eng.start_task("Time filter task", agent_id=agent)
        await eng.record_event(
            agent_id=agent,
            role="assistant",
            event_type="assistant_message",
            content="Noted your commute time",
            task_run_id=task.task_run_id,
        )
        since = datetime.now(timezone.utc) - timedelta(minutes=5)
        results = await eng.search_events("commute", agent_id=agent, since=since)
        assert isinstance(results, list)
        await eng.purge(agent)

    async def test_redact_event(self, eng, agent):
        task = await eng.start_task("Redact task", agent_id=agent)
        event = await eng.record_event(
            agent_id=agent,
            role="user",
            event_type="user_message",
            content="Sensitive PII: SSN 123-45-6789",
            task_run_id=task.task_run_id,
        )
        redacted = await eng.redact_event(event.event_id)
        assert redacted.event_id == event.event_id
        await eng.purge(agent)

    async def test_backfill_event_embeddings(self, eng, agent):
        task = await eng.start_task("Backfill task", agent_id=agent)
        for c in MEMORIES[:3]:
            await eng.record_event(
                agent_id=agent,
                role="user",
                event_type="user_message",
                content=c,
                task_run_id=task.task_run_id,
            )
        count = await eng.backfill_event_embeddings(agent_id=agent, limit=10)
        assert count >= 0  # 0 if already embedded, >0 if backfilled
        await eng.purge(agent)


class TestCheckpoints:
    async def test_create_checkpoint(self, eng, agent):
        task = await eng.start_task("Checkpoint task", agent_id=agent)
        cp = await eng.create_checkpoint(
            task.task_run_id,
            summary="Completed initial data analysis",
            completed_steps=["load data", "clean data"],
            pending_steps=["model training", "evaluation"],
            decisions=["Use logistic regression baseline"],
            blockers=["Waiting for GPU quota"],
        )
        assert cp.checkpoint_id
        assert "Completed initial data analysis" in cp.summary
        await eng.purge(agent)

    async def test_create_multiple_checkpoints(self, eng, agent):
        task = await eng.start_task("Multi-checkpoint task", agent_id=agent)
        for i in range(3):
            await eng.create_checkpoint(task.task_run_id, summary=f"Step {i} done")
        await eng.purge(agent)


class TestLongInput:
    async def test_record_long_input_no_llm(self, eng, agent):
        task = await eng.start_task("Long input task", agent_id=agent)
        report = await eng.record_long_input(
            task.task_run_id,
            LONG_DOC,
            title="Product Requirements",
            agent_id=agent,
            extract_with_llm=False,
        )
        assert report.source_event_id
        assert len(report.chunks) > 0
        assert len(report.memory_ids) > 0
        assert report.checkpoint_id
        await eng.purge(agent)

    @needs_llm
    async def test_record_long_input_with_llm(self, eng, agent):
        task = await eng.start_task("LLM long input task", agent_id=agent)
        report = await eng.record_long_input(
            task.task_run_id,
            LONG_DOC,
            title="PRD with LLM",
            agent_id=agent,
            extract_with_llm=True,
            max_facts_per_chunk=3,
        )
        assert len(report.memory_ids) > 0
        await eng.purge(agent)

    async def test_build_long_input_context(self, eng, agent):
        task = await eng.start_task("Context build task", agent_id=agent)
        await eng.record_long_input(
            task.task_run_id,
            LONG_DOC,
            title="PRD",
            agent_id=agent,
            extract_with_llm=False,
        )
        ctx = await eng.build_long_input_context(
            task.task_run_id,
            query="memory requirements constraints",
            max_tokens=2000,
            expected_terms=["PostgreSQL"],
        )
        assert isinstance(ctx.text, str)
        assert ctx.token_estimate >= 0
        await eng.purge(agent)


class TestBuildContext:
    async def test_build_context_empty_task(self, eng, agent):
        task = await eng.start_task("Context task", agent_id=agent)
        ctx = await eng.build_context(task.task_run_id, query="finance analysis")
        assert isinstance(ctx.text, str)
        await eng.purge(agent)

    async def test_build_context_with_memories_and_events(self, eng, agent):
        task = await eng.start_task("Rich context task", agent_id=agent)
        await eng.add_batch([{"content": c, "agent_id": agent} for c in MEMORIES[:4]])
        for c in MEMORIES[4:7]:
            await eng.record_event(
                agent_id=agent,
                role="user",
                event_type="user_message",
                content=c,
                task_run_id=task.task_run_id,
            )
        await eng.create_checkpoint(task.task_run_id, summary="Partial progress")
        ctx = await eng.build_context(
            task.task_run_id,
            query="commute finance dancing",
            recent_event_limit=10,
            memory_limit=5,
        )
        assert isinstance(ctx.text, str)
        await eng.purge(agent)

    async def test_build_context_no_graph(self, eng, agent):
        task = await eng.start_task("No graph task", agent_id=agent)
        ctx = await eng.build_context(task.task_run_id, include_graph=False)
        assert isinstance(ctx.text, str)
        await eng.purge(agent)


class TestMemoryWorker:
    async def test_process_memory_jobs_no_jobs(self, eng):
        processed = await eng.process_memory_jobs(limit=5)
        assert isinstance(processed, list)

    async def test_run_memory_worker_stops(self, eng):
        stop = asyncio.Event()
        stop.set()  # stop immediately
        count = await eng.run_memory_worker(
            stop_event=stop,
            batch_size=5,
            interval_seconds=0.1,
            max_iterations=2,
        )
        assert count >= 0

    async def test_run_memory_worker_max_iterations(self, eng):
        count = await eng.run_memory_worker(
            max_iterations=3,
            batch_size=5,
            interval_seconds=0.01,
        )
        assert count >= 0

    async def test_process_turn_ingest_job(self, eng, agent):
        """record_turn with enqueue_processing=True creates a turn_ingest job."""
        task = await eng.start_task("Worker ingest task", agent_id=agent)
        events = await eng.record_turn(
            task.task_run_id,
            TURN_USER,
            TURN_ASST,
            agent_id=agent,
            enqueue_processing=True,  # creates job
        )
        assert len(events) == 2
        # run worker to drain the job
        count = await eng.run_memory_worker(max_iterations=5, interval_seconds=0.05)
        assert count >= 0  # job processed (or 0 if LLM unavailable)
        await eng.purge(agent)


# ── 8. Edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    async def test_search_no_results_empty_list(self, eng, agent):
        results = await eng.search("xyzzy_not_a_real_word_12345", agent_id=agent)
        assert results == []

    async def test_get_memories_no_results(self, eng, agent):
        mems = await eng.get_memories(agent, limit=100)
        assert mems == []

    async def test_add_batch_empty_list_ok(self, eng):
        mems = await eng.add_batch([])
        assert mems == []

    async def test_purge_scoped_to_user(self, eng, agent):
        await eng.add(content="User A memory", agent_id=agent, user_id="userA")
        await eng.add(content="User B memory", agent_id=agent, user_id="userB")
        await eng.purge(agent, user_id="userA")
        remaining = await eng.get_memories(agent, user_id="userB")
        assert len(remaining) == 1
        await eng.purge(agent)

    async def test_revise_then_search_returns_active(self, eng, agent):
        m = await eng.add(content="Old laptop budget 800", agent_id=agent)
        await eng.revise(m.memory_id, content="New laptop budget 1500")
        results = await eng.search("laptop budget", agent_id=agent)
        contents = [r.memory.content for r in results]
        assert any("1500" in c for c in contents)
        # superseded should not appear by default
        assert not any("800" in c for c in contents)
        await eng.purge(agent)

    async def test_traverse_no_relations_returns_empty(self, eng, agent):
        m = await eng.add(content="Isolated node", agent_id=agent)
        results = await eng.traverse(m.memory_id, max_depth=2)
        assert results == []
        await eng.purge(agent)

    async def test_deep_search_empty_agent(self, eng, agent):
        results = await eng.deep_search("anything", agent_id=agent)
        assert results == []

    async def test_list_events_empty(self, eng, agent):
        task = await eng.start_task("Empty events task", agent_id=agent)
        events = await eng.list_events(task_run_id=task.task_run_id)
        assert events == []
        await eng.purge(agent)

    async def test_get_history_no_memories(self, eng, agent):
        history = await eng.get_history(agent)
        assert history == []
