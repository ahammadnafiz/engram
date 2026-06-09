"""Unit tests for Engram client helpers: get_context_block + add_conversation summary."""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock

import pytest


def make_engram():
    """Engram instance with connected state and mocked internals."""
    from engram.client import Engram

    eg = Engram()
    eg._connected = True
    eg._memory_store = AsyncMock()
    eg._sessions = AsyncMock()
    eg._llm = AsyncMock()
    eg._embedding = MagicMock()
    return eg


def sr(content: str, score: float, memory_type: str = "semantic"):
    from engram.memory.models import Memory, SearchResult

    return SearchResult(
        memory=Memory(agent_id="a", content=content, memory_type=memory_type),
        score=score,
    )


def mem(content: str, mid: str, memory_type: str = "semantic", metadata=None):
    from engram.memory.models import Memory

    return Memory(
        memory_id=mid,
        agent_id="agent",
        content=content,
        memory_type=memory_type,
        metadata=metadata or {},
    )


def session(summary: str | None):
    from engram.session.models import Session

    return Session(agent_id="a", summary=summary)


class TestGetContextBlock:
    @pytest.mark.asyncio
    async def test_empty_returns_empty_string(self) -> None:
        eg = make_engram()
        eg._memory_store.search = AsyncMock(return_value=[])
        assert await eg.get_context_block("q", "agent") == ""

    @pytest.mark.asyncio
    async def test_renders_ordered_bullets(self) -> None:
        eg = make_engram()
        eg._memory_store.search = AsyncMock(
            return_value=[sr("Fact A", 0.9), sr("Fact B", 0.5)]
        )
        out = await eg.get_context_block("q", "agent")
        assert out == "## Relevant memories\n- Fact A\n- Fact B"

    @pytest.mark.asyncio
    async def test_token_budget_truncates(self) -> None:
        eg = make_engram()
        eg._memory_store.search = AsyncMock(
            return_value=[sr("x" * 40, 0.9), sr("y" * 40, 0.5)]
        )
        # heuristic = len // 4; each "- " + 40 chars ~ 10 tokens. Budget 12 keeps one.
        out = await eg.get_context_block("q", "agent", max_tokens=12)
        assert "x" * 40 in out
        assert "y" * 40 not in out

    @pytest.mark.asyncio
    async def test_session_summary_prepended(self) -> None:
        eg = make_engram()
        eg._memory_store.search = AsyncMock(return_value=[sr("Fact A", 0.9)])
        eg._sessions.get = AsyncMock(return_value=session("Talked about pizza."))
        out = await eg.get_context_block("q", "agent", session_id="s1")
        assert out.startswith("## Conversation summary\nTalked about pizza.")
        assert "## Relevant memories\n- Fact A" in out

    @pytest.mark.asyncio
    async def test_group_by_type_renders_sections(self) -> None:
        eg = make_engram()
        eg._memory_store.search = AsyncMock(
            return_value=[
                sr("User likes jazz", 0.9, "semantic"),
                sr("User went to a concert", 0.8, "episodic"),
            ]
        )
        out = await eg.get_context_block("q", "agent", group_by_type=True)
        assert "## Semantic — user facts\n- User likes jazz" in out
        assert "## Episodic — events\n- User went to a concert" in out


class TestAddConversationSummary:
    def _wire(self, eg, stored_summary):
        from engram.llm.service import ExtractionResult

        eg._memory_store.search = AsyncMock(return_value=[])
        eg._sessions.get = AsyncMock(return_value=session(stored_summary))
        eg._llm.process_for_memory = AsyncMock(
            return_value=ExtractionResult(operations=[])
        )
        eg._llm.update_conversation_summary = AsyncMock(return_value="rolled")
        eg._sessions.update_summary = AsyncMock()

    @pytest.mark.asyncio
    async def test_loads_and_persists_session_summary(self) -> None:
        eg = make_engram()
        self._wire(eg, "prev")
        await eg.add_conversation("hi", "hello", "agent", session_id="s1")

        assert (
            eg._llm.process_for_memory.call_args.kwargs["conversation_summary"]
            == "prev"
        )
        eg._llm.update_conversation_summary.assert_awaited_once_with(
            "prev", "hi", "hello"
        )
        eg._sessions.update_summary.assert_awaited_once_with("s1", "rolled")

    @pytest.mark.asyncio
    async def test_update_summary_false_skips_roll(self) -> None:
        eg = make_engram()
        self._wire(eg, "prev")
        await eg.add_conversation(
            "hi", "hello", "agent", session_id="s1", update_summary=False
        )
        eg._llm.update_conversation_summary.assert_not_awaited()
        eg._sessions.update_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_summary_overrides_stored(self) -> None:
        eg = make_engram()
        self._wire(eg, "stored")
        await eg.add_conversation(
            "hi", "hello", "agent", session_id="s1", conversation_summary="explicit"
        )
        assert (
            eg._llm.process_for_memory.call_args.kwargs["conversation_summary"]
            == "explicit"
        )
        eg._sessions.get.assert_not_awaited()  # explicit provided, no load


class TestDeepSearch:
    @pytest.mark.asyncio
    async def test_merges_and_dedupes_by_id(self) -> None:
        from engram.memory.models import Memory, SearchResult

        def sr_id(content: str, score: float, mid: str) -> SearchResult:
            return SearchResult(
                memory=Memory(memory_id=mid, agent_id="a", content=content), score=score
            )

        eg = make_engram()
        eg._llm.expand_query = AsyncMock(return_value=["q2", "q3"])

        async def fake_search(q: str, agent_id: str, **kw):
            if q == "orig":
                return [sr_id("A", 0.5, "m1")]
            if q == "q2":
                return [sr_id("A", 0.8, "m1"), sr_id("B", 0.4, "m2")]
            return [sr_id("C", 0.6, "m3")]

        eg.search = AsyncMock(side_effect=fake_search)
        out = await eg.deep_search("orig", "agent", limit=10)

        ids = [r.memory.memory_id for r in out]
        assert ids == ["m1", "m3", "m2"]  # m1 deduped at higher score, sorted desc
        assert out[0].score == 0.8

    @pytest.mark.asyncio
    async def test_no_llm_falls_back_to_single_search(self) -> None:
        eg = make_engram()
        eg._llm = None
        eg.search = AsyncMock(return_value=[])
        await eg.deep_search("q", "agent")
        eg.search.assert_awaited_once()


class TestPolicyRecall:
    @pytest.mark.asyncio
    async def test_add_attaches_critical_slot_and_supersedes_conflicts(self) -> None:
        from engram.memory.models import Memory

        eg = make_engram()
        eg._memory_store.add = AsyncMock(
            return_value=Memory(
                memory_id="m_new",
                agent_id="agent",
                content="User is allergic to cashews",
                memory_type="profile",
                metadata={
                    "critical": True,
                    "critical_slot": "profile:allergy:cashews",
                    "conflict_key": "agent:user:profile:allergy:cashews",
                },
            )
        )
        eg._memory_store.supersede_conflicts = AsyncMock(return_value=1)

        out = await eg.add("User is allergic to cashews", "agent", user_id="user")

        create = eg._memory_store.add.call_args.args[0]
        assert create.memory_type == "profile"
        assert create.metadata["critical"] is True
        assert create.metadata["critical_slot"] == "profile:allergy:cashews"
        eg._memory_store.supersede_conflicts.assert_awaited_once_with(
            agent_id="agent",
            user_id="user",
            conflict_key="agent:user:profile:allergy:cashews",
            winner_memory_id="m_new",
        )
        assert out.memory_id == "m_new"

    @pytest.mark.asyncio
    async def test_custom_memory_policy_controls_type_and_slot(self) -> None:
        from engram import MemoryPolicy, SlotRule, TypeRule
        from engram.client import Engram
        from engram.memory.models import Memory

        policy = MemoryPolicy(
            name="custom_sales",
            type_rules=(TypeRule("project", (r"\baccount\b",)),),
            slot_rules=(SlotRule("sales:account_owner", (r"\baccount owner\b",)),),
        )
        eg = Engram(memory_policy=policy)
        eg._connected = True
        eg._memory_store = AsyncMock()
        eg._memory_store.add = AsyncMock(
            return_value=Memory(
                memory_id="m_sales",
                agent_id="agent",
                content="The account owner is Rina.",
                memory_type="project",
                metadata={
                    "critical": True,
                    "critical_slot": "sales:account_owner",
                    "conflict_key": "agent:user:sales:account_owner",
                },
            )
        )
        eg._memory_store.supersede_conflicts = AsyncMock(return_value=0)

        await eg.add("The account owner is Rina.", "agent", user_id="user")

        create = eg._memory_store.add.call_args.args[0]
        assert create.memory_type == "project"
        assert create.metadata["critical_slot"] == "sales:account_owner"
        eg._memory_store.supersede_conflicts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trace_recall_shows_critical_kept_and_superseded(self) -> None:
        eg = make_engram()
        critical = mem(
            "User is allergic to cashews",
            "m_critical",
            "profile",
            {"critical": True, "critical_slot": "profile:allergy"},
        )
        superseded = mem(
            "User is allergic to almonds",
            "m_old",
            "profile",
            {"critical": True, "status": "superseded", "superseded_by": "m_critical"},
        )
        eg._memory_store.list_policy_memories = AsyncMock(
            side_effect=[[critical], [critical, superseded]]
        )
        eg.deep_search = AsyncMock(
            return_value=[sr("Atlas Checkout rollback owner is Priya", 0.9, "project")]
        )

        trace = await eg.trace_recall(
            "final verification",
            "agent",
            user_id="user",
            expected_terms=["cashews", "Priya"],
            max_tokens=200,
        )

        assert trace.critical_memory_ids == ["m_critical"]
        assert trace.search_memory_ids
        assert trace.kept_memory_ids == ["m_critical", *trace.search_memory_ids]
        assert trace.superseded_memory_ids == ["m_old"]
        assert trace.missing_expected_terms == []
        assert "profile:allergy" in trace.context


class TestGraphPromptHelpers:
    @pytest.mark.asyncio
    async def test_traverse_many_delegates_to_graph(self) -> None:
        eg = make_engram()
        eg._graph = AsyncMock()
        eg._graph.traverse_many = AsyncMock(return_value=[])

        out = await eg.traverse_many(["m1", "m2"], max_depth=2, direction="any")

        assert out == []
        eg._graph.traverse_many.assert_awaited_once_with(
            ["m1", "m2"],
            max_depth=2,
            direction="any",
            relation_types=None,
            min_weight=0.0,
            limit_per_seed=25,
            total_limit=100,
            skip_missing=True,
        )

    def test_render_graph_context_delegates_to_graph(self) -> None:
        eg = make_engram()
        eg._graph = MagicMock()
        eg._graph.render_context.return_value = "## graph"

        out = eg.render_graph_context([], max_tokens=100)

        assert out == "## graph"
        eg._graph.render_context.assert_called_once()


class TestAddConversationGraphPreserving:
    @pytest.mark.asyncio
    async def test_contradiction_updates_in_place_not_delete_add(self) -> None:
        from engram.llm.service import (
            ExtractionResult,
            MemoryOperation,
            MemoryOperationType,
        )
        from engram.memory.models import Memory

        eg = make_engram()
        eg._memory_store.search = AsyncMock(return_value=[])
        eg._llm.process_for_memory = AsyncMock(
            return_value=ExtractionResult(
                operations=[
                    MemoryOperation(
                        operation=MemoryOperationType.DELETE,
                        content="User lives in Singapore",
                        target_id="old_1",
                    )
                ]
            )
        )
        eg.update = AsyncMock(
            return_value=Memory(
                memory_id="old_1", agent_id="a", content="User lives in Singapore"
            )
        )
        eg.forget = AsyncMock()
        eg.add = AsyncMock()

        out = await eg.add_conversation("moved", "ok", "agent")

        # Contradiction updates the existing row in place (id + relations survive)
        eg.update.assert_awaited_once_with(
            "old_1",
            content="User lives in Singapore",
            metadata=ANY,
        )
        eg.forget.assert_not_awaited()
        eg.add.assert_not_awaited()
        assert out[0].memory_id == "old_1"
        # per-fact retrieval callback is wired
        assert (
            eg._llm.process_for_memory.call_args.kwargs["retrieve_for_fact"] is not None
        )
