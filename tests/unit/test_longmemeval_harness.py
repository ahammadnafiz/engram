"""Unit tests for LongMemEval harness reading helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_harness():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "longmemeval_harness.py"
    spec = importlib.util.spec_from_file_location("longmemeval_harness", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _gold_sample() -> dict:
    return {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "what?",
        "answer": "blue",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/01/01", "2023/01/02"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "the answer is blue", "has_answer": True},
                {"role": "assistant", "content": "noted"},
            ],
            [{"role": "user", "content": "unrelated chatter"}],
        ],
    }


def test_has_answer_gold_label_not_stored_in_memory_metadata() -> None:
    """The gold answer label must never live in the retrievable store."""
    mems = harness.iter_memories(
        _gold_sample(), agent_id="a", memory_unit="turn", max_memory_chars=2000
    )
    assert mems  # sanity
    assert all("has_answer" not in m["metadata"] for m in mems)


def test_any_answer_memory_retrieved_recomputed_from_sample() -> None:
    """The evidence-recall metric is derived from the raw sample at scoring
    time, not from any stored flag — matching gold turns by (session, turn)."""
    sample = _gold_sample()
    gold_hit = {"metadata": {"original_session_id": "s1", "turn_index": 0}}
    non_gold = {"metadata": {"original_session_id": "s2", "turn_index": 0}}

    hit = harness.recall_metrics(sample, [gold_hit], context="", hypothesis="")
    miss = harness.recall_metrics(sample, [non_gold], context="", hypothesis="")
    assert hit["any_answer_memory_retrieved"] is True
    assert miss["any_answer_memory_retrieved"] is False


def test_exact_answer_metric_uses_token_sequence_not_substring() -> None:
    sample = _gold_sample() | {"answer": "Target"}
    retrieved = [{"metadata": {"original_session_id": "s1", "turn_index": 0}}]

    false_positive = harness.recall_metrics(
        sample,
        retrieved,
        context="Optimize ad targeting for social campaigns.",
        hypothesis="targeting",
    )
    true_positive = harness.recall_metrics(
        sample,
        retrieved,
        context="The coupon was redeemed at Target.",
        hypothesis="Target.",
    )

    assert false_positive["answer_exact_in_context"] is False
    assert false_positive["answer_exact_in_hypothesis"] is False
    assert true_positive["answer_exact_in_context"] is True
    assert true_positive["answer_exact_in_hypothesis"] is True


@pytest.mark.asyncio
async def test_answer_generation_delegates_to_reader(monkeypatch) -> None:
    """maybe_generate_answer forwards to the harness answer_from_evidence
    reader, passing the engram and the question/context/reading kwargs."""
    calls: list[dict] = []

    async def fake_answer_from_evidence(engram, **kwargs):
        calls.append({"engram": engram, **kwargs})
        return "Target"

    monkeypatch.setattr(harness, "answer_from_evidence", fake_answer_from_evidence)

    sentinel = object()
    out = await harness.maybe_generate_answer(
        sentinel,
        question="Where did I redeem a coupon?",
        question_date="2023/05/30",
        context="USER: I use Cartwheel from Target.\nUSER: I redeemed a coupon.",
        max_tokens=256,
        reading="con",
    )

    assert out == "Target"
    assert calls == [
        {
            "engram": sentinel,
            "question": "Where did I redeem a coupon?",
            "question_date": "2023/05/30",
            "context": "USER: I use Cartwheel from Target.\nUSER: I redeemed a coupon.",
            "max_tokens": 256,
            "reading": "con",
        }
    ]


def _completed_trace(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "error": None,
        "metrics": {
            "answer_session_recall": 1.0,
            "all_answer_sessions_recalled": True,
            "any_answer_memory_retrieved": True,
            "answer_exact_in_context": True,
            "answer_exact_in_hypothesis": True,
            "answer_word_coverage_in_context": 1.0,
            "answer_word_coverage_in_hypothesis": 1.0,
        },
    }


def _errored_trace(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "error": "StorageError: boom",
        "metrics": {},
    }


def test_summarize_excludes_errored_from_metrics_and_lists_their_ids() -> None:
    """Errored samples must not drag down accuracy (they never ran) and must be
    surfaced by id so a partial run is never mistaken for a complete one."""
    summary = harness.summarize([_completed_trace("ok-1"), _errored_trace("infra-2")])

    assert summary["total"] == 2
    assert summary["completed"] == 1
    assert summary["errors"] == 1
    assert summary["errored_question_ids"] == ["infra-2"]
    # Metrics are averaged over completed only — the errored sample is absent,
    # so a single good answer reads as a perfect rate rather than 0.5.
    assert summary["answer_exact_in_hypothesis_rate"] == 1.0
    assert summary["answer_session_recall"] == 1.0


def _ns(**kw):
    import argparse

    defaults = {
        "data_path": "data/longmemeval/longmemeval_s_cleaned.json",
        "sample": None,
        "question_types": None,
        "memory_unit": "turn",
        "max_memory_chars": 2000,
    }
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_store_fingerprint_stable_for_same_settings() -> None:
    # Reuse only saves money if the key is stable across identical runs.
    assert harness.store_fingerprint(_ns()) == harness.store_fingerprint(_ns())


def test_store_fingerprint_changes_with_ingestion_settings() -> None:
    # The riskiest failure mode is silently serving a stale store. The
    # fingerprint must change whenever what gets ingested changes.
    base = harness.store_fingerprint(_ns())
    assert harness.store_fingerprint(_ns(memory_unit="session")) != base
    assert harness.store_fingerprint(_ns(max_memory_chars=1000)) != base
    assert (
        harness.store_fingerprint(_ns(data_path="data/longmemeval/other.json")) != base
    )


# ---------------------------------------------------------------------------
# Reader/evidence helpers relocated from the Engram public API into this
# harness. The tests below were moved verbatim from tests/unit/test_client.py
# and now exercise the harness module functions, driving a mocked Engram
# through its public API only.
# ---------------------------------------------------------------------------


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


def sr(
    content: str,
    score: float,
    memory_type: str = "semantic",
    metadata=None,
    memory_id: str | None = None,
    session_id: str | None = None,
):
    from engram.memory.models import Memory, SearchResult

    return SearchResult(
        memory=Memory(
            memory_id=memory_id or f"mem_{abs(hash(content))}",
            agent_id="a",
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            session_id=session_id,
        ),
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


class TestNeighboringContextBlock:
    @pytest.mark.asyncio
    async def test_expands_retrieved_turn_with_metadata_neighbors(self) -> None:
        eg = make_engram()
        group = [
            mem(
                "[2023/05/29] USER: I use Cartwheel at Target.",
                "m0",
                "episodic",
                {
                    "original_session_id": "s1",
                    "turn_index": 0,
                    "haystack_date": "2023/05/29",
                },
            ),
            mem(
                "[2023/05/29] USER: I redeemed a coupon on coffee creamer.",
                "m1",
                "episodic",
                {
                    "original_session_id": "s1",
                    "turn_index": 1,
                    "haystack_date": "2023/05/29",
                    "has_answer": True,
                },
            ),
            mem(
                "[2023/05/29] ASSISTANT: That coupon came from email.",
                "m2",
                "episodic",
                {
                    "original_session_id": "s1",
                    "turn_index": 2,
                    "haystack_date": "2023/05/29",
                },
            ),
        ]
        eg._memory_store.list_memories = AsyncMock(return_value=group)
        result = sr(
            "[2023/05/29] USER: I redeemed a coupon on coffee creamer.",
            0.9,
            "episodic",
            metadata={
                "original_session_id": "s1",
                "turn_index": 1,
                "haystack_date": "2023/05/29",
                "has_answer": True,
            },
            memory_id="m1",
        )

        context, sources = await harness.get_neighboring_context_block(
            eg,
            [result],
            "agent",
            before=1,
            after=1,
            memory_types=["episodic"],
        )

        assert "Target" in context
        assert "coffee creamer" in context
        assert "email" in context
        assert [source["memory_id"] for source in sources] == ["m0", "m1", "m2"]
        eg._memory_store.list_memories.assert_awaited_once_with(
            "agent",
            user_id=None,
            session_id=None,
            metadata_filter={"original_session_id": "s1"},
            memory_types=["episodic"],
            limit=200,
        )

    @pytest.mark.asyncio
    async def test_uses_real_session_id_when_present(self) -> None:
        eg = make_engram()
        group = [
            mem(
                "USER: session opening",
                "m0",
                "episodic",
                {"turn_index": 0},
            ).model_copy(update={"session_id": "real-session"}),
            mem(
                "USER: exact retrieved fact",
                "m1",
                "episodic",
                {"turn_index": 1},
            ).model_copy(update={"session_id": "real-session"}),
        ]
        eg._memory_store.list_memories = AsyncMock(return_value=group)
        result = sr(
            "USER: exact retrieved fact",
            0.9,
            "episodic",
            metadata={"turn_index": 1},
            memory_id="m1",
            session_id="real-session",
        )

        context, _sources = await harness.get_neighboring_context_block(
            eg,
            [result],
            "agent",
            before=1,
            after=0,
            memory_types=["episodic"],
        )

        assert "session opening" in context
        eg._memory_store.list_memories.assert_awaited_once_with(
            "agent",
            user_id=None,
            session_id="real-session",
            metadata_filter=None,
            memory_types=["episodic"],
            limit=200,
        )

    @pytest.mark.asyncio
    async def test_budget_keeps_exact_retrieved_turn_before_neighbor_filler(
        self,
    ) -> None:
        eg = make_engram()
        group = [
            mem(
                "Long filler " * 200,
                "m0",
                "episodic",
                {"original_session_id": "s1", "turn_index": 0},
            ),
            mem(
                "The answer amount was $25.",
                "m1",
                "episodic",
                {"original_session_id": "s1", "turn_index": 1, "has_answer": True},
            ),
        ]
        eg._memory_store.list_memories = AsyncMock(return_value=group)
        result = sr(
            "The answer amount was $25.",
            0.9,
            "episodic",
            metadata={"original_session_id": "s1", "turn_index": 1},
            memory_id="m1",
        )

        context, sources = await harness.get_neighboring_context_block(
            eg,
            [result],
            "agent",
            before=1,
            after=0,
            max_tokens=12,
            memory_types=["episodic"],
        )

        assert "$25" in context
        assert [source["memory_id"] for source in sources] == ["m1"]

    @pytest.mark.asyncio
    async def test_prior_user_turns_preserve_linked_session_evidence(self) -> None:
        eg = make_engram()
        group = [
            mem(
                "[2023/05/29] USER: I organize coupons in a binder.",
                "m0",
                "episodic",
                {"original_session_id": "s1", "turn_index": 0, "turn_role": "user"},
            ),
            mem(
                "[2023/05/29] ASSISTANT: A binder is useful.",
                "m1",
                "episodic",
                {
                    "original_session_id": "s1",
                    "turn_index": 1,
                    "turn_role": "assistant",
                },
            ),
            mem(
                "[2023/05/29] USER: I use the Cartwheel app from Target.",
                "m2",
                "episodic",
                {"original_session_id": "s1", "turn_index": 2, "turn_role": "user"},
            ),
            mem(
                "[2023/05/29] ASSISTANT: Target often has offers.",
                "m3",
                "episodic",
                {
                    "original_session_id": "s1",
                    "turn_index": 3,
                    "turn_role": "assistant",
                },
            ),
            mem(
                "[2023/05/29] USER: I redeemed a $5 coupon on coffee creamer.",
                "m4",
                "episodic",
                {"original_session_id": "s1", "turn_index": 4, "turn_role": "user"},
            ),
        ]
        eg._memory_store.list_memories = AsyncMock(return_value=group)
        result = sr(
            "[2023/05/29] USER: I redeemed a $5 coupon on coffee creamer.",
            0.9,
            "episodic",
            metadata={"original_session_id": "s1", "turn_index": 4},
            memory_id="m4",
        )

        context, sources = await harness.get_neighboring_context_block(
            eg,
            [result],
            "agent",
            before=0,
            after=0,
            prior_user_turns=1,
            context_order="relevance",
            memory_types=["episodic"],
        )

        assert "Cartwheel app from Target" in context
        assert "coffee creamer" in context
        assert [source["memory_id"] for source in sources] == ["m2", "m4"]

    @pytest.mark.asyncio
    async def test_relevance_order_puts_top_retrieved_group_first(self) -> None:
        eg = make_engram()
        answer_group = [
            mem(
                "[2023/05/30] USER: Answer group opener.",
                "answer0",
                "episodic",
                {
                    "original_session_id": "answer",
                    "turn_index": 0,
                    "haystack_date": "2023/05/30",
                },
            ),
            mem(
                "[2023/05/30] USER: Exact answer.",
                "answer1",
                "episodic",
                {
                    "original_session_id": "answer",
                    "turn_index": 1,
                    "haystack_date": "2023/05/30",
                },
            ),
        ]
        old_group = [
            mem(
                "[2023/05/01] USER: Older distractor.",
                "old0",
                "episodic",
                {
                    "original_session_id": "old",
                    "turn_index": 0,
                    "haystack_date": "2023/05/01",
                },
            )
        ]

        async def fake_list_memories(*_args, **kwargs):
            if kwargs.get("metadata_filter") == {"original_session_id": "answer"}:
                return answer_group
            return old_group

        eg._memory_store.list_memories = AsyncMock(side_effect=fake_list_memories)
        results = [
            sr(
                "[2023/05/30] USER: Exact answer.",
                0.9,
                "episodic",
                metadata={
                    "original_session_id": "answer",
                    "turn_index": 1,
                    "haystack_date": "2023/05/30",
                },
                memory_id="answer1",
            ),
            sr(
                "[2023/05/01] USER: Older distractor.",
                0.8,
                "episodic",
                metadata={
                    "original_session_id": "old",
                    "turn_index": 0,
                    "haystack_date": "2023/05/01",
                },
                memory_id="old0",
            ),
        ]

        context, sources = await harness.get_neighboring_context_block(
            eg,
            results,
            "agent",
            before=1,
            after=0,
            context_order="relevance",
            memory_types=["episodic"],
        )

        assert context.startswith("[2023/05/30]")
        assert [source["memory_id"] for source in sources[:2]] == [
            "answer0",
            "answer1",
        ]


class TestSearchEvidenceSet:
    @pytest.mark.asyncio
    async def test_overfetches_and_round_robins_by_group(self) -> None:
        eg = make_engram()
        eg.deep_search = AsyncMock(
            return_value=[
                sr(
                    "session 1 top hit",
                    0.95,
                    metadata={"original_session_id": "s1"},
                    memory_id="s1a",
                ),
                sr(
                    "session 1 second hit",
                    0.94,
                    metadata={"original_session_id": "s1"},
                    memory_id="s1b",
                ),
                sr(
                    "session 2 hit",
                    0.80,
                    metadata={"original_session_id": "s2"},
                    memory_id="s2a",
                ),
                sr(
                    "session 3 hit",
                    0.70,
                    metadata={"original_session_id": "s3"},
                    memory_id="s3a",
                ),
            ]
        )

        out = await harness.search_evidence_set(
            eg,
            "q",
            "agent",
            limit=3,
            candidate_limit=4,
            rerank=False,
            max_per_group=1,
        )

        assert [r.memory.memory_id for r in out] == ["s1a", "s2a", "s3a"]
        eg.deep_search.assert_awaited_once_with(
            "q",
            "agent",
            user_id=None,
            limit=4,
            min_score=0.0,
            metadata_filter=None,
            memory_types=None,
            mode="hybrid",
            rerank=False,
        )

    @pytest.mark.asyncio
    async def test_preferred_role_reorders_nearby_turns(self) -> None:
        eg = make_engram()
        eg.deep_search = AsyncMock(
            return_value=[
                sr(
                    "ASSISTANT: generic response",
                    0.95,
                    metadata={"original_session_id": "a", "turn_role": "assistant"},
                    memory_id="assistant",
                ),
                sr(
                    "USER: exact preference",
                    0.94,
                    metadata={"original_session_id": "u", "turn_role": "user"},
                    memory_id="user",
                ),
            ]
        )

        out = await harness.search_evidence_set(
            eg,
            "q",
            "agent",
            limit=2,
            candidate_limit=2,
            rerank=False,
            preferred_role="user",
        )

        assert [r.memory.memory_id for r in out] == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_can_use_single_query_search_as_candidate_source(self) -> None:
        eg = make_engram()
        eg.search = AsyncMock(return_value=[sr("Fact", 0.9, memory_id="m1")])

        out = await harness.search_evidence_set(
            eg,
            "q",
            "agent",
            limit=1,
            candidate_limit=5,
            use_deep_search=False,
            rerank=False,
        )

        assert [r.memory.memory_id for r in out] == ["m1"]
        eg.search.assert_awaited_once_with(
            "q",
            "agent",
            user_id=None,
            limit=5,
            min_score=0.0,
            metadata_filter=None,
            memory_types=None,
            mode="hybrid",
            rerank=False,
        )


class TestAnswerFromEvidence:
    @pytest.mark.asyncio
    async def test_chain_of_note_verifies_exact_question_slot(self) -> None:
        eg = make_engram()
        eg._llm.complete = AsyncMock(
            side_effect=[
                "notes",
                (
                    '{"operation":"single","answer":"Target","insufficient":false,'
                    '"included":[{"entity":"Target"}],"excluded":[]}'
                ),
                (
                    '{"operation":"single","answer":"Target","insufficient":false,'
                    '"included":[{"entity":"Target"}],"excluded":[]}'
                ),
                "Target",
            ]
        )

        out = await harness.answer_from_evidence(
            eg,
            question="Where did I redeem a $5 coupon on coffee creamer?",
            question_date="2023/05/30",
            context=(
                "USER: I use the Cartwheel app from Target.\n"
                "USER: I redeemed a $5 coupon on coffee creamer."
            ),
            max_tokens=256,
            reading="con",
        )

        assert out == "Target"
        assert eg._llm.complete.await_count == 4
        notes_prompt = eg._llm.complete.await_args_list[0].args[0]
        ledger_prompt = eg._llm.complete.await_args_list[1].args[0]
        verifier_prompt = eg._llm.complete.await_args_list[2].args[0]
        final_prompt = eg._llm.complete.await_args_list[3].args[0]
        assert "candidate answers for the exact requested slot" in notes_prompt
        assert "wrong-slot facts to avoid" in notes_prompt
        assert "do not stop after the first match" in notes_prompt
        assert "<memory_context>" in notes_prompt
        assert "Build an aggregation ledger" in ledger_prompt
        assert "Return JSON only" in ledger_prompt
        assert "one row per candidate evidence item" in ledger_prompt
        assert "Deduplicate only true duplicates" in ledger_prompt
        assert "computed from the included rows" in ledger_prompt
        assert "superseded by a later update" in ledger_prompt
        assert "Audit and correct the ledger" in verifier_prompt
        assert "Merge only exact duplicates" in verifier_prompt
        assert "recency rule" in verifier_prompt
        assert "Answer strictly from the included rows" in final_prompt
        assert "<aggregation_ledger>" in final_prompt
        assert eg._llm.complete.await_args_list[0].kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_chain_of_note_count_question_reaches_final_llm(self) -> None:
        # The deterministic reducer is gone: the verified ledger is handed to a
        # final LLM call that is instructed to count the included rows itself.
        eg = make_engram()
        verified = (
            '{"operation":"count","insufficient":false,'
            '"included":[{"entity":"boots"},{"entity":"blazer"},{"entity":"coat"}],'
            '"excluded":[]}'
        )
        eg._llm.complete = AsyncMock(side_effect=["notes", verified, verified, "3"])

        out = await harness.answer_from_evidence(
            eg,
            question="How many things do I need to pick up?",
            context="ctx",
            reading="con",
        )

        assert out == "3"
        assert eg._llm.complete.await_count == 4
        final_prompt = eg._llm.complete.await_args_list[3].args[0]
        assert "count the included rows" in final_prompt
        assert '"entity": "coat"' in final_prompt

    @pytest.mark.asyncio
    async def test_chain_of_note_sum_question_reaches_final_llm(self) -> None:
        eg = make_engram()
        verified = (
            '{"operation":"sum","insufficient":false,'
            '"included":[{"amount":"$25"},{"amount":"$40"},{"amount":"$120"}],'
            '"excluded":[]}'
        )
        eg._llm.complete = AsyncMock(side_effect=["notes", verified, verified, "$185"])

        out = await harness.answer_from_evidence(
            eg,
            question="How much did I spend total?",
            context="ctx",
            reading="con",
        )

        assert out == "$185"
        assert eg._llm.complete.await_count == 4
        final_prompt = eg._llm.complete.await_args_list[3].args[0]
        assert "add the amounts on the included rows" in final_prompt

    @pytest.mark.asyncio
    async def test_chain_of_note_falls_back_when_ledger_is_not_json(self) -> None:
        eg = make_engram()
        eg._llm.complete = AsyncMock(side_effect=["notes", "not json", "Target"])

        out = await harness.answer_from_evidence(
            eg,
            question="Where did I redeem a coupon?",
            context="ctx",
            reading="con",
        )

        assert out == "Target"
        assert eg._llm.complete.await_count == 3
        final_prompt = eg._llm.complete.await_args_list[2].args[0]
        assert "Answer strictly from the included rows" in final_prompt

    @pytest.mark.asyncio
    async def test_direct_reader_also_uses_slot_rules(self) -> None:
        eg = make_engram()
        eg._llm.complete = AsyncMock(return_value="Target")

        out = await harness.answer_from_evidence(
            eg,
            question="Where did I redeem a coupon?",
            context="USER: I use Cartwheel from Target.",
            reading="direct",
        )

        assert out == "Target"
        prompt = eg._llm.complete.await_args.args[0]
        assert "exact slot requested" in prompt
        assert "merchant" in prompt

    @pytest.mark.asyncio
    async def test_no_llm_returns_empty_answer(self) -> None:
        eg = make_engram()
        eg._llm = None

        assert (
            await harness.answer_from_evidence(
                eg,
                question="q",
                context="ctx",
            )
            == ""
        )
