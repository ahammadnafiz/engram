"""Unit tests for LongMemEval harness reading helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
async def test_answer_generation_uses_public_engram_reader_api() -> None:
    class FakeEngram:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def answer_from_evidence(self, **kwargs):
            self.calls.append(kwargs)
            return "Target"

    engram = FakeEngram()
    out = await harness.maybe_generate_answer(
        engram,
        question="Where did I redeem a coupon?",
        question_date="2023/05/30",
        context="USER: I use Cartwheel from Target.\nUSER: I redeemed a coupon.",
        max_tokens=256,
        reading="con",
    )

    assert out == "Target"
    assert engram.calls == [
        {
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
