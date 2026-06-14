#!/usr/bin/env python3
"""Run LongMemEval end-to-end through Engram.

The harness ingests real LongMemEval histories into Engram, retrieves memories
for each question, and writes JSONL traces plus aggregate recall metrics.

Examples:
    python scripts/longmemeval_harness.py \
        --data-path data/longmemeval_oracle.json \
        --max-samples 10 \
        --output-dir runs/longmemeval-oracle

    python scripts/longmemeval_harness.py \
        --dataset oracle \
        --download \
        --generate-answers \
        --output-dir runs/longmemeval-oracle-qa
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import urlretrieve

from engram import Engram
from engram.core.config import get_settings
from engram.policy import MemoryPolicy

# Benchmark policy: do NOT retype memories. LongMemEval turns are pure
# conversation events (episodic by nature); the default policy reclassifies
# casual language ("don't" -> constraint, "project" -> project), which then
# (a) gets filtered out by episodic retrieval, (b) earns critical/conflict
# slots that can supersede and hide sibling evidence, and (c) injects
# non-episodic distractors into retrieval. Keeping every turn episodic with no
# slots is the faithful representation for this benchmark. (Disclosed in
# summary.json as memory_policy="benchmark-no-retype".)
BENCHMARK_POLICY = MemoryPolicy(
    name="benchmark-no-retype",
    type_rules=(),
    slot_rules=(),
    generic_critical_slots=False,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


DATASET_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
}

WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval through the real Engram storage/retrieval path."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to a LongMemEval JSON file. If omitted, use --dataset with --download.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_URLS),
        default="oracle",
        help="Dataset to download when --data-path is omitted.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the selected LongMemEval JSON into --cache-dir if needed.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/longmemeval"),
        help="Where downloaded LongMemEval JSON files are cached.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/longmemeval"),
        help="Directory for summary.json, traces.jsonl, and hypotheses.jsonl.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start offset into the dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to run. Omit for the full file.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        help="Run only this question_id. Can be repeated.",
    )
    parser.add_argument(
        "--question-type",
        action="append",
        help="Run only this question_type. Can be repeated.",
    )
    parser.add_argument(
        "--memory-unit",
        choices=("session", "turn"),
        default="turn",
        help="Store each haystack session as one memory, or each turn separately.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of memories to retrieve per question.",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "semantic", "keyword"),
        default="hybrid",
        help="Engram search mode.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum retrieval score.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4000,
        help="Approximate token budget for answer context construction.",
    )
    parser.add_argument(
        "--max-memory-chars",
        type=int,
        default=2000,
        help=(
            "Maximum fact characters stored per memory, preserving head and tail. "
            "Keep this below the current unique fact index row limit."
        ),
    )
    parser.add_argument(
        "--ingest-batch-size",
        type=int,
        default=64,
        help="How many memories to add per add_batch() call.",
    )
    parser.add_argument(
        "--agent-prefix",
        default="longmemeval",
        help="Prefix for temporary benchmark agent IDs.",
    )
    parser.add_argument(
        "--database-url",
        help="Override ENGRAM_DATABASE_URL.",
    )
    parser.add_argument(
        "--embedding-provider",
        help="Override ENGRAM_EMBEDDING_PROVIDER, e.g. sentence-transformers.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Override ENGRAM_EMBEDDING_MODEL.",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        help="Override ENGRAM_EMBEDDING_DIMENSION.",
    )
    parser.add_argument(
        "--allow-embedding-dimension-change",
        action="store_true",
        help="Allow Engram to resize/clear existing embeddings if dimensions differ.",
    )
    parser.add_argument(
        "--keep-memories",
        action="store_true",
        help="Do not purge per-sample benchmark memories after each question.",
    )
    parser.add_argument(
        "--reuse-store",
        action="store_true",
        help=(
            "Reuse a previously ingested store across runs. Agent IDs become "
            "deterministic (keyed on data file + ingestion settings), ingestion "
            "is skipped when the agent already holds the full memory set, and "
            "memories are never purged. Pay ingestion cost once, iterate on "
            "retrieval/reading for free. Re-ingests automatically if the data "
            "path, memory unit, or max memory chars change."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Re-order search candidates with Engram's local cross-encoder.",
    )
    parser.add_argument(
        "--deep-search",
        action="store_true",
        help=(
            "Use Engram.search_evidence_set() with multi-query deep search "
            "(LLM query expansion + merged hybrid searches) instead of "
            "single-query search. Improves recall for questions whose "
            "evidence spans sessions."
        ),
    )
    parser.add_argument(
        "--preferred-role",
        choices=("none", "user", "assistant"),
        default="none",
        help=(
            "Optional turn-role preference for Engram.search_evidence_set(). "
            "Use 'user' for user-memory questions, 'assistant' for "
            "assistant-memory questions, or 'none' for neutral ranking."
        ),
    )
    parser.add_argument(
        "--reading",
        choices=("direct", "con"),
        default="direct",
        help=(
            "Answer generation style, applied uniformly to every question "
            "(no per-question-type branching): 'direct' single-shot, or 'con' "
            "(generic chain-of-note). Whichever is chosen must be held constant "
            "across any systems being compared."
        ),
    )
    parser.add_argument(
        "--context-strategy",
        choices=("window", "session", "block"),
        default="window",
        help=(
            "How to build the answer context: 'window' expands retrieved turns "
            "with nearby turns from the same session, 'session' expands full "
            "top sessions, and 'block' uses Engram.get_context_block()."
        ),
    )
    parser.add_argument(
        "--evidence-window-size",
        type=int,
        default=2,
        metavar="N",
        help=(
            "For --context-strategy window, include N turns before and after "
            "each retrieved turn from the same LongMemEval session."
        ),
    )
    parser.add_argument(
        "--prior-user-turns",
        type=int,
        default=2,
        metavar="N",
        help=(
            "For --context-strategy window, include up to N earlier user turns "
            "from the same session as each retrieved hit. This preserves "
            "linked evidence established earlier in a conversation."
        ),
    )
    parser.add_argument(
        "--expand-sessions",
        type=int,
        default=0,
        metavar="N",
        help=(
            "For --context-strategy session, build the answer context from "
            "the full content of the top N retrieved sessions fetched back "
            "from Engram. 0 uses --limit sessions."
        ),
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Use configured Engram LLM provider to answer from retrieved context.",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=96,
        help="Max tokens for optional answer generation.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed sample instead of recording an error trace.",
    )
    return parser.parse_args()


def resolve_data_path(args: argparse.Namespace) -> Path:
    if args.data_path is not None:
        return args.data_path

    url = DATASET_URLS[args.dataset]
    filename = url.rsplit("/", 1)[-1]
    path = args.cache_dir / filename
    if path.exists():
        return path
    if not args.download:
        raise SystemExit(
            "No --data-path supplied and cached dataset is missing. "
            "Pass --download to fetch it."
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urlretrieve(url, path)
    return path


def load_samples(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    if args.question_id:
        wanted = set(args.question_id)
        data = [sample for sample in data if sample.get("question_id") in wanted]
    if args.question_type:
        wanted_types = set(args.question_type)
        data = [
            sample for sample in data if sample.get("question_type") in wanted_types
        ]

    data = data[args.offset :]
    if args.max_samples is not None:
        data = data[: args.max_samples]
    return data


def normalize(text: Any) -> str:
    return " ".join(normalized_tokens(text))


def normalized_tokens(text: Any) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def contains_normalized_phrase(needle: Any, haystack: Any) -> bool:
    phrase = normalized_tokens(needle)
    if not phrase:
        return False
    tokens = normalized_tokens(haystack)
    size = len(phrase)
    return any(tokens[start : start + size] == phrase for start in range(len(tokens)))


def content_words(text: Any) -> set[str]:
    return {
        word for word in WORD_RE.findall(str(text).lower()) if word not in STOPWORDS
    }


def bounded_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max(1, (max_chars - 32) // 2)
    return f"{text[:half]}\n[...truncated...]\n{text[-half:]}"


def render_turn(turn: dict[str, Any], date: str) -> str:
    role = str(turn.get("role", "unknown")).upper()
    return f"[{date}] {role}: {turn.get('content', '')}"


def render_session(session: Sequence[dict[str, Any]], date: str) -> str:
    return "\n".join(render_turn(turn, date) for turn in session)


def iter_memories(
    sample: dict[str, Any],
    *,
    agent_id: str,
    memory_unit: str,
    max_memory_chars: int,
) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    session_ids = sample.get("haystack_session_ids", [])
    dates = sample.get("haystack_dates", [])
    sessions = sample.get("haystack_sessions", [])

    for original_session_id, date, session in zip(
        session_ids, dates, sessions, strict=True
    ):
        if memory_unit == "session":
            # NOTE: the gold ``has_answer`` flag is deliberately NOT stored in
            # memory metadata. Putting the answer label inside the retrievable
            # store is a leakage risk; evidence-recall metrics recompute it from
            # the raw sample at scoring time instead (see recall_metrics).
            content = bounded_text(render_session(session, date), max_memory_chars)
            memories.append(
                {
                    "content": content,
                    "main_content": content,
                    "agent_id": agent_id,
                    "memory_type": "episodic",
                    "metadata": {
                        "source": "longmemeval",
                        "question_id": sample["question_id"],
                        "question_type": sample.get("question_type"),
                        "question_date": sample.get("question_date"),
                        "original_session_id": original_session_id,
                        "haystack_date": date,
                        "memory_unit": "session",
                    },
                }
            )
            continue

        for turn_index, turn in enumerate(session):
            content = bounded_text(render_turn(turn, date), max_memory_chars)
            memories.append(
                {
                    "content": content,
                    "main_content": content,
                    "agent_id": agent_id,
                    "memory_type": "episodic",
                    "metadata": {
                        "source": "longmemeval",
                        "question_id": sample["question_id"],
                        "question_type": sample.get("question_type"),
                        "question_date": sample.get("question_date"),
                        "original_session_id": original_session_id,
                        "haystack_date": date,
                        "turn_index": turn_index,
                        "turn_role": turn.get("role"),
                        "memory_unit": "turn",
                    },
                }
            )
    return memories


def chunks(
    items: Sequence[dict[str, Any]], size: int
) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def recall_metrics(
    sample: dict[str, Any],
    retrieved: Sequence[dict[str, Any]],
    context: str,
    hypothesis: str,
) -> dict[str, Any]:
    expected_sessions = set(sample.get("answer_session_ids", []))
    retrieved_sessions = {
        str(item["metadata"].get("original_session_id"))
        for item in retrieved
        if item.get("metadata", {}).get("original_session_id") is not None
    }
    recalled_sessions = expected_sessions & retrieved_sessions

    # Recompute the gold-evidence keys from the raw sample rather than reading a
    # stored ``has_answer`` flag — the answer label is intentionally kept out of
    # the retrievable store. Match retrieved memories back to gold turns (turn
    # unit) or gold sessions (session unit).
    gold_turn_keys: set[tuple[str, int]] = set()
    gold_answer_sessions: set[str] = set()
    for sid, session in zip(
        sample.get("haystack_session_ids", []),
        sample.get("haystack_sessions", []),
        strict=False,
    ):
        for ti, turn in enumerate(session):
            if turn.get("has_answer"):
                gold_turn_keys.add((str(sid), ti))
                gold_answer_sessions.add(str(sid))

    def _is_answer_memory(item: dict[str, Any]) -> bool:
        meta = item.get("metadata", {})
        sid = str(meta.get("original_session_id"))
        ti = meta.get("turn_index")
        if ti is not None:
            return (sid, int(ti)) in gold_turn_keys
        return sid in gold_answer_sessions

    answer = str(sample.get("answer", ""))
    answer_words = content_words(answer)
    context_words = content_words(context)
    hypothesis_words = content_words(hypothesis)

    return {
        "expected_answer_sessions": sorted(expected_sessions),
        "retrieved_sessions": sorted(retrieved_sessions),
        "recalled_answer_sessions": sorted(recalled_sessions),
        "answer_session_recall": (
            len(recalled_sessions) / len(expected_sessions)
            if expected_sessions
            else 0.0
        ),
        "all_answer_sessions_recalled": expected_sessions <= retrieved_sessions
        if expected_sessions
        else False,
        "any_answer_memory_retrieved": any(
            _is_answer_memory(item) for item in retrieved
        ),
        "answer_exact_in_context": contains_normalized_phrase(answer, context),
        "answer_exact_in_hypothesis": bool(hypothesis)
        and contains_normalized_phrase(answer, hypothesis),
        "answer_word_coverage_in_context": (
            len(answer_words & context_words) / len(answer_words)
            if answer_words
            else 0.0
        ),
        "answer_word_coverage_in_hypothesis": (
            len(answer_words & hypothesis_words) / len(answer_words)
            if answer_words and hypothesis
            else 0.0
        ),
    }


async def build_expanded_context(
    engram: Engram,
    *,
    agent_id: str,
    retrieved: Sequence[dict[str, Any]],
    n_sessions: int,
    max_tokens: int,
) -> str:
    """Render the full content of the top retrieved sessions, oldest first.

    Retrieved turns point at their source session; the complete sessions are
    fetched back from Engram so the reader sees each evidence turn with its
    surrounding conversation instead of an isolated snippet.
    """
    session_best_rank: dict[str, int] = {}
    for rank, item in enumerate(retrieved):
        session_id = item.get("metadata", {}).get("original_session_id")
        if session_id is not None and session_id not in session_best_rank:
            session_best_rank[str(session_id)] = rank
    top_sessions = sorted(session_best_rank, key=lambda s: session_best_rank[s])
    expanded = set(top_sessions[:n_sessions])

    blocks: list[str] = []
    for session_id in top_sessions[:n_sessions]:
        memories = await engram.get_memories(
            agent_id,
            metadata_filter={"original_session_id": session_id},
        )
        memories.sort(key=lambda m: m.metadata.get("turn_index", 0))
        if memories:
            blocks.append("\n".join(m.content for m in memories))

    # Depth + breadth: retrieved turns from sessions beyond the expanded
    # top-N stay in the context as individual lines, so widening one part
    # of the evidence never silently drops another.
    leftover = [
        item["content"]
        for item in retrieved
        if str(item.get("metadata", {}).get("original_session_id")) not in expanded
    ]
    if leftover:
        blocks.append("Other relevant memories:\n" + "\n".join(leftover))

    context = "\n\n".join(blocks)
    max_chars = max_tokens * 4  # same heuristic as get_context_block
    if len(context) > max_chars:
        context = context[:max_chars]
    return context


async def maybe_generate_answer(
    engram: Engram,
    *,
    question: str,
    question_date: str | None,
    context: str,
    max_tokens: int,
    reading: str = "direct",
) -> str:
    """Generate an answer from retrieved context through public Engram APIs."""
    return await engram.answer_from_evidence(
        question=question,
        question_date=question_date,
        context=context,
        max_tokens=max_tokens,
        reading=reading,
    )


async def database_vector_dimension(engram: Engram) -> int | None:
    storage = getattr(engram, "_storage", None)
    if storage is None:
        return None
    return await storage.fetchval(
        """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = 'agent_memory'::regclass
            AND attname = 'embedding'
        """
    )


async def assert_vector_dimension_matches(engram: Engram, *, label: str) -> int | None:
    embedding = getattr(engram, "_embedding", None)
    expected = getattr(embedding, "dimension", None)
    actual = await database_vector_dimension(engram)
    # atttypmod is the pgvector dimension directly; -1 means an unbounded
    # vector column (no fixed dimension), which accepts any width, so only a
    # positive, differing dimension is a genuine mismatch.
    if (
        expected is not None
        and actual is not None
        and actual > 0
        and actual != expected
    ):
        settings = getattr(engram, "_settings", None)
        provider = getattr(settings, "embedding_provider", "unknown")
        model = getattr(settings, "embedding_model", "unknown")
        raise RuntimeError(
            f"Embedding dimension mismatch {label}: database "
            f"agent_memory.embedding is vector({actual}) but configured "
            f"{provider}/{model} emits {expected}-dimensional vectors. "
            "Use a clean database/schema for the benchmark, or run with the "
            "same embedding provider/model that created the existing schema."
        )
    return actual


def store_fingerprint(args: argparse.Namespace) -> str:
    """Short hash of the settings that determine what gets ingested.

    Reused agent IDs must change whenever the ingested content would differ, so
    a stale store is never silently served. Data path, memory unit, and the
    per-memory char cap all change the stored rows.
    """
    raw = "|".join(
        [
            Path(resolve_data_path(args)).name,
            str(args.memory_unit),
            str(args.max_memory_chars),
        ]
    )
    return hashlib.md5(raw.encode()).hexdigest()[:10]


async def run_sample(
    engram: Engram,
    sample: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_id: str,
) -> dict[str, Any]:
    question_id = str(sample["question_id"])
    if args.reuse_store:
        agent_id = f"{args.agent_prefix}-store-{store_fingerprint(args)}-{question_id}"
    else:
        agent_id = f"{args.agent_prefix}-{run_id}-{question_id}"
    start = time.perf_counter()
    memories = iter_memories(
        sample,
        agent_id=agent_id,
        memory_unit=args.memory_unit,
        max_memory_chars=args.max_memory_chars,
    )

    try:
        ingested = True
        if args.reuse_store and memories:
            existing = await engram.get_memories(agent_id, limit=len(memories) + 1)
            if len(existing) >= len(memories):
                ingested = False  # store already complete; skip ingestion cost
            elif existing:
                # Partial store from an interrupted run: rebuild from scratch.
                await engram.purge(agent_id=agent_id)
        if ingested:
            for batch in chunks(memories, args.ingest_batch_size):
                await engram.add_batch(batch)

        if args.deep_search:
            results = await engram.search_evidence_set(
                query=sample["question"],
                agent_id=agent_id,
                limit=args.limit,
                min_score=args.min_score,
                mode=args.mode,
                rerank=args.rerank,
                use_deep_search=True,
                preferred_role=(
                    None if args.preferred_role == "none" else args.preferred_role
                ),
            )
        else:
            results = await engram.search(
                query=sample["question"],
                agent_id=agent_id,
                limit=args.limit,
                min_score=args.min_score,
                mode=args.mode,
                rerank=args.rerank,
            )
        retrieved = [
            {
                "memory_id": result.memory.memory_id,
                "score": result.score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "decay_score": result.decay_score,
                "content": result.memory.content,
                "metadata": result.memory.metadata,
            }
            for result in results
        ]

        context_sources: list[dict[str, Any]] = []
        effective_context_strategy = args.context_strategy
        if effective_context_strategy == "session":
            context = await build_expanded_context(
                engram,
                agent_id=agent_id,
                retrieved=retrieved,
                n_sessions=args.expand_sessions or args.limit,
                max_tokens=args.max_context_tokens,
            )
        elif effective_context_strategy == "window":
            context, context_sources = await engram.get_neighboring_context_block(
                results,
                agent_id,
                before=args.evidence_window_size,
                after=args.evidence_window_size,
                include_session_start=True,
                max_tokens=args.max_context_tokens,
                prior_user_turns=args.prior_user_turns,
                context_order="relevance",
            )
        else:
            context = await engram.get_context_block(
                sample["question"],
                agent_id,
                limit=args.limit,
                min_score=args.min_score,
                max_tokens=args.max_context_tokens,
                group_by_type=True,
                rerank=args.rerank,
            )
        hypothesis = ""
        effective_reading = args.reading
        if args.generate_answers:
            hypothesis = await maybe_generate_answer(
                engram,
                question=sample["question"],
                question_date=sample.get("question_date"),
                context=context,
                max_tokens=args.answer_max_tokens,
                reading=args.reading,
            )

        metrics = recall_metrics(sample, retrieved, context, hypothesis)
        return {
            "question_id": question_id,
            "question_type": sample.get("question_type"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "hypothesis": hypothesis,
            "agent_id": agent_id,
            "memory_count": len(memories),
            "ingested": ingested,
            "retrieved": retrieved,
            "context": context,
            "context_strategy": effective_context_strategy,
            "context_sources": context_sources,
            "reading": effective_reading,
            "metrics": metrics,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "error": None,
        }
    finally:
        if not args.keep_memories and not args.reuse_store:
            await engram.purge(agent_id=agent_id)


def summarize(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [trace for trace in traces if trace.get("error") is None]
    errored = [trace for trace in traces if trace.get("error") is not None]

    def avg_metric(name: str) -> float:
        values = [trace["metrics"][name] for trace in completed]
        return sum(values) / len(values) if values else 0.0

    def rate_metric(name: str) -> float:
        values = [bool(trace["metrics"][name]) for trace in completed]
        return sum(values) / len(values) if values else 0.0

    by_type: dict[str, dict[str, Any]] = {}
    for trace in completed:
        qtype = str(trace.get("question_type"))
        bucket = by_type.setdefault(qtype, {"count": 0, "traces": []})
        bucket["count"] += 1
        bucket["traces"].append(trace)

    by_type_summary = {}
    for qtype, bucket in by_type.items():
        subset = bucket["traces"]
        by_type_summary[qtype] = {
            "count": bucket["count"],
            "answer_session_recall": sum(
                item["metrics"]["answer_session_recall"] for item in subset
            )
            / len(subset),
            "all_answer_sessions_recalled_rate": sum(
                bool(item["metrics"]["all_answer_sessions_recalled"]) for item in subset
            )
            / len(subset),
            "any_answer_memory_retrieved_rate": sum(
                bool(item["metrics"]["any_answer_memory_retrieved"]) for item in subset
            )
            / len(subset),
            "answer_word_coverage_in_context": sum(
                item["metrics"]["answer_word_coverage_in_context"] for item in subset
            )
            / len(subset),
        }

    return {
        "total": len(traces),
        "completed": len(completed),
        "errors": len(errored),
        "errored_question_ids": [trace.get("question_id") for trace in errored],
        "answer_session_recall": avg_metric("answer_session_recall"),
        "all_answer_sessions_recalled_rate": rate_metric(
            "all_answer_sessions_recalled"
        ),
        "any_answer_memory_retrieved_rate": rate_metric("any_answer_memory_retrieved"),
        "answer_exact_in_context_rate": rate_metric("answer_exact_in_context"),
        "answer_exact_in_hypothesis_rate": rate_metric("answer_exact_in_hypothesis"),
        "answer_word_coverage_in_context": avg_metric(
            "answer_word_coverage_in_context"
        ),
        "answer_word_coverage_in_hypothesis": avg_metric(
            "answer_word_coverage_in_hypothesis"
        ),
        "by_question_type": by_type_summary,
    }


async def run(args: argparse.Namespace) -> None:
    data_path = resolve_data_path(args)
    samples = load_samples(data_path, args)
    if not samples:
        raise SystemExit("No samples selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = args.output_dir / "traces.jsonl"
    hypotheses_path = args.output_dir / "hypotheses.jsonl"
    summary_path = args.output_dir / "summary.json"
    run_id = uuid.uuid4().hex[:8]

    settings = get_settings()
    # LongMemEval histories often repeat details across sessions. Keep all
    # benchmark memories so evidence labels remain attributable.
    settings_update: dict[str, Any] = {"near_duplicate_threshold": 1.0}
    if args.embedding_provider:
        settings_update["embedding_provider"] = args.embedding_provider
    if args.embedding_model:
        settings_update["embedding_model"] = args.embedding_model
    if args.embedding_dimension is not None:
        settings_update["embedding_dimension"] = args.embedding_dimension
    if args.allow_embedding_dimension_change:
        settings_update["allow_embedding_dimension_change"] = True
    settings = settings.model_copy(update=settings_update)

    engram = Engram(
        settings=settings,
        database_url=args.database_url,
        memory_policy=BENCHMARK_POLICY,
    )
    await engram.connect()
    startup_vector_dimension = await assert_vector_dimension_matches(
        engram, label="at startup"
    )
    traces: list[dict[str, Any]] = []
    try:
        with (
            traces_path.open("w") as traces_file,
            hypotheses_path.open("w") as hypotheses_file,
        ):
            for index, sample in enumerate(samples, start=1):
                label = f"{index}/{len(samples)} {sample.get('question_id')}"
                print(f"Running {label}")
                await assert_vector_dimension_matches(engram, label=f"before {label}")
                try:
                    trace = await run_sample(
                        engram,
                        sample,
                        args=args,
                        run_id=run_id,
                    )
                except Exception as exc:
                    # Dimension mismatches are already caught fatally by the
                    # proactive pg_attribute check above, so any error here is a
                    # genuine per-sample failure. Abort on --fail-fast; otherwise
                    # record it for forensics but never let it reach the scored
                    # file as a blank hypothesis (an infra failure must not be
                    # graded as a wrong model answer).
                    if args.fail_fast:
                        raise
                    trace = {
                        "question_id": sample.get("question_id"),
                        "question_type": sample.get("question_type"),
                        "question": sample.get("question"),
                        "answer": sample.get("answer"),
                        "hypothesis": "",
                        "retrieved": [],
                        "context": "",
                        "metrics": {},
                        "elapsed_seconds": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                traces.append(trace)
                traces_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                traces_file.flush()
                # A sample that errored never ran; it must not appear in the
                # scored hypotheses file. The external LongMemEval scorer keys on
                # question_id and simply won't grade what isn't there, instead of
                # counting a blank as a model miss. summary.json's
                # errored_question_ids records what was skipped.
                if trace.get("error") is None:
                    hypotheses_file.write(
                        json.dumps(
                            {
                                "question_id": trace["question_id"],
                                "hypothesis": trace.get("hypothesis", ""),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    hypotheses_file.flush()

        summary = summarize(traces)
        summary["data_path"] = str(data_path)
        summary["memory_unit"] = args.memory_unit
        summary["search_mode"] = args.mode
        summary["limit"] = args.limit
        summary["generated_answers"] = bool(args.generate_answers)
        summary["rerank"] = bool(args.rerank)
        summary["context_strategy"] = args.context_strategy
        summary["evidence_window_size"] = args.evidence_window_size
        summary["prior_user_turns"] = args.prior_user_turns
        summary["max_context_tokens"] = args.max_context_tokens
        summary["answer_max_tokens"] = args.answer_max_tokens
        summary["expand_sessions"] = args.expand_sessions
        summary["deep_search"] = bool(args.deep_search)
        summary["evidence_set"] = bool(args.deep_search)
        summary["preferred_role"] = args.preferred_role
        summary["reading"] = args.reading
        summary["llm_provider"] = settings.llm_provider
        summary["llm_model"] = settings.llm_model
        summary["embedding_provider"] = settings.embedding_provider
        summary["embedding_model"] = settings.embedding_model
        summary["embedding_dimension"] = settings.embedding_dimension
        summary["database_vector_dimension"] = startup_vector_dimension
        summary["near_duplicate_threshold"] = settings.near_duplicate_threshold
        summary["max_memory_chars"] = args.max_memory_chars
        summary["min_score"] = args.min_score
        summary["memory_policy"] = BENCHMARK_POLICY.name
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Wrote {traces_path}")
        print(f"Wrote {hypotheses_path}")
        print(f"Wrote {summary_path}")
    finally:
        await engram.close()


def main() -> None:
    args = parse_args()
    if args.ingest_batch_size < 1:
        raise SystemExit("--ingest-batch-size must be >= 1")
    if args.max_memory_chars < 1:
        raise SystemExit("--max-memory-chars must be >= 1")
    if args.max_context_tokens < 1:
        raise SystemExit("--max-context-tokens must be >= 1")
    if args.evidence_window_size < 0:
        raise SystemExit("--evidence-window-size must be >= 0")
    if args.prior_user_turns < 0:
        raise SystemExit("--prior-user-turns must be >= 0")
    if args.generate_answers and not os.getenv("ENGRAM_LLM_PROVIDER"):
        print(
            "Warning: --generate-answers was set but ENGRAM_LLM_PROVIDER is not set; "
            "hypotheses will be empty unless settings configure an LLM provider."
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
