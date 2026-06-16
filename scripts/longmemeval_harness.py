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
    from collections.abc import Callable, Iterator, Sequence

    from engram.memory.models import Memory, SearchResult
    from engram.task.models import AgentEvent


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
        "--ingest-surface",
        choices=("memory", "event", "both"),
        default="memory",
        help=(
            "Where to ingest LongMemEval histories. 'memory' preserves the "
            "original benchmark path, 'event' writes the raw ledger with "
            "record_event(), and 'both' stores both surfaces."
        ),
    )
    parser.add_argument(
        "--retrieval-surface",
        choices=("memory", "event"),
        default="memory",
        help=(
            "Which Engram surface to evaluate: memory search or hybrid event "
            "search over the raw ledger."
        ),
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
        "--backfill-event-embeddings",
        action="store_true",
        help=(
            "After event ingestion/reuse, run bounded event-embedding backfill "
            "until no old event rows remain. New event rows are embedded on "
            "write; this is mainly for reused stores from older runs."
        ),
    )
    parser.add_argument(
        "--event-backfill-batch-size",
        type=int,
        default=1000,
        help="Batch size for --backfill-event-embeddings.",
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
            "Use the harness search_evidence_set() with multi-query deep "
            "search (LLM query expansion + merged hybrid searches) instead of "
            "single-query search. Improves recall for questions whose "
            "evidence spans sessions."
        ),
    )
    parser.add_argument(
        "--preferred-role",
        choices=("none", "user", "assistant"),
        default="none",
        help=(
            "Optional turn-role preference for the harness search_evidence_set(). "
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
        "--aggregation-full-session",
        action="store_true",
        help=(
            "For --context-strategy window, when a question is a counting/"
            "aggregation question (how many/how much/total/...), include every "
            "turn of each top-ranked matched session instead of a window, so "
            "counts are complete. Rank-gated to avoid pulling in distractors."
        ),
    )
    parser.add_argument(
        "--aggregation-context-tokens",
        type=int,
        default=16000,
        metavar="N",
        help=(
            "Answer-context token budget used only for aggregation questions "
            "when --aggregation-full-session is set (full sessions need room)."
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


def _event_type_for_role(role: str) -> tuple[str, str]:
    normalized = role.strip().lower()
    if normalized == "user":
        return "user", "user_message"
    if normalized == "assistant":
        return "assistant", "assistant_message"
    if normalized == "tool":
        return "tool", "tool_result"
    if normalized in {"system", "agent"}:
        return normalized, "system_note" if normalized == "system" else "agent_action"
    return "system", "system_note"


def iter_events(
    sample: dict[str, Any],
    *,
    agent_id: str,
    memory_unit: str,
    max_memory_chars: int,
) -> list[dict[str, Any]]:
    """Render LongMemEval histories as raw event-ledger rows.

    The metadata mirrors ``iter_memories()`` so recall metrics and context
    expansion can score either surface the same way.
    """
    events: list[dict[str, Any]] = []
    session_ids = sample.get("haystack_session_ids", [])
    dates = sample.get("haystack_dates", [])
    sessions = sample.get("haystack_sessions", [])

    for original_session_id, date, session in zip(
        session_ids, dates, sessions, strict=True
    ):
        session_id = f"{sample['question_id']}:{original_session_id}"
        if memory_unit == "session":
            content = bounded_text(render_session(session, date), max_memory_chars)
            events.append(
                {
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "role": "system",
                    "event_type": "observation",
                    "content": content,
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
            role, event_type = _event_type_for_role(str(turn.get("role", "unknown")))
            content = bounded_text(render_turn(turn, date), max_memory_chars)
            events.append(
                {
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "role": role,
                    "event_type": event_type,
                    "content": content,
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
    return events


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


# ============================================================================
# Evidence selection and reading
#
# These helpers were relocated out of the Engram public API: they are
# LongMemEval / QA-harness machinery (multi-call "chain-of-note" reading,
# session-diversified evidence selection, and turn-window expansion keyed on
# this benchmark's metadata schema), not general-purpose memory primitives.
# They drive Engram only through its public API (search, deep_search,
# get_memories, llm.complete), so the library itself stays domain-neutral.
# ============================================================================


async def search_evidence_set(
    engram: Engram,
    query: str,
    agent_id: str,
    *,
    user_id: str | None = None,
    limit: int = 10,
    candidate_limit: int | None = None,
    min_score: float = 0.0,
    metadata_filter: dict[str, Any] | None = None,
    memory_types: list[str] | None = None,
    mode: str = "hybrid",
    use_deep_search: bool = True,
    rerank: bool = True,
    diversify_metadata_key: str = "original_session_id",
    max_per_group: int = 3,
    preferred_role: str | None = None,
    role_metadata_key: str = "turn_role",
) -> list[SearchResult]:
    """Retrieve a broad, diverse evidence set for aggregation questions.

    Overfetches candidates (optionally via deep multi-query retrieval), applies
    a small role-aware rank bias, then round-robins by session/group before
    returning the final evidence set.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if max_per_group < 1:
        raise ValueError("max_per_group must be >= 1")

    max_search_limit = engram._settings.max_search_limit
    overfetch = candidate_limit
    if overfetch is None:
        overfetch = limit * engram._settings.search_candidate_multiplier
    overfetch = min(max(overfetch, limit), max_search_limit)

    if use_deep_search:
        candidates = await engram.deep_search(
            query,
            agent_id,
            user_id=user_id,
            limit=overfetch,
            min_score=min_score,
            metadata_filter=metadata_filter,
            memory_types=memory_types,
            mode=mode,
            rerank=rerank,
        )
    else:
        candidates = await engram.search(
            query,
            agent_id,
            user_id=user_id,
            limit=overfetch,
            min_score=min_score,
            metadata_filter=metadata_filter,
            memory_types=memory_types,
            mode=mode,
            rerank=rerank,
        )
    if not candidates:
        return []

    preferred = preferred_role.strip().lower() if preferred_role else None

    def role_rank_bias(result: SearchResult) -> float:
        if preferred is None:
            return 0.0
        role = result.memory.metadata.get(role_metadata_key)
        if not isinstance(role, str):
            return 0.0
        normalized = role.strip().lower()
        if normalized == preferred:
            return -1.5
        if preferred == "user" and normalized in {"assistant", "system", "tool"}:
            return 1.5
        return 0.0

    def group_key(result: SearchResult) -> str:
        memory = result.memory
        if memory.session_id:
            return f"session:{memory.session_id}"
        value = memory.metadata.get(diversify_metadata_key)
        if value is not None:
            return f"metadata:{value}"
        return f"memory:{memory.memory_id}"

    ranked = [
        result
        for _rank, result in sorted(
            enumerate(candidates),
            key=lambda item: (item[0] + role_rank_bias(item[1]), item[0]),
        )
    ]
    groups: dict[str, list[SearchResult]] = {}
    for result in ranked:
        groups.setdefault(group_key(result), []).append(result)

    selected: list[SearchResult] = []
    selected_ids: set[str] = set()
    for depth in range(max_per_group):
        for group_results in groups.values():
            if depth >= len(group_results):
                continue
            result = group_results[depth]
            selected.append(result)
            selected_ids.add(result.memory.memory_id)
            if len(selected) >= limit:
                return selected

    for result in ranked:
        if result.memory.memory_id in selected_ids:
            continue
        selected.append(result)
        if len(selected) >= limit:
            break
    return selected


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _memory_window_group(memory: Memory, metadata_key: str) -> str | None:
    if memory.session_id:
        return str(memory.session_id)
    value = memory.metadata.get(metadata_key)
    return str(value) if value is not None else None


_AGGREGATION_PATTERNS = (
    "how many",
    "how much",
    "how often",
    "number of",
    "altogether",
    "combined",
    "average",
    " each ",
)


def is_aggregation_question(question: str) -> bool:
    """Heuristically flag counting/aggregation questions.

    These need every relevant turn from each matched session to count without
    over- or under-counting, so a windowed context view is not enough.
    """
    q = f" {question.lower().strip()} "
    if "total" in q:
        return True
    return any(pattern in q for pattern in _AGGREGATION_PATTERNS)


async def get_neighboring_context_block(
    engram: Engram,
    results: list[SearchResult],
    agent_id: str,
    *,
    user_id: str | None = None,
    before: int = 2,
    after: int = 2,
    include_session_start: bool = False,
    max_tokens: int | None = None,
    token_counter: Callable[[str], int] | None = None,
    memory_types: list[str] | None = None,
    session_metadata_key: str = "original_session_id",
    turn_metadata_key: str = "turn_index",
    date_metadata_key: str = "haystack_date",
    role_metadata_key: str = "turn_role",
    group_limit: int = 200,
    priority_window_results: int = 3,
    prior_user_turns: int = 0,
    context_order: str = "chronological",
    whole_session_top_k: int = 0,
) -> tuple[str, list[dict[str, Any]]]:
    """Expand retrieved turn memories with neighboring session memories.

    When whole_session_top_k > 0, sessions whose best hit ranks above that
    cutoff contribute every turn (not just a window). The full session is
    already fetched per hit, so this adds no retrieval. Used for aggregation
    questions, where a windowed view drops mentions and breaks the count.

    Uses only stored Engram memories plus their session/turn metadata, read via
    the public get_memories() API. Useful when a search hit is near the answer
    but the necessary context is in adjacent turns from the same conversation.
    """
    if before < 0 or after < 0:
        raise ValueError("before and after must be >= 0")
    if group_limit < 1:
        raise ValueError("group_limit must be >= 1")
    if priority_window_results < 0:
        raise ValueError("priority_window_results must be >= 0")
    if prior_user_turns < 0:
        raise ValueError("prior_user_turns must be >= 0")
    if whole_session_top_k < 0:
        raise ValueError("whole_session_top_k must be >= 0")
    if context_order not in {"chronological", "relevance"}:
        raise ValueError("context_order must be 'chronological' or 'relevance'")

    count = token_counter or (lambda text: max(1, len(text) // 4))
    group_cache: dict[tuple[str, str], list[Memory]] = {}
    selected: dict[str, tuple[tuple[int, int, int], Memory]] = {}
    group_best_rank: dict[str, int] = {}

    def remember(memory: Memory, priority: tuple[int, int, int]) -> None:
        current = selected.get(memory.memory_id)
        if current is None or priority < current[0]:
            selected[memory.memory_id] = (priority, memory)

    async def fetch_group(memory: Memory, group_key: str) -> list[Memory]:
        cache_key = ("session" if memory.session_id else "metadata", group_key)
        cached = group_cache.get(cache_key)
        if cached is not None:
            return cached
        if memory.session_id:
            group = await engram.get_memories(
                agent_id,
                user_id=user_id,
                session_id=memory.session_id,
                memory_types=memory_types,
                limit=group_limit,
            )
        else:
            group = await engram.get_memories(
                agent_id,
                user_id=user_id,
                metadata_filter={session_metadata_key: group_key},
                memory_types=memory_types,
                limit=group_limit,
            )
        group_cache[cache_key] = group
        return group

    for rank, result in enumerate(results):
        memory = result.memory
        if rank < priority_window_results:
            remember(memory, (rank, 0, 0))
        else:
            remember(memory, (priority_window_results, 0, rank))

        turn_index = _metadata_int(memory.metadata, turn_metadata_key)
        group_key = _memory_window_group(memory, session_metadata_key)
        if turn_index is None or group_key is None:
            continue
        group_best_rank[group_key] = min(rank, group_best_rank.get(group_key, rank))

        group = await fetch_group(memory, group_key)
        by_turn = {
            idx: item
            for item in group
            if (idx := _metadata_int(item.metadata, turn_metadata_key)) is not None
        }
        if whole_session_top_k and group_best_rank[group_key] < whole_session_top_k:
            # Aggregation queries need every mention in a strongly-matched
            # session, not a window around each hit. The full session is already
            # in `group`, so this widens coverage without extra retrieval.
            indexes = set(by_turn)
        else:
            indexes = set(range(max(0, turn_index - before), turn_index + after + 1))
        if include_session_start:
            indexes.add(0)
        for idx in indexes:
            neighbor = by_turn.get(idx)
            if neighbor is None:
                continue
            distance = abs(idx - turn_index)
            if rank < priority_window_results:
                remember(neighbor, (rank, 0 if distance == 0 else 1, distance))
            else:
                remember(neighbor, (priority_window_results + 1, rank, distance))
        if prior_user_turns:
            prior_users = [
                item
                for item in group
                if (idx := _metadata_int(item.metadata, turn_metadata_key)) is not None
                and idx < turn_index
                and str(item.metadata.get(role_metadata_key, "")).lower() == "user"
            ]
            prior_users.sort(
                key=lambda item: _metadata_int(item.metadata, turn_metadata_key) or 0,
                reverse=True,
            )
            for prior in prior_users[:prior_user_turns]:
                idx = _metadata_int(prior.metadata, turn_metadata_key)
                assert idx is not None
                distance = turn_index - idx
                if rank < priority_window_results:
                    remember(prior, (rank, 2, distance))
                else:
                    remember(prior, (priority_window_results + 1, rank, distance))

    kept: list[Memory] = []
    used = 0
    for _priority, memory in sorted(selected.values(), key=lambda item: item[0]):
        cost = count(memory.content)
        if max_tokens is not None and kept and used + cost > max_tokens:
            continue
        kept.append(memory)
        used += cost

    def chronological_key(memory: Memory) -> tuple[str, str, int, str]:
        return (
            str(memory.metadata.get(date_metadata_key) or memory.created_at),
            _memory_window_group(memory, session_metadata_key) or "",
            _metadata_int(memory.metadata, turn_metadata_key) or 0,
            memory.memory_id,
        )

    if context_order == "relevance":
        kept.sort(
            key=lambda memory: (
                group_best_rank.get(
                    _memory_window_group(memory, session_metadata_key) or "",
                    len(results),
                ),
                *chronological_key(memory),
            )
        )
    else:
        kept.sort(key=chronological_key)
    sources = [
        {
            "memory_id": memory.memory_id,
            "session_id": memory.session_id,
            "group": _memory_window_group(memory, session_metadata_key),
            "turn_index": _metadata_int(memory.metadata, turn_metadata_key),
            "date": memory.metadata.get(date_metadata_key),
            "has_answer": bool(memory.metadata.get("has_answer")),
        }
        for memory in kept
    ]
    return "\n".join(memory.content for memory in kept), sources


def _event_group(event: AgentEvent) -> str | None:
    value = event.metadata.get("original_session_id")
    if value is not None:
        return str(value)
    return str(event.session_id) if event.session_id else None


def _event_turn_index(event: AgentEvent) -> int | None:
    return _metadata_int(event.metadata, "turn_index")


def _event_to_retrieved(event: AgentEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "session_id": event.session_id,
        "score": None,
        "semantic_score": None,
        "keyword_score": None,
        "decay_score": None,
        "content": event.content,
        "metadata": event.metadata,
    }


async def search_event_evidence_set(
    engram: Engram,
    query: str,
    agent_id: str,
    *,
    limit: int = 10,
    mode: str = "hybrid",
) -> list[AgentEvent]:
    """Retrieve LongMemEval evidence from the raw event ledger."""
    return await engram.search_events(
        query,
        agent_id=agent_id,
        limit=limit,
        mode=mode,
    )


async def get_neighboring_event_context_block(
    engram: Engram,
    events: Sequence[AgentEvent],
    agent_id: str,
    *,
    before: int = 2,
    after: int = 2,
    include_session_start: bool = False,
    max_tokens: int | None = None,
    group_limit: int = 200,
    priority_window_results: int = 3,
    prior_user_turns: int = 0,
    context_order: str = "chronological",
) -> tuple[str, list[dict[str, Any]]]:
    """Expand retrieved event hits with neighboring events from the same session."""
    if before < 0 or after < 0:
        raise ValueError("before and after must be >= 0")
    if group_limit < 1:
        raise ValueError("group_limit must be >= 1")
    if priority_window_results < 0:
        raise ValueError("priority_window_results must be >= 0")
    if prior_user_turns < 0:
        raise ValueError("prior_user_turns must be >= 0")
    if context_order not in {"chronological", "relevance"}:
        raise ValueError("context_order must be 'chronological' or 'relevance'")

    def count(text: str) -> int:
        return max(1, len(text) // 4)

    group_cache: dict[str, list[AgentEvent]] = {}
    selected: dict[str, tuple[tuple[int, int, int], AgentEvent]] = {}
    group_best_rank: dict[str, int] = {}

    def remember(event: AgentEvent, priority: tuple[int, int, int]) -> None:
        current = selected.get(event.event_id)
        if current is None or priority < current[0]:
            selected[event.event_id] = (priority, event)

    async def fetch_group(event: AgentEvent, group_key: str) -> list[AgentEvent]:
        cached = group_cache.get(group_key)
        if cached is not None:
            return cached
        if event.session_id:
            group = await engram.list_events(
                agent_id=agent_id,
                session_id=event.session_id,
                limit=group_limit,
            )
        else:
            group = [
                candidate
                for candidate in await engram.search_events(
                    group_key,
                    agent_id=agent_id,
                    limit=group_limit,
                    mode="keyword",
                )
                if _event_group(candidate) == group_key
            ]
        group_cache[group_key] = group
        return group

    for rank, event in enumerate(events):
        if rank < priority_window_results:
            remember(event, (rank, 0, 0))
        else:
            remember(event, (priority_window_results, 0, rank))

        turn_index = _event_turn_index(event)
        group_key = _event_group(event)
        if turn_index is None or group_key is None:
            continue
        group_best_rank[group_key] = min(rank, group_best_rank.get(group_key, rank))

        group = await fetch_group(event, group_key)
        by_turn = {
            idx: item for item in group if (idx := _event_turn_index(item)) is not None
        }
        indexes = set(range(max(0, turn_index - before), turn_index + after + 1))
        if include_session_start:
            indexes.add(0)
        for idx in indexes:
            neighbor = by_turn.get(idx)
            if neighbor is None:
                continue
            distance = abs(idx - turn_index)
            if rank < priority_window_results:
                remember(neighbor, (rank, 0 if distance == 0 else 1, distance))
            else:
                remember(neighbor, (priority_window_results + 1, rank, distance))
        if prior_user_turns:
            prior_users = [
                item
                for item in group
                if (idx := _event_turn_index(item)) is not None
                and idx < turn_index
                and str(item.metadata.get("turn_role", "")).lower() == "user"
            ]
            prior_users.sort(
                key=lambda item: _event_turn_index(item) or 0, reverse=True
            )
            for prior in prior_users[:prior_user_turns]:
                idx = _event_turn_index(prior)
                assert idx is not None
                distance = turn_index - idx
                if rank < priority_window_results:
                    remember(prior, (rank, 2, distance))
                else:
                    remember(prior, (priority_window_results + 1, rank, distance))

    kept: list[AgentEvent] = []
    used = 0
    for _priority, event in sorted(selected.values(), key=lambda item: item[0]):
        cost = count(event.content)
        if max_tokens is not None and kept and used + cost > max_tokens:
            continue
        kept.append(event)
        used += cost

    def chronological_key(event: AgentEvent) -> tuple[str, str, int, str]:
        return (
            str(event.metadata.get("haystack_date") or event.created_at),
            _event_group(event) or "",
            _event_turn_index(event) or 0,
            event.event_id,
        )

    if context_order == "relevance":
        kept.sort(
            key=lambda event: (
                group_best_rank.get(_event_group(event) or "", len(events)),
                *chronological_key(event),
            )
        )
    else:
        kept.sort(key=chronological_key)

    sources = [
        {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "group": _event_group(event),
            "turn_index": _event_turn_index(event),
            "date": event.metadata.get("haystack_date"),
        }
        for event in kept
    ]
    return "\n".join(event.content for event in kept), sources


async def build_expanded_event_context(
    engram: Engram,
    *,
    agent_id: str,
    retrieved: Sequence[dict[str, Any]],
    n_sessions: int,
    max_tokens: int,
) -> str:
    session_best_rank: dict[str, int] = {}
    session_ids: dict[str, str] = {}
    for rank, item in enumerate(retrieved):
        meta = item.get("metadata", {})
        group = meta.get("original_session_id")
        session_id = item.get("session_id")
        if group is not None and group not in session_best_rank:
            session_best_rank[str(group)] = rank
            if session_id:
                session_ids[str(group)] = str(session_id)
    top_groups = sorted(session_best_rank, key=lambda s: session_best_rank[s])
    expanded = set(top_groups[:n_sessions])

    blocks: list[str] = []
    for group in top_groups[:n_sessions]:
        session_id = session_ids.get(group)
        if not session_id:
            continue
        events = await engram.list_events(
            agent_id=agent_id,
            session_id=session_id,
            limit=500,
        )
        events.sort(key=lambda event: _event_turn_index(event) or 0)
        if events:
            blocks.append("\n".join(event.content for event in events))

    leftover = [
        item["content"]
        for item in retrieved
        if str(item.get("metadata", {}).get("original_session_id")) not in expanded
    ]
    if leftover:
        blocks.append("Other relevant events:\n" + "\n".join(leftover))

    context = "\n\n".join(blocks)
    max_chars = max_tokens * 4
    if len(context) > max_chars:
        context = context[:max_chars]
    return context


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.insert(0, stripped[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


async def _verify_aggregation_ledger(
    engram: Engram,
    *,
    date_line: str,
    context: str,
    notes: str,
    question: str,
    slot_rules: str,
    draft_ledger: dict[str, Any],
    max_tokens: int,
) -> str:
    assert engram.llm is not None
    return await engram.llm.complete(
        f"{date_line}<memory_context>\n{context}\n</memory_context>\n\n"
        f"<evidence_notes>\n{notes}\n</evidence_notes>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<slot_rules>\n{slot_rules}\n</slot_rules>\n\n"
        "<draft_ledger>\n"
        f"{json.dumps(draft_ledger, ensure_ascii=False)}\n"
        "</draft_ledger>\n\n"
        "Audit and correct the ledger against the raw memory context. "
        "Return corrected JSON only, with no markdown or extra prose, using "
        "the same schema as the draft. The final answer will be computed "
        "from included rows, so missing or extra included rows directly "
        "make the answer wrong. For count, list, or total questions, every "
        "included row must be a distinct answer-bearing item, event, action, "
        "or amount the question asks for. Exclude assistant tips, generic "
        "examples, and anything outside the question's scope, moving it to "
        "the excluded list with a reason rather than dropping it silently. "
        "Merge only exact duplicates; keep genuinely separate items apart "
        "even when they share an entity or category. For a single-answer "
        "question about the current or latest state of a fact that changed "
        "over time, keep only the most recent value by timestamp and exclude "
        "earlier superseded values; never apply this recency rule to count, "
        "sum, or list questions.",
        system=(
            "You audit structured evidence ledgers for memory QA. Be strict "
            "about the requested slot and exhaustive about included rows."
        ),
        max_tokens=max(1536, max_tokens * 2),
        temperature=0.0,
    )


async def answer_from_evidence(
    engram: Engram,
    *,
    question: str,
    context: str,
    question_date: str | None = None,
    max_tokens: int = 256,
    reading: str = "direct",
) -> str:
    """Answer a question from an assembled memory evidence context.

    ``reading="con"`` first extracts evidence notes, builds an include/exclude
    aggregation ledger, then verifies the answer slot against both the ledger
    and raw context. Returns "" if the engram has no LLM provider configured.
    """
    if engram.llm is None:
        return ""
    if reading not in {"direct", "con"}:
        raise ValueError("reading must be 'direct' or 'con'")

    date_line = f"Current date: {question_date}\n" if question_date else ""
    system = (
        "Answer the user's question using only the supplied memory evidence. "
        "If the evidence is insufficient, say you do not know. Keep the "
        "answer concise."
    )
    slot_rules = (
        "Before answering, identify the exact slot requested by the "
        "question. Do not substitute a related source, storage place, date, "
        "or description for the requested entity. For where questions, "
        "answer the place, merchant, or location of the action; do not "
        "answer where a coupon, item, message, or information was found "
        "unless that is exactly what was asked. If two nearby statements in "
        "the same conversation identify the same event or entity, link them "
        "internally before answering."
    )

    if reading == "con":
        notes = await engram.llm.complete(
            f"{date_line}<memory_context>\n{context}\n</memory_context>\n\n"
            f"<question>\n{question}\n</question>\n\n"
            "<instructions>\n"
            "Work through the memory context turn by turn and quote every "
            "part that could bear on the question. Do not skip a relevant "
            "turn and do not summarize away detail. Then list the evidence, "
            "including: "
            "(1) candidate answers for the exact requested slot, "
            "(2) nearby same-conversation facts that identify the same "
            "event/entity, and (3) related but wrong-slot facts to avoid. "
            "For count, total, or list questions, extract every distinct "
            "candidate item/amount/event; do not stop after the first match. "
            "Include category synonyms and subtypes that satisfy the "
            "question, rather than relying only on exact surface words. "
            "If a single statement describes several distinct things the "
            "question asks about, extract each one separately. "
            "Do not answer yet. If nothing is relevant, write exactly: "
            "No relevant information.\n"
            "</instructions>",
            system=(
                "You extract and verify evidence from conversation memory. "
                "Be precise, exhaustive, and preserve dates. Miss nothing "
                "that could answer the question."
            ),
            max_tokens=max(2048, max_tokens * 2),
            temperature=0.0,
        )
        ledger = await engram.llm.complete(
            f"{date_line}<memory_context>\n{context}\n</memory_context>\n\n"
            f"<evidence_notes>\n{notes}\n</evidence_notes>\n\n"
            f"<question>\n{question}\n</question>\n\n"
            f"<slot_rules>\n{slot_rules}\n</slot_rules>\n\n"
            "Build an aggregation ledger before answering. Return JSON only, "
            "with no markdown or extra prose, in this shape: "
            '{"operation":"count|sum|list|single",'
            '"answer":"only for single-answer questions if known",'
            '"insufficient":false,'
            '"included":[{"entity":"...","action":"...","amount":null,'
            '"source_quote":"...","reason":"..."}],'
            '"excluded":[{"entity":"...","action":"...","amount":null,'
            '"source_quote":"...","reason":"..."}]}. '
            "The entity field must name the specific item, event, or "
            "amount; do not use placeholder labels. "
            "If the question asks how many, how much total, which items, or "
            "asks for a list, make one row per candidate evidence item, as "
            "either included or excluded. Deduplicate only true duplicates, "
            "and keep genuinely separate items apart even when they share an "
            "entity or category. Put anything that does not qualify in the "
            "excluded list with a reason rather than dropping it silently. "
            "The final count, sum, or list is computed from the included "
            "rows, so each included row must be a distinct item the question "
            "asks for. For totals, set the numeric amount on every included "
            "row. Treat category synonyms and subtypes as candidates when "
            "they satisfy the question. "
            "For single-answer questions, use "
            'operation "single" and set "answer" to the exact requested '
            "slot. When the question asks for the current or latest state of "
            "a fact that changed over time (a running total, status, amount, "
            "or value the user updated), take the value from the most recent "
            "memory by timestamp and mark earlier conflicting values as "
            "excluded with reason 'superseded by a later update'. Do not "
            "apply recency to count, sum, or list questions, where every "
            "qualifying item counts regardless of when it was mentioned.",
            system=(
                "You build evidence ledgers for memory QA. Be exhaustive, "
                "separate included from excluded evidence, and compute any "
                "count or total explicitly."
            ),
            max_tokens=max(1536, max_tokens * 2),
            temperature=0.0,
        )
        parsed_ledger = _parse_json_object(ledger)
        if parsed_ledger is not None:
            verified = await _verify_aggregation_ledger(
                engram,
                date_line=date_line,
                context=context,
                notes=notes,
                question=question,
                slot_rules=slot_rules,
                draft_ledger=parsed_ledger,
                max_tokens=max_tokens,
            )
            verified_ledger = _parse_json_object(verified)
            if verified_ledger is not None:
                ledger = json.dumps(verified_ledger, ensure_ascii=False)

        prompt = (
            f"{date_line}<memory_context>\n{context}\n</memory_context>\n\n"
            f"<evidence_notes>\n{notes}\n</evidence_notes>\n\n"
            f"<aggregation_ledger>\n{ledger}\n</aggregation_ledger>\n\n"
            f"<question>\n{question}\n</question>\n\n"
            f"<slot_rules>\n{slot_rules}\n</slot_rules>\n\n"
            "Answer strictly from the included rows of the aggregation "
            "ledger; ignore excluded rows. Derive the answer from the "
            "operation: for a count question, count the included rows; for "
            "a total, add the amounts on the included rows; for a list, "
            "name each included row; for a single-answer question, give the "
            "answer it records. Work the arithmetic out step by step over "
            "the rows so the count or total is exact. If the ledger is "
            "marked insufficient or has no included rows, reply exactly: "
            "I do not know. Give only the final concise answer.\nAnswer:"
        )
    else:
        prompt = (
            f"{date_line}<memory_context>\n{context}\n</memory_context>\n\n"
            f"<question>\n{question}\n</question>\n\n"
            f"<slot_rules>\n{slot_rules}\n</slot_rules>\n\n"
            "Give only the final concise answer.\nAnswer:"
        )

    return await engram.llm.complete(
        prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=0.0,
    )


async def maybe_generate_answer(
    engram: Engram,
    *,
    question: str,
    question_date: str | None,
    context: str,
    max_tokens: int,
    reading: str = "direct",
) -> str:
    """Generate an answer from retrieved context through the harness reader."""
    return await answer_from_evidence(
        engram,
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
            str(args.ingest_surface),
        ]
    )
    return hashlib.md5(raw.encode()).hexdigest()[:10]


async def purge_benchmark_agent(engram: Engram, agent_id: str) -> None:
    """Remove benchmark memories and raw events for a temporary agent."""
    await engram.purge(agent_id=agent_id)
    storage = getattr(engram, "_storage", None)
    if storage is not None:
        await storage.execute("DELETE FROM agent_events WHERE agent_id = $1", agent_id)
        await storage.execute(
            "DELETE FROM agent_sessions WHERE agent_id = $1", agent_id
        )
        await storage.execute("DELETE FROM agents WHERE agent_id = $1", agent_id)


async def count_events(engram: Engram, agent_id: str, expected: int) -> int:
    events = await engram.list_events(agent_id=agent_id, limit=expected + 1)
    return len(events)


async def ingest_events(
    engram: Engram,
    events: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> None:
    storage = getattr(engram, "_storage", None)
    if storage is not None:
        sessions: dict[str, dict[str, Any]] = {}
        for event in events:
            session_id = event.get("session_id")
            if not session_id:
                continue
            meta = dict(event.get("metadata") or {})
            sessions.setdefault(
                str(session_id),
                {
                    "agent_id": event["agent_id"],
                    "user_id": event.get("user_id"),
                    "metadata": {
                        "source": "longmemeval",
                        "question_id": meta.get("question_id"),
                        "question_type": meta.get("question_type"),
                        "original_session_id": meta.get("original_session_id"),
                        "haystack_date": meta.get("haystack_date"),
                    },
                },
            )
        for session_id, data in sessions.items():
            await storage.execute(
                """
                INSERT INTO agents (agent_id, name)
                VALUES ($1, $1)
                ON CONFLICT (agent_id) DO NOTHING
                """,
                data["agent_id"],
            )
            if data["user_id"]:
                await storage.execute(
                    """
                    INSERT INTO users (user_id)
                    VALUES ($1)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    data["user_id"],
                )
            await storage.execute(
                """
                INSERT INTO agent_sessions (session_id, agent_id, user_id, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (session_id) DO NOTHING
                """,
                session_id,
                data["agent_id"],
                data["user_id"],
                json.dumps(data["metadata"]),
            )
    task_memory = getattr(engram, "_task_memory", None)
    if task_memory is None:
        for event in events:
            await engram.record_event(**event)
        return

    from engram.task.models import EventCreate

    for batch in chunks(list(events), batch_size):
        creates = [
            EventCreate(
                agent_id=event["agent_id"],
                role=event["role"],
                event_type=event["event_type"],
                content=event["content"],
                task_run_id=event.get("task_run_id"),
                session_id=event.get("session_id"),
                user_id=event.get("user_id"),
                payload=event.get("payload") or {},
                metadata=event.get("metadata") or {},
            )
            for event in batch
        ]
        await task_memory.record_events(creates)


async def maybe_backfill_events(
    engram: Engram, args: argparse.Namespace, agent_id: str
) -> int:
    if not args.backfill_event_embeddings:
        return 0
    total = 0
    while True:
        count = await engram.backfill_event_embeddings(
            limit=args.event_backfill_batch_size,
            agent_id=agent_id,
        )
        total += count
        if count == 0:
            return total


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
    events = iter_events(
        sample,
        agent_id=agent_id,
        memory_unit=args.memory_unit,
        max_memory_chars=args.max_memory_chars,
    )
    use_memory_ingest = args.ingest_surface in {"memory", "both"}
    use_event_ingest = args.ingest_surface in {"event", "both"}

    try:
        ingested = True
        if args.reuse_store:
            memory_complete = True
            event_complete = True
            if use_memory_ingest and memories:
                existing = await engram.get_memories(agent_id, limit=len(memories) + 1)
                memory_complete = len(existing) >= len(memories)
            if use_event_ingest and events:
                event_complete = await count_events(
                    engram, agent_id, len(events)
                ) >= len(events)
            if memory_complete and event_complete:
                ingested = False  # store already complete; skip ingestion cost
            else:
                # Partial store from an interrupted run: rebuild from scratch.
                await purge_benchmark_agent(engram, agent_id)
        if ingested:
            if use_memory_ingest:
                for batch in chunks(memories, args.ingest_batch_size):
                    await engram.add_batch(batch)
            if use_event_ingest:
                await ingest_events(
                    engram,
                    events,
                    batch_size=args.ingest_batch_size,
                )

        event_embeddings_backfilled = 0
        if use_event_ingest:
            event_embeddings_backfilled = await maybe_backfill_events(
                engram, args, agent_id
            )

        event_results = []
        if args.retrieval_surface == "event":
            event_results = await search_event_evidence_set(
                engram,
                query=sample["question"],
                agent_id=agent_id,
                limit=args.limit,
                mode=args.mode,
            )
            results = []
        elif args.deep_search:
            results = await search_evidence_set(
                engram,
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
        if args.retrieval_surface == "event":
            retrieved = [_event_to_retrieved(event) for event in event_results]
        else:
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
            if args.retrieval_surface == "event":
                context = await build_expanded_event_context(
                    engram,
                    agent_id=agent_id,
                    retrieved=retrieved,
                    n_sessions=args.expand_sessions or args.limit,
                    max_tokens=args.max_context_tokens,
                )
            else:
                context = await build_expanded_context(
                    engram,
                    agent_id=agent_id,
                    retrieved=retrieved,
                    n_sessions=args.expand_sessions or args.limit,
                    max_tokens=args.max_context_tokens,
                )
        elif effective_context_strategy == "window":
            if args.retrieval_surface == "event":
                context, context_sources = await get_neighboring_event_context_block(
                    engram,
                    event_results,
                    agent_id,
                    before=args.evidence_window_size,
                    after=args.evidence_window_size,
                    include_session_start=True,
                    max_tokens=args.max_context_tokens,
                    prior_user_turns=args.prior_user_turns,
                    context_order="relevance",
                )
            else:
                aggregating = args.aggregation_full_session and is_aggregation_question(
                    sample["question"]
                )
                whole_k = (args.expand_sessions or 3) if aggregating else 0
                window_tokens = (
                    args.aggregation_context_tokens
                    if aggregating
                    else args.max_context_tokens
                )
                context, context_sources = await get_neighboring_context_block(
                    engram,
                    results,
                    agent_id,
                    before=args.evidence_window_size,
                    after=args.evidence_window_size,
                    include_session_start=True,
                    max_tokens=window_tokens,
                    prior_user_turns=args.prior_user_turns,
                    context_order="relevance",
                    whole_session_top_k=whole_k,
                )
        else:
            if args.retrieval_surface == "event":
                max_chars = args.max_context_tokens * 4
                context = "\n".join(item["content"] for item in retrieved)[:max_chars]
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
            "memory_count": len(memories) if use_memory_ingest else 0,
            "event_count": len(events) if use_event_ingest else 0,
            "ingested": ingested,
            "ingest_surface": args.ingest_surface,
            "retrieval_surface": args.retrieval_surface,
            "event_embeddings_backfilled": event_embeddings_backfilled,
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
            await purge_benchmark_agent(engram, agent_id)


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
        summary["ingest_surface"] = args.ingest_surface
        summary["retrieval_surface"] = args.retrieval_surface
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
        summary["backfill_event_embeddings"] = bool(args.backfill_event_embeddings)
        summary["event_backfill_batch_size"] = args.event_backfill_batch_size
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
    if args.event_backfill_batch_size < 1:
        raise SystemExit("--event-backfill-batch-size must be >= 1")
    if args.retrieval_surface == "event" and args.ingest_surface == "memory":
        raise SystemExit(
            "--retrieval-surface event requires --ingest-surface event or both"
        )
    if args.retrieval_surface == "memory" and args.ingest_surface == "event":
        raise SystemExit(
            "--retrieval-surface memory requires --ingest-surface memory or both"
        )
    if args.generate_answers and not os.getenv("ENGRAM_LLM_PROVIDER"):
        print(
            "Warning: --generate-answers was set but ENGRAM_LLM_PROVIDER is not set; "
            "hypotheses will be empty unless settings configure an LLM provider."
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
