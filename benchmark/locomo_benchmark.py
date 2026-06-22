#!/usr/bin/env python3
"""Run LoCoMo (ACL 2024) end-to-end through Engram's advanced APIs.

Pipeline (per conversation, one agent per conversation):
  1. INGEST: Render each session's turns into pre-formed memory rows and
     bulk-insert via ``add_batch()`` — embeddings only, no LLM extraction.
     Sessions are inserted chronologically.

  2. RETRIEVE (per question): 3-surface evidence gathering:
     a) search(mode="hybrid", include_superseded=True) — wide candidate pool,
        diversified across sessions (round-robin, user turns first).
     b) recall(compose_answer=False) — structured current/previous/conflict
        lineage evidence with temporal anchor.
     c) get_lineage() — superseded predecessors for full history.
     (traverse_many() graph traversal is off by default — ingest creates no
      edges, so it is a no-op unless edges are added via add_relation().)

  3. GENERATE: Composer LLM answers each question from the evidence block
     using LoCoMo-optimized prompts (adapted from the reference prompts.py).

  4. JUDGE: Independent LLM judge scores each answer CORRECT or WRONG using
     the official LoCoMo JSON rubric (category-aware partial credit).

  5. SCORE: Accuracy by category (1=multi-hop, 2=temporal, 3=open-domain,
     4=single-hop). Category 5 (adversarial) excluded per spec.

Dataset: locomo10.json auto-downloaded from GitHub if not present.
Outputs: JSONL traces, judgments, and summary.json in the output directory.

Usage:
    poetry run python benchmark/locomo_benchmark.py \\
        --output-dir benchmark/runs/locomo-bench

    # Specific conversations only:
    poetry run python benchmark/locomo_benchmark.py \\
        --conversations 0,1,2

    # Local embeddings (no API cost):
    poetry run python benchmark/locomo_benchmark.py \\
        --local-embedding

    # Re-judge an existing run:
    poetry run python benchmark/locomo_benchmark.py \\
        --rejudge-only benchmark/runs/locomo-bench/traces.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment bootstrap (mirrors longmemeval_benchmark.py exactly)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

if os.environ.get("EMBEDDING_PROVIDER") and "ENGRAM_EMBEDDING_PROVIDER" not in os.environ:
    os.environ["ENGRAM_EMBEDDING_PROVIDER"] = os.environ["EMBEDDING_PROVIDER"]
if os.environ.get("OPENAI_API_KEY") and "ENGRAM_OPENAI_API_KEY" not in os.environ:
    os.environ["ENGRAM_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
if os.environ.get("ANTHROPIC_API_KEY") and "ENGRAM_ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ENGRAM_ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
if os.environ.get("ENGRAM_GEMENI_API_KEY") and "ENGRAM_GEMINI_API_KEY" not in os.environ:
    os.environ["ENGRAM_GEMINI_API_KEY"] = os.environ["ENGRAM_GEMENI_API_KEY"]
if os.environ.get("GEMINI_API_KEY") and "ENGRAM_GEMINI_API_KEY" not in os.environ:
    os.environ["ENGRAM_GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"]

if "ENGRAM_LLM_PROVIDER" not in os.environ:
    if os.environ.get("ENGRAM_ANTHROPIC_API_KEY"):
        os.environ["ENGRAM_LLM_PROVIDER"] = "anthropic"
    elif os.environ.get("ENGRAM_OPENAI_API_KEY"):
        os.environ["ENGRAM_LLM_PROVIDER"] = "openai"

from cost_tracking import CostAccumulator, count_tokens  # noqa: E402

from engram import Engram  # noqa: E402
from engram.core.config import get_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
# DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"
# DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"

DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

LOCAL_EMBEDDING_PROVIDER = "sentence-transformers"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIMENSION = 384

BENCHMARK_POLICY = "default"
DEFAULT_MAX_MEMORY_CHARS = 16000
DEFAULT_INGEST_BATCH_SIZE = 64

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]  # category 5 excluded per spec


# ===========================================================================
# Prompts
# ===========================================================================

# Adapted from the reference prompts.py ANSWER_GENERATION_PROMPT.
# Key changes for Engram: memories arrive as an evidence block with
# [ACTIVE]/[SUPERSEDED] tags rather than a flat chronological list,
# and the CURRENT:/SUPERSEDED: structured lineage from recall is prepended.
LOCOMO_COMPOSER_SYSTEM = """You are answering questions using retrieved memories from past conversations. Memories are tagged:
- [ACTIVE] — current, valid information
- [SUPERSEDED] — an older value since updated; use only for "what was it before?" questions
- CURRENT: / SUPERSEDED (until date): — structured recall lineage (highest authority)

IMPORTANT: The reference date is {reference_date}. All temporal reasoning must be relative to this date. Events occurred in 2022-2024. Never output 2025 or 2026.

Follow these steps IN ORDER:

## Step 1: SCAN ALL MEMORIES
Read EVERY memory. For each one relevant to the question, note it. Do NOT stop after the first relevant memory — important details are scattered. Give equal weight to all memories regardless of position.

## Step 2: ACTIVE vs SUPERSEDED
For current-state questions, use [ACTIVE] and CURRENT: memories only.
For "what was it before?" questions, use [SUPERSEDED] memories.
Most recent [ACTIVE] wins for conflicting values — use the latest.

## Step 3: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity.
In two-person conversations, both speakers' actions are relevant — check attribution carefully.

## Step 4: COMBINE AND CROSS-REFERENCE
- Combine facts from multiple memories about the same topic.
- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items.
- For counting: enumerate each distinct instance explicitly with date or context BEFORE giving a final count. List them out, then count the list.
- Connect related facts: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking," the nearby lake IS Lake Tahoe.
- Decompose complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts.

## Step 5: SELECT THE BEST ANSWER
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description.
- Report what someone actually DID, not what was offered or available.
- "Has not tried X yet" means X was NOT done. "Joined X" means it WAS done.
- When multiple memories repeat the same generic fact, a single memory with a specific answer wins.

## Step 6: TEMPORAL GROUNDING
- Use dates explicitly stated in memory text. Do not invent or estimate dates.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with dates, then pick the instance closest to (and before) the reference date.
- For "how long" questions, find start and end dates explicitly, then compute duration.

## Step 7: INCLUDE ALL RELEVANT ITEMS
If you found items during reasoning you're tempted to exclude — STOP. Include them unless you have STRONG evidence they are wrong. More items is better than fewer when there is supporting evidence.

## Step 8: COMMIT AND ANSWER
Give a direct, specific answer after "ANSWER:". NEVER say "not specified," "not mentioned," or "no record" — if ANY memory contains relevant information, answer from available evidence. No hedging, no caveats.

NEVER generate specific names, titles, places, or dates that do not appear in any memory.

For open-domain/opinion questions: Follow direct causal reasoning in memories. If memories show X does Y BECAUSE of Z, then without Z answer "likely no." A recent negative experience outweighs historical positive patterns."""


JUDGE_SYSTEM = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."

JUDGE_PROMPT = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. "Chocolate raspberry tart" = "chocolate cake with raspberries". "Shelter meal service" = "volunteering at a homeless shelter". Emotions in the same positive/negative family count: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" = "thrilled".

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"; "19 days" matches "two weeks"). Converting "last year" to the actual year (e.g., "2022" when conversations are in 2023) is CORRECT.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches.

6. **SAME REFERENT**: If the generated answer mentions the same named entity as the gold answer, mark CORRECT — even if additional details differ.

7. **FOCUS ON KNOWLEDGE**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 3)


def _preprocess_answer(category: int, answer: str) -> str:
    """Category 3 (open-domain): use only the part before the first semicolon."""
    if category == 3 and ";" in answer:
        return answer.split(";")[0].strip()
    return answer


# ===========================================================================
# Dataset
# ===========================================================================

def download_dataset(data_dir: Path) -> Path:
    """Download locomo10.json from GitHub if not present."""
    path = data_dir / "locomo10.json"
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LoCoMo-10 dataset to {path}...")
    urllib.request.urlretrieve(DATASET_URL, path)
    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) == 0:
        path.unlink()
        raise RuntimeError(f"Invalid dataset at {path}")
    print(f"Downloaded {len(data)} conversations.")
    return path


# ===========================================================================
# Conversation parsing
# ===========================================================================

def parse_locomo_date(date_str: str) -> datetime | None:
    """Parse LoCoMo date: '1:56 pm on 8 May, 2023'."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _locomo_date_to_human(date_str: str) -> str:
    """'1:56 pm on 8 May, 2023' → 'May 8, 2023'."""
    dt = parse_locomo_date(date_str)
    return dt.strftime("%B %d, %Y") if dt else date_str


def get_sorted_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """Extract and sort sessions chronologically.

    Returns list of (session_key, date_str, turns).
    """
    session_keys = [k for k in conversation if re.match(r"^session_\d+$", k)]
    paired = []
    for key in session_keys:
        date_str = conversation.get(f"{key}_date_time", "")
        turns = conversation[key]
        if isinstance(turns, list):
            paired.append((key, date_str, turns))

    def sort_key(item: tuple) -> tuple[int, float]:
        parsed = parse_locomo_date(item[1])
        if parsed:
            return (0, parsed.timestamp())
        m = re.search(r"\d+", item[0])
        return (1, float(m.group()) if m else 0.0)

    paired.sort(key=sort_key)
    return paired


def _render_turn(turn: dict, date_str: str, speaker_a: str) -> str:
    """Render a LoCoMo turn as '[date] Speaker: text [photo]'."""
    speaker = turn.get("speaker", "")
    text = turn.get("text", "")
    blip = turn.get("blip_caption", "")
    query = turn.get("query", "")

    if query and blip:
        photo = f" [Photo: {query} — {blip}]"
    elif query:
        photo = f" [Photo: {query}]"
    elif blip:
        photo = f" [Photo: {blip}]"
    else:
        photo = ""

    content = f"{text}{photo}".strip()
    if not content:
        return ""

    human_date = _locomo_date_to_human(date_str) if date_str else "unknown date"
    return f"[{human_date}] {speaker}: {content}"


# ===========================================================================
# Memory row building
# ===========================================================================

def build_memory_rows(
    conversation: dict,
    conv_idx: int,
    *,
    agent_id: str,
    user_id: str,
    max_memory_chars: int,
) -> list[dict[str, Any]]:
    """Render a LoCoMo conversation into add_batch() row dicts.

    Each turn becomes one episodic memory with the date anchored in the
    text. Sessions are ordered chronologically so last-writer-wins
    supersession resolves correctly.
    """
    rows: list[dict[str, Any]] = []
    speaker_a = conversation.get("speaker_a", "")

    for session_key, date_str, turns in get_sorted_sessions(conversation):
        for turn_index, turn in enumerate(turns):
            rendered = _render_turn(turn, date_str, speaker_a)
            if not rendered:
                continue

            if max_memory_chars > 0 and len(rendered) > max_memory_chars:
                rendered = rendered[:max_memory_chars]

            speaker = turn.get("speaker", "")
            role = "user" if speaker == speaker_a else "assistant"

            rows.append({
                "content": rendered,
                "main_content": rendered,
                "agent_id": agent_id,
                "user_id": user_id,
                "memory_type": "episodic",
                "metadata": {
                    "source": "locomo",
                    "conversation_idx": conv_idx,
                    "original_session_id": session_key,
                    "haystack_date": date_str,
                    "turn_index": turn_index,
                    "turn_role": role,
                    "speaker": speaker,
                },
            })

    return rows


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ===========================================================================
# Ingestion
# ===========================================================================

async def ingest_conversation(
    engram: Engram,
    conversation: dict,
    conv_idx: int,
    agent_id: str,
    user_id: str,
    *,
    max_memory_chars: int,
    batch_size: int,
) -> dict[str, Any]:
    """Bulk-ingest all sessions of a conversation via add_batch()."""
    rows = build_memory_rows(
        conversation, conv_idx,
        agent_id=agent_id,
        user_id=user_id,
        max_memory_chars=max_memory_chars,
    )
    full_context_text = "\n".join(
        (turn.get("text", "") or "") + (turn.get("blip_caption", "") or "")
        for _, _, turns in get_sorted_sessions(conversation)
        for turn in turns
    )
    full_context_chars = len(full_context_text)
    full_context_tokens = count_tokens(full_context_text)

    t0 = time.perf_counter()
    inserted = 0
    batches = 0
    errors = 0
    first_error: str | None = None

    for batch in _chunks(rows, batch_size):
        batches += 1
        try:
            affected = await engram.add_batch(batch)
            inserted += len(affected)
        except Exception as exc:
            errors += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"

    return {
        "rows": len(rows),
        "inserted": inserted,
        "batches": batches,
        "errors": errors,
        "first_error": first_error,
        "full_context_chars": full_context_chars,
        "full_context_tokens": full_context_tokens,
        "ingest_seconds": round(time.perf_counter() - t0, 3),
    }


# ===========================================================================
# Evidence Building (same as longmemeval_benchmark.py)
# ===========================================================================

def _to_human_date(date_str: str) -> str:
    return _locomo_date_to_human(date_str) if date_str else date_str


def _build_evidence_block(
    search_results: list[Any],
    recall_answer: Any | None,
    graph_context: str,
    lineage_superseded: list[Any] | None = None,
) -> str:
    lines: list[str] = []
    seen: set[str] = set()

    if recall_answer is not None:
        cur = getattr(recall_answer, "current", None)
        if cur is not None:
            seen.add(cur.memory_id)
            lines.append(f"CURRENT: {cur.fact or cur.content}")
        for mem in getattr(recall_answer, "previous", []) or []:
            when = mem.superseded_at or mem.valid_to or mem.created_at
            stamp = when.date().isoformat() if when else "unknown"
            seen.add(mem.memory_id)
            lines.append(f"SUPERSEDED (until {stamp}): {mem.fact or mem.content}")
        if getattr(recall_answer, "conflict_note", None):
            lines.append(f"CONFLICT: {recall_answer.conflict_note}")
        if getattr(recall_answer, "answer_text", "").strip():
            lines.append(f"RECALL NOTE: {recall_answer.answer_text.strip()}")

    for mem in lineage_superseded or []:
        if mem.memory_id in seen:
            continue
        seen.add(mem.memory_id)
        when = mem.superseded_at or mem.valid_to or mem.created_at
        stamp = when.date().isoformat() if when else "unknown"
        lines.append(f"SUPERSEDED (until {stamp}): {mem.fact or mem.content}")

    lines.append("\n## RETRIEVED MEMORIES (hybrid search)")
    current_date = None
    for result in search_results:
        mem = result.memory
        if mem.memory_id in seen:
            continue
        seen.add(mem.memory_id)
        mem_date = mem.metadata.get("haystack_date", "Unknown Date")
        if mem_date != current_date:
            lines.append(f"\n--- {_to_human_date(mem_date)} ---")
            current_date = mem_date
        status = getattr(mem, "status", None) or mem.metadata.get("status", "active")
        tag = "[SUPERSEDED]" if status == "superseded" else "[ACTIVE]"
        lines.append(f"- {tag} {mem.content}")

    if graph_context:
        lines.append(f"\n{graph_context}")

    return "\n".join(lines) if lines else "(no matching memory)"


def diversify_by_session(
    results: list[Any],
    *,
    limit: int,
    max_per_session: int,
    rerank: bool = False,
) -> list[Any]:
    """Round-robin candidate pool across sessions, user turns first."""
    if not results:
        return results

    def role_bias(r: Any) -> float:
        if rerank:
            return 0.0
        role = str(r.memory.metadata.get("turn_role", "")).lower()
        if role == "user":
            return -1.0
        if role in ("assistant", "system", "tool"):
            return 1.0
        return 0.0

    def group_key(r: Any) -> str:
        sid = r.memory.metadata.get("original_session_id")
        return str(sid) if sid is not None else r.memory.memory_id

    ranked = [
        r for _, r in sorted(
            enumerate(results),
            key=lambda it: (it[0] + role_bias(it[1]), it[0]),
        )
    ]
    groups: dict[str, list[Any]] = {}
    for r in ranked:
        groups.setdefault(group_key(r), []).append(r)

    selected: list[Any] = []
    seen: set[str] = set()
    for depth in range(max_per_session):
        for group in groups.values():
            if depth < len(group):
                r = group[depth]
                selected.append(r)
                seen.add(r.memory.memory_id)
                if len(selected) >= limit:
                    return selected
    for r in ranked:
        if r.memory.memory_id in seen:
            continue
        selected.append(r)
        if len(selected) >= limit:
            break
    return selected


# ===========================================================================
# Retrieval (same structure as longmemeval_benchmark.py)
# ===========================================================================

async def retrieve_evidence(
    engram: Engram,
    question: str,
    agent_id: str,
    user_id: str,
    search_limit: int,
    graph_depth: int,
    max_per_session: int,
    reference_date: str = "",
    rerank: bool = False,
    candidate_limit: int = 100,
) -> tuple[str, dict[str, Any]]:
    """Retrieve evidence using Engram's 4-surface APIs."""
    t0 = time.perf_counter()
    error = None
    evidence = ""
    n_search_hits = 0
    n_graph_hits = 0
    n_lineage_superseded = 0
    recall_intent = ""

    try:
        candidate_limit = (
            candidate_limit
            if rerank
            else min(max(search_limit * 3, search_limit), candidate_limit)
        )
        candidates = await engram.search(
            query=question,
            agent_id=agent_id,
            user_id=user_id,
            limit=candidate_limit,
            mode="hybrid",
            rerank=rerank,
            include_superseded=True,
        )
        search_results = diversify_by_session(
            candidates, limit=search_limit, max_per_session=max_per_session,
            rerank=rerank,
        )
        n_search_hits = len(search_results)

        search_results.sort(key=lambda r: r.memory.metadata.get("haystack_date", ""))

        recall_answer = None
        try:
            recall_answer = await engram.recall(
                question,
                agent_id,
                user_id=user_id,
                question_date=parse_locomo_date(reference_date),
                limit=max(search_limit // 2, 10),
                compose_answer=False,
            )
            recall_intent = getattr(recall_answer, "intent", "")
        except Exception as exc:
            recall_intent = f"error:{type(exc).__name__}"

        lineage_superseded: list[Any] = []
        seen_lineages: set[str] = set()
        for r in search_results:
            mem = r.memory
            lid = getattr(mem, "lineage_id", None)
            status = getattr(mem, "status", None) or mem.metadata.get("status", "active")
            if (
                status != "superseded"
                and lid
                and lid != mem.memory_id
                and lid not in seen_lineages
            ):
                seen_lineages.add(lid)
                try:
                    lineage = await engram.get_lineage(mem.memory_id)
                    lineage_superseded.extend(
                        m for m in lineage.memories
                        if (getattr(m, "status", None) == "superseded")
                    )
                except Exception:
                    pass
        n_lineage_superseded = len(lineage_superseded)

        graph_context = ""
        seed_ids = [
            r.memory.memory_id for r in search_results
            if r.memory.metadata.get("status") != "superseded"
        ][:5]
        if seed_ids and graph_depth > 0:
            try:
                graph_results = await engram.traverse_many(
                    start_memory_ids=seed_ids,
                    max_depth=graph_depth,
                    direction="any",
                    skip_missing=True,
                )
                if graph_results:
                    graph_context = engram.render_graph_context(graph_results)
                    n_graph_hits = len(graph_results)
            except Exception:
                pass

        evidence = _build_evidence_block(
            search_results, recall_answer, graph_context, lineage_superseded
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return evidence, {
        "search_hits": n_search_hits,
        "graph_hits": n_graph_hits,
        "lineage_superseded": n_lineage_superseded,
        "recall_intent": recall_intent,
        "retrieval_seconds": round(time.perf_counter() - t0, 3),
        "error": error,
    }


# ===========================================================================
# Answer Generation
# ===========================================================================

async def generate_answer(
    engram: Engram,
    question: str,
    reference_date_human: str,
    evidence: str,
    max_tokens: int,
) -> tuple[str, str, dict[str, int]]:
    """Generate answer from evidence using Engram's LLM.

    Returns (answer, model, usage) where usage is the provider's real token
    accounting for billed-cost reporting.
    """
    assert engram.llm is not None

    system = LOCOMO_COMPOSER_SYSTEM.format(reference_date=reference_date_human or "2023")
    user_prompt = (
        f"<engram_memory_evidence>\n{evidence}\n</engram_memory_evidence>\n\n"
        f"Reference Date: {reference_date_human or '2023'}\n"
        f"Question: {question}\n\n"
        "Work through Steps 1-8, then give your final answer after 'ANSWER:'."
    )
    resp = await engram.llm.complete_full(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    raw = resp.content.strip()
    answer = raw.rsplit("ANSWER:", 1)[-1].strip() if "ANSWER:" in raw else raw
    return answer, resp.model, resp.usage or {}


# ===========================================================================
# Judging
# ===========================================================================

def _parse_json_judgment(raw: str) -> bool:
    """Parse CORRECT/WRONG from judge JSON response."""
    text = (raw or "").strip()
    if not text:
        return False
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Direct JSON parse
    try:
        data = json.loads(text)
        return str(data.get("label", "")).upper() == "CORRECT"
    except json.JSONDecodeError:
        pass
    # Extract embedded JSON object
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return str(data.get("label", "")).upper() == "CORRECT"
        except json.JSONDecodeError:
            pass
    # Keyword fallback
    upper = text.upper()
    if re.search(r'\bCORRECT\b', upper) and not re.search(r'\bWRONG\b', upper):
        return True
    if re.search(r'\bWRONG\b', upper):
        return False
    return False


def _judge_temp(model: str) -> dict[str, float]:
    """opus-4-8 (and newer) reject the `temperature` param; omit it there and keep
    deterministic temperature=0 for models that still accept it."""
    return {} if "opus-4-8" in model.lower() else {"temperature": 0.0}


async def judge_all(
    answer_traces: list[dict[str, Any]],
    judge_model: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Judge answers with an independent LLM (JSON CORRECT/WRONG output)."""
    use_anthropic = judge_model.lower().startswith("claude")

    if use_anthropic:
        from anthropic import AsyncAnthropic
        api_key = (
            os.environ.get("ENGRAM_ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise SystemExit("Set ENGRAM_ANTHROPIC_API_KEY for the Claude judge.")
        client: Any = AsyncAnthropic(api_key=api_key)
    else:
        from openai import AsyncOpenAI
        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ENGRAM_OPENAI_API_KEY")
        )
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY for the judge.")
        client = AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(concurrency)

    async def verdict_for(prompt: str) -> tuple[str, dict[str, int]]:
        async with sem:
            if use_anthropic:
                resp = await client.messages.create(
                    model=judge_model,
                    max_tokens=256,
                    **_judge_temp(judge_model),
                    system=JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                usage = {
                    "input_tokens": getattr(resp.usage, "input_tokens", 0),
                    "output_tokens": getattr(resp.usage, "output_tokens", 0),
                }
                return " ".join(parts).strip(), usage
            resp = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            u = getattr(resp, "usage", None)
            usage = {
                "input_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
                "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
            }
            return (resp.choices[0].message.content or "").strip(), usage

    async def judge_one(trace: dict[str, Any]) -> dict[str, Any]:
        if trace.get("error"):
            return {
                "question_id": trace["question_id"],
                "category": trace.get("category"),
                "category_name": trace.get("category_name", ""),
                "correct": False,
                "verdict": "error",
                "judge_raw": "",
                "judge_model": judge_model,
                "judge_usage": {},
            }
        prompt = JUDGE_PROMPT.format(
            question=trace["question"],
            answer=str(trace["answer"]),
            response=trace["hypothesis"],
        )
        raw, judge_usage = await verdict_for(prompt)
        correct = _parse_json_judgment(raw)
        return {
            "question_id": trace["question_id"],
            "category": trace.get("category"),
            "category_name": trace.get("category_name", ""),
            "correct": correct,
            "verdict": "CORRECT" if correct else "WRONG",
            "judge_raw": raw[:400],
            "judge_model": judge_model,
            "judge_usage": judge_usage,
        }

    return await asyncio.gather(*(judge_one(t) for t in answer_traces))


# ===========================================================================
# Conversation runner
# ===========================================================================

async def run_conversation(
    engram: Engram,
    entry: dict,
    conv_idx: int,
    *,
    agent_id: str,
    user_id: str,
    search_limit: int,
    graph_depth: int,
    answer_max_tokens: int,
    max_per_session: int,
    max_memory_chars: int,
    ingest_batch_size: int,
    categories: list[int],
    max_questions: int | None,
    rerank: bool = False,
    candidate_limit: int = 100,
    question_filter: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ingest a conversation's sessions, then answer all in-scope questions.

    Returns (ingest_summary, list_of_traces).
    """
    conversation = entry["conversation"]
    sorted_sessions = get_sorted_sessions(conversation)

    # Reference date = last session's date (most recent context for temporal grounding)
    reference_date_raw = sorted_sessions[-1][1] if sorted_sessions else ""
    reference_date_human = _locomo_date_to_human(reference_date_raw) if reference_date_raw else "2023"

    # 1. INGEST all sessions
    ingest_summary = await ingest_conversation(
        engram, conversation, conv_idx, agent_id, user_id,
        max_memory_chars=max_memory_chars,
        batch_size=ingest_batch_size,
    )

    print(
        f"[conv{conv_idx}] ingested={ingest_summary.get('inserted')}/"
        f"{ingest_summary.get('rows')} in {ingest_summary.get('ingest_seconds')}s"
    )

    questions = entry.get("qa", entry.get("qa_pairs", []))
    traces: list[dict[str, Any]] = []

    if ingest_summary.get("errors"):
        # Partial ingest — mark all questions errored rather than score them wrong
        err_msg = (
            f"incomplete ingestion: {ingest_summary.get('inserted')}/"
            f"{ingest_summary.get('rows')} rows, {ingest_summary['errors']} error(s); "
            f"first: {ingest_summary.get('first_error')}"
        )
        for qi, qa in enumerate(questions):
            if qa.get("category") not in categories:
                continue
            if max_questions is not None and len(traces) >= max_questions:
                break
            category = qa.get("category", 0)
            traces.append({
                "question_id": f"conv{conv_idx}_q{qi}",
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "unknown"),
                "question": qa.get("question", ""),
                "answer": _preprocess_answer(category, str(qa.get("answer", ""))),
                "reference_date": reference_date_human,
                "hypothesis": "",
                "conversation_idx": conv_idx,
                "agent_id": agent_id,
                "composer_model": "",
                "ingest_summary": ingest_summary,
                "retrieval": {},
                "elapsed_seconds": 0.0,
                "error": err_msg,
            })
        return ingest_summary, traces

    # 2. Answer each in-scope question
    for qi, qa in enumerate(questions):
        if qa.get("category") not in categories:
            continue
        if question_filter is not None and f"conv{conv_idx}_q{qi}" not in question_filter:
            continue
        if max_questions is not None and len(traces) >= max_questions:
            break

        category = qa.get("category", 0)
        raw_answer = str(qa.get("answer", ""))
        processed_answer = _preprocess_answer(category, raw_answer)
        question = qa.get("question", "")
        start = time.perf_counter()
        error = None
        hypothesis = ""
        composer_model = ""
        composer_usage: dict[str, int] = {}
        retrieval_trace: dict[str, Any] = {}
        evidence = ""

        try:
            evidence, retrieval_trace = await retrieve_evidence(
                engram, question, agent_id, user_id,
                search_limit=search_limit,
                graph_depth=graph_depth,
                max_per_session=max_per_session,
                reference_date=reference_date_raw,
                rerank=rerank,
                candidate_limit=candidate_limit,
            )
            if engram.llm is not None:
                hypothesis, composer_model, composer_usage = await generate_answer(
                    engram, question, reference_date_human, evidence,
                    max_tokens=answer_max_tokens,
                )
            else:
                error = "No LLM configured"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        _elapsed = round(time.perf_counter() - start, 3)
        _retrieval_s = retrieval_trace.get("retrieval_seconds", 0.0)
        status = "ERR" if error else "OK"
        retr = retrieval_trace
        print(
            f"  [conv{conv_idx}_q{qi}] {status} "
            f"cat={CATEGORY_NAMES.get(category, category)} "
            f"search={retr.get('search_hits', '?')} "
            f"recall={retr.get('recall_intent', '?')} "
            f"| {_elapsed}s"
        )
        traces.append({
            "question_id": f"conv{conv_idx}_q{qi}",
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, "unknown"),
            "question": question,
            "answer": processed_answer,
            "reference_date": reference_date_human,
            "hypothesis": hypothesis,
            "conversation_idx": conv_idx,
            "agent_id": agent_id,
            "composer_model": composer_model,
            "ingest_summary": ingest_summary,
            "retrieval": retrieval_trace,
            "evidence": evidence,
            "evidence_chars": len(evidence),
            "evidence_tokens": count_tokens(evidence),
            "composer_usage": composer_usage,
            "generation_seconds": max(0.0, round(_elapsed - _retrieval_s, 3)),
            "elapsed_seconds": _elapsed,
            "error": error,
        })

    return ingest_summary, traces


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoCoMo benchmark through Engram's advanced APIs."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=_REPO_ROOT / "data" / "locomo" / "locomo10.json",
        help="Path to locomo10.json (auto-downloaded if missing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "benchmark" / "runs" / "locomo-benchmark",
    )
    parser.add_argument("--agent-prefix", default="locomo")
    parser.add_argument("--user-id", default="locomo-user")
    parser.add_argument(
        "--conversations",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated conversation indices (default: all 10).",
    )
    parser.add_argument(
        "--categories",
        default="1,2,3,4",
        help="Comma-separated categories (1=multi-hop 2=temporal 3=open-domain 4=single-hop; 5=adversarial excluded).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Independent LLM judge model (default {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Conversations processed concurrently (default 3).",
    )
    parser.add_argument(
        "--llm-model", default=DEFAULT_LLM_MODEL,
        help=f"Composer LLM model (default {DEFAULT_LLM_MODEL}).",
    )
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dimension", type=int, default=DEFAULT_EMBEDDING_DIMENSION)
    parser.add_argument(
        "--local-embedding", action="store_true",
        help=f"Use on-device sentence-transformers ({LOCAL_EMBEDDING_MODEL}, {LOCAL_EMBEDDING_DIMENSION}d).",
    )
    parser.add_argument("--max-memory-chars", type=int, default=DEFAULT_MAX_MEMORY_CHARS)
    parser.add_argument("--ingest-batch-size", type=int, default=DEFAULT_INGEST_BATCH_SIZE)
    parser.add_argument(
        "--search-limit", type=int, default=100,
        help="Memories per question after session-diversified selection.",
    )
    parser.add_argument(
        "--max-per-session", type=int, default=6,
        help="Max turns kept per session in the diversified evidence set.",
    )
    parser.add_argument(
        "--candidate-limit", type=int, default=100,
        help="Reranked candidates returned per question before diversification.",
    )
    parser.add_argument(
        "--max-search-limit", type=int, default=500,
        help="Pre-rerank candidate pool the cross-encoder scores (overfetch "
        "depth). Decoupled from --candidate-limit so recall can go deep without "
        "bloating the evidence window.",
    )
    parser.add_argument(
        "--question-id", action="append",
        help="Answer only this question_id (e.g. conv0_q104). Repeatable. "
        "Full conversations are still ingested; only these questions are scored.",
    )
    parser.add_argument("--graph-depth", type=int, default=0)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--answer-max-tokens", type=int, default=4000,
        help="Max tokens for composer LLM (generous for 8-step reasoning).",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Max questions per conversation (for quick testing).",
    )
    parser.add_argument("--no-purge", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--rejudge-only",
        type=Path,
        help="Re-score an existing traces.jsonl (no ingestion, no answering).",
    )
    parser.add_argument("--clean-db", action="store_true")

    args = parser.parse_args()

    if args.local_embedding:
        args.embedding_provider = LOCAL_EMBEDDING_PROVIDER
        if args.embedding_model == DEFAULT_EMBEDDING_MODEL:
            args.embedding_model = LOCAL_EMBEDDING_MODEL
        if args.embedding_dimension == DEFAULT_EMBEDDING_DIMENSION:
            args.embedding_dimension = LOCAL_EMBEDDING_DIMENSION

    return args


# ===========================================================================
# Scorecard helpers
# ===========================================================================

def _print_scorecard(judgments: list[dict[str, Any]]) -> None:
    total = len(judgments)
    correct = sum(j["correct"] for j in judgments)
    by_cat: dict[str, list[int]] = {}
    for j in judgments:
        cat = j.get("category_name", "unknown")
        if cat not in by_cat:
            by_cat[cat] = [0, 0]
        by_cat[cat][0] += 1
        by_cat[cat][1] += int(j["correct"])

    print("\n" + "=" * 60)
    if total:
        print(f"OVERALL  : {correct}/{total} = {correct / total * 100:.1f}%")
    print("\nby category:")
    for cat_name, (n, c) in sorted(by_cat.items()):
        pct = c / n * 100 if n else 0.0
        print(f"  {cat_name:15s} {c}/{n} = {pct:.1f}%")


# ===========================================================================
# Main
# ===========================================================================

async def rejudge_only(args: argparse.Namespace) -> None:
    traces = [
        json.loads(line)
        for line in args.rejudge_only.read_text().splitlines()
        if line.strip()
    ]
    print(f"re-judging {len(traces)} answers from {args.rejudge_only} with {args.judge_model}...")
    judgments = await judge_all(traces, args.judge_model, args.judge_concurrency)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(j) for j in judgments)
    )
    _print_scorecard(judgments)
    print(f"\njudgments written to {args.output_dir / 'judgments.jsonl'}")


async def main() -> None:
    args = parse_args()

    if args.rejudge_only is not None:
        await rejudge_only(args)
        return

    if not args.data_path.exists():
        args.data_path = download_dataset(args.data_path.parent)

    dataset = json.loads(args.data_path.read_text())
    if not isinstance(dataset, list):
        raise SystemExit(f"Expected a JSON list in {args.data_path}")

    conv_indices = [int(c) for c in args.conversations.split(",")]
    categories = [int(c) for c in args.categories.split(",")]

    # --question-id restricts scoring to specific questions. Derive the
    # conversations they belong to so we only ingest what we need.
    question_filter: set[str] | None = None
    if args.question_id:
        question_filter = set(args.question_id)
        wanted_convs = {int(q.split("_q")[0].removeprefix("conv")) for q in question_filter}
        conv_indices = [c for c in conv_indices if c in wanted_convs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]

    settings = get_settings()
    settings = settings.model_copy(update={
        "llm_provider": "anthropic",
        "llm_model": args.llm_model,
        "anthropic_api_key": (
            os.environ.get("ENGRAM_ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        ),
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "allow_embedding_dimension_change": True,
        "near_duplicate_threshold": 1.0,
        "max_search_limit": args.max_search_limit,
    })

    engram = Engram(settings=settings, memory_policy=BENCHMARK_POLICY)
    await engram.connect()

    if args.clean_db:
        print("Cleaning database...")
        storage = getattr(engram, "_storage", None)
        if storage is not None:
            await storage.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            embedding = getattr(engram, "_embedding", None)
            dim = embedding.dimension if embedding is not None else settings.embedding_dimension
            await storage.init_schema(embedding_dimension=dim)

    if engram.llm is None:
        raise SystemExit(
            "No LLM provider configured. Set ENGRAM_LLM_PROVIDER=anthropic and "
            "ENGRAM_ANTHROPIC_API_KEY in .env."
        )

    print(f"LLM      : {settings.llm_provider} / {settings.llm_model}")
    print(f"Embedding: {settings.embedding_provider} / {settings.embedding_model}")
    print(f"Judge    : {args.judge_model}")
    print(f"Rerank   : {args.rerank}  Search-limit: {args.search_limit}  Graph-depth: {args.graph_depth}")
    print(f"Convs    : {conv_indices}  Categories: {[CATEGORY_NAMES.get(c, c) for c in categories]}")

    all_traces: list[dict[str, Any]] = []
    traces_path = args.output_dir / "traces.jsonl"

    try:
        with traces_path.open("w") as traces_file:
            sem = asyncio.Semaphore(args.concurrency)

            async def process_one(conv_idx: int) -> None:
                async with sem:
                    if conv_idx >= len(dataset):
                        print(f"[conv{conv_idx}] out of range, skipping")
                        return

                    entry = dataset[conv_idx]
                    conversation = entry["conversation"]
                    speaker_a = conversation.get("speaker_a", "?")
                    speaker_b = conversation.get("speaker_b", "?")
                    agent_id = f"{args.agent_prefix}-{run_id}-conv{conv_idx}"

                    questions = entry.get("qa", entry.get("qa_pairs", []))
                    n_q = sum(1 for qa in questions if qa.get("category") in categories)
                    print(f"\n{'=' * 60}\n[conv{conv_idx}] {speaker_a} & {speaker_b} | {n_q} questions")

                    if not args.no_purge:
                        try:
                            purged = await engram.purge(agent_id, args.user_id)
                            if purged:
                                print(f"[conv{conv_idx}] purged {purged} pre-existing memories")
                        except Exception:
                            pass

                    try:
                        ingest_summary, traces = await run_conversation(
                            engram, entry, conv_idx,
                            agent_id=agent_id,
                            user_id=args.user_id,
                            search_limit=args.search_limit,
                            graph_depth=args.graph_depth,
                            answer_max_tokens=args.answer_max_tokens,
                            max_per_session=args.max_per_session,
                            max_memory_chars=args.max_memory_chars,
                            ingest_batch_size=args.ingest_batch_size,
                            categories=categories,
                            max_questions=args.max_questions,
                            rerank=args.rerank,
                            candidate_limit=args.candidate_limit,
                            question_filter=question_filter,
                        )
                    except Exception as exc:
                        if args.fail_fast:
                            raise
                        print(f"[conv{conv_idx}] ERROR: {exc}")
                        return

                    print(f"[conv{conv_idx}] done — {len(traces)} questions answered")

                    for trace in traces:
                        all_traces.append(trace)
                        traces_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        traces_file.flush()

                    if not args.no_purge:
                        try:
                            await engram.purge(agent_id, args.user_id)
                        except Exception:
                            pass

            await asyncio.gather(*[process_one(idx) for idx in conv_indices])

    finally:
        await engram.close()

    # Judge
    scored_count = sum(1 for t in all_traces if not t.get("error"))
    errored_count = len(all_traces) - scored_count
    print(f"\n{'=' * 60}")
    print(f"judging {len(all_traces)} answers with {args.judge_model} (+{errored_count} errored)...")
    judgments = await judge_all(all_traces, args.judge_model, args.judge_concurrency)

    (args.output_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(j) for j in judgments)
    )

    total = len(judgments)
    correct = sum(j["correct"] for j in judgments)
    by_cat: dict[str, list[int]] = {}
    for j in judgments:
        cat = j.get("category_name", "unknown")
        if cat not in by_cat:
            by_cat[cat] = [0, 0]
        by_cat[cat][0] += 1
        by_cat[cat][1] += int(j["correct"])

    summary = {
        "benchmark": "LoCoMo-10 (ACL 2024)",
        "pipeline": "add_batch(turns) -> search(hybrid, include_superseded) + recall(compose=False) + get_lineage -> composer LLM",
        "judge_model": args.judge_model,
        "answer_model": settings.llm_model,
        "memory_policy": BENCHMARK_POLICY,
        "ingest_batch_size": args.ingest_batch_size,
        "search_limit": args.search_limit,
        "max_per_session": args.max_per_session,
        "rerank": args.rerank,
        "graph_depth": args.graph_depth,
        "answer_max_tokens": args.answer_max_tokens,
        "conversations": conv_indices,
        "categories": categories,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "errored": errored_count,
        "by_category": {
            cat_name: {
                "correct": c,
                "total": n,
                "accuracy": round(c / n, 4) if n else 0.0,
            }
            for cat_name, (n, c) in sorted(by_cat.items())
        },
        "failed_ids": sorted(j["question_id"] for j in judgments if not j["correct"])[:50],
    }

    _ok = [t for t in all_traces if not t.get("error")]
    _el = [t.get("elapsed_seconds", 0.0) for t in _ok]
    _ret = [t.get("retrieval", {}).get("retrieval_seconds", 0.0) for t in _ok]
    _gen = [t.get("generation_seconds", 0.0) for t in _ok]
    _sh = [t.get("retrieval", {}).get("search_hits", 0) for t in _ok]
    _gh = [t.get("retrieval", {}).get("graph_hits", 0) for t in _ok]
    summary["latency"] = {
        "avg_total_s": round(sum(_el) / len(_el), 3) if _el else 0.0,
        "p50_total_s": _pct(_el, 50),
        "p95_total_s": _pct(_el, 95),
        "avg_retrieval_s": round(sum(_ret) / len(_ret), 3) if _ret else 0.0,
        "avg_generation_s": round(sum(_gen) / len(_gen), 3) if _gen else 0.0,
    }

    # Real token accounting + end-to-end cost (tiktoken compression baseline +
    # provider-billed usage). One CostAccumulator over every answered question.
    _judge_by_id = {j["question_id"]: j for j in judgments}
    cost = CostAccumulator()
    for t in _ok:
        cu = t.get("composer_usage") or {}
        cost.add_composer(t.get("composer_model") or summary["answer_model"], cu)
        j = _judge_by_id.get(t["question_id"], {})
        cost.add_judge(j.get("judge_model") or args.judge_model, j.get("judge_usage") or {})
        cost.add_compression(
            evidence_text=t.get("evidence", ""),
            real_input_tokens=int(cu.get("input_tokens") or cu.get("prompt_tokens") or 0),
            full_context_tokens=t.get("ingest_summary", {}).get("full_context_tokens", 0),
        )
    summary["cost"] = cost.summary()
    summary["context_efficiency"] = {
        "avg_evidence_tokens": summary["cost"]["context_compression"]["avg_evidence_tokens"],
        "avg_full_context_tokens": summary["cost"]["context_compression"]["avg_full_context_tokens"],
        "compression_pct": summary["cost"]["context_compression"]["compression_pct"],
        "avg_search_hits": round(sum(_sh) / len(_sh), 1) if _sh else 0.0,
        "avg_graph_hits": round(sum(_gh) / len(_gh), 1) if _gh else 0.0,
        "token_counter": "tiktoken/o200k_base",
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    _print_scorecard(judgments)
    _c = summary["cost"]
    print(
        f"\napi calls    : {_c['api_calls']['total']} "
        f"({_c['api_calls']['composer']} composer + {_c['api_calls']['judge']} judge)"
    )
    print(
        f"real cost    : ${_c['real_cost_usd']['total']} total "
        f"(${_c['real_cost_usd']['per_question']}/question) | "
        f"composer ${_c['real_cost_usd']['composer']} + judge ${_c['real_cost_usd']['judge']}"
    )
    print(
        f"compression  : {_c['context_compression']['avg_evidence_tokens']} evidence tok "
        f"vs {_c['context_compression']['avg_full_context_tokens']} full-context tok "
        f"= {_c['context_compression']['compression_pct']}% smaller"
    )
    print(
        f"vs full-ctx  : composer ${_c['full_context_baseline']['our_composer_cost_usd']} "
        f"vs ${_c['full_context_baseline']['projected_composer_cost_usd']} naive "
        f"= {_c['full_context_baseline']['savings_pct']}% cheaper "
        f"({_c['full_context_baseline']['cost_multiplier']}x)"
    )
    print(f"\nanswer model : {summary['answer_model']}  |  judge: {args.judge_model}")
    print(f"pipeline     : {summary['pipeline']}")
    print(f"artifacts    : {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
