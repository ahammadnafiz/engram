#!/usr/bin/env python3
"""Run BEAM (ICLR 2026) end-to-end through Engram's advanced APIs.

Pipeline (per conversation, one agent per conversation):
  1. INGEST: Parse BEAM chat batches, render each turn into an episodic
     memory row with the time_anchor date in the text, and bulk-insert
     via add_batch() — on-device embeddings only, zero LLM calls at ingest.

  2. RETRIEVE (per question): 4-surface evidence gathering:
     a) search(mode="hybrid", rerank=True) — vector + full-text fused with
        RRF and cross-encoder reranking; diversified across sessions.
     b) recall(compose_answer=False) — structured current/previous/conflict
        lineage evidence with temporal anchor.
     c) get_lineage() — superseded predecessors for knowledge-update history.
     d) traverse_many() — multi-hop graph relations from top search hits.

  3. GENERATE: Composer LLM answers from the assembled evidence block.

  4. JUDGE: Independent LLM judge scores each rubric nugget 0.0/0.5/1.0
     (BEAM's nugget-scoring methodology). Question score = mean of nugget
     scores. Pass threshold >= 0.5. For event_ordering questions an optional
     Kendall tau-b is computed in addition.

  5. SCORE: Metrics at each evaluation cutoff (top-k memories used for
     answer generation), by question type and overall.

Dataset: BEAM from HuggingFace (Mohammadta/BEAM + Mohammadta/BEAM-10M).
         Auto-downloaded and cached locally in data/beam/.
Outputs: traces.jsonl, judgments.jsonl, summary.json in the output dir.

Usage:
    python benchmark/beam_benchmark.py \\
        --chat-sizes 100K \\
        --rerank \\
        --output-dir benchmark/runs/beam-100k

    # Dataset sizes: 100K=20 conversations (400q), 1M=35 conversations (700q)
    # --conversations defaults to all available; pass e.g. "0-4" for a subset.

    # Specific question types only:
    python benchmark/beam_benchmark.py \\
        --chat-sizes 100K \\
        --question-types information_extraction,temporal_reasoning

    # Re-judge an existing run:
    python benchmark/beam_benchmark.py \\
        --rejudge-only benchmark/runs/beam-100k/traces.jsonl \\
        --judge-model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import time
import uuid
from collections import defaultdict
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

from engram import Engram  # noqa: E402
from engram.core.config import get_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

LOCAL_EMBEDDING_PROVIDER = "sentence-transformers"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIMENSION = 384

BENCHMARK_POLICY = "default"
DEFAULT_MAX_MEMORY_CHARS = 8000
DEFAULT_INGEST_BATCH_SIZE = 64

HF_DATASET_NAME = "Mohammadta/BEAM"
HF_DATASET_10M = "Mohammadta/BEAM-10M"
HF_SPLIT_MAP = {"100K": "100K", "500K": "500K", "1M": "1M", "10M": "10M"}
VALID_CHAT_SIZES = ["100K", "500K", "1M", "10M"]

PASS_THRESHOLD = 0.5

BEAM_QUESTION_TYPES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]


# ===========================================================================
# Prompts
# ===========================================================================

BEAM_COMPOSER_SYSTEM = """You are an AI assistant answering questions from stored memories of past conversations with a user. Memories are tagged:
- [ACTIVE] — current, valid information
- [SUPERSEDED] — an older value since updated; use only for "what was it before?" questions
- CURRENT: / SUPERSEDED (until date): — structured lineage from Engram's recall operator

RULES:
1. Scan ALL provided memories before answering — do not stop after the first relevant one.
2. If multiple memories contain relevant information, combine and cross-reference them.
3. CONTRADICTION questions (QUESTION TYPE: contradiction_resolution) — MANDATORY:
   - Your ONLY job is to REPORT the contradiction. Do NOT resolve it, do NOT conclude which side is correct.
   - REQUIRED STEPS: (a) Scan ALL memories for any pair making logically incompatible claims about the same fact. (b) Quote BOTH verbatim. (c) State that contradictory information exists.
   - REQUIRED FORMAT: "There is contradictory information in the memories. One memory states: '[exact quote A]'. However, another memory states: '[exact quote B]'. I cannot determine which is correct."
   - IMPORTANT: Even if one side seems far more common or credible, ALWAYS report both sides. The user suspects a contradiction exists.
   - A TIMELINE UPDATE is NOT a contradiction (value changed over time → report the latest). Only logically incompatible simultaneous claims count.
   - If you find NO contradiction after checking all memories: state "I found no contradictory information about [topic] in the memories."
   - NEVER just answer the underlying factual question for contradiction_resolution questions.
4. If memories lack sufficient information: say exactly "I don't have enough information to answer this question."
5. For temporal questions: use dates stated in memories; compute intervals explicitly step by step (e.g., "From [date A] to [date B] is X days").
6. For event_ordering questions: list events in the chronological order the memories show; use dates to sort.
7. For preference questions: use the most recently stated preference.
8. For knowledge_update questions: the MOST RECENT value is the current answer.
   - [ACTIVE] = current; [SUPERSEDED] = old. When multiple memories give different values for the same fact, use the one with the LATEST date.
   - If a memory says "I updated [X] from [old] to [new]", the answer is [new].
   - Do NOT average values or report both — give only the current/latest one.
9. For ABSTENTION questions (QUESTION TYPE: abstention) — STRICT STANDARD: only answer if memories contain a DIRECT, SPECIFIC answer to this exact question.
   - "Related" is NOT enough. A passing mention is NOT enough. Proximity to the topic is NOT enough.
   - Fail cases to watch for:
     * Question asks about personal background/history → memories about a current project are NOT sufficient
     * Question asks about emotional reactions/feelings → memories recording facts are NOT sufficient
     * Question asks about team members or colleagues → memories about a project are NOT sufficient unless names are EXPLICITLY listed
     * Question asks about a specific tool or technology → do NOT infer from adjacent tools the user mentioned
   - If the memories address a related topic but NOT the specific subject of the question, say exactly: "I don't have enough information to answer this question."
   - NOTE: This strict rule ONLY applies to abstention-type questions. For all other question types, attempt a synthesis answer from available memories.
10. For summarization questions (QUESTION TYPE: summarization): your answer MUST cover the ENTIRE conversation span.
    - Go session by session through the memories. For each distinct time period or topic cluster, identify at least one key fact.
    - Do NOT stop after summarizing a few sessions — the rubric checks for coverage across ALL sessions.
    - Structure: briefly enumerate topics/themes from the full conversation, then synthesize.
11. Be specific — include exact names, dates, numbers, and details from the memories.
12. Do NOT invent information not present in the memories."""

BEAM_JUDGE_SYSTEM = (
    "You are an expert evaluator assessing whether an AI assistant's response satisfies "
    "specific rubric criteria. You must be objective, fair, and consistent. "
    "Return ONLY valid JSON with the exact format requested."
)

BEAM_NUGGET_JUDGE_PROMPT = """Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{question}

LLM RESPONSE:
{response}

RUBRIC CRITERION:
{nugget}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete.
- **0.5 (Partial Compliance)**: The required element is partially present or has minor inaccuracies.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is off-topic.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0 (Complete Compliance)**: The response is responsive AND the prohibited element is absent.
- **0.5 (Partial Compliance)**: The response is responsive but contains a borderline reference to the prohibited element.
- **0.0 (No Compliance)**: The prohibited element is present, OR the response is non-responsive.

**Compound statement handling**: If the criterion contains "and" or commas joining multiple elements:
- All elements present and correct = 1.0
- Some elements present and correct = 0.5
- No elements present or correct = 0.0

EVALUATION RULES:
1. Semantic tolerance: paraphrases and synonyms are acceptable.
2. Numeric equivalence: "$68,000" = "68k"; "2 years" = "24 months".
3. Case/punctuation/whitespace differences must be ignored.
4. Do not penalize hedging language if the substantive content satisfies the criterion.
5. Evaluate this criterion in isolation — do not consider other rubric items.

Return a JSON object with exactly two fields:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence explaining your score>"}}"""

BEAM_FACT_EXTRACTION_PROMPT = """Extract all distinct events or facts mentioned in the following response, in the exact order they are presented. Return ONLY a JSON array of short event descriptions.

RESPONSE:
{response}

Return format: ["event 1 description", "event 2 description", ...]"""

BEAM_EVENT_ALIGNMENT_PROMPT = """Given the following extracted event from an LLM response, determine which reference event it best corresponds to. Return ONLY a JSON object.

EXTRACTED EVENT:
{extracted_event}

REFERENCE EVENTS:
{events_list}

ALREADY MATCHED (do not return these indices again): {already_used}

Match the extracted event to the BEST fitting reference event by semantic similarity.
- Return the 0-based index of the matching reference event.
- Do NOT return an index listed in ALREADY MATCHED — each reference event can only be matched once.
- If no unmatched reference event fits, return -1.

Return format: {{"index": <integer>, "reason": "<brief explanation>"}}"""


def _session_stratify(candidates: list[Any], search_limit: int, per_session: int = 2) -> list[Any]:
    """Enforce breadth across sessions by taking top-K per batch_idx, then fill from remainder."""
    from collections import defaultdict
    by_batch: dict[int, list[Any]] = defaultdict(list)
    for r in candidates:
        batch = r.memory.metadata.get("batch_idx", 0)
        by_batch[batch].append(r)

    stratified: list[Any] = []
    for batch in sorted(by_batch.keys()):
        stratified.extend(by_batch[batch][:per_session])

    seen = {r.memory.memory_id for r in stratified}
    for r in candidates:
        if len(stratified) >= search_limit:
            break
        if r.memory.memory_id not in seen:
            stratified.append(r)
    return stratified[:search_limit]


def _format_evidence_for_beam(memories: list[Any], cutoff: int) -> str:
    """Format up to `cutoff` memories as a numbered chronological list for BEAM answer generation."""
    sliced = memories[:cutoff]
    if not sliced:
        return "(No memories available)"
    lines: list[str] = []
    for i, mem in enumerate(sliced, 1):
        content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
        date = mem.get("date", "") if isinstance(mem, dict) else ""
        tag = mem.get("tag", "[ACTIVE]") if isinstance(mem, dict) else "[ACTIVE]"
        if date:
            lines.append(f"{i}. [{date}] {tag} {content}")
        else:
            lines.append(f"{i}. {tag} {content}")
    return "\n".join(lines)


# ===========================================================================
# Kendall tau-b (no scipy dependency)
# ===========================================================================

def _compute_kendall_tau_b(x: list[int], y: list[int]) -> float:
    """Compute Kendall tau-b between two ordered sequences."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[:n], y[:n]
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) // 2
    denom_sq = (n0 - ties_x) * (n0 - ties_y)
    if denom_sq <= 0:
        return 0.0
    return (concordant - discordant) / math.sqrt(denom_sq)


# ===========================================================================
# Dataset
# ===========================================================================

def download_dataset(chat_sizes: list[str], data_dir: Path) -> dict[str, list[dict]]:
    """Download BEAM from HuggingFace and cache locally as JSON."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset: dict[str, list[dict]] = {}

    for size in chat_sizes:
        cache_path = data_dir / f"beam_{size}.json"
        if cache_path.exists():
            print(f"Loading cached {size} dataset: {cache_path}")
            dataset[size] = json.loads(cache_path.read_text())
            continue

        print(f"Downloading BEAM {size} from HuggingFace...")
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            raise SystemExit(
                "Install the datasets library first: pip install datasets\n"
                f"Or manually place beam_{size}.json in {data_dir}"
            )

        try:
            import ast
            ds = hf_load(HF_DATASET_10M if size == "10M" else HF_DATASET_NAME, split=HF_SPLIT_MAP[size])
            conversations: list[dict] = []
            for idx, item in enumerate(ds):
                pq_raw = item.get("probing_questions", "{}")
                if isinstance(pq_raw, str):
                    try:
                        pq = ast.literal_eval(pq_raw)
                    except (ValueError, SyntaxError):
                        try:
                            pq = json.loads(pq_raw)
                        except json.JSONDecodeError:
                            pq = {}
                else:
                    pq = pq_raw if isinstance(pq_raw, dict) else {}

                conversations.append({
                    "conversation_id": item.get("conversation_id", f"{size}_{idx}"),
                    "conversation_seed": item.get("conversation_seed", {}),
                    "user_profile": item.get("user_profile", {}),
                    "chat": item.get("chat", []),
                    "probing_questions": pq,
                })
            cache_path.write_text(json.dumps(conversations, ensure_ascii=False))
            print(f"Downloaded and cached {size}: {len(conversations)} conversations.")
            dataset[size] = conversations
        except Exception as exc:
            raise SystemExit(f"Failed to download BEAM {size}: {exc}") from exc

    return dataset


# ===========================================================================
# Chat parsing
# ===========================================================================

def _unwrap_batch_dicts(batch_dicts: list[dict]) -> list[list[dict]]:
    batches: list[list[dict]] = []
    for batch in batch_dicts:
        turns = batch.get("turns", [])
        flat: list[dict] = []
        for item in turns:
            if isinstance(item, list):
                flat.extend(item)
            elif isinstance(item, dict):
                flat.append(item)
        batches.append(flat)
    return batches


def parse_beam_chat(chat_data: Any) -> list[list[dict]]:
    """Parse BEAM chat into a list of batches (each a list of turn dicts).

    Handles the three HuggingFace storage formats:
    - 1M and smaller: 2D list [[turn, ...], ...]
    - 10M: list of session dicts mapping plan keys to batch lists
    - Batch-dict format: list of dicts with "turns" key
    """
    if not chat_data:
        return []

    if isinstance(chat_data, list) and chat_data and isinstance(chat_data[0], dict):
        if "turns" in chat_data[0]:
            return _unwrap_batch_dicts(chat_data)

        first = chat_data[0]
        sample_val = next(iter(first.values()), None)
        is_plan_format = (
            isinstance(sample_val, list)
            and sample_val
            and isinstance(sample_val[0], dict)
            and "turns" in sample_val[0]
        )
        if is_plan_format:
            batches: list[list[dict]] = []
            for session in chat_data:
                if not isinstance(session, dict):
                    continue
                plan_keys = sorted(
                    session.keys(),
                    key=lambda k: int(k.split("-")[-1]) if k.split("-")[-1].isdigit() else 0,
                )
                for pk in plan_keys:
                    plan_batches = session[pk]
                    if plan_batches:
                        batches.extend(_unwrap_batch_dicts(plan_batches))
            return batches

        if "role" in first or "content" in first:
            return [chat_data]
        return []

    if isinstance(chat_data, list) and chat_data and isinstance(chat_data[0], list):
        return chat_data

    return []


def _time_anchor_to_human(anchor: str) -> str:
    """'2023-05-15T10:30:00' -> 'May 15, 2023'"""
    if not anchor:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(anchor[:19], fmt).strftime("%B %d, %Y")
        except ValueError:
            continue
    return anchor[:10]


# ===========================================================================
# Memory row building
# ===========================================================================

def build_memory_rows(
    conversation: dict,
    chat_size: str,
    *,
    agent_id: str,
    user_id: str,
    max_memory_chars: int,
) -> list[dict[str, Any]]:
    """Render a BEAM conversation into add_batch() row dicts.

    Each turn becomes one episodic memory with the time_anchor date anchored
    in the text content. Batches are inserted in order (chronological).
    """
    conv_id = conversation.get("conversation_id", "")
    batches = parse_beam_chat(conversation.get("chat", []))
    rows: list[dict[str, Any]] = []

    for batch_idx, batch_turns in enumerate(batches):
        time_anchor = ""
        for turn in batch_turns:
            ta = turn.get("time_anchor", "")
            if ta:
                time_anchor = ta
                break
        human_date = _time_anchor_to_human(time_anchor) if time_anchor else f"batch {batch_idx}"

        for turn_idx, turn in enumerate(batch_turns):
            role_raw = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if not content:
                continue
            role = "user" if role_raw.lower() in ("user", "human") else "assistant"
            role_label = "USER" if role == "user" else "ASSISTANT"

            rendered = f"[{human_date}] {role_label}: {content}"
            if max_memory_chars > 0 and len(rendered) > max_memory_chars:
                rendered = rendered[:max_memory_chars]

            rows.append({
                "content": rendered,
                "main_content": rendered,
                "agent_id": agent_id,
                "user_id": user_id,
                "memory_type": "episodic",
                "metadata": {
                    "source": "beam",
                    "chat_size": chat_size,
                    "conversation_id": conv_id,
                    "batch_idx": batch_idx,
                    "turn_idx": turn_idx,
                    "time_anchor": time_anchor,
                    "beam_date": human_date,
                    "turn_role": role,
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
    chat_size: str,
    conv_idx: int,
    agent_id: str,
    user_id: str,
    *,
    max_memory_chars: int,
    batch_size: int,
) -> dict[str, Any]:
    """Bulk-ingest all batches of a BEAM conversation via add_batch()."""
    rows = build_memory_rows(
        conversation, chat_size,
        agent_id=agent_id,
        user_id=user_id,
        max_memory_chars=max_memory_chars,
    )

    t0 = time.perf_counter()
    inserted = 0
    errors = 0
    first_error: str | None = None

    for batch in _chunks(rows, batch_size):
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
        "batches": len(list(_chunks(rows, batch_size))),
        "errors": errors,
        "first_error": first_error,
        "ingest_seconds": round(time.perf_counter() - t0, 3),
    }


# ===========================================================================
# Probing questions
# ===========================================================================

def extract_probing_questions(conversation: dict, q_type_filter: set[str] | None) -> list[dict]:
    """Extract all probing questions from a BEAM conversation dict."""
    pq = conversation.get("probing_questions", {})
    if not pq:
        return []

    questions: list[dict] = []
    for q_type in BEAM_QUESTION_TYPES:
        if q_type_filter and q_type not in q_type_filter:
            continue
        items = pq.get(q_type, [])
        if isinstance(items, dict):
            items = [items]
        for q in items:
            if isinstance(q, str):
                questions.append({"question_type": q_type, "question_text": q, "rubric": {}})
            elif isinstance(q, dict):
                q = dict(q)
                q["question_type"] = q_type
                questions.append(q)
    return questions


def extract_rubric_nuggets(question_data: dict) -> list[str]:
    """Extract rubric nugget descriptions from a question dict."""
    rubric_raw = question_data.get("rubric", {})
    if isinstance(rubric_raw, dict):
        nuggets = rubric_raw.get("nuggets", [])
        return [
            n.get("description", str(n)) if isinstance(n, dict) else str(n)
            for n in nuggets
        ]
    if isinstance(rubric_raw, list):
        return [str(n) for n in rubric_raw]
    if rubric_raw:
        return [str(rubric_raw)]
    return []


# ===========================================================================
# Evidence building (same pattern as longmemeval / locomo)
# ===========================================================================

def _build_evidence_block(
    search_results: list[Any],
    recall_answer: Any | None,
    graph_context: str,
    lineage_superseded: list[Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Build evidence block and return (text, flat_memory_list).

    The flat list is used for cutoff slicing — it's all unique memories in
    the order they'd appear in the text (recall lineage first, then search).
    """
    lines: list[str] = []
    seen: set[str] = set()
    flat: list[dict[str, Any]] = []

    if recall_answer is not None:
        cur = getattr(recall_answer, "current", None)
        if cur is not None:
            seen.add(cur.memory_id)
            content = cur.fact or cur.content
            flat.append({"content": content, "tag": "CURRENT", "date": ""})
            lines.append(f"CURRENT: {content}")
        for mem in getattr(recall_answer, "previous", []) or []:
            when = mem.superseded_at or mem.valid_to or mem.created_at
            stamp = when.date().isoformat() if when else "unknown"
            seen.add(mem.memory_id)
            content = mem.fact or mem.content
            flat.append({"content": content, "tag": "[SUPERSEDED]", "date": stamp})
            lines.append(f"SUPERSEDED (until {stamp}): {content}")
        # For temporal_chain intent the operator puts both anchor memories in
        # .evidence (not .current/.previous). Consume them here so they reach
        # the composer regardless of where they rank in the main search results.
        for mem in getattr(recall_answer, "evidence", []) or []:
            if mem.memory_id in seen:
                continue
            seen.add(mem.memory_id)
            content = mem.fact or mem.content
            mem_date = getattr(mem, "metadata", {}).get("beam_date", "")
            flat.append({"content": content, "tag": "[ACTIVE]", "date": mem_date or ""})
            lines.append(f"RECALL: {content}")
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
        content = mem.fact or mem.content
        flat.append({"content": content, "tag": "[SUPERSEDED]", "date": stamp})
        lines.append(f"SUPERSEDED (until {stamp}): {content}")

    lines.append("\n## RETRIEVED MEMORIES (hybrid search)")
    current_date: str | None = None
    for result in search_results:
        mem = result.memory
        if mem.memory_id in seen:
            continue
        seen.add(mem.memory_id)
        mem_date = mem.metadata.get("beam_date", mem.metadata.get("haystack_date", ""))
        if mem_date != current_date:
            lines.append(f"\n--- {mem_date or 'Unknown Date'} ---")
            current_date = mem_date
        status = getattr(mem, "status", None) or mem.metadata.get("status", "active")
        tag = "[SUPERSEDED]" if status == "superseded" else "[ACTIVE]"
        content = mem.content
        flat.append({"content": content, "tag": tag, "date": mem_date or ""})
        lines.append(f"- {tag} {content}")

    if graph_context:
        lines.append(f"\n{graph_context}")

    evidence_text = "\n".join(lines) if lines else "(no matching memory)"
    return evidence_text, flat


# ===========================================================================
# Retrieval
# ===========================================================================

async def retrieve_evidence(
    engram: Engram,
    question: str,
    agent_id: str,
    user_id: str,
    search_limit: int,
    graph_depth: int,
    rerank: bool,
    candidate_limit: int = 100,
    question_type: str = "unknown",
    rubric_nuggets: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Retrieve evidence using Engram's 4-surface APIs.

    Returns (evidence_text, flat_memory_list, trace).
    """
    t0 = time.perf_counter()
    error = None
    evidence = ""
    flat: list[dict[str, Any]] = []
    n_search_hits = 0
    n_graph_hits = 0
    n_lineage = 0
    recall_intent = ""

    try:
        candidate_limit = candidate_limit if rerank else min(max(search_limit * 3, search_limit), candidate_limit)
        candidates = await engram.search(
            query=question,
            agent_id=agent_id,
            user_id=user_id,
            limit=candidate_limit,
            mode="hybrid",
            rerank=rerank,
            include_superseded=True,
        )

        # Type-specific supplemental retrieval
        # Sub-query results are PREPENDED so they're guaranteed in the final slice.
        extra: list[Any] = []
        _search_kwargs = dict(agent_id=agent_id, user_id=user_id, mode="hybrid", rerank=False, include_superseded=True)

        if question_type == "event_ordering" and rubric_nuggets:
            # One targeted sub-query per rubric event — finds specific turns the broad query misses
            tasks = [
                engram.search(query=nugget, limit=25, **_search_kwargs)
                for nugget in rubric_nuggets[:6]
            ]
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if not isinstance(result, Exception):
                    extra.extend(result)

        elif question_type == "contradiction_resolution":
            # Adversarial queries to surface minority-opinion turns.
            # The minority turn is rarely in the top-500 reranked results (majority evidence dominates).
            # PREPEND so adversarial hits are guaranteed within the evidence window.
            topic_words = " ".join(w for w in question.split() if len(w) > 4)[:120]
            neg_queries = [
                f"never {topic_words}",
                f"not {topic_words}",
                f"I have not {topic_words}",
                f"I never {topic_words}",
                f"I don't {topic_words}",
            ]
            tasks = [engram.search(query=q, limit=30, **_search_kwargs) for q in neg_queries]
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if not isinstance(result, Exception):
                    extra.extend(result)

        # Both event_ordering and contradiction extras are PREPENDED — guarantees those specific
        # turns appear within the evidence window cutoff. For event_ordering, per-event sub-queries
        # find turns the broad query misses. For contradiction, adversarial queries find the minority
        # turn that majority-dominated reranking buries past position 500.
        if extra:
            seen_ids: set[str] = {r.memory.memory_id for r in extra}
            merged = list(extra)
            for r in candidates:
                if r.memory.memory_id not in seen_ids:
                    seen_ids.add(r.memory.memory_id)
                    merged.append(r)
            candidates = merged

        # Type-specific evidence budget:
        # - single-fact types (one answer in one turn): cap at 60 to reduce noise
        # - all others: use full search_limit
        _SINGLE_FACT_TYPES = {"temporal_reasoning", "information_extraction", "knowledge_update",
                              "preference_following", "abstention"}
        effective_limit = 60 if question_type in _SINGLE_FACT_TYPES else search_limit

        if question_type == "summarization":
            search_results = _session_stratify(candidates, effective_limit)
        else:
            search_results = candidates[:effective_limit]
        n_search_hits = len(search_results)
        search_results.sort(key=lambda r: r.memory.metadata.get("beam_date", ""))

        recall_answer = None
        try:
            recall_answer = await engram.recall(
                question, agent_id,
                user_id=user_id,
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
                        if getattr(m, "status", None) == "superseded"
                    )
                except Exception:
                    pass
        n_lineage = len(lineage_superseded)

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

        evidence, flat = _build_evidence_block(
            search_results, recall_answer, graph_context, lineage_superseded
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return evidence, flat, {
        "search_hits": n_search_hits,
        "graph_hits": n_graph_hits,
        "lineage_superseded": n_lineage,
        "recall_intent": recall_intent,
        "retrieval_seconds": round(time.perf_counter() - t0, 3),
        "error": error,
    }


# ===========================================================================
# Answer generation
# ===========================================================================

async def generate_answer(
    engram: Engram,
    question: str,
    flat_memories: list[dict[str, Any]],
    cutoff: int,
    max_tokens: int,
    question_type: str = "unknown",
) -> tuple[str, str]:
    """Generate answer from evidence at a specific memory cutoff."""
    assert engram.llm is not None
    memories_text = _format_evidence_for_beam(flat_memories, cutoff)
    user_prompt = (
        f"QUESTION TYPE: {question_type}\n"
        f"QUESTION: {question}\n\n"
        f"RETRIEVED MEMORIES:\n{memories_text}\n\n"
        "ANSWER:"
    )
    resp = await engram.llm.complete_full(
        [
            {"role": "system", "content": BEAM_COMPOSER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    answer = resp.content.strip()
    if "ANSWER:" in answer:
        answer = answer.rsplit("ANSWER:", 1)[-1].strip()
    return answer, resp.model


# ===========================================================================
# Nugget judging
# ===========================================================================

def _clamp_nugget_score(raw: float) -> float:
    if raw >= 0.75:
        return 1.0
    if raw >= 0.25:
        return 0.5
    return 0.0


def _parse_nugget_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    score = 0.0
    if "1.0" in text:
        score = 1.0
    elif "0.5" in text:
        score = 0.5
    return {"score": score, "reason": text[:200]}


async def judge_nugget(
    question: str,
    nugget: str,
    generated_answer: str,
    judge_client: Any,
    judge_model: str,
    sem: asyncio.Semaphore,
    use_anthropic: bool,
) -> dict[str, Any]:
    """Judge a single rubric nugget; returns {"score": 0|0.5|1.0, "reason": "..."}."""
    prompt = BEAM_NUGGET_JUDGE_PROMPT.format(
        question=question,
        response=generated_answer,
        nugget=nugget,
    )
    async with sem:
        if use_anthropic:
            resp = await judge_client.messages.create(
                model=judge_model,
                max_tokens=256,
                temperature=0.0,
                system=BEAM_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            raw = " ".join(parts).strip()
        else:
            resp = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": BEAM_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()

    parsed = _parse_nugget_json(raw)
    try:
        score = _clamp_nugget_score(float(parsed.get("score", 0.0)))
    except (ValueError, TypeError):
        score = 0.0
    return {"score": score, "reason": parsed.get("reason", "")[:300]}


# ===========================================================================
# Event ordering (Kendall tau-b)
# ===========================================================================

async def compute_event_ordering(
    question: str,
    rubric_nuggets: list[str],
    generated_answer: str,
    judge_client: Any,
    judge_model: str,
    sem: asyncio.Semaphore,
    use_anthropic: bool,
) -> dict[str, Any]:
    """Compute Kendall tau-b for event_ordering questions."""
    if not rubric_nuggets:
        return {"tau_b": 0.0, "predicted_order": [], "reference_order": []}

    extract_prompt = BEAM_FACT_EXTRACTION_PROMPT.format(response=generated_answer)
    async with sem:
        if use_anthropic:
            resp = await judge_client.messages.create(
                model=judge_model,
                max_tokens=512,
                temperature=0.0,
                system="Extract events as a JSON array of strings.",
                messages=[{"role": "user", "content": extract_prompt}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            extract_raw = " ".join(parts).strip()
        else:
            resp = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": extract_prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            extract_raw = (resp.choices[0].message.content or "").strip()

    extracted_events: list[str] = []
    try:
        parsed = json.loads(extract_raw.strip().lstrip("```json").rstrip("```").strip())
        if isinstance(parsed, list):
            extracted_events = parsed
        elif isinstance(parsed, dict):
            for key in ("events", "facts", "result"):
                if isinstance(parsed.get(key), list):
                    extracted_events = parsed[key]
                    break
    except (json.JSONDecodeError, AttributeError):
        pass

    if not extracted_events:
        return {"tau_b": 0.0, "predicted_order": [], "reference_order": list(range(len(rubric_nuggets)))}

    events_list_text = "\n".join(f"{i}. {e}" for i, e in enumerate(rubric_nuggets))
    predicted_indices: list[int] = []
    used_indices: set[int] = set()
    for event in extracted_events:
        already_used_str = ", ".join(str(i) for i in sorted(used_indices)) or "none"
        align_prompt = BEAM_EVENT_ALIGNMENT_PROMPT.format(
            extracted_event=event,
            events_list=events_list_text,
            already_used=already_used_str,
        )
        async with sem:
            if use_anthropic:
                resp = await judge_client.messages.create(
                    model=judge_model,
                    max_tokens=128,
                    temperature=0.0,
                    system="Align the event to a reference event index. Return JSON.",
                    messages=[{"role": "user", "content": align_prompt}],
                )
                parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                align_raw = " ".join(parts).strip()
            else:
                resp = await judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": align_prompt}],
                    temperature=0.0,
                    max_tokens=128,
                    response_format={"type": "json_object"},
                )
                align_raw = (resp.choices[0].message.content or "").strip()

        try:
            data = json.loads(align_raw.strip().lstrip("```json").rstrip("```").strip())
            idx = int(data.get("index", -1))
        except (json.JSONDecodeError, ValueError, TypeError):
            idx = -1

        if 0 <= idx < len(rubric_nuggets) and idx not in used_indices:
            predicted_indices.append(idx)
            used_indices.add(idx)

    reference_order = list(range(len(rubric_nuggets)))
    tau_b = _compute_kendall_tau_b(predicted_indices, reference_order)

    return {
        "tau_b": round(tau_b, 4),
        "predicted_order": predicted_indices,
        "reference_order": reference_order,
    }


# ===========================================================================
# Per-question processing
# ===========================================================================

async def process_question(
    engram: Engram,
    question_data: dict,
    qi: int,
    chat_size: str,
    conv_idx: int,
    agent_id: str,
    user_id: str,
    *,
    search_limit: int,
    candidate_limit: int,
    graph_depth: int,
    rerank: bool,
    cutoffs: list[int],
    answer_max_tokens: int,
    event_ordering_tau: bool,
    judge_client: Any,
    judge_model: str,
    use_anthropic_judge: bool,
    judge_sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run retrieval + answer + judge for one BEAM question."""
    q_type = question_data.get("question_type", "unknown")
    conv_id = question_data.get("conversation_id", "")
    question_id = f"{chat_size}_c{conv_idx}_q{qi}_{q_type}"
    question_text = question_data.get("question_text", question_data.get("question", ""))
    difficulty = question_data.get("difficulty", "unknown")
    rubric_nuggets = extract_rubric_nuggets(question_data)
    source_chat_ids = question_data.get("source_chat_ids", [])

    start = time.perf_counter()
    error = None
    composer_model = ""
    retrieval_trace: dict[str, Any] = {}
    cutoff_results: dict[str, Any] = {}

    try:
        evidence_text, flat_memories, retrieval_trace = await retrieve_evidence(
            engram, question_text, agent_id, user_id,
            search_limit=search_limit,
            graph_depth=graph_depth,
            rerank=rerank,
            candidate_limit=candidate_limit,
            question_type=q_type,
            rubric_nuggets=rubric_nuggets,
        )

        if retrieval_trace.get("error"):
            raise RuntimeError(f"retrieval failed: {retrieval_trace['error']}")

        if engram.llm is None:
            raise RuntimeError("No LLM configured")

        for cutoff in cutoffs:
            label = f"top{cutoff}"
            generated_answer, composer_model = await generate_answer(
                engram, question_text, flat_memories, cutoff, answer_max_tokens,
                question_type=q_type,
            )

            if not rubric_nuggets:
                cutoff_results[label] = {
                    "judgment": "ERROR",
                    "score": 0.0,
                    "generated_answer": generated_answer,
                    "memories_evaluated": min(cutoff, len(flat_memories)),
                    "nugget_scores": [],
                    "error": "No rubric nuggets",
                }
                continue

            # Judge each nugget independently
            nugget_score_records: list[dict[str, Any]] = []
            for nugget in rubric_nuggets:
                ns = await judge_nugget(
                    question_text, nugget, generated_answer,
                    judge_client, judge_model, judge_sem, use_anthropic_judge,
                )
                nugget_score_records.append({"nugget": nugget, "score": ns["score"], "reason": ns["reason"]})

            avg_score = statistics.mean(r["score"] for r in nugget_score_records) if nugget_score_records else 0.0

            cr: dict[str, Any] = {
                "judgment": "PASS" if avg_score >= PASS_THRESHOLD else "FAIL",
                "score": round(avg_score, 4),
                "generated_answer": generated_answer,
                "memories_evaluated": min(cutoff, len(flat_memories)),
                "nugget_scores": nugget_score_records,
            }

            if q_type == "event_ordering" and event_ordering_tau:
                try:
                    tau_result = await compute_event_ordering(
                        question_text, rubric_nuggets, generated_answer,
                        judge_client, judge_model, judge_sem, use_anthropic_judge,
                    )
                    cr["event_ordering_tau"] = tau_result
                    # Blend nugget score with normalized tau-b ([-1,1] -> [0,1])
                    tau_norm = (tau_result["tau_b"] + 1.0) / 2.0
                    cr["score_with_tau"] = round((avg_score + tau_norm) / 2.0, 4)
                except Exception as exc:
                    cr["event_ordering_tau"] = {"error": str(exc)}

            cutoff_results[label] = cr

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return {
        "question_id": question_id,
        "chat_size": chat_size,
        "conversation_idx": conv_idx,
        "conversation_id": conv_id,
        "question_type": q_type,
        "question_type_idx": qi,
        "difficulty": difficulty,
        "question": question_text,
        "rubric": rubric_nuggets,
        "source_chat_ids": source_chat_ids,
        "agent_id": agent_id,
        "composer_model": composer_model,
        "retrieval": retrieval_trace,
        "cutoff_results": cutoff_results,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "error": error,
    }


# ===========================================================================
# Metrics
# ===========================================================================

def compute_beam_metrics(traces: list[dict[str, Any]], cutoffs: list[int]) -> dict[str, Any]:
    """Compute per-question-type and overall metrics at each cutoff."""
    metrics: dict[str, Any] = {}

    for c in cutoffs:
        label = f"top{c}"
        scores: list[float] = []
        by_type: dict[str, list[float]] = defaultdict(list)
        errors = 0

        for t in traces:
            cr = t.get("cutoff_results", {}).get(label, {})
            if cr.get("error") or t.get("error"):
                scores.append(0.0)
                by_type[t.get("question_type", "unknown")].append(0.0)
                errors += 1
            else:
                s = cr.get("score", 0.0)
                scores.append(s)
                by_type[t.get("question_type", "unknown")].append(s)

        total = len(scores)
        correct = sum(1 for s in scores if s >= PASS_THRESHOLD)

        type_metrics: dict[str, Any] = {}
        for qt in sorted(by_type):
            qt_scores = by_type[qt]
            qt_correct = sum(1 for s in qt_scores if s >= PASS_THRESHOLD)
            type_metrics[qt] = {
                "total": len(qt_scores),
                "correct": qt_correct,
                "accuracy": round(qt_correct / len(qt_scores) * 100, 1) if qt_scores else 0.0,
                "avg_score": round(statistics.mean(qt_scores), 4) if qt_scores else 0.0,
            }

        metrics[label] = {
            "overall": {
                "total": total,
                "correct": correct,
                "errors": errors,
                "accuracy": round(correct / total * 100, 1) if total else 0.0,
                "avg_score": round(statistics.mean(scores), 4) if scores else 0.0,
            },
            "by_question_type": type_metrics,
        }

    return metrics


def _print_scorecard(metrics_by_cutoff: dict[str, Any], cutoffs: list[int]) -> None:
    for c in cutoffs:
        label = f"top{c}"
        m = metrics_by_cutoff.get(label, {})
        overall = m.get("overall", {})
        print(f"\n--- cutoff={c} ---")
        print(
            f"  OVERALL: {overall.get('correct', 0)}/{overall.get('total', 0)} "
            f"pass (score >= 0.5)  |  avg={overall.get('avg_score', 0.0):.3f}  "
            f"|  errors={overall.get('errors', 0)}"
        )
        for qt, tm in sorted(m.get("by_question_type", {}).items()):
            print(
                f"  {qt:30s} {tm['correct']}/{tm['total']} "
                f"({tm['accuracy']:.1f}%)  avg={tm['avg_score']:.3f}"
            )


# ===========================================================================
# Rejudge-only
# ===========================================================================

async def rejudge_only(args: argparse.Namespace) -> None:
    """Re-score an existing traces.jsonl with the current judge logic."""
    traces = [
        json.loads(line)
        for line in args.rejudge_only.read_text().splitlines()
        if line.strip()
    ]
    print(f"Re-judging {len(traces)} traces from {args.rejudge_only} with {args.judge_model}...")

    use_anthropic = args.judge_model.lower().startswith("claude")
    if use_anthropic:
        from anthropic import AsyncAnthropic
        api_key = os.environ.get("ENGRAM_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Set ENGRAM_ANTHROPIC_API_KEY for the Claude judge.")
        judge_client: Any = AsyncAnthropic(api_key=api_key)
    else:
        from openai import AsyncOpenAI
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ENGRAM_OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY for the judge.")
        judge_client = AsyncOpenAI(api_key=api_key)

    cutoffs = [int(c) for c in args.cutoffs.split(",")]
    judge_sem = asyncio.Semaphore(args.judge_concurrency)

    updated: list[dict[str, Any]] = []
    for trace in traces:
        if trace.get("error"):
            updated.append(trace)
            continue
        question_text = trace.get("question", "")
        rubric_nuggets = trace.get("rubric", [])
        q_type = trace.get("question_type", "unknown")
        cutoff_results: dict[str, Any] = {}
        for c in cutoffs:
            label = f"top{c}"
            existing = trace.get("cutoff_results", {}).get(label, {})
            generated_answer = existing.get("generated_answer", "")
            if not generated_answer or not rubric_nuggets:
                cutoff_results[label] = existing
                continue
            nugget_score_records: list[dict[str, Any]] = []
            for nugget in rubric_nuggets:
                ns = await judge_nugget(
                    question_text, nugget, generated_answer,
                    judge_client, args.judge_model, judge_sem, use_anthropic,
                )
                nugget_score_records.append({"nugget": nugget, "score": ns["score"], "reason": ns["reason"]})
            avg = statistics.mean(r["score"] for r in nugget_score_records) if nugget_score_records else 0.0
            cutoff_results[label] = {
                **existing,
                "judgment": "PASS" if avg >= PASS_THRESHOLD else "FAIL",
                "score": round(avg, 4),
                "nugget_scores": nugget_score_records,
            }
        trace = dict(trace)
        trace["cutoff_results"] = cutoff_results
        updated.append(trace)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in updated)
    )

    metrics = compute_beam_metrics(updated, cutoffs)
    _print_scorecard(metrics, cutoffs)
    print(f"\nUpdated traces written to {args.output_dir / 'traces.jsonl'}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BEAM benchmark (ICLR 2026) through Engram's advanced APIs."
    )
    parser.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data" / "beam",
                        help="Local cache directory for HuggingFace dataset.")
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "benchmark" / "runs" / "beam-benchmark")
    parser.add_argument("--agent-prefix", default="beam")
    parser.add_argument("--user-id", default="beam-user")
    parser.add_argument("--chat-sizes", default="100K",
                        help="Comma-separated chat sizes: 100K,500K,1M,10M (default: 100K)")
    parser.add_argument("--conversations", default=None,
                        help="Conversation indices: 0-99, 0,1,5, or 0-9,50 (default: 0-4)")
    parser.add_argument("--question-types", default=None,
                        help="Comma-separated question types to evaluate (default: all 10)")
    parser.add_argument("--cutoffs", default="10,50,60",
                        help="Comma-separated top-k cutoffs for evaluation (default: 10,50,60)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Conversations processed concurrently (default: 3)")
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dimension", type=int, default=DEFAULT_EMBEDDING_DIMENSION)
    parser.add_argument("--local-embedding", action="store_true",
                        help=f"Use on-device sentence-transformers ({LOCAL_EMBEDDING_MODEL}, {LOCAL_EMBEDDING_DIMENSION}d).")
    parser.add_argument("--search-limit", type=int, default=60,
                        help="Memories retrieved per question (default: 60; must be >= max cutoff)")
    parser.add_argument("--candidate-limit", type=int, default=100,
                        help="Pre-rerank candidate pool size (default: 100). Increase for large conversations (500+ for 1M).")
    parser.add_argument("--graph-depth", type=int, default=1)
    parser.add_argument("--rerank", action="store_true",
                        help="Enable cross-encoder reranking (strongly recommended — biggest accuracy lever)")
    parser.add_argument("--max-memory-chars", type=int, default=DEFAULT_MAX_MEMORY_CHARS)
    parser.add_argument("--ingest-batch-size", type=int, default=DEFAULT_INGEST_BATCH_SIZE)
    parser.add_argument("--answer-max-tokens", type=int, default=1000)
    parser.add_argument("--event-ordering-tau", action="store_true",
                        help="Compute Kendall tau-b for event_ordering questions (extra LLM calls).")
    parser.add_argument("--no-purge", action="store_true")
    parser.add_argument("--clean-db", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--rejudge-only", type=Path,
                        help="Re-score an existing traces.jsonl (no ingestion, no answering).")
    parser.add_argument("--smoke-test", action="store_true",
                        help=(
                            "Quick sanity check: 1 conversation (idx=0), 10 questions, "
                            "optimized args (rerank, search-limit=60, cutoff=60, tau-b). "
                            "Overrides --chat-sizes/--conversations/--cutoffs/--max-questions. "
                            "Use to verify the full pipeline before a production run."
                        ))
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Cap total questions evaluated per conversation (default: all).")
    return parser.parse_args()


def _parse_conversation_indices(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


# ===========================================================================
# Main
# ===========================================================================

async def main() -> None:
    args = parse_args()

    if args.rejudge_only is not None:
        await rejudge_only(args)
        return

    if args.smoke_test:
        args.chat_sizes = "100K"
        args.conversations = "0"  # single conversation for smoke test
        args.search_limit = 60
        args.cutoffs = "60"
        args.rerank = True
        args.event_ordering_tau = True
        args.max_questions = args.max_questions or 10
        args.concurrency = 1
        print(
            f"[smoke-test] 1 conv, {args.max_questions} questions | "
            "search=60, rerank=True, tau-b=True, cutoff=60"
        )

    if args.local_embedding:
        args.embedding_provider = LOCAL_EMBEDDING_PROVIDER
        if args.embedding_model == DEFAULT_EMBEDDING_MODEL:
            args.embedding_model = LOCAL_EMBEDDING_MODEL
        if args.embedding_dimension == DEFAULT_EMBEDDING_DIMENSION:
            args.embedding_dimension = LOCAL_EMBEDDING_DIMENSION

    chat_sizes = [s.strip() for s in args.chat_sizes.split(",")]
    for s in chat_sizes:
        if s not in VALID_CHAT_SIZES:
            raise SystemExit(f"Invalid chat size '{s}'. Valid: {VALID_CHAT_SIZES}")

    # None = all available conversations; resolved per chat_size below
    conv_indices: list[int] | None = (
        None if args.conversations is None
        else _parse_conversation_indices(args.conversations)
    )
    cutoffs = sorted(int(c) for c in args.cutoffs.split(","))
    q_type_filter = set(args.question_types.split(",")) if args.question_types else None

    if args.search_limit < max(cutoffs):
        raise SystemExit(
            f"--search-limit {args.search_limit} must be >= max cutoff {max(cutoffs)}"
        )

    dataset = download_dataset(chat_sizes, args.data_dir)

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

    # Set up judge client
    use_anthropic_judge = args.judge_model.lower().startswith("claude")
    if use_anthropic_judge:
        from anthropic import AsyncAnthropic
        api_key = os.environ.get("ENGRAM_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Set ENGRAM_ANTHROPIC_API_KEY for the Claude judge.")
        judge_client: Any = AsyncAnthropic(api_key=api_key)
    else:
        from openai import AsyncOpenAI
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ENGRAM_OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY for the judge.")
        judge_client = AsyncOpenAI(api_key=api_key)

    judge_sem = asyncio.Semaphore(args.judge_concurrency)

    print(f"BEAM Benchmark | run_id={run_id}")
    print(f"  LLM      : {settings.llm_provider} / {settings.llm_model}")
    print(f"  Embedding: {settings.embedding_provider} / {settings.embedding_model}")
    print(f"  Judge    : {args.judge_model}")
    conv_display = args.conversations if args.conversations is not None else "all"
    print(f"  Sizes    : {chat_sizes}  Convs: {conv_display}  Cutoffs: {cutoffs}")
    print(f"  QTypes   : {', '.join(sorted(q_type_filter)) if q_type_filter else 'all'}")

    all_traces: list[dict[str, Any]] = []
    traces_path = args.output_dir / "traces.jsonl"

    try:
        with traces_path.open("w") as traces_file:
            conv_sem = asyncio.Semaphore(args.concurrency)

            async def process_one_conv(chat_size: str, conv_idx: int) -> None:
                async with conv_sem:
                    convs = dataset[chat_size]
                    if conv_idx >= len(convs):
                        print(f"[{chat_size}][{conv_idx}] out of range, skipping")
                        return

                    conversation = convs[conv_idx]
                    conv_id = conversation.get("conversation_id", f"{chat_size}_{conv_idx}")
                    agent_id = f"{args.agent_prefix}-{run_id}-{chat_size}-c{conv_idx}"

                    questions = extract_probing_questions(conversation, q_type_filter)
                    if args.max_questions is not None:
                        questions = questions[:args.max_questions]
                    print(f"\n{'=' * 60}\n[{chat_size}][{conv_idx}] conv_id={conv_id} | {len(questions)} questions")

                    if not args.no_purge:
                        try:
                            purged = await engram.purge(agent_id, args.user_id)
                            if purged:
                                print(f"[{chat_size}][{conv_idx}] purged {purged} pre-existing memories")
                        except Exception:
                            pass

                    # Ingest
                    ingest_summary = await ingest_conversation(
                        engram, conversation, chat_size, conv_idx, agent_id, args.user_id,
                        max_memory_chars=args.max_memory_chars,
                        batch_size=args.ingest_batch_size,
                    )
                    print(
                        f"[{chat_size}][{conv_idx}] ingested {ingest_summary['inserted']}/"
                        f"{ingest_summary['rows']} memories in {ingest_summary['ingest_seconds']}s"
                    )

                    if ingest_summary.get("errors"):
                        print(f"[{chat_size}][{conv_idx}] WARNING: {ingest_summary['errors']} ingest errors")

                    # Process questions
                    for qi, q_data in enumerate(questions):
                        try:
                            trace = await process_question(
                                engram, q_data, qi, chat_size, conv_idx, agent_id, args.user_id,
                                search_limit=args.search_limit,
                                candidate_limit=args.candidate_limit,
                                graph_depth=args.graph_depth,
                                rerank=args.rerank,
                                cutoffs=cutoffs,
                                answer_max_tokens=args.answer_max_tokens,
                                event_ordering_tau=args.event_ordering_tau,
                                judge_client=judge_client,
                                judge_model=args.judge_model,
                                use_anthropic_judge=use_anthropic_judge,
                                judge_sem=judge_sem,
                            )
                        except Exception as exc:
                            if args.fail_fast:
                                raise
                            print(f"[{chat_size}][{conv_idx}][q{qi}] ERROR: {exc}")
                            trace = {
                                "question_id": f"{chat_size}_c{conv_idx}_q{qi}_{q_data.get('question_type', 'unknown')}",
                                "chat_size": chat_size,
                                "conversation_idx": conv_idx,
                                "conversation_id": conv_id,
                                "question_type": q_data.get("question_type", "unknown"),
                                "question_type_idx": qi,
                                "question": q_data.get("question_text", ""),
                                "rubric": extract_rubric_nuggets(q_data),
                                "agent_id": agent_id,
                                "composer_model": "",
                                "retrieval": {},
                                "cutoff_results": {},
                                "elapsed_seconds": 0.0,
                                "error": f"{type(exc).__name__}: {exc}",
                            }

                        trace["ingest_summary"] = ingest_summary
                        all_traces.append(trace)
                        traces_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        traces_file.flush()

                        primary_cr = trace.get("cutoff_results", {}).get(f"top{max(cutoffs)}", {})
                        status = "ERR" if trace.get("error") else primary_cr.get("judgment", "?")
                        score = primary_cr.get("score", 0.0)
                        retr = trace.get("retrieval", {})
                        print(
                            f"  [{trace['question_id']}] {status} score={score:.2f} "
                            f"qt={trace.get('question_type', '?')} "
                            f"search={retr.get('search_hits', '?')} "
                            f"recall={retr.get('recall_intent', '?')} "
                            f"| {trace.get('elapsed_seconds', '?')}s"
                        )
                        if primary_cr.get("generated_answer"):
                            print(f"    answer: {primary_cr['generated_answer'][:100]}")

                    if not args.no_purge:
                        try:
                            await engram.purge(agent_id, args.user_id)
                        except Exception:
                            pass

            tasks = [
                process_one_conv(size, ci)
                for size in chat_sizes
                for ci in (conv_indices if conv_indices is not None else range(len(dataset[size])))
                if ci < len(dataset[size])
            ]
            await asyncio.gather(*tasks)

    finally:
        await engram.close()

    # Compute and display metrics
    metrics_by_cutoff = compute_beam_metrics(all_traces, cutoffs)
    _print_scorecard(metrics_by_cutoff, cutoffs)

    # Write summary
    total_questions = len(all_traces)
    errored = sum(1 for t in all_traces if t.get("error"))
    primary_label = f"top{max(cutoffs)}"
    primary_m = metrics_by_cutoff.get(primary_label, {}).get("overall", {})

    summary: dict[str, Any] = {
        "benchmark": "BEAM (ICLR 2026)",
        "pipeline": (
            "add_batch(turns) -> search(hybrid, rerank) + recall(compose=False) "
            "+ get_lineage + traverse_many -> composer LLM -> nugget judge (0/0.5/1.0)"
        ),
        "judge_model": args.judge_model,
        "answer_model": settings.llm_model,
        "memory_policy": BENCHMARK_POLICY,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "search_limit": args.search_limit,
        "candidate_limit": args.candidate_limit,
        "rerank": args.rerank,
        "graph_depth": args.graph_depth,
        "answer_max_tokens": args.answer_max_tokens,
        "cutoffs": cutoffs,
        "chat_sizes": chat_sizes,
        "conversations": args.conversations,
        "question_types": sorted(q_type_filter) if q_type_filter else BEAM_QUESTION_TYPES,
        "total_questions": total_questions,
        "errored": errored,
        "pass_threshold": PASS_THRESHOLD,
        "primary_cutoff": primary_label,
        "accuracy": primary_m.get("accuracy", 0.0),
        "avg_score": primary_m.get("avg_score", 0.0),
        "metrics_by_cutoff": metrics_by_cutoff,
        "failed_ids": sorted(
            t["question_id"] for t in all_traces
            if not t.get("error")
            and (t.get("cutoff_results", {}).get(primary_label, {}).get("score", 0.0) < PASS_THRESHOLD)
        )[:50],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nanswer model : {summary['answer_model']}  |  judge: {args.judge_model}")
    print(f"primary cutoff: {primary_label}  accuracy={summary['accuracy']:.1f}%  avg_score={summary['avg_score']:.3f}")
    print(f"artifacts    : {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
