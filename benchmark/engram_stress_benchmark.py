#!/usr/bin/env python3
"""Run the custom Engram Memory Stress Test against the real Engram core.

Pipeline (one self-contained run):
  1. INGEST: Render each context session as a pre-formed memory row and
     bulk-insert with add_batch() -- embeddings only, no per-session LLM
     fact-extraction. The date_marker is anchored in each row's text so
     temporal questions resolve correctly. This mirrors longmemeval_benchmark.py:
     ingestion is cheap and parallel, all reasoning over conflicting/updated
     facts moves to the retrieval + composer layer.

  2. RETRIEVE: Multi-surface evidence gathering (same stack as longmemeval):
     a) search(mode="hybrid", include_superseded=True) -- wide candidate pool,
        diversified across sessions down to the evidence budget.
     b) recall(compose_answer=False) -- structured current/previous/conflict
        lineage evidence.
     c) get_lineage() -- superseded predecessors of retrieved active facts.
     (traverse_many() graph traversal is off by default -- ingest creates no
      edges, so it is a no-op unless edges are added via add_relation().)

  3. GENERATE: A separate composer LLM call writes the final answer from
     the assembled evidence block, using the same COMPOSER_SYSTEM as
     longmemeval_benchmark.py.

  4. JUDGE: An independent LLM judge scores each answer against the
     structured expected_output rubric, honoring evaluation_rule and
     distractors_to_ignore. Abstention questions use the dedicated
     ABSTENTION_JUDGE_PROMPT; normal questions use the unified JUDGE_PROMPT
     with the rubric injected as the correct answer.

Outputs JSONL traces, per-question judgments, and a summary.json into the
run directory, plus an overall + per-category accuracy table on stdout.

Config comes from .env (same ENGRAM_* vars as longmemeval_benchmark.py).

Usage:
    poetry run python benchmark/engram_stress_benchmark.py \\
        --output-dir benchmark/runs/engram-stress

    # Different judge model:
    poetry run python benchmark/engram_stress_benchmark.py \\
        --judge-model claude-haiku-4-5-20251001 \\
        --output-dir benchmark/runs/engram-stress

    # Local embeddings (no API cost):
    poetry run python benchmark/engram_stress_benchmark.py \\
        --local-embedding \\
        --output-dir benchmark/runs/engram-stress-local

    # Re-judge an existing run:
    poetry run python benchmark/engram_stress_benchmark.py \\
        --rejudge-only benchmark/runs/engram-stress/traces.jsonl \\
        --output-dir benchmark/runs/engram-stress
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment bootstrap (mirrors longmemeval_benchmark.py)
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
from engram.policy import MemoryPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"

DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

LOCAL_EMBEDDING_PROVIDER = "sentence-transformers"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIMENSION = 384

DEFAULT_INGEST_BATCH_SIZE = 64

# The stress suite mixes durable profile facts (allergy, preferences) with
# project facts that genuinely supersede each other (p95 thresholds, demo
# city, launch date). The default policy's slot/critical rules drive supersede
# + conflict resolution -- which is what the suite actually tests.
STRESS_POLICY = "default"


# ===========================================================================
# Prompts
# ===========================================================================

COMPOSER_SYSTEM = """You are a personal assistant with access to memories from past conversations with a user. Answer the question using information from the memories below. Be direct and concise.

IMPORTANT: Today's date is {question_date}. All relative time expressions MUST be computed relative to this date.

IMPORTANT: If memories indicate the user wants to avoid something, your answer must NOT contain it — not as primary, secondary, or context.

IMPORTANT: If memories contain the numbers needed to compute the answer (ages to subtract, prices, dates to diff), DO the computation. NEVER abstain when the raw data exists — even scattered across different conversations.

IMPORTANT: Keep your responses short. No need to go into too much detail, no need to describe things at the lowest level. You can generally describe events and ideas abstractly.

IMPORTANT: Pay close attention to the EXACT entity in the question. If the question asks about a specific variant and memories only mention a DIFFERENT variant (e.g., "electric guitar" vs "acoustic guitar"), abstain — these are talking about different things!

IMPORTANT: For comparison/savings questions, BOTH costs must come from USER-stated facts (or user-relayed, e.g., "my friend said"). Do NOT use assistant-provided general info. If only one side has a user-stated cost, abstain.

IMPORTANT: If the query uses a specific but WRONG role/title/entity (e.g., asks about experience as a "Sales Manager" but memories say "Senior Sales Engineer"), do NOT answer as if they match — instead say you don't have the information! Always lean towards abstention in these cases! Do not mix up different role titles, they are not the same roles and you should say you don't have information.

Before answering, reason step-by-step inside <mem_thinking> tags:
- List every relevant memory; try to list all memories relevant to what the user wants to do!
- For counting: enumerate each item with date. Apply the question's EXACT verb/qualifier strictly. Count multiple items in a single memory separately. Do a SECOND full scan of all memories after initial count. Verify each item is a completed action (past tense), not a plan.
- For cross-topic computation: scan ALL memories for each needed fact independently. List: (a) what you need, (b) where each appears, (c) the computation.
- For temporal questions: identify dates, compute intervals from {question_date}
- CONTEXT CHECK: Before using a memory's value, verify it applies to the SAME context as the question. List the context of each memory and only use values from the matching context.
- State your conclusion

The user will only see text outside the <mem_thinking> tags.

Rules:

1. **Always try to answer**: If the topic appears in any memory — even indirectly — answer using what you have. Don't refuse for one missing detail.

2. **Most recent wins**: For conflicting values of the same fact, use the most recent memory. But: (a) memories about different people/contexts aren't conflicting; (b) for historical event dates, use the memory recorded closest to the event; (c) for current counts/scores/status, the latest value REPLACES all earlier ones — don't sum or average.

3. **ACTIVE vs SUPERSEDED memories**: Memories tagged [ACTIVE] are the truth now. Memories tagged [SUPERSEDED] are old values that have been overwritten. For current-state questions, use [ACTIVE] memories. For "what was it before?" or "originally" questions, use [SUPERSEDED] memories. Never present a superseded value as the current answer.

4. **Time-bounded questions**: Compute the date window from {question_date}. Show date arithmetic in <mem_thinking>. Scan EVERY memory for events in range.

5. **Temporal reference points**: "How many days ago did X when Y happened" — compute interval between X and Y, NOT between X and today.

6. **Counting and ordering**: Scan ALL memories first to last. Build a numbered list in <mem_thinking> with date and position. Deduplicate by matching dates/descriptions.

7. **Use only the memories**: Don't invent numbers, prices, addresses, or details not present in the evidence.

8. **When to abstain**: Say "The information provided is not enough" when the topic is genuinely unmentioned or the question asks about a specific event that doesn't exist.

9. **Yes/no and comparison**: "Did I ever do X?" with no matching memory = "No." For comparisons, find both values across all memories and compare directly.

10. **Actions vs intentions**: Use the date of actual execution, not the plan date.

11. **User facts vs assistant advice**: "User..." = actual experience. "Assistant..." = advice. Prefer user-stated facts for personal questions.

12. **Connect memories across topics**: Facts needed for computation are often in unrelated conversations. Search ALL memories for each fact independently.

13. **Keep entities distinct**: Do not merge two people who share a name. Attribute work only when the evidence explicitly links that person to it.

14. **ACTIVE / CURRENT memories are the truth now. SUPERSEDED / PREVIOUS memories are old values**: report the current value, and mention an old value only to show what changed. Never present a superseded value as the current answer.
"""


# ===========================================================================
# Judge Prompts (ported from longmemeval_benchmark.py)
# ===========================================================================

JUDGE_PROMPT = """I will give you a question, a correct answer (or rubric), and a model response. Decide whether the model response is correct.

CORE PRINCIPLE — Semantic equivalence: Judge by MEANING, not exact words. Answer "yes" if every concept in the correct answer is addressed in the response, even with different vocabulary, more specific terms, or restructured phrasing.

IMPORTANT BIAS CHECK: You have a tendency to say "no" too quickly. Before concluding "no", you MUST verify the answer is truly wrong, not just differently worded. When in doubt, lean toward "yes".

Rules:

**Equivalence & Supersets**
- Equivalent or superset responses are correct. Extra details are fine unless proven to be factually wrong.
- Same factual meaning with different phrasing = correct.

**Lists & Compound Terms**
- For list answers, match each item by semantic meaning. A concept is covered if restated via synonyms, sub-concepts, or related terms.
- If some items are listed as "or"s, "maybe"s and potential answers, it's okay if the answer does not include those.

**Numbers & Precision**
- Hedging ("at least 3", "approximately") is fine if the core number matches. A range that includes the correct answer is correct.
- More precise answers are correct: "22 days" matches "3 weeks"; "over $270" matches "$270."
- Rough answers are correct: "about nine months" ≈ "9 months"; off-by-one errors on days/weeks/months are acceptable.

**Dates & Temporal**
- Date format variations are equivalent.
- Outdated info alongside the correct updated answer is acceptable if the current value is identified.

**Counting Edge Cases**
- If correct answer is "0" or "nothing found," model saying "not enough information" is also correct.
- Similarly, if correct answer is "not enough information", model saying "0" or "nothing found" is also correct.

**Abstention Matching**
- If correct answer = unanswerable/abstention, ANY phrasing that conveys "I don't have this information" is correct.
- The key test: does the response REFUSE to answer the question? If yes, it matches an abstention ground truth, period.

FINAL CHECK: Before answering "no," reason through:
1. What is the core factual claim or intent of the correct answer?
2. Does the model response address that same claim, even in different words?
3. Is the response a superset (correct answer + extra details)?
4. For numbers: does the core number match, ignoring hedging/qualifiers?
5. For abstentions: does the response effectively decline to answer?
Only answer "no" if, after this analysis, a core concept is entirely unaddressed or contradicted.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Think step-by-step in <judge_thinking> tags, then give your final verdict as exactly "yes" or "no" on a new line after the closing tag."""


ABSTENTION_JUDGE_PROMPT = """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}
Explanation: {answer}
Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only."""


# Stress-test rubric keys that describe HOW to grade, not what to answer.
# Folding these into the required rubric causes false negatives.
_JUDGE_META_KEYS = {
    "supporting_context",
    "evaluation_rule",
    "distractors_to_ignore",
    "lineage",
}


# ===========================================================================
# Stress-test judge helpers
# ===========================================================================

def flatten_expected(expected: dict[str, Any], indent: int = 0) -> str:
    """Render structured expected_output as readable key: value lines."""
    lines: list[str] = []
    pad = "  " * indent
    for key, value in expected.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(flatten_expected(value, indent + 1))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                lines.append(f"{pad}{key}:")
                for item in value:
                    lines.append(flatten_expected(item, indent + 1))
            else:
                joined = ", ".join(str(v) for v in value)
                lines.append(f"{pad}{key}: {joined}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(line for line in lines if line.strip())


def is_abstention(expected: dict[str, Any], category: str) -> bool:
    """Detect fabrication/abstention prompts (answer must be 'not in memory')."""
    if "hallucination" in category.lower() or "fabrication" in category.lower():
        return True
    blob = json.dumps(expected).lower()
    return any(
        marker in blob
        for marker in ("unknown / unrecorded", "unrecorded", "not available", "not recorded")
    )


def build_judge_prompt(prompt_item: dict[str, Any], response: str) -> str:
    """Route abstention questions to ABSTENTION_JUDGE_PROMPT and normal
    questions to JUDGE_PROMPT with the rubric injected as the correct answer."""
    question = prompt_item["prompt"]
    category = prompt_item.get("category", "")
    expected = prompt_item["expected_output"]

    if is_abstention(expected, category):
        return ABSTENTION_JUDGE_PROMPT.format(
            question=question,
            answer=flatten_expected(expected),
            response=response,
        )

    required = {k: v for k, v in expected.items() if k not in _JUDGE_META_KEYS}
    gold = flatten_expected(required) or flatten_expected(expected)

    parts = [gold]
    distractors = expected.get("distractors_to_ignore")
    if distractors:
        parts.append(
            f"\nDistractors (superseded values — mentioning as past context is OK, "
            f"but NOT as the current answer): {', '.join(map(str, distractors))}"
        )
    rule = expected.get("evaluation_rule")
    if rule:
        parts.append(f"\nAdditional grading rule: {rule}")
    context = expected.get("supporting_context")
    if context:
        parts.append(
            f"\nBackground context (do NOT require the response to restate this): {context}"
        )

    return JUDGE_PROMPT.format(
        question=question,
        answer="\n".join(parts),
        response=response,
    )


# ===========================================================================
# Helpers
# ===========================================================================

def _to_human_date(date_str: str) -> str:
    """Convert date strings (yyyy/mm/dd or yyyy-mm-dd) to human-readable form."""
    from datetime import datetime
    m = re.match(r"(\d{4}/\d{2}/\d{2})", date_str)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y/%m/%d").strftime("%A, %B %d, %Y")
        except ValueError:
            pass
    m = re.match(r"(\d{4}-\d{2}-\d{2})", date_str)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").strftime("%A, %B %d, %Y")
        except ValueError:
            pass
    return date_str


def _parse_question_date(date_str: str) -> "datetime | None":
    """Parse date strings into a datetime for recall's temporal reasoning."""
    from datetime import datetime
    if not date_str:
        return None
    # LME format: "2023/05/20 (Sat) 02:21"
    m = re.match(r"(\d{4}/\d{2}/\d{2})(?:\s+\(\w+\))?(?:\s+(\d{2}:\d{2}))?", date_str)
    if m:
        fmt = "%Y/%m/%d %H:%M" if m.group(2) else "%Y/%m/%d"
        raw = f"{m.group(1)} {m.group(2)}" if m.group(2) else m.group(1)
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    # ISO format: "2024-01-15"
    m = re.match(r"(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2}))?", date_str)
    if m:
        fmt = "%Y-%m-%dT%H:%M" if m.group(2) else "%Y-%m-%d"
        raw = f"{m.group(1)}T{m.group(2)}" if m.group(2) else m.group(1)
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ===========================================================================
# Ingestion
# ===========================================================================

def build_memory_rows(
    sessions: list[dict[str, Any]],
    *,
    agent_id: str,
    user_id: str,
) -> list[dict[str, Any]]:
    """Render context_sessions into add_batch() row dicts.

    Each session statement becomes one episodic memory with the date_marker
    anchored in the text. Uses the same metadata keys as longmemeval_benchmark
    (haystack_date, original_session_id) so the evidence-building helpers work
    unchanged.
    """
    rows: list[dict[str, Any]] = []
    for sess in sessions:
        content = f"[{sess['date_marker']}] {sess['content']}"
        rows.append(
            {
                "content": content,
                "main_content": content,
                "agent_id": agent_id,
                "user_id": user_id,
                "memory_type": "episodic",
                "metadata": {
                    "source": "engram-stress",
                    "original_session_id": str(sess["session_id"]),
                    "session_id": sess["session_id"],
                    "haystack_date": sess["date_marker"],
                },
            }
        )
    return rows


async def ingest_sessions(
    engram: Engram,
    sessions: list[dict[str, Any]],
    agent_id: str,
    user_id: str,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Bulk-ingest all context sessions via add_batch() (no per-session LLM)."""
    rows = build_memory_rows(sessions, agent_id=agent_id, user_id=user_id)

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
        "ingest_seconds": round(time.perf_counter() - t0, 3),
    }


# ===========================================================================
# Evidence Building (mirrors longmemeval_benchmark.py)
# ===========================================================================

def _build_evidence_block(
    search_results: list[Any],
    recall_answer: Any | None,
    graph_context: str,
    lineage_superseded: list[Any] | None = None,
) -> str:
    """Assemble the evidence block from Engram's retrieval surfaces."""
    lines: list[str] = []
    seen: set[str] = set()

    # 1) Recall's structured lineage signal first (current vs superseded).
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

    # 2) Lineage history for retrieved active facts.
    for mem in lineage_superseded or []:
        if mem.memory_id in seen:
            continue
        seen.add(mem.memory_id)
        when = mem.superseded_at or mem.valid_to or mem.created_at
        stamp = when.date().isoformat() if when else "unknown"
        lines.append(f"SUPERSEDED (until {stamp}): {mem.fact or mem.content}")

    # 3) Hybrid search results: active + superseded, grouped by date, tagged.
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

    # 4) Graph relations for multi-hop context.
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
    """Round-robin a candidate pool across sessions, user turns first.

    Ported from longmemeval_benchmark.py. Prevents the evidence budget from
    collapsing onto a single session's near-duplicate turns.
    """
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


async def retrieve_evidence(
    engram: Engram,
    question: str,
    agent_id: str,
    user_id: str,
    search_limit: int,
    graph_depth: int,
    max_per_session: int,
    question_date: str = "",
    rerank: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Retrieve evidence using the full longmemeval retrieval stack.

    Returns:
        (evidence_text, retrieval_trace)
    """
    t0 = time.perf_counter()
    error = None
    evidence = ""
    n_search_hits = 0
    n_graph_hits = 0
    n_lineage_superseded = 0
    recall_intent = ""

    try:
        candidate_limit = 100 if rerank else min(max(search_limit * 3, search_limit), 100)
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

        search_results.sort(
            key=lambda r: r.memory.metadata.get("haystack_date", "")
        )

        recall_answer = None
        try:
            recall_answer = await engram.recall(
                question,
                agent_id,
                user_id=user_id,
                question_date=_parse_question_date(question_date),
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

    trace = {
        "search_hits": n_search_hits,
        "graph_hits": n_graph_hits,
        "lineage_superseded": n_lineage_superseded,
        "recall_intent": recall_intent,
        "retrieval_seconds": round(time.perf_counter() - t0, 3),
        "error": error,
    }
    return evidence, trace


# ===========================================================================
# Answer Generation (mirrors longmemeval_benchmark.py)
# ===========================================================================

async def generate_answer(
    engram: Engram,
    question: str,
    question_date: str,
    evidence: str,
    max_tokens: int,
    user_name: str = "",
) -> tuple[str, str]:
    """Generate answer from evidence using Engram's LLM.

    Returns:
        (answer_text, model_name)
    """
    assert engram.llm is not None

    system = COMPOSER_SYSTEM.format(question_date=question_date or "the present day")
    user_name_note = (
        f"The user's name is {user_name}. When asked who they are or when listing "
        f"them as a team member, use their name ({user_name}), not 'the user' or 'you'.\n\n"
        if user_name else ""
    )

    user_prompt = (
        f"{user_name_note}"
        f"<engram_memory_evidence>\n{evidence}\n</engram_memory_evidence>\n\n"
        f"Today's Date: {question_date or 'the present day'}\n"
        f"Question: {question}\n\n"
        "IMPORTANT: You MUST provide your full thinking in <mem_thinking> tags "
        "BEFORE giving your answer.\nReasoning and answer:"
    )
    resp = await engram.llm.complete_full(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.content.strip(), resp.model


# ===========================================================================
# Judging
# ===========================================================================

def _parse_yes_no_judgment(raw: str) -> bool:
    """Extract the final yes/no verdict from judge output.

    Handles <judge_thinking> blocks from the unified JUDGE_PROMPT and plain
    "yes"/"no" replies from ABSTENTION_JUDGE_PROMPT.
    """
    text = (raw or "").strip()
    if not text:
        return False
    after_cot = re.split(r"</judge_thinking>|</thinking>", text, flags=re.IGNORECASE)
    verdict_region = after_cot[-1].strip() if after_cot else text
    verdict_lines = [
        line.strip().lower() for line in verdict_region.splitlines() if line.strip()
    ]
    for line in reversed(verdict_lines):
        if line == "yes":
            return True
        if line == "no":
            return False
    token_matches = re.findall(r"\b(yes|no)\b", verdict_region.lower())
    if token_matches:
        return token_matches[-1] == "yes"
    return text.lower().startswith("yes")


async def judge_all(
    answer_traces: list[dict[str, Any]],
    prompts_by_id: dict[int, dict[str, Any]],
    judge_model: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Independently judge each answer. Routes to Anthropic for claude-* models,
    otherwise OpenAI."""
    use_anthropic = judge_model.lower().startswith("claude")

    if use_anthropic:
        from anthropic import AsyncAnthropic

        api_key = os.environ.get("ENGRAM_ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise SystemExit("Set ENGRAM_ANTHROPIC_API_KEY for the Claude judge.")
        anthropic_client = AsyncAnthropic(api_key=api_key)
    else:
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "ENGRAM_OPENAI_API_KEY"
        )
        if not api_key:
            raise SystemExit(
                "Set OPENAI_API_KEY or ENGRAM_OPENAI_API_KEY for the judge."
            )
        openai_client = AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(concurrency)

    async def verdict_for(prompt: str) -> str:
        async with sem:
            if use_anthropic:
                resp = await anthropic_client.messages.create(
                    model=judge_model,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                return " ".join(parts).strip()
            resp = await openai_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()

    async def judge_one(trace: dict[str, Any]) -> dict[str, Any]:
        item = prompts_by_id[trace["id"]]
        category = trace.get("category", "")
        if trace.get("error"):
            return {
                "id": trace["id"],
                "category": category,
                "correct": False,
                "verdict": "error",
                "abstention": is_abstention(item["expected_output"], category),
            }
        prompt = build_judge_prompt(item, trace["hypothesis"])
        raw = await verdict_for(prompt)
        correct = _parse_yes_no_judgment(raw)
        return {
            "id": trace["id"],
            "category": category,
            "correct": correct,
            "verdict": "yes" if correct else "no",
            "judge_raw": raw[:300],
            "abstention": is_abstention(item["expected_output"], category),
        }

    return await asyncio.gather(*(judge_one(t) for t in answer_traces))


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Engram Memory Stress Test through Engram's advanced APIs."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=_REPO_ROOT / "data" / "longmemeval" / "test.json",
        help="Path to the stress-test JSON (default: data/longmemeval/test.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "benchmark" / "runs" / "engram-stress",
        help="Directory for traces / judgments / summary.",
    )
    parser.add_argument("--agent-id", default="engram-stress")
    parser.add_argument("--user-id", default="stress-user")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Independent LLM judge model (default {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of prompts to answer concurrently (default 5).",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model for answer composition (default {DEFAULT_LLM_MODEL}).",
    )
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding provider (default {DEFAULT_EMBEDDING_PROVIDER}).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model (default {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSION,
        help=f"Embedding dimension (default {DEFAULT_EMBEDDING_DIMENSION}).",
    )
    parser.add_argument(
        "--local-embedding",
        action="store_true",
        help=(
            "Use on-device sentence-transformers embeddings "
            f"({LOCAL_EMBEDDING_MODEL}, {LOCAL_EMBEDDING_DIMENSION}d) instead of "
            "an API provider: no cost, no rate limits."
        ),
    )
    parser.add_argument(
        "--ingest-batch-size",
        type=int,
        default=DEFAULT_INGEST_BATCH_SIZE,
        help=f"Sessions per add_batch() call (default {DEFAULT_INGEST_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=40,
        help=(
            "Final memories per question after session-diversified selection "
            "(active + superseded). Default 40."
        ),
    )
    parser.add_argument(
        "--max-per-session",
        type=int,
        default=4,
        help="Max memories kept per session in the diversified evidence set (default 4).",
    )
    parser.add_argument(
        "--graph-depth",
        type=int,
        default=0,
        help="Max graph traversal depth from search seeds. 0 to disable. "
        "Off by default: ingest does not create edges, so traversal is a "
        "no-op unless edges are populated via add_relation().",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=4000,
        help=(
            "Max tokens for the composer LLM answer. Must be generous: the "
            "composer does step-by-step work inside <mem_thinking> before "
            "emitting the user-facing answer (default 4000)."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Cross-encoder rerank candidates before selection (needs sentence-transformers).",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Do NOT purge the agent's memories before ingesting.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion (reuse memories already stored for this agent/user).",
    )
    parser.add_argument(
        "--clean-db",
        action="store_true",
        help="Drop and recreate the database schema before running.",
    )
    parser.add_argument(
        "--rejudge-only",
        type=Path,
        help="Re-score an existing traces.jsonl (no ingestion, no answering).",
    )
    args = parser.parse_args()

    if args.local_embedding:
        args.embedding_provider = LOCAL_EMBEDDING_PROVIDER
        if args.embedding_model == DEFAULT_EMBEDDING_MODEL:
            args.embedding_model = LOCAL_EMBEDDING_MODEL
        if args.embedding_dimension == DEFAULT_EMBEDDING_DIMENSION:
            args.embedding_dimension = LOCAL_EMBEDDING_DIMENSION

    return args


# ===========================================================================
# Rejudge helper
# ===========================================================================

async def rejudge_only(
    args: argparse.Namespace,
    data: dict[str, Any],
) -> None:
    """Re-score an existing traces.jsonl with the current judge logic."""
    traces = [
        json.loads(line)
        for line in args.rejudge_only.read_text().splitlines()
        if line.strip()
    ]
    dataset_by_id = {p["id"]: p for p in data["evaluation_prompts"]}
    prompts_by_id = {
        t["id"]: dataset_by_id.get(
            t["id"],
            {
                "prompt": t.get("question", ""),
                "category": t.get("category", ""),
                "expected_output": t["expected_output"],
            },
        )
        for t in traces
    }
    print(
        f"re-judging {len(traces)} answers from {args.rejudge_only} "
        f"with {args.judge_model}..."
    )
    judgments = await judge_all(traces, prompts_by_id, args.judge_model, args.judge_concurrency)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(j) for j in judgments)
    )

    total = len(judgments)
    correct = sum(j["correct"] for j in judgments)
    by_cat: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for j in judgments:
        by_cat[j["category"]][0] += 1
        by_cat[j["category"]][1] += int(j["correct"])

    print("\n" + "=" * 60)
    print(f"OVERALL : {correct}/{total} = {correct / total * 100:.1f}%" if total else "No samples")
    print("\nby category:")
    for cat, (n, c) in sorted(by_cat.items()):
        print(f"  {cat[:40]:40s} {c}/{n} = {c / n * 100:.1f}%")
    print(f"\njudgments written to {out_dir / 'judgments.jsonl'}")


# ===========================================================================
# Main
# ===========================================================================

async def main() -> None:
    args = parse_args()
    data = json.loads(args.data_path.read_text())

    if args.rejudge_only is not None:
        await rejudge_only(args, data)
        return

    sessions = data["context_sessions"]
    prompts = data["evaluation_prompts"]
    prompts_by_id = {p["id"]: p for p in prompts}

    # Use the explicit question_date from the dataset (the last session date).
    # Do NOT derive via max() — month names don't sort alphabetically by calendar
    # order ("May" > "Jun" as strings), so max() returns the wrong date.
    question_date = data.get("question_date", sessions[-1]["date_marker"])
    user_name = data.get("user_name", "")

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    engram = Engram(settings=settings, memory_policy=STRESS_POLICY)
    await engram.connect()

    if args.clean_db:
        print("Cleaning database (dropping public schema)...")
        storage = getattr(engram, "_storage", None)
        if storage is not None:
            await storage.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            embedding = getattr(engram, "_embedding", None)
            dim = (
                embedding.dimension
                if embedding is not None
                else settings.embedding_dimension
            )
            await storage.init_schema(embedding_dimension=dim)

    if engram.llm is None:
        raise SystemExit(
            "No LLM provider configured. Set ENGRAM_LLM_PROVIDER=anthropic and "
            "ENGRAM_ANTHROPIC_API_KEY in .env (or ENGRAM_LLM_PROVIDER=openai "
            "with ENGRAM_OPENAI_API_KEY)."
        )

    print(f"LLM provider : {settings.llm_provider} / {settings.llm_model}")
    print(f"Embedding    : {settings.embedding_provider} / {settings.embedding_model}")
    print(f"Judge model  : {args.judge_model}")
    print(f"Ingest       : add_batch / batch={args.ingest_batch_size}")
    print(f"Sessions     : {len(sessions)}")
    print(f"Prompts      : {len(prompts)}")
    print(f"Question date: {question_date}")
    print(f"User name    : {user_name or '(not set)'}")

    answer_traces: list[dict[str, Any]] = []
    ingest_summary: dict[str, Any] = {}

    try:
        if not args.skip_ingest:
            if not args.no_purge:
                purged = await engram.purge(args.agent_id, args.user_id)
                print(f"purged {purged} pre-existing memories for fresh run")
            print(f"\ningesting {len(sessions)} sessions via add_batch()...")
            ingest_summary = await ingest_sessions(
                engram, sessions, args.agent_id, args.user_id,
                batch_size=args.ingest_batch_size,
            )
            print(
                f"  inserted={ingest_summary['inserted']}/{ingest_summary['rows']} "
                f"batches={ingest_summary['batches']} errors={ingest_summary['errors']} "
                f"time={ingest_summary['ingest_seconds']}s"
            )
            if ingest_summary.get("errors"):
                raise RuntimeError(
                    f"incomplete ingestion: "
                    f"{ingest_summary.get('inserted')}/{ingest_summary.get('rows')} rows, "
                    f"{ingest_summary['errors']} batch error(s); "
                    f"first: {ingest_summary.get('first_error')}"
                )

        print(f"\nanswering {len(prompts)} evaluation prompts...")
        traces_path = args.output_dir / "traces.jsonl"
        with traces_path.open("w") as traces_file:
            sem = asyncio.Semaphore(args.concurrency)

            async def process_one(item: dict[str, Any]) -> None:
                async with sem:
                    question = item["prompt"]
                    t0 = time.perf_counter()
                    error = None
                    hypothesis = ""
                    composer_model = ""
                    retrieval_trace: dict[str, Any] = {}
                    evidence = ""

                    try:
                        evidence, retrieval_trace = await retrieve_evidence(
                            engram, question, args.agent_id, args.user_id,
                            search_limit=args.search_limit,
                            graph_depth=args.graph_depth,
                            max_per_session=args.max_per_session,
                            question_date=question_date,
                            rerank=args.rerank,
                        )
                        hypothesis, composer_model = await generate_answer(
                            engram, question, question_date, evidence,
                            max_tokens=args.answer_max_tokens,
                            user_name=user_name,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"

                    trace = {
                        "id": item["id"],
                        "category": item.get("category", ""),
                        "question": question,
                        "expected_output": item["expected_output"],
                        "hypothesis": hypothesis,
                        "composer_model": composer_model,
                        "retrieval": retrieval_trace,
                        "evidence": evidence,
                        "elapsed_seconds": round(time.perf_counter() - t0, 3),
                        "error": error,
                    }
                    answer_traces.append(trace)
                    traces_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    traces_file.flush()

                    status = "ERR" if error else (
                        f"[{retrieval_trace.get('recall_intent', '?')}|"
                        f"{retrieval_trace.get('search_hits', '?')}h]"
                    )
                    print(
                        f"  Q{item['id']:>2} {status:>20} "
                        f"{trace.get('elapsed_seconds', '?')}s  "
                        f"{question[:55]}"
                    )
                    if error:
                        print(f"       error: {error}")

            tasks = [process_one(item) for item in prompts]
            await asyncio.gather(*tasks)

    finally:
        await engram.close()

    # Persist ingest summary
    (args.output_dir / "ingest.json").write_text(
        json.dumps(ingest_summary, indent=2, ensure_ascii=False)
    )

    scored = [t for t in answer_traces if not t.get("error")]
    errored_count = len(answer_traces) - len(scored)
    print(f"\n{'=' * 60}")
    print(
        f"judging {len(answer_traces)} answers with {args.judge_model} "
        f"(+{errored_count} errored)..."
    )
    judgments = await judge_all(
        answer_traces, prompts_by_id, args.judge_model, args.judge_concurrency
    )

    (args.output_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(j) for j in judgments)
    )

    total = len(judgments)
    correct = sum(j["correct"] for j in judgments)
    by_cat: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for j in judgments:
        by_cat[j["category"]][0] += 1
        by_cat[j["category"]][1] += int(j["correct"])

    abst = [j for j in judgments if j.get("abstention")]
    abst_correct = sum(j["correct"] for j in abst)

    summary = {
        "benchmark": data.get("benchmark_name"),
        "version": data.get("version"),
        "pipeline": (
            "add_batch(rendered sessions) -> "
            "search(hybrid, include_superseded) + recall(compose=False) + "
            "get_lineage + traverse_many -> composer LLM"
        ),
        "judge_model": args.judge_model,
        "answer_model": settings.llm_model,
        "memory_policy": STRESS_POLICY,
        "ingest_batch_size": args.ingest_batch_size,
        "search_limit": args.search_limit,
        "max_per_session": args.max_per_session,
        "rerank": args.rerank,
        "graph_depth": args.graph_depth,
        "answer_max_tokens": args.answer_max_tokens,
        "sessions_ingested": ingest_summary.get("inserted", 0),
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "errored": errored_count,
        "by_category": {
            cat: {"correct": c, "total": n, "accuracy": round(c / n, 4)}
            for cat, (n, c) in sorted(by_cat.items())
        },
        "abstention": {
            "total": len(abst),
            "correct": abst_correct,
            "accuracy": round(abst_correct / len(abst), 4) if abst else 0.0,
        },
        "failed_ids": sorted(j["id"] for j in judgments if not j["correct"]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\n" + "=" * 60)
    print(f"answer model : {summary['answer_model']}  |  judge: {args.judge_model}")
    print(f"pipeline     : {summary['pipeline']}")
    print(f"OVERALL      : {correct}/{total} = {summary['accuracy'] * 100:.1f}%")
    print("\nby category:")
    for cat, (n, c) in sorted(by_cat.items()):
        print(f"  {cat[:40]:40s} {c}/{n} = {c / n * 100:.1f}%")
    if abst:
        print(
            f"\nabstention : {abst_correct}/{len(abst)} = "
            f"{summary['abstention']['accuracy'] * 100:.1f}%"
        )
    if summary["failed_ids"]:
        print(f"\nfailed question ids: {summary['failed_ids']}")
    print(f"\nartifacts written to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
