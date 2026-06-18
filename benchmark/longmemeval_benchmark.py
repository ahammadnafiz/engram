#!/usr/bin/env python3
"""Run LongMemEval (ICLR 2025) end-to-end through Engram's advanced APIs.

Pipeline (per question, fully isolated by agent_id):
  1. INGEST: Render each haystack session into pre-formed memory rows
     (one per turn, or one per whole session via ``--memory-unit``) and
     bulk-insert them with ``add_batch()`` -- embeddings only, no per-turn
     LLM fact-extraction. This mirrors ``longmemeval_harness.py``: ingestion
     becomes cheap and parallel, and all reasoning over conflicting/updated
     facts moves to the retrieval + composer layer. The date is anchored in
     each row's text (``[date] ROLE: content``) so temporal questions resolve
     correctly, and ``original_session_id``/``haystack_date`` ride along in
     metadata for chronological grouping at retrieval time.

  2. RETRIEVE: Multi-surface evidence gathering:
     a) ``search(mode="hybrid", include_superseded=True)`` -- a wide candidate
        pool, diversified across sessions (round-robin, user turns first) down
        to the evidence budget; active + historical facts explicitly tagged.
     b) ``recall(compose_answer=False, question_date=...)`` -- structured
        current/previous/conflict lineage evidence (read from the top-level
        RecallAnswer fields), with temporal intents anchored to the question
        date.
     c) ``get_lineage()`` -- superseded predecessors of retrieved active facts,
        so a corrected value's prior history is never lost.
     d) ``traverse_many()`` -- multi-hop graph relations from the top search
        hits, rendered as context.

  3. GENERATE: A separate composer LLM call writes the final answer from
     the assembled evidence block. The prompt is optimized for Engram's
     tagged [ACTIVE] / [SUPERSEDED] output.

  4. JUDGE: An independent LLM judge (default gpt-4o) scores each answer
     against the gold answer using per-question-type prompts from the
     official LongMemEval evaluation rubric.

Outputs JSONL traces, per-question judgments, and a summary.json into the
run directory, plus overall + per-type accuracy on stdout.

This run makes billable LLM calls (recall + composing + judging) plus
batched embedding calls at ingest. Ingestion no longer runs per-turn LLM
extraction. Config comes from .env (same ENGRAM_* vars as the stress
benchmark).

Usage:
    poetry run python scripts/longmemeval_benchmark.py \\
        --output-dir runs/longmemeval-bench \\
        --max-samples 10

    # Full 500-question run:
    poetry run python scripts/longmemeval_benchmark.py \\
        --output-dir runs/longmemeval-full

    # Different judge model:
    poetry run python scripts/longmemeval_benchmark.py \\
        --judge-model claude-sonnet-4-20250514 \\
        --output-dir runs/longmemeval-claude-judge

    # Re-judge an existing run (no DB, no ingestion, no answering):
    poetry run python scripts/longmemeval_benchmark.py \\
        --rejudge-only runs/longmemeval-bench/traces.jsonl \\
        --judge-model gpt-4o \\
        --output-dir runs/longmemeval-bench
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
# Environment bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

# Map legacy env vars
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

# Auto-detect LLM provider: prefer Anthropic if its key is set, else OpenAI.
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

# Default models: Anthropic Claude Haiku for both answering and judging.
DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"

# Embedding model
DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

# Local (sentence-transformers) embedding defaults for --local-embedding.
# Runs entirely on-device: no API, no cost, no rate limits.
LOCAL_EMBEDDING_PROVIDER = "sentence-transformers"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIMENSION = 384

# LongMemEval histories are conversational events. We keep the default policy
# so that add_conversation() exercises the full extraction, supersession, and
# conflict-resolution pipeline -- which is the whole point of benchmarking
# Engram's advanced APIs vs. raw vector stores.
BENCHMARK_POLICY = "default"

# Default ingestion knobs (mirrors longmemeval_harness.py).
# max_memory_chars is generous: the unique fact index is on md5(fact), so large
# facts don't hit the btree row-size limit, and turns should not be truncated.
DEFAULT_MEMORY_UNIT = "turn"
DEFAULT_MAX_MEMORY_CHARS = 16000
DEFAULT_INGEST_BATCH_SIZE = 64

# LongMemEval question types (from the official dataset).
QUESTION_TYPES = [
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]

# ---------------------------------------------------------------------------
# Smoke test: curated question IDs for quick validation.
# These span multiple question types and difficulty levels.
# ---------------------------------------------------------------------------
SMOKE_TEST_QUESTION_IDS = [
    "51a45a95",
    "a82c026e",
    "0a995998",
    "gpt4_59c863d7",
    "gpt4_f2262a51",
    "c4a1ceb8",
    "gpt4_2f8be40d",
    "88432d0a",
    "d23cf73b",
    "gpt4_7fce9456",
    "7024f17c",
    "gpt4_5501fe77",
    "2318644b",
    "2ce6a0f2",
    "gpt4_d12ceb0e",
    "a9f6b44c",
    "d851d5ba",
    "5a7937c8",
    "gpt4_ab202e7f",
    "edced276",
    "bf659f65",
    "0edc2aef",
]


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
- List every relevant memory; try to list all memories relevant to what the user wants to do! Eg. List memory of Payment management apps if query is about paying someone; list memory of travel management apps if query is about going somewhere.

- For counting: enumerate each item with date. Apply the question's EXACT verb/qualifier strictly (e.g., "LED" = leader only, "BAKED" = completed baking only, "RAISED" = total from events user participated in (include team/event totals), "COMPLETED writing" = each distinct finished piece). Count multiple items in a single memory separately. Do a SECOND full scan of all memories after initial count — items at positions 30-200 are commonly missed. Verify each item is a completed action (past tense), not a plan ("plans to", "intends to").
- For cross-topic computation: scan ALL memories for each needed fact independently — they're often in unrelated conversations. List: (a) what you need, (b) where each appears, (c) the computation.
- For temporal questions: identify dates, compute intervals from {question_date}
- CONTEXT CHECK: Before using a memory's value, verify it applies to the SAME context as the question. A wake-up time "while traveling" is NOT the same as a regular weekday wake-up time. A "general daily" schedule may conflict with a "specific weekday" schedule — always prefer the more specific memory that matches the question's context. List the context of each memory (weekday routine vs. travel vs. weekend vs. specific day) and only use values from the matching context.
- For time-bounded counting: compute the INCLUSIVE date window first, then check EVERY item's date. Err on inclusion for ambiguous dates.
- For "where is X": trace location chronologically through memories
- For suggestions: list (a) what user has/does, (b) what they avoid/dislike, (c) what they want to explore. Check every suggestion against (b) before including.
- State your conclusion

The user will only see text outside the <mem_thinking> tags.

Rules:

1. **Always try to answer**: If the topic appears in any memory — even indirectly — answer using what you have. Don't refuse for one missing detail.

2. **Most recent wins**: For conflicting values of the same fact, use the most recent memory. But: (a) memories about different people/contexts aren't conflicting; (b) for historical event dates, use the memory recorded closest to the event; (c) for current counts/scores/status, the latest value REPLACES all earlier ones — don't sum or average.

Similarly, when memories give two numbers for the same metric (e.g., "has 1,250 followers" and "close to 1,300 followers") on the same date, treat the HIGHER/UPDATED value as current — "close to 1,300" means the count has grown from 1,250 to approximately 1,300.

3. **ACTIVE vs SUPERSEDED memories**: Memories tagged [ACTIVE] are the truth now. Memories tagged [SUPERSEDED] are old values that have been overwritten. For current-state questions, use [ACTIVE] memories. For "what was it before?" or "originally" questions, use [SUPERSEDED] memories. Never present a superseded value as the current answer.

4. **Time-bounded questions**: Compute the date window from {question_date}. Show date arithmetic in <mem_thinking>. Scan EVERY memory for events in range. "Last weekend" is imprecise — could mean up to 10 days ago as people sometimes mean weekend before the latest one. "Last 3 months" can include boundary days of the 4th month back.

"Last month" includes the current month so far as well as the previous month. Eg. "last month" in Late May includes all of April. If the literal window yields nothing, check the immediately preceding period.

5. **Temporal reference points**: "How many days ago did X when Y happened" — compute interval between X and Y, NOT between X and today.

6. **Counting and ordering**: Scan ALL memories first to last. Build a numbered list in <mem_thinking> with date and position. Deduplicate by matching dates/descriptions. Count items in a single memory separately.
Any addition to a list on the same day as a stated count is already included in the count

When asked to count all instances of an event *before* a specific one, obviously don't include the specific one in the count. Eg. "how many restaurants did i visit before eating at Pizza Hut?". Obviously don't include Pizza Hut in the count

7. **Use only the memories**: Don't invent numbers, prices, or addresses.

8. **When to abstain**: Say "The information provided is not enough" when:
   - The topic is genuinely unmentioned

- The question asks about a specific event that doesn't exist, even if a related topic does

- IMPORTANT: If the query uses a specific but WRONG role/title/entity (e.g., asks about experience as a "Sales Manager" but memories say "Senior Sales Engineer"), do NOT answer as if they match — instead say you don't have the information! Always lean towards abstention in these cases! Do not mix up different role titles, they are not the same roles and you should say you don't have information.

   - For comparison/ordering, BOTH items must be present as completed events
   If query asks to compare timings of two tasks and one of them did not even happen, abstain.
   Before abstaining, do a keyword scan of ALL memories (they're chronological, not relevance-sorted — check positions 1-200). Only abstain if NO keywords match.
   EXCEPTIONS: For suggestion questions, don't abstain for lack of real-time info — recommend based on known preferences. If you lack exact brand but have the store, output the store.

9. **Yes/no and comparison**: "Did I ever do X?" with no matching memory = "No." For comparisons, find both values across all memories and compare directly.

10. **Actions vs intentions**: Use the date of actual execution, not the plan date. "Decided to" or "took X for servicing" = action initiated. Only treat as plan if explicit future-tense ("plans to", "will"). A plan with a specified date and no update = assume completed on that date. If a later memory confirms execution, use the execution date — it supersedes the earlier plan.

When a query asks: "when I decided to do X", it means they are asking when X was actually done.

11. **User facts vs assistant advice**: "User..." = actual experience. "Assistant..." = advice. Prefer user-stated facts for personal questions. Don't convert currencies unless user stated the conversion.

12. **Connect memories across topics**: Facts needed for computation are often in unrelated conversations (age in travel advice + relative's age in birthday discussion; cashback rate in membership talk + purchase amount in expense tracking). Search ALL memories for each fact independently.

13. **Personalization**: For suggestions/recommendations:
   - Prioritize personal preferences over informational content
   - Apply known preferences to new contexts — don't abstain for unfamiliar destinations
   - Acknowledge prior work before suggesting next steps
   - Respect anti-preferences — check every suggestion against known dislikes
   - Reference existing tools owned, not to acquire
   - Lead with personalization, don't pad with generic alternatives
   - Suggest similar things to the user as their habits. Eg. Logging basketball scores in a app they do usually. Eg. Adding travel logs to a travel logging app they use usually.
   - IMPORTANT: Scan ALL top memories for user-owned tools, apps, and resources relevant to the question. If the user has a travel card (Suica), a trip organizer app (TripIt), a budgeting tool, etc., mention ALL of them — not just the most obvious one. Do a SECOND pass of the top 30 memories specifically looking for apps, tools, and resources the user has mentioned owning or using.

14. **Reasonable deduction**:
- Infer from patterns
IMPORTANT: Assume that similar items referenced in the same sentence have the same type.
Eg. "User ate lunch, which was the third meal with this chicken fajitas". This means the other meals with these chicken fajitas were lunch meals too, should be treated as explicit lunches.

15. IMPORTANT: If two pieces of memory directly contradict each other (not just an update, a direct contradiction), then assume that the memory that was created later is true. Doesn't matter if a different one "appears" more reliable. If on the same day, trust the one at a later time.

- Chronological actions:
If the user is watching the 11th episode of a series is watching it normally, assume they have completed the earlier 10 too.

- If you lack a name but have a description, answer with the description.

**Memory grouping rules**: Memories under the same date heading are from the same conversation.
- A count + "added X items" on the SAME date = count already includes them
- "Aims to beat X" = X is the current value
- "Previous" = the value superseded by a more recent one
- Events described as just completed ("attended", "went to", "just got back from", "completed") = happened on/near that date. Undated actions = assume the event happened on the memory's date.

# Misc Rules
- Count class projects too when asked about users' projects. Class projects = projects.
- Most old (Eg. ancestral, vintage, heritage) items count as antiques too!
- If you don't have chords for a song (but have notes), output the notes. Song notes count as chord progressions.
- Starting a *diorama project* (eg. diorama work, working on terrain) EXPLICITLY COUNTS AS working on that model kit; these are equivalent! Always count such items.
- Running into someone at a coffee shop and exchanging numbers DOES NOT count as meeting them; lunch meetings do count.
- Potlucks/feasts/birthday parties count as dinner parties (BBQ doesn't).
- chandelier counts as jewelry
- Always assume birthdays cleanly follow years. Ie. User was 22 in 2022; they will be 23 in 2023.
- "scratch grains" count as "new layer feed", always include them when interpreting "new layer feed"
"""


# ===========================================================================
# Judge Prompts (official LongMemEval rubrics)
# ===========================================================================

# Single unified judge prompt for ALL question types (incl. abstention). Ported
# verbatim from the memory-benchmarks reference harness. It fixes the per-type
# rubric's false-negatives: the judge over-says "no", so it carries an explicit
# bias check plus superset/semantic-equivalence rules, and emits a verdict after
# a <judge_thinking> block (parsed by _parse_yes_no_judgment).
JUDGE_PROMPT = """I will give you a question, a correct answer (or rubric), and a model response. Decide whether the model response is correct.

CORE PRINCIPLE — Semantic equivalence: Judge by MEANING, not exact words. Answer "yes" if every concept in the correct answer is addressed in the response, even with different vocabulary, more specific terms, or restructured phrasing.

IMPORTANT BIAS CHECK: You have a tendency to say "no" too quickly. Before concluding "no", you MUST verify the answer is truly wrong, not just differently worded. When in doubt, lean toward "yes".

Rules:

**Equivalence & Supersets**
- Equivalent or superset responses are correct. Extra details are fine unless proven to be factually wrong. Extra qualifiers are fine unless proven to be wrong. E.g., "a blue dress and a matching necklace" is correct when the answer is "a blue dress."
- If a response captures the most specific part (exact item/place/name) but omits a broader container, it's correct.
- Same factual meaning with different phrasing = correct (e.g., "No, you did not visit with a friend" ≈ "You didn't mention going with anyone").
- Adding scope qualifiers like "regular-season" or "excluding X" is fine as long as the core value is correct. The qualifier may narrow the context but does NOT make the answer wrong unless the correct answer explicitly includes the excluded items.

**Lists & Compound Terms**
- For list answers, match each item by semantic meaning. A concept is covered if restated via synonyms, sub-concepts, or related terms. Adding methodological detail or rewording verbs to near-synonyms is acceptable.
- A broad term like "A and B significance" is covered if the response addresses the topic area through related specific terms, even without naming each component literally.
- If some items as listed as "or"s, "maybe"s and potential answers, it's okay if the answer does not include those.
- If two items in a list achieve the same purpose, listing just one of them is fine.

IMPORTANT: The "anti-preference" items are very specific!
Eg. Someone "not interested in general AI topics" could be very interested in specific AI topics in general AI *conferences*; those are not the same thing and should be accepted! topics != conferences

**Numbers & Precision**
- Hedging ("at least 3", "approximately") is fine if the core number matches. A range that includes the correct answer is correct.
Generally, if the user themself would be satisfied by the response, it is acceptable. Ie. If the answer is conditional on information they would have (eg. their birthday, some hidden dependent information), and would be correct with that information, that is acceptable.
- More precise answers are correct: "22 days" matches "3 weeks"; "over $270" matches "$270."; "9 1/2 months" matches "9 months";

- Rough answers are correct: "about nine months" ≈ "9 months; "8 months and 20 days" matches "9 months";

- Off-by-one errors on days/weeks/months are acceptable.
- Approximate unit conversions are equivalent: "14 weeks" ≈ "3 months", "6 months" ≈ "half a year."
- Round time ranges generously: 7 months and 16 days ≈ 8 months.
- Notes instead of chords are acceptable when justified
- A correct number with added context (e.g., "about 5 months ago (around December 2022)") is correct — the parenthetical date is supplementary, not a contradiction.

**Dates & Temporal**
- Date format variations are equivalent: "February 1st" = "Feb 1, 2023" = "on February 1."
- Same-day event ordering swaps are acceptable.
- Outdated info alongside the correct updated answer is acceptable if the current value is identified.
- "recent" is upto 6 years ago, which means 2017+
- References like "last weekend", "last Wednesday", etc. are imprecise - people sometimes mean the weekend/Wednesday before the latest one if they're near it. "Last 3 months" can include boundary days of the 4th month back. "Last month" includes the current month so far. Be flexible with such timestamps

**Counting Edge Cases**
- If correct answer is "0" or "nothing found," model saying "not enough information" is also correct.
- Similarly, If correct answer is "not enough information", model saying "0" or "nothing found," is also correct.

**Preference/Personalization Rubrics** (apply in order):
1. Correct if the response demonstrates awareness of user's personal context (preferences, habits, interests). Need not satisfy every rubric point.
2. Primary criterion: do main suggestions align with what the user WANTS?
3. Anti-preferences: evaluate the OVERALL thrust, not keyword scanning. If the response largely suggests correct options, minor incidental references to "not-preferred" things are fine.
4. Mentioning a phone app as a MEANS to a preferred activity (e.g., meditation app for sleep) is not "suggesting phone use." Judge by the activity, not delivery mechanism.
5. "May not prefer" = mild preference, not hard prohibition. Secondary/context-dependent inclusion is fine.
6. Explicit acknowledgment of anti-preferences (e.g., "keep screens off") strengthens correctness.
7. Context-dependent suggestions are acceptable (reading is fine on a bus even if rubric flags visual attention activities). Adjacent genres alongside preferred ones are additive, not contradictory.
8. If the rubric mentions specific user resources/tools (e.g., "Suica card", "TripIt app"), the response is correct if it demonstrates awareness of the user's MAIN personal context even if it does not name every specific tool. The rubric is a guide, not a checklist.

**Abstention Matching**
- If correct answer = unanswerable/abstention, ANY phrasing that conveys "I don't have this information" is correct, regardless of what partial context is mentioned or omitted.
- Saying "not enough information" while mentioning partial related context = correct abstention.
- Saying "no record of X" or "only have plans for X, not actual dates" = correct abstention.
- The key test: does the response REFUSE to answer the question? If yes, it matches an abstention ground truth, period.

FINAL CHECK: Before answering "no," you MUST reason through these steps:
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


# Dedicated rubric for *_abs (unanswerable) questions. The unified judge does
# general semantic-equivalence reasoning and gets distracted by the partial
# context a correctly-abstaining answer includes -- wrongly failing it. This
# narrow prompt asks only "did the model decline to answer?" and is far more
# reliable on abstention (restored 30/30 in the original per-type judge).
ABSTENTION_JUDGE_PROMPT = """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}
Explanation: {answer}
Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only."""


def get_judge_prompt(
    question_type: str,
    question_id: str,
    question: str,
    answer: str,
    response: str,
) -> str:
    """Pick the judge prompt: a dedicated abstention rubric for *_abs questions,
    the unified semantic-equivalence rubric for everything else (question_type is
    unused, kept for call-site compatibility)."""
    if question_id.endswith("_abs"):
        return ABSTENTION_JUDGE_PROMPT.format(
            question=question, answer=str(answer), response=response
        )
    return JUDGE_PROMPT.format(question=question, answer=str(answer), response=response)


# ===========================================================================
# Helpers
# ===========================================================================

def _to_human_date(date_str: str) -> str:
    """Convert LongMemEval date format '2023/05/20 (Sat) 02:21' to a human date."""
    # Extract just the date part
    match = re.match(r"(\d{4}/\d{2}/\d{2})", date_str)
    if match:
        from datetime import datetime
        try:
            dt = datetime.strptime(match.group(1), "%Y/%m/%d")
            return dt.strftime("%A, %B %d, %Y")
        except ValueError:
            pass
    return date_str


def _parse_question_date(date_str: str) -> "datetime | None":
    """Parse LongMemEval question_date into a datetime for recall's temporal
    reasoning ('yesterday', 'last week' resolve relative to this)."""
    from datetime import datetime
    if not date_str:
        return None
    m = re.match(r"(\d{4}/\d{2}/\d{2})(?:\s+\(\w+\))?(?:\s+(\d{2}:\d{2}))?", date_str)
    if not m:
        return None
    fmt = "%Y/%m/%d %H:%M" if m.group(2) else "%Y/%m/%d"
    raw = f"{m.group(1)} {m.group(2)}" if m.group(2) else m.group(1)
    try:
        return datetime.strptime(raw, fmt)
    except ValueError:
        return None


def _bounded_text(text: str, max_chars: int) -> str:
    """Truncate to max_chars keeping head and tail (harness parity)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max(1, (max_chars - 32) // 2)
    return f"{text[:half]}\n[...truncated...]\n{text[-half:]}"


def _render_turn(turn: dict[str, Any], date: str) -> str:
    """Render a single turn as '[date] ROLE: content'."""
    role = str(turn.get("role", "unknown")).upper()
    return f"[{date}] {role}: {turn.get('content', '')}"


def _render_session(session: list[dict[str, Any]], date: str) -> str:
    """Render an entire session as one date-anchored block."""
    return "\n".join(_render_turn(turn, date) for turn in session)


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Split items into batches of at most ``size``."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_memory_rows(
    sample: dict[str, Any],
    *,
    agent_id: str,
    user_id: str,
    memory_unit: str,
    max_memory_chars: int,
) -> list[dict[str, Any]]:
    """Render a question's haystack into add_batch() row dicts.

    Ported from ``longmemeval_harness.iter_memories``. No LLM extraction:
    each turn (or whole session) becomes one episodic memory verbatim, with
    the date anchored in the text and routing metadata attached. ``user_id``
    is set on every row so the benchmark's user-scoped search/purge match.
    """
    rows: list[dict[str, Any]] = []
    session_ids = sample.get("haystack_session_ids", [])
    dates = sample.get("haystack_dates", [])
    sessions = sample.get("haystack_sessions", [])

    # Insert chronologically: conflict_key supersession (driven by the memory
    # policy's slot rules) is last-writer-wins by insert order, so the most
    # recent value of a slotted fact must be ingested last to win the slot.
    indexed = sorted(zip(session_ids, dates, sessions), key=lambda x: x[1])

    for original_session_id, date, session in indexed:
        if memory_unit == "session":
            content = _bounded_text(_render_session(session, date), max_memory_chars)
            rows.append(
                {
                    "content": content,
                    "main_content": content,
                    "agent_id": agent_id,
                    "user_id": user_id,
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
            content = _bounded_text(_render_turn(turn, date), max_memory_chars)
            rows.append(
                {
                    "content": content,
                    "main_content": content,
                    "agent_id": agent_id,
                    "user_id": user_id,
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
    return rows


# ===========================================================================
# Ingestion
# ===========================================================================

async def ingest_sessions(
    engram: Engram,
    sample: dict[str, Any],
    agent_id: str,
    user_id: str,
    *,
    memory_unit: str,
    max_memory_chars: int,
    batch_size: int,
) -> dict[str, Any]:
    """Bulk-ingest a question's haystack via add_batch() (no per-turn LLM).

    Renders every turn (or whole session) into pre-formed episodic rows and
    inserts them in ``batch_size`` chunks. Embeddings are batched per chunk;
    there is no fact-extraction, dedup, or supersede decisioning at ingest
    time -- that work moves to retrieval + the composer. Returns a summary
    rather than per-pair traces.
    """
    rows = build_memory_rows(
        sample,
        agent_id=agent_id,
        user_id=user_id,
        memory_unit=memory_unit,
        max_memory_chars=max_memory_chars,
    )

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
        "memory_unit": memory_unit,
        "rows": len(rows),
        "inserted": inserted,
        "batches": batches,
        "errors": errors,
        "first_error": first_error,
        "ingest_seconds": round(time.perf_counter() - t0, 3),
    }


# ===========================================================================
# Evidence Building
# ===========================================================================

def _build_evidence_block(
    search_results: list[Any],
    recall_answer: Any | None,
    graph_context: str,
    lineage_superseded: list[Any] | None = None,
) -> str:
    """Assemble the evidence block from Engram's retrieval surfaces.

    Combines:
    - search(include_superseded=True): active + historical facts, tagged.
    - recall(compose_answer=False): structured current/previous/conflict lineage.
      RecallAnswer exposes current/previous/conflict_note at the TOP LEVEL
      (``.evidence`` is just the flat supporting-memory list).
    - get_lineage(): superseded predecessors of retrieved active facts, so a
      corrected value's history is never lost even when the old row didn't
      independently match the query.
    - traverse_many(): rendered graph relations for multi-hop reasoning.
    """
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

    # 2) Lineage history for retrieved active facts: surface superseded
    #    predecessors so "what was it before" / knowledge-update questions have
    #    the prior values even if they weren't top search hits.
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

    # 3) Graph relations for multi-hop context.
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

    Ported from ``longmemeval_harness.search_evidence_set``. Plain vector
    search clusters: for an aggregation question ("how many X") the top hits
    collapse onto the single most-similar conversation, so the budget is spent
    on near-duplicate turns of one session while the other qualifying events go
    unretrieved (undercount). This spreads selection across sessions, and within
    each session prefers USER turns over ASSISTANT turns -- assistant chatter
    (e.g. suggested garnishes) otherwise gets miscounted as user facts
    (overcount).

    When ``rerank`` is set the candidate order is already a cross-encoder
    relevance ranking, so the user-first nudge is dropped: it actively buries
    the ASSISTANT turn that *is* the answer for "remind me what you recommended"
    (single-session-assistant) questions. Relevance order is respected as-is,
    with only the cross-session round-robin applied.
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

    # Stable sort: original rank, nudged by role so user turns lead each group.
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
    # Backfill any remaining budget by original rank.
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
    """Retrieve evidence using Engram's advanced APIs.

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
    retrieved_session_ids: list[str] = []

    try:
        # 1) Hybrid search with superseded memories included. Overfetch a wider
        # candidate pool, then diversify across sessions down to search_limit so
        # aggregation questions see many conversations, not one cluster.
        # SearchQuery hard-caps limit at 100 (memory/models.py), so the candidate
        # pool can't exceed that; diversification only reduces coverage when the
        # final search_limit is set below 100.
        # When reranking, fetch the full 100-candidate pool so the cross-encoder
        # scores everything before we down-select; without it, fetch only what
        # the diversified budget needs.
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
        # Record which haystack sessions were surfaced, for lexical-miss triage:
        # comparing these against the gold answer_session_ids separates a
        # retrieval miss (answer-bearing session never retrieved) from a reading
        # error (retrieved but answered wrong).
        retrieved_session_ids = sorted({
            str(r.memory.metadata.get("original_session_id"))
            for r in search_results
            if r.memory.metadata.get("original_session_id") is not None
        })

        # Sort by date for chronological presentation
        search_results.sort(
            key=lambda r: r.memory.metadata.get("haystack_date", "")
        )

        # 2) Recall as evidence aid (NOT the answer). Pass the question's
        # reference date so temporal intents ("yesterday", "last week") resolve
        # correctly, and a limit aligned with the evidence budget.
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
            # recall is best-effort; don't fail the question over it
            recall_intent = f"error:{type(exc).__name__}"

        # 3) Lineage preservation: for retrieved ACTIVE facts that belong to a
        # real conflict lineage (lineage_id != memory_id), pull get_lineage and
        # surface the superseded predecessors so corrected values keep their
        # history. Gated to lineaged facts only -> cheap (most turns stand alone).
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
                    pass  # best-effort enrichment
        n_lineage_superseded = len(lineage_superseded)

        # 4) Graph traversal from top search hits
        graph_context = ""
        seed_ids = [
            r.memory.memory_id for r in search_results
            if r.memory.metadata.get("status") != "superseded"
        ][:5]  # Top 5 active seeds
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
                pass  # Graph is best-effort enrichment

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
        "retrieved_session_ids": retrieved_session_ids,
        "retrieval_seconds": round(time.perf_counter() - t0, 3),
        "error": error,
    }
    return evidence, trace


# ===========================================================================
# Answer Generation
# ===========================================================================

async def generate_answer(
    engram: Engram,
    question: str,
    question_date: str,
    evidence: str,
    max_tokens: int,
) -> tuple[str, str]:
    """Generate answer from evidence using Engram's LLM.

    Returns:
        (answer_text, model_name)
    """
    assert engram.llm is not None

    system = COMPOSER_SYSTEM.format(question_date=question_date)

    # Mirror the reference harness layout: evidence first, the question LAST, and
    # an explicit "Reasoning and answer:" primer. Question-last plus the primer
    # measurably reduces premature abstention vs. a bare question turn.
    user_prompt = (
        f"<engram_memory_evidence>\n{evidence}\n</engram_memory_evidence>\n\n"
        f"Today's Date: {question_date}\n"
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

    The unified judge emits reasoning in <judge_thinking> then a verdict line.
    Take the region after the closing tag, prefer a standalone yes/no line
    (last wins), then fall back to the last yes/no token, then startswith.
    Ported from the reference harness so scoring matches exactly.
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
    judge_model: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Judge answers with an independent LLM.

    Routes to Anthropic for claude-* models, otherwise OpenAI.
    """
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
        # The unified judge thinks in <judge_thinking> before the verdict, so it
        # needs real headroom (the old 10-token cap truncated mid-reasoning).
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
        if trace.get("error"):
            return {
                "question_id": trace["question_id"],
                "question_type": trace.get("question_type", ""),
                "correct": False,
                "verdict": "error",
                "abstention": trace["question_id"].endswith("_abs"),
            }
        prompt = get_judge_prompt(
            question_type=trace.get("question_type", ""),
            question_id=trace["question_id"],
            question=trace["question"],
            answer=str(trace["answer"]),
            response=trace["hypothesis"],
        )
        raw = await verdict_for(prompt)
        correct = _parse_yes_no_judgment(raw)
        return {
            "question_id": trace["question_id"],
            "question_type": trace.get("question_type", ""),
            "correct": correct,
            "verdict": "yes" if correct else "no",
            "judge_raw": raw[:300],
            "abstention": trace["question_id"].endswith("_abs"),
        }

    return await asyncio.gather(*(judge_one(t) for t in answer_traces))


# ===========================================================================
# Main runner
# ===========================================================================

async def run_sample(
    engram: Engram,
    sample: dict[str, Any],
    *,
    agent_id: str,
    user_id: str,
    search_limit: int,
    graph_depth: int,
    answer_max_tokens: int,
    skip_ingest: bool,
    memory_unit: str,
    max_memory_chars: int,
    ingest_batch_size: int,
    max_per_session: int,
    rerank: bool = False,
) -> dict[str, Any]:
    """Run a single LongMemEval question end-to-end."""
    question_id = str(sample["question_id"])
    question = sample["question"]
    answer = sample["answer"]
    question_date = sample.get("question_date", "")
    question_type = sample.get("question_type", "")
    start = time.perf_counter()
    error = None
    hypothesis = ""
    composer_model = ""
    ingest_summary: dict[str, Any] = {}
    retrieval_trace: dict[str, Any] = {}
    evidence = ""

    try:
        # 1. INGEST
        if not skip_ingest:
            ingest_summary = await ingest_sessions(
                engram, sample, agent_id, user_id,
                memory_unit=memory_unit,
                max_memory_chars=max_memory_chars,
                batch_size=ingest_batch_size,
            )
            # Refuse to score a question whose store is incomplete: a partial
            # haystack (e.g. dropped batches from an embedding rate limit) would
            # masquerade as a wrong answer and poison the accuracy number. Mark
            # it errored so it is surfaced, not silently counted.
            if ingest_summary.get("errors"):
                raise RuntimeError(
                    f"incomplete ingestion: "
                    f"{ingest_summary.get('inserted')}/{ingest_summary.get('rows')} "
                    f"rows, {ingest_summary['errors']} batch error(s); "
                    f"first: {ingest_summary.get('first_error')}"
                )

        # 2. RETRIEVE
        evidence, retrieval_trace = await retrieve_evidence(
            engram, question, agent_id, user_id,
            search_limit=search_limit,
            graph_depth=graph_depth,
            max_per_session=max_per_session,
            question_date=question_date,
            rerank=rerank,
        )

        # 3. GENERATE
        if engram.llm is not None:
            hypothesis, composer_model = await generate_answer(
                engram, question, question_date, evidence,
                max_tokens=answer_max_tokens,
            )
        else:
            error = "No LLM configured; cannot generate answer"

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "question_date": question_date,
        "hypothesis": hypothesis,
        "agent_id": agent_id,
        "composer_model": composer_model,
        "ingest_summary": ingest_summary,
        "retrieval": retrieval_trace,
        "evidence": evidence,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark through Engram's advanced APIs."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=_REPO_ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json",
        help="Path to the LongMemEval JSON dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "benchmark" / "runs" / "longmemeval-benchmark",
        help="Directory for traces / judgments / summary.",
    )
    parser.add_argument("--agent-prefix", default="lmebench")
    parser.add_argument("--user-id", default="lme-user")
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
        help="Number of questions to process concurrently (default 5).",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model for recall + answer composition (default {DEFAULT_LLM_MODEL}). Ingestion uses embeddings only.",
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
            "an API provider: no cost, no rate limits. Switches the embedding "
            "provider/model/dimension defaults together; explicit "
            "--embedding-model / --embedding-dimension still win."
        ),
    )
    parser.add_argument(
        "--memory-unit",
        choices=("turn", "session"),
        default=DEFAULT_MEMORY_UNIT,
        help=(
            "Ingestion granularity: 'turn' stores each message as its own "
            "memory, 'session' stores each whole session as one memory "
            f"(default {DEFAULT_MEMORY_UNIT})."
        ),
    )
    parser.add_argument(
        "--max-memory-chars",
        type=int,
        default=DEFAULT_MAX_MEMORY_CHARS,
        help=(
            "Max characters stored per memory, preserving head and tail "
            f"(default {DEFAULT_MAX_MEMORY_CHARS}). Use 0 to disable truncation."
        ),
    )
    parser.add_argument(
        "--ingest-batch-size",
        type=int,
        default=DEFAULT_INGEST_BATCH_SIZE,
        help=f"Memories per add_batch() call (default {DEFAULT_INGEST_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=100,
        help=(
            "Final memories per question after session-diversified selection "
            "(active + superseded). A wider candidate pool is overfetched and "
            "round-robined across sessions down to this number."
        ),
    )
    parser.add_argument(
        "--max-per-session",
        type=int,
        default=4,
        help=(
            "Max turns kept per session in the diversified evidence set. Lower "
            "values spread coverage across more conversations (better for "
            "aggregation/counting questions)."
        ),
    )
    parser.add_argument(
        "--graph-depth",
        type=int,
        default=1,
        help="Max graph traversal depth from search seeds. 0 to disable.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Cross-encoder rerank the full 100-candidate pool against the "
            "question before session-diversified selection. Cuts irrelevant "
            "turns out of the evidence block (precision). Requires the optional "
            "sentence-transformers dependency (already present with "
            "--local-embedding)."
        ),
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=4000,
        help=(
            "Max tokens for the composer LLM answer. Must be generous: the "
            "composer prompt does verbose step-by-step work inside "
            "<mem_thinking>, and a small budget gets consumed before the "
            "user-facing answer is emitted (truncation = silent wrong answer)."
        ),
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
        help="Maximum number of questions to run. Omit for the full dataset.",
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
        "--no-purge",
        action="store_true",
        help="Do NOT purge the agent's memories before ingesting.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion (reuse already-stored memories for this agent).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed sample.",
    )
    parser.add_argument(
        "--rejudge-only",
        type=Path,
        help="Re-score an existing traces.jsonl (no DB, no ingestion, no "
        "answer generation). Cheapest way to apply judge fixes.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only the 22 curated smoke-test questions (quick validation).",
    )
    parser.add_argument(
        "--clean-db",
        action="store_true",
        help="Drop and recreate the database schema before running.",
    )
    args = parser.parse_args()

    # --local-embedding flips the embedding defaults to the on-device provider,
    # but only where the user did not explicitly override them.
    if args.local_embedding:
        args.embedding_provider = LOCAL_EMBEDDING_PROVIDER
        if args.embedding_model == DEFAULT_EMBEDDING_MODEL:
            args.embedding_model = LOCAL_EMBEDDING_MODEL
        if args.embedding_dimension == DEFAULT_EMBEDDING_DIMENSION:
            args.embedding_dimension = LOCAL_EMBEDDING_DIMENSION

    return args


def load_samples(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load and filter LongMemEval samples."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    # --smoke-test overrides --question-id with the curated set
    if args.smoke_test:
        wanted = set(SMOKE_TEST_QUESTION_IDS)
        data = [s for s in data if s.get("question_id") in wanted]
        # Preserve the curated order
        order = {qid: i for i, qid in enumerate(SMOKE_TEST_QUESTION_IDS)}
        data.sort(key=lambda s: order.get(s.get("question_id", ""), 999))
    elif args.question_id:
        wanted = set(args.question_id)
        data = [s for s in data if s.get("question_id") in wanted]

    if args.question_type:
        wanted_types = set(args.question_type)
        data = [s for s in data if s.get("question_type") in wanted_types]

    data = data[args.offset:]
    if args.max_samples is not None:
        data = data[:args.max_samples]
    return data


async def rejudge_only(args: argparse.Namespace) -> None:
    """Re-score an existing traces.jsonl with the current judge logic."""
    traces = [
        json.loads(line)
        for line in args.rejudge_only.read_text().splitlines()
        if line.strip()
    ]
    print(
        f"re-judging {len(traces)} answers from {args.rejudge_only} "
        f"with {args.judge_model}..."
    )
    judgments = await judge_all(traces, args.judge_model, args.judge_concurrency)

    # Write results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(j) for j in judgments)
    )

    total = len(judgments)
    correct = sum(j["correct"] for j in judgments)
    by_type: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for j in judgments:
        by_type[j["question_type"]][0] += 1
        by_type[j["question_type"]][1] += int(j["correct"])

    print("\n" + "=" * 60)
    print(f"OVERALL : {correct}/{total} = {correct / total * 100:.1f}%" if total else "No samples")
    print("\nby question_type:")
    for qtype, (n, c) in sorted(by_type.items()):
        print(f"  {qtype:35s} {c}/{n} = {c / n * 100:.1f}%")
    print(f"\njudgments written to {args.output_dir / 'judgments.jsonl'}")


async def main() -> None:
    args = parse_args()

    if args.rejudge_only is not None:
        await rejudge_only(args)
        return

    data_path = args.data_path
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    samples = load_samples(data_path, args)
    if not samples:
        raise SystemExit("No samples selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]

    # Build Engram settings: Anthropic LLM + Gemini embeddings (overridable via CLI)
    settings = get_settings()
    settings = settings.model_copy(update={
        "llm_model": args.llm_model,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "allow_embedding_dimension_change": True,
        "near_duplicate_threshold": 1.0,  # keep all benchmark memories
    })

    engram = Engram(settings=settings, memory_policy=BENCHMARK_POLICY)
    await engram.connect()

    if args.clean_db:
        print("Cleaning database (dropping public schema)...")
        storage = getattr(engram, "_storage", None)
        if storage is not None:
            await storage.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            # Re-initialize the schema AT THE ACTIVE EMBEDDING DIMENSION. Without
            # this, init_schema() recreates the vector column at the schema
            # default (1536), so non-1536 providers (e.g. local 384d) fail every
            # insert with "expected 1536 dimensions". Mirror connect() and use
            # the provider's real dimension.
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
    print(f"Ingest       : add_batch / unit={args.memory_unit} "
          f"batch={args.ingest_batch_size} max_chars={args.max_memory_chars}")
    print(f"Samples      : {len(samples)}")

    traces: list[dict[str, Any]] = []
    
    try:
        traces_path = args.output_dir / "traces.jsonl"
        with traces_path.open("w") as traces_file:
            sem = asyncio.Semaphore(args.concurrency)

            async def process_one(index: int, sample: dict[str, Any]) -> None:
                async with sem:
                    question_id = str(sample["question_id"])
                    agent_id = f"{args.agent_prefix}-{run_id}-{question_id}"
                    label = f"{index}/{len(samples)} [{question_id}]"

                    print(f"\n{'=' * 60}\nRunning {label}: {sample['question'][:70]}\n  sessions: {len(sample.get('haystack_sessions', []))}")

                    # Purge pre-existing memories for isolation
                    if not args.no_purge and not args.skip_ingest:
                        try:
                            purged = await engram.purge(agent_id, args.user_id)
                            if purged:
                                print(f"[{question_id}] purged {purged} pre-existing memories")
                        except Exception:
                            pass  # Agent may not exist yet

                    try:
                        trace = await run_sample(
                            engram,
                            sample,
                            agent_id=agent_id,
                            user_id=args.user_id,
                            search_limit=args.search_limit,
                            graph_depth=args.graph_depth,
                            answer_max_tokens=args.answer_max_tokens,
                            skip_ingest=args.skip_ingest,
                            memory_unit=args.memory_unit,
                            max_memory_chars=args.max_memory_chars,
                            ingest_batch_size=args.ingest_batch_size,
                            max_per_session=args.max_per_session,
                            rerank=args.rerank,
                        )
                    except Exception as exc:
                        if args.fail_fast:
                            raise
                        trace = {
                            "question_id": question_id,
                            "question_type": sample.get("question_type", ""),
                            "question": sample.get("question", ""),
                            "answer": sample.get("answer", ""),
                            "question_date": sample.get("question_date", ""),
                            "hypothesis": "",
                            "agent_id": agent_id,
                            "composer_model": "",
                            "ingest_summary": {},
                            "retrieval": {},
                            "evidence": "",
                            "elapsed_seconds": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }

                    traces.append(trace)
                    traces_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    traces_file.flush()

                    status = "ERR" if trace.get("error") else "OK"
                    retrieval = trace.get("retrieval", {})
                    ingest = trace.get("ingest_summary", {})
                    print(
                        f"[{question_id}] {status} | "
                        f"ingested={ingest.get('inserted', '?')}/{ingest.get('rows', '?')} "
                        f"ingest_errors={ingest.get('errors', '?')} "
                        f"search={retrieval.get('search_hits', '?')} "
                        f"lineage={retrieval.get('lineage_superseded', '?')} "
                        f"graph={retrieval.get('graph_hits', '?')} "
                        f"recall_intent={retrieval.get('recall_intent', '?')} | "
                        f"{trace.get('elapsed_seconds', '?')}s"
                    )
                    if ingest.get("errors") and ingest.get("first_error"):
                        print(f"[{question_id}] first ingest error: {ingest['first_error']}")
                    if trace.get("hypothesis"):
                        print(f"[{question_id}] answer: {trace['hypothesis'][:120]}")

                    # Cleanup: purge after each question to avoid cross-contamination
                    if not args.no_purge and not args.skip_ingest:
                        try:
                            await engram.purge(agent_id, args.user_id)
                        except Exception:
                            pass

            tasks = [process_one(index, sample) for index, sample in enumerate(samples, start=1)]
            await asyncio.gather(*tasks)

    finally:
        await engram.close()

    # Judge
    scored = [t for t in traces if not t.get("error")]
    errored_count = len(traces) - len(scored)
    print(f"\n{'=' * 60}")
    print(f"judging {len(scored)} answers with {args.judge_model} "
          f"(+{errored_count} errored)...")
    judgments = await judge_all(traces, args.judge_model, args.judge_concurrency)

    # Persist
    (args.output_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(j) for j in judgments)
    )

    total = len(judgments)
    correct = sum(j["correct"] for j in judgments)
    by_type: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for j in judgments:
        by_type[j["question_type"]][0] += 1
        by_type[j["question_type"]][1] += int(j["correct"])

    abst = [j for j in judgments if j["abstention"]]
    abst_correct = sum(j["correct"] for j in abst)

    summary = {
        "benchmark": "LongMemEval (longmemeval_s_cleaned)",
        "pipeline": "add_batch(rendered turns/sessions) -> search(include_superseded) + recall(compose=False) + traverse_many -> composer LLM",
        "judge_model": args.judge_model,
        "answer_model": settings.llm_model,
        "memory_policy": BENCHMARK_POLICY,
        "memory_unit": args.memory_unit,
        "max_memory_chars": args.max_memory_chars,
        "ingest_batch_size": args.ingest_batch_size,
        "search_limit": args.search_limit,
        "rerank": args.rerank,
        "graph_depth": args.graph_depth,
        "answer_max_tokens": args.answer_max_tokens,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "errored": errored_count,
        "by_question_type": {
            qtype: {"correct": c, "total": n, "accuracy": round(c / n, 4)}
            for qtype, (n, c) in sorted(by_type.items())
        },
        "abstention": {
            "total": len(abst),
            "correct": abst_correct,
            "accuracy": round(abst_correct / len(abst), 4) if abst else 0.0,
        },
        "failed_ids": sorted(
            j["question_id"] for j in judgments if not j["correct"]
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # Print scorecard
    print("\n" + "=" * 60)
    print(f"answer model : {summary['answer_model']}  |  judge: {args.judge_model}")
    print(f"pipeline     : {summary['pipeline']}")
    print(f"OVERALL      : {correct}/{total} = {summary['accuracy'] * 100:.1f}%")
    print("\nby question_type:")
    for qtype, (n, c) in sorted(by_type.items()):
        print(f"  {qtype:35s} {c}/{n} = {c / n * 100:.1f}%")
    if abst:
        print(
            f"\nabstention (_abs) : {abst_correct}/{len(abst)} = "
            f"{summary['abstention']['accuracy'] * 100:.1f}%"
        )
    if summary["failed_ids"]:
        print(f"\nfailed question ids: {summary['failed_ids'][:20]}")
        if len(summary["failed_ids"]) > 20:
            print(f"  ... and {len(summary['failed_ids']) - 20} more")
    print(f"\nartifacts written to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
