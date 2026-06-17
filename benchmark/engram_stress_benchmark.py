#!/usr/bin/env python3
"""Run the custom Engram Memory Stress Test against the real Engram core.

This benchmark is purpose-built for ``data/longmemeval/test.json`` (the
"Engram Memory Stress Test Suite"), whose shape differs from LongMemEval: a
chronological list of single-statement ``context_sessions`` followed by
``evaluation_prompts`` with structured ``expected_output`` rubrics.

Pipeline (one self-contained run):
  1. Ingest each session in chronological order through ``add_conversation()``
     -- the real LLM fact-extraction + supersede/contradiction pipeline. This
     is what the suite stresses (overwrites, lineage, entity separation,
     negation), so we exercise it rather than raw ``add()``.
  2. Answer every evaluation prompt the way a production app does: gather
     evidence with intent-independent retrieval (``deep_search``) PLUS
     ``recall(compose_answer=False)`` structured lineage evidence, then make a
     SEPARATE composer LLM call to write the final answer. Recall is a
     retrieval/evidence aid here, not the answer generator -- this avoids
     making the score hostage to recall's single-intent router (which drops
     ~half of these questions into the no-retrieval ``chat``/``event`` paths).
  3. Judge each answer with an independent LLM judge (default gpt-4o) against a
     flattened view of ``expected_output``, honoring ``evaluation_rule`` and
     ``distractors_to_ignore``. Fabrication/abstention prompts use an
     abstention rubric.

Outputs JSONL traces, per-question judgments, and a summary.json into the
run directory, plus an overall + per-category accuracy table on stdout.

This run makes billable OpenAI calls (ingestion extraction + recall +
judging). Config comes from .env (same ENGRAM_* vars as examples/chatbot.py).

Usage:
    poetry run python scripts/engram_stress_benchmark.py \
        --output-dir runs/engram-stress

    # different judge / answer model is via .env (ENGRAM_LLM_MODEL); judge:
    poetry run python scripts/engram_stress_benchmark.py --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env exactly like examples/chatbot.py: map legacy aliases, then default
# the LLM provider on when an OpenAI key is present.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)
if os.environ.get("EMBEDDING_PROVIDER") and "ENGRAM_EMBEDDING_PROVIDER" not in os.environ:
    os.environ["ENGRAM_EMBEDDING_PROVIDER"] = os.environ["EMBEDDING_PROVIDER"]
if os.environ.get("OPENAI_API_KEY") and "ENGRAM_OPENAI_API_KEY" not in os.environ:
    os.environ["ENGRAM_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
if os.environ.get("ENGRAM_OPENAI_API_KEY") and "ENGRAM_LLM_PROVIDER" not in os.environ:
    os.environ["ENGRAM_LLM_PROVIDER"] = "openai"

from engram import Engram  # noqa: E402
from engram.policy import MemoryPolicy  # noqa: E402

# The stress suite mixes durable profile facts (allergy, preferences) with
# project facts that genuinely supersede each other (p95 thresholds, demo
# city, launch date). The default policy's slot/critical rules are what make
# supersede + conflict resolution work, so -- unlike the LongMemEval harness,
# which forces every turn episodic -- we keep the default policy here.
STRESS_POLICY = "default"

ACK = (
    "Understood. I've noted that and will keep it in mind."
)

# Separate composer call: Engram supplies the evidence, this LLM writes the
# answer. Kept deliberately general (not tuned to the 30 questions): it must
# use only the evidence, prefer current over superseded values, enumerate
# every relevant item for list/aggregation questions, and abstain instead of
# fabricating when the evidence does not contain the answer.
COMPOSER_SYSTEM = """You answer questions using the Engram memory evidence provided.

Treat the evidence as authoritative and answer ASSERTIVELY:
- If the answer appears anywhere in the evidence, state it directly and
  confidently. NEVER say you lack memory, and do NOT hedge ("I don't have an
  entry that explicitly says..."), when the evidence already contains the fact.
  A booked/ticketed/planned detail in the evidence IS the answer to a question
  about it.
- Only say "I do not have that in memory" when NOTHING in the evidence is
  relevant to the question. Never invent facts that are absent.
- ACTIVE / CURRENT memories are the truth now. SUPERSEDED / PREVIOUS memories
  are old values: report the current value, and mention an old value only to
  show what changed. Never present a superseded value as the current answer.
- Answer every part of the question. For "why" questions, give the stated
  reason from the evidence. For owner/approval/threshold/date/count fields,
  state the exact value.
- For list, "everything", "all", "across all projects", or summary questions,
  enumerate EVERY matching item in the evidence, not just the first.
- Keep entities distinct: do not merge two people who share a name, and do not
  attribute work to someone the evidence does not link to it. When a fact names
  only a shared first name (e.g. two different "Mahi"s), attribute it only if
  the evidence makes clear which person; otherwise leave it out.
- For membership / involvement questions ("is X involved in A, B, or C",
  "which projects is X on"), count a project ONLY when the evidence explicitly
  ties that person to that named project. Do NOT infer involvement from a
  shared class, study group, workshop, or mere co-occurrence in another
  context -- a classmate in a course is not thereby a member of your company.
- For food/restaurant/travel planning, include any remembered allergies and
  hard constraints that apply.
Be concise and factual."""


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


# Keys in expected_output that describe HOW to grade or give background -- they
# are NOT answer content the model must recite. Folding these into the required
# rubric (the old bug) caused false negatives: a correct "under 110ms" was
# failed for not also restating the supporting_context paragraph.
_JUDGE_META_KEYS = {
    "supporting_context",
    "evaluation_rule",
    "distractors_to_ignore",
    # Provenance/background: how the current state was reached. For "what is X
    # now" questions the answer is the active value, so demanding the whole
    # migration chain be recited produced false negatives (e.g. a correct
    # "MatriCare right now" failed for not restating fizlabai -> Kaggle ->
    # MatriCare). Lineage questions whose ANSWER is the chain use
    # ``chronological_arc``, which stays required.
    "lineage",
}


def build_judge_prompt(prompt_item: dict[str, Any], response: str) -> str:
    """Build a rubric-aware yes/no judge prompt for one evaluation prompt."""
    question = prompt_item["prompt"]
    category = prompt_item.get("category", "")
    expected = prompt_item["expected_output"]
    # Required rubric = only the answer-bearing fields; meta fields are handled
    # separately (or dropped) so they are never demanded as recited content.
    required = {k: v for k, v in expected.items() if k not in _JUDGE_META_KEYS}
    gold = flatten_expected(required) or flatten_expected(expected)

    if is_abstention(expected, category):
        return (
            "I will give you a question that is NOT answerable from the stored "
            "memory, the explanation of why, and a model response. Answer 'yes' "
            "only if the model correctly states the information is unknown / not "
            "in memory and does NOT fabricate a plausible-sounding reason. "
            "Answer 'no' if the model invents an explanation.\n\n"
            f"Question: {question}\n"
            f"Why unanswerable / expected: {flatten_expected(expected)}\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly abstain instead of fabricating? "
            "Answer yes or no only."
        )

    distractors = expected.get("distractors_to_ignore")
    distractor_note = ""
    if distractors:
        distractor_note = (
            "\nThese are superseded / distractor values: "
            f"{', '.join(map(str, distractors))}. It is CORRECT for the response "
            "to mention them as PAST or historical context (e.g. 'it was "
            "originally X, then Y'). Mark the response INCORRECT ONLY if it "
            "presents one of them as the CURRENT / active answer."
        )
    rule = expected.get("evaluation_rule")
    rule_note = f"\nAdditional grading rule: {rule}" if rule else ""
    context = expected.get("supporting_context")
    # Context is for the judge's understanding only; the model need not recite it.
    context_note = (
        f"\nBackground (do NOT require the response to restate this): {context}"
        if context
        else ""
    )

    return (
        "I will give you a question, a structured correct answer (the rubric), "
        "and a model response. Answer 'yes' if the response correctly contains "
        "ALL the key facts in the rubric. If the rubric lists multiple required "
        "items (people, values, dates, a chronological sequence), the response "
        "must include every one of them to be correct; a partial answer is "
        "'no'. Equivalent phrasing is fine."
        f"{distractor_note}{rule_note}{context_note}\n\n"
        f"Question: {question}\n"
        f"Correct Answer (rubric):\n{gold}\n\n"
        f"Model Response: {response}\n\n"
        "Is the model response fully correct? Answer yes or no only."
    )


async def ingest(engram: Engram, sessions: list[dict[str, Any]], agent_id: str,
                 user_id: str) -> list[dict[str, Any]]:
    """Ingest sessions chronologically through the real extraction pipeline."""
    traces: list[dict[str, Any]] = []
    for sess in sessions:
        # Anchor the date in the message text: add_conversation stamps
        # created_at=now, so the only durable temporal signal for absolute-date
        # questions is the date inside the fact content itself.
        message = f"[{sess['date_marker']}] {sess['content']}"
        t0 = time.perf_counter()
        affected = await engram.add_conversation(
            user_message=message,
            assistant_response=ACK,
            agent_id=agent_id,
            user_id=user_id,
            session_id=None,  # skip rolling-summary maintenance; recall() ignores it
        )
        traces.append(
            {
                "session_id": sess["session_id"],
                "date_marker": sess["date_marker"],
                "content": sess["content"],
                "affected_count": len(affected),
                "affected": [
                    {"memory_id": m.memory_id, "type": m.memory_type, "content": m.content}
                    for m in affected
                ],
                "ingest_seconds": round(time.perf_counter() - t0, 3),
            }
        )
        print(
            f"  session {sess['session_id']:>2} [{sess['date_marker']}] "
            f"-> {len(affected)} memory op(s)"
        )
    return traces


def _build_evidence_block(
    hits: list[Any],
    recall_answer: Any,
    superseded_hits: list[Any],
) -> str:
    """Assemble the evidence block from the Engram read APIs that earn their keep:

    - deep_search: high-recall ACTIVE facts, ranked (precision signal).
    - recall(compose_answer=False): structured current/previous/conflict lineage.
    - search(include_superseded=True): OLDER/original values, for "first",
      "originally", or "all over time" questions that active-only search drops.

    A full get_memories() inventory was tried as an aggregation recall floor and
    removed: dumping every fact diluted precision (it re-surfaced stale/active
    distractors and overwhelmed enumeration), regressing pointed and membership
    questions while fixing none of its target. Per-entity query expansion in
    expand_query covers aggregation recall instead.
    """
    lines: list[str] = []
    seen: set[str] = set()

    def emit(prefix: str, mem: Any) -> None:
        if mem.memory_id in seen:
            return
        seen.add(mem.memory_id)
        lines.append(f"{prefix} {mem.fact or mem.content}")

    # 1) recall's structured signal first -- it disambiguates current vs old.
    if recall_answer is not None:
        if recall_answer.current is not None:
            emit("CURRENT:", recall_answer.current)
        for mem in recall_answer.previous:
            when = mem.superseded_at or mem.valid_to or mem.created_at
            stamp = when.date().isoformat() if when else "unknown"
            emit(f"SUPERSEDED (until {stamp}):", mem)
        if recall_answer.conflict_note:
            lines.append(f"CONFLICT: {recall_answer.conflict_note}")

    # 2) ranked active facts (precision).
    lines.append("\n## MOST RELEVANT (current facts)")
    for hit in hits:
        emit(f"[{hit.memory.memory_type}]", hit.memory)

    # 3) older/superseded values for history/"original"/"over time" questions.
    older = [h.memory for h in superseded_hits if h.memory.status == "superseded"]
    if older:
        lines.append(
            "\n## OLDER / SUPERSEDED VALUES (prior history; use only for "
            "'first', 'originally', or 'over time' questions)"
        )
        for mem in older:
            if mem.memory_id in seen:
                continue
            seen.add(mem.memory_id)
            lines.append(f"[was] {mem.fact or mem.content}")

    return "\n".join(lines) if lines else "(no matching memory)"


async def answer_prompts(engram: Engram, prompts: list[dict[str, Any]],
                         agent_id: str, user_id: str,
                         limit: int, rerank: bool = False) -> list[dict[str, Any]]:
    """Answer each prompt: gather evidence (deep_search + recall), then compose
    the final answer with a SEPARATE LLM call."""
    traces: list[dict[str, Any]] = []
    for item in prompts:
        question = item["prompt"]
        t0 = time.perf_counter()
        error = None
        answer_text = ""
        intent = ""
        n_hits = 0
        llm_model = ""
        evidence = ""
        try:
            # Intent-independent high-recall retrieval (multi-query hybrid).
            # rerank overfetches candidates and re-orders them with the local
            # cross-encoder against the original question before truncating.
            hits = await engram.deep_search(
                question, agent_id, user_id=user_id, limit=limit, n_queries=4,
                rerank=rerank,
            )
            n_hits = len(hits)
            # Recall as an evidence aid (NOT the answer): structured lineage.
            recall_answer = await engram.recall(
                question, agent_id, user_id=user_id, compose_answer=False
            )
            intent = recall_answer.intent
            # Older/original values for "first"/"originally"/"over time" Qs.
            superseded_hits = await engram.search(
                question, agent_id, user_id=user_id, limit=8,
                include_superseded=True,
            )

            evidence = _build_evidence_block(hits, recall_answer, superseded_hits)
            resp = await engram.llm.complete_full(
                [
                    {"role": "system", "content": COMPOSER_SYSTEM},
                    {
                        "role": "system",
                        "content": f"<engram_memory_evidence>\n{evidence}\n</engram_memory_evidence>",
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=600,
                temperature=0.0,
            )
            answer_text = resp.content.strip()
            llm_model = resp.model
        except Exception as exc:  # noqa: BLE001 - record, don't abort the batch
            error = f"{type(exc).__name__}: {exc}"
        traces.append(
            {
                "id": item["id"],
                "category": item.get("category", ""),
                "question": question,
                "expected_output": item["expected_output"],
                "hypothesis": answer_text,
                "recall_intent": intent,
                "deep_search_hits": n_hits,
                "evidence": evidence,
                "composer_model": llm_model,
                "answer_seconds": round(time.perf_counter() - t0, 3),
                "error": error,
            }
        )
        status = "ERR" if error else f"[{intent}|{n_hits}h]"
        print(f"  Q{item['id']:>2} {status:>14} {question[:66]}")
    return traces


async def judge_all(prompts_by_id: dict[int, dict[str, Any]],
                    answer_traces: list[dict[str, Any]], judge_model: str,
                    concurrency: int) -> list[dict[str, Any]]:
    """Independently judge each answer with an LLM. Routes to Anthropic when the
    judge model is a ``claude-*`` model, otherwise OpenAI."""
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
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                return " ".join(parts).strip().lower()
            resp = await openai_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            return (resp.choices[0].message.content or "").strip().lower()

    async def judge_one(trace: dict[str, Any]) -> dict[str, Any]:
        item = prompts_by_id[trace["id"]]
        if trace.get("error"):
            # An errored answer is a failed answer; count it wrong, don't judge.
            return {
                "id": trace["id"],
                "category": trace["category"],
                "correct": False,
                "verdict": "error",
                "abstention": is_abstention(item["expected_output"], trace["category"]),
            }
        verdict = await verdict_for(build_judge_prompt(item, trace["hypothesis"]))
        return {
            "id": trace["id"],
            "category": trace["category"],
            "correct": verdict.startswith("yes"),
            "verdict": verdict,
            "abstention": is_abstention(item["expected_output"], trace["category"]),
        }

    return await asyncio.gather(*(judge_one(t) for t in answer_traces))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=_REPO_ROOT / "data" / "longmemeval" / "test.json",
        help="Path to the stress-test JSON (default: data/longmemeval/test.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "runs" / "engram-stress",
        help="Directory for traces / judgments / summary.",
    )
    parser.add_argument("--agent-id", default="engram-stress")
    parser.add_argument("--user-id", default="stress-user")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="Independent LLM judge model (default gpt-4o).",
    )
    parser.add_argument("--judge-concurrency", type=int, default=6)
    parser.add_argument(
        "--recall-limit",
        type=int,
        default=12,
        help="Max evidence items recall() retrieves per question.",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Do NOT purge the agent's memories before ingesting (default: purge).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion (reuse memories already stored for this agent/user).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Re-order deep_search candidates with the local cross-encoder "
        "reranker before composing (needs sentence-transformers).",
    )
    parser.add_argument(
        "--rejudge-only",
        type=Path,
        help="Re-score an existing traces.jsonl (only judge calls; no DB, no "
        "ingestion, no answer generation). Cheapest way to apply judge fixes.",
    )
    return parser.parse_args()


def _write_summary(data: dict[str, Any], judgments: list[dict[str, Any]],
                   output_dir: Path, judge_model: str, sessions_ingested: int) -> None:
    """Persist judgments + summary.json and print the scorecard."""
    (output_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(j) for j in judgments)
    )
    total = len(judgments)
    correct = sum(j["correct"] for j in judgments)
    by_cat: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for j in judgments:
        by_cat[j["category"]][0] += 1
        by_cat[j["category"]][1] += int(j["correct"])
    summary = {
        "benchmark": data.get("benchmark_name"),
        "judge_model": judge_model,
        "answer_model": os.environ.get("ENGRAM_LLM_MODEL", "unknown"),
        "sessions_ingested": sessions_ingested,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_category": {
            cat: {"correct": c, "total": n} for cat, (n, c) in sorted(by_cat.items())
        },
        "failed_ids": sorted(j["id"] for j in judgments if not j["correct"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"OVERALL : {correct}/{total} = {summary['accuracy'] * 100:.1f}%")
    print("\nby category:")
    for cat, (n, c) in sorted(by_cat.items()):
        print(f"  {cat[:40]:40s} {c}/{n}")
    print(f"\nfailed question ids: {summary['failed_ids']}")
    print(f"artifacts written to {output_dir}")


async def rejudge_only(args: argparse.Namespace, data: dict[str, Any]) -> None:
    """Re-score an existing traces.jsonl with the current judge logic."""
    traces = [
        json.loads(line)
        for line in args.rejudge_only.read_text().splitlines()
        if line.strip()
    ]
    # traces.jsonl carries question/category/expected_output, so we can rebuild
    # the judge inputs without the dataset, but prefer the live dataset by id.
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
    print(f"re-judging {len(traces)} answers from {args.rejudge_only} "
          f"with {args.judge_model}...")
    judgments = await judge_all(
        prompts_by_id, traces, args.judge_model, args.judge_concurrency
    )
    out_dir = args.rejudge_only.parent
    _write_summary(data, judgments, out_dir, args.judge_model, sessions_ingested=0)


async def main() -> None:
    args = parse_args()
    data = json.loads(args.data_path.read_text())

    if args.rejudge_only is not None:
        await rejudge_only(args, data)
        return

    sessions = data["context_sessions"]
    prompts = data["evaluation_prompts"]
    prompts_by_id = {p["id"]: p for p in prompts}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    engram = Engram(memory_policy=STRESS_POLICY)
    await engram.connect()
    if engram.llm is None:
        raise SystemExit(
            "No LLM provider configured. Set ENGRAM_LLM_PROVIDER=openai and "
            "ENGRAM_OPENAI_API_KEY in .env."
        )

    ingest_traces: list[dict[str, Any]] = []
    try:
        if not args.skip_ingest:
            if not args.no_purge:
                purged = await engram.purge(args.agent_id, args.user_id)
                print(f"purged {purged} pre-existing memories for fresh run")
            print(f"ingesting {len(sessions)} sessions (real extraction pipeline)...")
            ingest_traces = await ingest(engram, sessions, args.agent_id, args.user_id)

        print(f"\nanswering {len(prompts)} evaluation prompts with recall()...")
        answer_traces = await answer_prompts(
            engram, prompts, args.agent_id, args.user_id, args.recall_limit,
            rerank=args.rerank,
        )
    finally:
        await engram.close()

    print(f"\njudging {len(answer_traces)} answers with {args.judge_model}...")
    judgments = await judge_all(
        prompts_by_id, answer_traces, args.judge_model, args.judge_concurrency
    )

    # Persist everything.
    (args.output_dir / "ingest.jsonl").write_text(
        "\n".join(json.dumps(t) for t in ingest_traces)
    )
    (args.output_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(t) for t in answer_traces)
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

    summary = {
        "benchmark": data.get("benchmark_name"),
        "version": data.get("version"),
        "judge_model": args.judge_model,
        "answer_model": os.environ.get("ENGRAM_LLM_MODEL", "unknown"),
        "answer_path": "deep_search + recall(compose_answer=False) -> separate composer LLM",
        "memory_policy": STRESS_POLICY,
        "sessions_ingested": len(ingest_traces),
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_category": {
            cat: {"correct": c, "total": n, "accuracy": round(c / n, 4)}
            for cat, (n, c) in sorted(by_cat.items())
        },
        "failed_ids": sorted(j["id"] for j in judgments if not j["correct"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"answer model : {summary['answer_model']}  |  judge: {args.judge_model}")
    print(f"OVERALL      : {correct}/{total} = {summary['accuracy'] * 100:.1f}%")
    print("\nby category:")
    for cat, (n, c) in sorted(by_cat.items()):
        print(f"  {cat[:40]:40s} {c}/{n}")
    if summary["failed_ids"]:
        print(f"\nfailed question ids: {summary['failed_ids']}")
    print(f"\nartifacts written to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
