#!/usr/bin/env python3
"""Official-rubric LLM judge for LongMemEval harness output.

Reproduces the question-type-specific judge prompts from the LongMemEval
reference evaluation (``get_anscheck_prompt``), including the preference
rubric, temporal off-by-one tolerance, knowledge-update handling, and the
abstention check for ``*_abs`` questions.

Scores a harness ``traces.jsonl`` (which already carries question, gold
answer, hypothesis, and question_type) and reports overall plus per-type
accuracy. The judge model is independent of the answer model on purpose.

Usage:
    OPENAI_API_KEY=... python3 scripts/lme_judge.py \
        --traces runs/lme-s-clean-generic/traces.jsonl \
        --judge-model gpt-4o

If OPENAI_API_KEY is unset, ENGRAM_OPENAI_API_KEY is used.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
) -> str:
    """Build the official LongMemEval judge prompt for one item."""
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            f"Question: {question}\nExplanation: {answer}\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
    if task == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. In addition, do not penalize off-by-one errors for the number "
            "of days. If the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 "
            "days when the answer is 18), the model's response is still "
            "correct.\n\n"
            f"Question: {question}\nCorrect Answer: {answer}\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response "
            "should be considered as correct as long as the updated answer is "
            "the required answer.\n\n"
            f"Question: {question}\nCorrect Answer: {answer}\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if task == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. The "
            "model does not need to reflect all the points in the rubric. The "
            "response is correct as long as it recalls and utilizes the user's "
            "personal information correctly.\n\n"
            f"Question: {question}\nRubric: {answer}\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    return (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct "
        "answer or contains all the intermediate steps to get the correct "
        "answer, you should also answer yes. If the response only contains a "
        "subset of the information required by the answer, answer no.\n\n"
        f"Question: {question}\nCorrect Answer: {answer}\n"
        f"Model Response: {response}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


async def judge_row(
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    row: dict[str, Any],
) -> dict[str, Any]:
    qid = str(row.get("question_id", ""))
    abstention = qid.endswith("_abs")
    prompt = get_anscheck_prompt(
        row.get("question_type", ""),
        str(row.get("question", "")),
        str(row.get("answer", "")),
        str(row.get("hypothesis", "")),
        abstention=abstention,
    )
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    correct = verdict.startswith("yes")
    return {
        "question_id": qid,
        "question_type": row.get("question_type", ""),
        "abstention": abstention,
        "correct": correct,
        "verdict": verdict,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("runs/lme-s-clean-generic/traces.jsonl"),
        help="Harness traces.jsonl to score.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="OpenAI model used as judge (LongMemEval reference uses gpt-4o).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max concurrent judge calls.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write per-question judgments (default: alongside traces).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
        "ENGRAM_OPENAI_API_KEY"
    )
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or ENGRAM_OPENAI_API_KEY.")

    rows = load_rows(args.traces)
    scored = [r for r in rows if not r.get("error")]
    errored = len(rows) - len(scored)

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(judge_row(client, args.judge_model, sem, r) for r in scored)
    )

    # Honest denominator: an errored sample is a sample the system failed to
    # answer, so it counts as incorrect. Report accuracy over every attempted
    # question (judged + errored), and the judged-only rate separately.
    overall_correct = sum(r["correct"] for r in results)
    attempted = len(rows)
    judged = len(results)

    by_type: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["question_type"]][0] += 1
        by_type[r["question_type"]][1] += int(r["correct"])

    abst = [r for r in results if r["abstention"]]
    abst_correct = sum(r["correct"] for r in abst)

    out_path = args.output or args.traces.with_name("judgments.jsonl")
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"judge model       : {args.judge_model}")
    print(f"attempted         : {attempted} (judged: {judged}, errored: {errored})")
    print(
        f"OVERALL (errors=wrong): {overall_correct}/{attempted} = "
        f"{(overall_correct / attempted * 100) if attempted else 0:.1f}%"
    )
    print(
        f"judged-only        : {overall_correct}/{judged} = "
        f"{(overall_correct / judged * 100) if judged else 0:.1f}%"
    )
    print("\nby question_type:")
    for qtype, (n, c) in sorted(by_type.items()):
        print(f"  {qtype:28s} {c}/{n} = {c / n * 100:.1f}%")
    if abst:
        print(
            f"\nabstention (_abs) : {abst_correct}/{len(abst)} = "
            f"{abst_correct / len(abst) * 100:.1f}%"
        )
    print(f"\nper-question judgments written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
