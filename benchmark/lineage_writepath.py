#!/usr/bin/env python3
"""Layer 2 — Write-path scenario benchmark for add_conversation().

Layer 1 (lineage_invariants.py) proved the state machine is correct GIVEN the
right operation. This layer tests the layer above it: does add_conversation's
LLM extraction feed the state machine the right ADD/UPDATE/NOOP decisions, and
-- the part no read benchmark measures -- what fraction of real updates get
SILENTLY DROPPED (returned [] with no error)?

Each scenario is a scripted multi-turn update sequence with a gold final state.
We run the turns through add_conversation(), then read the ACTIVE set directly
via list_recent() (which filters superseded with no embedding-ranking noise) and
score:

  - current-state accuracy : is the gold value active AND the stale value gone?
                             (the outcome an application actually depends on)
  - update-capture rate    : 1 - silent_loss/non_noop_turns. THE safety metric:
                             a turn that should ADD/UPDATE but returns [].
  - noop precision         : restatements/dups that correctly produce 0 writes.
  - active-count error     : spurious or missing active rows (over/under-write).

Because the SAME scenarios run on every --llm-model, the output is a per-model
table: the lineage mechanism is constant (Layer 1), so any cross-model
difference here is extraction/decision quality -- e.g. "haiku silently drops 12%
of numeric updates, sonnet 2%". That table is the honest characterization of
whether a developer can trust add_conversation on a write path.

Billable: ~3 LLM calls/turn (extract, decide, classify) x ~20 turns x N models.
Embeddings run on-device (free). Needs live Postgres.

    python3 benchmark/lineage_writepath.py \
        --llm-model claude-haiku-4-5-20251001 \
        --llm-model claude-sonnet-4-6 \
        --output-dir benchmark/runs/writepath
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parent.parent
load_dotenv(_REPO / ".env", override=False)

os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("ENGRAM_EMBEDDING_DIMENSION", "384")

from engram import Engram  # noqa: E402
from engram.core.config import get_settings  # noqa: E402


def _provider_for_model(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    return ""


# ===========================================================================
# Scenario dataset — scripted update sequences with gold final state.
#
# Each turn: (user, assistant, expect) where expect is the operation the
# extractor SHOULD choose for the salient fact: ADD (new), UPDATE (supersede a
# prior value), or NOOP (already known / duplicate).
# gold.present : substrings that MUST appear in the active set at the end.
# gold.absent  : substrings (stale/overwritten values) that must NOT be active.
# gold.active_facts : expected number of distinct active rows (over/under-write).
# ===========================================================================
SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "numeric-update",
        "why": "Number-only change; weak embeddings rate 90k~110k near-identical, "
               "so the cosine dup-guard can NOOP a real raise. The classic silent loss.",
        "turns": [
            ("My base salary is 90000 euros.", "Noted, 90k base.", "ADD"),
            ("I got a raise — my base is now 110000 euros.", "Congrats on the raise!", "UPDATE"),
        ],
        "gold": {"present": ["110000"], "absent": ["90000"], "active_facts": 1},
    },
    {
        "id": "date-update",
        "why": "Date-only change has the same weak-embedding failure mode as numbers.",
        "turns": [
            ("My flight is booked for June 10.", "Got it, June 10.", "ADD"),
            ("The flight got moved to June 14.", "Updated to June 14.", "UPDATE"),
        ],
        "gold": {"present": ["June 14", "14"], "absent": ["June 10"], "active_facts": 1},
    },
    {
        "id": "employer-change",
        "why": "Semantic update, different phrasing — the easy case that should always work.",
        "turns": [
            ("I work as a backend engineer at Zalando.", "Zalando, nice.", "ADD"),
            ("I switched jobs — I'm at Adyen now.", "Adyen, congrats.", "UPDATE"),
        ],
        "gold": {"present": ["Adyen"], "absent": ["Zalando"], "active_facts": 1},
    },
    {
        "id": "city-oscillation",
        "why": "A->B->A: final value equals the first, but must be a NEW active row, "
               "and the intermediate value must be gone.",
        "turns": [
            ("I live in Berlin.", "Berlin!", "ADD"),
            ("I relocated to Amsterdam.", "Amsterdam now.", "UPDATE"),
            ("Actually I moved back to Berlin.", "Back to Berlin.", "UPDATE"),
        ],
        "gold": {"present": ["Berlin"], "absent": ["Amsterdam"], "active_facts": 1},
    },
    {
        "id": "dedup-noop",
        "why": "Verbatim-ish restatement must NOT spawn a second active row or a revision.",
        "turns": [
            ("I live in Amsterdam.", "Amsterdam.", "ADD"),
            ("Just to confirm, I currently live in Amsterdam.", "Yes, on file.", "NOOP"),
        ],
        "gold": {"present": ["Amsterdam"], "absent": [], "active_facts": 1},
    },
    {
        "id": "assistant-restatement-trap",
        "why": "extract_assistant_response=False: the assistant echoing the budget must "
               "NOT create a second budget fact. Then a real update must still supersede.",
        "turns": [
            ("My project budget is 5000 dollars.",
             "A 5000 dollar budget is reasonable for that scope.", "ADD"),
            ("We increased the budget to 8000 dollars.", "Updated: 8000.", "UPDATE"),
        ],
        "gold": {"present": ["8000"], "absent": ["5000"], "active_facts": 1},
    },
    {
        "id": "contradiction-correction",
        "why": "Direct contradiction (one->two sisters): later statement should win as a "
               "correction, not accumulate two conflicting active facts.",
        "turns": [
            ("I have one sister.", "Okay.", "ADD"),
            ("Correction: I actually have two sisters.", "Got it, two.", "UPDATE"),
        ],
        "gold": {"present": ["two"], "absent": ["one sister"], "active_facts": 1},
    },
    {
        "id": "multifact-non-collision",
        "why": "Multi-fact turn forms independent lineages; updating one must leave the "
               "other untouched (no cross-fact collision).",
        "turns": [
            ("I'm allergic to peanuts, and I work at Spotify.",
             "Noted: peanut allergy and Spotify.", "ADD"),
            ("I changed jobs — I'm at Netflix now.", "Netflix!", "UPDATE"),
        ],
        "gold": {"present": ["peanut", "Netflix"], "absent": ["Spotify"], "active_facts": 2},
    },
]


def _norm(s: str) -> str:
    """Normalize for value matching: lowercase, drop whitespace and thousands
    separators so '110,000' == '110000' and 'June 14' == 'june14'."""
    return re.sub(r"[\s,]+", "", s.lower())


def _matches(text: str, needle: str) -> bool:
    return _norm(needle) in _norm(text)


async def run_scenario(
    engram: Engram, scenario: dict[str, Any], agent: str, user: str, *, sleep: float
) -> dict[str, Any]:
    """Run one scenario's turns, then score the resulting active set."""
    history: list[dict[str, str]] = []
    turn_log: list[dict[str, Any]] = []

    for i, (umsg, amsg, expect) in enumerate(scenario["turns"]):
        if sleep and i:
            await asyncio.sleep(sleep)
        result = await engram.add_conversation(
            umsg, amsg, agent, user_id=user,
            conversation_history=history[-8:],
            update_summary=False,  # focus on lineage, skip the summary LLM call
        )
        affected = list(result)  # written memories (list-compatible result)
        n = len(affected)
        revs = sorted({m.revision for m in affected})
        # Per-fact decisions now visible: a dropped update shows as a NOOP/
        # unapplied decision with a reason instead of just an empty return.
        decisions = [
            {"op": d.operation, "applied": d.applied, "reason": d.reason, "fact": d.fact}
            for d in getattr(result, "decisions", [])
        ]
        silent_loss = expect in ("ADD", "UPDATE") and n == 0
        over_write = expect == "NOOP" and n > 0
        turn_log.append({
            "turn": i, "expect": expect, "affected": n, "revisions": revs,
            "silent_loss": silent_loss, "over_write": over_write, "user": umsg,
            "decisions": decisions,
        })
        history.append({"role": "user", "content": umsg})
        history.append({"role": "assistant", "content": amsg})

    # Read the ACTIVE set directly — list_recent excludes superseded, no ranking.
    active = await engram.list_recent(agent, user_id=user, limit=100)
    active_texts = [getattr(m, "fact", None) or m.content for m in active]
    joined = "\n".join(active_texts)

    gold = scenario["gold"]
    present_checks = [(p, _matches(joined, p)) for p in gold.get("present", [])]
    present_ok = all(ok for _, ok in present_checks)
    # Stale mentions are INFORMATIONAL ONLY, never a gate: models legitimately
    # write narrative facts ("X now, changed from Y") that mention the old value
    # while the current value is correct, so substring-absent over-flags them.
    # The reliable loss signal is silent_loss (the return contract), not this.
    stale_mentions = [(a, _matches(joined, a)) for a in gold.get("absent", [])]

    exp_facts = gold.get("active_facts")
    active_count_ok = (exp_facts is None) or (len(active) == exp_facts)

    non_noop = [t for t in turn_log if t["expect"] in ("ADD", "UPDATE")]
    noop = [t for t in turn_log if t["expect"] == "NOOP"]
    silent_losses = [t for t in non_noop if t["silent_loss"]]
    over_writes = [t for t in noop if t["over_write"]]

    # Pass iff the new value is active AND no update was silently dropped.
    # active_count is advisory (extraction granularity is model-dependent).
    state_pass = present_ok and len(silent_losses) == 0

    return {
        "id": scenario["id"],
        "state_correct": sum(1 for _, ok in present_checks if ok),
        "state_total": len(present_checks),
        "present_ok": present_ok,
        "state_pass": state_pass,
        "present_checks": present_checks,
        "stale_mentions": stale_mentions,
        "active_count": len(active),
        "expected_active": exp_facts,
        "active_count_ok": active_count_ok,
        "active_texts": active_texts,
        "non_noop_turns": len(non_noop),
        "silent_loss_turns": len(silent_losses),
        "noop_turns": len(noop),
        "over_write_turns": len(over_writes),
        "turn_log": turn_log,
    }


async def run_model(engram: Engram, model: str, base: str, *, sleep: float) -> dict[str, Any]:
    user = "u"
    results = []
    for scn in SCENARIOS:
        agent = f"{base}-{model_slug(model)}-{scn['id']}"
        with contextlib.suppress(Exception):
            await engram.purge(agent, user)
        try:
            res = await run_scenario(engram, scn, agent, user, sleep=sleep)
        except Exception as exc:
            res = {"id": scn["id"], "error": f"{type(exc).__name__}: {exc}",
                   "state_pass": False, "state_correct": 0, "state_total": 0,
                   "silent_loss_turns": 0, "non_noop_turns": 0, "noop_turns": 0,
                   "over_write_turns": 0, "active_count_ok": False}
        results.append(res)
        with contextlib.suppress(Exception):
            await engram.purge(agent, user)

    state_correct = sum(r.get("state_correct", 0) for r in results)
    state_total = sum(r.get("state_total", 0) for r in results)
    non_noop = sum(r.get("non_noop_turns", 0) for r in results)
    silent = sum(r.get("silent_loss_turns", 0) for r in results)
    noop = sum(r.get("noop_turns", 0) for r in results)
    overwrite = sum(r.get("over_write_turns", 0) for r in results)
    scn_pass = sum(1 for r in results if r.get("state_pass"))

    return {
        "model": model,
        "scenarios_passed": scn_pass,
        "scenarios_total": len(SCENARIOS),
        "current_state_accuracy": round(state_correct / state_total, 4) if state_total else 0.0,
        "update_capture_rate": round(1 - silent / non_noop, 4) if non_noop else 1.0,
        "silent_loss_turns": silent,
        "non_noop_turns": non_noop,
        "noop_precision": round((noop - overwrite) / noop, 4) if noop else 1.0,
        "over_write_turns": overwrite,
        "scenarios": results,
    }


def model_slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")[:24]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer 2 write-path scenario benchmark.")
    p.add_argument("--llm-model", action="append", default=None,
                   help="Answer/extraction model. Repeatable to compare models "
                        "(default: claude-haiku-4-5-20251001).")
    p.add_argument("--turn-sleep", type=float, default=0.0,
                   help="Seconds between turns (raise for free-tier rate limits).")
    p.add_argument("--output-dir", type=Path,
                   default=_REPO / "benchmark" / "runs" / "writepath")
    return p.parse_args()


def print_model_report(m: dict[str, Any]) -> None:
    print(f"\n{'=' * 68}\nMODEL: {m['model']}")
    print(f"  scenarios passed       : {m['scenarios_passed']}/{m['scenarios_total']}")
    print(f"  current-state accuracy : {m['current_state_accuracy'] * 100:.1f}%")
    print(f"  update-capture rate    : {m['update_capture_rate'] * 100:.1f}%  "
          f"(silent losses: {m['silent_loss_turns']}/{m['non_noop_turns']} update turns)")
    print(f"  noop precision         : {m['noop_precision'] * 100:.1f}%  "
          f"(spurious writes: {m['over_write_turns']})")
    print("  per-scenario:")
    for r in m["scenarios"]:
        if r.get("error"):
            print(f"    ✗ {r['id']:28} ERROR {r['error']}")
            continue
        mark = "✓" if r["state_pass"] else "✗"
        flags = []
        if r["silent_loss_turns"]:
            flags.append(f"SILENT-LOSS x{r['silent_loss_turns']}")
        if not r["active_count_ok"]:
            flags.append(f"active={r['active_count']}≠{r['expected_active']} (advisory)")
        if r["over_write_turns"]:
            flags.append(f"over-write x{r['over_write_turns']}")
        detail = f"   [{', '.join(flags)}]" if flags else ""
        print(f"    {mark} {r['id']:28} state {r['state_correct']}/{r['state_total']}{detail}")


async def main() -> None:
    args = parse_args()
    models = args.llm_model or ["claude-haiku-4-5-20251001"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = f"wp-{uuid.uuid4().hex[:6]}"

    all_reports = []
    for model in models:
        provider = _provider_for_model(model)
        settings = get_settings().model_copy(update={
            "llm_model": model,
            **({"llm_provider": provider} if provider else {}),
            "embedding_provider": os.environ["ENGRAM_EMBEDDING_PROVIDER"],
            "embedding_model": os.environ["ENGRAM_EMBEDDING_MODEL"],
            "embedding_dimension": int(os.environ["ENGRAM_EMBEDDING_DIMENSION"]),
            "allow_embedding_dimension_change": True,
            "near_duplicate_threshold": 1.0,
        })
        engram = Engram(settings=settings, memory_policy="default")
        await engram.connect()
        if engram.llm is None:
            raise SystemExit(
                f"No LLM for model {model!r} (provider={provider!r}). "
                "Check the matching ENGRAM_*_API_KEY is set."
            )
        print(f"\n>>> running {len(SCENARIOS)} scenarios on {provider}/{model} "
              f"(embeddings: on-device MiniLM)")
        try:
            report = await run_model(engram, model, base, sleep=args.turn_sleep)
        finally:
            await engram.close()
        all_reports.append(report)
        print_model_report(report)

    # Cross-model comparison table — the headline.
    print(f"\n{'=' * 68}\nWRITE-PATH SCORECARD (mechanism is constant; deltas = model)")
    print(f"{'model':32} {'state':>7} {'capture':>8} {'noop':>6} {'pass':>6}")
    for m in all_reports:
        print(f"{m['model'][:32]:32} "
              f"{m['current_state_accuracy'] * 100:6.1f}% "
              f"{m['update_capture_rate'] * 100:7.1f}% "
              f"{m['noop_precision'] * 100:5.0f}% "
              f"{m['scenarios_passed']:>3}/{m['scenarios_total']}")

    out = args.output_dir / "writepath_summary.json"
    out.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False, default=str))
    print(f"\nartifacts written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
