#!/usr/bin/env python3
"""Layer 1 — Lineage state-machine invariant suite (deterministic).

Tests the WRITE path's mechanism, NOT its model-gated intelligence. It drives
the store directly via add() + revise() + conflict_key routing -- no LLM, no
fact extraction, no similarity decisioning -- and asserts the append-only
lineage invariants after every operation. Because no model is involved, this
must pass 100% on any configuration; a failure here is a real engine bug, not
an extraction miss.

What it proves (the differentiator that the read benchmarks bypass):
  - supersession creates rev N+1 active + flips the predecessor to superseded
    with valid links, under both lineage-formation paths (explicit revise,
    implicit conflict_key);
  - exactly one active head per lineage, ever (no zero, no two);
  - the head pointer (memory_lineages.current_memory_id) tracks the active row;
  - get_lineage / get_history reconstruct a contiguous, link-valid timeline;
  - oscillation (A->B->A) makes a NEW row, never resurrects the old one;
  - unrelated lineages never collide;
  - concurrent revises on one lineage serialize to a single active head.

Embeddings run on-device (MiniLM, free); no API key or LLM is required. Needs a
live Postgres (docker compose up -d postgres). Exits non-zero on any failure so
it can gate CI.

    python3 benchmark/lineage_invariants.py
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parent.parent
load_dotenv(_REPO / ".env", override=False)

# Deterministic, free, no API: on-device embeddings. The LLM is never called.
os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("ENGRAM_EMBEDDING_DIMENSION", "384")

from engram import Engram  # noqa: E402
from engram.core.config import get_settings  # noqa: E402


# ===========================================================================
# Assertion harness
# ===========================================================================
class Checks:
    """Collects named pass/fail results; a scenario can register many."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, bool, str]] = []
        self._scenario = ""

    def scenario(self, name: str) -> None:
        self._scenario = name

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append((self._scenario, name, bool(passed), detail))

    def eq(self, name: str, got: Any, want: Any) -> None:
        self.check(name, got == want, f"got={got!r} want={want!r}")

    def report(self) -> bool:
        by_scn: dict[str, list[tuple[str, bool, str]]] = {}
        for scn, name, ok, detail in self.results:
            by_scn.setdefault(scn, []).append((name, ok, detail))
        n_pass = sum(1 for _, _, ok, _ in self.results if ok)
        n_total = len(self.results)
        print("\n" + "=" * 68)
        print("LINEAGE INVARIANT SUITE (Layer 1 — deterministic, no LLM)")
        print("=" * 68)
        for scn, items in by_scn.items():
            ok_n = sum(1 for _, ok, _ in items if ok)
            print(f"\n{scn}  ({ok_n}/{len(items)})")
            for name, ok, detail in items:
                mark = "✓" if ok else "✗"
                line = f"  {mark} {name}"
                if not ok and detail:
                    line += f"   [{detail}]"
                print(line)
        print("\n" + "-" * 68)
        verdict = "ALL INVARIANTS HOLD" if n_pass == n_total else "INVARIANT VIOLATION(S)"
        print(f"{verdict}: {n_pass}/{n_total} checks passed")
        return n_pass == n_total


# ===========================================================================
# Invariant helpers — assert on a fetched lineage
# ===========================================================================
def active_rows(memories: list[Any]) -> list[Any]:
    return [m for m in memories if m.status == "active"]


def superseded_rows(memories: list[Any]) -> list[Any]:
    return [m for m in memories if m.status == "superseded"]


async def assert_lineage_well_formed(
    c: Checks, engram: Engram, seed_id: str, *, expected_revisions: int, prefix: str
) -> Any:
    """Core invariants that must hold for ANY lineage after ANY operation."""
    lin = await engram.get_lineage(seed_id)
    mems = lin.memories
    active = active_rows(mems)
    superseded = superseded_rows(mems)

    c.eq(f"{prefix}: revision count", len(mems), expected_revisions)
    c.check(f"{prefix}: exactly one active head", len(active) == 1,
            f"active={len(active)}")
    c.eq(f"{prefix}: superseded count", len(superseded), expected_revisions - 1)

    if active:
        head = active[0]
        # Head pointer agrees with the lone active row.
        c.eq(f"{prefix}: head pointer == active row",
             lin.current_memory_id, head.memory_id)
        # get_current resolves to the same head from any seed in the lineage.
        cur = await engram.get_current(seed_id)
        c.eq(f"{prefix}: get_current == head", cur.memory_id, head.memory_id)
        # Active head is the max revision, and is NOT marked superseded.
        c.eq(f"{prefix}: head is max revision",
             head.revision, max(m.revision for m in mems))
        c.check(f"{prefix}: head has no superseded_by",
                head.superseded_by_memory_id is None,
                f"superseded_by={head.superseded_by_memory_id}")
        c.check(f"{prefix}: head has no valid_to",
                head.valid_to is None, f"valid_to={head.valid_to}")

    # Every superseded row has a valid forward link to a real row in the lineage.
    ids = {m.memory_id for m in mems}
    for m in superseded:
        c.check(f"{prefix}: superseded r{m.revision} links forward",
                m.superseded_by_memory_id in ids,
                f"superseded_by={m.superseded_by_memory_id} not in lineage")
        c.check(f"{prefix}: superseded r{m.revision} has valid_to",
                m.valid_to is not None, "valid_to is None")

    # No two rows claim to be superseded BY the same row's slot in a way that
    # would imply a fork: each superseded_by target should appear once as a head
    # of supersession (chain, not tree).
    targets = [m.superseded_by_memory_id for m in superseded]
    c.check(f"{prefix}: supersession is a chain (no forks)",
            len(targets) == len(set(targets)),
            f"duplicate supersede targets in {targets}")

    # Revisions are contiguous 1..N.
    revs = sorted(m.revision for m in mems)
    c.eq(f"{prefix}: revisions contiguous 1..N", revs, list(range(1, expected_revisions + 1)))
    return lin


async def count_rows(engram: Engram, agent: str, user: str) -> tuple[int, int]:
    """(active_count, total_incl_superseded) for an agent/user, via search."""
    active = await engram.search("the user fact", agent, user_id=user, limit=100,
                                 mode="hybrid", include_superseded=False)
    allm = await engram.search("the user fact", agent, user_id=user, limit=100,
                               mode="hybrid", include_superseded=True)
    return len(active), len(allm)


# ===========================================================================
# Scenarios
# ===========================================================================
async def scenario_explicit_revise_chain(c: Checks, engram: Engram, base: str) -> None:
    """add -> revise -> revise. The canonical explicit lineage."""
    c.scenario("S1 explicit revise chain (add->revise->revise)")
    agent, user = f"{base}-s1", "u"
    m1 = await engram.add("The user's role is junior engineer.", agent,
                          user_id=user, memory_type="semantic")
    await assert_lineage_well_formed(c, engram, m1.memory_id,
                                     expected_revisions=1, prefix="after add")
    m2 = await engram.revise(m1.memory_id, content="The user's role is senior engineer.",
                             reason="promotion")
    c.check("revise returns new active row r2", m2.status == "active" and m2.revision == 2,
            f"status={m2.status} rev={m2.revision}")
    c.eq("revise stays in same lineage", m2.lineage_id, m1.lineage_id)
    await assert_lineage_well_formed(c, engram, m1.memory_id,
                                     expected_revisions=2, prefix="after revise#1")
    await engram.revise(m1.memory_id, content="The user's role is staff engineer.",
                        reason="promotion-2")
    lin = await assert_lineage_well_formed(c, engram, m1.memory_id,
                                           expected_revisions=3, prefix="after revise#2")
    c.eq("final head value is the latest revision",
         (active_rows(lin.memories)[0].content if active_rows(lin.memories) else None),
         "The user's role is staff engineer.")
    a, t = await count_rows(engram, agent, user)
    c.check("count reconciles (1 active + 2 superseded = 3)", a == 1 and t == 3,
            f"active={a} total={t}")
    await engram.purge(agent, user)


async def scenario_conflict_key_supersession(c: Checks, engram: Engram, base: str) -> None:
    """Two add()s sharing a conflict_key: the second must supersede the first
    WITHOUT an explicit memory_id (implicit lineage formation)."""
    c.scenario("S2 implicit conflict_key supersession")
    agent, user = f"{base}-s2", "u"
    key = f"{agent}:{user}:city"
    a1 = await engram.add("The user lives in Berlin.", agent, user_id=user,
                          memory_type="profile", metadata={"conflict_key": key})
    a2 = await engram.add("The user lives in Amsterdam.", agent, user_id=user,
                          memory_type="profile", metadata={"conflict_key": key})
    c.eq("second add lands in the same lineage", a2.lineage_id, a1.lineage_id)
    c.check("second add is active r2", a2.status == "active" and a2.revision == 2,
            f"status={a2.status} rev={a2.revision}")
    lin = await assert_lineage_well_formed(c, engram, a1.memory_id,
                                           expected_revisions=2, prefix="after 2nd add")
    head = active_rows(lin.memories)[0]
    c.eq("head value is the newer city", head.content, "The user lives in Amsterdam.")
    await engram.purge(agent, user)


async def scenario_oscillation_no_resurrection(c: Checks, engram: Engram, base: str) -> None:
    """A -> B -> A: returning to the original VALUE must mint a NEW row, never
    flip the original superseded row back to active."""
    c.scenario("S3 oscillation A->B->A (no resurrection)")
    agent, user = f"{base}-s3", "u"
    m1 = await engram.add("The user's status is single.", agent, user_id=user)
    await engram.revise(m1.memory_id, content="The user's status is married.")
    await engram.revise(m1.memory_id, content="The user's status is single.")
    lin = await assert_lineage_well_formed(c, engram, m1.memory_id,
                                           expected_revisions=3, prefix="after A->B->A")
    head = active_rows(lin.memories)[0]
    c.eq("head VALUE returned to original", head.content, "The user's status is single.")
    c.check("head is a NEW row, not the resurrected original",
            head.memory_id != m1.memory_id, f"head id == original {m1.memory_id}")
    orig = next((m for m in lin.memories if m.memory_id == m1.memory_id), None)
    c.check("original row stays superseded (not resurrected)",
            orig is not None and orig.status == "superseded",
            f"original status={getattr(orig, 'status', None)}")
    await engram.purge(agent, user)


async def scenario_no_false_collision(c: Checks, engram: Engram, base: str) -> None:
    """Revising one lineage must not touch an unrelated lineage."""
    c.scenario("S4 unrelated lineages don't collide")
    agent, user = f"{base}-s4", "u"
    city = await engram.add("The user lives in Oslo.", agent, user_id=user,
                            memory_type="profile",
                            metadata={"conflict_key": f"{agent}:{user}:city"})
    job = await engram.add("The user works at Spotify.", agent, user_id=user,
                           memory_type="profile",
                           metadata={"conflict_key": f"{agent}:{user}:job"})
    c.check("two adds form two distinct lineages",
            city.lineage_id != job.lineage_id, "lineage_ids collided")
    # Revise the city lineage; the job lineage must be byte-for-byte untouched.
    await engram.revise(city.memory_id, content="The user lives in Bergen.")
    job_lin = await engram.get_lineage(job.memory_id)
    c.eq("job lineage still single revision", len(job_lin.memories), 1)
    c.eq("job lineage head unchanged", job_lin.current_memory_id, job.memory_id)
    c.check("job row still active", active_rows(job_lin.memories)[0].status == "active"
            if job_lin.memories else False)
    a, t = await count_rows(engram, agent, user)
    c.check("counts: 2 active + 1 superseded = 3", a == 2 and t == 3,
            f"active={a} total={t}")
    await engram.purge(agent, user)


async def scenario_history_fidelity(c: Checks, engram: Engram, base: str) -> None:
    """get_history must reconstruct added/revised/superseded with valid links."""
    c.scenario("S5 history timeline fidelity")
    agent, user = f"{base}-s5", "u"
    m1 = await engram.add("The user's plan is the free tier.", agent, user_id=user)
    m2 = await engram.revise(m1.memory_id, content="The user's plan is the pro tier.")
    events = await engram.get_history(agent, user_id=user, limit=100,
                                      include_superseded=True)
    types = [e.event_type for e in events]
    c.check("has an 'added' event", "added" in types, f"types={types}")
    c.check("has a 'revised' event", "revised" in types, f"types={types}")
    c.check("has a 'superseded' event", "superseded" in types, f"types={types}")
    ids = {m1.memory_id, m2.memory_id}
    revised = next((e for e in events if e.event_type == "revised"), None)
    c.check("revised event links to predecessor",
            revised is not None and revised.previous_memory_id == m1.memory_id,
            f"prev={getattr(revised, 'previous_memory_id', None)} want={m1.memory_id}")
    superseded = next((e for e in events if e.event_type == "superseded"), None)
    c.check("superseded event links to successor",
            superseded is not None and superseded.superseded_by_memory_id == m2.memory_id,
            f"by={getattr(superseded, 'superseded_by_memory_id', None)} want={m2.memory_id}")
    c.check("all event memories belong to the lineage",
            all(e.memory.memory_id in ids for e in events),
            f"stray ids in {[e.memory.memory_id for e in events]}")
    await engram.purge(agent, user)


async def scenario_concurrent_revise(c: Checks, engram: Engram, base: str) -> None:
    """Fire N concurrent revises on one lineage. Whatever the interleaving, the
    end state must have exactly ONE active head and a consistent chain (the
    advisory-lock / CAS guarantee against a double-active fork)."""
    c.scenario("S6 concurrent revise -> single active head")
    agent, user = f"{base}-s6", "u"
    m1 = await engram.add("The user's score is 0.", agent, user_id=user)
    results = await asyncio.gather(
        engram.revise(m1.memory_id, content="The user's score is 1."),
        engram.revise(m1.memory_id, content="The user's score is 2."),
        engram.revise(m1.memory_id, content="The user's score is 3."),
        return_exceptions=True,
    )
    n_ok = sum(1 for r in results if not isinstance(r, Exception))
    c.check("at least one concurrent revise succeeded", n_ok >= 1, f"ok={n_ok}")
    lin = await engram.get_lineage(m1.memory_id)
    active = active_rows(lin.memories)
    c.check("exactly one active head after concurrency", len(active) == 1,
            f"active={len(active)} (FORK = double-active bug)")
    a, _t = await count_rows(engram, agent, user)
    c.eq("counts reconcile (1 active + rest superseded)", a, 1)
    c.check("no orphaned/duplicate head pointer",
            lin.current_memory_id == (active[0].memory_id if active else None),
            f"head={lin.current_memory_id}")
    await engram.purge(agent, user)


# ===========================================================================
# Runner
# ===========================================================================
async def main() -> None:
    settings = get_settings().model_copy(update={
        "embedding_provider": os.environ["ENGRAM_EMBEDDING_PROVIDER"],
        "embedding_model": os.environ["ENGRAM_EMBEDDING_MODEL"],
        "embedding_dimension": int(os.environ["ENGRAM_EMBEDDING_DIMENSION"]),
        "allow_embedding_dimension_change": True,
        "near_duplicate_threshold": 1.0,  # never collapse — we assert exact rows
    })
    engram = Engram(settings=settings, memory_policy="default")
    await engram.connect()
    print(f"Embedding : {settings.embedding_provider} / {settings.embedding_model} "
          f"({settings.embedding_dimension}d)   |   LLM: not used")

    base = f"lineage-inv-{uuid.uuid4().hex[:8]}"
    c = Checks()
    scenarios = [
        scenario_explicit_revise_chain,
        scenario_conflict_key_supersession,
        scenario_oscillation_no_resurrection,
        scenario_no_false_collision,
        scenario_history_fidelity,
        scenario_concurrent_revise,
    ]
    try:
        for scn in scenarios:
            try:
                await scn(c, engram, base)
            except Exception as exc:  # a thrown error IS a failure, not a crash
                c.check(f"{scn.__name__} raised", False, f"{type(exc).__name__}: {exc}")
    finally:
        # Best-effort cleanup of any namespace a scenario left behind on error.
        for i in range(1, len(scenarios) + 1):
            with contextlib.suppress(Exception):
                await engram.purge(f"{base}-s{i}", "u")
        await engram.close()

    ok = c.report()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
