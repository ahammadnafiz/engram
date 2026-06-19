"""The memory operator: classify a question, route to the right recall
surface(s), and compose a source-backed answer.

This is the high-level "ask my memory anything" layer. It requires a
configured LLM (used to classify intent and compose the final answer); the
underlying recall surfaces (``search``, ``search_events``, ``explain_memory``,
``get_lineage``) remain usable without one.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from engram.core.exceptions import ConfigurationError
from engram.recall.models import RecallAnswer, RecallIntent, RecallSource
from engram.recall.temporal import resolve_timeframe

if TYPE_CHECKING:
    from datetime import datetime

    from engram.client import Engram
    from engram.core._types import AgentId, UserId
    from engram.memory.models import Memory

_VALID_INTENTS = {"current", "historical", "event", "lineage", "temporal_chain", "chat"}

_CLASSIFY_SYSTEM = """You route the CURRENT USER MESSAGE for a personal-memory assistant.
Return ONLY a JSON object, no prose, no code fences:
{"intent": "current|historical|event|lineage|temporal_chain|chat", "topic": "<keywords>", "when": "<temporal phrase or empty>", "anchors": []}

intents:
- current: what is true now ("when is my meeting", "where do I live now")
- historical: what changed or the old value ("what was my meeting before I changed it", "what did I update")
- event: what was said/asked in the conversation ("what did I ask yesterday", "what prompt did I use")
- lineage: the full timeline of one topic ("show the history of my meeting time")
- temporal_chain: an interval or gap between TWO distinct events at different points in time
  ("how many days between X and Y", "how long after X did Y happen", "how much time passed between X and Y")
- chat: not a memory recall request; the user is sharing new facts, correcting
  facts, asking a general non-memory question, or giving an instruction

critical routing rules:
- If the user is telling you new information, giving a correction, or asking
  you to remember something, return chat. Do not search memory for a fact that
  the user just authored in this message.
- If the user is asking what you know/remember about them, their profile,
  schedule, preferences, history, prior messages, or saved facts, return one of
  current/historical/event/lineage/temporal_chain.
- A sentence can mention a memory topic without being a recall request.
  Declarative first-person updates are chat; questions about saved values are
  recall.
- A question asking for a stored value, owner, reason, count, or list is recall
  even when it uses "we", "our", "the team", or a project name instead of "my",
  and even when a context clause comes before the question. Route these to
  current (or historical/lineage if they ask about an old or changed value).
- event is only for what was literally said or asked in the conversation. A
  question about a stored fact (which database, who is on the team, what was
  bought) is current, not event, even if the fact came up earlier.
- temporal_chain requires TWO separable events. "How long did my sprint last?" is
  current (one event), but "how many days between starting project X and completing it?"
  is temporal_chain (two events).

anchors field:
- For temporal_chain: exactly 2 short search phrases, one per event anchor.
  Example: "how many days after I started soccer practice did I reserve the court?"
  → anchors: ["started soccer practice", "reserved the court"]
- For all other intents: anchors must be [] (empty array).

examples:
- "i have a meeting at 3pm in zoom" -> {"intent":"chat","topic":"meeting","when":"","anchors":[]}
- "remember my meeting is at 3pm in zoom" -> {"intent":"chat","topic":"meeting","when":"","anchors":[]}
- "actually my meeting moved to 10pm" -> {"intent":"chat","topic":"meeting time","when":"","anchors":[]}
- "i live in dhaka and study cse at uiu" -> {"intent":"chat","topic":"location education","when":"","anchors":[]}
- "what is my name" -> {"intent":"current","topic":"name","when":"","anchors":[]}
- "where do i live" -> {"intent":"current","topic":"location","when":"","anchors":[]}
- "what time is my meeting" -> {"intent":"current","topic":"meeting time","when":"","anchors":[]}
- "what latency threshold must we hit before we launch?" -> {"intent":"current","topic":"latency threshold launch","when":"","anchors":[]}
- "who is managing the launch pipeline?" -> {"intent":"current","topic":"launch pipeline manager","when":"","anchors":[]}
- "which database did we pick and why?" -> {"intent":"current","topic":"database choice reason","when":"","anchors":[]}
- "what was my meeting time before i changed it" -> {"intent":"historical","topic":"meeting time","when":"","anchors":[]}
- "what did i ask yesterday about the chatbot" -> {"intent":"event","topic":"chatbot","when":"yesterday","anchors":[]}
- "show the history of my meeting time" -> {"intent":"lineage","topic":"meeting time","when":"","anchors":[]}
- "how many days between starting my ML project and submitting it?" -> {"intent":"temporal_chain","topic":"ML project start submission","when":"","anchors":["started ML project","submitted ML project"]}
- "how long after I set my sprint deadline did I adjust it?" -> {"intent":"temporal_chain","topic":"sprint deadline","when":"","anchors":["set sprint deadline","adjusted sprint deadline"]}
- "explain postgres indexing" -> {"intent":"chat","topic":"postgres indexing","when":"","anchors":[]}

topic: the subject to search for, reduced to keywords (e.g. "meeting time", "current city").
when: a temporal phrase only if the question contains one (e.g. "yesterday", "last week"), else "".
anchors: for temporal_chain only — exactly 2 short search phrases for the two events."""

_COMPOSE_SYSTEM = """Answer the user's question using only the memory evidence provided.

Treat the evidence as authoritative and answer it directly:
- If the answer is present in the evidence, state it plainly and confidently.
  Do not say you lack memory, and do not hedge ("I don't have an entry that
  explicitly says..."), when the evidence already contains the fact. A planned,
  booked, or recorded detail in the evidence is the answer to a question about it.
- Say you have no memory of it only when nothing in the evidence is relevant.
  Never invent details that are absent.
- CURRENT is the value now; PREVIOUS values are superseded history. State the
  current value, and mention a previous value only to show what changed.
- Answer every part of the question. For list, "everything", or "across all"
  questions, include every matching item in the evidence, not just the first.
Be concise and factual."""


def _parse_classification(raw: str) -> tuple[RecallIntent, str, str, list[str]]:
    """Parse the classifier's JSON; fall back to current-intent on any error."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        intent = str(data.get("intent", "current")).lower()
        if intent not in _VALID_INTENTS:
            intent = "current"
        topic = str(data.get("topic") or "").strip()
        when = str(data.get("when") or "").strip()
        raw_anchors = data.get("anchors") or []
        anchors = [str(a).strip() for a in raw_anchors if a][:2]
        # temporal_chain without two anchors degrades to current-intent search
        if intent == "temporal_chain" and len(anchors) < 2:
            intent = "current"
            anchors = []
        return intent, topic, when, anchors  # type: ignore[return-value]
    except (json.JSONDecodeError, ValueError, AttributeError):
        return "current", "", "", []


def _split_lineage(members: list[Memory]) -> tuple[Memory | None, list[Memory]]:
    """Split lineage revisions into (active head, superseded newest-first)."""
    ordered = sorted(members, key=lambda m: m.revision)
    current = next((m for m in ordered if m.status == "active"), None)
    previous = [m for m in reversed(ordered) if m.status == "superseded"]
    return current, previous


def _source(memory: Memory) -> RecallSource:
    return RecallSource(
        memory_id=memory.memory_id,
        session_id=memory.session_id,
        created_at=memory.created_at,
        status=memory.status,
        source=str(memory.metadata.get("source"))
        if memory.metadata.get("source")
        else None,
    )


def _evidence_block(answer: RecallAnswer) -> str:
    """Render retrieved evidence into a compact prompt block."""
    lines: list[str] = []
    seen_memory_ids: set[str] = set()
    if answer.current is not None:
        lines.append(f"CURRENT: {answer.current.fact or answer.current.content}")
        seen_memory_ids.add(answer.current.memory_id)
    for mem in answer.previous:
        when = mem.superseded_at or mem.valid_to or mem.created_at
        stamp = when.date().isoformat() if when else "unknown"
        lines.append(f"PREVIOUS (until {stamp}): {mem.fact or mem.content}")
        seen_memory_ids.add(mem.memory_id)
    for ev in answer.events:
        stamp = ev.created_at.date().isoformat() if ev.created_at else "unknown"
        lines.append(f"EVENT [{ev.role} {stamp}]: {ev.content}")
    for res in answer.evidence:
        if res.memory_id in seen_memory_ids:
            continue
        lines.append(f"MEMORY: {res.fact or res.content}")
        seen_memory_ids.add(res.memory_id)
    return "\n".join(lines) if lines else "(no matching memory)"


async def recall(
    engram: Engram,
    question: str,
    agent_id: AgentId,
    *,
    user_id: UserId | None = None,
    question_date: datetime | None = None,
    limit: int = 10,
    compose_answer: bool = True,
) -> RecallAnswer:
    """Classify ``question``, route to the right surface(s), and compose an answer.

    Raises:
        ConfigurationError: If no LLM is configured.
        ValueError: If ``question`` is empty.
    """
    if not question.strip():
        raise ValueError("recall question must not be empty")
    if engram.llm is None:
        raise ConfigurationError(
            "recall() requires a configured LLM. Use search()/search_events()/"
            "explain_memory() for LLM-free recall surfaces."
        )

    # Step 1: classify intent + extract topic, temporal phrase, and event anchors.
    raw = await engram.llm.complete(question, system=_CLASSIFY_SYSTEM, temperature=0)
    intent, topic, when_phrase, anchors = _parse_classification(raw)
    topic = topic or question
    since, until = resolve_timeframe(when_phrase, base=question_date)

    trace: dict[str, Any] = {
        "intent": intent,
        "topic": topic,
        "when_phrase": when_phrase,
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "anchors": anchors,
        "raw_classification": raw,
    }

    current: Memory | None = None
    previous: list[Memory] = []
    when_changed: datetime | None = None
    sources: list[RecallSource] = []
    evidence: list[Memory] = []
    events = []
    conflict_note: str | None = None

    # Step 2: route to the matching surface(s).
    if intent == "chat":
        return RecallAnswer(answer_text="", intent=intent, trace=trace)

    if intent == "temporal_chain":
        # Two parallel searches — one per event anchor — then merge by date.
        # This solves two-hop temporal questions where both events need to be in
        # evidence simultaneously (e.g. "how many days between starting X and Y?").
        tasks = [
            engram.search(anchor, agent_id, user_id=user_id, limit=limit)
            for anchor in anchors
        ]
        results_per_anchor = await asyncio.gather(*tasks, return_exceptions=True)
        seen_ids: set[str] = set()
        for result in results_per_anchor:
            if isinstance(result, Exception):
                continue
            for r in result:
                if r.memory.memory_id not in seen_ids:
                    seen_ids.add(r.memory.memory_id)
                    evidence.append(r.memory)
        # Sort chronologically so the composer can compute the interval directly.
        evidence.sort(key=lambda m: m.created_at or m.valid_from or m.updated_at)  # type: ignore[arg-type]
        current = evidence[0] if evidence else None
        sources = [_source(m) for m in evidence]

    elif intent == "event":
        events = await engram.search_events(
            topic,
            agent_id=agent_id,
            user_id=user_id,
            since=since,
            until=until,
            limit=limit,
        )
        sources = [
            RecallSource(
                event_id=e.event_id,
                session_id=e.session_id,
                created_at=e.created_at,
                source="event",
            )
            for e in events
        ]
    elif intent == "lineage":
        results = await engram.search(topic, agent_id, user_id=user_id, limit=1)
        if results:
            lineage = await engram.get_lineage(results[0].memory.memory_id)
            current, previous = _split_lineage(lineage.memories)
            evidence = sorted(lineage.memories, key=lambda m: m.revision)
            sources = [_source(m) for m in evidence]
            if current and current.valid_from:
                when_changed = current.valid_from
    else:  # current or historical
        include_superseded = intent == "historical"
        results = await engram.search(
            topic,
            agent_id,
            user_id=user_id,
            limit=limit,
            include_superseded=include_superseded,
        )
        evidence = [r.memory for r in results]
        active = [m for m in evidence if m.status == "active"]
        current = active[0] if active else (evidence[0] if evidence else None)
        sources = [_source(m) for m in evidence]

        # Historical: pull the FULL lineage so every prior value is available,
        # not just the direct supersede edge (which can be an uninformative
        # intermediate revision).
        if intent == "historical" and current is not None:
            lineage = await engram.get_lineage(current.memory_id)
            head, previous = _split_lineage(lineage.memories)
            current = head or current
            if current.valid_from:
                when_changed = current.valid_from
            sources = [_source(current), *[_source(m) for m in previous]]

        # Conflict: two distinct active facts in the same lineage slot.
        by_lineage = [m for m in active if m.lineage_id]
        seen: dict[str, str] = {}
        for m in by_lineage:
            assert m.lineage_id is not None
            prior = seen.get(m.lineage_id)
            if prior is not None and prior != (m.fact or m.content):
                conflict_note = (
                    "Multiple active facts conflict for the same topic; "
                    "the most relevant is reported as current."
                )
                break
            seen[m.lineage_id] = m.fact or m.content

    partial = RecallAnswer(
        answer_text="",
        intent=intent,
        current=current,
        previous=previous,
        when_changed=when_changed,
        sources=sources,
        conflict_note=conflict_note,
        evidence=evidence,
        events=events,
        trace=trace,
    )

    if not compose_answer:
        return partial

    # Step 3: compose the prose answer grounded in retrieved evidence.
    compose_prompt = f"Question: {question}\n\nEvidence:\n{_evidence_block(partial)}"
    answer_text = await engram.llm.complete(
        compose_prompt, system=_COMPOSE_SYSTEM, temperature=0
    )

    return partial.model_copy(update={"answer_text": answer_text.strip()})
