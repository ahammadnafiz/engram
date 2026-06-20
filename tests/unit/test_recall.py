"""Unit tests for the memory recall operator and temporal resolver."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.core.exceptions import ConfigurationError
from engram.memory.models import (
    Memory,
    MemoryLineage,
    SearchResult,
)
from engram.recall.operator import _CLASSIFY_SYSTEM, _parse_classification, recall
from engram.recall.temporal import resolve_timeframe
from engram.task.models import AgentEvent

BASE = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


def _mem(
    fact: str, *, status: str = "active", mem_id: str = "mem_1", **kw: object
) -> Memory:
    return Memory(
        memory_id=mem_id,
        agent_id="agent",
        user_id="user",
        content=fact,
        fact=fact,
        status=status,
        **kw,  # type: ignore[arg-type]
    )


def _engram(classify_json: str, prose: str) -> MagicMock:
    eg = MagicMock()
    eg._ensure_connected = MagicMock()
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=[classify_json, prose])
    eg.llm = llm
    eg.search = AsyncMock(return_value=[])
    eg.search_events = AsyncMock(return_value=[])
    eg.get_lineage = AsyncMock()
    eg.explain_memory = AsyncMock()
    return eg


class TestParseClassification:
    def test_plain_json(self) -> None:
        intent, topic, when, anchors = _parse_classification(
            '{"intent": "historical", "topic": "meeting", "when": "yesterday", "anchors": []}'
        )
        assert (intent, topic, when) == ("historical", "meeting", "yesterday")
        assert anchors == []

    def test_code_fenced_json(self) -> None:
        intent, topic, _, _anchors = _parse_classification(
            '```json\n{"intent": "event", "topic": "chatbot", "when": "", "anchors": []}\n```'
        )
        assert intent == "event"
        assert topic == "chatbot"

    def test_malformed_falls_back_to_current(self) -> None:
        assert _parse_classification("not json at all") == ("current", "", "", [])

    def test_invalid_intent_coerced_to_current(self) -> None:
        intent, _, _, _anchors = _parse_classification(
            '{"intent": "banana", "topic": "x"}'
        )
        assert intent == "current"

    def test_chat_intent_is_supported(self) -> None:
        intent, topic, when, anchors = _parse_classification(
            '{"intent": "chat", "topic": "", "when": "", "anchors": []}'
        )
        assert (intent, topic, when) == ("chat", "", "")
        assert anchors == []

    def test_temporal_chain_extracts_anchors(self) -> None:
        intent, _topic, _when, anchors = _parse_classification(
            '{"intent": "temporal_chain", "topic": "project start finish", "when": "", '
            '"anchors": ["started ML project", "submitted ML project"]}'
        )
        assert intent == "temporal_chain"
        assert anchors == ["started ML project", "submitted ML project"]

    def test_temporal_chain_without_two_anchors_degrades(self) -> None:
        intent, _, _, anchors = _parse_classification(
            '{"intent": "temporal_chain", "topic": "project", "when": "", "anchors": ["only one"]}'
        )
        assert intent == "current"
        assert anchors == []

    def test_classifier_prompt_has_fact_vs_recall_counterexamples(self) -> None:
        """Regression guard for declarative facts being routed as recall.

        The model needs explicit counterexamples because "I have a meeting" and
        "what time is my meeting" share the same topic but require different
        chatbot paths.
        """
        assert '"i have a meeting at 3pm in zoom" -> {"intent":"chat"' in (
            _CLASSIFY_SYSTEM
        )
        assert '"what is my name" -> {"intent":"current"' in _CLASSIFY_SYSTEM
        assert '"what time is my meeting" -> {"intent":"current"' in _CLASSIFY_SYSTEM


class TestResolveTimeframe:
    def test_yesterday_is_full_day(self) -> None:
        pytest.importorskip("dateparser")
        since, until = resolve_timeframe("yesterday", base=BASE)
        assert since == datetime(2026, 6, 14, tzinfo=timezone.utc)
        assert until is not None and until.date() == since.date()

    @pytest.mark.parametrize("phrase", ["", "   ", "zzzz nonsense"])
    def test_empty_or_unparseable_returns_none(self, phrase: str) -> None:
        assert resolve_timeframe(phrase, base=BASE) == (None, None)


class TestRecallRouting:
    @pytest.mark.asyncio
    async def test_current_intent_returns_active_fact(self) -> None:
        eg = _engram(
            '{"intent": "current", "topic": "meeting"}', "Your meeting is at 10 PM."
        )
        eg.search = AsyncMock(
            return_value=[
                SearchResult(memory=_mem("meeting is at 10 PM"), score=0.9),
                SearchResult(
                    memory=_mem("meeting is in room 4", mem_id="mem_2"),
                    score=0.8,
                ),
            ]
        )

        answer = await recall(eg, "when is my meeting?", "agent", user_id="user")

        assert answer.intent == "current"
        assert answer.current is not None and "10 PM" in answer.current.fact  # type: ignore[operator]
        assert answer.answer_text == "Your meeting is at 10 PM."
        compose_prompt = eg.llm.complete.await_args_list[1].args[0]
        assert "CURRENT: meeting is at 10 PM" in compose_prompt
        assert "MEMORY: meeting is in room 4" in compose_prompt
        # current intent must NOT request superseded rows.
        assert eg.search.await_args.kwargs["include_superseded"] is False

    @pytest.mark.asyncio
    async def test_historical_intent_returns_full_lineage_previous(self) -> None:
        """Historical recall must surface ALL prior values from the lineage,
        not just the direct supersede edge (regression: an uninformative
        intermediate revision hid the original value)."""
        eg = _engram(
            '{"intent": "historical", "topic": "manager"}',
            "Your manager was Alice before Bob.",
        )
        alice = _mem("manager is Alice", status="superseded", mem_id="m1", revision=1)
        mid = _mem(
            "manager changed to Bob", status="superseded", mem_id="m2", revision=2
        )
        bob = _mem("manager is Bob", mem_id="m3", revision=3, valid_from=BASE)
        eg.search = AsyncMock(return_value=[SearchResult(memory=bob, score=0.9)])
        eg.get_lineage = AsyncMock(
            return_value=MemoryLineage(
                lineage_id="lin_1", current_memory_id="m3", memories=[alice, mid, bob]
            )
        )

        answer = await recall(eg, "who was my manager before?", "agent", user_id="user")

        assert answer.intent == "historical"
        assert eg.search.await_args.kwargs["include_superseded"] is True
        prev_ids = {m.memory_id for m in answer.previous}
        assert prev_ids == {"m1", "m2"}  # the original value (Alice) is present
        assert answer.when_changed == BASE

    @pytest.mark.asyncio
    async def test_event_intent_searches_ledger_with_timeframe(self) -> None:
        pytest.importorskip("dateparser")
        eg = _engram(
            '{"intent": "event", "topic": "chatbot", "when": "yesterday"}',
            "You asked about making memory jobs automatic.",
        )
        ev = AgentEvent(
            event_id="evt_1",
            agent_id="agent",
            role="user",
            event_type="user_message",
            content="how do I make memory jobs automatic?",
        )
        eg.search_events = AsyncMock(return_value=[ev])

        answer = await recall(
            eg,
            "what did I ask yesterday about the chatbot?",
            "agent",
            user_id="user",
            question_date=BASE,
        )

        assert answer.intent == "event"
        assert [e.event_id for e in answer.events] == ["evt_1"]
        # temporal phrase resolved and passed as a bounded window.
        assert eg.search_events.await_args.kwargs["since"] is not None
        assert answer.sources[0].event_id == "evt_1"

    @pytest.mark.asyncio
    async def test_lineage_intent_returns_timeline(self) -> None:
        eg = _engram('{"intent": "lineage", "topic": "meeting"}', "3 PM -> 10 PM.")
        old = _mem("meeting is at 3 PM", status="superseded", mem_id="m1", revision=1)
        new = _mem("meeting is at 10 PM", mem_id="m2", revision=2, valid_from=BASE)
        eg.search = AsyncMock(return_value=[SearchResult(memory=new, score=0.9)])
        eg.get_lineage = AsyncMock(
            return_value=MemoryLineage(
                lineage_id="lin_1", current_memory_id="m2", memories=[old, new]
            )
        )

        answer = await recall(
            eg, "show my meeting time history", "agent", user_id="user"
        )

        assert answer.intent == "lineage"
        assert answer.current is not None and answer.current.memory_id == "m2"
        assert [m.memory_id for m in answer.previous] == ["m1"]

    @pytest.mark.asyncio
    async def test_chat_intent_does_not_compose_no_memory_answer(self) -> None:
        eg = _engram('{"intent": "chat", "topic": "", "when": ""}', "unused")

        answer = await recall(
            eg,
            "i have a meeting at 3pm in zoom",
            "agent",
            user_id="user",
        )

        assert answer.intent == "chat"
        assert answer.answer_text == ""
        eg.search.assert_not_awaited()
        eg.search_events.assert_not_awaited()
        assert eg.llm.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_compose_answer_false_returns_evidence_without_final_llm(
        self,
    ) -> None:
        eg = _engram(
            '{"intent": "current", "topic": "meeting"}',
            "should not be used",
        )
        eg.search = AsyncMock(
            return_value=[SearchResult(memory=_mem("meeting is at 3 PM"), score=0.9)]
        )

        answer = await recall(
            eg,
            "what time is my meeting?",
            "agent",
            user_id="user",
            compose_answer=False,
        )

        assert answer.intent == "current"
        assert answer.answer_text == ""
        assert answer.current is not None
        assert answer.current.fact == "meeting is at 3 PM"
        assert eg.llm.complete.await_count == 1


class TestRecallGuards:
    @pytest.mark.asyncio
    async def test_no_llm_raises_configuration_error(self) -> None:
        eg = MagicMock()
        eg._ensure_connected = MagicMock()
        eg.llm = None

        with pytest.raises(ConfigurationError, match="requires a configured LLM"):
            await recall(eg, "when is my meeting?", "agent")

    @pytest.mark.asyncio
    async def test_empty_question_raises(self) -> None:
        eg = _engram('{"intent": "current"}', "x")

        with pytest.raises(ValueError, match="must not be empty"):
            await recall(eg, "   ", "agent")
