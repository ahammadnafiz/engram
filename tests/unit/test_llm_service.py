"""Unit tests for LLMService summary helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def make_service(reply: str, finish_reason: str = "stop"):
    from engram.llm.service import LLMService
    from engram.providers.llm.protocol import LLMResponse

    provider = MagicMock()
    provider.model = "test-model"
    provider.complete_text = AsyncMock(return_value=reply)
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            content=reply, model="test-model", finish_reason=finish_reason
        )
    )
    return LLMService(provider=provider)


class TestUpdateConversationSummary:
    """Tests for LLMService.update_conversation_summary()."""

    @pytest.mark.asyncio
    async def test_first_summary_strips_output(self) -> None:
        svc = make_service("  User likes pizza.  ")
        out = await svc.update_conversation_summary(None, "I like pizza", "Nice!")
        assert out == "User likes pizza."
        svc._provider.complete_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_first_summary_prompt_is_write_mode(self) -> None:
        svc = make_service("summary")
        await svc.update_conversation_summary(None, "hi", "hello")
        prompt = svc._provider.complete_text.call_args.kwargs["prompt"]
        assert "Write a summary" in prompt
        assert "<none/>" in prompt

    @pytest.mark.asyncio
    async def test_update_includes_previous_summary(self) -> None:
        svc = make_service("Updated.")
        await svc.update_conversation_summary("Old summary", "New msg", "Reply")
        prompt = svc._provider.complete_text.call_args.kwargs["prompt"]
        assert "Old summary" in prompt
        assert "Update the existing summary" in prompt


class TestProcessForMemoryPerFact:
    """Tests for per-fact dedup retrieval in process_for_memory()."""

    @pytest.mark.asyncio
    async def test_retrieve_for_fact_called_per_fact(self) -> None:
        from engram.llm.service import LLMService, MemoryOperationType
        from engram.providers.llm.protocol import LLMResponse

        provider = MagicMock()
        provider.model = "test-model"
        # Extraction returns two facts; with empty candidates each evaluates to ADD
        # without a further LLM call.
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="Fact one here\nFact two here",
                model="test-model",
                finish_reason="stop",
            )
        )
        svc = LLMService(provider=provider)

        seen: list[str] = []

        async def retrieve(fact: str) -> list[tuple[str, str, float]]:
            seen.append(fact)
            return []

        result = await svc.process_for_memory(
            "msg", "resp", [], retrieve_for_fact=retrieve
        )

        assert seen == ["Fact one here", "Fact two here"]
        assert len(result.operations) == 2
        assert all(op.operation == MemoryOperationType.ADD for op in result.operations)


class TestTruncationHandling:
    """Truncated LLM output must not store partial facts silently."""

    @pytest.mark.asyncio
    async def test_truncated_extraction_drops_last_fact(self) -> None:
        svc = make_service(
            "User works at AskTuring\nUser lives in Dhaka\nUser is building an AI mem",
            finish_reason="length",
        )
        facts = await svc.extract_facts("msg", "resp")

        assert facts == ["User works at AskTuring", "User lives in Dhaka"]

    @pytest.mark.asyncio
    async def test_anthropic_max_tokens_reason_also_detected(self) -> None:
        svc = make_service(
            "User works at AskTuring\nUser lives in Dh",
            finish_reason="max_tokens",
        )
        facts = await svc.extract_facts("msg", "resp")

        assert facts == ["User works at AskTuring"]

    @pytest.mark.asyncio
    async def test_complete_extraction_keeps_all_facts(self) -> None:
        svc = make_service(
            "User works at AskTuring\nUser lives in Dhaka",
            finish_reason="stop",
        )
        facts = await svc.extract_facts("msg", "resp")

        assert facts == ["User works at AskTuring", "User lives in Dhaka"]

    @pytest.mark.asyncio
    async def test_extraction_prompt_allows_fictional_reference_codes(self) -> None:
        svc = make_service("NONE")
        await svc.extract_facts(
            "fictional: the recovery hint ends with 47-Kilo",
            "I will remember the fictional hint.",
        )

        prompt = svc._provider.complete.call_args.args[0][0]["content"]
        assert "fictional/test reference codes" in prompt
        assert "Do not extract real passwords" in prompt
        assert "47-Kilo" in prompt


class TestDuplicateScoreSpace:
    """The 0.92 duplicate threshold must compare cosine similarity, not the
    hybrid combined score (which decays below 0.92 within hours)."""

    @pytest.mark.asyncio
    async def test_stale_exact_duplicate_detected_via_semantic_score(self) -> None:
        from engram.llm.service import MemoryOperationType

        # Single LLM call: extraction. If dup detection fails, a second call
        # (evaluate_memory_operation, via complete_text) would happen.
        svc = make_service("User works at AskTuring")

        async def retrieve(fact: str):
            # Memory last accessed a day ago: combined hybrid score sags to
            # 0.80, but cosine similarity is still 0.99.
            return [("m1", "User works at AskTuring", 0.80, 0.99)]

        result = await svc.process_for_memory(
            "I work at AskTuring", "Nice!", [], retrieve_for_fact=retrieve
        )

        assert len(result.operations) == 1
        assert result.operations[0].operation == MemoryOperationType.NOOP
        svc._provider.complete_text.assert_not_awaited()  # no evaluate call

    @pytest.mark.asyncio
    async def test_three_tuples_still_supported(self) -> None:
        from engram.llm.service import MemoryOperationType

        svc = make_service("User works at AskTuring")

        async def retrieve(fact: str):
            return [("m1", "User works at AskTuring", 0.95)]

        result = await svc.process_for_memory(
            "I work at AskTuring", "Nice!", [], retrieve_for_fact=retrieve
        )

        assert result.operations[0].operation == MemoryOperationType.NOOP


class TestEvaluateMemoryOperation:
    """Tests for evaluate_memory_operation() target resolution."""

    @pytest.mark.asyncio
    async def test_update_with_unresolvable_target_falls_back_to_add(self) -> None:
        """An UPDATE/DELETE whose TARGET can't be resolved must not drop the
        fact: the safe fallback is ADD (per the prompt's own default rule)."""
        from engram.llm.service import MemoryOperationType

        svc = make_service(
            "OPERATION: UPDATE\nTARGET: 1 or 2\nMERGED: merged text\nREASON: unsure"
        )
        op = await svc.evaluate_memory_operation(
            "User moved to Berlin",
            [("mem_a", "User lives in Dhaka"), ("mem_b", "User likes tea")],
        )

        # "1 or 2" parses to digits "12" -> index 11 -> out of range -> no target
        assert op.operation == MemoryOperationType.ADD
        assert op.content == "User moved to Berlin"


    @pytest.mark.asyncio
    async def test_delete_with_no_target_falls_back_to_add(self) -> None:
        from engram.llm.service import MemoryOperationType

        svc = make_service(
            "OPERATION: DELETE\nTARGET: none\nMERGED: none\nREASON: contradiction"
        )
        op = await svc.evaluate_memory_operation(
            "User switched to BRAC Bank",
            [("mem_a", "User banks at City Bank")],
        )

        assert op.operation == MemoryOperationType.ADD
        assert op.content == "User switched to BRAC Bank"

    @pytest.mark.asyncio
    async def test_update_with_valid_target_stays_update(self) -> None:
        from engram.llm.service import MemoryOperationType

        svc = make_service(
            "OPERATION: UPDATE\nTARGET: 2\nMERGED: merged fact text\nREASON: detail"
        )
        op = await svc.evaluate_memory_operation(
            "Nadia lives in Toronto",
            [("mem_a", "irrelevant"), ("mem_b", "User's sister is Nadia")],
        )

        assert op.operation == MemoryOperationType.UPDATE
        assert op.target_id == "mem_b"
        assert op.content == "merged fact text"


class TestQueryExpansionPrompt:
    @pytest.mark.asyncio
    async def test_expand_query_prompt_covers_indirect_constraints(self) -> None:
        svc = make_service("food allergies\ncancelled plans\nschedule boundaries")
        await svc.expand_query("plan dinner and coffee")

        prompt = svc._provider.complete_text.call_args.kwargs["prompt"]
        assert "food queries should search for allergies" in prompt
        assert "cancelled, superseded, replaced" in prompt


class TestClassifyFacts:
    """Tests for LLMService.classify_facts() and classify_types wiring."""

    @pytest.mark.asyncio
    async def test_parses_types(self) -> None:
        svc = make_service("1: semantic\n2: episodic\n3: procedural")
        out = await svc.classify_facts(["a", "b", "c"])
        assert out == ["semantic", "episodic", "procedural"]

    @pytest.mark.asyncio
    async def test_defaults_unparseable_to_semantic(self) -> None:
        svc = make_service("nonsense output")
        assert await svc.classify_facts(["a", "b"]) == ["semantic", "semantic"]

    @pytest.mark.asyncio
    async def test_empty(self) -> None:
        svc = make_service("")
        assert await svc.classify_facts([]) == []

    @pytest.mark.asyncio
    async def test_compound_type_resolves_deterministically(self) -> None:
        """ "task/decision" contains two valid types; resolution must use a
        fixed priority order, not set iteration order."""
        svc = make_service("1: task/decision")
        out = await svc.classify_facts(["The team decided to ship"])
        assert out == ["decision"]  # longest match wins, deterministically

    @pytest.mark.asyncio
    async def test_classify_types_tags_operations(self) -> None:
        from engram.llm.service import LLMService
        from engram.providers.llm.protocol import LLMResponse

        def resp(content: str) -> LLMResponse:
            return LLMResponse(
                content=content, model="test-model", finish_reason="stop"
            )

        provider = MagicMock()
        provider.model = "test-model"
        provider.complete = AsyncMock(
            side_effect=[
                resp("User attended a concert\nUser likes jazz"),  # extraction
                resp("1: episodic\n2: semantic"),  # classification
            ]
        )
        svc = LLMService(provider=provider)

        async def retrieve(fact: str) -> list[tuple[str, str, float]]:
            return []

        result = await svc.process_for_memory(
            "m", "r", [], retrieve_for_fact=retrieve, classify_types=True
        )
        assert [op.memory_type for op in result.operations] == ["episodic", "semantic"]


class TestExpandQuery:
    """Tests for LLMService.expand_query() (HyDE)."""

    @pytest.mark.asyncio
    async def test_parses_and_strips_numbering(self) -> None:
        svc = make_service(
            "Where does the user live?\nUser's home city\n1. user location\n- residence"
        )
        out = await svc.expand_query("where does user live", n_queries=4)
        assert out == [
            "Where does the user live?",
            "User's home city",
            "user location",
            "residence",
        ]

    @pytest.mark.asyncio
    async def test_caps_to_n_queries(self) -> None:
        svc = make_service("query one\nquery two\nquery three")
        assert len(await svc.expand_query("x", n_queries=2)) == 2
