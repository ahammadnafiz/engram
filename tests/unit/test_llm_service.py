"""Unit tests for LLMService summary helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def make_service(reply: str):
    from engram.llm.service import LLMService

    provider = MagicMock()
    provider.model = "test-model"
    provider.complete_text = AsyncMock(return_value=reply)
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

        provider = MagicMock()
        provider.model = "test-model"
        # Extraction returns two facts; with empty candidates each evaluates to ADD
        # without a further LLM call.
        provider.complete_text = AsyncMock(return_value="Fact one here\nFact two here")
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
    async def test_classify_types_tags_operations(self) -> None:
        from engram.llm.service import LLMService

        provider = MagicMock()
        provider.model = "test-model"
        provider.complete_text = AsyncMock(
            side_effect=[
                "User attended a concert\nUser likes jazz",  # extraction
                "1: episodic\n2: semantic",  # classification
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
