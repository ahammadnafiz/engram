"""Unit tests for the OpenAI-backed chatbot example orchestration modes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def load_chatbot_module():
    path = Path(__file__).resolve().parents[2] / "examples" / "chatbot.py"
    spec = importlib.util.spec_from_file_location("engram_chatbot_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def memory(content: str, memory_id: str = "mem_1"):
    return SimpleNamespace(
        memory_id=memory_id,
        memory_type="semantic",
        content=content,
    )


def search_result(content: str, memory_id: str = "mem_2"):
    return SimpleNamespace(memory=memory(content, memory_id))


def fake_engram():
    engram = SimpleNamespace()
    engram.llm = SimpleNamespace(
        complete_full=AsyncMock(
            return_value=SimpleNamespace(
                content=" remembered answer ", model="fake-llm"
            )
        )
    )
    engram.recall_critical = AsyncMock(return_value=[memory("Critical fact")])
    engram.get_context_block = AsyncMock(return_value="## Relevant memories\n- Fact A")
    engram.trace_recall = AsyncMock(
        return_value=SimpleNamespace(
            context="## Memory Recall\n- Trace fact",
            kept_memory_ids=["mem_trace"],
            missing_expected_terms=[],
        )
    )
    engram.deep_search = AsyncMock(return_value=[search_result("Deep fact")])
    engram.list_recent = AsyncMock(return_value=[memory("Recent fact")])
    engram.build_context = AsyncMock(return_value=SimpleNamespace(text="Task context"))
    engram.record_turn = AsyncMock()
    engram.process_memory_jobs = AsyncMock(return_value=[])
    return engram


def make_bot(module, engram):
    bot = module.MemoryChatbot()
    bot.engram = engram
    bot.task_id = "task_1"
    bot.session_id = "session_1"
    return bot


@pytest.mark.asyncio
async def test_fast_mode_uses_one_context_block_without_deep_calls(monkeypatch):
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RECALL_MODE", "fast")
    monkeypatch.setattr(module, "MEMORY_JOBS_MODE", "deferred")
    monkeypatch.setattr(module, "RERANK_MODE", "auto")
    engram = fake_engram()
    bot = make_bot(module, engram)

    response = await bot.reply("What do you remember?")

    assert response == "remembered answer"
    engram.recall_critical.assert_awaited_once()
    engram.get_context_block.assert_awaited_once()
    assert engram.get_context_block.call_args.kwargs["rerank"] is False
    engram.trace_recall.assert_not_awaited()
    engram.deep_search.assert_not_awaited()
    engram.list_recent.assert_not_awaited()
    engram.build_context.assert_not_awaited()
    engram.process_memory_jobs.assert_not_awaited()

    messages = engram.llm.complete_full.call_args.args[0]
    prompt = "\n".join(message["content"] for message in messages)
    assert "Critical fact" in prompt
    assert "Fact A" in prompt

    metadata = engram.record_turn.call_args.kwargs["metadata"]
    assert metadata["recall_mode"] == "fast"
    assert metadata["critical_memory_count"] == 1


@pytest.mark.asyncio
async def test_deep_mode_keeps_trace_recall_out_of_hot_path(monkeypatch):
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RECALL_MODE", "deep")
    monkeypatch.setattr(module, "MEMORY_JOBS_MODE", "deferred")
    monkeypatch.setattr(module, "RERANK_MODE", "auto")
    engram = fake_engram()
    bot = make_bot(module, engram)

    await bot.reply("Find broad memory")

    engram.trace_recall.assert_not_awaited()
    engram.deep_search.assert_awaited_once()
    assert engram.get_context_block.call_args.kwargs["rerank"] is True
    assert engram.deep_search.call_args.kwargs["rerank"] is True
    engram.list_recent.assert_awaited_once()
    engram.build_context.assert_awaited_once()

    metadata = engram.record_turn.call_args.kwargs["metadata"]
    assert metadata["recall_mode"] == "deep"
    assert "trace_kept_memory_ids" not in metadata


@pytest.mark.asyncio
async def test_debug_mode_includes_trace_metadata_and_inline_jobs(monkeypatch):
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RECALL_MODE", "debug")
    monkeypatch.setattr(module, "MEMORY_JOBS_MODE", "inline")
    monkeypatch.setattr(module, "RERANK_MODE", "auto")
    engram = fake_engram()
    bot = make_bot(module, engram)

    await bot.reply("Debug recall")

    engram.trace_recall.assert_awaited_once()
    engram.deep_search.assert_awaited_once()
    engram.process_memory_jobs.assert_awaited_once_with(limit=10)

    metadata = engram.record_turn.call_args.kwargs["metadata"]
    assert metadata["recall_mode"] == "debug"
    assert metadata["trace_kept_memory_ids"] == ["mem_trace"]
    assert metadata["missing_expected_terms"] == []


@pytest.mark.asyncio
async def test_rerank_true_forces_fast_mode_reranking(monkeypatch):
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RECALL_MODE", "fast")
    monkeypatch.setattr(module, "MEMORY_JOBS_MODE", "deferred")
    monkeypatch.setattr(module, "RERANK_MODE", "true")
    engram = fake_engram()
    bot = make_bot(module, engram)

    await bot.reply("What do you remember?")

    assert engram.get_context_block.call_args.kwargs["rerank"] is True
