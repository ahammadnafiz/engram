"""Unit tests for the OpenAI-backed chatbot example orchestration modes."""

from __future__ import annotations

import importlib.util
import os
from datetime import datetime, timezone
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


def memory(
    content: str,
    memory_id: str = "mem_1",
    *,
    lineage_id: str = "lin_1",
    revision: int = 1,
    status: str = "active",
):
    return SimpleNamespace(
        memory_id=memory_id,
        lineage_id=lineage_id,
        revision=revision,
        status=status,
        memory_type="semantic",
        importance=0.5,
        content=content,
        fact=content,
        created_at=datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc),
        valid_to=None,
        superseded_at=None,
    )


def search_result(content: str, memory_id: str = "mem_2"):
    return SimpleNamespace(memory=memory(content, memory_id))


def history_event(
    event_type: str,
    content: str,
    memory_id: str,
    *,
    previous_memory_id: str | None = None,
    superseded_by_memory_id: str | None = None,
    reason: str | None = None,
):
    return SimpleNamespace(
        event_type=event_type,
        occurred_at=datetime(2026, 6, 15, 10, 30, tzinfo=timezone.utc),
        memory=memory(content, memory_id),
        current_memory_id="mem_new",
        previous_memory_id=previous_memory_id,
        superseded_by_memory_id=superseded_by_memory_id,
        reason=reason,
    )


def fake_engram():
    old_memory = memory(
        "The user's live chatbot demo city is Dhaka.",
        "mem_old",
        revision=1,
        status="superseded",
    )
    new_memory = memory(
        "The user's live chatbot demo city is Singapore.",
        "mem_new",
        revision=2,
        status="active",
    )
    lineage = SimpleNamespace(
        lineage_id="lin_1",
        current_memory_id="mem_new",
        memories=[new_memory, old_memory],
    )
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
    engram.get_history = AsyncMock(
        return_value=[
            history_event(
                "revised",
                "The user's live chatbot demo city is Singapore.",
                "mem_new",
                previous_memory_id="mem_old",
                reason="demo_correction",
            ),
            history_event(
                "superseded",
                "The user's live chatbot demo city is Dhaka.",
                "mem_old",
                superseded_by_memory_id="mem_new",
            ),
        ]
    )
    engram.recall = AsyncMock(
        return_value=SimpleNamespace(
            answer_text="The current city is Singapore; the previous city was Dhaka.",
            intent="historical",
            current=new_memory,
            previous=[old_memory],
            when_changed=datetime(2026, 6, 15, 10, 30, tzinfo=timezone.utc),
            evidence=[new_memory, old_memory],
            events=[
                SimpleNamespace(
                    role="user",
                    created_at=datetime(2026, 6, 15, 10, 20, tzinfo=timezone.utc),
                    content="I moved from Dhaka to Singapore.",
                )
            ],
            conflict_note=None,
            sources=[],
            trace={"topic": "city"},
        )
    )
    engram.build_context = AsyncMock(return_value=SimpleNamespace(text="Task context"))
    engram.record_turn = AsyncMock()
    engram.process_memory_jobs = AsyncMock(return_value=[])
    engram.add = AsyncMock(return_value=old_memory)
    engram.revise = AsyncMock(return_value=new_memory)
    engram.get_current = AsyncMock(return_value=new_memory)
    engram.get_lineage = AsyncMock(return_value=lineage)
    engram.explain_memory = AsyncMock(
        return_value=SimpleNamespace(
            memory=old_memory,
            current=new_memory,
            lineage=lineage,
            supersedes=[],
            superseded_by=new_memory,
        )
    )
    return engram


def memory_job(status: str):
    return SimpleNamespace(status=status)


def make_bot(module, engram):
    bot = module.MemoryChatbot()
    bot.engram = engram
    bot.task_id = "task_1"
    bot.session_id = "session_1"
    return bot


def test_memory_jobs_default_is_inline(monkeypatch):
    monkeypatch.delenv("ENGRAM_CHATBOT_MEMORY_JOBS", raising=False)
    module = load_chatbot_module()

    assert module.MEMORY_JOBS_MODE == "inline"


def test_recall_mode_default_is_operator(monkeypatch):
    monkeypatch.delenv("ENGRAM_CHATBOT_RECALL_MODE", raising=False)
    module = load_chatbot_module()

    assert module.RECALL_MODE == "operator"


def test_standard_gemini_key_maps_to_engram_key(monkeypatch):
    def fake_load_dotenv(*_args, **_kwargs):
        return False

    monkeypatch.delenv("ENGRAM_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("dotenv.load_dotenv", fake_load_dotenv)

    load_chatbot_module()

    assert os.environ["ENGRAM_GEMINI_API_KEY"] == "test-key"


def test_default_stack_is_local_embeddings_and_gemini(monkeypatch):
    for var in (
        "ENGRAM_EMBEDDING_PROVIDER",
        "ENGRAM_EMBEDDING_MODEL",
        "ENGRAM_EMBEDDING_DIMENSION",
        "ENGRAM_LLM_PROVIDER",
        "ENGRAM_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_a, **_k: False)

    load_chatbot_module()

    assert os.environ["ENGRAM_EMBEDDING_PROVIDER"] == "sentence-transformers"
    assert os.environ["ENGRAM_EMBEDDING_MODEL"] == "all-MiniLM-L6-v2"
    assert os.environ["ENGRAM_EMBEDDING_DIMENSION"] == "384"
    assert os.environ["ENGRAM_LLM_PROVIDER"] == "gemini"
    assert os.environ["ENGRAM_LLM_MODEL"] == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_operator_mode_uses_recall_evidence_then_chat_llm(monkeypatch):
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RECALL_MODE", "operator")
    monkeypatch.setattr(module, "MEMORY_JOBS_MODE", "deferred")
    monkeypatch.setattr(module, "RERANK_MODE", "auto")
    engram = fake_engram()
    bot = make_bot(module, engram)

    response = await bot.reply("What changed about my city?")

    engram.recall.assert_awaited_once_with(
        "What changed about my city?",
        module.AGENT_ID,
        user_id=module.USER_ID,
        compose_answer=False,
    )
    assert response == "remembered answer"
    engram.llm.complete_full.assert_awaited_once()
    engram.get_context_block.assert_awaited_once()
    engram.deep_search.assert_not_awaited()
    engram.recall_critical.assert_awaited_once()
    engram.get_history.assert_awaited_once()

    messages = engram.llm.complete_full.call_args.args[0]
    prompt = "\n".join(message["content"] for message in messages)
    assert '<engram_recall_evidence intent="historical">' in prompt
    assert "current: The user's live chatbot demo city is Singapore." in prompt
    assert "previous until" in prompt
    assert "<engram_memory_history>" in prompt

    metadata = engram.record_turn.call_args.kwargs["metadata"]
    assert metadata["llm_model"] == "fake-llm"
    assert metadata["recall_intent"] == "historical"
    assert metadata["recall_previous_count"] == 1
    assert metadata["recall_event_count"] == 1
    assert metadata["recall_source_count"] == 0
    assert metadata["operator_route"] == "recall_chat"
    assert metadata["memory_history_count"] == 2


@pytest.mark.asyncio
async def test_operator_mode_uses_chat_path_for_user_authored_facts(monkeypatch):
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RECALL_MODE", "operator")
    monkeypatch.setattr(module, "MEMORY_JOBS_MODE", "inline")
    monkeypatch.setattr(module, "RERANK_MODE", "auto")
    engram = fake_engram()
    engram.recall = AsyncMock(
        return_value=SimpleNamespace(
            answer_text="",
            intent="chat",
            current=None,
            previous=[],
            evidence=[],
            events=[],
            sources=[],
            conflict_note=None,
            trace={"topic": "meeting"},
        )
    )
    bot = make_bot(module, engram)

    response = await bot.reply("i have a meeting at 3pm in zoom")

    assert response == "remembered answer"
    engram.recall.assert_awaited_once_with(
        "i have a meeting at 3pm in zoom",
        module.AGENT_ID,
        user_id=module.USER_ID,
        compose_answer=False,
    )
    engram.llm.complete_full.assert_awaited_once()
    engram.get_context_block.assert_awaited_once()
    engram.process_memory_jobs.assert_awaited_once_with(limit=module.MEMORY_JOBS_LIMIT)

    metadata = engram.record_turn.call_args.kwargs["metadata"]
    assert metadata["llm_model"] == "fake-llm"
    assert metadata["recall_intent"] == "chat"
    assert metadata["operator_route"] == "chat"
    assert metadata["memory_history_count"] == 2


@pytest.mark.asyncio
async def test_fast_mode_uses_active_recall_and_timeline_context(monkeypatch):
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
    engram.get_history.assert_awaited_once_with(
        module.AGENT_ID,
        user_id=module.USER_ID,
        limit=module.MEMORY_HISTORY_LIMIT,
        include_superseded=True,
    )
    engram.process_memory_jobs.assert_not_awaited()

    messages = engram.llm.complete_full.call_args.args[0]
    prompt = "\n".join(message["content"] for message in messages)
    assert "Critical fact" in prompt
    assert "Fact A" in prompt
    assert "<engram_memory_history>" in prompt
    assert "The user's live chatbot demo city is Dhaka." in prompt
    assert "The user's live chatbot demo city is Singapore." in prompt

    metadata = engram.record_turn.call_args.kwargs["metadata"]
    assert metadata["recall_mode"] == "fast"
    assert metadata["critical_memory_count"] == 1
    assert metadata["memory_history_count"] == 2


@pytest.mark.asyncio
async def test_fast_mode_history_does_not_depend_on_keyword_routing(monkeypatch):
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RECALL_MODE", "fast")
    monkeypatch.setattr(module, "MEMORY_JOBS_MODE", "deferred")
    monkeypatch.setattr(module, "RERANK_MODE", "auto")
    engram = fake_engram()
    bot = make_bot(module, engram)

    await bot.reply("Can you reconstruct that schedule change from my memory?")

    engram.get_history.assert_awaited_once_with(
        module.AGENT_ID,
        user_id=module.USER_ID,
        limit=module.MEMORY_HISTORY_LIMIT,
        include_superseded=True,
    )
    messages = engram.llm.complete_full.call_args.args[0]
    prompt = "\n".join(message["content"] for message in messages)
    assert "<engram_memory_history>" in prompt
    assert "Use this block only for questions about previous values" in prompt
    assert "revised" in prompt
    assert "superseded" in prompt
    assert "The user's live chatbot demo city is Dhaka." in prompt
    assert "The user's live chatbot demo city is Singapore." in prompt

    metadata = engram.record_turn.call_args.kwargs["metadata"]
    assert metadata["memory_history_count"] == 2


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
    engram.process_memory_jobs.assert_awaited_once_with(limit=module.MEMORY_JOBS_LIMIT)

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


@pytest.mark.asyncio
async def test_lineage_command_uses_lineage_api(monkeypatch):
    module = load_chatbot_module()
    engram = fake_engram()
    bot = make_bot(module, engram)

    keep_running = await module.run_command(bot, "/lineage mem_old")

    assert keep_running is True
    engram.get_current.assert_awaited_once_with("mem_old")
    engram.get_lineage.assert_awaited_once_with("mem_old")
    engram.explain_memory.assert_awaited_once_with("mem_old")


@pytest.mark.asyncio
async def test_revise_command_creates_new_revision(monkeypatch):
    module = load_chatbot_module()
    engram = fake_engram()
    bot = make_bot(module, engram)

    keep_running = await module.run_command(
        bot, "/revise mem_old The user's live chatbot demo city is Singapore."
    )

    assert keep_running is True
    engram.revise.assert_awaited_once_with(
        "mem_old",
        content="The user's live chatbot demo city is Singapore.",
        metadata={"source": "manual_chatbot_revision"},
        reason="manual_chatbot_revision",
    )
    engram.get_current.assert_awaited_once_with("mem_old")
    engram.get_lineage.assert_awaited_once_with("mem_old")


@pytest.mark.asyncio
async def test_history_command_uses_history_api(monkeypatch):
    module = load_chatbot_module()
    engram = fake_engram()
    bot = make_bot(module, engram)

    keep_running = await module.run_command(bot, "/history 25")

    assert keep_running is True
    engram.get_history.assert_awaited_once_with(
        module.AGENT_ID,
        user_id=module.USER_ID,
        limit=25,
        include_superseded=True,
    )


@pytest.mark.asyncio
async def test_history_command_with_memory_id_uses_lineage(monkeypatch):
    module = load_chatbot_module()
    engram = fake_engram()
    bot = make_bot(module, engram)

    keep_running = await module.run_command(bot, "/history mem_old")

    assert keep_running is True
    engram.get_history.assert_not_awaited()
    engram.get_lineage.assert_awaited_once_with("mem_old")
    engram.explain_memory.assert_awaited_once_with("mem_old")


@pytest.mark.asyncio
async def test_jobs_command_processes_queued_memory_jobs(monkeypatch):
    module = load_chatbot_module()
    engram = fake_engram()
    engram.process_memory_jobs = AsyncMock(
        return_value=[memory_job("completed"), memory_job("failed")]
    )
    bot = make_bot(module, engram)

    keep_running = await module.run_command(bot, "/jobs")

    assert keep_running is True
    engram.process_memory_jobs.assert_awaited_once_with(limit=module.MEMORY_JOBS_LIMIT)


@pytest.mark.asyncio
async def test_demo_uses_lineage_api(monkeypatch):
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RECALL_MODE", "fast")
    monkeypatch.setattr(module, "MEMORY_JOBS_MODE", "deferred")
    monkeypatch.setattr(module, "RERANK_MODE", "auto")
    engram = fake_engram()
    bot = make_bot(module, engram)

    await module.run_demo(bot)

    engram.add.assert_awaited_once()
    engram.revise.assert_awaited_once()
    engram.get_current.assert_awaited_once_with("mem_old")
    engram.get_lineage.assert_awaited_once_with("mem_old")
    engram.explain_memory.assert_awaited_once_with("mem_old")
    engram.llm.complete_full.assert_awaited_once()
