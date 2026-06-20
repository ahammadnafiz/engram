"""Unit tests for the Engram-backed chatbot example."""

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
    # Retrieval/ingest surfaces used by reply().
    engram.search = AsyncMock(return_value=[])
    engram.traverse_many = AsyncMock(return_value=[])
    engram.render_graph_context = lambda *_a, **_k: ""
    engram.add_batch = AsyncMock(return_value=[])
    return engram


def make_bot(module, engram):
    bot = module.MemoryChatbot()
    bot.engram = engram
    bot.task_id = "task_1"
    bot.session_id = "session_1"
    return bot


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
    assert os.environ["ENGRAM_LLM_MODEL"] == "gemini-3.1-flash-lite"


@pytest.mark.asyncio
async def test_reply_composes_then_stores_via_add_batch(monkeypatch):
    """Each turn ingests through the raw add_batch floor, never add_conversation.

    The hybrid was removed: co-locating the two writers made the extractor read
    every fact as already-present and NOOP it. This guards against re-wiring it.
    """
    module = load_chatbot_module()
    monkeypatch.setattr(module, "RERANK_MODE", "auto")
    engram = fake_engram()
    engram.add_conversation = AsyncMock()  # must stay untouched by the floor path
    bot = make_bot(module, engram)

    response = await bot.reply("What's my current city?")

    # Composed an answer over retrieved evidence...
    assert response == "remembered answer"
    engram.llm.complete_full.assert_awaited_once()

    # ...and stored the turn via the add_batch floor, not add_conversation.
    engram.add_batch.assert_awaited_once()
    engram.add_conversation.assert_not_awaited()
    rows = engram.add_batch.call_args.args[0]
    assert rows[0]["metadata"]["source"] == "chatbot"
    assert rows[0]["metadata"]["turn_role"] == "user"


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
