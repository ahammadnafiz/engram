"""Unit tests for engram.mcp.server — chunking-based ingestion and evidence assembly."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers imported directly from the module under test
# ---------------------------------------------------------------------------
from engram.mcp.server import (
    CANDIDATE_LIMIT,
    CHUNK_TOKENS,
    MAX_PER_SESSION,
    SEARCH_LIMIT,
    _build_evidence_block,
    _chunk_text,
    _result_text,
    save_turn,
)


# ===========================================================================
# _result_text — no truncation
# ===========================================================================
class TestResultText:
    def test_string_passthrough(self):
        assert _result_text("hello world") == "hello world"

    def test_strips_whitespace(self):
        assert _result_text("  hi  ") == "hi"

    def test_list_of_text_blocks(self):
        blocks = [
            {"type": "text", "text": "first"},
            {"type": "image", "data": "..."},
            {"type": "text", "text": "second"},
        ]
        assert _result_text(blocks) == "first\nsecond"

    def test_long_content_not_truncated(self):
        long = "x" * 10_000
        assert _result_text(long) == long

    def test_empty_string(self):
        assert _result_text("") == ""


# ===========================================================================
# _chunk_text — Chonkie integration with fallback
# ===========================================================================
class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = _chunk_text("This is a short sentence.")
        assert len(chunks) >= 1
        # The original text must survive intact (possibly as one chunk)
        assert "short sentence" in " ".join(chunks)

    def test_long_text_multiple_chunks(self):
        # ~4 x CHUNK_TOKENS chars guarantees at least 2 chunks from Chonkie
        long = ("word " * (CHUNK_TOKENS * 4)).strip()
        chunks = _chunk_text(long)
        assert len(chunks) > 1

    def test_chunks_cover_full_text(self):
        """No content should be silently dropped — verify total character count.

        Chonkie uses a character tokenizer so it can bisect words at chunk
        boundaries.  We therefore check that the total stripped content across
        all chunks equals the original text rather than doing word-level checks.
        """
        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)
        chunks = _chunk_text(text)
        # All characters must be present across chunks (whitespace may differ at
        # boundaries, so compare stripped non-space content)
        original_chars = text.replace(" ", "")
        chunk_chars = "".join(c.replace(" ", "") for c in chunks)
        assert chunk_chars == original_chars

    def test_fallback_when_chonkie_missing(self):
        """If chonkie is not installed, _chunk_text must return the original text."""
        with patch("engram.chunking.chonkie_recursive_spans", side_effect=ImportError):
            chunks = _chunk_text("some text")
        assert chunks == ["some text"]

    def test_fallback_on_chonkie_exception(self):
        with patch(
            "engram.chunking.chonkie_recursive_spans", side_effect=RuntimeError("boom")
        ):
            chunks = _chunk_text("some text")
        assert chunks == ["some text"]

    def test_empty_string(self):
        chunks = _chunk_text("")
        # Either one empty chunk or no chunks — either is fine; must not raise
        assert isinstance(chunks, list)


# ===========================================================================
# _build_evidence_block — full content, no truncation marks
# ===========================================================================
def _make_memory(
    memory_id: str,
    content: str,
    status: str = "active",
    chat_date: str = "2026-06-25",
    fact: str | None = None,
) -> MagicMock:
    mem = MagicMock()
    mem.memory_id = memory_id
    mem.content = content
    mem.fact = fact
    mem.status = status
    mem.metadata = {"chat_date": chat_date, "status": status}
    mem.superseded_at = None
    mem.valid_to = None
    mem.created_at = None
    return mem


def _make_result(memory: MagicMock, score: float = 1.0) -> MagicMock:
    r = MagicMock()
    r.memory = memory
    r.score = score
    return r


class TestBuildEvidenceBlock:
    def test_active_memory_tagged(self):
        mem = _make_memory("m1", "I prefer dark mode")
        block = _build_evidence_block([_make_result(mem)], [])
        assert "[ACTIVE]" in block
        assert "I prefer dark mode" in block

    def test_superseded_memory_tagged(self):
        mem = _make_memory("m1", "old preference", status="superseded")
        block = _build_evidence_block([_make_result(mem)], [])
        assert "[SUPERSEDED]" in block

    def test_date_header_present(self):
        mem = _make_memory("m1", "note", chat_date="2026-06-25")
        block = _build_evidence_block([_make_result(mem)], [])
        assert "June 25, 2026" in block

    def test_long_content_not_truncated(self):
        long_content = "detail " * 300  # ~2 100 chars, well over the old 500-char cap
        mem = _make_memory("m1", long_content.strip())
        block = _build_evidence_block([_make_result(mem)], [])
        assert "…" not in block
        assert "(evidence truncated" not in block
        assert "detail" in block

    def test_lineage_superseded_listed_first(self):
        import datetime

        sup = MagicMock()
        sup.memory_id = "sup1"
        sup.content = "old value"
        sup.fact = "old value"
        sup.status = "superseded"
        sup.superseded_at = datetime.date(2026, 1, 1)
        sup.valid_to = None
        sup.created_at = None
        sup.superseded_at = MagicMock()
        sup.superseded_at.date.return_value = datetime.date(2026, 1, 1)

        active = _make_memory("m2", "new value")
        block = _build_evidence_block([_make_result(active)], [sup])
        assert block.index("SUPERSEDED") < block.index("RETRIEVED MEMORIES")

    def test_deduplication(self):
        mem = _make_memory("same-id", "content")
        result = _make_result(mem)
        # Passing same memory in both lists — should appear only once in output
        block = _build_evidence_block([result], [mem])
        assert block.count("content") == 1

    def test_no_results_returns_section_header(self):
        block = _build_evidence_block([], [])
        assert "RETRIEVED MEMORIES" in block

    def test_multiple_dates_grouped(self):
        m1 = _make_memory("m1", "note A", chat_date="2026-06-20")
        m2 = _make_memory("m2", "note B", chat_date="2026-06-25")
        block = _build_evidence_block([_make_result(m1), _make_result(m2)], [])
        assert "June 20, 2026" in block
        assert "June 25, 2026" in block


# ===========================================================================
# save_turn — Chonkie chunking produces multiple rows for long content
# ===========================================================================
@pytest.mark.asyncio
class TestSaveTurn:
    def _make_engram(self, session_id: str = "sess-1") -> MagicMock:
        eng = MagicMock()

        task = MagicMock()
        task.session_id = session_id
        eng.list_tasks = AsyncMock(return_value=[task])

        # add_batch returns a list with one element per row stored
        async def _add_batch(rows):  # type: ignore[no-untyped-def]
            return [MagicMock() for _ in rows]

        eng.add_batch = _add_batch
        return eng

    async def test_short_turn_stores_two_rows(self):
        eng = self._make_engram()
        n = await save_turn(eng, "hello", "hi there")
        # one user row + one assistant row
        assert n == 2

    async def test_empty_assistant_stores_one_row(self):
        eng = self._make_engram()
        n = await save_turn(eng, "hello", "")
        assert n == 1

    async def test_long_user_message_stored_as_multiple_rows(self):
        eng = self._make_engram()
        long_user = ("word " * (CHUNK_TOKENS * 5)).strip()
        n = await save_turn(eng, long_user, "ok")
        # long user text → multiple chunks + 1 assistant row
        assert n > 2

    async def test_long_assistant_response_stored_as_multiple_rows(self):
        eng = self._make_engram()
        long_assistant = ("sentence. " * (CHUNK_TOKENS * 5)).strip()
        n = await save_turn(eng, "question", long_assistant)
        assert n > 2

    async def test_chunk_metadata_added_for_multi_chunk(self):
        stored_rows: list[list[dict]] = []

        async def capture_batch(rows):  # type: ignore[no-untyped-def]
            stored_rows.append(rows)
            return [MagicMock() for _ in rows]

        eng = self._make_engram()
        eng.add_batch = capture_batch

        long_user = ("word " * (CHUNK_TOKENS * 5)).strip()
        await save_turn(eng, long_user, "short reply")

        all_rows = stored_rows[0]
        user_rows = [r for r in all_rows if r["metadata"]["turn_role"] == "user"]
        if len(user_rows) > 1:
            for r in user_rows:
                assert "chunk_index" in r["metadata"]
                assert "chunk_count" in r["metadata"]
                assert r["metadata"]["chunk_count"] == len(user_rows)

    async def test_no_chunk_metadata_for_single_chunk(self):
        stored_rows: list[list[dict]] = []

        async def capture_batch(rows):  # type: ignore[no-untyped-def]
            stored_rows.append(rows)
            return [MagicMock() for _ in rows]

        eng = self._make_engram()
        eng.add_batch = capture_batch

        await save_turn(eng, "short", "short reply")

        all_rows = stored_rows[0]
        for r in all_rows:
            assert "chunk_index" not in r["metadata"]

    async def test_fallback_when_chonkie_unavailable(self):
        """If chonkie import fails, save_turn must still store one row per role."""
        with patch("engram.mcp.server._chunk_text", side_effect=lambda t: [t]):
            eng = self._make_engram()
            n = await save_turn(eng, "hello", "world")
        assert n == 2


# ===========================================================================
# Tuning-constant sanity — output budget stays within the target range
# ===========================================================================
class TestTuningConstants:
    def test_recall_output_token_budget(self):
        """SEARCH_LIMIT x CHUNK_TOKENS must stay in the 3k-8k token range.

        Too low → poor accuracy (not enough context).
        Too high → context bloat that degrades host LLM reasoning.
        """
        max_output_tokens = SEARCH_LIMIT * CHUNK_TOKENS
        assert 3_000 <= max_output_tokens <= 8_000, (
            f"SEARCH_LIMIT={SEARCH_LIMIT} x CHUNK_TOKENS={CHUNK_TOKENS} = "
            f"{max_output_tokens} tokens — outside the 3k-8k target window"
        )

    def test_reranker_compression_ratio(self):
        """CANDIDATE_LIMIT / SEARCH_LIMIT should be ≥ 4 so reranking has room to work."""
        ratio = CANDIDATE_LIMIT / SEARCH_LIMIT
        assert ratio >= 4, (
            f"CANDIDATE_LIMIT={CANDIDATE_LIMIT} / SEARCH_LIMIT={SEARCH_LIMIT} = {ratio:.1f} "
            f"— reranker needs at least 4:1 candidate-to-result ratio for meaningful filtering"
        )

    def test_per_session_cap_below_search_limit(self):
        """MAX_PER_SESSION must be less than SEARCH_LIMIT to enforce diversity."""
        assert MAX_PER_SESSION < SEARCH_LIMIT
