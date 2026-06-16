"""Unit tests for long-input chunking, anchoring, and context helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.client import Engram

DOC = """# Project Atlas Requirements

The checkout flow must complete in under 200ms at p95.
All payments shall be processed through the new gateway.

## Constraints

Never deploy on Fridays after 2 PM.
The deadline for phase one is March 15.

## Background

This is a longer background section. It contains several sentences that
describe the history of the project in some detail. The original system
was built in 2019 and has been extended many times since then. Each
extension added complexity that the team now wants to remove entirely.
"""


class TestChunkAnchors:
    """char_start/char_end must be exact spans into the source text."""

    def test_every_chunk_is_exact_substring(self) -> None:
        eg = Engram()
        chunks = eg._split_long_input(DOC)

        assert chunks, "expected at least one chunk"
        for chunk in chunks:
            assert DOC[chunk.char_start : chunk.char_end] == chunk.text, (
                f"anchor drift in chunk {chunk.chunk_id}: "
                f"span={chunk.char_start}-{chunk.char_end}"
            )

    def test_anchors_exact_with_forced_subsplits(self) -> None:
        """Sub-splitting long sections must not drift the offsets."""
        body = "  Leading spaces here. " + ("This sentence repeats. " * 200)
        text = f"# Heading\n\n{body}\n"
        eg = Engram()

        chunks = eg._split_long_input(text, max_chunk_tokens=100)

        assert len(chunks) > 1, "expected the section to be sub-split"
        for chunk in chunks:
            assert text[chunk.char_start : chunk.char_end] == chunk.text

    def test_anchors_exact_without_headings(self) -> None:
        text = "just a plain paragraph with no headings at all"
        eg = Engram()
        chunks = eg._split_long_input(text)

        assert len(chunks) == 1
        assert text[chunks[0].char_start : chunks[0].char_end] == chunks[0].text


class TestChonkieBackend:
    """Optional chonkie recursive chunker, with builtin fallback."""

    def _engram_chonkie(self) -> Engram:
        from engram.core.config import EngramSettings

        return Engram(settings=EngramSettings(long_input_chunker="chonkie"))

    def test_chonkie_chunks_have_exact_anchors(self) -> None:
        pytest.importorskip("chonkie")
        eg = self._engram_chonkie()

        chunks = eg._split_long_input(DOC, max_chunk_tokens=60)

        assert chunks, "expected chonkie to produce chunks"
        for chunk in chunks:
            assert DOC[chunk.char_start : chunk.char_end] == chunk.text
            assert chunk.kind  # classified by the pipeline, not chonkie

    def test_falls_back_to_builtin_when_chonkie_unavailable(self, monkeypatch) -> None:
        import engram.chunking

        # Simulate chonkie missing/failed: the span helper returns None.
        monkeypatch.setattr(
            engram.chunking, "chonkie_recursive_spans", lambda *_a, **_k: None
        )
        eg = self._engram_chonkie()

        chunks = eg._split_long_input(DOC)

        # The builtin splitter ran: it detects headings, which chonkie does not.
        assert any(c.heading == "Project Atlas Requirements" for c in chunks)
        for chunk in chunks:
            assert DOC[chunk.char_start : chunk.char_end] == chunk.text


class TestQueryTerms:
    """Short but critical tokens like p95/SLA must match source chunks."""

    def test_three_char_terms_included(self) -> None:
        eg = Engram()
        terms = eg._query_terms("what is the p95 target in the SLA?")

        assert "p95" in terms
        assert "sla" in terms
        # Noise words below 3 chars stay excluded
        assert "is" not in terms


class TestDocumentFactExtraction:
    """Long-input chunks are documents, not user chat; the extraction prompt
    must not phrase document content as facts about 'the User'."""

    @pytest.mark.asyncio
    async def test_extracts_lines_with_document_prompt(self) -> None:
        from engram.llm.service import LLMService
        from engram.providers.llm.protocol import LLMResponse

        provider = MagicMock()
        provider.model = "test-model"
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content=(
                    "The checkout p95 target is 200ms\n"
                    "Payments go through the new gateway"
                ),
                model="test-model",
                finish_reason="stop",
            )
        )
        svc = LLMService(provider=provider)

        facts = await svc.extract_document_facts(
            "The checkout flow must complete in under 200ms at p95...",
            kind="requirement",
            heading="Project Atlas Requirements",
        )

        assert facts == [
            "The checkout p95 target is 200ms",
            "Payments go through the new gateway",
        ]
        prompt = provider.complete.call_args.args[0][0]["content"]
        assert "document" in prompt.lower()
        assert "Project Atlas Requirements" in prompt

    @pytest.mark.asyncio
    async def test_truncated_document_extraction_drops_last_line(self) -> None:
        from engram.llm.service import LLMService
        from engram.providers.llm.protocol import LLMResponse

        provider = MagicMock()
        provider.model = "test-model"
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="Complete fact here\nCut off mid-sent",
                model="test-model",
                finish_reason="length",
            )
        )
        svc = LLMService(provider=provider)

        facts = await svc.extract_document_facts("text")

        assert facts == ["Complete fact here"]
