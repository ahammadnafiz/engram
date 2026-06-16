"""Optional recursive chunking backend for long-input ingestion.

Engram's builtin long-input splitter (``Engram._split_long_input``) is
structure/heading aware and dependency free. When the optional ``chonkie``
package is installed and selected via ``ENGRAM_LONG_INPUT_CHUNKER=chonkie``,
this module provides a token-aware recursive splitter as an alternative.

It returns character-anchored spans in the same shape the long-input pipeline
already assembles ``LongInputChunk`` objects from, and returns ``None`` when
chonkie is unavailable (or fails) so the caller falls back to the builtin
splitter — installs without ``chonkie`` keep working unchanged.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (heading, body, char_start, char_end). heading is None here — the long-input
# pipeline classifies/labels each span itself; chonkie only decides boundaries.
ChunkSpan = tuple[str | None, str, int, int]


def chonkie_recursive_spans(
    text: str,
    *,
    max_chunk_tokens: int,
) -> list[ChunkSpan] | None:
    """Split ``text`` into recursive chunks using chonkie's RecursiveChunker.

    Args:
        text: The raw long input to split.
        max_chunk_tokens: Approximate token budget per chunk. Mapped to chonkie's
            character tokenizer at ~4 chars/token to match the builtin splitter's
            sizing, so the two backends stay comparable.

    Returns:
        Character-anchored ``(heading, body, char_start, char_end)`` spans, or
        ``None`` if chonkie is not installed or chunking failed (caller falls
        back to the builtin splitter).
    """
    try:
        from chonkie import RecursiveChunker
    except ImportError:
        logger.debug("chonkie not installed; falling back to builtin chunker")
        return None

    # chonkie's character tokenizer counts characters, so size in characters
    # (~4 chars/token) to roughly match the builtin splitter's chunk sizes.
    chunk_chars = max(1, max_chunk_tokens * 4)
    try:
        chunker = RecursiveChunker(tokenizer="character", chunk_size=chunk_chars)
        chunks = chunker.chunk(text)
    except Exception as e:  # chonkie failure must not break ingestion
        logger.warning("chonkie chunking failed (%s); falling back to builtin", e)
        return None

    spans: list[ChunkSpan] = []
    for chunk in chunks:
        raw = chunk.text
        body = raw.strip()
        if not body:
            continue
        # Re-anchor onto the stripped body so char offsets stay exact spans
        # into the source text (the pipeline cites these for source-backed
        # answers).
        lead = len(raw) - len(raw.lstrip())
        start = int(chunk.start_index) + lead
        spans.append((None, body, start, start + len(body)))
    return spans or None
