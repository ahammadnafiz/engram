#!/usr/bin/env python3
"""
Long Input Memory Demo
======================

Shows how to ingest a large prompt or legal/source document with source
anchors, typed memories, a manifest checkpoint, deterministic recall, and
traceable context assembly.

Usage:
    python examples/long_input_usage.py

Provider setup:
    ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
    ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2

Or with OpenAI:
    ENGRAM_EMBEDDING_PROVIDER=openai
    ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
    ENGRAM_OPENAI_API_KEY=...
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


LONG_PROMPT = """
# Legal Review Instructions
The agent must answer with citations to source chunks.
The agent must not rely on summaries when an exact clause is available.
Next Wednesday is the review deadline.

# Section 1 Audit Logs
The vendor shall maintain audit logs for seven years.
The logs must include administrator access, policy changes, data export events,
and failed authentication attempts.

# Section 2 Liability
The supplier shall not cap liability for confidentiality breaches, data misuse,
or willful misconduct. Ordinary service issues may be capped at twelve months of fees.

# Section 3 Indemnity
The vendor must indemnify the customer for third-party claims arising from
intellectual property infringement, security incidents caused by vendor negligence,
and unauthorized disclosure.

# Section 4 Agent Task Requirements
The agent must produce a risk table with columns risk, trigger, source clause,
owner, and mitigation. The agent should flag uncapped liability, missing audit
exports, and weak breach notification language.
"""


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def main() -> None:
    from engram import Engram

    async with Engram(memory_policy="legal") as engram:
        task = await engram.start_task(
            "Review a long legal prompt with source-grounded memory",
            "legal-review-agent",
            user_id="demo-user",
            metadata={"example": "long_input_usage"},
        )

        print_section("Ingest Long Input")
        report = await engram.record_long_input(
            task.task_run_id,
            LONG_PROMPT,
            title="Vendor SaaS legal review",
            max_chunk_tokens=220,
            extract_with_llm=False,
            metadata={"example": "long_input_usage"},
        )
        print(f"Task:          {report.task_run_id}")
        print(f"Source event:  {report.source_event_id}")
        print(f"Chunks:        {len(report.chunks)}")
        print(f"Memories:      {len(report.memory_ids)}")
        print(f"Checkpoint:    {report.checkpoint_id}")
        print(f"Time notes:    {report.trace['time_notes']}")

        print_section("Chunk Anchors")
        for chunk in report.chunks[:4]:
            print(
                f"- {chunk.chunk_id} [{chunk.kind}] "
                f"chars={chunk.char_start}-{chunk.char_end} hash={chunk.quote_hash}"
            )

        print_section("Build Source-Grounded Context")
        context = await engram.build_long_input_context(
            task.task_run_id,
            query="audit logs liability indemnity risk table citations",
            max_tokens=3500,
            source_chunk_limit=4,
            expected_terms=["audit logs", "liability", "indemnify", "citations"],
        )
        print(f"Context tokens: {context.token_estimate}")
        print(f"Missing terms:  {context.trace['missing_expected_terms']}")
        print(f"Chunks kept:    {context.metadata['source_chunks_kept']}")

        print_section("Context Preview")
        print(context.text[:2500])

        print_section("Recall Trace")
        recall = context.trace["recall"]
        print(f"Critical memories: {len(recall['critical_memory_ids'])}")
        print(f"Search memories:   {len(recall['search_memory_ids'])}")
        print(f"Kept memories:     {len(recall['kept_memory_ids'])}")
        print(f"Trimmed memories:  {len(recall['trimmed_memory_ids'])}")
        print(f"Superseded hidden: {len(recall['superseded_memory_ids'])}")


if __name__ == "__main__":
    asyncio.run(main())
