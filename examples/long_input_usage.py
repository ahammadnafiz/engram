#!/usr/bin/env python3
"""End-to-end long-input and source-grounded context demo.

Run:
    python examples/long_input_usage.py

This example uses heuristic extraction by default so it works without an LLM
provider. Set ENGRAM_LLM_PROVIDER and provider credentials to enable richer
fact extraction and answer_from_evidence output.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=False)
os.environ.setdefault(
    "ENGRAM_DATABASE_URL",
    "postgresql://engram:engram_secret@localhost:5432/engram",
)
os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


DOCUMENT = """
# Legal Review Instructions
The agent must answer with citations to source chunks.
The agent must prefer exact source text over distilled memory summaries.
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


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def item(label: str, value: Any) -> None:
    print(f"{label:<24} {value}")


def preview(text: str, limit: int = 1600) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


async def main() -> None:
    from engram import Engram

    agent_id = f"legal-review-{uuid.uuid4().hex[:8]}"
    user_id = "demo-user"

    async with Engram(memory_policy="legal") as engram:
        section("1. Start Review Task")
        task = await engram.start_task(
            "Review vendor agreement with source-grounded memory",
            agent_id,
            user_id=user_id,
            metadata={"example": "long_input_usage"},
        )
        item("task", task.task_run_id)
        item("llm_enabled", engram.llm is not None)

        section("2. Record Long Input")
        report = await engram.record_long_input(
            task.task_run_id,
            DOCUMENT,
            title="Vendor SaaS legal review",
            max_chunk_tokens=220,
            extract_with_llm=engram.llm is not None,
            max_facts_per_chunk=5,
            metadata={"document_id": "vendor-saas-demo"},
        )
        item("source_event", report.source_event_id)
        item("chunks", len(report.chunks))
        item("anchored_memories", len(report.memory_ids))
        item("checkpoint", report.checkpoint_id)
        item("time_notes", report.trace["time_notes"])

        print("\nChunk anchors:")
        for chunk in report.chunks:
            print(
                f"- {chunk.chunk_id} [{chunk.kind}] "
                f"chars={chunk.char_start}-{chunk.char_end} hash={chunk.quote_hash}"
            )

        section("3. Build Long-Input Context")
        context = await engram.build_long_input_context(
            task.task_run_id,
            query="audit logs liability indemnity risk table citations",
            max_tokens=3500,
            source_chunk_limit=4,
            expected_terms=["audit logs", "liability", "indemnify", "citations"],
        )
        item("token_estimate", context.token_estimate)
        item("missing_terms", context.trace["missing_expected_terms"])
        item("source_chunks_kept", context.metadata["source_chunks_kept"])
        print(preview(context.text))

        section("4. Recall Trace")
        recall = context.trace["recall"]
        item("critical_memories", len(recall["critical_memory_ids"]))
        item("search_memories", len(recall["search_memory_ids"]))
        item("kept_memories", len(recall["kept_memory_ids"]))
        item("trimmed_memories", len(recall["trimmed_memory_ids"]))
        item("superseded_hidden", len(recall["superseded_memory_ids"]))

        section("5. Evidence Set And Neighboring Context")
        evidence = await engram.search_evidence_set(
            "What audit log and liability obligations matter?",
            agent_id,
            user_id=user_id,
            limit=6,
            memory_types=["constraint", "task", "tool_result", "semantic"],
            rerank=False,
        )
        neighbor_context, sources = await engram.get_neighboring_context_block(
            evidence,
            agent_id,
            user_id=user_id,
            before=1,
            after=1,
            max_tokens=1800,
        )
        item("evidence_hits", len(evidence))
        item("neighbor_sources", len(sources))
        print(preview(neighbor_context, 1200))

        section("6. Optional Evidence Answer")
        answer = await engram.answer_from_evidence(
            question="Which audit log and liability obligations matter?",
            context=context.text,
            reading="con",
        )
        print(answer or "Skipped because no LLM provider is configured.")

        section("7. Task Resume Context And Cleanup")
        task_context = await engram.build_context(
            task.task_run_id,
            query="resume legal review",
            max_tokens=2200,
        )
        print(preview(task_context.text, 1400))

        completed = await engram.complete_task(
            task.task_run_id,
            outcome="Long-input demo completed",
        )
        purged = await engram.purge(agent_id, user_id=user_id)
        item("completed_status", completed.status)
        item("purged_memories", purged)


if __name__ == "__main__":
    asyncio.run(main())
