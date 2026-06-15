#!/usr/bin/env python3
"""End-to-end Engram API tour.

Run:
    python examples/basic_usage.py

Local setup:
    docker compose up -d postgres
    export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
    export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
    export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2

This script keeps LLM-backed calls optional. If no LLM provider is configured,
conversation extraction and LLM answer generation are skipped or return an
empty answer, while the rest of the API still runs.
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


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def item(label: str, value: Any) -> None:
    print(f"{label:<24} {value}")


def preview(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


async def main() -> None:
    from engram import Engram

    run_id = uuid.uuid4().hex[:8]
    agent_id = f"api-tour-{run_id}"
    user_id = "sarah"

    async with Engram(memory_policy="coding_agent") as engram:
        section("1. Health And Setup")
        health = await engram.health_check()
        item("status", health.get("status"))
        item("agent_id", agent_id)
        item("llm_enabled", engram.llm is not None)

        section("2. Add, Batch Add, And Policy Metadata")
        allergy = await engram.add(
            "User is allergic to shellfish",
            agent_id,
            user_id=user_id,
            main_content="[USER]: Shellfish makes me sick.\n[AI]: I will avoid it.",
            metadata={"category": "health"},
        )
        repo_rule = await engram.add(
            "Repo constraint: never revert user changes without approval",
            agent_id,
            user_id=user_id,
        )
        batch = await engram.add_batch(
            [
                {
                    "content": "Project Atlas checkout rollback owner is Priya",
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "memory_type": "project",
                    "metadata": {
                        "project": "atlas_checkout",
                        "original_session_id": "demo-session-1",
                        "turn_index": 0,
                        "turn_role": "user",
                    },
                },
                {
                    "content": "Tool result: pytest tests/unit -q passed",
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "memory_type": "tool_result",
                    "metadata": {
                        "project": "atlas_checkout",
                        "original_session_id": "demo-session-1",
                        "turn_index": 1,
                        "turn_role": "tool",
                    },
                },
                {
                    "content": "Decision: use cached inventory reads for launch week",
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "memory_type": "decision",
                    "metadata": {
                        "project": "atlas_checkout",
                        "original_session_id": "demo-session-1",
                        "turn_index": 2,
                        "turn_role": "assistant",
                    },
                },
            ]
        )
        item("single_memory", allergy.memory_id)
        item("policy_type", repo_rule.memory_type)
        item("critical_slot", repo_rule.metadata.get("critical_slot"))
        item("batch_count", len(batch))

        section("3. Search Modes, Deep Search, Critical Recall, Trace")
        for mode in ("hybrid", "semantic", "keyword"):
            results = await engram.search(
                "rollback owner pytest launch",
                agent_id,
                user_id=user_id,
                mode=mode,
                limit=3,
            )
            item(f"{mode}_hits", [preview(r.memory.content, 50) for r in results])

        deep = await engram.deep_search(
            "What should the agent remember before launch?",
            agent_id,
            user_id=user_id,
            limit=5,
        )
        critical = await engram.recall_critical(
            agent_id,
            user_id=user_id,
            memory_types=["constraint", "project", "tool_result", "decision"],
        )
        trace = await engram.trace_recall(
            "launch checklist: rollback owner pytest repo rules",
            agent_id,
            user_id=user_id,
            expected_terms=["Priya", "pytest", "never revert"],
            max_tokens=1000,
        )
        item("deep_hits", len(deep))
        item("critical_hits", len(critical))
        item("trace_missing", trace.missing_expected_terms)
        print(trace.context)

        section("4. Context Block And Plain Memory Reads")
        block = await engram.get_context_block(
            "repo and launch context",
            agent_id,
            user_id=user_id,
            memory_types=["constraint", "project", "decision", "tool_result"],
            group_by_type=True,
            max_tokens=800,
        )
        print(block)
        filtered = await engram.get_memories(
            agent_id,
            user_id=user_id,
            metadata_filter={"project": "atlas_checkout"},
            limit=10,
        )
        item("filtered_memories", [m.memory_id for m in filtered])

        section("5. Get, Update, Reinforce, List Recent")
        fetched = await engram.get(allergy.memory_id)
        updated = await engram.update(
            fetched.memory_id,
            content="User is allergic to shellfish and cashews",
            metadata={"source": "manual_update"},
        )
        reinforced = await engram.reinforce(updated.memory_id, importance_boost=0.2)
        recent = await engram.list_recent(agent_id, user_id=user_id, limit=5)
        item("updated_fact", updated.content)
        item("importance", f"{reinforced.importance:.2f}")
        item("recent_count", len(recent))

        section("6. Graph Relations")
        await engram.relate(
            batch[0].memory_id,
            batch[2].memory_id,
            relation_type="supports",
            weight=0.8,
        )
        await engram.relate(
            repo_rule.memory_id,
            batch[2].memory_id,
            relation_type="supports",
            weight=0.7,
        )
        graph = await engram.traverse(
            batch[2].memory_id,
            max_depth=2,
            direction="inbound",
        )
        graph_many = await engram.traverse_many(
            [batch[0].memory_id, repo_rule.memory_id],
            direction="any",
            total_limit=10,
        )
        item("traverse_hits", len(graph))
        print(engram.render_graph_context(graph_many, max_tokens=500))

        section("7. Sessions")
        async with engram.session(agent_id, user_id=user_id) as session:
            session_memory = await engram.add(
                "User asked to keep launch answers concise",
                agent_id,
                user_id=user_id,
                session_id=session.session_id,
                memory_type="preference",
            )
            item("session_id", session.session_id)
            item("session_memory", session_memory.memory_id)

        section("8. Task Memory, Events, Checkpoints, Jobs")
        task = await engram.start_task(
            "Prepare Atlas checkout launch memory",
            agent_id,
            user_id=user_id,
            metadata={"example": "basic_usage"},
        )
        event = await engram.record_event(
            agent_id=agent_id,
            task_run_id=task.task_run_id,
            user_id=user_id,
            role="tool",
            event_type="tool_result",
            content="ruff check src tests: All checks passed",
            payload={"command": "ruff check src tests", "exit_code": 0},
        )
        events = await engram.record_turn(
            task.task_run_id,
            user_message="What remains before launch?",
            assistant_response="Confirm rollback owner and pytest status.",
            tool_calls=[{"name": "pytest", "result": "274 passed"}],
            artifacts=[{"path": "docs/api-reference.md", "type": "markdown"}],
        )
        checkpoint = await engram.create_checkpoint(
            task.task_run_id,
            "Launch memory has rollback owner, repo rule, and test result.",
            completed_steps=["Stored critical facts", "Recorded launch turn"],
            pending_steps=["Review final context"],
            source_event_ids=[event.event_id, *[e.event_id for e in events]],
        )
        jobs = await engram.process_memory_jobs(limit=10)
        task_context = await engram.build_context(
            task.task_run_id,
            query="resume launch work",
            max_tokens=1600,
        )
        listed_events = await engram.list_events(task_run_id=task.task_run_id, limit=10)
        item("task", task.task_run_id)
        item("checkpoint", checkpoint.checkpoint_id)
        item("events", len(listed_events))
        item("jobs", [job.status for job in jobs])
        print(preview(task_context.text, 1000))

        section("9. Long Input")
        long_text = """
        # Launch Review
        The rollback owner is Priya. The launch depends on pytest staying green.

        # Risk Notes
        Do not revert user changes without approval. Keep answers concise.
        """
        report = await engram.record_long_input(
            task.task_run_id,
            long_text,
            title="Launch review packet",
            extract_with_llm=False,
            max_chunk_tokens=120,
        )
        long_context = await engram.build_long_input_context(
            task.task_run_id,
            query="rollback owner pytest revert approval",
            expected_terms=["Priya", "pytest", "approval"],
            max_tokens=1800,
        )
        item("chunks", len(report.chunks))
        item("anchored_memories", len(report.memory_ids))
        item("long_missing", long_context.trace["missing_expected_terms"])

        section("10. Evidence Retrieval And Reading")
        # Evidence/aggregation reading is composed from public primitives:
        # high-recall retrieval, a rendered context block, then the LLM.
        question = "Who owns rollback and what test result matters?"
        evidence = await engram.deep_search(
            question,
            agent_id,
            user_id=user_id,
            limit=5,
        )
        evidence_context = await engram.get_context_block(
            question,
            agent_id,
            user_id=user_id,
            max_tokens=1200,
        )
        answer = ""
        if engram.llm is not None:
            answer = await engram.llm.complete(
                f"Context:\n{evidence_context}\n\n{question}\nAnswer concisely.",
            )
        item("evidence_hits", len(evidence))
        item("llm_answer", answer or "skipped because no LLM provider is configured")

        section("11. Optional LLM Conversation Extraction")
        if engram.llm is None:
            print("Skipping add_conversation(); no LLM provider is configured.")
        else:
            memories = await engram.add_conversation(
                user_message="I prefer launch updates as a short risk table.",
                assistant_response="I will format launch updates as a risk table.",
                agent_id=agent_id,
                user_id=user_id,
            )
            item("conversation_memories", len(memories))

        section("12. Redaction, Status, Forget, Cleanup")
        redacted = await engram.redact_event(event.event_id)
        paused = await engram.pause_task(task.task_run_id, outcome="Demo paused")
        resumable = await engram.list_tasks(
            agent_id=agent_id,
            user_id=user_id,
            status=["active", "paused"],
        )
        completed = await engram.complete_task(
            task.task_run_id,
            outcome="API tour completed",
        )
        deleted = await engram.forget(session_memory.memory_id)
        purged = await engram.purge(agent_id)
        item("redacted_at", redacted.redacted_at)
        item("paused_status", paused.status)
        item("resumable_tasks", len(resumable))
        item("completed_status", completed.status)
        item("forgot_session_memory", deleted)
        item("purged_memories", purged)


if __name__ == "__main__":
    asyncio.run(main())
