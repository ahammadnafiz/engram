#!/usr/bin/env python3
"""Real Engram-backed chatbot: on-device embeddings + Gemini for responses.

This is the optimized reference stack — free, on-device sentence-transformers
embeddings (`all-MiniLM-L6-v2`, 384-d) for retrieval and Google's
`gemini-3.5-flash` for answer generation, driven by the same composer rules
used in the Engram LongMemEval benchmark.

Run:
    export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
    export ENGRAM_GEMINI_API_KEY=...   # or GEMINI_API_KEY
    python examples/chatbot.py

The embedding/LLM stack defaults are applied below and can be overridden with
the standard ENGRAM_* environment variables (e.g. ENGRAM_LLM_MODEL). A bare
GEMINI_API_KEY (the google-genai convention) is mapped to ENGRAM_GEMINI_API_KEY
for convenience.

Non-interactive checks:
    python examples/chatbot.py --once "What do you remember about me?"
    python examples/chatbot.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from engram.core.exceptions import DatabaseConnectionError

gemini_api_key_alias = os.environ.get("GEMINI_API_KEY")
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
gemini_api_key_alias = os.environ.get("GEMINI_API_KEY") or gemini_api_key_alias

# Map the bare google-genai convention key onto Engram's namespaced variable.
if gemini_api_key_alias and "ENGRAM_GEMINI_API_KEY" not in os.environ:
    os.environ["ENGRAM_GEMINI_API_KEY"] = gemini_api_key_alias

# Optimized default stack: free on-device embeddings + Gemini for responses.
# setdefault keeps any value already supplied via the environment or .env.
os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("ENGRAM_EMBEDDING_DIMENSION", "384")
os.environ.setdefault("ENGRAM_LLM_PROVIDER", "gemini")
os.environ.setdefault("ENGRAM_LLM_MODEL", "gemini-3.1-flash-lite")

AGENT_ID = os.environ.get("ENGRAM_CHATBOT_AGENT_ID", "engram-chatbot")
USER_ID = os.environ.get("ENGRAM_CHATBOT_USER_ID", "default-user")
HISTORY_LIMIT = 10
RECALL_MODE = os.environ.get("ENGRAM_CHATBOT_RECALL_MODE", "operator").lower()
MEMORY_JOBS_MODE = os.environ.get("ENGRAM_CHATBOT_MEMORY_JOBS", "inline").lower()
MEMORY_JOBS_LIMIT = int(os.environ.get("ENGRAM_CHATBOT_MEMORY_JOBS_LIMIT", "10"))
RERANK_MODE = os.environ.get("ENGRAM_CHATBOT_RERANK", "auto").lower()
BROAD_MEMORY_LIMIT = int(os.environ.get("ENGRAM_CHATBOT_BROAD_MEMORY_LIMIT", "60"))
BROAD_MEMORY_CHARS = int(os.environ.get("ENGRAM_CHATBOT_BROAD_MEMORY_CHARS", "3600"))
MEMORY_HISTORY_LIMIT = int(os.environ.get("ENGRAM_CHATBOT_HISTORY_LIMIT", "30"))
VALID_RECALL_MODES = {"operator", "fast", "deep", "debug"}
VALID_MEMORY_JOBS_MODES = {"inline", "deferred"}
VALID_RERANK_MODES = {"auto", "true", "false"}
COLOR_ENABLED = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
)

SYSTEM_PROMPT = """You are a helpful assistant with persistent Engram memory. Today's date is {today}. Use the supplied memory and task context as the only source for remembered facts. If the context does not contain something, say that you do not have it in memory. When the user tells you a durable preference, profile fact, project detail, decision, or instruction, acknowledge it naturally; Engram will store it after the turn.

Answering rules (ported from the Engram LongMemEval benchmark composer):

- Compute when the data exists. If memory holds the numbers needed (ages, prices, dates to diff), do the arithmetic instead of refusing — even when the facts are scattered across different conversations. Compute every relative time expression relative to today's date above.

- Respect avoidances. If memory indicates the user wants to avoid something (allergy, dislike, hard constraint), your answer must NOT contain it — not as a primary, secondary, or fallback suggestion.

- Match the exact entity. Pay close attention to the specific entity/variant/role in the question. If the question asks about one variant and memory only mentions a DIFFERENT one (e.g. "electric" vs "acoustic guitar", "Sales Manager" vs "Senior Sales Engineer"), do not treat them as the same — say you do not have that information.

- Enumerate everything for list / aggregation / counting questions. List every relevant memory rather than the first few, and scan the full context twice because matching items are commonly scattered far apart. Count items in a single memory separately, and only count completed actions (past tense), not plans.

- Most recent wins. For conflicting values of the same fact, use the most recent memory; for "what was it before / originally" questions, use the older value. Memories about different people or contexts are not conflicting.

- ACTIVE vs SUPERSEDED. Memories tagged [ACTIVE] are the truth now; [SUPERSEDED] memories are old values that were overwritten. Never present a superseded value as the current answer — use it only for "before / originally" questions.

- Match context. Before using a memory's value, verify it applies to the SAME context as the question (a "while traveling" routine is not a regular weekday routine); prefer the more specific memory that matches.

When correcting an outdated or false value, name both the outdated value and the correct value. For "why" questions, include the supporting memory detail. Keep answers concise and directly useful. If <engram_query_specific_must_use> is non-empty, every line in it is mandatory for the current answer.

Reason step by step inside <mem_thinking>...</mem_thinking> tags first: list every relevant memory, do any counting / temporal / cross-topic computation, check avoidances and context, then state your conclusion. The user only sees text OUTSIDE the <mem_thinking> tags, so put your final answer there."""

_THINKING_CLOSE = "</mem_thinking>"


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(today=date.today().isoformat())


def strip_thinking(text: str) -> str:
    """Drop the hidden <mem_thinking> scratchpad, keeping the final answer."""
    lowered = text.lower()
    if _THINKING_CLOSE in lowered:
        return text[lowered.rfind(_THINKING_CLOSE) + len(_THINKING_CLOSE) :].strip()
    return text.strip()


def preview(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def format_timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(timespec="minutes")
        except TypeError:
            return value.isoformat()
    return str(value)


def paint(text: str, code: str) -> str:
    if not COLOR_ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(text: str) -> str:
    return paint(text, "2")


def bold(text: str) -> str:
    return paint(text, "1")


def accent(text: str) -> str:
    return paint(text, "36")


def good(text: str) -> str:
    return paint(text, "32")


def warn(text: str) -> str:
    return paint(text, "33")


def bad(text: str) -> str:
    return paint(text, "31")


def terminal_width() -> int:
    return max(72, min(104, shutil.get_terminal_size((88, 20)).columns))


def rule(label: str = "") -> str:
    width = terminal_width()
    if not label:
        return dim("-" * width)
    prefix = f" {label} "
    return dim(prefix + "-" * max(0, width - len(prefix)))


def print_header() -> None:
    print()
    print(rule("engram"))
    print(f"{bold('Engram Memory Chat')} | {dim('persistent Gemini-backed recall')}")
    print(dim("Type a message to chat, or /help for commands."))


def print_table(rows: list[tuple[str, Any]]) -> None:
    for key, value in rows:
        print(f"{dim(f'{key:<20}')} {value}")


def print_status_panel(rows: list[tuple[str, Any]]) -> None:
    print_header()
    print(rule("session"))
    print_table(rows)
    print(rule())


def print_notice(message: str, *, level: str = "info") -> None:
    prefix = {
        "info": accent("engram"),
        "ok": good("engram"),
        "warn": warn("engram"),
        "error": bad("error"),
    }.get(level, accent("engram"))
    print(f"{prefix} {dim(message)}")


def print_response(text: str) -> None:
    width = terminal_width() - 4
    print()
    print(f"{accent('assistant')} {dim('-' * max(1, terminal_width() - 10))}")
    for raw_line in text.splitlines() or [""]:
        if not raw_line.strip():
            print()
            continue
        if raw_line.lstrip().startswith(("-", "*")):
            print(f"  {raw_line}")
            continue
        print(
            textwrap.fill(
                raw_line,
                width=width,
                initial_indent="  ",
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        )


def prompt_text() -> str:
    return f"{accent('you')} {dim('> ')}" if COLOR_ENABLED else "you> "


def require_real_config() -> None:
    if RECALL_MODE not in VALID_RECALL_MODES:
        raise SystemExit(
            "Invalid ENGRAM_CHATBOT_RECALL_MODE. Use 'operator', 'fast', "
            "'deep', or 'debug'."
        )
    if MEMORY_JOBS_MODE not in VALID_MEMORY_JOBS_MODES:
        raise SystemExit(
            "Invalid ENGRAM_CHATBOT_MEMORY_JOBS. Use 'inline' or 'deferred'."
        )
    if RERANK_MODE not in VALID_RERANK_MODES:
        raise SystemExit(
            "Invalid ENGRAM_CHATBOT_RERANK. Use 'auto', 'true', or 'false'."
        )

    if not os.environ.get("ENGRAM_GEMINI_API_KEY"):
        raise SystemExit(
            "Missing required environment variable: ENGRAM_GEMINI_API_KEY "
            "(or GEMINI_API_KEY).\nGet a key at https://aistudio.google.com/apikey "
            "and set it before running the chatbot."
        )


class MemoryChatbot:
    def __init__(self) -> None:
        self.engram = None
        self.task_id: str | None = None
        self.session_id: str | None = None
        self._session_context: Any | None = None
        self.history: list[dict[str, str]] = []

    async def connect(self) -> None:
        require_real_config()

        from engram import Engram

        self.engram = Engram(memory_policy="default")
        await self.engram.connect()
        if self.engram.llm is None:
            raise RuntimeError(
                "LLM provider is disabled. Set ENGRAM_LLM_PROVIDER=gemini and "
                "ENGRAM_GEMINI_API_KEY before running examples/chatbot.py."
            )

        await self._resume_or_start_task()
        health = await self.engram.health_check()
        health_status = str(health.get("status"))
        print_status_panel(
            [
                (
                    "health",
                    good(health_status)
                    if health_status == "healthy"
                    else warn(health_status),
                ),
                ("agent", AGENT_ID),
                ("user", USER_ID),
                ("task", self.task_id),
                ("session", self.session_id),
                (
                    "embedding_provider",
                    os.environ.get(
                        "ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers"
                    ),
                ),
                (
                    "embedding_model",
                    os.environ.get("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                ),
                ("embedding_dim", os.environ.get("ENGRAM_EMBEDDING_DIMENSION", "384")),
                ("llm_provider", os.environ.get("ENGRAM_LLM_PROVIDER", "gemini")),
                ("llm_model", os.environ.get("ENGRAM_LLM_MODEL", "gemini-3.5-flash")),
                ("recall_mode", RECALL_MODE),
                ("memory_jobs", MEMORY_JOBS_MODE),
                ("rerank", self._rerank_enabled()),
                ("broad_memory_limit", BROAD_MEMORY_LIMIT),
            ]
        )
        if MEMORY_JOBS_MODE == "inline":
            await self.process_jobs(show_empty=False, reason="from earlier runs")

    async def _resume_or_start_task(self) -> None:
        assert self.engram is not None

        tasks = await self.engram.list_tasks(
            agent_id=AGENT_ID,
            user_id=USER_ID,
            status=["active", "paused"],
            limit=1,
        )
        if tasks:
            task = tasks[0]
            self.task_id = task.task_run_id
            self.session_id = task.session_id
            return

        self._session_context = self.engram.session(
            AGENT_ID,
            user_id=USER_ID,
            metadata={"application": "examples/chatbot.py"},
        )
        session = await self._session_context.__aenter__()
        self.session_id = session.session_id
        task = await self.engram.start_task(
            "Run a real Gemini-backed chatbot with persistent Engram memory",
            AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            metadata={"application": "examples/chatbot.py"},
        )
        self.task_id = task.task_run_id

    async def close(self) -> None:
        if self.engram is None:
            return
        if self.task_id is not None:
            with contextlib.suppress(Exception):
                await self.engram.pause_task(
                    self.task_id,
                    outcome="Chatbot process exited; task can be resumed later.",
                )
        if self._session_context is not None:
            with contextlib.suppress(Exception):
                await self._session_context.__aexit__(None, None, None)
        await self.engram.close()

    async def reply(self, message: str) -> str:
        assert self.engram is not None
        assert self.task_id is not None

        if RECALL_MODE == "operator":
            response, recall_metadata, llm_model = await self._operator_reply(message)
        else:
            messages, recall_metadata = await self._build_prompt_messages(message)
            llm_response = await self.engram.llm.complete_full(
                messages,
                max_tokens=1200,
                temperature=0.4,
            )
            response = llm_response.content.strip()
            llm_model = llm_response.model

        # Drop the hidden <mem_thinking> scratchpad before storing/printing.
        response = strip_thinking(response)

        metadata = {
            "application": "examples/chatbot.py",
            "llm_model": llm_model,
            "recall_mode": RECALL_MODE,
        }
        metadata.update(recall_metadata)

        await self.engram.record_turn(
            self.task_id,
            message,
            response,
            agent_id=AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            metadata=metadata,
        )
        jobs = await self._process_memory_jobs()
        self._print_memory_job_summary(jobs)

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        self.history = self.history[-HISTORY_LIMIT:]

        return response

    async def _build_prompt_messages(
        self,
        message: str,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        if RECALL_MODE == "fast":
            return await self._build_fast_prompt_messages(message)
        return await self._build_deep_prompt_messages(
            message,
            include_trace=RECALL_MODE == "debug",
        )

    async def _operator_reply(
        self,
        message: str,
    ) -> tuple[str, dict[str, Any], str]:
        """Route with recall, then answer with the chatbot LLM prompt.

        recall() classifies the message's intent (current / historical / event
        / lineage / chat) and retrieves from the matching surface(s), but the
        chatbot owns final answer generation. That lets broad history/update
        questions use the timeline block instead of returning recall's narrow
        "no matching memory" prose.
        """
        assert self.engram is not None

        answer = await self.engram.recall(
            message,
            AGENT_ID,
            user_id=USER_ID,
            compose_answer=False,
        )
        metadata = {
            "recall_intent": answer.intent,
            "recall_previous_count": len(answer.previous),
            "recall_event_count": len(answer.events),
            "recall_source_count": len(answer.sources),
            "operator_route": "chat" if answer.intent == "chat" else "recall_chat",
        }

        messages, chat_metadata = await self._build_operator_prompt_messages(
            message,
            answer,
        )
        llm_response = await self.engram.llm.complete_full(
            messages,
            max_tokens=1200,
            temperature=0.4,
        )
        metadata.update(chat_metadata)
        return llm_response.content.strip(), metadata, llm_response.model

    async def _build_operator_prompt_messages(
        self,
        message: str,
        answer: Any,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        assert self.engram is not None

        messages, metadata = await self._build_fast_prompt_messages(message)
        recall_block = self._render_recall_evidence_block(answer)
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    f'<engram_recall_evidence intent="{answer.intent}">\n'
                    f"{recall_block}\n"
                    "</engram_recall_evidence>"
                ),
            },
        )
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Use the Engram recall evidence, memory context, and memory "
                    "history as source material. For before/after, changed, "
                    "updated, revised, added, deleted, or conversation-history "
                    "questions, prefer the memory history timeline. If recall "
                    "evidence is empty but the history timeline has matching "
                    "entries, answer from the history timeline instead of saying "
                    "there is no memory."
                ),
            },
        )
        return messages, metadata

    def _render_recall_evidence_block(self, answer: Any) -> str:
        lines = []
        trace = getattr(answer, "trace", {}) or {}
        topic = trace.get("topic")
        if topic:
            lines.append(f"topic: {topic}")
        if answer.current is not None:
            lines.append(f"current: {answer.current.fact or answer.current.content}")
        for mem in answer.previous:
            when = mem.superseded_at or mem.valid_to or mem.created_at
            stamp = when.date().isoformat() if when else "unknown"
            lines.append(f"previous until {stamp}: {mem.fact or mem.content}")
        for mem in answer.evidence:
            lines.append(f"memory [{mem.status}]: {mem.fact or mem.content}")
        for event in answer.events[:8]:
            stamp = (
                format_timestamp(event.created_at) if event.created_at else "unknown"
            )
            lines.append(f"event [{event.role} {stamp}]: {event.content}")
        if answer.conflict_note:
            lines.append(f"conflict: {answer.conflict_note}")
        return "\n".join(lines) if lines else "No direct recall evidence."

    async def _build_fast_prompt_messages(
        self,
        message: str,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        assert self.engram is not None

        critical_memories = await self.engram.recall_critical(
            AGENT_ID,
            user_id=USER_ID,
            limit=12,
        )
        critical_block = self._render_memories_block(critical_memories, max_chars=1200)
        memory_block = await self.engram.get_context_block(
            message,
            AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            limit=8,
            max_tokens=900,
            group_by_type=True,
            rerank=self._rerank_enabled(),
        )
        history_block, history_count = await self._build_history_block()
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "system",
                "content": (
                    "<engram_critical_memory>\n"
                    f"{critical_block}\n"
                    "</engram_critical_memory>"
                ),
            },
            {
                "role": "system",
                "content": (
                    f"<engram_memory_context>\n{memory_block}\n</engram_memory_context>"
                ),
            },
            {
                "role": "system",
                "content": (
                    f"<engram_memory_history>\n{history_block}\n"
                    "</engram_memory_history>"
                ),
            },
            *self.history[-HISTORY_LIMIT:],
            {"role": "user", "content": message},
        ]
        return messages, {
            "critical_memory_count": len(critical_memories),
            "memory_history_count": history_count,
        }

    async def _build_deep_prompt_messages(
        self,
        message: str,
        *,
        include_trace: bool,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        assert self.engram is not None
        assert self.task_id is not None

        trace = None
        if include_trace:
            trace = await self.engram.trace_recall(
                message,
                AGENT_ID,
                user_id=USER_ID,
                limit=12,
                max_tokens=1400,
                expected_terms=self._expected_terms(message),
            )
        memory_block = await self.engram.get_context_block(
            message,
            AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            limit=10,
            max_tokens=1000,
            group_by_type=True,
            rerank=self._rerank_enabled(),
        )
        deep_hits = await self.engram.deep_search(
            message,
            AGENT_ID,
            user_id=USER_ID,
            limit=16,
            n_queries=4,
            rerank=self._rerank_enabled(),
        )
        deep_memory_block = self._render_deep_memory_block(deep_hits)
        broad_memories = await self.engram.list_recent(
            AGENT_ID,
            user_id=USER_ID,
            limit=BROAD_MEMORY_LIMIT,
        )
        broad_memory_block = self._render_memories_block(
            broad_memories,
            max_chars=BROAD_MEMORY_CHARS,
        )
        attention_memory_block = self._render_attention_memory_block(broad_memories)
        query_attention_block = self._render_query_attention_memory_block(
            message,
            broad_memories,
        )
        history_block, history_count = await self._build_history_block()
        task_context = await self.engram.build_context(
            self.task_id,
            query=message,
            max_tokens=1600,
            recent_event_limit=10,
            memory_limit=10,
            checkpoint_limit=2,
            include_graph=True,
        )

        metadata: dict[str, Any] = {"memory_history_count": history_count}
        trace_context = ""
        if trace is not None:
            trace_context = trace.context
            metadata.update(
                {
                    "trace_kept_memory_ids": trace.kept_memory_ids,
                    "missing_expected_terms": trace.missing_expected_terms,
                }
            )

        user_content = message
        if query_attention_block:
            user_content = (
                "Mandatory Engram memories for this question. "
                "Every line below matches the current question; include every "
                "line in your answer:\n"
                f"{query_attention_block}\n\n"
                f"User question: {message}"
            )

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "system",
                "content": (
                    '<engram_query_specific_must_use priority="highest">\n'
                    f"{query_attention_block}\n"
                    "</engram_query_specific_must_use>"
                ),
            },
            {
                "role": "system",
                "content": f"<engram_memory_context>\n{memory_block}\n</engram_memory_context>",
            },
        ]
        if trace_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "<engram_recall_trace>\n"
                        f"{trace_context}\n"
                        "</engram_recall_trace>"
                    ),
                }
            )
        messages.extend(
            [
                {
                    "role": "system",
                    "content": f"<engram_deep_memory>\n{deep_memory_block}\n</engram_deep_memory>",
                },
                {
                    "role": "system",
                    "content": f"<engram_recent_memory_safety_net>\n{broad_memory_block}\n</engram_recent_memory_safety_net>",
                },
                {
                    "role": "system",
                    "content": f"<engram_attention_memory>\n{attention_memory_block}\n</engram_attention_memory>",
                },
                {
                    "role": "system",
                    "content": (
                        f"<engram_memory_history>\n{history_block}\n"
                        "</engram_memory_history>"
                    ),
                },
                {
                    "role": "system",
                    "content": f"<engram_task_context>\n{task_context.text}\n</engram_task_context>",
                },
                *self.history[-HISTORY_LIMIT:],
                {"role": "user", "content": user_content},
            ]
        )
        return messages, metadata

    async def _process_memory_jobs(self) -> list[Any]:
        assert self.engram is not None

        if MEMORY_JOBS_MODE == "inline":
            return await self.engram.process_memory_jobs(limit=MEMORY_JOBS_LIMIT)
        print_notice(
            "memory job queued; run /jobs, process_memory_jobs(), or "
            "ENGRAM_CHATBOT_MEMORY_JOBS=inline"
        )
        return []

    def _print_memory_job_summary(self, jobs: list[Any], *, reason: str = "") -> None:
        completed = len([job for job in jobs if job.status == "completed"])
        failed = len([job for job in jobs if job.status == "failed"])
        suffix = f" {reason}" if reason else ""
        if completed:
            print_notice(f"processed {completed} memory job(s){suffix}", level="ok")
        if failed:
            print_notice(f"{failed} memory job(s) failed{suffix}", level="error")

    def _rerank_enabled(self) -> bool:
        if RERANK_MODE == "true":
            return True
        if RERANK_MODE == "false":
            return False
        return RECALL_MODE in {"deep", "debug"}

    def _render_deep_memory_block(
        self, results: list[Any], max_chars: int = 2400
    ) -> str:
        lines = []
        used = 0
        seen = set()
        for result in results:
            memory = result.memory
            if memory.memory_id in seen:
                continue
            seen.add(memory.memory_id)
            line = f"- [{memory.memory_type}] {memory.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _render_memories_block(self, memories: list[Any], max_chars: int) -> str:
        lines = []
        used = 0
        for memory in memories:
            line = f"- [{memory.memory_type}] {memory.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    async def _build_history_block(self) -> tuple[str, int]:
        assert self.engram is not None

        events = await self.engram.get_history(
            AGENT_ID,
            user_id=USER_ID,
            limit=MEMORY_HISTORY_LIMIT,
            include_superseded=True,
        )
        if not events:
            return "No memory history is available.", 0

        return self._render_history_block(events), len(events)

    def _render_history_block(self, events: list[Any], max_chars: int = 3200) -> str:
        header = (
            "Use this block only for questions about previous values, updates, "
            "adds, changes, revisions, or history. Active memories are current. "
            "Superseded memories are old values; do not treat them as current."
        )
        lines = [header]
        used = len(header)
        for event in events:
            memory = event.memory
            relation = ""
            if event.event_type == "revised" and event.previous_memory_id:
                relation = f" previous={event.previous_memory_id[:12]}"
            elif event.event_type == "superseded" and event.superseded_by_memory_id:
                relation = f" superseded_by={event.superseded_by_memory_id[:12]}"
            reason = f" reason={event.reason}" if event.reason else ""
            line = (
                f"- {event.event_type} at {format_timestamp(event.occurred_at)} "
                f"[{memory.memory_type} r{memory.revision} {memory.status}] "
                f"{memory.content}{relation}{reason}"
            )
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _render_attention_memory_block(
        self,
        memories: list[Any],
        max_chars: int = 1800,
    ) -> str:
        keywords = (
            "avoid",
            "allerg",
            "no longer",
            "superseded",
            "cancel",
            "instead",
            "not ",
            "must",
            "before",
            "owner",
            "threshold",
        )
        lines = []
        used = 0
        for memory in memories:
            content_lower = memory.content.lower()
            if not any(keyword in content_lower for keyword in keywords):
                continue
            line = f"- [{memory.memory_type}] {memory.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _render_query_attention_memory_block(
        self,
        message: str,
        memories: list[Any],
        max_chars: int = 1600,
    ) -> str:
        lowered = message.lower()
        keyword_groups: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
            (
                ("dinner", "restaurant", "food", "meal", "thai", "coffee"),
                ("shellfish", "allerg", "vegetarian", "quiet", "live music", "avoid"),
            ),
            (
                (
                    "older",
                    "old plan",
                    "superseded",
                    "cancelled",
                    "canceled",
                    "replaced",
                ),
                ("superseded", "cancel", "no longer"),
            ),
            (
                ("owner", "approval", "threshold", "safety rule", "launch"),
                ("owner", "must", "before", "threshold", "latency", "p95"),
            ),
        ]
        needles: set[str] = set()
        for triggers, terms in keyword_groups:
            if any(trigger in lowered for trigger in triggers):
                needles.update(terms)
        if not needles:
            return ""

        lines = []
        used = 0
        for memory in memories:
            content_lower = memory.content.lower()
            if not any(term in content_lower for term in needles):
                continue
            line = f"- [{memory.memory_type}] {memory.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _expected_terms(self, message: str) -> list[str]:
        terms = []
        lowered = message.lower()
        for raw in lowered.replace("\n", " ").split():
            word = raw.strip(".,!?;:()[]{}\"'")
            if len(word) >= 5 and word not in {"about", "there", "would", "could"}:
                terms.append(word)
            if len(terms) == 6:
                break
        return terms

    async def remember(self, text: str) -> None:
        assert self.engram is not None
        memory = await self.engram.add(
            text,
            AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            memory_type="semantic",
            metadata={"source": "manual_chatbot_memory"},
        )
        print(rule("stored"))
        print_table(
            [
                ("memory_id", memory.memory_id),
                ("lineage_id", memory.lineage_id),
                ("revision", memory.revision),
                ("status", memory.status),
                ("content", preview(memory.content)),
            ]
        )

    async def memories(self) -> None:
        assert self.engram is not None
        memories = await self.engram.list_recent(AGENT_ID, user_id=USER_ID, limit=15)
        if not memories:
            print_notice("no memories stored yet", level="warn")
            return
        print(rule("memories"))
        for memory in memories:
            print(
                f"  {accent(memory.memory_id[:16])} "
                f"{dim(f'[{memory.memory_type} r{memory.revision} {memory.status} {memory.importance:.2f}]')} "
                f"{preview(memory.content, 120)}"
            )

    async def process_jobs(self, *, show_empty: bool = True, reason: str = "") -> None:
        assert self.engram is not None
        jobs = await self.engram.process_memory_jobs(limit=MEMORY_JOBS_LIMIT)
        if jobs:
            self._print_memory_job_summary(jobs, reason=reason)
        elif show_empty:
            print_notice("no queued memory jobs", level="ok")

    async def show_history(self, rest: str = "") -> None:
        assert self.engram is not None
        arg = rest.strip()
        if arg.startswith("mem_"):
            await self.lineage(arg)
            return

        include_superseded = True
        limit = 20
        if arg == "active":
            include_superseded = False
        elif arg.isdigit():
            limit = max(1, min(100, int(arg)))
        elif arg:
            print_notice("usage: /history [active|limit|memory_id]", level="warn")
            return

        events = await self.engram.get_history(
            AGENT_ID,
            user_id=USER_ID,
            limit=limit,
            include_superseded=include_superseded,
        )
        if not events:
            print_notice("no memory history yet", level="warn")
            return

        print(rule("history"))
        for event in events:
            marker = {
                "added": "+",
                "revised": "~",
                "superseded": "x",
            }.get(event.event_type, "?")
            relation = ""
            if event.event_type == "revised" and event.previous_memory_id:
                relation = f" prev={event.previous_memory_id[:12]}"
            elif event.event_type == "superseded" and event.superseded_by_memory_id:
                relation = f" by={event.superseded_by_memory_id[:12]}"
            print(
                f"  {accent(marker)} {accent(event.event_type.ljust(10))} "
                f"{dim(format_timestamp(event.occurred_at))} "
                f"{accent(event.memory.memory_id[:16])} "
                f"{dim(f'[{event.memory.memory_type} r{event.memory.revision} {event.memory.status}]')} "
                f"{preview(event.memory.content, 110)}{dim(relation)}"
            )
            if event.reason:
                print(f"    {dim('reason')} {event.reason}")

    async def revise(self, memory_id: str, content: str) -> None:
        assert self.engram is not None
        memory = await self.engram.revise(
            memory_id,
            content=content,
            metadata={"source": "manual_chatbot_revision"},
            reason="manual_chatbot_revision",
        )
        current = await self.engram.get_current(memory_id)
        lineage = await self.engram.get_lineage(memory_id)
        print(rule("revised"))
        print_table(
            [
                ("new_memory_id", memory.memory_id),
                ("current", current.memory_id),
                ("lineage_id", lineage.lineage_id),
                ("revisions", len(lineage.memories)),
                ("content", preview(memory.content)),
            ]
        )

    async def lineage(self, memory_id: str) -> None:
        assert self.engram is not None
        current = await self.engram.get_current(memory_id)
        lineage = await self.engram.get_lineage(memory_id)
        explanation = await self.engram.explain_memory(memory_id)
        print(rule("lineage"))
        print_table(
            [
                ("lineage_id", lineage.lineage_id),
                ("current", current.memory_id),
                ("selected", explanation.memory.memory_id),
                ("selected_status", explanation.memory.status),
                (
                    "superseded_by",
                    explanation.superseded_by.memory_id
                    if explanation.superseded_by
                    else None,
                ),
                ("supersedes", [m.memory_id for m in explanation.supersedes]),
            ]
        )
        print(rule("revisions"))
        for memory in lineage.memories:
            marker = "*" if memory.memory_id == current.memory_id else " "
            print(
                f"{marker} {accent(memory.memory_id[:16])} "
                f"{dim(f'r{memory.revision} {memory.status}')} "
                f"{preview(memory.content, 120)}"
            )

    async def search(self, query: str) -> None:
        assert self.engram is not None
        results = await self.engram.search(
            query,
            AGENT_ID,
            user_id=USER_ID,
            limit=8,
            mode="hybrid",
            include_superseded=True,
        )
        if not results:
            print_notice("no matches", level="warn")
            return
        print(rule("search"))
        for result in results:
            await self.engram.reinforce(result.memory.memory_id, 0.02)
            print(
                f"  {accent(f'{result.score:.3f}')} "
                f"{dim(result.memory.memory_type)} "
                f"{preview(result.memory.content)}"
            )

    async def recall(self, question: str) -> None:
        assert self.engram is not None
        answer = await self.engram.recall(question, AGENT_ID, user_id=USER_ID)
        print(rule(f"recall [{answer.intent}]"))
        print(f"  {answer.answer_text}")
        if answer.previous:
            prev = "; ".join(preview(m.fact or m.content, 60) for m in answer.previous)
            print(f"  {dim('previously:')} {prev}")
        if answer.when_changed:
            print(f"  {dim('changed:')} {answer.when_changed.date().isoformat()}")
        if answer.conflict_note:
            print(f"  {warn('conflict:')} {answer.conflict_note}")

    async def context(self, query: str) -> None:
        assert self.engram is not None
        assert self.task_id is not None
        block = await self.engram.get_context_block(
            query,
            AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            group_by_type=True,
            max_tokens=1200,
        )
        task_context = await self.engram.build_context(
            self.task_id,
            query=query,
            max_tokens=1400,
            include_graph=True,
        )
        print(rule("memory context"))
        print(block or "No memory context.")
        print()
        print(rule("task context"))
        print(task_context.text or "No task context.")

    async def trace(self, query: str) -> None:
        assert self.engram is not None
        trace = await self.engram.trace_recall(
            query,
            AGENT_ID,
            user_id=USER_ID,
            expected_terms=self._expected_terms(query),
            max_tokens=1200,
        )
        print(rule("trace"))
        print(trace.context or "No recall context.")
        print_table(
            [
                ("critical", len(trace.critical_memory_ids)),
                ("search", len(trace.search_memory_ids)),
                ("kept", len(trace.kept_memory_ids)),
                ("trimmed", len(trace.trimmed_memory_ids)),
                ("missing_terms", trace.missing_expected_terms),
            ]
        )

    async def task(self) -> None:
        assert self.engram is not None
        assert self.task_id is not None
        current = await self.engram.get_task(self.task_id)
        tasks = await self.engram.list_tasks(
            agent_id=AGENT_ID,
            user_id=USER_ID,
            status=["active", "paused"],
            limit=5,
        )
        print(rule("task"))
        print_table(
            [
                ("current", f"{current.task_run_id} ({current.status})"),
                ("session", current.session_id),
                ("resumable", [task.task_run_id for task in tasks]),
            ]
        )

    async def forget(self, memory_id: str) -> None:
        assert self.engram is not None
        deleted = await self.engram.forget(memory_id)
        print(rule("forget"))
        print_table([("deleted", deleted)])

    async def clear(self) -> None:
        assert self.engram is not None
        count = await self.engram.purge(AGENT_ID, user_id=USER_ID)
        self.history.clear()
        print(rule("clear"))
        print_table([("purged", count)])


COMMANDS = [
    ("/remember <fact>", "store a durable fact immediately"),
    ("/revise <memory_id> <fact>", "create a new active revision"),
    ("/lineage <memory_id>", "show current head and revision history"),
    ("/history [active|limit|memory_id]", "show memory add/update timeline"),
    ("/memories", "list recent Engram memories"),
    ("/jobs", "process queued memory extraction jobs now"),
    ("/search <query>", "search memories and reinforce hits"),
    ("/recall <question>", "ask memory: current/historical/event/lineage answer"),
    ("/context <query>", "show memory and task prompt context"),
    ("/trace <query>", "inspect trace_recall decisions"),
    ("/task", "show the resumable task and session"),
    ("/forget <memory_id>", "delete one memory"),
    ("/clear", "purge this chatbot user's memories"),
    ("/help", "show this command palette"),
    ("/quit", "exit"),
]


def print_help() -> None:
    print()
    print(rule("commands"))
    width = max(len(command) for command, _ in COMMANDS)
    for command, description in COMMANDS:
        print(f"  {accent(command.ljust(width))}  {dim(description)}")
    print()
    print(
        dim(
            "Plain text runs: recall router -> Gemini chat when needed -> "
            "record_turn -> memory jobs."
        )
    )
    print(rule())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real Gemini-backed Engram memory chatbot."
    )
    parser.add_argument(
        "--once",
        metavar="MESSAGE",
        help="send one message, print the response, and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run one deterministic memory-backed demo turn and exit",
    )
    return parser.parse_args(argv)


async def run_command(bot: MemoryChatbot, line: str) -> bool:
    command, _, rest = line.partition(" ")
    command = command.lower()
    rest = rest.strip()

    if command in {"/quit", "/q"}:
        return False
    if command == "/help":
        print_help()
    elif command == "/remember" and rest:
        await bot.remember(rest)
    elif command == "/revise" and rest:
        memory_id, _, content = rest.partition(" ")
        if memory_id and content.strip():
            await bot.revise(memory_id, content.strip())
        else:
            print_notice("usage: /revise <memory_id> <fact>", level="warn")
    elif command == "/lineage" and rest:
        await bot.lineage(rest)
    elif command == "/history":
        await bot.show_history(rest)
    elif command == "/memories":
        await bot.memories()
    elif command == "/jobs":
        await bot.process_jobs()
    elif command == "/search" and rest:
        await bot.search(rest)
    elif command == "/recall" and rest:
        await bot.recall(rest)
    elif command == "/context" and rest:
        await bot.context(rest)
    elif command == "/trace" and rest:
        await bot.trace(rest)
    elif command == "/task":
        await bot.task()
    elif command == "/forget" and rest:
        await bot.forget(rest)
    elif command == "/clear":
        clear_prompt = (
            warn("clear") + " " + dim("Delete this user's chatbot memories? y/N: ")
        )
        confirm = await asyncio.to_thread(
            input,
            clear_prompt,
        )
        if confirm.lower() == "y":
            await bot.clear()
    else:
        print_notice("unknown or incomplete command, type /help", level="warn")
    return True


async def run_once(bot: MemoryChatbot, message: str) -> None:
    print(rule("once"))
    print_table([("message", preview(message, 100))])
    response = await bot.reply(message)
    print_response(response)


async def run_demo(bot: MemoryChatbot) -> None:
    assert bot.engram is not None
    old = await bot.engram.add(
        "The user's live chatbot demo city is Dhaka.",
        AGENT_ID,
        user_id=USER_ID,
        session_id=bot.session_id,
        memory_type="profile",
        metadata={
            "source": "examples/chatbot.py --demo",
            "conflict_key": f"{AGENT_ID}:{USER_ID}:demo:city",
        },
    )
    new = await bot.engram.revise(
        old.memory_id,
        content="The user's live chatbot demo city is Singapore.",
        metadata={"source": "examples/chatbot.py --demo"},
        reason="demo_correction",
    )
    current = await bot.engram.get_current(old.memory_id)
    lineage = await bot.engram.get_lineage(old.memory_id)
    explanation = await bot.engram.explain_memory(old.memory_id)
    print(rule("demo lineage"))
    print_table(
        [
            ("old_memory_id", old.memory_id),
            ("new_memory_id", new.memory_id),
            ("lineage_id", lineage.lineage_id),
            ("current", current.memory_id),
            ("old_status", explanation.memory.status),
            (
                "old_superseded_by",
                explanation.superseded_by.memory_id
                if explanation.superseded_by
                else None,
            ),
            (
                "revisions",
                [f"r{memory.revision}:{memory.status}" for memory in lineage.memories],
            ),
        ]
    )
    await run_once(
        bot,
        "Using Engram memory only, what is my live chatbot demo city?",
    )


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bot = MemoryChatbot()
    try:
        await bot.connect()
    except DatabaseConnectionError as exc:
        print_notice(f"database connection failed: {exc}", level="error")
        print_notice(
            "Check that ENGRAM_DATABASE_URL matches the running Postgres "
            "container credentials. If you used docker-setup.sh, the password "
            "in .env may differ from an older existing Docker volume.",
            level="warn",
        )
        return
    try:
        if args.demo:
            await run_demo(bot)
            return
        if args.once:
            await run_once(bot, args.once)
            return

        print_help()
        while True:
            try:
                line = (await asyncio.to_thread(input, prompt_text())).strip()
            except EOFError:
                break
            if not sys.stdin.isatty():
                print()
            if not line:
                continue
            if line.startswith("/"):
                if not await run_command(bot, line):
                    break
            else:
                if COLOR_ENABLED:
                    print_notice("recalling memory and calling Gemini")
                response = await bot.reply(line)
                print_response(response)
    finally:
        await bot.close()
        if not sys.stdin.isatty():
            print()
        print_notice("bye", level="ok")


if __name__ == "__main__":
    asyncio.run(main())
