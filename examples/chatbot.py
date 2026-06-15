#!/usr/bin/env python3
"""Real Engram-backed chatbot using OpenAI embeddings and chat completions.

Run:
    export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
    export ENGRAM_EMBEDDING_PROVIDER=openai
    export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
    export ENGRAM_EMBEDDING_DIMENSION=1536
    export ENGRAM_LLM_PROVIDER=openai
    export ENGRAM_LLM_MODEL=gpt-4o-mini
    export ENGRAM_OPENAI_API_KEY=sk-...
    python examples/chatbot.py

If you have EMBEDDING_PROVIDER=openai from an older shell snippet, this script
maps it to ENGRAM_EMBEDDING_PROVIDER for convenience.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from engram.core.exceptions import DatabaseConnectionError

embedding_provider_alias = os.environ.get("EMBEDDING_PROVIDER")
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

if embedding_provider_alias:
    os.environ["ENGRAM_EMBEDDING_PROVIDER"] = embedding_provider_alias
if os.environ.get("ENGRAM_OPENAI_API_KEY") and "ENGRAM_LLM_PROVIDER" not in os.environ:
    os.environ["ENGRAM_LLM_PROVIDER"] = "openai"

AGENT_ID = os.environ.get("ENGRAM_CHATBOT_AGENT_ID", "engram-chatbot")
USER_ID = os.environ.get("ENGRAM_CHATBOT_USER_ID", "default-user")
HISTORY_LIMIT = 10
RECALL_MODE = os.environ.get("ENGRAM_CHATBOT_RECALL_MODE", "fast").lower()
MEMORY_JOBS_MODE = os.environ.get("ENGRAM_CHATBOT_MEMORY_JOBS", "deferred").lower()
RERANK_MODE = os.environ.get("ENGRAM_CHATBOT_RERANK", "auto").lower()
BROAD_MEMORY_LIMIT = int(os.environ.get("ENGRAM_CHATBOT_BROAD_MEMORY_LIMIT", "60"))
BROAD_MEMORY_CHARS = int(os.environ.get("ENGRAM_CHATBOT_BROAD_MEMORY_CHARS", "3600"))
VALID_RECALL_MODES = {"fast", "deep", "debug"}
VALID_MEMORY_JOBS_MODES = {"inline", "deferred"}
VALID_RERANK_MODES = {"auto", "true", "false"}
COLOR_ENABLED = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
)

SYSTEM_PROMPT = """You are a helpful assistant with persistent Engram memory.
Use the supplied memory and task context as the only source for remembered facts.
If the context does not contain something, say that you do not have it in memory.
When the user tells you a durable preference, profile fact, project detail,
decision, or instruction, acknowledge it naturally; Engram will store it after
the turn. Answer every part of the user's question. For "why" questions, include
the supporting memory detail. When correcting an outdated or false value, name
both the outdated value and the correct value. Keep answers concise and directly
useful.

For list or aggregation questions, enumerate every relevant matching memory in
the supplied context instead of choosing only the first examples. For planning
food, restaurants, travel, or meetings, include any remembered allergies,
avoidances, hard constraints, and preferences that apply. If the user asks for
an owner, approval, threshold, date, document, or other named field, include that
field explicitly when it exists in memory. If
<engram_query_specific_must_use> is non-empty, every line in it is mandatory for
the current answer."""


def preview(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


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
    print(f"{bold('Engram Memory Chat')} | {dim('persistent OpenAI-backed recall')}")
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


def require_real_openai_config() -> None:
    missing = []
    if not os.environ.get("ENGRAM_OPENAI_API_KEY"):
        missing.append("ENGRAM_OPENAI_API_KEY")

    if RECALL_MODE not in VALID_RECALL_MODES:
        raise SystemExit(
            "Invalid ENGRAM_CHATBOT_RECALL_MODE. Use 'fast', 'deep', or 'debug'."
        )
    if MEMORY_JOBS_MODE not in VALID_MEMORY_JOBS_MODES:
        raise SystemExit(
            "Invalid ENGRAM_CHATBOT_MEMORY_JOBS. Use 'inline' or 'deferred'."
        )
    if RERANK_MODE not in VALID_RERANK_MODES:
        raise SystemExit(
            "Invalid ENGRAM_CHATBOT_RERANK. Use 'auto', 'true', or 'false'."
        )

    provider = os.environ.get("ENGRAM_EMBEDDING_PROVIDER", "openai")
    if provider != "openai":
        raise SystemExit(
            "examples/chatbot.py is configured as a real OpenAI-backed chatbot. "
            f"Set ENGRAM_EMBEDDING_PROVIDER=openai, got {provider!r}."
        )

    if missing:
        raise SystemExit(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + "\nSet OpenAI embeddings/chat config before running the chatbot."
        )


class MemoryChatbot:
    def __init__(self) -> None:
        self.engram = None
        self.task_id: str | None = None
        self.session_id: str | None = None
        self._session_context: Any | None = None
        self.history: list[dict[str, str]] = []

    async def connect(self) -> None:
        require_real_openai_config()

        from engram import Engram

        self.engram = Engram(memory_policy="default")
        await self.engram.connect()
        if self.engram.llm is None:
            raise RuntimeError(
                "LLM provider is disabled. Set ENGRAM_LLM_PROVIDER=openai and "
                "ENGRAM_OPENAI_API_KEY before running examples/chatbot.py."
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
                    os.environ.get("ENGRAM_EMBEDDING_PROVIDER", "openai"),
                ),
                (
                    "embedding_model",
                    os.environ.get("ENGRAM_EMBEDDING_MODEL", "text-embedding-3-small"),
                ),
                ("embedding_dim", os.environ.get("ENGRAM_EMBEDDING_DIMENSION", "auto")),
                ("llm_provider", os.environ.get("ENGRAM_LLM_PROVIDER", "openai")),
                ("llm_model", os.environ.get("ENGRAM_LLM_MODEL", "gpt-4o-mini")),
                ("recall_mode", RECALL_MODE),
                ("memory_jobs", MEMORY_JOBS_MODE),
                ("rerank", self._rerank_enabled()),
                ("broad_memory_limit", BROAD_MEMORY_LIMIT),
            ]
        )

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
            "Run a real OpenAI-backed chatbot with persistent Engram memory",
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

        messages, recall_metadata = await self._build_prompt_messages(message)
        llm_response = await self.engram.llm.complete_full(
            messages,
            max_tokens=700,
            temperature=0.4,
        )
        response = llm_response.content.strip()

        metadata = {
            "application": "examples/chatbot.py",
            "llm_model": llm_response.model,
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

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        self.history = self.history[-HISTORY_LIMIT:]

        processed = len([job for job in jobs if job.status == "completed"])
        if processed:
            print_notice(f"processed {processed} memory job(s)", level="ok")
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
            *self.history[-HISTORY_LIMIT:],
            {"role": "user", "content": message},
        ]
        return messages, {"critical_memory_count": len(critical_memories)}

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
        task_context = await self.engram.build_context(
            self.task_id,
            query=message,
            max_tokens=1600,
            recent_event_limit=10,
            memory_limit=10,
            checkpoint_limit=2,
            include_graph=True,
        )

        metadata: dict[str, Any] = {}
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
            {"role": "system", "content": SYSTEM_PROMPT},
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
            return await self.engram.process_memory_jobs(limit=10)
        print_notice(
            "memory job queued, run process_memory_jobs() or run_memory_worker() later"
        )
        return []

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
            [("memory_id", memory.memory_id), ("content", preview(memory.content))]
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
                f"{dim(f'[{memory.memory_type} {memory.importance:.2f}]')} "
                f"{preview(memory.content, 120)}"
            )

    async def search(self, query: str) -> None:
        assert self.engram is not None
        results = await self.engram.search(query, AGENT_ID, user_id=USER_ID, limit=8)
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
    ("/memories", "list recent Engram memories"),
    ("/search <query>", "search memories and reinforce hits"),
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
    print(dim("Plain text runs: recall -> OpenAI chat -> record_turn -> memory jobs."))
    print(rule())


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
    elif command == "/memories":
        await bot.memories()
    elif command == "/search" and rest:
        await bot.search(rest)
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


async def main() -> None:
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
    print_help()
    try:
        while True:
            line = (await asyncio.to_thread(input, prompt_text())).strip()
            if not sys.stdin.isatty():
                print()
            if not line:
                continue
            if line.startswith("/"):
                if not await run_command(bot, line):
                    break
            else:
                if COLOR_ENABLED:
                    print_notice("recalling memory and calling OpenAI")
                response = await bot.reply(line)
                print_response(response)
    finally:
        await bot.close()
        if not sys.stdin.isatty():
            print()
        print_notice("bye", level="ok")


if __name__ == "__main__":
    asyncio.run(main())
