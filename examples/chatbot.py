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
MEMORY_JOBS_MODE = os.environ.get("ENGRAM_CHATBOT_MEMORY_JOBS", "inline").lower()
BROAD_MEMORY_LIMIT = int(os.environ.get("ENGRAM_CHATBOT_BROAD_MEMORY_LIMIT", "60"))
BROAD_MEMORY_CHARS = int(os.environ.get("ENGRAM_CHATBOT_BROAD_MEMORY_CHARS", "3600"))

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


def print_table(rows: list[tuple[str, Any]]) -> None:
    for key, value in rows:
        print(f"{key:<20} {value}")


def require_real_openai_config() -> None:
    missing = []
    if not os.environ.get("ENGRAM_OPENAI_API_KEY"):
        missing.append("ENGRAM_OPENAI_API_KEY")

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
        print_table(
            [
                ("health", health.get("status")),
                ("agent", AGENT_ID),
                ("user", USER_ID),
                ("task", self.task_id),
                ("session", self.session_id),
                ("embedding_provider", os.environ.get("ENGRAM_EMBEDDING_PROVIDER", "openai")),
                ("embedding_model", os.environ.get("ENGRAM_EMBEDDING_MODEL", "text-embedding-3-small")),
                ("embedding_dim", os.environ.get("ENGRAM_EMBEDDING_DIMENSION", "auto")),
                ("llm_provider", os.environ.get("ENGRAM_LLM_PROVIDER", "openai")),
                ("llm_model", os.environ.get("ENGRAM_LLM_MODEL", "gpt-4o-mini")),
                ("memory_jobs", MEMORY_JOBS_MODE),
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
        )
        deep_hits = await self.engram.deep_search(
            message,
            AGENT_ID,
            user_id=USER_ID,
            limit=16,
            n_queries=4,
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
                    "<engram_query_specific_must_use priority=\"highest\">\n"
                    f"{query_attention_block}\n"
                    "</engram_query_specific_must_use>"
                ),
            },
            {
                "role": "system",
                "content": f"<engram_memory_context>\n{memory_block}\n</engram_memory_context>",
            },
            {
                "role": "system",
                "content": f"<engram_recall_trace>\n{trace.context}\n</engram_recall_trace>",
            },
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
        llm_response = await self.engram.llm.complete_full(
            messages,
            max_tokens=700,
            temperature=0.4,
        )
        response = llm_response.content.strip()

        await self.engram.record_turn(
            self.task_id,
            message,
            response,
            agent_id=AGENT_ID,
            user_id=USER_ID,
            session_id=self.session_id,
            metadata={
                "application": "examples/chatbot.py",
                "llm_model": llm_response.model,
                "trace_kept_memory_ids": trace.kept_memory_ids,
                "missing_expected_terms": trace.missing_expected_terms,
            },
        )
        if MEMORY_JOBS_MODE == "inline":
            jobs = await self.engram.process_memory_jobs(limit=10)
        elif MEMORY_JOBS_MODE == "deferred":
            jobs = []
            print(
                "[engram] memory job queued; run process_memory_jobs() "
                "or run_memory_worker() later"
            )
        else:
            raise RuntimeError(
                "Invalid ENGRAM_CHATBOT_MEMORY_JOBS. Use 'inline' or 'deferred'."
            )

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        self.history = self.history[-HISTORY_LIMIT:]

        processed = len([job for job in jobs if job.status == "completed"])
        if processed:
            print(f"[engram] processed {processed} memory job(s)")
        return response

    def _render_deep_memory_block(self, results: list[Any], max_chars: int = 2400) -> str:
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
                ("older", "old plan", "superseded", "cancelled", "canceled", "replaced"),
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
        print_table([("memory_id", memory.memory_id), ("content", preview(memory.content))])

    async def memories(self) -> None:
        assert self.engram is not None
        memories = await self.engram.list_recent(AGENT_ID, user_id=USER_ID, limit=15)
        if not memories:
            print("No memories stored yet.")
            return
        for memory in memories:
            print(
                f"{memory.memory_id} [{memory.memory_type}] "
                f"{memory.importance:.2f} {preview(memory.content)}"
            )

    async def search(self, query: str) -> None:
        assert self.engram is not None
        results = await self.engram.search(query, AGENT_ID, user_id=USER_ID, limit=8)
        if not results:
            print("No matches.")
            return
        for result in results:
            await self.engram.reinforce(result.memory.memory_id, 0.02)
            print(f"{result.score:.3f} {preview(result.memory.content)}")

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
        print(block or "No memory context.")
        print("\nTask context:")
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
        print_table([("deleted", deleted)])

    async def clear(self) -> None:
        assert self.engram is not None
        count = await self.engram.purge(AGENT_ID, user_id=USER_ID)
        self.history.clear()
        print_table([("purged", count)])


HELP = """
Commands:
  /remember <fact>      Store a durable fact immediately with add()
  /memories             List recent Engram memories for this user
  /search <query>       Search memories and reinforce the retrieved hits
  /context <query>      Show the prompt context Engram would send to the LLM
  /trace <query>        Inspect trace_recall() retrieval decisions
  /task                 Show the resumable task/session backing this chat
  /forget <memory_id>   Delete one memory
  /clear                Purge this chatbot user's memories
  /help                 Show this help
  /quit                 Exit

Plain text sends a real OpenAI chat completion with Engram memory context,
then records the turn and processes memory jobs so future replies can recall it.
"""


async def run_command(bot: MemoryChatbot, line: str) -> bool:
    command, _, rest = line.partition(" ")
    command = command.lower()
    rest = rest.strip()

    if command in {"/quit", "/q"}:
        return False
    if command == "/help":
        print(HELP)
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
        confirm = await asyncio.to_thread(input, "Delete this user's chatbot memories? y/N: ")
        if confirm.lower() == "y":
            await bot.clear()
    else:
        print("Unknown or incomplete command. Type /help.")
    return True


async def main() -> None:
    bot = MemoryChatbot()
    try:
        await bot.connect()
    except DatabaseConnectionError as exc:
        print(f"Database connection failed: {exc}")
        print(
            "Check that ENGRAM_DATABASE_URL matches the running Postgres "
            "container credentials. If you used docker-setup.sh, the password "
            "in .env may differ from an older existing Docker volume."
        )
        return
    print(HELP)
    try:
        while True:
            line = (await asyncio.to_thread(input, "you> ")).strip()
            if not line:
                continue
            if line.startswith("/"):
                if not await run_command(bot, line):
                    break
            else:
                response = await bot.reply(line)
                print(f"bot> {response}")
    finally:
        await bot.close()
        print("bye")


if __name__ == "__main__":
    asyncio.run(main())
