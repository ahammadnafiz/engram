#!/usr/bin/env python3
"""
Personal Chatbot with Persistent Memory - Engram Demo

Demonstrates the full Engram API:
  • engram.add()        - Store memories
  • engram.start_task() - Start/resume long-running task memory
  • engram.record_turn()- Write raw user/assistant events to the task ledger
  • engram.build_context() - Build prompt context from task state + memories
  • engram.process_memory_jobs() - Derive facts/checkpoints from queued turns
  • engram.search()     - Hybrid search (semantic + keyword)
  • engram.get()        - Retrieve by ID
  • engram.update()     - Modify memories
  • engram.reinforce()  - Boost importance (memory decay)
  • engram.forget()     - Delete single memory
  • engram.purge()      - Clear all memories
  • engram.list_recent()- Browse memories
  • engram.relate()     - Create memory relations
  • engram.traverse()   - Graph traversal
  • engram.session()    - Session management

Usage:
    python chatbot.py

Commands:
    /memories       Show recent memories
    /search <q>     Hybrid search
    /graph          Show memory relations
    /task           Show active task memory context
    /worker         Process queued memory jobs now
    /forget         Clear all memories
    /help           Show commands
    /quit           Exit
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Config. Environment variables win so production runs can choose providers.
EMBEDDING_PROVIDER = os.environ.get(
    "ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers"
)
EMBEDDING_MODEL = os.environ.get("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

LLM_PROVIDER = os.environ.get("ENGRAM_LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("ENGRAM_LLM_MODEL", "gpt-4o-mini")

os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", EMBEDDING_PROVIDER)
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", EMBEDDING_MODEL)
if os.environ.get("OPENAI_API_KEY"):
    os.environ.setdefault("ENGRAM_OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from engram import EmbeddingService, Engram, LLMService  # noqa: E402
from engram.core.config import clear_settings_cache  # noqa: E402

clear_settings_cache()

AGENT_ID = "assistant"
USER_ID = "user"

# Sliding window config
MAX_HISTORY = 20
CONTEXT_WINDOW = 10
MAX_CHARS = 4000

SYSTEM_PROMPT = """<role>
You are a friendly personal assistant with persistent memory. You remember facts about the user across conversations.
</role>

<memory_rules>
- The facts provided in <memories> are YOUR ground truth - they come from previous conversations
- ALWAYS use these facts when answering questions about the user
- Pay special attention to: name, job, company, projects, interests, preferences
- For multi-part questions, answer each part separately using all available memory context
- If asked about projects/work, check memories for specific project names
- NEVER contradict or guess beyond what's in your memories
- If asked about something not in your memories, say "I don't have that in my memory"
</memory_rules>

<personality>
- Be warm, friendly, and conversational
- Show genuine interest in the user
- Keep responses concise unless detail is requested
- Ask follow-up questions to learn more
</personality>

<response_format>
- Reference specific facts naturally (don't say "according to my memory")
- If user corrects a fact, acknowledge it
</response_format>"""


class MemoryChatbot:
    """Chatbot using Engram's task memory architecture."""

    def __init__(self):
        self.engram: Engram | None = None
        self.embedding: EmbeddingService | None = None
        self.llm: LLMService | None = None
        self.history: list[dict] = []
        self._tasks: list[asyncio.Task] = []
        self.task_run_id: str | None = None
        self.last_recall_trace = None

    # =========================================================================
    # Connection & Lifecycle
    # =========================================================================

    async def connect(self):
        """Connect to Engram and initialize services."""
        self.engram = Engram()
        await self.engram.connect()

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "ENGRAM_OPENAI_API_KEY"
        )

        # EmbeddingService - for vector embeddings
        # sentence-transformers is local (no API key needed)
        if EMBEDDING_PROVIDER == "sentence-transformers":
            self.embedding = EmbeddingService.from_provider(
                EMBEDDING_PROVIDER,
                model=EMBEDDING_MODEL,
            )
        else:
            self.embedding = EmbeddingService.from_provider(
                EMBEDDING_PROVIDER,
                model=EMBEDDING_MODEL,
                api_key=api_key,
            )

        # LLMService - for chat and fact extraction
        self.llm = LLMService.from_provider(
            LLM_PROVIDER, model=LLM_MODEL, api_key=api_key
        )

        # Health check
        health = await self.engram.health_check()
        status = "✓" if health.get("status") == "healthy" else "⚠"
        print(f"  Database: {status}")
        print(f"  Embedding: {self.embedding.model} ({self.embedding.dimension}d)")
        print(f"  LLM: {self.llm.model}\n")

        active_tasks = await self.engram.list_tasks(
            agent_id=AGENT_ID,
            user_id=USER_ID,
            status=["active", "paused"],
            limit=1,
        )
        if active_tasks:
            task = active_tasks[0]
            self.task_run_id = task.task_run_id
            if task.status == "paused":
                # Reuse the same task row; new turns will continue the ledger.
                print(f"  Resumed paused memory task: {task.task_run_id}")
            else:
                print(f"  Resumed active memory task: {task.task_run_id}")
        else:
            task = await self.engram.start_task(
                "Maintain persistent memory for the personal assistant chat",
                AGENT_ID,
                user_id=USER_ID,
                metadata={"example": "chatbot"},
            )
            self.task_run_id = task.task_run_id
            print(f"  Started memory task: {task.task_run_id}")

    async def close(self):
        """Wait for pending tasks and close."""
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            print(f"  Saving {len(pending)} memories...")
            try:
                # Increased timeout to allow complex fact extraction to complete
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=45.0
                )
            except asyncio.TimeoutError:
                remaining = len([t for t in self._tasks if not t.done()])
                if remaining:
                    print(f"  ⚠ {remaining} tasks still pending (timeout)")
        await self.pause_task()
        if self.engram:
            await self.engram.close()

    # =========================================================================
    # Sliding Window
    # =========================================================================

    def _get_context(self) -> list[dict]:
        """Get sliding window of recent conversation."""
        window, chars = [], 0
        for msg in reversed(self.history):
            if len(window) >= CONTEXT_WINDOW or chars > MAX_CHARS:
                break
            window.append(msg)
            chars += len(msg.get("content", ""))
        return list(reversed(window))

    def _trim_history(self):
        """Keep history bounded."""
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    # =========================================================================
    # ENGRAM API: Search & Reinforce
    # =========================================================================

    def _needs_high_recall(self, query: str) -> bool:
        """Return True for broad prompts that need more than a top-k search."""
        q = query.lower()
        broad_markers = (
            "everything",
            "summarize",
            "what do you remember",
            "based on everything",
            "based on memory",
            "final verification",
            "exact",
            "all targets",
            "checklist",
            "brief",
        )
        return (
            any(marker in q for marker in broad_markers)
            or q.count("?") > 1
            or ("," in q and (" and " in q or ":" in q))
        )

    async def recall(
        self,
        query: str,
        limit: int = 6,
        max_chars: int = 3500,
        *,
        high_recall: bool = False,
    ) -> str:
        """
        engram.search() - Hybrid search with semantic + keyword matching
        engram.deep_search() - Multi-query retrieval for broad questions
        engram.reinforce() - Boost importance of used memories

        Returns both fact and main_content for richer LLM context.
        Context budget management: truncates main_content if too large.
        """
        if not self.engram:
            return ""

        if high_recall:
            trace = await self.engram.trace_recall(
                query=query,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                limit=limit,
                min_score=0.15,
                max_tokens=max_chars // 4,
                use_deep_search=True,
            )
            self.last_recall_trace = trace
            return trace.context
        else:
            # Hybrid search: combines vector similarity + BM25 keyword matching
            # (Engram uses hybrid mode by default, searches on fact column)
            results = await self.engram.search(
                query=query,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                limit=limit,
                min_score=0.2,
            )
            self.last_recall_trace = None

        relevant = [r for r in results if r.score >= (0.15 if high_recall else 0.2)]
        if not relevant:
            return ""

        # Reinforce used memories (increases importance over time)
        for r in relevant:
            boost = 0.02 + (r.score * 0.08)
            self._tasks.append(
                asyncio.create_task(self.engram.reinforce(r.memory.memory_id, boost))
            )

        # Build context with both fact and main_content
        # Context budget management to avoid overflowing LLM context window
        lines = []
        chars_used = 0
        max_main_content_len = 220 if high_recall else 150

        for r in relevant:
            # Always include the fact (this is what matched)
            fact = r.memory.fact or r.memory.content
            line = f"- {fact}"

            # Include main_content only if budget allows
            main_content = r.memory.main_content
            if main_content and chars_used + len(main_content) < max_chars:
                # Truncate if too long
                if len(main_content) > max_main_content_len:
                    main_content = main_content[:max_main_content_len] + "..."
                line += f"\n  ({main_content})"
                chars_used += len(main_content)

            lines.append(line)
            chars_used += len(fact)

        return "\n".join(lines)

    # =========================================================================
    # ENGRAM API: Durable Task Memory Ingestion
    # =========================================================================

    async def learn(self, user_msg: str, bot_msg: str):
        """
        Record the raw turn and process the queued memory job.

        record_turn() stores user/assistant events in agent_events and queues
        a turn_ingest job. process_memory_jobs() turns that raw ledger entry
        into checkpoints and, when an LLM provider is configured, searchable
        facts in agent_memory.
        """
        if not self.engram or not self.task_run_id:
            return

        try:
            await self.engram.record_turn(
                self.task_run_id,
                user_msg,
                bot_msg,
                metadata={"source": "chatbot_example"},
            )
            jobs = await self.engram.process_memory_jobs(limit=5)
            if jobs and os.environ.get("DEBUG_FACTS"):
                print(f"  Processed memory jobs: {[j.status for j in jobs]}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if os.environ.get("DEBUG_FACTS"):
                print(f"  Memory job failed: {exc}")

    async def task_context(
        self,
        query: str,
        max_tokens: int = 1200,
        *,
        memory_limit: int = 8,
        recent_event_limit: int = 12,
        checkpoint_limit: int = 2,
    ) -> str:
        """Build prompt-ready context from task memory and fact memory."""
        if not self.engram or not self.task_run_id:
            return ""

        context = await self.engram.build_context(
            self.task_run_id,
            query=query,
            max_tokens=max_tokens,
            recent_event_limit=recent_event_limit,
            memory_limit=memory_limit,
            checkpoint_limit=checkpoint_limit,
            include_graph=True,
        )
        return context.text

    async def process_memory_backlog(self) -> int:
        """Process queued memory jobs on demand."""
        if not self.engram:
            return 0
        jobs = await self.engram.process_memory_jobs(limit=20)
        return len(jobs)

    async def pause_task(self):
        """Pause the active task before shutdown."""
        if self.engram and self.task_run_id:
            await self.engram.pause_task(
                self.task_run_id, outcome="Chatbot session ended"
            )

    # =========================================================================
    # Chat
    # =========================================================================

    async def chat(self, user_input: str) -> str:
        """Generate response with memory context."""
        if not self.llm:
            return "Error: LLM not configured"

        high_recall = self._needs_high_recall(user_input)

        # Long-term memory via hybrid search, or deep search for broad prompts.
        memories = await self.recall(
            user_input,
            limit=14 if high_recall else 6,
            max_chars=6500 if high_recall else 3500,
            high_recall=high_recall,
        )

        # Long-running task context: goal, recent events, checkpoints, facts,
        # artifacts, and graph expansion assembled under one prompt budget.
        task_memory = await self.task_context(
            user_input,
            max_tokens=2800 if high_recall else 1200,
            memory_limit=18 if high_recall else 8,
            recent_event_limit=24 if high_recall else 12,
            checkpoint_limit=3 if high_recall else 2,
        )

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if task_memory:
            messages.append(
                {
                    "role": "system",
                    "content": f"<task_memory>\n{task_memory}\n</task_memory>",
                }
            )
        if memories:
            messages.append(
                {"role": "system", "content": f"<memories>\n{memories}\n</memories>"}
            )

        # Short-term context via sliding window
        messages.extend(self._get_context())
        messages.append({"role": "user", "content": user_input})

        # Generate
        response = await self.llm.complete_full(messages)
        reply = response.content

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})
        self._trim_history()

        # Learn in background
        self._tasks.append(asyncio.create_task(self.learn(user_input, reply)))

        return reply

    # =========================================================================
    # Commands - Full Engram API Demo
    # =========================================================================

    async def show_memories(self, limit: int = 10):
        """engram.list_recent() - List memories sorted by recency."""
        if not self.engram:
            return

        memories = await self.engram.list_recent(
            agent_id=AGENT_ID,
            user_id=USER_ID,
            limit=limit,
        )

        print(f"\n📝 Memories ({len(memories)})")
        for m in memories:
            fact = m.fact or m.content
            print(f"  [{m.importance:.0%}] {fact[:55]}...")
            if m.main_content:
                # Show truncated main_content
                ctx = m.main_content[:60].replace("\n", " ")
                print(f"           └─ {ctx}...")
        print()

    async def search_memories(self, query: str):
        """engram.search() - Hybrid search demo (searches on fact column)."""
        if not self.engram:
            return

        results = await self.engram.search(
            query=query,
            agent_id=AGENT_ID,
            user_id=USER_ID,
            limit=5,
        )

        print(f"\n🔍 Hybrid Search: '{query}'")
        if not results:
            print("  No matches")
        else:
            for r in results:
                fact = r.memory.fact or r.memory.content
                print(f"  [{r.score:.0%}] {fact[:50]}...")
                if r.memory.main_content:
                    ctx = r.memory.main_content[:50].replace("\n", " ")
                    print(f"           └─ {ctx}...")
        print()

    async def show_graph(self):
        """engram.traverse() + traverse_many() - Show memory relations."""
        if not self.engram:
            return

        memories = await self.engram.list_recent(
            agent_id=AGENT_ID,
            user_id=USER_ID,
            limit=3,
        )

        if not memories:
            print("\n📊 No memories to traverse\n")
            return

        # Traverse from most recent memory
        results = await self.engram.traverse(
            start_memory_id=memories[0].memory_id,
            max_depth=2,
        )

        print(f"\n📊 Memory Graph (from: {memories[0].content[:30]}...)")
        if not results:
            print("  No connections")
        else:
            for r in results:
                print(f"  └─ [{r.depth}] {r.content[:45]}...")

        expanded = await self.engram.traverse_many(
            [m.memory_id for m in memories[:3]],
            max_depth=2,
            direction="any",
            total_limit=8,
        )
        block = self.engram.render_graph_context(expanded, max_tokens=300)
        if block:
            print("\nPrompt-ready graph block:")
            print(block)
        print()

    async def show_task(self):
        """engram.build_context() - Show active task memory context."""
        if not self.engram or not self.task_run_id:
            print("\nNo active task memory\n")
            return

        task = await self.engram.get_task(self.task_run_id)
        context = await self.task_context("current conversation state", max_tokens=900)
        print(f"\n🧭 Task Memory ({task.status})")
        print(f"  ID:   {task.task_run_id}")
        print(f"  Goal: {task.goal}")
        print("\n" + (context or "  No task context yet"))
        print()

    async def clear_memories(self):
        """engram.purge() - Delete all memories."""
        if not self.engram:
            return

        count = await self.engram.purge(agent_id=AGENT_ID)
        self.history.clear()
        print(f"🗑️  Deleted {count} memories\n")


# =============================================================================
# Main
# =============================================================================


def print_help():
    print("""
Commands:
  /memories     List recent memories (engram.list_recent)
  /search <q>   Hybrid search (engram.search)
  /graph        Show memory graph + prompt block (traverse_many/render_context)
  /task         Show active long-running task context (engram.build_context)
  /worker       Process queued memory jobs (engram.process_memory_jobs)
  /forget       Clear all (engram.purge)
  /help         This help
  /quit         Exit

Engram API Used:
  start_task, list_tasks, record_turn, build_context,
  process_memory_jobs, pause_task,
  add, search, get, update, reinforce, forget, purge,
  list_recent, relate, traverse, health_check
""")


async def main():
    print("🧠 Engram Memory Chatbot\n")
    print("Connecting...")

    bot = MemoryChatbot()

    try:
        await bot.connect()
        print("Type /help for commands.\n")

        while True:
            try:
                user_input = (await asyncio.to_thread(input, "You: ")).strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd = user_input.split()[0].lower()

                    if cmd in ("/quit", "/q"):
                        break
                    elif cmd == "/help":
                        print_help()
                    elif cmd == "/memories":
                        await bot.show_memories()
                    elif cmd == "/search":
                        q = user_input[7:].strip()
                        if q:
                            await bot.search_memories(q)
                        else:
                            print("Usage: /search <query>")
                    elif cmd == "/graph":
                        await bot.show_graph()
                    elif cmd == "/task":
                        await bot.show_task()
                    elif cmd == "/worker":
                        count = await bot.process_memory_backlog()
                        print(f"Processed {count} memory jobs\n")
                    elif cmd == "/forget":
                        if (
                            await asyncio.to_thread(input, "Delete all? (y/n): ")
                        ).lower() == "y":
                            await bot.clear_memories()
                    else:
                        print(f"Unknown: {cmd}")
                    continue

                response = await bot.chat(user_input)
                print(f"Bot: {response}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}\n")

    finally:
        print("\n💾 Saving...")
        await bot.close()
        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
