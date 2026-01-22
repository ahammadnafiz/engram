#!/usr/bin/env python3
"""Personal chatbot with persistent memory using Engram."""

import asyncio
import os
import sys
from pathlib import Path

# Load .env FIRST
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Config - set BEFORE importing engram (settings are cached)
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"

os.environ["ENGRAM_EMBEDDING_PROVIDER"] = EMBEDDING_PROVIDER
os.environ["ENGRAM_EMBEDDING_MODEL"] = EMBEDDING_MODEL
if os.environ.get("OPENAI_API_KEY"):
    os.environ["ENGRAM_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

# NOW import engram (after env vars are set)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from engram import Engram, EmbeddingService, LLMService
from engram.llm.service import MemoryOperationType
from engram.core.config import clear_settings_cache
clear_settings_cache()  # Force reload settings

AGENT_ID = "assistant"
USER_ID = "user"

# Sliding window config
MAX_HISTORY_MESSAGES = 20  # Keep last N messages in memory
CONTEXT_WINDOW_MESSAGES = 10  # Send last N to LLM
MAX_CONTEXT_CHARS = 4000  # Approximate token limit for context

SYSTEM_PROMPT = """You are a friendly personal assistant who remembers everything about the user.

Your memory contains facts from previous conversations. Use them naturally - don't announce "I remember..." every time, just incorporate what you know into your responses like a real friend would.

Guidelines:
- Be warm, conversational, and concise
- Reference past details naturally when relevant
- When user shares new info (name, preferences, goals), briefly acknowledge you'll remember
- If memories seem outdated, politely ask if things have changed
- Don't repeat back memories verbatim - weave them into conversation

You're not just an assistant - you're someone who genuinely knows and cares about the user."""



class Chatbot:
    def __init__(self):
        self.engram: Engram | None = None
        self.embedding: EmbeddingService | None = None
        self.llm: LLMService | None = None
        self.history: list[dict] = []  # Full conversation history
        self._tasks: list[asyncio.Task] = []
        self._max_pending_tasks = 50  # Prevent unbounded task accumulation

    def _get_context_window(self) -> list[dict]:
        """Get sliding window of recent conversation history for LLM context."""
        if not self.history:
            return []
        
        # Start with most recent messages
        window = []
        total_chars = 0
        
        # Work backwards from most recent
        for msg in reversed(self.history):
            msg_len = len(msg.get("content", ""))
            
            # Stop if we exceed limits
            if len(window) >= CONTEXT_WINDOW_MESSAGES:
                break
            if total_chars + msg_len > MAX_CONTEXT_CHARS:
                break
            
            window.append(msg)
            total_chars += msg_len
        
        # Return in chronological order
        return list(reversed(window))

    def _trim_history(self):
        """Trim history to max size, keeping most recent."""
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def _cleanup_tasks(self):
        """Remove completed tasks from the task list to prevent memory leaks."""
        self._tasks = [t for t in self._tasks if not t.done()]

    def _add_background_task(self, coro):
        """Add a background task with cleanup to prevent unbounded growth."""
        # Cleanup completed tasks periodically
        if len(self._tasks) >= self._max_pending_tasks:
            self._cleanup_tasks()
        
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task

    async def connect(self):
        self.engram = Engram()
        await self.engram.connect()
        
        # Create services with explicit providers
        api_key = os.environ.get("OPENAI_API_KEY")
        
        self.embedding = EmbeddingService.from_provider(
            EMBEDDING_PROVIDER,
            model=EMBEDDING_MODEL,
            api_key=api_key,
        )
        self.llm = LLMService.from_provider(
            LLM_PROVIDER,
            model=LLM_MODEL,
            api_key=api_key,
        )
        
        # Health check (non-fatal - network can be flaky)
        try:
            health = await self.engram.health_check()
            if health.get("status") != "healthy":
                print(f"  ⚠ Health warning: {health.get('components', {}).get('embedding', {}).get('error', 'unknown')}")
        except Exception as e:
            print(f"  ⚠ Health check failed: {e}")
        
        print(f"\n  Embedding: {self.embedding.model} ({self.embedding.dimension}d)")
        print(f"  LLM:       {self.llm.model}")
        print(f"  Database:  ✓ connected\n")

    async def close(self):
        """Gracefully close the chatbot, waiting for pending tasks."""
        # Clean up and wait for all pending background tasks
        self._cleanup_tasks()
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            # Give tasks a reasonable timeout to complete
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for t in pending:
                    if not t.done():
                        t.cancel()
        
        if self.engram:
            await self.engram.close()

    async def get_context(self, query: str, limit: int = 5) -> str:
        """Get relevant memories and reinforce them (used = more important)."""
        if not self.engram:
            return ""
        results = await self.engram.search(
            query=query, agent_id=AGENT_ID, user_id=USER_ID, limit=limit
        )
        if not results:
            return ""
        
        # Filter by relevance threshold
        relevant = [r for r in results if r.score > 0.3]
        if not relevant:
            return ""
        
        # Reinforce retrieved memories (they're being used!)
        for r in relevant:
            self._add_background_task(self._reinforce_memory(r.memory.memory_id, r.score))
        
        return "\n".join(f"- {r.memory.content}" for r in relevant)

    async def _reinforce_memory(self, memory_id: str, score: float):
        """Boost memory importance based on retrieval relevance."""
        if not self.engram:
            return
        try:
            # Higher relevance = bigger boost (0.02 to 0.1)
            boost = 0.02 + (score * 0.08)
            await self.engram.reinforce(memory_id, boost)
        except Exception as e:
            # Log but don't interrupt - reinforcement is non-critical
            import logging
            logging.debug(f"Failed to reinforce memory {memory_id}: {e}")

    async def store_memory(self, content: str):
        """Store a memory, avoiding duplicates."""
        if not self.engram:
            return
        try:
            # Check for duplicates (high similarity = already exists)
            existing = await self.engram.search(
                query=content, agent_id=AGENT_ID, user_id=USER_ID, limit=1
            )
            if existing and existing[0].score > 0.85:
                return  # Already have this fact
            
            await self.engram.add(
                content=content,
                agent_id=AGENT_ID,
                user_id=USER_ID,
            )
        except Exception as e:
            print(f"⚠ Memory error: {e}")

    async def extract_and_store_facts(self, user_msg: str, bot_msg: str):
        """Extract and process facts using intelligent memory operations."""
        if not self.llm or not self.engram:
            return
        
        # Run both extractions concurrently
        await asyncio.gather(
            self._extract_user_facts(user_msg, bot_msg),
            self._extract_conversation_topic(user_msg, bot_msg),
            return_exceptions=True,  # Don't fail if one errors
        )

    async def _extract_user_facts(self, user_msg: str, bot_msg: str):
        """Extract USER-specific facts (name, preferences, personal info)."""
        if not self.llm or not self.engram:
            return
        
        try:
            facts = await self.llm.extract_facts(
                user_msg,
                bot_msg,
                conversation_history=self.history[-10:],
            )
            
            if not facts:
                return
            
            # Process each fact with its own similarity search
            for fact in facts:
                await self._process_single_fact(fact, memory_type="user_fact")
                
        except Exception:
            pass  # Non-critical, don't interrupt chat

    async def _extract_conversation_topic(self, user_msg: str, bot_msg: str):
        """Extract TOPIC/KNOWLEDGE from the conversation using summarization."""
        if not self.llm or not self.engram:
            return
        
        try:
            # Skip short or trivial exchanges (greetings, commands, etc.)
            if len(bot_msg) < 100 or len(user_msg) < 10:
                return
            
            # Skip if user message is a command
            if user_msg.strip().startswith("/"):
                return
            
            # Build conversation text for summarization
            conversation_text = f"""Question: {user_msg}

Answer: {bot_msg}"""
            
            # Use the existing summarize method with bullet style for clarity
            topic_summary = await self.llm.summarize(
                text=conversation_text,
                max_length=50,  # Keep it concise
                style="concise",
            )
            topic_summary = topic_summary.strip()
            
            # Skip if no substantial summary or just echoes the question
            if (not topic_summary or 
                len(topic_summary) < 15 or
                topic_summary.lower().startswith("the user") or
                "no information" in topic_summary.lower()):
                return
            
            # Prefix to make it clear this is a topic discussed
            topic_content = f"Discussed: {topic_summary}"
            
            # Check for duplicates
            existing = await self.engram.search(
                query=topic_content,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                limit=3,
            )
            
            # Skip if very similar topic already exists
            for r in existing:
                if r.score > 0.75:
                    return
            
            # Store the topic with metadata
            await self.engram.add(
                content=topic_content,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                metadata={
                    "type": "conversation_topic",
                    "user_query": user_msg[:100],
                },
            )
            
        except Exception:
            pass  # Non-critical

    async def _process_single_fact(self, fact: str, memory_type: str = "user_fact"):
        """Process a single fact with targeted similarity search."""
        if not self.llm or not self.engram:
            return
        
        try:
            # Search specifically for this fact's topic
            similar = await self.engram.search(
                query=fact,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                limit=5,
            )
            
            # Check for high-similarity duplicates first
            for r in similar:
                if r.score > 0.85:
                    # Very similar - likely duplicate, skip
                    return
            
            # Get relevant memories for operation evaluation (2-tuple format: id, content)
            existing_memories: list[tuple[str, str]] = [
                (r.memory.memory_id, r.memory.content) 
                for r in similar if r.score > 0.35
            ]
            
            metadata = {"type": memory_type}
            
            if not existing_memories:
                # No similar memories, just add
                await self.engram.add(
                    content=fact, 
                    agent_id=AGENT_ID, 
                    user_id=USER_ID,
                    metadata=metadata,
                )
                return
            
            # Evaluate operation with LLM
            operation = await self.llm.evaluate_memory_operation(fact, existing_memories)
            await self._execute_memory_operation(operation, metadata)
            
        except Exception:
            # Fallback: try to add
            try:
                await self.engram.add(
                    content=fact, 
                    agent_id=AGENT_ID, 
                    user_id=USER_ID,
                    metadata={"type": memory_type},
                )
            except Exception:
                pass

    async def _execute_memory_operation(self, op, metadata: dict | None = None):
        """Execute a single memory operation."""
        if not self.engram:
            return
        
        metadata = metadata or {"type": "user_fact"}
        
        try:
            if op.operation == MemoryOperationType.ADD:
                await self.engram.add(
                    content=op.content,
                    agent_id=AGENT_ID,
                    user_id=USER_ID,
                    metadata=metadata,
                )
            
            elif op.operation == MemoryOperationType.UPDATE and op.target_id:
                await self.engram.update(op.target_id, content=op.content)
            
            elif op.operation == MemoryOperationType.DELETE and op.target_id:
                await self.engram.forget(op.target_id)
                # Add the replacement fact
                await self.engram.add(
                    content=op.content,
                    agent_id=AGENT_ID,
                    user_id=USER_ID,
                    metadata=metadata,
                )
            
            # NOOP - do nothing
            
        except Exception:
            # Fallback: try to add
            try:
                await self.engram.add(
                    content=op.content,
                    agent_id=AGENT_ID,
                    user_id=USER_ID,
                    metadata=metadata,
                )
            except Exception:
                pass

    async def chat(self, user_input: str) -> str:
        if not self.llm:
            return "LLM not configured"
        
        # Get relevant memories from long-term storage
        memory_context = await self.get_context(user_input)
        
        # Build messages with sliding window
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add long-term memory context
        if memory_context:
            messages.append({
                "role": "system", 
                "content": f"Relevant memories:\n{memory_context}"
            })
        
        # Add sliding window of recent conversation (short-term context)
        messages.extend(self._get_context_window())
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = await self.llm.complete_full(messages)
        reply = response.content
        
        # Update history and trim if needed
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})
        self._trim_history()
        
        # Extract facts in background (tracked for graceful shutdown)
        self._add_background_task(self.extract_and_store_facts(user_input, reply))
        
        return reply

    async def show_memories(self, limit: int = 10):
        if not self.engram:
            return
        memories = await self.engram.list_recent(
            agent_id=AGENT_ID, user_id=USER_ID, limit=limit
        )
        print(f"\n📝 Recent Memories ({len(memories)})")
        for m in memories:
            mem_type = m.metadata.get("type", "unknown") if m.metadata else "unknown"
            icon = "👤" if mem_type == "user_fact" else "💬" if mem_type == "conversation_topic" else "📌"
            print(f"  {icon} [{mem_type}] {m.content[:55]}...")
        print()

    async def search_memories(self, query: str):
        if not self.engram:
            return
        results = await self.engram.search(
            query=query, agent_id=AGENT_ID, user_id=USER_ID, limit=5
        )
        print(f"\n🔍 Search: '{query}'")
        for r in results:
            mem_type = r.memory.metadata.get("type", "unknown") if r.memory.metadata else "unknown"
            icon = "👤" if mem_type == "user_fact" else "💬" if mem_type == "conversation_topic" else "📌"
            print(f"  {icon} [{r.score:.2f}] {r.memory.content[:45]}...")
        print()

    async def forget_all(self):
        if not self.engram:
            return
        count = await self.engram.purge(agent_id=AGENT_ID)
        self.history.clear()
        print(f"🗑️ Deleted {count} memories\n")

    async def consolidate_memories(self):
        """Find and merge similar memories to reduce redundancy."""
        if not self.engram or not self.llm:
            return
        
        print("\n🔄 Consolidating memories...")
        
        # Get all memories
        memories = await self.engram.list_recent(agent_id=AGENT_ID, user_id=USER_ID, limit=100)
        if len(memories) < 2:
            print("  Not enough memories to consolidate.\n")
            return
        
        merged = 0
        deleted = 0
        checked = set()
        
        for mem in memories:
            if mem.memory_id in checked:
                continue
            checked.add(mem.memory_id)
            
            # Find similar memories
            similar = await self.engram.search(
                query=mem.content,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                limit=5,
            )
            
            # Look for high-similarity pairs (excluding self)
            for r in similar:
                if r.memory.memory_id == mem.memory_id:
                    continue
                if r.memory.memory_id in checked:
                    continue
                if r.score > 0.80:  # High similarity = likely redundant
                    # Ask LLM which to keep or how to merge
                    operation = await self.llm.evaluate_memory_operation(
                        mem.content,
                        [(r.memory.memory_id, r.memory.content)],
                    )
                    
                    if operation.operation == MemoryOperationType.NOOP:
                        # Duplicate - delete the newer one
                        await self.engram.forget(r.memory.memory_id)
                        checked.add(r.memory.memory_id)
                        deleted += 1
                        print(f"  Deleted duplicate: {r.memory.content[:40]}...")
                    elif operation.operation == MemoryOperationType.UPDATE:
                        # Merge them
                        await self.engram.update(mem.memory_id, content=operation.content)
                        await self.engram.forget(r.memory.memory_id)
                        checked.add(r.memory.memory_id)
                        merged += 1
                        print(f"  Merged: {operation.content[:50]}...")
        
        print(f"\n  Merged: {merged}, Deleted: {deleted}\n")

    def show_config(self):
        print(f"\n⚙️ Config")
        print(f"  Embedding: {self.embedding.model if self.embedding else 'N/A'} ({self.embedding.dimension if self.embedding else 0}d)")
        print(f"  LLM:       {self.llm.model if self.llm else 'N/A'}")
        print()


def print_help():
    print("""
Commands:
  /memories     Show recent memories
  /search <q>   Search memories
  /consolidate  Merge similar memories
  /forget       Clear all memories
  /config       Show config
  /help         This help
  /quit         Exit
""")


async def main():
    print("🧠 Engram Chatbot")
    
    bot = Chatbot()
    
    try:
        print("Connecting...")
        await bot.connect()
        print("Type /help for commands.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    cmd = user_input.split()[0].lower()
                    
                    if cmd in ("/quit", "/exit", "/q"):
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
                    elif cmd == "/consolidate":
                        await bot.consolidate_memories()
                    elif cmd == "/forget":
                        if input("Sure? (y/n): ").lower() == "y":
                            await bot.forget_all()
                    elif cmd == "/config":
                        bot.show_config()
                    else:
                        print(f"Unknown: {cmd}")
                    continue
                
                print("...", end="\r")
                response = await bot.chat(user_input)
                print(f"Bot: {response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}\n")
    
    finally:
        print("\n💾 Saving...")
        await bot.close()
        print("Bye!")


if __name__ == "__main__":
    asyncio.run(main())