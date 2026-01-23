#!/usr/bin/env python3
"""
Personal Chatbot with Persistent Memory - Engram Demo

Demonstrates the full Engram API:
  • engram.add()        - Store memories
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

# Config
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"

os.environ["ENGRAM_EMBEDDING_PROVIDER"] = EMBEDDING_PROVIDER
os.environ["ENGRAM_EMBEDDING_MODEL"] = EMBEDDING_MODEL
if os.environ.get("OPENAI_API_KEY"):
    os.environ["ENGRAM_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from engram import Engram, EmbeddingService, LLMService
from engram.core.config import clear_settings_cache
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
- ALWAYS use these facts when answering questions about the user or people they mentioned
- NEVER contradict, modify, or guess beyond what's in your memories
- If asked about something not in your memories, honestly say "I don't have that in my memory"
- When referencing memories, be natural - don't say "according to my memory" every time
</memory_rules>

<personality>
- Be warm, friendly, and conversational
- Show genuine interest in the user
- Keep responses concise unless detail is requested
- Ask follow-up questions to learn more about the user
</personality>

<response_format>
- Reference specific facts from memories when relevant
- If user corrects a fact, acknowledge it (the system will update the memory)
- Don't repeat facts verbatim - weave them naturally into conversation
</response_format>"""


class MemoryChatbot:
    """Chatbot using full Engram API for persistent memory."""
    
    def __init__(self):
        self.engram: Engram | None = None
        self.embedding: EmbeddingService | None = None
        self.llm: LLMService | None = None
        self.history: list[dict] = []
        self._tasks: list[asyncio.Task] = []
    
    # =========================================================================
    # Connection & Lifecycle
    # =========================================================================
    
    async def connect(self):
        """Connect to Engram and initialize services."""
        self.engram = Engram()
        await self.engram.connect()
        
        api_key = os.environ.get("OPENAI_API_KEY")
        
        # EmbeddingService - for vector embeddings
        self.embedding = EmbeddingService.from_provider(
            EMBEDDING_PROVIDER,
            model=EMBEDDING_MODEL,
            api_key=api_key,
        )
        
        # LLMService - for chat and fact extraction
        self.llm = LLMService.from_provider(LLM_PROVIDER, model=LLM_MODEL, api_key=api_key)
        
        # Health check
        health = await self.engram.health_check()
        status = "✓" if health.get("status") == "healthy" else "⚠"
        print(f"  Database: {status}")
        print(f"  Embedding: {self.embedding.model} ({self.embedding.dimension}d)")
        print(f"  LLM: {self.llm.model}\n")
    
    async def close(self):
        """Wait for pending tasks and close."""
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            print(f"  Saving {len(pending)} memories...")
            try:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=15.0)
            except asyncio.TimeoutError:
                pass
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
    
    async def recall(self, query: str, limit: int = 5, max_chars: int = 3000) -> str:
        """
        engram.search() - Hybrid search with semantic + keyword matching
        engram.reinforce() - Boost importance of used memories
        
        Returns both fact and main_content for richer LLM context.
        Context budget management: truncates main_content if too large.
        """
        if not self.engram:
            return ""
        
        # Hybrid search: combines vector similarity + BM25 keyword matching
        # (Engram uses hybrid mode by default, searches on fact column)
        results = await self.engram.search(
            query=query,
            agent_id=AGENT_ID,
            user_id=USER_ID,
            limit=limit,
        )
        
        relevant = [r for r in results if r.score > 0.3]
        if not relevant:
            return ""
        
        # Reinforce used memories (increases importance over time)
        for r in relevant:
            boost = 0.02 + (r.score * 0.08)
            self._tasks.append(asyncio.create_task(
                self.engram.reinforce(r.memory.memory_id, boost)
            ))
        
        # Build context with both fact and main_content
        # Context budget management to avoid overflowing LLM context window
        lines = []
        chars_used = 0
        max_main_content_len = 150  # Truncate long main_content
        
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
    # ENGRAM API: Fact Extraction & Memory Operations
    # =========================================================================
    
    async def learn(self, user_msg: str, bot_msg: str):
        """
        Extract facts and use engram.add(), engram.update(), engram.forget()
        Two-column system: fact (embedded) + main_content (context, not embedded)
        """
        if not self.llm or not self.engram:
            return
        
        try:
            # LLM extracts atomic facts about the user
            facts = await self.llm.extract_facts(
                user_msg, bot_msg,
                conversation_history=self.history[-6:],
            )
            
            for fact in facts:
                # Pass conversation context for main_content
                await self._process_fact(fact, user_msg, bot_msg)
                
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
    
    async def _process_fact(self, fact: str, user_msg: str, bot_msg: str):
        """Process a fact using full Engram memory operations.
        
        Two-column system:
        - fact: The extracted user fact (embedded for hybrid search)
        - main_content: "[USER]: {msg}\n[AI]: {summary}" (not embedded, preserved for context)
        """
        if not self.engram or not self.llm:
            return
        
        # Search for similar memories (hybrid by default)
        similar = await self.engram.search(
            query=fact,
            agent_id=AGENT_ID,
            user_id=USER_ID,
            limit=3,
        )
        
        # Skip exact duplicates (very high similarity)
        for r in similar:
            if r.score > 0.90:  # Raised threshold - was 0.85
                return
        
        # Only consider memories with high relevance for dedup decisions
        # Lower threshold means more false positives (NOOP when should ADD)
        existing = [(r.memory.memory_id, r.memory.content) for r in similar if r.score > 0.55]  # Raised from 0.4
        
        # Build main_content: preserve conversation context (not embedded)
        ai_summary = await self._summarize_response(bot_msg)
        main_content = f"[USER]: {user_msg}\n[AI]: {ai_summary}"
        
        if not existing:
            # engram.add() - Store new memory with two-column system
            # - content (fact): embedded for search
            # - main_content: preserved context (not embedded)
            new_mem = await self.engram.add(
                content=fact,  # This is the fact - gets embedded
                main_content=main_content,  # Context - NOT embedded
                agent_id=AGENT_ID,
                user_id=USER_ID,
                metadata={"type": "user_fact"},
            )
            # Create relation to recent memories for graph connectivity
            await self._link_to_recent(new_mem.memory_id)
            return
        
        # LLM decides: ADD, UPDATE, DELETE, or NOOP
        op = await self.llm.evaluate_memory_operation(fact, existing)
        
        if op.operation.value == "ADD":
            new_mem = await self.engram.add(
                content=op.content,
                main_content=main_content,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                metadata={"type": "user_fact"},
            )
            await self._link_to_recent(new_mem.memory_id)
            
        elif op.operation.value == "UPDATE" and op.target_id:
            # engram.update() - Modify existing memory (fact only)
            await self.engram.update(op.target_id, content=op.content)
            
        elif op.operation.value == "DELETE" and op.target_id:
            # engram.forget() - Delete outdated memory
            await self.engram.forget(op.target_id)
            new_mem = await self.engram.add(
                content=op.content,
                main_content=main_content,
                agent_id=AGENT_ID,
                user_id=USER_ID,
                metadata={"type": "user_fact"},
            )
            await self._link_to_recent(new_mem.memory_id)
    
    async def _summarize_response(self, response: str, max_length: int = 15) -> str:
        """Summarize AI response for main_content storage.
        
        Uses LLMService.summarize(text, max_length, style) from engram.llm.service.
        Falls back to simple truncation if LLM unavailable.
        
        Args:
            response: The AI response text to summarize.
            max_length: Approximate max length in words (passed to LLMService.summarize).
            
        Returns:
            Concise summary of the response.
        """
        if not self.llm:
            # Fallback: simple truncation
            words = response.split()
            if len(words) <= max_length:
                return response
            return " ".join(words[:max_length]) + "..."
        
        try:
            # LLMService.summarize(text, max_length, style) - from engram/llm/service.py
            summary = await self.llm.summarize(
                response,
                max_length=max_length,
                style="concise",
            )
            return summary
        except Exception:
            # Fallback on error
            words = response.split()
            if len(words) <= max_length:
                return response
            return " ".join(words[:max_length]) + "..."
    
    async def _link_to_recent(self, memory_id: str):
        """
        engram.relate() - Create relations between memories
        Links new memory to recent ones for graph traversal
        """
        if not self.engram:
            return
        
        try:
            recent = await self.engram.list_recent(
                agent_id=AGENT_ID,
                user_id=USER_ID,
                limit=3,
            )
            for mem in recent:
                if mem.memory_id != memory_id:
                    await self.engram.relate(
                        source_id=memory_id,
                        target_id=mem.memory_id,
                        relation_type="related_to",
                    )
                    break  # Link to most recent only
        except Exception:
            pass
    
    # =========================================================================
    # Chat
    # =========================================================================
    
    async def chat(self, user_input: str) -> str:
        """Generate response with memory context."""
        if not self.llm:
            return "Error: LLM not configured"
        
        # Long-term memory via hybrid search
        memories = await self.recall(user_input)
        
        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memories:
            messages.append({
                "role": "system", 
                "content": f"<memories>\n{memories}\n</memories>"
            })
        
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
        """engram.traverse() - Show memory relations."""
        if not self.engram:
            return
        
        memories = await self.engram.list_recent(
            agent_id=AGENT_ID,
            user_id=USER_ID,
            limit=1,
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
  /graph        Show memory graph (engram.traverse)
  /forget       Clear all (engram.purge)
  /help         This help
  /quit         Exit

Engram API Used:
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
                user_input = input("You: ").strip()
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
                    elif cmd == "/forget":
                        if input("Delete all? (y/n): ").lower() == "y":
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