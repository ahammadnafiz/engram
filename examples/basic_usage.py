#!/usr/bin/env python3
"""
Engram Comprehensive Demo
=========================

Advanced demo showcasing Engram's full capabilities with realistic content,
complex relationships, and thorough testing of all API operations.

Two-Column Memory System:
    Engram uses a cost-effective two-column approach:
    - `content` (fact): The extracted fact - EMBEDDED for hybrid search
    - `main_content`: Full conversation context - NOT embedded (free storage)
    
    This preserves full context without increasing embedding API costs.

This demo uses sentence-transformers (local, free embeddings) by default,
but works with ANY provider thanks to Engram's pluggable architecture.

Provider Options:
    # Local (free) - Default for this demo
    ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
    ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
    
    # OpenAI (cloud)
    ENGRAM_EMBEDDING_PROVIDER=openai
    ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
    ENGRAM_OPENAI_API_KEY=sk-...
    
    # Cohere
    ENGRAM_EMBEDDING_PROVIDER=cohere
    ENGRAM_EMBEDDING_MODEL=embed-english-v3.0
    ENGRAM_COHERE_API_KEY=...
    
    # Ollama (local)
    ENGRAM_EMBEDDING_PROVIDER=ollama
    ENGRAM_EMBEDDING_MODEL=nomic-embed-text

Usage:
    # Default (sentence-transformers)
    python examples/basic_usage.py
    
    # With OpenAI
    ENGRAM_EMBEDDING_PROVIDER=openai python examples/basic_usage.py

Operations Demonstrated:
    - health_check()  - System health verification
    - add()           - Individual memory storage (with main_content)
    - add_batch()     - Bulk memory insertion
    - search()        - Hybrid semantic + keyword search
    - get()           - Retrieve by ID
    - list_recent()   - Chronological listing
    - update()        - Modify existing memories
    - reinforce()     - Boost importance scores
    - relate()        - Create knowledge graph edges
    - traverse()      - Multi-hop graph exploration
    - session()       - Scoped conversation context
    - forget()        - Delete single memory
    - purge()         - Bulk delete by agent
"""

import asyncio
import os

# Default to local embeddings (no API key required)
os.environ.setdefault("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
os.environ.setdefault("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Load .env if exists (allows overriding with any provider)
from pathlib import Path
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def print_header(title: str, emoji: str = "📌"):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{emoji} {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n  ▶ {title}")
    print("  " + "-" * 40)


async def main():
    from engram import Engram
    
    print_header("🧠 Engram Comprehensive Demo", "🚀")
    
    # Show which provider is being used
    provider = os.environ.get("ENGRAM_EMBEDDING_PROVIDER", "sentence-transformers")
    model = os.environ.get("ENGRAM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    print(f"""
    Provider: {provider}
    Model:    {model}
    
    This demo simulates an AI assistant learning about a user
    over multiple conversations, building a knowledge graph,
    and retrieving relevant context for conversations.
    """)
    
    async with Engram() as memory:
        agent_id = "personal-assistant"
        user_id = "sarah_chen"
        
        # ================================================================
        # PHASE 1: HEALTH CHECK
        # ================================================================
        print_header("Health Check", "🏥")
        status = await memory.health_check()
        components = status.get('components', {})
        db_status = components.get('database', {})
        emb_status = components.get('embedding', {})
        print(f"  Overall:    {status.get('status', 'unknown')}")
        print(f"  Database:   {db_status.get('status', 'unknown')}")
        print(f"  Embeddings: {emb_status.get('status', 'unknown')}")
        print(f"  Dimension:  {emb_status.get('dimension', 'unknown')}")
        print(f"  Version:    {status.get('version', 'unknown')}")
        
        # ================================================================
        # PHASE 2: BUILDING USER PROFILE (Complex Content)
        # ================================================================
        print_header("Building User Profile", "👤")
        
        # Personal Information
        # Demonstrating the TWO-COLUMN MEMORY SYSTEM:
        #   - content: The fact (EMBEDDED for search)
        #   - main_content: Full conversation context (NOT embedded, preserved for context)
        print_subheader("Personal Details (Two-Column System)")
        personal_memories = []
        
        personal_memories.append(await memory.add(
            content="Sarah Chen is a 32-year-old software architect living in Seattle",
            main_content="[USER]: I'm Sarah Chen, 32, and I work as a software architect. I moved to Seattle from San Francisco in 2022.\n[AI]: Welcome to Seattle! That's an exciting career move.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "personal", "confidence": 0.95}
        ))
        print(f"  ✓ Added: Personal background (with context)")
        
        personal_memories.append(await memory.add(
            content="Sarah has a golden retriever named Apollo who is 4 years old",
            main_content="[USER]: I have a golden retriever named Apollo, he's 4. Adopted him from a shelter!\n[AI]: Apollo sounds wonderful! Rescue dogs are the best.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "personal", "subcategory": "pets"}
        ))
        print(f"  ✓ Added: Pet information (with context)")
        
        personal_memories.append(await memory.add(
            content="Sarah is allergic to shellfish - discovered during Maine trip at age 25",
            main_content="[USER]: I'm allergic to shellfish. Found out the hard way in Maine when I was 25.\n[AI]: That's important to know. I'll remember to avoid seafood recommendations.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "health", "severity": "high"}
        ))
        print(f"  ✓ Added: Health/allergy info (with context)")
        
        # Professional Information
        print_subheader("Professional Details")
        work_memories = []
        
        work_memories.append(await memory.add(
            content="Sarah works as a Principal Software Architect at CloudScale Inc, focusing on distributed systems and microservices architecture. She leads a team of 8 engineers.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "work", "company": "CloudScale Inc"}
        ))
        print(f"  ✓ Added: Current job")
        
        work_memories.append(await memory.add(
            content="Sarah is an expert in Python, Go, and Rust. She prefers Rust for performance-critical systems but uses Python for rapid prototyping and data analysis scripts.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "skills", "type": "programming"}
        ))
        print(f"  ✓ Added: Programming skills")
        
        work_memories.append(await memory.add(
            content="Sarah is currently working on a project called 'Project Aurora' - a real-time data streaming platform that needs to handle 10 million events per second with sub-millisecond latency.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "work", "project": "Aurora", "priority": "high"}
        ))
        print(f"  ✓ Added: Current project")
        
        work_memories.append(await memory.add(
            content="Sarah mentioned she's frustrated with her current CI/CD pipeline - builds take over 45 minutes and the flaky tests are causing deployment delays. She's considering migrating to Bazel.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "work", "sentiment": "frustrated", "topic": "devops"}
        ))
        print(f"  ✓ Added: Work frustration")
        
        # Preferences
        print_subheader("Preferences & Interests")
        pref_memories = []
        
        pref_memories.append(await memory.add(
            content="Sarah prefers dark mode in all applications and IDEs. She uses the Dracula theme in VS Code and has her terminal configured with Catppuccin Mocha colors.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "preferences", "type": "ui"}
        ))
        print(f"  ✓ Added: UI preferences")
        
        pref_memories.append(await memory.add(
            content="For productivity, Sarah uses the Pomodoro technique with 45-minute focus sessions. She blocks all notifications during deep work and prefers async communication over meetings.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "preferences", "type": "productivity"}
        ))
        print(f"  ✓ Added: Productivity style")
        
        pref_memories.append(await memory.add(
            content="Sarah is learning Japanese in her spare time using Anki flashcards and watching anime without subtitles. She's at JLPT N3 level and plans to take N2 next year.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "hobbies", "type": "language_learning"}
        ))
        print(f"  ✓ Added: Language learning hobby")
        
        pref_memories.append(await memory.add(
            content="Sarah is an avid hiker and has completed several trails in the Cascades. Her goal is to summit Mount Rainier next summer. She typically hikes with Apollo on weekends.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "hobbies", "type": "outdoor"}
        ))
        print(f"  ✓ Added: Hiking hobby")
        
        pref_memories.append(await memory.add(
            content="Sarah drinks oat milk lattes - she's lactose intolerant but not vegan. Her favorite coffee shop is 'Analog Coffee' near her office where she goes every morning around 8:30 AM.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"category": "preferences", "type": "food", "routine": True}
        ))
        print(f"  ✓ Added: Coffee preference")
        
        # ================================================================
        # PHASE 3: BATCH INSERT (Recent Conversations)
        # ================================================================
        print_header("Batch Insert - Recent Conversations", "📦")
        
        # Batch insert also supports main_content for two-column system
        conversation_batch = [
            {
                "content": "Sarah wants to implement circuit breakers in microservices using Hystrix pattern in Go",
                "main_content": "[USER]: How do I implement circuit breakers in Go?\n[AI]: The Hystrix pattern is great for preventing cascade failures.",
                "agent_id": agent_id,
                "user_id": user_id,
                "metadata": {"category": "technical", "topic": "resilience"}
            },
            {
                "content": "Sarah's team is adopting OpenTelemetry for distributed tracing, migrating from Jaeger",
                "main_content": "[USER]: We're moving to OpenTelemetry from Jaeger.\n[AI]: Good choice for vendor-neutral instrumentation!",
                "agent_id": agent_id,
                "user_id": user_id,
                "metadata": {"category": "technical", "topic": "observability"}
            },
            {
                "content": "Sarah submitted a KubeCon talk proposal about Kubernetes to Nomad migration",
                "agent_id": agent_id,
                "user_id": user_id,
                "metadata": {"category": "events", "event": "KubeCon"}
            },
            {
                "content": "Sarah's birthday is March 15th - prefers small gatherings over surprise parties",
                "main_content": "[USER]: My birthday is March 15th. I don't like surprises.\n[AI]: Noted! Small gatherings it is.",
                "agent_id": agent_id,
                "user_id": user_id,
                "metadata": {"category": "personal", "type": "birthday"}
            },
            {
                "content": "Sarah is re-reading 'Designing Data-Intensive Applications' - considers it the best technical book",
                "agent_id": agent_id,
                "user_id": user_id,
                "metadata": {"category": "books", "genre": "technical"}
            },
            {
                "content": "Sarah has 1:1 with manager every Tuesday at 2 PM for career discussions",
                "agent_id": agent_id,
                "user_id": user_id,
                "metadata": {"category": "work", "type": "meeting", "recurring": True}
            },
        ]
        
        batch_results = await memory.add_batch(conversation_batch)
        print(f"  ✓ Added {len(batch_results)} conversation memories in batch")
        for mem in batch_results:
            print(f"    - {mem.content[:60]}...")
        
        # ================================================================
        # PHASE 4: SEARCH TESTING
        # ================================================================
        print_header("Search Testing - Hybrid Search", "🔍")
        
        # Test 1: Semantic search for programming
        print_subheader("Test 1: Semantic - 'What programming languages?'")
        results = await memory.search(
            query="What programming languages does she know and prefer?",
            agent_id=agent_id,
            user_id=user_id,
            limit=3
        )
        for r in results:
            fact = r.memory.fact or r.memory.content
            print(f"  [{r.score:.3f}] {fact[:70]}...")
            # Show main_content if available (two-column system)
            if r.memory.main_content:
                ctx = r.memory.main_content[:60].replace("\n", " ")
                print(f"           └─ Context: {ctx}...")
        
        # Test 2: Search for work-related context
        print_subheader("Test 2: Work Context - 'current project challenges'")
        results = await memory.search(
            query="What challenges is she facing at work right now?",
            agent_id=agent_id,
            user_id=user_id,
            limit=3
        )
        for r in results:
            print(f"  [{r.score:.3f}] {r.memory.content[:70]}...")
        
        # Test 3: Food and dietary restrictions (shows main_content)
        print_subheader("Test 3: Dietary - 'food allergies restrictions'")
        results = await memory.search(
            query="Does she have any food allergies or dietary restrictions?",
            agent_id=agent_id,
            user_id=user_id,
            limit=3
        )
        for r in results:
            fact = r.memory.fact or r.memory.content
            print(f"  [{r.score:.3f}] {fact[:70]}...")
            # Display main_content to show conversation context is preserved
            if r.memory.main_content:
                ctx = r.memory.main_content[:80].replace("\n", " | ")
                print(f"           └─ {ctx}...")
        
        # Test 4: Hobbies and interests
        print_subheader("Test 4: Hobbies - 'weekend activities interests'")
        results = await memory.search(
            query="What does she do for fun on weekends?",
            agent_id=agent_id,
            user_id=user_id,
            limit=4
        )
        for r in results:
            print(f"  [{r.score:.3f}] {r.memory.content[:70]}...")
        
        # Test 5: Specific keyword search
        print_subheader("Test 5: Keyword - 'Apollo dog'")
        results = await memory.search(
            query="Apollo golden retriever dog pet",
            agent_id=agent_id,
            user_id=user_id,
            limit=2
        )
        for r in results:
            print(f"  [{r.score:.3f}] {r.memory.content[:70]}...")
        
        # Test 6: Technical tools
        print_subheader("Test 6: Technical - 'Kubernetes observability tools'")
        results = await memory.search(
            query="What tools is she using for Kubernetes observability and tracing?",
            agent_id=agent_id,
            user_id=user_id,
            limit=3
        )
        for r in results:
            print(f"  [{r.score:.3f}] {r.memory.content[:70]}...")
        
        # ================================================================
        # PHASE 5: GET & UPDATE
        # ================================================================
        print_header("Get & Update Operations", "✏️")
        
        # Get specific memory
        print_subheader("Get by ID")
        work_mem = work_memories[0]
        retrieved = await memory.get(work_mem.memory_id)
        print(f"  ID: {retrieved.memory_id}")
        print(f"  Content: {retrieved.content[:60]}...")
        print(f"  Importance: {retrieved.importance}")
        print(f"  Access Count: {retrieved.access_count}")
        print(f"  Created: {retrieved.created_at}")
        
        # Update memory with new information
        print_subheader("Update Memory")
        print(f"  Before: {work_memories[2].content[:50]}...")
        updated = await memory.update(
            memory_id=work_memories[2].memory_id,
            content="Sarah successfully launched 'Project Aurora' - the real-time data streaming platform now handles 15 million events per second with 0.5ms latency. The team celebrated with a launch party.",
            metadata={"status": "completed", "celebration": True}
        )
        print(f"  After:  {updated.content[:50]}...")
        
        # ================================================================
        # PHASE 6: REINFORCE (Importance Boosting)
        # ================================================================
        print_header("Reinforce - Importance Boosting", "💪")
        
        # Simulate the allergy info being useful in conversation
        allergy_mem = personal_memories[2]  # Shellfish allergy
        
        print_subheader("Reinforcing Critical Health Info")
        print(f"  Memory: {allergy_mem.content[:50]}...")
        
        before = await memory.get(allergy_mem.memory_id)
        print(f"  Before: importance={before.importance:.2f}, access_count={before.access_count}")
        
        # Reinforce multiple times (as if it's been useful in several conversations)
        for i in range(3):
            await memory.reinforce(allergy_mem.memory_id)
        
        after = await memory.get(allergy_mem.memory_id)
        print(f"  After (3x reinforce): importance={after.importance:.2f}, access_count={after.access_count}")
        
        # Now search should rank this higher
        print_subheader("Search After Reinforcement")
        results = await memory.search(
            query="health information allergies",
            agent_id=agent_id,
            limit=2
        )
        for r in results:
            print(f"  [{r.score:.3f}] importance={r.memory.importance:.2f} | {r.memory.content[:50]}...")
        
        # ================================================================
        # PHASE 7: LIST RECENT
        # ================================================================
        print_header("List Recent Memories", "📋")
        
        recent = await memory.list_recent(agent_id=agent_id, limit=5)
        print(f"  Most recent {len(recent)} memories:")
        for i, m in enumerate(recent, 1):
            print(f"  {i}. [{m.created_at.strftime('%H:%M:%S')}] {m.content[:55]}...")
        
        # ================================================================
        # PHASE 8: CREATE RELATIONSHIPS
        # ================================================================
        print_header("Create Knowledge Graph Relationships", "🔗")
        
        # Link programming skills to current project
        print_subheader("Linking Skills to Project")
        await memory.relate(
            source_id=work_memories[1].memory_id,  # Programming skills (Python, Go, Rust)
            target_id=work_memories[2].memory_id,  # Project Aurora
            relation_type="supports",
            weight=0.95
        )
        print(f"  ✓ Skills --[supports]--> Project Aurora")
        
        # Link hiking hobby to pet
        print_subheader("Linking Hobbies")
        await memory.relate(
            source_id=pref_memories[3].memory_id,  # Hiking hobby
            target_id=personal_memories[1].memory_id,  # Apollo the dog
            relation_type="related_to",
            weight=0.9
        )
        print(f"  ✓ Hiking --[related_to]--> Apollo (hiking companion)")
        
        # Link work frustration to CI/CD topic
        await memory.relate(
            source_id=work_memories[3].memory_id,  # CI/CD frustration
            target_id=work_memories[2].memory_id,  # Project Aurora
            relation_type="related_to",
            weight=0.8
        )
        print(f"  ✓ CI/CD issues --[related_to]--> Project Aurora")
        
        # Link coffee to productivity
        await memory.relate(
            source_id=pref_memories[4].memory_id,  # Coffee preference
            target_id=pref_memories[1].memory_id,  # Productivity style
            relation_type="supports",
            weight=0.7
        )
        print(f"  ✓ Morning coffee --[supports]--> Productivity routine")
        
        # Link Seattle to hiking
        await memory.relate(
            source_id=personal_memories[0].memory_id,  # Lives in Seattle
            target_id=pref_memories[3].memory_id,  # Hiking (Cascades)
            relation_type="related_to",
            weight=0.85
        )
        print(f"  ✓ Seattle location --[related_to]--> Cascade hiking")
        
        # ================================================================
        # PHASE 9: GRAPH TRAVERSAL
        # ================================================================
        print_header("Graph Traversal", "🌐")
        
        print_subheader("Traverse from Project Aurora (inbound - what links TO it)")
        related = await memory.traverse(
            start_memory_id=work_memories[2].memory_id,  # Project Aurora
            max_depth=2,
            min_weight=0.5,
            direction="inbound"  # Skills and CI/CD link TO Aurora
        )
        print(f"  Starting from: Project Aurora")
        print(f"  Found {len(related)} connected memories:")
        for r in related:
            indent = "    " * r.depth
            print(f"  {indent}[depth={r.depth}] {r.content[:50]}...")
        
        print_subheader("Traverse from Hiking (2 hops, any direction)")
        related = await memory.traverse(
            start_memory_id=pref_memories[3].memory_id,  # Hiking
            max_depth=2,
            min_weight=0.5,
            direction="any"  # Follow relations in both directions
        )
        print(f"  Starting from: Hiking hobby")
        print(f"  Found {len(related)} connected memories:")
        for r in related:
            indent = "    " * r.depth
            print(f"  {indent}[depth={r.depth}] {r.content[:50]}...")
        
        # ================================================================
        # PHASE 10: SESSION CONTEXT
        # ================================================================
        print_header("Session-Scoped Conversation", "💬")
        
        async with memory.session(agent_id=agent_id, user_id=user_id) as session:
            print(f"  Session ID: {session.session_id}")
            
            # Simulate a multi-turn conversation
            print_subheader("Adding Conversation Turns")
            
            turn1 = await memory.add(
                content="User: 'I need help preparing for my KubeCon talk. What angle should I take?'",
                agent_id=agent_id,
                user_id=user_id,
                session_id=session.session_id,
                metadata={"turn": 1, "speaker": "user"}
            )
            print(f"  Turn 1: User asks about KubeCon talk")
            
            turn2 = await memory.add(
                content="Assistant: 'Based on your experience with Project Aurora, you could focus on the real-world challenges of handling 15M events/sec. What specific aspect interests you most?'",
                agent_id=agent_id,
                user_id=user_id,
                session_id=session.session_id,
                metadata={"turn": 2, "speaker": "assistant"}
            )
            print(f"  Turn 2: Assistant suggests angle")
            
            turn3 = await memory.add(
                content="User: 'I want to discuss our migration from Kubernetes to Nomad for edge computing - the tradeoffs and lessons learned.'",
                agent_id=agent_id,
                user_id=user_id,
                session_id=session.session_id,
                metadata={"turn": 3, "speaker": "user"}
            )
            print(f"  Turn 3: User clarifies focus")
            
            turn4 = await memory.add(
                content="Assistant: 'Great choice! Your hands-on experience with both platforms is valuable. Shall we outline the key migration challenges you faced?'",
                agent_id=agent_id,
                user_id=user_id,
                session_id=session.session_id,
                metadata={"turn": 4, "speaker": "assistant"}
            )
            print(f"  Turn 4: Assistant confirms direction")
            
            # Search within this conversation context
            print_subheader("Search for Relevant Context")
            context = await memory.search(
                query="KubeCon presentation topic migration",
                agent_id=agent_id,
                limit=5
            )
            print(f"  Query: 'KubeCon presentation topic migration'")
            for c in context:
                is_session = c.memory.session_id == session.session_id
                marker = "📍" if is_session else "  "
                print(f"  {marker}[{c.score:.3f}] {c.memory.content[:55]}...")
        
        print(f"  Session ended: {session.session_id}")
        
        # ================================================================
        # PHASE 11: FORGET (Delete Single Memory)
        # ================================================================
        print_header("Forget - Delete Single Memory", "🗑️")
        
        # Add some temporary memories
        temp1 = await memory.add(
            content="Sarah mentioned she's considering buying a new MacBook Pro M3 but hasn't decided yet.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"temporary": True}
        )
        temp2 = await memory.add(
            content="Sarah asked about a random trivia question that's not worth remembering.",
            agent_id=agent_id,
            user_id=user_id,
            metadata={"temporary": True}
        )
        
        print(f"  Created 2 temporary memories:")
        print(f"    - {temp1.memory_id[:20]}... (MacBook)")
        print(f"    - {temp2.memory_id[:20]}... (trivia)")
        
        # Search before delete
        print_subheader("Before Deletion")
        results = await memory.search("MacBook Pro trivia", agent_id=agent_id, limit=3)
        print(f"  Found {len(results)} results for 'MacBook Pro trivia'")
        for r in results:
            if "MacBook" in r.memory.content or "trivia" in r.memory.content:
                print(f"    ✓ Found: {r.memory.content[:50]}...")
        
        # Delete using forget()
        deleted1 = await memory.forget(temp1.memory_id)
        deleted2 = await memory.forget(temp2.memory_id)
        print(f"  ✓ Deleted memory 1: {deleted1}")
        print(f"  ✓ Deleted memory 2: {deleted2}")
        
        # Search after delete
        print_subheader("After Deletion")
        results = await memory.search("MacBook Pro trivia", agent_id=agent_id, limit=3)
        found_deleted = any("MacBook" in r.memory.content or "trivia" in r.memory.content for r in results)
        print(f"  Found deleted memories in search: {found_deleted}")
        
        # Verify with get
        try:
            await memory.get(temp1.memory_id)
            print(f"  Error: Memory 1 still exists!")
        except Exception as e:
            print(f"  ✓ Memory 1 confirmed deleted ({type(e).__name__})")
        
        # ================================================================
        # PHASE 12: PURGE (Bulk Delete by Agent)
        # ================================================================
        print_header("Purge - Bulk Delete Demo", "💀")
        
        # Create a separate agent for purge demo
        purge_agent = "purge-test-agent"
        
        # Add some memories for the test agent
        await memory.add(
            content="Test memory 1 for purge demo",
            agent_id=purge_agent,
            metadata={"test": True}
        )
        await memory.add(
            content="Test memory 2 for purge demo",
            agent_id=purge_agent,
            metadata={"test": True}
        )
        await memory.add(
            content="Test memory 3 for purge demo",
            agent_id=purge_agent,
            metadata={"test": True}
        )
        print(f"  Created 3 test memories for agent '{purge_agent}'")
        
        # Verify they exist
        before_purge = await memory.list_recent(agent_id=purge_agent, limit=10)
        print(f"  Before purge: {len(before_purge)} memories")
        
        # Purge all memories for this agent
        deleted_count = await memory.purge(agent_id=purge_agent)
        print(f"  ✓ Purged {deleted_count} memories")
        
        # Verify they're gone
        after_purge = await memory.list_recent(agent_id=purge_agent, limit=10)
        print(f"  After purge: {len(after_purge)} memories")
        
        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        print_header("Demo Complete!", "🎉")
        
        # Final stats
        final_memories = await memory.list_recent(agent_id=agent_id, limit=100)
        print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║                    DEMO STATISTICS                       ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Embedding Provider:    {provider:<30} ║
  ║  Embedding Model:       {model:<30} ║
  ║  Total memories:        ~25+                             ║
  ║  Relationships created: 5                                ║
  ║  Search queries tested: 8                                ║
  ║  Sessions created:      1                                ║
  ║  Memories forgotten:    2                                ║
  ║  Memories purged:       3                                ║
  ╚══════════════════════════════════════════════════════════╝
        """)
        
        print("""
  Operations Demonstrated:
  ━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ health_check()   - System health verification
  ✓ add()            - Individual memory storage (with main_content)
  ✓ add_batch()      - Bulk memory insertion (with main_content)
  ✓ search()         - Hybrid semantic + keyword search
  ✓ get()            - Retrieve by ID
  ✓ list_recent()    - Chronological listing
  ✓ update()         - Modify existing memories
  ✓ reinforce()      - Boost importance scores
  ✓ relate()         - Create knowledge graph edges
  ✓ traverse()       - Multi-hop graph exploration
  ✓ session()        - Scoped conversation context
  ✓ forget()         - Delete single memory
  ✓ purge()          - Bulk delete by agent
  
  Two-Column Memory System:
  ━━━━━━━━━━━━━━━━━━━━━━━━━
  • content (fact)  - Embedded for hybrid search ($)
  • main_content    - Full context preserved (free)
  
  Cost-effective: Only facts are embedded!
  
  Provider-Agnostic Architecture:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  This demo works with ANY embedding provider:
  • sentence-transformers (local, free)
  • openai (cloud)
  • cohere (cloud)
  • ollama (local)
  • huggingface (cloud/local)
  
  Just set ENGRAM_EMBEDDING_PROVIDER to switch!
  
  🧠 Engram: AI Memory Made Simple
        """)


if __name__ == "__main__":
    asyncio.run(main())
