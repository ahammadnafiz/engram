## User API Guide

### 1. Installation & Setup (30 seconds)

```python
# pip install engram

from engram import Engram

# User provides their own embedding function
async def my_embed(texts: list[str]) -> list[list[float]]:
    """Any embedding provider works - OpenAI, Cohere, local, etc."""
    import openai
    response = await openai.embeddings.create(
        input=texts, 
        model="text-embedding-3-small"
    )
    return [e.embedding for e in response.data]

# Initialize - that's it!
memory = Engram(
    database_url="postgresql+asyncpg://user:pass@localhost/engram",
    embedding_fn=my_embed,
    embedding_dim=1536,
)
```

### 2. Basic Usage

```python
# ============ ADD MEMORIES ============
# Simple - just content and user_id
memory_id = await memory.add(
    "User's name is Alice",
    user_id="user_123"
)

# With metadata
memory_id = await memory.add(
    "Alice prefers dark mode",
    user_id="user_123",
    metadata={"type": "preference", "category": "ui"}
)

# ============ SEARCH ============
# Simple search - returns most relevant + recent
results = await memory.search(
    "What are Alice's preferences?",
    user_id="user_123"
)

for r in results:
    print(f"[{r['score']:.2f}] {r['content']}")
# [0.92] Alice prefers dark mode
# [0.85] Alice likes Python over JavaScript

# ============ GET BY ID ============
specific = await memory.get(memory_id)

# ============ UPDATE ============
await memory.update(memory_id, content="Alice STRONGLY prefers dark mode")

# ============ DELETE ============
await memory.forget(memory_id)  # Soft delete (can be restored)
await memory.purge(memory_id)   # Hard delete (permanent)
```

### 3. Search Options

```python
# Default search - balanced scoring
results = await memory.search("user preferences", user_id="user_123")

# Prioritize recent memories
results = await memory.search(
    "user preferences",
    user_id="user_123",
    decay_weight=0.5,  # Higher = more recent bias (default 0.25)
)

# Get more results
results = await memory.search(
    "user preferences",
    user_id="user_123",
    limit=20,  # Default is 10
)

# Filter by metadata
results = await memory.search(
    "preferences",
    user_id="user_123",
    filters={"type": "preference"}  # Only preference-type memories
)
```

### 4. Sessions (Conversation Context)

```python
# Track memories within a conversation session
async with memory.session(user_id="user_123") as session:
    # Add memories to this session
    await session.add("User asked about Python frameworks")
    await session.add("Recommended FastAPI for their use case")
    await session.add("User chose FastAPI")
    
    # Search within session
    context = await session.search("what framework?", limit=5)
    
    # Get all session memories
    all_memories = await session.get_all()

# Session persists - can resume later
async with memory.session(user_id="user_123", session_id=existing_session_id) as session:
    # Continue the conversation
    await session.add("User had questions about FastAPI deployment")
```

### 5. Graph Relationships

```python
# Create relationships between memories
await memory.relate(
    source_id=mem1_id,
    target_id=mem2_id,
    relation_type="causes",  # or "relates_to", "contradicts", etc.
    weight=0.8
)

# Traverse the graph
related = await memory.traverse(
    start_memory_id=mem1_id,
    relation_types=["causes", "relates_to"],
    max_hops=2,
    min_weight=0.5
)

for r in related:
    print(f"Hop {r['hop_depth']}: {r['content']}")
```

### 6. Bulk Operations

```python
# Add many memories at once (100x faster than individual adds)
memory_ids = await memory.add_batch([
    {"content": "User likes coffee", "user_id": "user_123"},
    {"content": "User works remotely", "user_id": "user_123"},
    {"content": "User is in PST timezone", "user_id": "user_123"},
])

# List recent memories (no search, just time-ordered)
recent = await memory.list_recent(user_id="user_123", limit=20)
```

### 7. Memory Reinforcement

```python
# When a memory is useful, reinforce it (boosts score in future searches)
await memory.reinforce(memory_id)

# This internally:
# - Increments memory_strength
# - Resets last_accessed_at (resets decay)
# - Increments access_count
```

### 8. Health Check (Production)

```python
# Check system health
status = await memory.health_check()
# {"database": "healthy", "latency_ms": 2.5}

if status["database"] != "healthy":
    alert_ops_team()
```

---

## Complete Chatbot Example

```python
import asyncio
from engram import Engram

async def chatbot():
    # Setup
    async def embed(texts):
        import openai
        r = await openai.embeddings.create(input=texts, model="text-embedding-3-small")
        return [e.embedding for e in r.data]
    
    memory = Engram(
        database_url="postgresql+asyncpg://localhost/engram",
        embedding_fn=embed,
        embedding_dim=1536,
    )
    
    user_id = "alice_123"
    
    # Simulate conversation
    async with memory.session(user_id=user_id) as session:
        
        # User message 1
        user_input = "Hi! I'm Alice and I work at TechCorp"
        await session.add(f"User said: {user_input}")
        
        # User message 2  
        user_input = "I prefer Python for backend development"
        await session.add(f"User said: {user_input}")
        
        # Later... user asks a question
        user_input = "What programming languages have we discussed?"
        
        # Get relevant context
        context = await session.search(user_input, limit=5)
        
        # Build prompt with memory context
        context_str = "\n".join([f"- {m['content']}" for m in context])
        prompt = f"""Previous context:
{context_str}

User: {user_input}
Assistant:"""
        
        print(prompt)
        # Previous context:
        # - User said: I prefer Python for backend development
        # - User said: Hi! I'm Alice and I work at TechCorp
        #
        # User: What programming languages have we discussed?
        # Assistant:
    
    await memory.close()

asyncio.run(chatbot())
```

---

## API Summary

| Method | Purpose | Example |
|--------|---------|---------|
| `add()` | Store a memory | `await memory.add("fact", user_id="u1")` |
| `add_batch()` | Store many memories | `await memory.add_batch([...])` |
| `search()` | Find relevant memories | `await memory.search("query", user_id="u1")` |
| `get()` | Get by ID | `await memory.get(memory_id)` |
| `list_recent()` | Get recent (no search) | `await memory.list_recent(user_id="u1")` |
| `update()` | Modify content | `await memory.update(id, content="new")` |
| `reinforce()` | Boost memory | `await memory.reinforce(id)` |
| `forget()` | Soft delete | `await memory.forget(id)` |
| `purge()` | Hard delete | `await memory.purge(id)` |
| `relate()` | Create relationship | `await memory.relate(id1, id2, "causes")` |
| `traverse()` | Graph traversal | `await memory.traverse(id, max_hops=2)` |
| `session()` | Conversation context | `async with memory.session(...) as s:` |
| `health_check()` | System status | `await memory.health_check()` |
| `close()` | Cleanup | `await memory.close()` |

The API is designed to be **intuitive** - users don't need to understand SQL, embeddings, or decay formulas. They just `add()` and `search()`.