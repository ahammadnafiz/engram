# API Reference

Complete API documentation for Engram.

## Engram Client

The main entry point for all memory operations.

```python
from engram import Engram

async with Engram() as engram:
    # Use engram methods here
    ...
```

### Connection

#### `connect()`

Connect to the database and initialize services.

```python
engram = Engram()
await engram.connect()
# ... use engram ...
await engram.close()
```

#### `close()`

Close connections and cleanup resources.

#### Context Manager

Recommended way to use Engram:

```python
async with Engram() as engram:
    memory = await engram.add(content="Hello", agent_id="agent_1")
# Automatically closes on exit
```

---

## Memory Operations

### `add()`

Add a new memory.

```python
memory = await engram.add(
    content: str,              # Required: the fact to store (embedded)
    agent_id: str,             # Required: agent ID
    user_id: str = None,       # Optional: user ID
    session_id: str = None,    # Optional: session ID
    main_content: str = None,  # Optional: full context (NOT embedded)
    metadata: dict = None,     # Optional: key-value metadata
) -> Memory
```

**Two-Column System:**
- `content` → Stored in `fact` column, **embedded** for semantic search
- `main_content` → Stored separately, **NOT embedded** (cost-effective)

**Example:**
```python
# Basic usage (fact only)
memory = await engram.add(
    content="User prefers dark mode",
    agent_id="assistant",
    user_id="user_123",
    metadata={"type": "preference"}
)

# With conversation context (two-column)
memory = await engram.add(
    content="User prefers dark mode",  # Fact (embedded)
    agent_id="assistant",
    user_id="user_123",
    main_content="[USER]: I like dark themes\n[AI]: Noted your preference!",  # Context (not embedded)
)
print(f"Created: {memory.memory_id}")
```

---

### `add_batch()`

Add multiple memories efficiently (batch embedding).

```python
memories = await engram.add_batch(
    memories: list[dict],  # List of memory dicts
) -> list[Memory]
```

**Example:**
```python
memories = await engram.add_batch([
    {"content": "User likes Python", "agent_id": "assistant"},
    {"content": "User works in finance", "agent_id": "assistant", "metadata": {"source": "chat"}},
    # With two-column system
    {
        "content": "User is learning ML",  # Fact (embedded)
        "agent_id": "assistant",
        "main_content": "[USER]: I'm taking an ML course\n[AI]: That's great!",  # Context
    },
])
print(f"Created {len(memories)} memories")
```

---

### `get()`

Get a memory by ID. Also updates access timestamp.

```python
memory = await engram.get(
    memory_id: str,  # Memory ID
) -> Memory
```

**Raises:** `MemoryNotFoundError` if not found.

**Example:**
```python
memory = await engram.get("mem_abc123")
print(f"Content: {memory.content}")
print(f"Importance: {memory.importance}")
```

---

### `update()`

Update an existing memory.

```python
memory = await engram.update(
    memory_id: str,           # Memory ID
    content: str = None,      # New content (re-embeds)
    importance: float = None, # New importance (0.0-1.0)
    metadata: dict = None,    # Metadata to merge
) -> Memory
```

**Example:**
```python
# Update content
memory = await engram.update("mem_abc123", content="User prefers light mode now")

# Update importance
memory = await engram.update("mem_abc123", importance=0.9)

# Add metadata
memory = await engram.update("mem_abc123", metadata={"verified": True})
```

---

### `reinforce()`

Boost a memory's importance (for memory decay system).

```python
memory = await engram.reinforce(
    memory_id: str,             # Memory ID
    importance_boost: float = 0.1,  # Amount to boost (capped at 1.0)
) -> Memory
```

**Example:**
```python
# Memory was useful, boost its importance
memory = await engram.reinforce("mem_abc123", importance_boost=0.15)
print(f"New importance: {memory.importance}")
```

---

### `forget()`

Delete a single memory.

```python
deleted = await engram.forget(
    memory_id: str,  # Memory ID
) -> bool  # True if deleted, False if not found
```

**Example:**
```python
if await engram.forget("mem_abc123"):
    print("Memory deleted")
```

---

### `purge()`

Delete all memories for an agent.

```python
count = await engram.purge(
    agent_id: str,         # Agent ID
    user_id: str = None,   # Optional: filter by user
) -> int  # Number deleted
```

**Example:**
```python
count = await engram.purge(agent_id="assistant")
print(f"Deleted {count} memories")
```

---

### `list_recent()`

List recent memories ordered by creation time.

```python
memories = await engram.list_recent(
    agent_id: str,         # Agent ID
    user_id: str = None,   # Optional: filter by user
    limit: int = 10,       # Max results
) -> list[Memory]
```

**Example:**
```python
memories = await engram.list_recent(agent_id="assistant", limit=5)
for m in memories:
    print(f"[{m.importance:.0%}] {m.content}")
```

---

## Search Operations

### `search()`

Search memories using hybrid search (semantic + keyword + decay + importance).

```python
results = await engram.search(
    query: str,              # Search query
    agent_id: str,           # Agent ID
    user_id: str = None,     # Optional: filter by user
    limit: int = 10,         # Max results
    min_score: float = 0.0,  # Minimum score threshold
) -> list[SearchResult]
```

**How It Works:**

Engram uses **hybrid search** by default, combining:
- **Semantic similarity** - Vector embeddings for meaning
- **Keyword matching** - BM25/full-text for exact terms
- **Time decay** - Recent memories rank higher
- **Importance scoring** - Frequently used memories rank higher

Results are fused using Reciprocal Rank Fusion (RRF).

**Example:**
```python
# Hybrid search (default)
results = await engram.search(
    query="user preferences for UI",
    agent_id="assistant",
    limit=5,
)

for r in results:
    print(f"[{r.score:.0%}] {r.memory.content}")
```

**SearchResult Fields:**

```python
result.score          # Combined relevance score (0.0-1.0)
result.memory         # The Memory object
result.memory.content # Memory content
result.memory.importance  # Importance score
result.memory.metadata    # Metadata dict
```

---

## Graph Operations

### `relate()`

Create a relation between two memories.

```python
await engram.relate(
    source_id: str,                    # Source memory ID
    target_id: str,                    # Target memory ID
    relation_type: str = "related_to", # Relation type
    weight: float = 1.0,               # Relation weight (0.0-1.0)
    metadata: dict = None,             # Optional metadata
)
```

**Relation Types:**
- `related_to` - General association
- `causes` - Causal relationship
- `supports` - Supporting evidence
- `contradicts` - Conflicting information
- `temporal` - Time-based sequence
- `part_of` - Hierarchical membership

**Example:**
```python
# Create a memory chain
mem1 = await engram.add(content="User wants to learn Python", agent_id="assistant")
mem2 = await engram.add(content="User enrolled in Python course", agent_id="assistant")

await engram.relate(
    source_id=mem1.memory_id,
    target_id=mem2.memory_id,
    relation_type="causes",
)
```

---

### `traverse()`

Traverse the memory graph from a starting point.

```python
results = await engram.traverse(
    start_memory_id: str,              # Starting memory ID
    max_depth: int = 3,                # Max traversal depth
    direction: str = "outbound",       # Direction: "outbound", "inbound", "any"
    relation_types: list[str] = None,  # Filter by relation types
    min_weight: float = 0.0,           # Minimum weight to follow
    limit: int = 50,                   # Max results
) -> list[TraversalResult]
```

**Example:**
```python
results = await engram.traverse(
    start_memory_id="mem_abc123",
    max_depth=2,
    direction="outbound",
)

for r in results:
    print(f"Depth {r.depth}: {r.content}")
```

**TraversalResult Fields:**

```python
result.memory_id      # Memory ID
result.content        # Memory content
result.depth          # Hops from start
result.relation_type  # How it was reached
result.path           # Full path from start
```

---

## Session Management

### `session()`

Create a conversation session (async context manager).

```python
async with engram.session(
    agent_id: str,         # Agent ID
    user_id: str = None,   # Optional: user ID
    metadata: dict = None, # Optional: session metadata
) as session:
    # session.session_id
    # session.is_active
    ...
```

**Example:**
```python
async with engram.session(agent_id="assistant", user_id="user_123") as session:
    print(f"Session: {session.session_id}")
    
    # Add memory to this session
    memory = await engram.add(
        content="User asked about weather",
        agent_id="assistant",
        session_id=session.session_id,
    )
# Session automatically ended on exit
```

---

## Health Check

### `health_check()`

Check system health.

```python
status = await engram.health_check() -> dict
```

**Returns:**
```python
{
    "status": "healthy",  # or "unhealthy"
    "components": {
        "database": {"status": "healthy", "latency_ms": 5.2},
        "embedding": {"status": "healthy", "provider": "openai"},
    }
}
```

**Example:**
```python
health = await engram.health_check()
if health["status"] == "healthy":
    print("All systems operational")
```

---

## Services

### EmbeddingService

Generate vector embeddings.

```python
from engram import EmbeddingService

# From settings
embedding = EmbeddingService.from_settings()

# From specific provider
embedding = EmbeddingService.from_provider(
    "openai",
    model="text-embedding-3-small",
    api_key="sk-...",
)

# Use
vector = await embedding.embed("Hello world")  # Returns list[float]
vectors = await embedding.embed_batch(["Hello", "World"])  # Batch

# Properties
embedding.model      # Model name
embedding.dimension  # Vector dimension (e.g., 1536)
```

**Providers:** `openai`, `sentence-transformers`, `cohere`, `ollama`, `huggingface`

---

### LLMService

High-level LLM operations.

```python
from engram import LLMService

# Create
llm = LLMService.from_provider(
    "openai",
    model="gpt-4o-mini",
    api_key="sk-...",
)

# Chat completion
response = await llm.complete_full(messages=[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
])
print(response.content)

# Simple text completion
text = await llm.complete("What is 2+2?")

# Fact extraction
facts = await llm.extract_facts(
    user_message="I'm Nafiz, I work in AI",
    assistant_response="Nice to meet you!",
)
# Returns: ["User's name is Nafiz", "User works in AI"]

# Summarization
summary = await llm.summarize(
    text="Long text here...",
    max_length=50,
    style="concise",
)

# Memory operation evaluation
operation = await llm.evaluate_memory_operation(
    new_fact="User now prefers tea",
    existing_memories=[("mem_123", "User likes coffee")],
)
# Returns: MemoryOperation(operation=UPDATE, content="User prefers tea over coffee")
```

**Providers:** `openai`, `anthropic`, `ollama`, `groq`, `litellm`

---

## Models

### Memory

```python
@dataclass
class Memory:
    memory_id: str        # Unique ID (mem_...)
    agent_id: str         # Agent ID
    user_id: str | None   # User ID
    session_id: str | None
    
    # Two-Column System
    content: str          # Alias for fact (backward compatible)
    fact: str | None      # Extracted user fact (EMBEDDED)
    main_content: str | None  # Full context [USER]:...\n[AI]:... (NOT embedded)
    
    embedding: list[float] | None  # Vector of fact
    importance: float     # 0.0-1.0, default 0.5
    access_count: int     # Times accessed
    created_at: datetime
    last_accessed_at: datetime
    metadata: dict
```

### SearchResult

```python
@dataclass
class SearchResult:
    memory: Memory        # The matched memory
    score: float          # Relevance score (0.0-1.0)
    
    # Access both columns via memory:
    # result.memory.fact         → What matched (embedded)
    # result.memory.main_content → Full context (not embedded)
    # result.memory.content      → Alias for fact
```

### TraversalResult

```python
@dataclass
class TraversalResult:
    memory_id: str
    content: str
    depth: int
    relation_type: str
    path: list[str]
```

### MemoryOperation

```python
@dataclass
class MemoryOperation:
    operation: MemoryOperationType  # ADD, UPDATE, DELETE, NOOP
    content: str
    target_id: str | None  # For UPDATE/DELETE
    reason: str
```

---

## Configuration

Environment variables:

```bash
# Database
ENGRAM_DATABASE_URL=postgresql://user:pass@localhost:5432/engram

# Embedding
ENGRAM_EMBEDDING_PROVIDER=openai
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
ENGRAM_EMBEDDING_DIMENSION=1536

# LLM
ENGRAM_LLM_PROVIDER=openai
ENGRAM_LLM_MODEL=gpt-4o-mini

# API Keys
ENGRAM_OPENAI_API_KEY=sk-...
ENGRAM_ANTHROPIC_API_KEY=sk-ant-...

# Search weights (must sum to 1.0)
ENGRAM_SEMANTIC_WEIGHT=0.5
ENGRAM_KEYWORD_WEIGHT=0.3
ENGRAM_RECENCY_WEIGHT=0.1
ENGRAM_IMPORTANCE_WEIGHT=0.1
```

See [Configuration Guide](configuration.md) for full details.

---

## Exceptions

```python
from engram.core.exceptions import (
    EngramError,           # Base exception
    DatabaseConnectionError,
    StorageError,
    ValidationError,
    EmbeddingError,
    MemoryNotFoundError,
    SessionError,
    GraphError,
    ConfigurationError,
)
```

**Example:**
```python
from engram.core.exceptions import MemoryNotFoundError

try:
    memory = await engram.get("invalid_id")
except MemoryNotFoundError:
    print("Memory not found")
```

