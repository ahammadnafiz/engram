# Engram Chatbot - Concept Document

> A personal AI assistant demonstrating the full Engram API.

## Overview

The Engram Chatbot demonstrates how to build an AI assistant with **persistent memory**. Unlike typical chatbots that forget everything when closed, this chatbot:

- Stores **user facts** extracted by LLM (name, preferences, goals)
- Uses **hybrid search** (semantic + keyword) to retrieve relevant context
- **Reinforces** frequently used memories (higher importance = higher ranking)
- Builds a **memory graph** by linking related memories
- Uses **sliding window** for efficient short-term context

## Engram API Coverage

The chatbot demonstrates these Engram methods:

| Method | Purpose | Used In |
|--------|---------|---------|
| `engram.add()` | Store new memory | `_process_fact()` |
| `engram.search()` | Hybrid search | `recall()`, `search_memories()` |
| `engram.update()` | Modify memory | `_process_fact()` |
| `engram.reinforce()` | Boost importance | `recall()` |
| `engram.forget()` | Delete memory | `_process_fact()` |
| `engram.purge()` | Clear all | `clear_memories()` |
| `engram.list_recent()` | Browse memories | `show_memories()` |
| `engram.relate()` | Create relations | `_link_to_recent()` |
| `engram.traverse()` | Graph traversal | `show_graph()` |
| `engram.health_check()` | Status check | `connect()` |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MEMORY CHATBOT                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ Sliding Window  │  │ recall()        │  │ chat()                  │  │
│  │ (Short-term)    │  │ (Long-term)     │  │ (Response Generation)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│     ENGRAM CLIENT     │ │  LLMService     │ │  EmbeddingService       │
│                       │ │                 │ │                         │
│ • add, search, update │ │ • complete_full │ │ • embed (vectors)       │
│ • reinforce, forget   │ │ • extract_facts │ │ • batch embedding       │
│ • relate, traverse    │ │ • evaluate_op   │ │                         │
└───────────────────────┘ └─────────────────┘ └─────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    POSTGRESQL + PGVECTOR                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  agent_memory: content, embedding, importance, metadata, ...    │    │
│  │  memory_relations: source_id, target_id, relation_type, ...     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Conversation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. USER INPUT: "I'm Nafiz, I work in AI"                               │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. MEMORY RETRIEVAL: recall()                                          │
│     await engram.search(query="I'm Nafiz, I work in AI", ...)           │
│                                                                         │
│     Returns relevant memories (if any):                                 │
│     - "User studies data science" (score: 0.45)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. CONTEXT BUILDING: chat()                                            │
│                                                                         │
│     messages = [                                                        │
│       {role: "system", content: SYSTEM_PROMPT},                         │
│       {role: "system", content: "About the user:\n- User studies..."},  │
│       ...sliding window of recent conversation...,                      │
│       {role: "user", content: "I'm Nafiz, I work in AI"}                │
│     ]                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. LLM COMPLETION                                                      │
│     response = await llm.complete_full(messages)                        │
│                                                                         │
│     Bot: "Nice to meet you, Nafiz! AI is such a fascinating field..."   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. BACKGROUND LEARNING: learn()                                        │
│                                                                         │
│     facts = await llm.extract_facts(user_msg, bot_msg)                  │
│     → ["User's name is Nafiz", "User works in AI"]                      │
│                                                                         │
│     For each fact:                                                      │
│       await _process_fact(fact)  # ADD, UPDATE, DELETE, or skip         │
│       await _link_to_recent()    # Create graph relations               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Memory Operations

When storing facts, the chatbot uses intelligent deduplication:

| Operation | When Used | Example |
|-----------|-----------|---------|
| **ADD** | New fact | "User has a cat named Luna" |
| **UPDATE** | Augments existing | "User's cat Luna is 2 years old" |
| **DELETE** | Contradicts existing | "User moved to NYC" replaces "User lives in Dhaka" |
| **NOOP** | Duplicate | "User's name is Nafiz" already exists |

### Deduplication Flow

```python
# In _process_fact()

# 1. Search for similar memories
similar = await engram.search(query=fact, ...)

# 2. Skip exact duplicates (score > 0.85)
for r in similar:
    if r.score > 0.85:
        return  # Already have this fact

# 3. Check for related memories (score > 0.4)
existing = [(id, content) for r in similar if r.score > 0.4]

# 4. If no related memories, just add
if not existing:
    await engram.add(content=fact, ...)
    return

# 5. Otherwise, let LLM decide the operation
op = await llm.evaluate_memory_operation(fact, existing)
# → ADD, UPDATE, DELETE, or NOOP
```

## Memory Reinforcement

When memories are retrieved and used, they get **reinforced**:

```python
# In recall()
for r in relevant:
    # Higher relevance = bigger boost (0.02 to 0.1)
    boost = 0.02 + (r.score * 0.08)
    await engram.reinforce(r.memory.memory_id, boost)
```

This implements **memory decay + importance boosting**:
- Unused memories gradually lose importance (decay)
- Frequently accessed memories gain importance (reinforcement)
- Search results rank higher-importance memories higher

## Memory Graph

New memories are linked to recent ones for graph connectivity:

```python
# In _link_to_recent()
recent = await engram.list_recent(agent_id, user_id, limit=3)
for mem in recent:
    if mem.memory_id != new_memory_id:
        await engram.relate(
            source_id=new_memory_id,
            target_id=mem.memory_id,
            relation_type="related_to",
        )
        break  # Link to most recent only
```

Use `/graph` command to visualize:

```
📊 Memory Graph (from: User's name is Nafiz...)
  └─ [1] User works in AI...
  └─ [2] User studied data science...
```

## Sliding Window (Short-term Memory)

Recent conversation is kept in-memory for LLM context:

```python
# Configuration
MAX_HISTORY = 20        # Total messages stored
CONTEXT_WINDOW = 10     # Messages sent to LLM
MAX_CHARS = 4000        # Character limit

# In _get_context()
def _get_context(self) -> list[dict]:
    window, chars = [], 0
    for msg in reversed(self.history):
        if len(window) >= CONTEXT_WINDOW or chars > MAX_CHARS:
            break
        window.append(msg)
        chars += len(msg.get("content", ""))
    return list(reversed(window))
```

**Key distinction**:
- `self.history` = Temporary, in-memory (lost on restart)
- Engram storage = Persistent, searchable (permanent)

## Configuration

```python
# Providers
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"

# Identifiers
AGENT_ID = "assistant"
USER_ID = "user"

# Context limits
MAX_HISTORY = 20
CONTEXT_WINDOW = 10
MAX_CHARS = 4000
```

## Commands

| Command | Description | Engram API |
|---------|-------------|------------|
| `/memories` | Show recent memories | `list_recent()` |
| `/search <q>` | Hybrid search | `search()` |
| `/graph` | Show memory relations | `traverse()` |
| `/forget` | Clear all memories | `purge()` |
| `/help` | Show commands | - |
| `/quit` | Exit | - |

## Example Session

```
🧠 Engram Memory Chatbot

Connecting...
  Database: ✓
  Embedding: text-embedding-3-small (1536d)
  LLM: gpt-4o-mini

Type /help for commands.

You: Hi, I'm Nafiz and I work in AI
Bot: Hey Nafiz! Nice to meet you. AI is such a fascinating field - 
     what kind of work do you do in AI?

You: /memories
📝 Memories (2)
  [50%] User's name is Nafiz...
  [50%] User works in AI...

You: /search AI
🔍 Hybrid Search: 'AI'
  [87%] User works in AI...

You: /graph
📊 Memory Graph (from: User works in AI...)
  └─ [1] User's name is Nafiz...
```

## Database Commands

### View All Memories

```bash
docker exec engram-postgres psql -U engram -d engram -c "
SELECT 
    LEFT(content, 60) as content,
    ROUND(importance::numeric, 2) as importance,
    to_char(created_at, 'MM-DD HH24:MI') as created
FROM agent_memory 
ORDER BY created_at DESC 
LIMIT 20;
"
```

### Memory Statistics

```bash
docker exec engram-postgres psql -U engram -d engram -c "
SELECT 
    COUNT(*) as total_memories,
    ROUND(AVG(importance)::numeric, 3) as avg_importance,
    MAX(access_count) as max_access
FROM agent_memory;
"
```

### View Relations

```bash
docker exec engram-postgres psql -U engram -d engram -c "
SELECT 
    LEFT(s.content, 30) as source,
    r.relation_type,
    LEFT(t.content, 30) as target
FROM memory_relations r
JOIN agent_memory s ON r.source_memory_id = s.memory_id
JOIN agent_memory t ON r.target_memory_id = t.memory_id
LIMIT 10;
"
```

### Clear All Memories

```bash
docker exec engram-postgres psql -U engram -d engram -c "
DELETE FROM agent_memory WHERE agent_id = 'assistant';
"
```

## Code Structure

```
chatbot.py (~470 lines)
│
├── Configuration (lines 1-64)
│   ├── Imports and env setup
│   ├── Provider config
│   └── Constants (AGENT_ID, sliding window limits)
│
├── MemoryChatbot class
│   │
│   ├── Connection & Lifecycle
│   │   ├── connect() - Initialize Engram + services
│   │   └── close() - Wait for tasks, cleanup
│   │
│   ├── Sliding Window
│   │   ├── _get_context() - Get recent messages
│   │   └── _trim_history() - Bound history size
│   │
│   ├── Search & Reinforce
│   │   └── recall() - engram.search() + reinforce()
│   │
│   ├── Fact Learning
│   │   ├── learn() - Extract facts from conversation
│   │   ├── _process_fact() - ADD/UPDATE/DELETE logic
│   │   └── _link_to_recent() - engram.relate()
│   │
│   ├── Chat
│   │   └── chat() - Main response generation
│   │
│   └── Commands
│       ├── show_memories() - engram.list_recent()
│       ├── search_memories() - engram.search()
│       ├── show_graph() - engram.traverse()
│       └── clear_memories() - engram.purge()
│
└── Main Loop
    ├── print_help()
    └── main() - REPL with command handling
```

---

*Built with [Engram](https://github.com/ahammadnafiz/engram) - AI Memory Layer for LLM Applications*
