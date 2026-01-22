# Engram Chatbot - Concept Document

> A personal AI assistant with persistent, searchable memory using Engram.

## Overview

The Engram Chatbot demonstrates how to build an AI assistant that **truly remembers** conversations across sessions. Unlike typical chatbots that forget everything when you close the window, this chatbot:

- Remembers **user facts** (name, preferences, goals, relationships)
- Remembers **conversation topics** (what was discussed, explained, asked)
- Uses **hybrid search** to retrieve relevant context for each response
- **Reinforces** memories that are frequently accessed (more important = higher ranking)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CHATBOT LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  Context Window │  │  Memory Search  │  │  LLM Completion         │  │
│  │  (Short-term)   │  │  (Long-term)    │  │  (Response Generation)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│     ENGRAM CORE       │ │  LLM SERVICE    │ │  EMBEDDING SERVICE      │
│  ┌─────────────────┐  │ │                 │ │                         │
│  │ Memory Store    │  │ │ - Completions   │ │ - Text → Vector         │
│  │ Graph Traversal │  │ │ - Summarize     │ │ - Batch Embedding       │
│  │ Session Manager │  │ │ - Extract Facts │ │ - Caching               │
│  └─────────────────┘  │ └─────────────────┘ └─────────────────────────┘
└───────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    POSTGRESQL + PGVECTOR                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  agent_memory: content, embedding, importance, metadata, ...    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Memory Types

The chatbot stores two types of memories:

### 1. User Facts (`type: "user_fact"`)

Personal information about the user, extracted by LLM:

| Category | Examples |
|----------|----------|
| Identity | "User's name is Nafiz", "User lives in Dhaka" |
| Preferences | "User's favorite dish is kacchi" |
| Professional | "User studied data science" |
| Relationships | "User's sister is named Nadia" |
| Goals | "User wants to learn machine learning" |

**Extraction**: Uses `LLMService.extract_facts()` with a comprehensive prompt that identifies atomic facts from conversation.

### 2. Conversation Topics (`type: "conversation_topic"`)

What was discussed/explained, extracted via summarization:

| Category | Examples |
|----------|----------|
| Technical | "Discussed: Transformer attention mechanism with self-attention and multi-head attention" |
| Explanations | "Discussed: Project management ticket systems for tracking tasks" |
| Q&A | "Discussed: Python implementation of decoder blocks in PyTorch" |

**Extraction**: Uses `LLMService.summarize()` to create concise summaries of substantive exchanges.

## Conversation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. USER INPUT: "explain me attention paper"                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. MEMORY RETRIEVAL                                                    │
│     engram.search(query="explain me attention paper", agent_id, ...)    │
│                                                                         │
│     Returns relevant memories:                                          │
│     - "User studied data science" (score: 0.45)                         │
│     - "Discussed: Neural network basics" (score: 0.52)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. CONTEXT BUILDING                                                    │
│                                                                         │
│     messages = [                                                        │
│       {role: "system", content: SYSTEM_PROMPT},                         │
│       {role: "system", content: "Relevant memories:\n- User studied..." │
│       ...sliding window of recent conversation...,                      │
│       {role: "user", content: "explain me attention paper"}             │
│     ]                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. LLM COMPLETION                                                      │
│     response = await llm.complete_full(messages)                        │
│                                                                         │
│     Bot: "The Attention paper introduced the Transformer..."            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. MEMORY EXTRACTION (Background, Parallel)                            │
│                                                                         │
│     ┌─────────────────────────┐  ┌────────────────────────────────────┐ │
│     │  _extract_user_facts()  │  │  _extract_conversation_topic()     │ │
│     │                         │  │                                    │ │
│     │  → (none in this case)  │  │  → "Discussed: Transformer         │ │
│     │                         │  │     attention mechanism..."        │ │
│     └─────────────────────────┘  └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  6. MEMORY STORAGE                                                      │
│     engram.add(                                                         │
│       content="Discussed: Transformer attention mechanism...",          │
│       metadata={type: "conversation_topic", user_query: "explain..."}   │
│     )                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Memory Operations (Mem0-Style)

When storing user facts, the chatbot uses intelligent deduplication:

### Operation Types

| Operation | When Used | Example |
|-----------|-----------|---------|
| **ADD** | Completely new fact | "User has a cat named Luna" (no prior pet info) |
| **UPDATE** | Augments existing | "User's cat Luna is 2 years old" + existing "User has cat Luna" |
| **DELETE** | Contradicts existing | "User switched to BRAC Bank" replaces "User banks at Standard Chartered" |
| **NOOP** | Duplicate | "User lives in Dhaka" already exists |

### Deduplication Flow

```
New Fact: "User's LinkedIn is linkedin.com/in/nafiz"

    │
    ▼ Search similar memories
    
Existing: "User's LinkedIn profile is linkedin.com/in/nafiz" (score: 0.92)

    │
    ▼ LLM evaluates operation
    
Decision: NOOP (semantically equivalent)
```

## Memory Reinforcement

When memories are retrieved and used, they get **reinforced**:

```python
# In get_context()
for r in relevant:
    # Higher relevance = bigger boost (0.02 to 0.1)
    boost = 0.02 + (r.score * 0.08)
    await engram.reinforce(r.memory.memory_id, boost)
```

This implements **memory decay + importance boosting**:
- Unused memories gradually lose importance (decay)
- Frequently accessed memories gain importance (reinforcement)
- Search results rank higher-importance memories higher

## Sliding Window (Short-term Memory)

Recent conversation history is kept in-memory for LLM context:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  self.history = [                                                       │
│    {role: "user", content: "my name is Nafiz"},                         │  Older
│    {role: "assistant", content: "Nice to meet you!"},                   │    │
│    {role: "user", content: "explain attention"},                        │    │
│    {role: "assistant", content: "The attention mechanism..."},          │    │
│    {role: "user", content: "give me the math"},                         │    ▼
│    {role: "assistant", content: "Here's the formula..."},               │  Newer
│  ]                                                                      │
│                                                                         │
│  Limits:                                                                │
│  - MAX_HISTORY_MESSAGES = 20 (stored in RAM)                            │
│  - CONTEXT_WINDOW_MESSAGES = 10 (sent to LLM)                           │
│  - MAX_CONTEXT_CHARS = 4000 (approximate token limit)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key distinction**:
- `self.history` = Temporary, in-memory (lost on restart)
- Engram storage = Persistent, searchable (permanent)

## Configuration

```python
# Providers
EMBEDDING_PROVIDER = "openai"      # or "sentence-transformers", "cohere"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_PROVIDER = "openai"            # or "anthropic", "ollama"
LLM_MODEL = "gpt-4o-mini"

# Identifiers
AGENT_ID = "assistant"             # Groups memories by agent
USER_ID = "user"                   # Groups memories by user

# Context limits
MAX_HISTORY_MESSAGES = 20          # In-memory conversation buffer
CONTEXT_WINDOW_MESSAGES = 10       # Messages sent to LLM
MAX_CONTEXT_CHARS = 4000           # Approximate token limit
```

## Commands

| Command | Description |
|---------|-------------|
| `/memories` | Show recent memories with types (👤 user_fact, 💬 topic) |
| `/search <query>` | Search memories by semantic similarity |
| `/consolidate` | Merge similar memories to reduce redundancy |
| `/forget` | Clear all memories (with confirmation) |
| `/config` | Show current configuration |
| `/help` | Show available commands |
| `/quit` | Exit the chatbot |

## Example Session

```
🧠 Engram Chatbot
Connecting...
  Embedding: text-embedding-3-small (1536d)
  LLM:       gpt-4o-mini
  Database:  ✓ connected

Type /help for commands.

You: Hi, I'm Nafiz and I live in Dhaka
Bot: Hey Nafiz! Nice to meet you. How's life in Dhaka treating you?

You: explain me the attention mechanism
Bot: The attention mechanism allows models to focus on relevant parts of 
     the input when generating output. In transformers, self-attention 
     computes relationships between all positions in a sequence...
     [detailed explanation]

You: /memories
📝 Recent Memories (2)
  👤 [user_fact] User's name is Nafiz...
  👤 [user_fact] User lives in Dhaka...
  💬 [conversation_topic] Discussed: Transformer attention mechanism...

You: /search attention
🔍 Search: 'attention'
  💬 [0.87] Discussed: Transformer attention mechanism with self-att...
  👤 [0.31] User's name is Nafiz...
```

## Design Decisions

### Why two memory types?

| User Facts | Conversation Topics |
|------------|---------------------|
| Stable over time | Ephemeral knowledge |
| High recall priority | Contextual retrieval |
| Deduplicated aggressively | Accumulated over time |
| "Who is the user?" | "What did we discuss?" |

### Why parallel extraction?

Running `_extract_user_facts()` and `_extract_conversation_topic()` concurrently:
- Doesn't block the user waiting for memory storage
- LLM calls are I/O bound, parallelism helps
- Either can fail without affecting the other

### Why background tasks?

Memory extraction is non-critical:
- User gets response immediately
- Facts are stored asynchronously
- Graceful shutdown waits for pending tasks

## Future Enhancements

1. **Memory graphs**: Link related memories (e.g., "Luna" → "User's cat")
2. **Session awareness**: Group memories by conversation session
3. **Importance decay visualization**: Show memory "freshness"
4. **Multi-user support**: Separate memories per user
5. **Memory export/import**: Backup and restore memories
6. **Conversation threading**: Track conversation branches

---

*Built with [Engram](https://github.com/your-repo/engram) - AI Memory Layer for LLM Applications*

