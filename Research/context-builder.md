# Context Builder Module Documentation

> **Simplified and Modular Context Window Management**

This document provides comprehensive documentation for the `context_builder.py` module, which implements a clean, maintainable approach to building context windows with dedicated source classes, weight-based scoring, and greedy knapsack packing.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Data Classes](#data-classes)
5. [Context Sources](#context-sources)
6. [ContextBuilder Class](#contextbuilder-class)
7. [Algorithms](#algorithms)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)
10. [API Reference](#api-reference)

---

## Overview

The Context Builder module provides optimal context window construction for LLM conversations. It uses a modular source-based architecture where each source handles one type of retrieval, and a unified builder assembles the final context using a greedy knapsack algorithm.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Context Building Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query ─────────────────────────────────────────────────►      │
│              │                                                  │
│              ▼                                                  │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                   Context Sources                       │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│   │  │ Recent   │ │ Semantic │ │ Current  │ │Important │   │   │
│   │  │ Window   │ │ Search   │ │ Topic    │ │ Messages │   │   │
│   │  │ (1.0)    │ │ (0.6)    │ │ (0.8)    │ │ (0.7)    │   │   │
│   │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │   │
│   │       │            │            │            │          │   │
│   └───────┼────────────┼────────────┼────────────┼──────────┘   │
│           │            │            │            │              │
│           └────────────┴────────────┴────────────┘              │
│                              │                                  │
│                              ▼                                  │
│                   ┌────────────────────┐                       │
│                   │ Early Deduplication │                       │
│                   │  (merge by index)   │                       │
│                   └──────────┬─────────┘                       │
│                              │                                  │
│                              ▼                                  │
│                   ┌────────────────────┐                       │
│                   │  Greedy Knapsack   │                       │
│                   │  (score + budget)  │                       │
│                   └──────────┬─────────┘                       │
│                              │                                  │
│                              ▼                                  │
│                   ┌────────────────────┐                       │
│                   │ Chronological Sort │                       │
│                   └──────────┬─────────┘                       │
│                              │                                  │
│                              ▼                                  │
│                        Final Context                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Single Responsibility
Each source class handles exactly ONE type of retrieval:
- `RecentWindowSource` - Last N messages
- `SemanticSource` - Similarity-based retrieval
- `CurrentTopicSource` - Topic segment continuity
- `ImportantMessageSource` - Tools, artifacts, pinned
- `EntitySource` - Entity-based matches

### 2. Weight-Based Scoring
Each source has a configurable weight that affects final scoring:

| Source | Default Weight | Base Score | Final Score |
|--------|---------------|------------|-------------|
| Recent | 1.0 | 1.0 | 1.0 |
| Topic | 0.8 | 0.9 | 0.72 |
| Important | 0.7 | 0.85 | 0.595 |
| Semantic | 0.6 | varies | varies × 0.6 |
| Entity | 0.5 | 0.7 | 0.35 |

### 3. Early Deduplication
Messages found by multiple sources are merged:
- Scores are summed
- Sources are unioned
- Avoids redundant processing

### 4. Greedy Knapsack Packing
Optimal budget usage:
- Sort candidates by score (descending)
- Add messages while budget allows
- Skip oversized messages, continue to next

### 5. No Emergency Fallbacks
Designed to always fit within budget by construction.

---

## Architecture

```
core/context_builder.py
├── Data Classes
│   ├── Message                    # Wrapper with metadata
│   └── Context                    # Final assembled context
├── Abstract Base
│   └── ContextSource              # Interface for sources
├── Source Implementations
│   ├── RecentWindowSource         # Recent messages
│   ├── SemanticSource             # Semantic similarity
│   ├── CurrentTopicSource         # Topic continuity
│   ├── ImportantMessageSource     # Tools/artifacts
│   └── EntitySource               # Entity matches
└── ContextBuilder Class
    ├── build()                    # Main entry point
    ├── _gather_candidates()       # Collect from sources
    ├── _knapsack_select()         # Greedy selection
    └── _get_checkpoints()         # Checkpoint summaries
```

---

## Data Classes

### Message

Wrapper for a message with scoring metadata:

```python
@dataclass
class Message:
    index: int                      # Position in conversation
    content: Dict[str, Any]         # Original message dict
    tokens: int                     # Token count
    score: float = 0.0              # Importance score
    sources: Set[str] = field(...)  # Which sources found it
```

**Features:**
- Hashable by index for deduplication
- Equality comparison by index

### Context

Final assembled context result:

```python
@dataclass
class Context:
    messages: List[Dict[str, Any]]  # Selected messages
    total_tokens: int               # Total token count
    sources_used: Dict[str, int]    # Count per source
    checkpoints: List[str] = []     # Checkpoint summaries
```

---

## Context Sources

All sources inherit from the abstract `ContextSource` class:

```python
class ContextSource(ABC):
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight
    
    @abstractmethod
    def get_candidates(self, query, conversation_history, exclude_indices) -> List[Message]:
        pass
```

### RecentWindowSource

**Purpose:** Always include recent messages for conversation continuity.

```python
RecentWindowSource(weight=1.0, window_size=5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight` | float | 1.0 | Scoring weight (highest) |
| `window_size` | int | 5 | Number of recent messages |

**Behavior:**
- Returns last `window_size` messages
- Assigns score of `1.0 × weight`
- Source name: `"recent_window"`

### SemanticSource

**Purpose:** Retrieve semantically similar messages to the query.

```python
SemanticSource(weight=0.6, semantic_retriever=retriever, top_k=10)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight` | float | 0.6 | Scoring weight |
| `semantic_retriever` | object | None | Retriever instance |
| `top_k` | int | 10 | Max results |

**Behavior:**
- Uses semantic retriever with temporal scoring
- Matches results to conversation by ID or content
- Score = `relevance_score × weight`
- Source name: `"semantic_matches"`

### CurrentTopicSource

**Purpose:** Keep current topic segment intact for coherence.

```python
CurrentTopicSource(weight=0.8, segmenter=segmenter)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight` | float | 0.8 | Scoring weight |
| `segmenter` | object | None | Topic segmenter |

**Behavior:**
- Gets current segment from segmenter
- Returns all messages in segment range
- Score = `0.9 × weight`
- Source name: `"current_segment"`

### ImportantMessageSource

**Purpose:** Include messages with tool calls, artifacts, or pinned status.

```python
ImportantMessageSource(weight=0.7, pinned_ids=set())
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight` | float | 0.7 | Scoring weight |
| `pinned_ids` | Set[int] | empty | Manually pinned message IDs |

**Importance Criteria:**
1. Message ID in `pinned_ids`
2. Contains tool calls (various formats)
3. Contains artifacts (via metadata)

**Detection Methods:**
```python
# Tool call detection checks:
- msg['tool_calls']           # LangChain format
- part.function_call          # Gemini-style parts
- part.function_response      # Tool responses
- metadata['has_tool_calls']  # Metadata flags
- metadata['is_tool_response']
- role == 'tool'              # ToolMessage role

# Artifact detection:
- metadata['has_artifact']
```

- Score = `0.85 × weight`
- Source name: `"important_messages"`

### EntitySource

**Purpose:** Include messages mentioning entities relevant to the query.

```python
EntitySource(weight=0.5, entity_graph=graph)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight` | float | 0.5 | Scoring weight |
| `entity_graph` | object | None | Entity graph instance |

**Behavior:**
- Queries entity graph for relevant message IDs
- Returns messages containing related entities
- Score = `0.7 × weight`
- Source name: `"entity_matches"`

---

## ContextBuilder Class

The main orchestrator that assembles context from multiple sources.

### Constructor

```python
def __init__(self,
             budget: int,
             token_counter,
             sources: Optional[List[ContextSource]] = None,
             checkpoint_manager=None,
             search_fusion=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `budget` | int | Maximum tokens for context |
| `token_counter` | callable | Function to count tokens |
| `sources` | List[ContextSource] | Ordered list of sources |
| `checkpoint_manager` | object | For checkpoint summaries |
| `search_fusion` | object | Reused for embeddings |

### Main Method: build()

```python
def build(self, query: str, conversation_history: List[Dict]) -> Context
```

**Pipeline:**

1. **Get Checkpoints** - Compact summaries from checkpoint manager
2. **Gather Candidates** - Collect from all sources with deduplication
3. **Set Token Counts** - Calculate tokens for each candidate
4. **Greedy Selection** - Select highest-scoring within budget
5. **Sort Chronologically** - Maintain conversation order
6. **Assemble Context** - Create final Context object

---

## Algorithms

### Candidate Gathering with Deduplication

```
For each source in sources:
    candidates = source.get_candidates(query, history, excluded)
    
    For each candidate:
        If index already in candidates_by_index:
            existing.score += candidate.score    # Sum scores
            existing.sources |= candidate.sources # Union sources
        Else:
            candidates_by_index[index] = candidate
```

**Result:** Messages found by multiple sources get higher combined scores.

### Greedy Knapsack Selection

```
┌─────────────────────────────────────────────────────────────┐
│                 Greedy Knapsack Algorithm                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: candidates (sorted by score DESC), budget          │
│                                                             │
│   selected = []                                             │
│   used_tokens = 0                                           │
│                                                             │
│   For each candidate (highest score first):                 │
│       If used_tokens + candidate.tokens ≤ budget:           │
│           selected.append(candidate)                        │
│           used_tokens += candidate.tokens                   │
│       Else:                                                 │
│           Skip (log debug message)                          │
│           Continue to next candidate                        │
│                                                             │
│   Return selected                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight:** By skipping rather than stopping, we can fit more smaller high-value messages even if one large message doesn't fit.

### Checkpoint Integration

```
1. Query checkpoint_manager with search_fusion embeddings
2. Get top 2 relevant checkpoints
3. Format as: "[Checkpoint #N: summary...]"
4. Limit to 15% of total budget
5. Add to context before messages
```

---

## Usage Examples

### Basic Setup

```python
from core.context_builder import (
    ContextBuilder,
    RecentWindowSource,
    SemanticSource,
    CurrentTopicSource,
    ImportantMessageSource,
    EntitySource
)

# Token counter function
def count_tokens(messages):
    if isinstance(messages, str):
        return len(messages.split())
    return sum(len(m.get('content', '').split()) for m in messages)

# Initialize sources
sources = [
    RecentWindowSource(weight=1.0, window_size=5),
    SemanticSource(weight=0.6, semantic_retriever=retriever, top_k=10),
    CurrentTopicSource(weight=0.8, segmenter=segmenter),
    ImportantMessageSource(weight=0.7, pinned_ids={42, 100}),
    EntitySource(weight=0.5, entity_graph=graph)
]

# Create builder
builder = ContextBuilder(
    budget=4000,
    token_counter=count_tokens,
    sources=sources,
    checkpoint_manager=checkpoint_mgr,
    search_fusion=fusion
)
```

### Building Context

```python
# Full conversation history
conversation = [
    {"role": "user", "content": "Hello, I'm working on a Python project"},
    {"role": "assistant", "content": "Great! What kind of project?"},
    # ... many more messages
]

# Current query
query = "How do I connect to the database?"

# Build context
context = builder.build(query, conversation)

# Use results
print(f"Selected {len(context.messages)} messages")
print(f"Using {context.total_tokens} tokens")
print(f"Sources: {context.sources_used}")

# context.messages is ready for LLM
for checkpoint in context.checkpoints:
    print(f"Summary: {checkpoint}")
```

### Custom Source

```python
class QuestionSource(ContextSource):
    """Include unanswered questions"""
    
    def __init__(self, weight: float = 0.6):
        super().__init__("unanswered_questions", weight)
    
    def get_candidates(self, query, conversation_history, exclude_indices):
        candidates = []
        
        for i, msg in enumerate(conversation_history):
            if i in exclude_indices:
                continue
            
            if msg.get('role') == 'user' and '?' in msg.get('content', ''):
                # Check if next message answers it
                if i + 1 < len(conversation_history):
                    next_msg = conversation_history[i + 1]
                    if next_msg.get('role') != 'assistant':
                        # Unanswered question
                        candidates.append(Message(
                            index=i,
                            content=msg,
                            tokens=0,
                            score=0.8 * self.weight,
                            sources={self.name}
                        ))
        
        return candidates

# Add to sources
sources.append(QuestionSource(weight=0.6))
```

### Adjusting Weights

```python
# Prioritize semantic search over recency
sources = [
    RecentWindowSource(weight=0.5, window_size=3),      # Reduced
    SemanticSource(weight=1.0, semantic_retriever=r),   # Increased
    ImportantMessageSource(weight=0.8),                 # Increased
]

# For task-focused conversations
sources = [
    RecentWindowSource(weight=1.0, window_size=10),     # More recent
    ImportantMessageSource(weight=1.0),                 # Tools important
    SemanticSource(weight=0.3),                         # Less semantic
]
```

---

## Configuration

### Budget Allocation

```python
# Recommended budget distribution
total_budget = 4000

# Checkpoints: 15% max
checkpoint_budget = int(total_budget * 0.15)  # 600 tokens

# Messages: remaining 85%
message_budget = total_budget - checkpoint_budget  # 3400 tokens
```

### Source Weights

| Use Case | Recent | Semantic | Topic | Important | Entity |
|----------|--------|----------|-------|-----------|--------|
| **Default** | 1.0 | 0.6 | 0.8 | 0.7 | 0.5 |
| **Research** | 0.5 | 1.0 | 0.6 | 0.8 | 0.7 |
| **Coding** | 0.8 | 0.5 | 0.6 | 1.0 | 0.4 |
| **Chat** | 1.0 | 0.4 | 0.9 | 0.5 | 0.3 |

### Window Sizes

```python
# Short context (quick responses)
RecentWindowSource(window_size=3)

# Standard (balanced)
RecentWindowSource(window_size=5)

# Long context (detailed work)
RecentWindowSource(window_size=10)
```

---

## API Reference

### ContextSource Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_candidates()` | `query, history, exclude` | `List[Message]` | Get candidate messages |

### ContextBuilder Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `build()` | `query, conversation_history` | `Context` | Build optimal context |

### Internal Methods

| Method | Description |
|--------|-------------|
| `_gather_candidates()` | Collect from all sources with dedup |
| `_knapsack_select()` | Greedy selection within budget |
| `_get_checkpoints()` | Get checkpoint summaries |
| `_count_message_tokens()` | Count tokens in messages |

### Context Object

```python
@dataclass
class Context:
    messages: List[Dict[str, Any]]  # Selected messages (chronological)
    total_tokens: int               # Total tokens used
    sources_used: Dict[str, int]    # {"recent_window": 5, "semantic_matches": 3}
    checkpoints: List[str]          # Checkpoint summary strings
```

---

## Source Name Mapping

For UI/logging consistency:

| Source Class | `.name` Property |
|--------------|------------------|
| `RecentWindowSource` | `"recent_window"` |
| `SemanticSource` | `"semantic_matches"` |
| `CurrentTopicSource` | `"current_segment"` |
| `ImportantMessageSource` | `"important_messages"` |
| `EntitySource` | `"entity_matches"` |

---

## Error Handling

Each source handles its own errors gracefully:

```python
# Semantic retrieval failure
try:
    results = self.semantic_retriever.retrieve(...)
except Exception as e:
    logger.warning(f"Semantic retrieval failed: {e}")
    return []  # Empty list, not exception

# Similar pattern for all sources
```

The builder continues with available sources even if some fail.

---

## Performance Notes

- **Early Deduplication**: Prevents redundant token counting
- **Single Pass Selection**: O(n log n) for sorting + O(n) for selection
- **Reused SearchFusion**: Avoids recreating embedding model
- **Lazy Token Counting**: Only counts tokens for actual candidates
- **Budget-First Design**: Never exceeds token budget

---

## Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    Integration Map                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐     ┌───────────────────┐               │
│   │ SemanticRetrieval │ │ TopicSegmenter    │               │
│   └───────┬──────┘     └────────┬──────────┘               │
│           │                     │                           │
│           ▼                     ▼                           │
│   ┌───────────────┐    ┌───────────────────┐               │
│   │ SemanticSource│    │ CurrentTopicSource│               │
│   └───────┬───────┘    └────────┬──────────┘               │
│           │                     │                           │
│           └──────────┬──────────┘                           │
│                      ▼                                      │
│              ┌───────────────┐                             │
│              │ContextBuilder │                             │
│              └───────┬───────┘                             │
│                      │                                      │
│           ┌──────────┼──────────┐                          │
│           ▼          ▼          ▼                          │
│   ┌────────────┐ ┌────────┐ ┌──────────────┐              │
│   │EntityGraph │ │Checkpoints│ │SearchFusion│              │
│   └────────────┘ └────────┘ └──────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Logging

The module provides detailed logging at INFO and DEBUG levels:

```
INFO: 🏗️ Building context (budget: 4000 tokens)
DEBUG:   Checkpoints: 2 summaries, 450 tokens
DEBUG:   Total candidates: 45 messages
DEBUG:   Skipping message 12 (800 tokens, over budget)
DEBUG:   Selected: 18 messages
INFO: ✓ Context built: 18 messages, 3850/4000 tokens
INFO:   Sources: {'recent_window': 5, 'semantic_matches': 8, 'current_segment': 5}
```

---

*Documentation generated for `core/context_builder.py`*
