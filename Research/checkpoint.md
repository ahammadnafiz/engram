# Checkpoint System Documentation

> **Next-Generation Memory Management for Long-Running Conversations**

This document provides comprehensive documentation for the `checkpoints.py` module, which implements an advanced checkpoint system for managing conversational memory with diff-based storage, multi-resolution compression, and active forgetting.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Data Classes](#data-classes)
5. [Checkpoint Class](#checkpoint-class)
6. [CheckpointManager Class](#checkpointmanager-class)
7. [Core Algorithms](#core-algorithms)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)
10. [API Reference](#api-reference)

---

## Overview

The Checkpoint System is designed to efficiently manage long-running conversation memory by:

- **Storing only what's new** - Diff-based memory avoids redundant storage
- **Compressing old memories** - Multi-resolution tiers reduce storage over time
- **Forgetting obsolete info** - LLM-driven cleanup keeps memory relevant
- **Maintaining backward compatibility** - API remains unchanged from legacy implementations

```
┌─────────────────────────────────────────────────────────────────┐
│                     Checkpoint System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Messages ─────► Diff Extraction ─────► Knowledge State        │
│                        │                      │                 │
│                        ▼                      ▼                 │
│               ┌─────────────────┐    ┌──────────────────┐      │
│               │   Checkpoint    │    │  Accumulated     │      │
│               │   (Novel Info)  │    │  Facts/Entities  │      │
│               └────────┬────────┘    └──────────────────┘      │
│                        │                                        │
│         ┌──────────────┼──────────────┐                        │
│         ▼              ▼              ▼                        │
│      HOT           WARM          COOL/FROZEN                   │
│   (Full Detail)  (Summarized)  (Key Concepts)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Diff-Based Memory
Instead of storing complete conversation segments, the system extracts and stores only:
- **Novel facts** - New information not in current state
- **Updated facts** - Corrections to existing knowledge
- **Obsolete facts** - Information that should be removed
- **New entities** - People, places, technologies mentioned
- **New topics** - Subject domains discussed

### 2. Multi-Resolution Compression
Checkpoints are organized into tiers based on age:

| Tier | Age (Checkpoints) | Message Storage | Summary Length |
|------|-------------------|-----------------|----------------|
| **HOT** | 0-5 | Full | Full |
| **WARM** | 6-15 | None | Full |
| **COOL** | 16-30 | None | Truncated (100 chars) |
| **FROZEN** | 30+ | None | Ultra-truncated (50 chars) |

### 3. Active Forgetting
Periodic LLM-driven cleanup that:
- Reviews stored facts for relevance
- Identifies outdated or superseded information
- Merges similar facts to reduce redundancy
- Removes noise and temporary context

### 4. Semantic Deduplication
Uses embedding similarity (threshold: 0.85) to avoid storing semantically equivalent facts.

---

## Architecture

```
memory/checkpoints.py
├── Enums
│   └── CompressionTier          # HOT, WARM, COOL, FROZEN
├── Data Classes
│   ├── MemoryDiff               # Delta from previous state
│   └── KnowledgeState           # Accumulated knowledge
├── Checkpoint Class
│   ├── Core fields (backward compatible)
│   └── Diff-based fields (new)
└── CheckpointManager Class
    ├── Diff-Based Memory Methods
    ├── Multi-Resolution Compression
    ├── Active Forgetting
    ├── Legacy Methods
    └── Public API
```

---

## Data Classes

### CompressionTier

```python
class CompressionTier(Enum):
    HOT = "hot"          # Full fidelity (recent)
    WARM = "warm"        # Summarized
    COOL = "cool"        # Key concepts only
    FROZEN = "frozen"    # Ultra-compressed
```

### MemoryDiff

Represents the delta/change from the previous state:

```python
@dataclass
class MemoryDiff:
    novel_facts: List[str]              # New facts learned
    updated_facts: List[Dict[str, str]] # {"old": ..., "new": ...}
    obsolete_facts: List[str]           # Facts to remove
    new_entities: List[str]             # New entities mentioned
    new_topics: List[str]               # New topics discussed
    is_novel: bool                      # Whether diff contains new info
```

**Methods:**
- `to_dict()` - Serialize to dictionary

### KnowledgeState

Maintains the accumulated knowledge across all checkpoints:

```python
@dataclass
class KnowledgeState:
    facts: Dict[str, str]                    # fact_hash -> fact_text
    entities: Set[str]                        # Known entities
    topics: Set[str]                          # Known topics
    fact_embeddings: Dict[str, np.ndarray]   # For semantic dedup
```

**Methods:**
- `get_summary()` - Returns concise summary of current state

---

## Checkpoint Class

Represents a single checkpoint in conversation history.

### Constructor

```python
def __init__(self, 
             checkpoint_id: int,
             message_range: tuple,        # (start_index, end_index)
             messages: List[Dict[str, Any]])
```

### Core Fields (Backward Compatible)

| Field | Type | Description |
|-------|------|-------------|
| `checkpoint_id` | `int` | Unique identifier |
| `message_range` | `tuple` | (start_index, end_index) |
| `messages` | `List[Dict]` | Raw messages in checkpoint |
| `created_at` | `float` | Unix timestamp |
| `summary` | `str` | Text summary of checkpoint |
| `key_entities` | `List[str]` | Extracted entities |
| `decisions` | `List[str]` | Facts/decisions stored |
| `open_questions` | `List[str]` | Unresolved questions |
| `artifacts` | `List[str]` | Created files/documents |
| `topics` | `List[str]` | Discussion topics |
| `embedding` | `np.ndarray` | Semantic embedding |

### New Fields (Diff-Based)

| Field | Type | Description |
|-------|------|-------------|
| `diff` | `MemoryDiff` | Delta from previous state |
| `compression_tier` | `CompressionTier` | Current compression level |
| `usage_count` | `int` | Access count (meta-learning) |
| `last_accessed` | `float` | Last access timestamp |

### Methods

```python
def to_dict(self) -> Dict[str, Any]
    """Convert checkpoint to dictionary (backward compatible)"""

@classmethod
def from_dict(cls, data: Dict[str, Any], messages: List[Dict] = None) -> 'Checkpoint'
    """Create checkpoint from dictionary"""

def mark_accessed(self)
    """Track usage for meta-learning"""
```

---

## CheckpointManager Class

The main manager class for creating, storing, and retrieving checkpoints.

### Constructor

```python
def __init__(self,
             llm,                              # LangChain chat model
             model: str,                       # Model name
             search_fusion,                    # SearchFusion for embeddings
             persist_dir: Optional[str] = None,
             checkpoint_interval: int = 20,   # Create every N messages
             max_checkpoints: int = 50)       # Max in memory
```

### Class Constants

```python
TIER_CONFIG = {
    CompressionTier.HOT:    {'max_age_checkpoints': 5,  'store_messages': True},
    CompressionTier.WARM:   {'max_age_checkpoints': 15, 'store_messages': False},
    CompressionTier.COOL:   {'max_age_checkpoints': 30, 'store_messages': False},
    CompressionTier.FROZEN: {'max_age_checkpoints': None, 'store_messages': False},
}

DEDUP_THRESHOLD = 0.85  # Similarity threshold for deduplication
```

### Internal State

| Attribute | Type | Description |
|-----------|------|-------------|
| `checkpoints` | `List[Checkpoint]` | Stored checkpoints |
| `checkpoint_counter` | `int` | ID counter |
| `messages_since_last_checkpoint` | `int` | Message counter |
| `checkpoint_embeddings` | `List[np.ndarray]` | For semantic search |
| `knowledge_state` | `KnowledgeState` | Accumulated knowledge |

---

## Core Algorithms

### Diff Extraction Algorithm

```
1. Get current knowledge state summary
2. Format messages for analysis (sample if > 30)
3. Prompt LLM to extract:
   - NOVEL_FACTS: New information
   - UPDATED_FACTS: Corrections (old -> new)
   - OBSOLETE_FACTS: Outdated info
   - NEW_ENTITIES: People, places, tech
   - NEW_TOPICS: Discussion subjects
4. Parse response and create MemoryDiff
5. If LLM fails, use heuristic extraction
```

**Heuristic Fallback:**
- Regex patterns for personal statements ("I am...", "My name is...")
- Extract capitalized words as entities
- Filter against noise words

### Multi-Resolution Compression Algorithm

```
For each checkpoint (oldest to newest):
    1. Calculate age = total_checkpoints - position
    2. Determine tier based on age:
       - age ≤ 5:  HOT
       - age ≤ 15: WARM
       - age ≤ 30: COOL
       - age > 30: FROZEN
    3. If tier changed, apply compression:
       - WARM: Remove messages
       - COOL: Truncate summary to 100 chars
       - FROZEN: Truncate to 50 chars, limit all lists
```

### Active Forgetting Algorithm

```
Every 5 checkpoints:
1. Collect up to 50 facts with usage counts
2. Prompt LLM to categorize:
   - KEEP: High-value, frequently used
   - COMPRESS: Merge similar facts
   - FORGET: Outdated, superseded, noise
3. Apply recommendations:
   - Remove forgotten facts from knowledge state
   - Update fact embeddings accordingly
```

### Super-Checkpoint Creation

When `max_checkpoints` is exceeded:
```
1. Select oldest 25% of checkpoints
2. Create new super-checkpoint:
   - Combine message ranges
   - Merge summaries (truncate to 200 chars)
   - Union of topics, entities (limited)
   - Mark as FROZEN tier
3. Replace old checkpoints with super-checkpoint
```

---

## Usage Examples

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from memory.checkpoints import CheckpointManager
from memory.search_fusion import SearchFusion

# Initialize
llm = ChatOpenAI(model="gpt-4")
search_fusion = SearchFusion(...)

manager = CheckpointManager(
    llm=llm,
    model="gpt-4",
    search_fusion=search_fusion,
    persist_dir="./checkpoints",
    checkpoint_interval=20,
    max_checkpoints=50
)

# Track messages
manager.increment_message_counter()

# Check if checkpoint needed
if manager.should_create_checkpoint():
    checkpoint = manager.create_checkpoint(
        messages=recent_messages,
        start_index=100
    )
```

### Searching Checkpoints

```python
# Semantic search
query_embedding = search_fusion.embed_text("user's Python preferences")
relevant = manager.search_checkpoints(query_embedding, top_k=3)

for cp in relevant:
    print(f"Checkpoint #{cp.checkpoint_id}: {cp.summary}")
    print(f"  Topics: {cp.topics}")
    print(f"  Facts: {cp.decisions}")
```

### Accessing Knowledge State

```python
# Get accumulated knowledge
summary = manager.get_knowledge_summary()
print(summary)
# Output: "Facts: User prefers Python, Works at TechCorp | Entities: Python, TechCorp"

# Get all stored facts
facts = manager.get_all_facts()
for fact in facts:
    print(f"  - {fact}")
```

### Manual Forgetting

```python
# Trigger cleanup when needed
manager.force_forgetting()
```

### Getting Statistics

```python
stats = manager.get_stats()
print(f"Checkpoints: {stats['checkpoints']}")
print(f"Facts: {stats['facts']}")
print(f"Tier distribution: {stats['tiers']}")
```

---

## Configuration

### Checkpoint Interval

Controls how often checkpoints are created:

```python
checkpoint_interval=20  # Create checkpoint every 20 messages
```

**Recommendations:**
- **10-15**: For detailed, fact-heavy conversations
- **20-30**: For general conversations (default)
- **40+**: For light, casual interactions

### Max Checkpoints

Controls memory usage:

```python
max_checkpoints=50  # Keep max 50 checkpoints in memory
```

When exceeded, old checkpoints are merged into super-checkpoints.

### Deduplication Threshold

```python
DEDUP_THRESHOLD = 0.85  # Cosine similarity threshold
```

**Adjustments:**
- **0.9+**: More strict, allows more similar facts
- **0.8-0.85**: Balanced (default)
- **0.7-**: Aggressive deduplication

### Forgetting Interval

```python
self._forgetting_interval = 5  # Run cleanup every 5 checkpoints
```

---

## API Reference

### Public Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `should_create_checkpoint()` | - | `bool` | Check if checkpoint needed |
| `create_checkpoint()` | `messages, start_index` | `Optional[Checkpoint]` | Create new checkpoint |
| `increment_message_counter()` | - | - | Increment message count |
| `search_checkpoints()` | `query_embedding, top_k=3` | `List[Checkpoint]` | Semantic search |
| `get_checkpoint_by_id()` | `checkpoint_id` | `Optional[Checkpoint]` | Get by ID |
| `get_recent_checkpoints()` | `count=3` | `List[Checkpoint]` | Get recent checkpoints |
| `get_knowledge_summary()` | - | `str` | Get knowledge summary |
| `get_all_facts()` | - | `List[str]` | Get all stored facts |
| `force_forgetting()` | - | - | Trigger active forgetting |
| `get_stats()` | - | `Dict[str, Any]` | Get statistics |
| `to_dict()` | - | `Dict[str, Any]` | Export manager state |

### Persistence

Files created in `persist_dir`:

```
persist_dir/
├── checkpoint_0001.json
├── checkpoint_0002.json
├── ...
└── knowledge_state.json
```

**checkpoint_XXXX.json:**
```json
{
  "checkpoint_id": 1,
  "message_range": [0, 19],
  "summary": "Discussed Python preferences...",
  "key_entities": ["Python", "Flask"],
  "decisions": ["User prefers Python over Java"],
  "topics": ["programming", "web development"],
  "compression_tier": "hot",
  "usage_count": 5,
  "diff": {
    "novel_facts": ["User prefers Python"],
    "is_novel": true
  }
}
```

**knowledge_state.json:**
```json
{
  "facts": {
    "a1b2c3d4": "User prefers Python over Java",
    "e5f6g7h8": "User works at TechCorp"
  },
  "entities": ["Python", "Java", "TechCorp"],
  "topics": ["programming", "career"]
}
```

---

## Error Handling

The system includes multiple fallback mechanisms:

1. **LLM Failure** → Heuristic extraction
2. **Embedding Failure** → Skip deduplication check
3. **Parsing Failure** → Apply diff directly to checkpoint
4. **Persistence Failure** → Continue in-memory operation

All errors are logged with appropriate severity levels.

---

## Performance Considerations

- **Message Sampling**: For large message sets (>30), samples are taken from beginning, middle, and end
- **Fact Limits**: Maximum 50 facts processed per forgetting cycle
- **Embedding Caching**: Fact embeddings stored in `KnowledgeState` for reuse
- **Lazy Compression**: Compression applied only when tier changes

---

## Dependencies

```python
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import logging
import json
import os
import hashlib
from langchain_core.messages import HumanMessage  # For LLM calls
from memory.entity_graph import extract_entities_simple  # For entity extraction
```

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0 | Initial checkpoint system |
| 2.0 (Current) | Diff-based memory, multi-resolution compression, active forgetting, semantic deduplication |

---

*Documentation generated for `memory/checkpoints.py`*
