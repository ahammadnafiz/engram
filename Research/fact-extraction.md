# Fact Extraction Module Documentation

> **Semantic LLM-Based Fact Extraction and Storage System**

This document provides comprehensive documentation for the `fact_extraction.py` module, which implements pure LLM-based fact extraction with semantic deduplication and FAISS-powered vector storage.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Extraction Prompt](#extraction-prompt)
5. [Data Classes](#data-classes)
6. [FactExtractor Class](#factextractor-class)
7. [FactStore Class](#factstore-class)
8. [Deduplication Algorithm](#deduplication-algorithm)
9. [Usage Examples](#usage-examples)
10. [Configuration](#configuration)
11. [API Reference](#api-reference)

---

## Overview

The Fact Extraction module provides a clean, LLM-first approach to extracting and storing memorable facts from conversations. Unlike rule-based systems, this module relies entirely on the LLM's understanding to determine what's worth remembering.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fact Extraction Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Message ──► Should Extract? ──► LLM Extraction           │
│                         │                    │                  │
│                    (role=user,              │                  │
│                     len > 10)               ▼                  │
│                                      ┌─────────────┐           │
│                                      │  Parse JSON │           │
│                                      │  Response   │           │
│                                      └──────┬──────┘           │
│                                             │                  │
│                                             ▼                  │
│                                    ┌────────────────┐          │
│                                    │  Validate &    │          │
│                                    │  Cache Facts   │          │
│                                    └────────┬───────┘          │
│                                             │                  │
│                                             ▼                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                      FactStore                          │  │
│   │  ┌──────────┐   ┌───────────────┐   ┌──────────────┐   │  │
│   │  │  FAISS   │   │  Similarity   │   │  Persistence │   │  │
│   │  │  Index   │◄──│   Check       │──►│  (pickle)    │   │  │
│   │  └──────────┘   └───────────────┘   └──────────────┘   │  │
│   │                        │                               │  │
│   │            ┌───────────┼───────────┐                   │  │
│   │            ▼           ▼           ▼                   │  │
│   │         ADD        UPDATE       NOOP                   │  │
│   │       (new)       (merge)    (duplicate)               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Pure LLM-Based Extraction
- **No hardcoded rules** - The LLM decides what's salient
- **Context-aware** - Uses conversation summary and recent messages
- **Structured output** - Returns JSON with fact, entities, and type

### 2. Semantic Deduplication
- **FAISS-powered** - Fast vector similarity search
- **Two-tier thresholds**:
  - `> 0.90`: Exact duplicate → Skip (NOOP)
  - `> 0.78`: Similar → Merge (UPDATE)
  - `< 0.78`: New → Add (ADD)

### 3. Correction Handling
- Automatically detects correction-type facts
- Supersedes related outdated facts
- Maintains version history

### 4. Persistent Storage
- FAISS index for vector search
- Pickle-based metadata storage
- Automatic loading on initialization

---

## Architecture

```
memory/fact_extraction.py
├── Constants
│   └── EXTRACTION_PROMPT          # LLM prompt template
├── Data Classes
│   ├── ExtractedFact              # Single extracted fact
│   └── StoredFact                 # Persisted fact with metadata
├── FactExtractor Class
│   ├── should_extract_facts()     # Pre-filter check
│   ├── extract_facts()            # Main extraction
│   └── _parse_response()          # JSON parsing
└── FactStore Class
    ├── Deduplication Methods
    ├── Search Methods
    ├── Persistence Methods
    └── Query Methods
```

---

## Extraction Prompt

The extraction logic is defined entirely in a single prompt template:

```python
EXTRACTION_PROMPT = """You are a memory extraction system. Extract facts worth remembering from this message.

**EXTRACT:**
- Personal information (names, preferences, relationships)
- Technical details (tools, frameworks, requirements)
- Goals, problems, deadlines
- Important entities (people, places, projects)
- Corrections to previous information

**SKIP:**
- Greetings, thanks, filler words
- Vague statements without specific information
- General knowledge (e.g., "Python is a language")

{context_section}

**Message:**
{message}

**Return JSON array (empty [] if nothing worth remembering):**
[
  {"fact": "concise fact statement", "entities": ["entity1", "entity2"], "type": "preference|requirement|context|correction"}
]

**JSON only, no explanation:**"""
```

### Fact Types

| Type | Description | Example |
|------|-------------|---------|
| `preference` | User preferences and likes | "User prefers dark mode" |
| `requirement` | Project or task requirements | "API must support OAuth 2.0" |
| `context` | Background information | "User works at TechCorp" |
| `correction` | Updates to previous facts | "User now uses Python 3.11 (was 3.9)" |

---

## Data Classes

### ExtractedFact

Represents a single fact extracted from a message:

```python
@dataclass
class ExtractedFact:
    fact: str           # The fact statement
    entities: List[str] # Related entities
    fact_type: str      # preference|requirement|context|correction
```

**Methods:**
- `to_dict()` - Convert to dictionary

### StoredFact

A fact stored in the FactStore with full metadata:

```python
@dataclass
class StoredFact:
    id: str                      # Unique identifier (MD5 hash)
    fact: str                    # Fact text
    entities: List[str]          # Related entities
    fact_type: str               # Type of fact
    context_name: str            # Context/conversation name
    message_id: int              # Source message ID
    mention_count: int = 1       # Times mentioned
    created: int                 # Unix timestamp
    last_seen: int               # Last mention timestamp
    status: str = "active"       # active|superseded
    versions: List[Dict] = []    # Version history
```

**Methods:**
- `to_dict()` - Serialize to dictionary
- `from_dict(data)` - Create from dictionary

---

## FactExtractor Class

Handles LLM-based fact extraction from messages.

### Constructor

```python
def __init__(self, llm, min_message_length: int = 10)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | ChatModel | required | LangChain chat model |
| `min_message_length` | int | 10 | Minimum chars to process |

### Key Methods

#### should_extract_facts()

```python
def should_extract_facts(self, message: str, role: str) -> bool
```

Pre-filters messages before extraction:
- Only processes `user` messages
- Skips messages shorter than `min_message_length`

#### extract_facts()

```python
def extract_facts(
    self,
    message: str,
    conversation_summary: Optional[str] = None,
    recent_messages: Optional[List[str]] = None
) -> List[Dict]
```

Main extraction method:

1. Check cache for previously extracted facts
2. Build context section from summary and recent messages
3. Call LLM with formatted prompt
4. Parse JSON response
5. Validate fact structure
6. Cache and return results

**Returns:** List of fact dictionaries with keys: `fact`, `entities`, `type`

### Internal Methods

| Method | Description |
|--------|-------------|
| `_build_context_section()` | Formats context for prompt |
| `_parse_response()` | Extracts JSON array from LLM response |
| `_validate_fact()` | Validates fact structure (3-50 words) |

---

## FactStore Class

Persistent fact storage with FAISS-powered semantic deduplication.

### Constructor

```python
def __init__(
    self,
    fusion=None,                    # SearchFusion for embeddings
    persist_dir: Optional[str] = None
)
```

### Class Constants

```python
DUPLICATE_THRESHOLD = 0.90  # Above = duplicate, skip
UPDATE_THRESHOLD = 0.78     # Above = similar, merge
```

### Storage Structure

```
persist_dir/
└── facts/
    ├── facts.index           # FAISS index
    └── facts_metadata.pkl    # Fact metadata (pickle)
```

### Deduplication Operations

| Operation | Similarity | Action |
|-----------|------------|--------|
| **NOOP** | > 0.90 | Skip, increment mention_count |
| **UPDATE** | 0.78 - 0.90 | Merge facts, update embedding |
| **ADD** | < 0.78 | Add as new fact |
| **CORRECTION** | Any (type=correction) | Supersede related facts |

---

## Deduplication Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│                  add_fact() Flow                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: fact_text, entities, fact_type                     │
│                        │                                    │
│                        ▼                                    │
│           ┌────────────────────────┐                        │
│           │ Is type="correction"?  │                        │
│           └───────────┬────────────┘                        │
│              YES │         │ NO                             │
│                  ▼         ▼                                │
│        ┌──────────────┐  ┌──────────────┐                   │
│        │ Handle       │  │ Get Embedding │                  │
│        │ Correction   │  └───────┬──────┘                   │
│        │ (supersede)  │          │                          │
│        └──────────────┘          ▼                          │
│                        ┌──────────────────┐                 │
│                        │ Search FAISS     │                 │
│                        │ (top 10 similar) │                 │
│                        └────────┬─────────┘                 │
│                                 │                           │
│              ┌──────────────────┼──────────────────┐        │
│              ▼                  ▼                  ▼        │
│        score > 0.90       0.78 < score ≤ 0.90   score ≤ 0.78│
│              │                  │                  │        │
│              ▼                  ▼                  ▼        │
│         ┌────────┐        ┌─────────┐       ┌──────────┐    │
│         │  NOOP  │        │ UPDATE  │       │   ADD    │    │
│         │(skip)  │        │ (merge) │       │  (new)   │    │
│         └────────┘        └─────────┘       └──────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Correction Handling

When a fact has `type="correction"`:

1. Embed the correction fact
2. Search for related facts (by similarity + entity overlap)
3. Mark related facts as `status="superseded"`
4. Add correction as new active fact
5. Track superseded fact IDs in `supersedes` field

---

## Usage Examples

### Basic Fact Extraction

```python
from langchain_openai import ChatOpenAI
from memory.fact_extraction import FactExtractor, FactStore
from memory.search_fusion import SearchFusion

# Initialize
llm = ChatOpenAI(model="gpt-4")
extractor = FactExtractor(llm)

# Check if extraction needed
message = "I'm a Python developer working at TechCorp"
if extractor.should_extract_facts(message, role="user"):
    facts = extractor.extract_facts(
        message,
        conversation_summary="General introductions",
        recent_messages=["Hi there!", "Tell me about yourself"]
    )
    
    for fact in facts:
        print(f"Fact: {fact['fact']}")
        print(f"  Entities: {fact['entities']}")
        print(f"  Type: {fact['type']}")
```

**Output:**
```
Fact: User is a Python developer
  Entities: ['Python']
  Type: context

Fact: User works at TechCorp
  Entities: ['TechCorp']
  Type: context
```

### Using FactStore

```python
# Initialize store with vector search
fusion = SearchFusion(...)
store = FactStore(fusion=fusion, persist_dir="./memory/facts")

# Add facts
for fact in facts:
    fact_id, operation = store.add_fact(
        fact_text=fact['fact'],
        entities=fact['entities'],
        fact_type=fact['type'],
        context_name="main_chat",
        message_id=42
    )
    print(f"{operation}: {fact_id}")
```

### Searching Facts

```python
# Semantic search
results = store.search_facts(
    query="user's programming skills",
    top_k=5,
    status_filter="active"
)

for r in results:
    print(f"[{r['score']:.2f}] {r['fact']['fact']}")
```

### Handling Corrections

```python
# User corrects previous information
correction_fact = {
    'fact': 'User now uses Python 3.12 instead of 3.9',
    'entities': ['Python'],
    'type': 'correction'
}

fact_id, operation = store.add_fact(
    fact_text=correction_fact['fact'],
    entities=correction_fact['entities'],
    fact_type=correction_fact['type'],
    context_name="main_chat",
    message_id=100
)

# operation = "CORRECTION"
# Related facts about Python version are now superseded
```

### Context-Specific Queries

```python
# Get facts for a specific context
context_facts = store.get_context_facts("project_alpha", limit=20)

# Get recent facts (last 24 hours)
recent = store.get_recent_facts(hours=24)

# Get facts by type
preferences = store.get_facts_by_type("preference")
```

### Statistics

```python
stats = store.get_statistics()
print(f"Total facts: {stats['total_facts']}")
print(f"Active facts: {stats['active_facts']}")
print(f"Vector count: {stats['vector_count']}")
print(f"Contexts: {stats['contexts']}")
print(f"Types: {stats['fact_types']}")
```

---

## Configuration

### Similarity Thresholds

```python
# In FactStore class
DUPLICATE_THRESHOLD = 0.90  # Skip if similarity above
UPDATE_THRESHOLD = 0.78     # Merge if similarity above
```

**Tuning Guidelines:**

| Scenario | DUPLICATE | UPDATE | Effect |
|----------|-----------|--------|--------|
| Aggressive dedup | 0.85 | 0.70 | Fewer facts, more merging |
| Default | 0.90 | 0.78 | Balanced |
| Conservative | 0.95 | 0.85 | More facts, less merging |

### Minimum Message Length

```python
extractor = FactExtractor(llm, min_message_length=10)
```

Increase to skip more messages, decrease to extract from shorter messages.

### Search Scoring

Facts are scored with boosts for:
- **Recency**: `0.1 / (1.0 + days_since_last_seen)`
- **Corrections**: `+0.15` for correction-type facts

---

## API Reference

### FactExtractor Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `should_extract_facts()` | `message, role` | `bool` | Pre-filter check |
| `extract_facts()` | `message, summary?, recent?` | `List[Dict]` | Extract facts via LLM |

### FactStore Methods

#### Core Operations

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `add_fact()` | `fact_text, entities, fact_type, context_name, message_id` | `Tuple[str, str]` | Add/dedupe fact |

#### Search Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `search_facts()` | `query, top_k=20, status_filter="active"` | `List[Dict]` | Semantic search |
| `get_context_facts()` | `context_name, limit=30` | `List[Dict]` | Facts by context |
| `get_recent_facts()` | `hours=24, context_name?` | `List[Dict]` | Recent facts |
| `get_facts_by_type()` | `fact_type, context_name?` | `List[Dict]` | Facts by type |

#### Management Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_statistics()` | - | `Dict` | Store statistics |
| `clear_context_facts()` | `context_name` | `int` | Remove context facts |
| `persist()` | - | - | Force save to disk |

### Return Types

#### add_fact() Returns

```python
(fact_id: str, operation: str)
# operation: "ADD" | "UPDATE" | "NOOP" | "CORRECTION"
```

#### search_facts() Result Item

```python
{
    'fact_id': str,
    'score': float,
    'fact': {
        'id': str,
        'fact': str,
        'entities': List[str],
        'type': str,
        'context_name': str,
        'mention_count': int,
        'created': int,
        'last_seen': int,
        'status': str
    }
}
```

---

## Persistence Format

### facts_metadata.pkl

Pickle file containing:

```python
{
    'facts': {
        'fact_id_1': {
            'id': 'fact_id_1',
            'fact': 'User prefers Python',
            'entities': ['Python'],
            'type': 'preference',
            'context_name': 'main_chat',
            'message_id': 42,
            'mention_count': 3,
            'created': 1705849200,
            'last_seen': 1705935600,
            'status': 'active'
        },
        ...
    },
    'fact_ids': ['fact_id_1', 'fact_id_2', ...]
}
```

### facts.index

FAISS IndexIDMap containing normalized embeddings for all facts.

---

## Error Handling

### Extraction Errors

- **LLM failure**: Returns empty list `[]`
- **JSON parse error**: Logs warning, returns empty list
- **Validation failure**: Filters out invalid facts

### Storage Errors

- **FAISS not installed**: Vector deduplication disabled, simple storage used
- **Embedding update failure**: Triggers full index rebuild
- **Persistence failure**: Logs error, continues in-memory

---

## Dependencies

```python
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import json
import logging
import time
import hashlib
import pickle
import os
from pathlib import Path
import numpy as np
import faiss  # Optional, for vector search
from langchain_core.messages import HumanMessage
```

---

## Performance Notes

- **Caching**: Extracted facts cached by message hash
- **Batch search**: FAISS searches up to 10 similar facts at once
- **Status filtering**: Done post-search, queries up to 3x top_k
- **Lazy rebuild**: Index rebuilt only on embedding update failure

---

## Integration with Other Modules

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory System                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐      ┌────────────────┐                 │
│   │ FactExtractor│ ───► │   FactStore    │                 │
│   └──────────────┘      └────────┬───────┘                 │
│          ▲                       │                         │
│          │                       ▼                         │
│   ┌──────────────┐      ┌────────────────┐                 │
│   │ SearchFusion │ ◄──► │  Checkpoints   │                 │
│   │ (embeddings) │      │   (summary)    │                 │
│   └──────────────┘      └────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- **SearchFusion**: Provides embeddings for semantic deduplication
- **Checkpoints**: Uses extracted facts for knowledge state
- **Context Builder**: Retrieves facts for context injection

---

*Documentation generated for `memory/fact_extraction.py`*
