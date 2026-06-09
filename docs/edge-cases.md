# Edge Cases for Engram Memory System

This page lists behavior that application developers should test explicitly.
The current architecture adds deterministic critical recall, conflict metadata,
task memory, and long-input ingestion; those features have their own edge cases
in addition to vector search and CRUD behavior.

## 1. **Data Integrity**

### Duplicate Facts with Slight Variations
```python
# These all mean the same thing but might not dedupe:
"User's name is Nafiz"
"User is called Nafiz"
"User name: Nafiz"
"The user's name is Nafiz"

# Edge case: Score 0.84 (below 0.85 threshold) → stores duplicate
```

### Empty or Whitespace-Only Content
```python
await engram.add(content="   ", agent_id="assistant")  # Should reject
await engram.add(content="", agent_id="assistant")      # Should reject
await engram.add(content="\n\n\n", agent_id="assistant") # Should reject
```

### Extremely Long Content
```python
# Fact extraction produces 5000-char fact
fact = "User mentioned that " + ("very " * 1000) + "important"

# Edge case: Exceeds embedding model token limit (8192 for OpenAI)
await engram.add(content=fact, agent_id="assistant")  # Will it truncate or fail?
```

### Unicode and Special Characters
```python
# Emojis, non-Latin scripts, special chars
await engram.add(content="User likes 🍕 and lives in München", ...)
await engram.add(content="用户喜欢机器学习", ...)  # Chinese
await engram.add(content="User's email: test@example.com", ...)
await engram.add(content="Query: SELECT * FROM users WHERE id=1", ...)  # SQL injection in content?
```

### NULL Values in Optional Fields
```python
await engram.add(
    content="Fact",
    agent_id="assistant",
    user_id=None,        # Should work
    session_id=None,     # Should work
    main_content=None,   # Should work
    metadata=None        # Should default to {}
)
```

---

## 2. **Search & Retrieval**

### No Results Found
```python
# Brand new user with no memories
results = await engram.search(query="user preferences", agent_id="assistant", user_id="new_user")
# Edge case: Empty list, how does chatbot handle?
```

### Query Matches Both Fact and Main Content
```python
# Fact: "User works at AskTuring"
# Main_content: "[USER]: I work at AskTuring as ML engineer..."

# Query: "AskTuring"
# Edge case: Does it return duplicate-like results from same memory?
```

### Query Contains Stop Words Only
```python
results = await engram.search(query="the and or but", ...)  # No meaningful terms
results = await engram.search(query="", ...)                # Empty query
results = await engram.search(query="   ", ...)             # Whitespace only
```

### Extremely Long Query
```python
# User pastes entire essay as query
query = "..." * 10000  # 10K char query
results = await engram.search(query=query, ...)
# Edge case: Embedding API token limit exceeded
```

Use `record_long_input()` for multi-thousand-token prompts or documents. Use
`search()` for concise retrieval queries.

### Score Ties
```python
# Multiple memories have identical scores (e.g., 0.753)
# Which one ranks first? Is ordering stable?
```

### Time Decay Edge Cases
```python
# Memory created exactly NOW
memory = await engram.add(...)
results = await engram.search(...)  # Decay = 0.995^0 = 1.0?

# Memory from year 2000 (before cutoff)
# Decay = 0.995^(24*365*25) = basically 0
# Does it still appear in results?
```

---

## 3. **Concurrent Operations**

### Race Condition on Duplicate Detection
```python
# Two requests process same fact simultaneously
# Request A: searches, finds no duplicate, starts adding
# Request B: searches, finds no duplicate, starts adding
# Result: Both add the same fact (unique constraint should catch, but does it?)
```

### Simultaneous Reinforcement
```python
# Two processes reinforce same memory at same time
# Process A: importance = 0.5, boost 0.1 → 0.6
# Process B: importance = 0.5, boost 0.1 → 0.6
# Expected: 0.7, Actual: 0.6? (lost update problem)
```

### Search During Add
```python
# Search is running (slow query)
# Add completes and commits
# Does search see the new memory mid-execution?
```

### Simultaneous Corrections To The Same Critical Slot
```python
# Request A: "User is allergic to cashews"
# Request B: "Correction: user is not allergic to cashews"
# Both produce the same conflict_key
# Edge case: Which memory remains active?
```

The application should treat the latest committed row with `status=active` as
authoritative and inspect superseded rows during audits.

---

## 4. **Memory Decay System**

### Importance Overflow
```python
# Memory starts at 0.9
# Reinforced 20 times with 0.1 boost each
# Expected: capped at 1.0
# Actual: 2.9? (if no cap logic)
```

### Negative Importance
```python
# Memory with importance 0.1
# Decay reduces it repeatedly
# Can importance go negative? Should it be deleted?
```

### Zero Importance Memories
```python
# Memory decayed to importance = 0.0
# Does it still appear in search? Should it be purged?
```

### Clock Skew / Time Zones
```python
# Server A: created_at = 2025-01-23 10:00:00 UTC
# Server B: last_accessed_at = 2025-01-23 09:00:00 EST (1 hour behind)
# Decay calculation: negative hours_elapsed?
```

---

## 5. **Graph Relationships**

### Circular References
```python
# Memory A relates to B
# Memory B relates to C
# Memory C relates to A (cycle)

results = await engram.traverse(start_id=A, max_depth=10)
# Edge case: Infinite loop or handles cycles?
```

### Orphaned Relations
```python
# Memory A deleted, but relation A→B still exists
# Traverse from B finds dangling reference
# Does it fail or skip gracefully?
```

### Self-Reference
```python
await engram.relate(source_id=memory.id, target_id=memory.id)
# Memory relates to itself - should this be allowed?
```

### Max Depth Exceeded
```python
# Graph has 100 levels of depth
results = await engram.traverse(start_id=A, max_depth=3)
# Only returns 3 levels, but user expects full graph
```

---

## 6. **Session Management**

### Session Timeout Race
```python
async with engram.session(...) as session:
    # Long operation (30 min)
    await asyncio.sleep(1800)
    # Session expired mid-operation
    await engram.add(session_id=session.session_id)  # Fails?
```

### Nested Sessions
```python
async with engram.session(...) as session1:
    async with engram.session(...) as session2:
        # Same user, two active sessions?
        # Should this be allowed?
        ...
```

### Session Without User
```python
async with engram.session(agent_id="assistant", user_id=None) as session:
    # Session with no user - is this valid?
    await engram.add(session_id=session.session_id, user_id="user_123")
    # user_id mismatch - should this be rejected?
```

---

## 7. **Embedding Provider Issues**

### API Rate Limits
```python
# Embed 1000 facts in quick succession
for fact in facts:
    await engram.add(content=fact, ...)
# Edge case: OpenAI rate limit (3000 RPM) exceeded
```

### Embedding Dimension Mismatch
```python
# Database configured for 1536 dimensions
# Switch to model with 768 dimensions
await engram.add(...)  # Inserts 768-dim vector into 1536-dim column?
```

### Embedding Cache Poisoning
```python
# Cached embedding for "User likes coffee"
# User updates to "User likes tea"
# Search still uses old cached embedding for "User likes coffee"
```

### Provider Downtime
```python
# OpenAI API is down
await engram.add(content="fact", ...)  # Hangs? Times out? Retries?
```

---

## 8. **Critical Recall And Conflict Resolution**

### Critical Fact Not In Final Prompt
```python
trace = await engram.trace_recall(
    "What can I order for dinner?",
    agent_id="assistant",
    user_id="user",
    expected_terms=["shellfish"],
    max_tokens=100,
)

# Edge case: allergy exists, but max_tokens is too small
assert "shellfish" in trace.missing_expected_terms
assert trace.trimmed_memory_ids
```

### Superseded Fact Still Appears In Search
```python
# Old: "User lives in Dhaka"
# New: "User moved to Seattle"
# Edge case: old fact should not appear in normal search
```

Verify:

```python
trace = await engram.trace_recall("where does the user live?", "assistant")
assert old_memory_id in trace.superseded_memory_ids
assert old_memory_id not in trace.kept_memory_ids
```

### Wrong Critical Slot
```python
# "User has a shellfish allergy" and "User has a peanut allergy"
# Edge case: policy collapses both into profile:allergy instead of separate allergens
```

Use custom `SlotRule` objects when domain-specific slots need stricter parsing.

---

## 9. **Task Memory**

### Worker Not Running
```python
await engram.record_turn(task_id, user_msg, assistant_msg)
# No process_memory_jobs() or run_memory_worker()
# Edge case: raw events exist, but derived facts/checkpoints lag behind
```

Mitigation: build task context includes recent events, but search recall may not
see newly derived facts until jobs are processed.

### Failed Memory Job
```sql
SELECT job_id, error FROM memory_jobs WHERE status = 'failed';
```

Failed jobs preserve the raw event ledger. Fix the underlying issue and decide
whether to retry or create a replacement event/job.

### Redacted Events And Existing Derived Memories
```python
await engram.redact_event(event_id)
# Edge case: derived memories from the old event may still exist
```

Redaction clears the event. Applications that need strict privacy deletion
should also delete or supersede derived memories linked by `source_event_id`.

---

## 10. **Long Input And Legal Documents**

### Relative Time In Long Prompts
```python
await engram.record_long_input(
    task_id,
    "We need this reviewed tomorrow and next Friday.",
)
```

Engram records time notes where it can, but applications should normalize
relative dates before high-stakes use when exact deadlines matter.

### Missing Source Chunk
```python
context = await engram.build_long_input_context(
    task_id,
    query="termination notice",
    expected_terms=["termination", "notice"],
)
```

If `context.trace["missing_expected_terms"]` is non-empty, the application should
ask for clarification, retrieve more source chunks, or refuse exact-document
answers.

### Source Chunk Too Large For Prompt
```python
context = await engram.build_long_input_context(
    task_id,
    query="summarize all obligations",
    max_tokens=1000,
)
```

Relevant chunks may be trimmed. Inspect `context.trace["trimmed_sections"]`.

### Citation Metadata Missing

Engram stores character spans and quote hashes. If you need PDF page numbers,
line numbers, or OCR bounding boxes, add those fields in `metadata` before
calling `record_long_input()`.

---

## 8. **Database Issues**

### Connection Pool Exhausted
```python
# 50 concurrent requests, pool max = 10
# 40 requests waiting for connection
# Request times out after 30 seconds?
```

### Transaction Rollback Mid-Operation
```python
# Add memory, relation, and update importance in transaction
# Database crashes after add but before relation
# Partial state? Rollback? Retry?
```

### Vector Index Corruption
```python
# HNSW index corrupted (rare but possible)
await engram.search(...)  # Returns wrong results or crashes?
```

### Out of Disk Space
```python
# PostgreSQL disk full
await engram.add(...)  # Fails silently? Raises exception?
```

---

## 9. **Multi-Tenancy**

### User ID Leak
```python
# User A's query accidentally includes User B's user_id
results = await engram.search(
    query="my secrets",
    agent_id="assistant",
    user_id="user_B"  # Should be user_A
)
# Returns User B's data to User A!
```

### Agent ID Collision
```python
# Two different apps use same agent_id="assistant"
# App A and App B see each other's memories
```

### Missing User Filtering
```python
# Forget to pass user_id in search
results = await engram.search(query="password", agent_id="assistant")
# Returns ALL users' memories!
```

---

## 10. **Fact Extraction (LLM)**

### LLM Returns Malformed JSON
```python
# Expected: ["fact1", "fact2"]
# Actual: "Here are the facts:\n- fact1\n- fact2"  # Not JSON
await llm.extract_facts(...)  # Parsing fails?
```

### LLM Returns Empty List
```python
# User: "Hello"
# Bot: "Hi!"
facts = await llm.extract_facts(...)  # Returns []
# Nothing to store - is this handled?
```

### LLM Hallucinates Facts
```python
# User: "I like pizza"
# LLM extracts: ["User likes pizza", "User is Italian"]  # Hallucination!
# False fact stored in memory
```

### LLM Timeout
```python
# LLM API takes 60+ seconds
facts = await llm.extract_facts(...)  # Request times out mid-extraction
```

---

## 11. **Memory Operations (ADD/UPDATE/DELETE)**

### Evaluate Operation Returns Invalid Action
```python
# LLM returns operation="MERGE" (not in enum)
# Expected: ADD, UPDATE, DELETE, NOOP
operation = await llm.evaluate_memory_operation(...)
# Edge case: Unknown operation type
```

### Update Target Doesn't Exist
```python
# LLM says UPDATE memory "mem_xyz"
# But "mem_xyz" was deleted 5 seconds ago
await engram.update("mem_xyz", ...)  # 404 error?
```

### Circular Update Chain
```python
# New fact: "User works at Google"
# Existing: "User works at AskTuring" (id=mem_1)
# LLM: UPDATE mem_1
# But mem_1 was already superseded by mem_2
# Edge case: Chasing superseded memories
```

---

## 12. **API/Integration Edge Cases**

### FastAPI Lifespan Failure
```python
# engram.connect() fails in lifespan
# App starts anyway with engram=None
# First request crashes with "Engram not initialized"
```

### Request Canceled Mid-Operation
```python
# User cancels HTTP request
# Memory add is in progress
# Does transaction rollback? Memory leaked?
```

### Async Context Manager Not Used
```python
engram = Engram()
await engram.connect()
# User forgets await engram.close()
# Connection pool never cleaned up → memory leak
```

---

## 13. **Configuration Edge Cases**

### Weights Don't Sum to 1.0
```python
ENGRAM_WEIGHT_SEMANTIC=0.5
ENGRAM_WEIGHT_KEYWORD=0.3
ENGRAM_WEIGHT_DECAY=0.3  # Sum = 1.1
ENGRAM_WEIGHT_IMPORTANCE=0.1

# Should validation catch this?
```

### Decay Rate Outside Valid Range
```python
ENGRAM_DECAY_RATE=1.5  # Greater than 1.0 (memories get MORE relevant over time?!)
ENGRAM_DECAY_RATE=-0.5  # Negative (undefined behavior)
```

### Missing Required Config
```python
# No ENGRAM_DATABASE_URL set
engram = Engram()
await engram.connect()  # Should fail immediately, not at first query
```

---

## 14. **Two-Column System Specific**

### Main Content Without Fact
```python
await engram.add(
    content="",  # Empty fact
    main_content="[USER]: Hello\n[AI]: Hi!"  # Has context
)
# Should this be rejected? Fact is required for search
```

### Fact and Main Content Mismatch
```python
await engram.add(
    content="User likes pizza",
    main_content="[USER]: I hate pizza\n[AI]: Noted!"  # Contradicts fact
)
# Which one is "truth"?
```

### Extremely Long Main Content
```python
# User pastes 50,000-char essay in chat
main_content = f"[USER]: {essay_50k_chars}\n[AI]: Got it!"
await engram.add(content="User wrote essay", main_content=main_content)
# Edge case: TEXT column limit? Token limit when retrieved?
```

### Main Content Contains Special Formatting
```python
main_content = "[USER]: <script>alert('xss')</script>\n[AI]: Okay"
# If displayed in UI later, could cause XSS?
```

---

## 15. **Performance Edge Cases**

### Search Returns 10,000 Results
```python
# All memories match query equally
results = await engram.search(query="the", limit=10000)
# Edge case: Massive result set, memory/network limits
```

### Embedding Cache Fills Up
```python
# Cache size = 1000, but need to embed 10,000 unique texts
# Cache thrashes (constant evictions)
# Performance degrades 10x
```

### HNSW Index Rebuild During Query
```python
# Vacuum/reindex running in background
await engram.search(...)  # Slow or locks?
```

---

## Summary Table

| Category | Critical Edge Cases | Likelihood | Impact |
|----------|---------------------|------------|--------|
| **Data Integrity** | Duplicate detection false negatives | High | Medium |
| **Search** | Empty results, query overflow | Medium | High |
| **Concurrency** | Race conditions on add/reinforce | Medium | High |
| **Decay** | Importance overflow/underflow | Low | Medium |
| **Graph** | Circular references, orphaned relations | Medium | Medium |
| **Sessions** | Timeout race, nested sessions | Low | Low |
| **Embeddings** | Rate limits, dimension mismatch | High | Critical |
| **Database** | Connection pool exhaustion | Medium | Critical |
| **Multi-tenancy** | User ID leak | Low | Critical |
| **LLM** | Hallucinated facts, malformed JSON | High | High |
| **Two-column** | Fact/context mismatch | Low | Medium |
| **Performance** | Cache thrashing, massive results | Medium | High |

**Most Critical to Test:**
1. ❗ Embedding provider rate limits/failures
2. ❗ User ID filtering (security)
3. ❗ Connection pool exhaustion
4. ❗ Race conditions on duplicate detection
5. ❗ LLM hallucinations stored as facts
