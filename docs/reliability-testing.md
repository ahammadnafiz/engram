# Reliability Testing

This page lists failure modes application teams should test around Engram. It is
not a promise that Engram handles every case automatically. It identifies where
the library has guards and where the application must add policy.

## Test Database Isolation

Integration tests should never target a development or production database.
Use an explicit test URL:

```bash
export ENGRAM_TEST_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram_test
pytest tests/integration -q --run-integration
```

The test harness respects `ENGRAM_TEST_DATABASE_URL` and does not let `.env`
override a caller-provided database URL.

## Data Integrity

### Empty Facts

`MemoryCreate` requires non-empty `content`.

```python
from pydantic import ValidationError

from engram.memory.models import MemoryCreate


try:
    MemoryCreate(content="", agent_id="assistant")
except ValidationError as exc:
    print(exc)
```

### Long Facts And Long Context

`Memory.content` and `MemoryCreate.content` allow up to 100,000 characters.
`main_content` allows up to 200,000 characters. Embedding input is truncated by
`ENGRAM_EMBEDDING_MAX_INPUT_CHARS` before it reaches the provider.

For large source documents, prefer `record_long_input()` instead of one huge
fact:

```python
report = await engram.record_long_input(
    task.task_run_id,
    text=large_document,
    title="Vendor MSA",
)
```

### Duplicate And Near-Duplicate Facts

Engram uses an exact fact uniqueness index and a vector near-duplicate guard.
Test both exact and reworded duplicates in your domain.

```python
first = await engram.add("User reports to Priya", "assistant", user_id="sarah")
second = await engram.add("User's manager is Priya", "assistant", user_id="sarah")

print(first.memory_id)
print(second.memory_id)
```

Tune `ENGRAM_NEAR_DUPLICATE_THRESHOLD` if your domain needs more or less
aggressive duplicate suppression.

## Search And Recall

### Empty Result Sets

Your application must handle empty search results.

```python
results = await engram.search(
    "preferences",
    "assistant",
    user_id="new_user",
)

if not results:
    prompt_memory = ""
```

### Empty Or Stop-Word Queries

`SearchQuery` requires a non-empty query. Treat empty user prompts as application
input errors before calling Engram.

```python
query = user_message.strip()
if query:
    results = await engram.search(query, "assistant", user_id=user_id)
else:
    results = []
```

### Broad Prompts

For broad prompts, use `trace_recall()` with expected terms so missed recall is
visible.

```python
trace = await engram.trace_recall(
    "Can we launch today?",
    "assistant",
    user_id=user_id,
    expected_terms=["rollback owner", "error rate"],
)

if trace.missing_expected_terms:
    logger.warning("missing expected memory terms: %s", trace.missing_expected_terms)
```

## Conflict Resolution

Critical facts can share a `conflict_key`. The newer active fact supersedes the
older one.

```python
old = await engram.add(
    "User is allergic to cashews",
    "assistant",
    user_id="sarah",
)
new = await engram.add(
    "Correction: user is not allergic to cashews",
    "assistant",
    user_id="sarah",
)

trace = await engram.trace_recall(
    "Can the user eat cashews?",
    "assistant",
    user_id="sarah",
)

print(old.memory_id in trace.superseded_memory_ids)
print(new.memory_id in trace.kept_memory_ids)
```

If a domain needs stricter slots, define explicit `SlotRule` objects. Generic
content-digest slots are off by default because they do not match reworded
corrections.

## Concurrency

The store uses database constraints and advisory locking around near-duplicate
add paths. Still test concurrency around your highest-volume writes.

```python
import asyncio


async def add_fact(text: str):
    return await engram.add(text, "assistant", user_id="sarah")


await asyncio.gather(
    add_fact("User reports to Priya"),
    add_fact("User's manager is Priya"),
)
```

For multi-step application workflows, wrap your own operations in application
transactions or make them idempotent.

## Embedding Providers

### Rate Limits And Downtime

Cloud embedding calls can rate-limit or fail. Decide whether the application
should fail fast, retry, or queue work.

```python
try:
    await engram.add("User prefers concise answers", "assistant", user_id=user_id)
except Exception as exc:
    logger.exception("memory write failed: %s", exc)
```

### Dimension Mismatch

Switching providers can change vector dimension. Engram blocks destructive
dimension changes when existing embeddings are present.

```bash
export ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=false
```

Use a fresh database for experiments that switch between OpenAI embeddings and
sentence-transformer embeddings.

## Graph Relations

Recursive traversal has depth and limit guards. Test cycles, missing seeds, and
direction filters.

```python
await engram.relate(a.memory_id, b.memory_id, "related_to")
await engram.relate(b.memory_id, c.memory_id, "related_to")
await engram.relate(c.memory_id, a.memory_id, "related_to")

results = await engram.traverse(a.memory_id, max_depth=3, direction="any")
```

Use `traverse_many(..., skip_missing=True)` when prompt assembly should continue
even if one seed was deleted.

## Task Memory

### Worker Not Running

If no worker runs, raw events still exist, but derived memories and checkpoints
lag behind.

```python
await engram.record_turn(task.task_run_id, user_message, assistant_response)

context = await engram.build_context(task.task_run_id, query=user_message)
```

`build_context()` includes recent events, so the task can still resume, but
ordinary memory search may not see derived facts until jobs are processed.

### Failed Jobs

```sql
SELECT job_id, attempts, error, updated_at
FROM memory_jobs
WHERE status = 'failed'
ORDER BY updated_at DESC;
```

Failed jobs preserve the raw ledger. Fix the underlying cause and decide whether
to retry, requeue, or create a replacement event.

### Redaction

`redact_event()` clears event content and payload. It does not automatically
delete memories derived from that event.

```python
await engram.redact_event(event_id)
derived = await engram.get_memories(
    "assistant",
    user_id=user_id,
    metadata_filter={"source_event_id": event_id},
)
```

Strict privacy deletion should also delete or supersede derived memories.

## Long Input

### Relative Dates

Engram records time notes when it can, but high-stakes apps should normalize
relative dates before ingestion.

```python
await engram.record_long_input(
    task.task_run_id,
    "We need this reviewed tomorrow and next Friday.",
    metadata={"received_at": "2026-06-14T09:00:00Z"},
)
```

### Missing Expected Terms

```python
context = await engram.build_long_input_context(
    task.task_run_id,
    query="termination notice",
    expected_terms=["termination", "notice"],
)

if context.trace["missing_expected_terms"]:
    raise RuntimeError("source context missing required terms")
```

### Citation Metadata

Engram stores character spans and quote hashes. If you need PDF page numbers,
line numbers, or OCR boxes, add them in `metadata` before ingestion.

## Multi-Tenancy

Engram scopes data by `agent_id` and optional `user_id`; it does not enforce
authorization.

```python
results = await engram.search(
    "password reset",
    "support-agent",
    user_id=authenticated_user_id,
)
```

Do not trust a client-supplied `user_id`. Resolve it from your auth layer.

## Configuration

Test configuration validation in CI:

```python
from pydantic import ValidationError

from engram import EngramSettings


try:
    EngramSettings(
        weight_semantic=0.5,
        weight_keyword=0.3,
        weight_decay=0.3,
        weight_importance=0.1,
    )
except ValidationError as exc:
    print(exc)
```

Common failures:

- weights do not sum to `1.0`
- `max_pool_size` is lower than `min_pool_size`
- invalid `text_search_config`
- missing optional provider package
- missing API key for a cloud provider
- embedding dimension mismatch against a populated database

## Minimum Test Matrix

| Area | Test |
|------|------|
| database isolation | integration suite uses a throwaway database |
| data integrity | empty facts fail, long facts are handled, duplicates collapse |
| search | empty results, type filters, metadata filters, min score |
| critical recall | critical memories appear before ordinary hits |
| conflict resolution | corrected facts supersede older active facts |
| task memory | events and jobs commit together, worker creates checkpoints |
| privacy | redaction plus derived-memory cleanup policy |
| long input | expected terms and source chunk IDs are present |
| multi-tenancy | every app route resolves `user_id` from auth |
| operations | failed jobs and pool saturation are monitored |
