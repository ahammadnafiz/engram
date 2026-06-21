# Reliability & Testing

This guide outlines the failure modes, edge cases, and security boundaries application teams must test when integrating Engram. 

> [!NOTE]
> Engram guarantees database-level integrity, vector storage, and memory consistency. However, it **does not** enforce application-level authorization or handle external LLM network retries automatically.

---

## 1. Test Database Isolation

Your automated test suite (e.g., `pytest`) must never execute against a development or production database.

To isolate tests, provide a specific test database URL:

```bash
export ENGRAM_TEST_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram_test
pytest tests/integration -q --run-integration
```

> [!TIP]
> Engram's internal test harness strictly respects `ENGRAM_TEST_DATABASE_URL` and will safely ignore the standard `.env` file if the test URL is set.

---

## 2. Multi-Tenancy & Authorization

Engram scopes all data by `agent_id` and (optionally) `user_id`. It **does not** enforce authorization.

> [!CAUTION]  
> Never trust a client-supplied `user_id` from a JSON body or URL parameter. You must resolve the `user_id` strictly from your web framework's secure authentication layer (e.g., a JWT token) before passing it to Engram.

**Failure Mode Test:** Attempt to search memories using a valid `user_id` but with a mocked, invalid JWT token. The web layer should block the request before Engram is ever called.

---

## 3. Data Integrity & Ingestion

### Empty or Massive Facts
Engram relies on `pydantic` for validation.
- **Empty Facts**: `MemoryCreate` will throw a `ValidationError` if `content` is empty.
- **Massive Facts**: `content` is capped at 100,000 characters. `main_content` is capped at 200,000. Text exceeding `ENGRAM_EMBEDDING_MAX_INPUT_CHARS` will be safely truncated before reaching the embedding provider.

### Concurrency and Duplicates
Engram uses a mix of exact-hash uniqueness and vector cosine-similarity thresholds (`ENGRAM_NEAR_DUPLICATE_THRESHOLD`) to prevent fact duplication.

**Test:** Fire concurrent writes using `asyncio.gather` to ensure your database pool and Engram's advisory locks handle race conditions.

```python
import asyncio

async def add_fact(text: str):
    return await engram.add(text, "assistant", user_id="sarah")

await asyncio.gather(
    add_fact("User reports to Priya"),
    add_fact("User's manager is Priya"),
)
```

---

## 4. Intelligent Recall & Conflicts

### Expected Term Validation
For high-stakes tasks, use the `trace_recall()` operator to explicitly assert that Engram retrieved a specific concept.

```python
trace = await engram.trace_recall(
    query="Can we launch today?",
    agent_id="assistant",
    user_id=user_id,
    expected_terms=["rollback owner", "error rate"],
)

if trace.missing_expected_terms:
    raise RuntimeError(f"Safety guard triggered. Missing: {trace.missing_expected_terms}")
```

### Conflict Resolution
When an agent learns a new, contradictory critical fact, Engram uses the `conflict_key` to supersede the old fact. 

**Test:** Ensure the older fact's ID moves to `trace.superseded_memory_ids` and the new fact is in `trace.kept_memory_ids`.

### Write-Path Lineage (`add_conversation`)

The lineage write path has two failure sources that need separate tests, because conflating them hides which layer is wrong:

**Layer 1 — mechanism (deterministic, no LLM).** Drive `add()` / `revise()` / `conflict_key` directly and assert the supersession state machine: exactly one active head per lineage, the head pointer agrees with the active row, superseded rows carry valid forward links, oscillation (A→B→A) mints a *new* row rather than resurrecting the old one, unrelated lineages don't collide, and concurrent revises serialize to a single active head. This must be 100% on any model. `benchmark/lineage_invariants.py` runs it (free, no API) and exits non-zero on violation, so it can gate CI.

**Layer 2 — write-time intelligence (LLM-gated).** Run scripted update sequences through `add_conversation()` and check the **update-capture rate**: an intended ADD/UPDATE that comes back with no written memory is a dropped update. `benchmark/lineage_writepath.py` covers numeric/date changes, oscillation, dedup, the assistant-restatement trap, contradictions, and multi-fact non-collision across models.

> [!WARNING]
> `add_conversation()` can decide `NOOP` for a fact that was actually a real update (most often a number/date change a weak embedding rates as a duplicate, or a value reversal). It now returns a list-compatible `ConversationResult`; **inspect `.decisions`** to catch this — each extracted fact carries its `operation`, `applied`, and `reason`, so a skipped update is visible (`operation="NOOP", applied=False`) instead of silently missing from the written list. On a write path, treat an unexpected `NOOP` on a stated change as an error to surface or override.

---

## 5. Network & Infrastructure

### Embedding Dimension Mismatch
If you change your embedding model (e.g., OpenAI `1536` to local `384`), Engram will throw a `ConfigurationError` on boot to prevent wiping your vectors. 

**Test:** Verify your app fails safely if this occurs, or export `ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true` if you have an automated re-embedding pipeline.

### Worker Outages
If your memory worker goes offline, raw events will continue to queue in `memory_jobs`, but new semantic facts won't be derived.
- **Test**: Shut down the worker. Verify that `engram.build_context()` still allows the user to continue their current task using the raw recent events, even if deep semantic search degrades.

---

## 6. Privacy & Redaction

When a user requests data deletion or an event is flagged for PII, you must trigger redaction.

> [!WARNING]
> Calling `engram.redact_event(event_id)` securely scrubs the raw ledger payload. However, it **does not** automatically hunt down and delete semantic facts derived from that event. You must implement a policy to `forget()` or `supersede()` those derived memories based on your application's privacy requirements.

---

## 7. The Minimum Test Matrix

Application teams should use this matrix to build their integration tests:

| Domain | Required Integration Test |
|--------|---------------------------|
| **Isolation** | Suite executes strictly against a throwaway/test database. |
| **Auth Boundaries** | Endpoints fail if `user_id` is missing or spoofed. |
| **Data Integrity** | Submitting empty facts throws errors; near-duplicates gracefully collapse. |
| **Recall Quality** | `trace_recall` confirms critical memories appear before ordinary vector hits. |
| **Conflict Logic** | Correcting a user's preference supersedes the old active fact. |
| **Task Resiliency** | Tasks can resume from checkpoints even if the background worker lags. |
| **Privacy Cascade** | Scrubbing an event also triggers the deletion of downstream derived facts. |
| **Long Documents** | `expected_terms` and source `chunk_ids` are verified during exact-document queries. |
