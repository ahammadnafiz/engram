# Production Guide

Engram is an alpha-stage cognitive architecture. This guide outlines the best practices, security requirements, and operational strategies required to run it safely and reliably in a production environment.

> [!WARNING]
> Because Engram is in active development, the database schema and API contracts are subject to change. Ensure you have automated backups and a staged deployment environment before upgrading versions.

---

## 1. Deployment Checklist

Before taking an Engram-backed application live, ensure you have addressed the following:

- [ ] **Database Setup**: You are running PostgreSQL 16+ with the `pgvector` and `pg_trgm` extensions enabled.
- [ ] **Process Isolation**: You have separated your synchronous API web servers from your asynchronous background Memory Workers.
- [ ] **Multi-Tenant Security**: Every `engram` operation is strictly scoped by `user_id` and `agent_id` at the API layer.
- [ ] **Connection Pooling**: You are using PgBouncer (or similar) in front of PostgreSQL, and have tuned `ENGRAM_MIN_POOL_SIZE` and `ENGRAM_MAX_POOL_SIZE` appropriately for your app nodes.
- [ ] **Policy Selection**: You have explicitly chosen or written a `MemoryPolicy` that matches your domain's needs (e.g., `coding_agent`, `legal`).
- [ ] **Metrics**: You are actively monitoring the `memory_jobs` table for backlog saturation and derivation failures.

---

## 2. Recommended Runtime Architecture

Engram is designed to be highly concurrent, but to achieve low latency for the end user, you must separate **retrieval** from **derivation**.

| Component | Responsibility | Scaling Strategy |
|-----------|----------------|------------------|
| **API / Web Process** | Executes `engram.recall()` to fetch context, streams LLM responses, and appends raw events to the ledger. | Scale horizontally based on user traffic. |
| **Memory Worker** | Runs `engram.run_memory_worker()` continuously. Pops raw events off the queue, calls the LLM to extract facts, and creates task checkpoints. | Scale vertically or horizontally based on LLM throughput limits and job queue depth. |
| **PostgreSQL** | Stores vectors, task ledgers, and handles recursive graph queries. | Scale vertically (RAM) to keep the active `pgvector` indexes in memory. |

---

## 3. The Production Turn Loop

In production, you should rely on the `recall()` operator to intelligently route the user's intent, rather than manually hacking together vector searches and keyword lookups.

```python
async def handle_turn(engram, task_id, agent_id, user_id, user_message):
    
    # 1. Intelligently fetch evidence (Vector + Ledger + Lineage)
    trace = await engram.recall(
        query=user_message,
        agent_id=agent_id,
        user_id=user_id,
        compose_answer=False  # We only want the context, not an auto-reply
    )

    # 2. Call your LLM, streaming the result to the user
    response = await call_your_llm(
        memory_context=trace.context,
        user_message=user_message,
    )

    # 3. Append to the immutable ledger and queue background processing
    await engram.record_turn(
        task_id,
        user_message=user_message,
        assistant_response=response,
        agent_id=agent_id,
        user_id=user_id,
        enqueue_processing=True,  # Crucial: defers the heavy extraction
    )

    return response
```

### The Worker Process
In a completely separate deployment service (e.g., a background container):

```python
from engram import Engram

async def boot_worker():
    async with Engram(memory_policy="coding_agent") as engram:
        # Runs infinitely, processing 20 events per second
        await engram.run_memory_worker(batch_size=20, interval_seconds=1.0)
```

---

## 4. Security & Privacy

Engram stores raw conversations and distilled insights. Treat this data as highly sensitive.

- **App-Level Auth**: Engram does not enforce authentication. Your FastAPI/Django/Express layer must authenticate the user before passing their `user_id` down to Engram.
- **Provider Leaks**: Be extremely careful about which LLM provider you use for background derivation. Do not send sensitive enterprise data to public models unless you have a zero-retention data processing agreement (DPA).
- **Data Deletion**: To comply with GDPR or CCPA, use `purge(user_id="...")` to destroy all semantic facts, and ensure your app layer drops the corresponding task ledgers.
- **Redaction**: If a user pastes API keys or PII, use the `redact_event()` API on the raw ledger, and supersede any semantic facts derived from that event.

---

## 5. Long Inputs & Exact Context

For legal, compliance, or financial applications, vector similarity is not enough. You must be able to cite the exact source document.

Do not dump 50-page documents into standard `add()` calls. Use `record_long_input()` to securely chunk the document, generate precise quote hashes, and create bounded artifact events. 

> [!TIP]
> When building enterprise apps, force your LLM's system prompt to cite the `chunk_id` or `quote_hash` returned by `build_long_input_context()`. This guarantees hallucination-free citations.

---

## 6. Observability & Failure Modes

When running Engram in production, monitor the following signals:

| Signal | Metric / Query | Mitigation |
|--------|----------------|------------|
| **Worker Saturation** | Count of `pending` rows in `memory_jobs` | If the backlog grows indefinitely, scale up your worker instances or switch to a faster extraction LLM model. |
| **Extraction Failures** | Count of `failed` rows in `memory_jobs` | Typically caused by LLM rate limits. Engram will backoff and retry, but sustained failures require requesting higher quotas from OpenAI/Anthropic. |
| **Missed Retrieval** | Monitor the `missing_expected_terms` list in the `RecallTrace` object. | Tune your `MemoryPolicy` to pin facts to `critical_slots` so they bypass vector math entirely. |
| **Vector Index Wipes** | Application crash on boot with `ConfigurationError`. | You changed the embedding model dimension on an existing DB. You must export `ENGRAM_ALLOW_EMBEDDING_DIMENSION_CHANGE=true` and manually trigger a re-embedding script. |

---

## 7. FastAPI Integration Sketch

This is the recommended way to manage the Engram connection pool inside a modern async Python web framework.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from engram import Engram

# Global client
engram: Engram | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engram
    # Initialize the client and DB connection pool on startup
    engram = Engram(memory_policy="default")
    await engram.connect()
    try:
        yield
    finally:
        # Drain connections cleanly on shutdown
        await engram.close()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(body: dict, current_user: User = Depends(get_user)):
    assert engram is not None
    
    # Intelligently route and fetch context
    trace = await engram.recall(
        query=body["message"], 
        user_id=current_user.id
    )
    
    # ... Stream LLM, record_turn, etc ...
```
