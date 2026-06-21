<p align="center">
  <img src="assets/engram-banner.png" alt="Engram - AI Memory Layer" width="600">
</p>

<p align="center">
  <strong>Persistent memory infrastructure for long-running AI agents</strong>
</p>

<p align="center">
  <a href="#benchmark-results">Benchmarks</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#examples">Examples</a>
</p>

---

## Benchmark results

Engram is evaluated on three long-term memory benchmarks. All runs use on-device embeddings (`all-MiniLM-L6-v2`, 384-d, free) with **no LLM calls at ingestion** — raw episodic turns stored verbatim via `add_batch()`, all reasoning deferred to query time via hybrid search + cross-encoder rerank + `recall()`.

**Honest caveats**: composer and judge are the same model family (`claude-sonnet-4-6`) across all three runs, which is a known leniency bias. `add_batch()` bypasses Engram's LLM-based extraction pipeline — these are floor numbers. `add_conversation()` is expected to score higher on structured fact types.

![Engram benchmark results](assets/engram-benchmark.svg)

| Benchmark | Questions | Accuracy | Composer |
|---|---|---|---|
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) | 500 | **89.8%** | claude-sonnet-4-6 |
| [LoCoMo-10](https://github.com/snap-research/locomo) (ACL 2024) | 1,540 | **85.7%** | claude-sonnet-4-6 |
| [BEAM 1M](https://github.com/mohammadtavakoli78/BEAM) (ICLR 2026) | 700 | **79.6%** | claude-sonnet-4-6 |

**LongMemEval-S breakdown** (500 questions):

| Question type | Accuracy |
|---|---|
| single-session-user | 98.6% |
| knowledge-update | 97.4% |
| abstention | 96.7% |
| single-session-assistant | 94.6% |
| temporal-reasoning | 89.5% |
| single-session-preference | 83.3% |
| multi-session | 80.5% |

**LoCoMo-10 breakdown** (1,540 questions across 10 long conversations):

| Category | Accuracy |
|---|---|
| multi-hop | 87.9% |
| temporal | 86.9% |
| single-hop | 86.6% |
| open-domain | 67.7% |

**BEAM 1M breakdown** (700 questions, 10 question types, nugget scoring):

| Question type | Accuracy |
|---|---|
| abstention | 97.1% |
| preference_following | 92.9% |
| instruction_following | 90.0% |
| information_extraction | 84.3% |
| event_ordering | 81.4% |
| knowledge_update | 81.4% |
| multi_session_reasoning | 81.4% |
| contradiction_resolution | 80.0% |
| temporal_reasoning | 65.7% |
| summarization | 41.4% |

Summarization (41.4%) is a known architectural floor: relevance-ranked retrieval is precision-optimized, not coverage-maximizing. Full methodology, ablation table, and reproduce commands: [docs/benchmarks.md](docs/benchmarks.md).

---

Engram is a beta memory management library for AI agents and LLM
applications. It provides persistent, searchable memory using PostgreSQL and
pgvector, with hybrid retrieval, durable task ledgers, typed critical memory,
source-anchored long-input ingestion, and traceable recall.

> Status: **beta**. The architecture and local test suite are in good shape,
> but public APIs may still change before a stable release.

<p align="center">
  <img src="assets/chatbot_image.png" alt="Engram Memory Chatbot — your second brain in the terminal" width="720">
  <br>
  <em>See it working — the Engram-backed memory chatbot running as your second brain in the terminal.</em>
</p>

## The Problem: Why Conversational Memory Fails

Long-term memory for LLMs is fundamentally a database engineering problem, not just a context window problem. Most memory systems fail due to a few common patterns:

1. **Derivation Drift:** Summarizing old summaries eventually degrades the information (like a photocopy of a photocopy).
2. **Stale Context Dominance:** Old, frequently discussed facts crowd out recent, updated truths.
3. **Compaction Information Loss:** Storing only summaries loses the exact wording of what was said; storing only raw transcripts leaves the agent without any high-level understanding.

Engram was built to structurally mitigate these issues, offering perfect preservation and high-fidelity interpretation simultaneously.

## Features

- **Hybrid Search** - Combines vector similarity, keyword matching (BM25), time decay, and importance scoring using Reciprocal Rank Fusion
- **Two-Column Memory System** - Embed only facts for search, store full conversation context separately (cost-effective)
- **Memory Decay** - Exponential decay prioritizes recent and frequently accessed memories
- **Graph Traversal** - Multi-hop reasoning through typed memory relationships using recursive CTEs
- **Long-Running Task Memory** - Durable task runs, raw event ledger, checkpoints, and background memory jobs for agents that run across many turns
- **Session Management** - Track conversation context with session IDs and rolling summaries
- **Pluggable Providers** - Support for OpenAI, Anthropic, Cohere, Ollama, Sentence Transformers, and custom providers
- **Configurable Memory Policies** - Domain-specific typing and conflict slots for personal, coding, legal, and custom agents
- **Long Input Ingestion** - Store full prompts/documents, chunk by structure, extract anchored memories, and build source-grounded context
- **Recall Observability** - Trace whether memories were stored, ranked, kept, trimmed, superseded, or missing
- **Operational Foundation** - ACID storage, connection pooling, and guarded vector-dimension alignment

## Installation

```bash
pip install engram
```

With specific providers:

```bash
pip install engram[openai]                # OpenAI embeddings and LLM
pip install engram[anthropic]             # Anthropic Claude LLM
pip install engram[cohere]                # Cohere embeddings
pip install engram[http]                  # Ollama/Hugging Face/Groq HTTP providers
pip install engram[litellm]               # LiteLLM universal LLM provider
pip install engram[sentence-transformers] # Local embeddings (free)
pip install engram[examples]              # Example helper dependencies
pip install engram[all]                   # All providers
```

## Quick Start

### 1. Start the Database

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
pip install -e ".[dev,examples,sentence-transformers]"
docker compose up -d postgres
```

### 2. Configure Environment

```bash
# .env
ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Use Engram

```python
import asyncio
from engram import Engram

async def main():
    async with Engram() as engram:
        # Store a memory
        memory = await engram.add(
            content="User prefers dark mode",
            agent_id="assistant",
            user_id="user_123",
        )

        # Store with conversation context (two-column system)
        memory = await engram.add(
            content="User is learning Python",           # Fact (embedded)
            main_content="[USER]: I'm learning Python\n[AI]: Great choice!",  # Context (not embedded)
            agent_id="assistant",
            user_id="user_123",
        )

        # Search memories
        results = await engram.search(
            query="user preferences",
            agent_id="assistant",
            user_id="user_123",
        )

        for r in results:
            print(f"[{r.score:.2f}] {r.memory.content}")

        # Long-running agent task memory
        task = await engram.start_task(
            "Help the user plan a Python learning path",
            agent_id="assistant",
            user_id="user_123",
        )

        # Store the raw turn in the event ledger and queue memory derivation
        await engram.record_turn(
            task.task_run_id,
            user_message="I want to learn Python for data analysis.",
            assistant_response="Let's focus on pandas, notebooks, and small projects.",
        )

        # Process queued work: facts + checkpoints
        await engram.process_memory_jobs(limit=10)

        # Build prompt-ready context for the next agent turn
        context = await engram.build_context(
            task.task_run_id,
            query="Python data analysis learning plan",
            max_tokens=2000,
        )
        print(context.text)

asyncio.run(main())
```

## Long Input and Legal/Source Documents

For large prompts, legal documents, or multi-day sessions, store the exact
source and derive anchored memories from chunks instead of relying only on a
summary.

```python
from engram import Engram

async with Engram(memory_policy="legal") as engram:
    task = await engram.start_task(
        "Review vendor contract",
        agent_id="legal-agent",
        user_id="user_123",
    )

    report = await engram.record_long_input(
        task.task_run_id,
        contract_text,
        title="Vendor SaaS review",
        max_chunk_tokens=500,
    )

    context = await engram.build_long_input_context(
        task.task_run_id,
        query="audit logs liability indemnity risk table",
        expected_terms=["audit logs", "liability", "indemnify"],
    )

    print(context.text)
    print(context.trace["missing_expected_terms"])
```

`record_long_input()` keeps the raw input event, records source chunks, stores
chunk metadata (`source_event_id`, `chunk_id`, `section`, character range, and
`quote_hash`), normalizes common relative dates, and creates a manifest
checkpoint.

## Memory Policies

Policies control memory typing, deterministic critical recall, and conflict
slots. Built-in presets are:

| Policy | Use case |
|--------|----------|
| `default` | General personal/product/task agents |
| `legal` | Source-grounded legal or contract review |
| `coding_agent` | Repo-aware coding agents and tool-result memory |

```python
from engram import Engram, MemoryPolicy, SlotRule, TypeRule

sales_policy = MemoryPolicy(
    name="sales",
    type_rules=(TypeRule("project", (r"\baccount\b",)),),
    slot_rules=(SlotRule("sales:account_owner", (r"\baccount owner\b",)),),
)

async with Engram(memory_policy=sales_policy) as engram:
    await engram.add("The account owner is Rina.", "sales-agent")
```

## Architecture

Engram handles messy, unstructured streams of information and distills them into a pristine 2nd Brain. It uses a converged architecture where all operations run in PostgreSQL:

```
┌─────────────────────────────────────────────────────────────────────┐
│                            ENGRAM                                   │
├─────────────────────────────────────────────────────────────────────┤
│   Embedding Service          │          LLM Service                 │
│   (OpenAI, Local, etc.)      │          (OpenAI, Anthropic, etc.)   │
├──────────────────────────────┴──────────────────────────────────────┤
│                       PostgreSQL + pgvector                         │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│   │  Vectors   │  │  Full-text │  │   Graph    │  │   JSONB    │    │
│   │   (HNSW)   │  │   (GIN)    │  │ Relations  │  │  Metadata  │    │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Episodic vs. Semantic Separation

Engram refuses to pick between raw transcripts and derived summaries. It separates the two to prevent **Compaction Loss**:

| Column | Embedded | Purpose |
|--------|----------|---------|
| `fact` | Yes | Canonical concise facts for semantic and keyword search |
| `content` | Yes | Backward-compatible alias of `fact` in the API |
| `main_content` | No | Full conversation context (cost-effective storage) |

### Lineages & `supersedes`

When a user updates a fact, Engram does *not* destructively overwrite the old fact. Instead, it inserts the new fact, changes the old fact's status to `superseded`, and draws a physical `supersedes` graph edge between them.
* Active searches only retrieve the active head, eliminating **Stale Context Dominance**.
* The old fact is perfectly preserved for auditing or historical lineage queries (`get_lineage()`), eliminating **Derivation Drift**.

### Long-Running Task Memory

For agents that need to work across many turns or process restarts, Engram
stores task memory separately from extracted facts:

| Record | Purpose |
|--------|---------|
| `agent_task_runs` | Durable unit of work with status, goal, outcome, and metadata |
| `agent_events` | Raw append-only ledger of user, assistant, agent, tool, artifact, and error events |
| `agent_checkpoints` | Compact summaries of current state, decisions, blockers, and artifacts |
| `memory_jobs` | Durable background queue for deriving facts/checkpoints from raw turns |

```python
task = await engram.start_task("Investigate a production incident", "sre-agent")

await engram.record_turn(
    task.task_run_id,
    "Latency is high in checkout.",
    "I will inspect recent deploys and database metrics.",
    tool_calls=[{"name": "query_metrics", "service": "checkout"}],
)

await engram.process_memory_jobs()

context = await engram.build_context(task.task_run_id, max_tokens=200000)
```

### Critical Recall and Traceability

For broad or multi-part prompts, use `trace_recall()` to inspect what was
included or missed:

```python
trace = await engram.trace_recall(
    "final verification: city allergy codename rollback owner",
    "assistant",
    user_id="user_123",
    expected_terms=["Denver", "cashews", "quartz-falcon", "Priya"],
)

print(trace.context)
print(trace.missing_expected_terms)
print(trace.trimmed_memory_ids)
print(trace.superseded_memory_ids)
```

### Hybrid Search

Search combines multiple signals using Reciprocal Rank Fusion:

```
score = 0.40 × semantic_similarity
      + 0.20 × keyword_score
      + 0.25 × time_decay
      + 0.15 × importance
```

## Provider Support

### Embedding Providers

| Provider | Installation | Example Models |
|----------|--------------|----------------|
| `openai` | `pip install openai` | `text-embedding-3-small`, `text-embedding-3-large` |
| `sentence-transformers` | `pip install sentence-transformers` | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` |
| `cohere` | `pip install cohere` | `embed-english-v3.0`, `embed-multilingual-v3.0` |
| `ollama` | Ollama server | `nomic-embed-text`, `mxbai-embed-large` |
| `huggingface` | `pip install httpx` | Any model via Inference API |

### LLM Providers

| Provider | Installation | Example Models |
|----------|--------------|----------------|
| `openai` | `pip install openai` | `gpt-4o-mini`, `gpt-4o` |
| `anthropic` | `pip install anthropic` | `claude-3-haiku`, `claude-3-sonnet` |
| `ollama` | Ollama server | `llama3.2`, `mistral` |
| `groq` | `pip install httpx` | `llama-3.1-8b-instant`, `mixtral-8x7b` |
| `litellm` | `pip install litellm` | 100+ models via unified API |

### Custom Providers

```python
from engram import embedding_registry, EmbeddingProvider

@embedding_registry.register("custom")
class CustomEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "default"):
        self._model = model
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        # Implementation
        return [0.0] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]
```

## Examples

### Memory Chatbot

```bash
python examples/chatbot.py
```

The chatbot runs the benchmark pipeline live: each turn is stored verbatim via
`add_batch()` (on-device embeddings, no LLM at ingest), retrieved over 4 surfaces
(hybrid search + rerank, `recall()`, `get_lineage()`, graph traversal), and
answered by a single composer call. Reranking defaults to
`ENGRAM_CHATBOT_RERANK=auto`; set `ENGRAM_CHATBOT_RERANK=true` to force the
cross-encoder or `false` to disable it. It does **not** use `add_conversation()`
at ingest — see [Why not `add_conversation()`](docs/examples.md) for the reason.

### Programmatic Usage

```python
from engram import Engram

async def example():
    async with Engram() as engram:
        # Add memories
        m1 = await engram.add(
            content="User's name is Alice",
            agent_id="bot",
            main_content="[USER]: I'm Alice\n[AI]: Nice to meet you!",
        )
        m2 = await engram.add(
            content="Alice works in finance",
            agent_id="bot",
        )

        # Create graph relations
        await engram.relate(m1.memory_id, m2.memory_id, "related_to", weight=0.8)

        # Traverse the graph
        related = await engram.traverse(
            start_memory_id=m1.memory_id,
            max_depth=2,
            direction="outbound",
        )

        # Multi-seed graph expansion for prompt assembly
        graph = await engram.traverse_many(
            [m1.memory_id, m2.memory_id],
            max_depth=2,
            direction="any",
        )
        graph_context = engram.render_graph_context(graph, max_tokens=500)
        print(graph_context)

        # Reinforce important memories
        await engram.reinforce(m1.memory_id, importance_boost=0.2)

        # Search with hybrid ranking
        results = await engram.search("Alice", agent_id="bot", limit=5)
        for r in results:
            print(f"Fact: {r.memory.fact}")
            print(f"Context: {r.memory.main_content}")
```

## Configuration

All settings are configured via environment variables with the `ENGRAM_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_DATABASE_URL` | `postgresql://localhost:5432/engram` | PostgreSQL connection string |
| `ENGRAM_EMBEDDING_PROVIDER` | `openai` | Embedding provider |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `ENGRAM_LLM_PROVIDER` | - | LLM provider (optional) |
| `ENGRAM_LLM_MODEL` | `gpt-4o-mini` | LLM model |
| `ENGRAM_WEIGHT_SEMANTIC` | `0.40` | Semantic search weight |
| `ENGRAM_WEIGHT_KEYWORD` | `0.20` | Keyword search weight |
| `ENGRAM_WEIGHT_DECAY` | `0.25` | Time decay weight |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.15` | Importance weight |
| `ENGRAM_DECAY_RATE` | `0.995` | Decay rate per hour |

## Docker Commands

```bash
# Start database
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Reset (delete all data)
docker compose down -v
```

## Development

A `Makefile` wraps the common workflows. Run `make help` to list every target.

```bash
make quickstart   # start the database and install dev dependencies
make dev          # install with dev + all extras and pre-commit hooks
make test         # run unit tests
make lint         # run the Ruff linter
make format       # apply Ruff fixes and formatting
make type-check   # run mypy
make docs-serve   # serve the docs locally with live reload
```

See the [Operations guide](./docs/operations.md#make-targets) for the full target list.

## Documentation

The docs in `docs/` are the source for the MkDocs site.

- [Quickstart](./docs/quickstart.md) - install, configure, and run the first flow
- [Core Concepts](./docs/concepts.md) - memory types, policies, search, graph, and task memory
- [Task Memory](./docs/task-memory.md) - tasks, events, checkpoints, workers, long input, and evidence APIs
- [API Reference](./docs/api-reference.md) - current public signatures and tested examples
- [Configuration](./docs/configuration.md) - environment variables, provider extras, and safety flags
- [Production Guide](./docs/production-guide.md) - deployment shape, observability, and privacy boundaries
- [Operations](./docs/operations.md) - Docker, psql, migrations, workers, and verification commands
- [Examples](./docs/examples.md) - included scripts and what they demonstrate

Additional project docs:

- [CHANGELOG.md](./CHANGELOG.md) - release notes
- [SECURITY.md](./SECURITY.md) - privacy, deletion, and API-key guidance

## Production Caveats

Engram stores user facts, source documents, and agent events. Treat it as
sensitive infrastructure:

- Scope memories by `agent_id` and `user_id`.
- Use `forget()`, `purge()`, `redact_event()`, and task deletion workflows for
  retention/deletion requirements.
- For legal or exact-document answers, retrieve source chunks first and memory
  summaries second.
- Review `trace_recall()` output for broad prompts and production incidents.
- Start public deployments with explicit retention, tenant isolation, backup,
  and access-control policies.

## Requirements

- Python 3.10+
- PostgreSQL 16+ with pgvector
- Docker and Docker Compose (for database)

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for the AI community</sub>
</p>
