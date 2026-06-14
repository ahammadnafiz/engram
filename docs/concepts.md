# Core Concepts

This page explains the memory model behind the current Engram API.

## Memory Is Layered

Engram does not treat memory as one table of text. It separates durable state by
what the agent needs to do with it.

| Layer | Stored in | Used for |
|-------|-----------|----------|
| Fact memory | `agent_memory` | search, deterministic recall, conflict resolution |
| Graph memory | `memory_relations` | related decisions, constraints, facts, and tool results |
| Session memory | `agent_sessions` | conversation grouping and rolling summaries |
| Task memory | `agent_task_runs`, `agent_events`, `agent_checkpoints`, `memory_jobs` | resumable long-running work |

Small chatbots can start with fact memory. Coding agents, legal review agents,
research agents, and assistants that run for days should use task memory and
fact memory together.

## Two-Column Fact Memory

A memory stores a concise fact and optional source context.

```python
memory = await engram.add(
    "User is allergic to shellfish",
    "assistant",
    user_id="user_123",
    main_content="[USER]: Shellfish makes me sick.\n[AI]: I will avoid it.",
)
```

| Field | Embedded | Role |
|-------|----------|------|
| `content` / `fact` | yes | concise fact for vector and keyword search |
| `main_content` | no | source conversation, quote, document chunk, or other support |

This keeps embedding cost low while preserving the context that produced the
fact.

## Memory Types

Every memory has a `memory_type`.

| Type | Example |
|------|---------|
| `semantic` | generic durable fact |
| `episodic` | dated event or narrative |
| `procedural` | rule or process |
| `profile` | name, city, allergy, manager |
| `project` | codename, owner, launch date, target metric |
| `task` | requirement, pending work, completed work |
| `preference` | communication style, UI preference |
| `constraint` | hard rule, deadline, safety limit |
| `decision` | correction, approval, chosen approach |
| `tool_result` | pytest output, load test result, API response |

Filter by type when a prompt has a narrow purpose.

```python
constraints = await engram.search(
    "repo rules",
    "codex",
    user_id="nafiz",
    memory_types=["constraint", "decision"],
)
```

## Memory Policies

`MemoryPolicy` runs before storage. It can:

- infer a more specific `memory_type`
- mark a memory as critical
- assign a deterministic `critical_slot`
- derive a scoped `conflict_key`

```python
async with Engram(memory_policy="coding_agent") as engram:
    memory = await engram.add(
        "Repo constraint: never revert user changes without approval",
        "codex",
        user_id="nafiz",
    )

    print(memory.memory_type)
    print(memory.metadata["critical_slot"])
```

Built-in policies:

| Policy | Use |
|--------|-----|
| `default` | personal assistants and general agents |
| `legal` | legal or exact-document review |
| `coding_agent` | repository constraints, implementation decisions, tool output |

Custom policies use `TypeRule` and `SlotRule`.

```python
from engram import MemoryPolicy, SlotRule, TypeRule

support_policy = MemoryPolicy(
    name="support",
    type_rules=(
        TypeRule("profile", (r"\baccount id\b", r"\bplan\b")),
        TypeRule("constraint", (r"\bSLA\b", r"\bescalate\b")),
    ),
    slot_rules=(
        SlotRule("support:account_plan", (r"\bplan\b",), ("profile",)),
        SlotRule("support:sla", (r"\bSLA\b",), ("constraint",)),
    ),
)

async with Engram(memory_policy=support_policy) as engram:
    await engram.add("Account plan is Enterprise", "support-agent")
```

## Critical Memory

Critical facts should not depend only on vector similarity. Policy metadata lets
Engram recall them directly.

```json
{
  "critical": true,
  "critical_slot": "coding:repo_constraint",
  "conflict_key": "codex:nafiz:coding:repo_constraint",
  "status": "active",
  "version": 1
}
```

Use `recall_critical()` when you need the pinned facts directly.

```python
critical = await engram.recall_critical(
    "codex",
    user_id="nafiz",
    memory_types=["constraint", "preference", "decision"],
)
```

`trace_recall()` puts critical memories ahead of search-ranked memories in the
prompt budget.

## Conflict Resolution

When a new memory has the same `conflict_key` as an older active memory, Engram
marks the older row as superseded instead of deleting it.

```json
{
  "status": "superseded",
  "superseded_by": "mem_new",
  "superseded_at": "2026-06-14T..."
}
```

Normal search hides superseded rows. Trace APIs still report superseded IDs so
you can debug corrections.

## Search

`search()` supports three modes:

| Mode | Behavior |
|------|----------|
| `hybrid` | vector search + full-text search + decay + importance |
| `semantic` | vector similarity |
| `keyword` | PostgreSQL full-text search |

Search also supports `metadata_filter`, `memory_types`, `min_score`, and
optional local reranking.

```python
results = await engram.search(
    "black friday latency target",
    "codex",
    user_id="nafiz",
    metadata_filter={"project": "atlas_checkout"},
    memory_types=["project", "constraint", "tool_result"],
    mode="hybrid",
)
```

Use `deep_search()` for broad prompts. It asks the configured LLM for query
variants and fuses the result ranks. Without an LLM provider, it behaves like
one `search()` call.

## Recall Observability

`trace_recall()` answers the question "why did this memory appear or disappear
from the prompt?"

```python
trace = await engram.trace_recall(
    "Can we launch today?",
    "assistant",
    user_id="user_123",
    expected_terms=["rollback owner", "error rate"],
    max_tokens=1000,
)
```

| Field | Debug question |
|-------|----------------|
| `critical_memory_ids` | Was it pinned as critical? |
| `search_memory_ids` | Did search retrieve it? |
| `ranked_memory_ids` | Was it eligible after dedupe? |
| `kept_memory_ids` | Did it fit in the final context? |
| `trimmed_memory_ids` | Was it cut by token budget? |
| `superseded_memory_ids` | Was an old fact hidden by conflict rules? |
| `missing_expected_terms` | Did caller-provided required terms fail to appear? |

## Graph Relations

Graph relations connect memories, not tasks. Use them when one retrieved fact
should bring along related decisions, constraints, or tool results.

```python
requirement = await engram.add(
    "Task requirement: checkout p95 must stay under 250 ms",
    "codex",
    user_id="nafiz",
)
decision = await engram.add(
    "Decision: use cached inventory reads during launch week",
    "codex",
    user_id="nafiz",
)

await engram.relate(
    source_id=requirement.memory_id,
    target_id=decision.memory_id,
    relation_type="supports",
    weight=0.8,
)

graph = await engram.traverse_many(
    [requirement.memory_id, decision.memory_id],
    max_depth=2,
    direction="any",
)

block = engram.render_graph_context(graph, max_tokens=800)
```

## Task Memory

Task memory is for work that spans turns or process restarts.

| Model | Meaning |
|-------|---------|
| `TaskRun` | goal, status, owner, outcome |
| `AgentEvent` | raw ledger event |
| `TaskCheckpoint` | compact state snapshot |
| `MemoryJob` | background derivation work |

Typical loop:

```python
task = await engram.start_task("Refactor the memory framework", "codex")
context = await engram.build_context(task.task_run_id, query="resume")
response = await call_llm(context.text)
await engram.record_turn(task.task_run_id, user_message, response)
await engram.process_memory_jobs(limit=10)
```

Run `run_memory_worker()` in a worker process for production.

## Long Input

Long prompts and documents should be chunked and source-anchored.

```python
report = await engram.record_long_input(
    task.task_run_id,
    text=large_prompt,
    title="Legal review packet",
)

context = await engram.build_long_input_context(
    task.task_run_id,
    query="termination obligations",
    expected_terms=["termination", "notice"],
)
```

Each chunk records a chunk ID, kind, heading, character span, quote hash, source
event ID, and metadata. Use source chunks for legal, financial, compliance, or
exact-document answers. Use distilled memories for speed and continuity.

## Which API To Use

| Need | API |
|------|-----|
| Store one fact | `add()` |
| Store many facts | `add_batch()` |
| Extract facts from a conversation | `add_conversation()` |
| Search relevant facts | `search()` |
| Search broad prompts | `deep_search()` |
| Build debuggable prompt memory | `trace_recall()` |
| Build compact prompt memory | `get_context_block()` |
| Start resumable work | `start_task()` |
| Record raw turns or tools | `record_turn()` / `record_event()` |
| Build task-resume context | `build_context()` |
| Ingest a huge prompt or document | `record_long_input()` |
| Build source-anchored answer context | `build_long_input_context()` |
| Retrieve diverse evidence | `search_evidence_set()` |
| Add neighboring turn context | `get_neighboring_context_block()` |
