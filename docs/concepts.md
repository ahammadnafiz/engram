# Core concepts

This page covers the memory model behind the Engram API: how memories are
stored, typed, pinned, searched, and traced.

## Memory is layered

Engram doesn't dump everything into one table of text. It splits durable state
by what the agent actually needs to do with it.

| Layer | Stored in | Used for |
|-------|-----------|----------|
| Fact memory | `agent_memory` | search, deterministic recall, conflict resolution |
| Graph memory | `memory_relations` | linking related decisions, constraints, facts, and tool results |
| Session memory | `agent_sessions` | grouping a conversation and its rolling summary |
| Task memory | `agent_task_runs`, `agent_events`, `agent_checkpoints`, `memory_jobs` | resumable long-running work |

A small chatbot can live entirely in fact memory. Agents that run for hours or
days, such as coding, legal-review, or research agents, will want task memory
and fact memory together.

## Facts have two columns

A memory holds a short fact and, optionally, the context it came from.

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
| `fact` | yes | the concise fact, used for vector and keyword search |
| `content` | yes | backward-compatible alias of `fact` |
| `main_content` | no | the source conversation, quote, or document chunk |

Only the fact gets embedded, so you keep embedding costs down without throwing
away the context that produced it. Write new code against `fact` and
`main_content`. `content` is kept so older `engram.add(content=...)` calls and
existing rows keep working.

## Memory types

Every memory carries a `memory_type`.

| Type | Example |
|------|---------|
| `semantic` | a generic durable fact |
| `episodic` | a dated event or narrative |
| `procedural` | a rule or process |
| `profile` | name, city, allergy, manager |
| `project` | codename, owner, launch date, target metric |
| `task` | a requirement, pending work, or completed work |
| `preference` | communication style, UI preference |
| `constraint` | a hard rule, deadline, or safety limit |
| `decision` | a correction, approval, or chosen approach |
| `tool_result` | pytest output, a load-test result, an API response |

Filter by type when a prompt only needs one slice of memory.

```python
constraints = await engram.search(
    "repo rules",
    "codex",
    user_id="nafiz",
    memory_types=["constraint", "decision"],
)
```

## Policies

A `MemoryPolicy` runs just before a memory is stored. It can infer a more
specific `memory_type`, flag the memory as critical, give it a stable
`critical_slot`, and derive a scoped `conflict_key`.

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

Three policies ship with Engram:

| Policy | Use it for |
|--------|------------|
| `default` | personal assistants and general agents |
| `legal` | legal or exact-document review |
| `coding_agent` | repo constraints, implementation decisions, tool output |

The default ships with generic rules only: durable personal facts (name, city,
manager, communication style) plus allergy handling. Domain vocabulary belongs
in `legal`, `coding_agent`, or a policy you write yourself with `TypeRule` and
`SlotRule`.

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

## Critical memory

Some facts are too important to leave to vector similarity. Policy metadata lets
Engram recall them directly, no query required.

```json
{
  "critical": true,
  "critical_slot": "coding:repo_constraint",
  "conflict_key": "codex:nafiz:coding:repo_constraint",
  "status": "active",
  "version": 1
}
```

Call `recall_critical()` when you want those pinned facts on their own:

```python
critical = await engram.recall_critical(
    "codex",
    user_id="nafiz",
    memory_types=["constraint", "preference", "decision"],
)
```

`trace_recall()` already does this for you: it places critical memories ahead of
the search-ranked ones when it fills the prompt budget.

## Conflict resolution

When a new memory shares a `conflict_key` with an older active one, Engram marks
the old row superseded rather than deleting it.

```json
{
  "status": "superseded",
  "superseded_by": "mem_new",
  "superseded_at": "2026-06-14T..."
}
```

Regular search hides superseded rows. The trace APIs still report their IDs, so
when a correction goes wrong you can see exactly what got hidden and why.

## Search

`search()` runs in one of three modes.

| Mode | What it does |
|------|--------------|
| `hybrid` | vector search, full-text search, recency decay, and importance, fused together |
| `semantic` | vector similarity only |
| `keyword` | PostgreSQL full-text search only |

It also takes `metadata_filter`, `memory_types`, `min_score`, and optional local
reranking.

```python
results = await engram.search(
    "API latency budget for launch",
    "codex",
    user_id="nafiz",
    metadata_filter={"project": "payments"},
    memory_types=["project", "constraint", "tool_result"],
    mode="hybrid",
)
```

For broad or open-ended prompts, use `deep_search()`. It asks the configured LLM
for a few query variants, runs them, and fuses the rankings. With no LLM
provider it just falls back to a single `search()`.

## Recall observability

`trace_recall()` answers one question: why did this memory show up in the prompt,
or why didn't it?

```python
trace = await engram.trace_recall(
    "Can we launch today?",
    "assistant",
    user_id="user_123",
    expected_terms=["rollback owner", "error rate"],
    max_tokens=1000,
)
```

| Field | Tells you |
|-------|-----------|
| `critical_memory_ids` | was it pinned as critical? |
| `search_memory_ids` | did search retrieve it? |
| `ranked_memory_ids` | did it survive dedupe? |
| `kept_memory_ids` | did it fit the final context? |
| `trimmed_memory_ids` | did the token budget cut it? |
| `superseded_memory_ids` | did conflict resolution hide an older value? |
| `missing_expected_terms` | did a term you required fail to appear? |

## Graph relations

Relations connect memories to each other. Reach for them when one retrieved fact
should drag along the decisions, constraints, or results tied to it.

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

## Task memory

Task memory holds work that spans turns or survives a process restart.

| Model | What it is |
|-------|------------|
| `TaskRun` | the goal, status, owner, and outcome |
| `AgentEvent` | a raw ledger event |
| `TaskCheckpoint` | a compact state snapshot |
| `MemoryJob` | a background derivation job |

The usual loop:

```python
task = await engram.start_task("Refactor the memory framework", "codex")
context = await engram.build_context(task.task_run_id, query="resume")
response = await call_llm(context.text)
await engram.record_turn(task.task_run_id, user_message, response)
await engram.process_memory_jobs(limit=10)
```

In production, run `run_memory_worker()` in its own process instead of
processing jobs inline.

## Long input

Long prompts and documents get chunked and anchored to their source.

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

Each chunk records a chunk ID, kind, heading, character span, quote hash, and
source event ID. Use the source chunks when an answer has to be exact (legal,
financial, compliance). Use the distilled memories when you care more about speed
and continuity.

## Which API to use

| You want to | Use |
|-------------|-----|
| Store one fact | `add()` |
| Store many facts | `add_batch()` |
| Pull facts out of a conversation | `add_conversation()` |
| Search relevant facts | `search()` |
| Search broad or vague prompts | `deep_search()` |
| Build prompt memory you can debug | `trace_recall()` |
| Build a compact prompt block | `get_context_block()` |
| Start resumable work | `start_task()` |
| Record raw turns or tool calls | `record_turn()` / `record_event()` |
| Build resume context for a task | `build_context()` |
| Ingest a huge prompt or document | `record_long_input()` |
| Build source-anchored answer context | `build_long_input_context()` |
| Pull a session or group for neighbor context | `get_memories()` |
