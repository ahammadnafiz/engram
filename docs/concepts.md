# Core Concepts

This guide covers the memory model behind Engram. It explains how memories are structured, categorized, stored, and intelligently recalled by your autonomous agents.

> [!NOTE]
> Engram is not just a vector database wrapper. It is a fully featured cognitive architecture that handles memory conflicts, graph relations, immutable task ledgers, and intelligent intent classification.

---

## 1. The Memory Layers

Engram doesn't dump everything into a single text column. It separates durable state logically based on what the agent needs to do with it.

| Layer | Storage Table | Purpose |
|-------|---------------|---------|
| **Fact Memory** | `agent_memory` | Storing atomic facts for vector search, deterministic recall, and conflict resolution. |
| **Graph Memory** | `memory_relations` | Linking related decisions, constraints, facts, and tool results into an associative network. |
| **Session Memory** | `agent_sessions` | Grouping a conversational boundary and maintaining a rolling summary. |
| **Task Memory** | `agent_events`, `agent_checkpoints` | Maintaining an immutable ledger of agent actions for resumable, long-running work. |

A simple chatbot might only ever use Fact and Session memory. Advanced agents (like coding or research assistants) will heavily utilize Task and Graph memory.

---

## 2. Anatomy of a Memory

To optimize retrieval costs and preserve context, Engram splits knowledge into two distinct components: the **Fact** and the **Context**.

| Field | Embedded? | Description |
|-------|-----------|-------------|
| `fact` | **Yes** | The concise, distilled fact. This is embedded for vector search and indexed for keyword search. *(Note: `content` is a backward-compatible alias for this field).* |
| `main_content` | **No** | The raw conversational context, quote, or document chunk that produced the fact. |

By only embedding the `fact`, you keep embedding costs minimal without permanently losing the conversational context (`main_content`) that led to the decision.

---

## 3. Memory Types and Policies

Every memory carries a `memory_type` (e.g., `semantic`, `episodic`, `preference`, `constraint`, `decision`). This allows you to aggressively filter prompt contexts—for example, fetching *only* constraints and decisions before executing code.

### Memory Policies
A `MemoryPolicy` runs dynamically just before a memory is stored. It evaluates the fact and can:
1. Infer a more specific `memory_type`.
2. Assign a `critical_slot` (pinning the fact).
3. Assign a `conflict_key` to manage updates to the same conceptual fact.

> [!TIP]
> Engram ships with three built-in policies: `"default"`, `"legal"`, and `"coding_agent"`. You can easily create your own domain-specific policies using `TypeRule` and `SlotRule`.

---

## 4. Intelligent Recall & Intent (The Cognitive Operator)

Rather than forcing the developer to manually parse user prompts and decide which search mode to use, Engram provides the high-level `recall()` operator. 

When you ask Engram a question via `recall()`, it uses the LLM to classify the user's **Intent**:
- `current`: "What is my allergy?" (Retrieves the active head of the fact).
- `historical`: "What was my meeting before I changed it?" (Retrieves superseded lineage history).
- `event`: "When did you run the tests?" (Searches the immutable task ledger).
- `lineage`: "How has my project target changed over time?"

It also automatically maps **Temporal Phrases** (e.g., "last week", "yesterday") into strict database time windows before executing the hybrid search.

---

## 5. Conflict Resolution & Lineages

Agents frequently learn updated information that invalidates old facts (e.g., a user moves to a new city). 

When a new memory is added that shares a `conflict_key` with an existing active memory, Engram does **not** delete the old memory. Instead, it:
1. Links them via a `lineage_id`.
2. Marks the older row's status as `superseded`.
3. Marks the new row as the active `current` head.

Standard hybrid searches automatically filter out `superseded` memories, ensuring your prompt isn't polluted with outdated facts. However, because the data isn't deleted, you can still trace the timeline of a changing fact using `get_lineage()` or `get_history()`.

---

## 6. Search Modalities

When bypassing the high-level `recall()` operator, you can query facts directly using three distinct modes:

| Mode | Methodology |
|------|-------------|
| `hybrid` | Fuses pgvector similarity, Postgres full-text search, recency decay, and importance scores. |
| `semantic` | Pure pgvector cosine similarity. |
| `keyword` | Pure Postgres full-text search. |

### Deep Search
For broad or vague prompts, use `deep_search()`. This asks the LLM to generate multiple permutations of the query, executes them concurrently, and fuses the rankings using Reciprocal Rank Fusion (RRF).

### Critical Memory Retrieval
Some facts (like hard API constraints or life-threatening allergies) are too important to leave to the probabilities of vector math. By assigning a `critical_slot` via a Policy, you can fetch these facts directly using `recall_critical()`, completely bypassing vector search.

---

## 7. Graph Relations

Facts don't exist in a vacuum. Engram allows you to create directional edges (with optional weights) between memories. 

For example, a `decision` memory can be related to a `constraint` memory via a `supports` relation. When you retrieve the decision, you can use `traverse()` or `traverse_many()` to follow the graph edges via recursive Postgres CTEs and pull the supporting constraints directly into your prompt.

---

## 8. Task Memory and the Event Ledger

For long-running agents, Engram maintains a durable `TaskRun`. 

Everything the agent does—user messages, tool calls, tool results, observations, and errors—is appended to an **immutable event ledger**. 

> [!IMPORTANT]
> The raw ledger is authoritative. Derived facts are generated asynchronously from the ledger via background `memory_jobs`.

Because the ledger can grow massive over long tasks, you don't inject the whole ledger into the prompt. Instead, you create periodic **Checkpoints** (compact state summaries). The `build_context()` API intelligently assembles a budgeted prompt block containing the most recent events, the latest checkpoint, and relevant semantic facts.

---

## 9. Long Input Processing

When an agent needs to process massive context blocks (like an entire codebase diff or a 50-page legal contract), passing it directly into the LLM is inefficient.

`record_long_input()` splits massive text chunks using heading-aware or token-aware chunking (via `chonkie`). It creates `Artifact` events anchored directly to exact character spans (quote hashes). You can then use `build_long_input_context()` to retrieve only the hyper-relevant source chunks necessary to answer a specific question, preventing context window bloat while maintaining strict citable evidence.
