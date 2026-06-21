# Benchmarks

Engram ships three reproducible benchmark scripts covering progressively harder long-term memory tasks. The numbers below come from running them against real databases with publicly available datasets. All three use on-device embeddings (free, no API cost at ingest) and the same retrieval pipeline.

> [!WARNING]
> **Honest setup note**: all three benchmarks ingest memories via `add_batch()` — raw episodic turns stored verbatim, no LLM extraction at ingest time. This is a deliberate floor measurement of Engram's retrieval layer. `add_conversation()` (full LLM-based extraction, deduplication, and supersession) is expected to score higher on structured fact types like `knowledge-update` and `preference-following` but was not used here. Both LLM composer and judge use the same model family (`claude-sonnet-4-6`), which is known to be lenient relative to cross-family judging. For stricter scores, run `--rejudge-only` with a different judge family.

---

## Results at a glance

| Benchmark | Dataset | Questions | Accuracy |
|---|---|---|---|
| **LongMemEval-S** (ICLR 2025) | isolated per-question haystacks | 500 | **89.8%** |
| **LoCoMo-10** (ACL 2024) | 10 long-running conversations | 1,540 | **85.7%** |
| **BEAM 1M** (ICLR 2026) | 35 conversations, 10 question types | 700 | **79.6%** |

![Engram benchmark results](assets/engram-benchmark.svg)

---

## LongMemEval-S — 89.8%

**Dataset**: 500 questions, each with its own isolated haystack of chat histories  
**Composer & Judge**: `claude-sonnet-4-6`  
**Embeddings**: `all-MiniLM-L6-v2` (384-d, on-device)  
**Retrieval**: hybrid search + cross-encoder rerank, session-diversified (max 4 turns/session), 60 memories/question  
**Graph depth**: 0 (disabled)

| Question type | Accuracy | Raw |
|---|---|---|
| single-session-user | 98.6% | 69 / 70 |
| knowledge-update | 97.4% | 76 / 78 |
| abstention | 96.7% | 29 / 30 |
| single-session-assistant | 94.6% | 53 / 56 |
| temporal-reasoning | 89.5% | 119 / 133 |
| single-session-preference | 83.3% | 25 / 30 |
| multi-session | 80.5% | 107 / 133 |
| **Overall** | **89.8%** | **449 / 500** |

```mermaid
---
config:
  theme: base
  themeVariables:
    xyChart:
      backgroundColor: "transparent"
      plotColorPalette: "#6d28d9"
      xAxisLabelColor: "#374151"
      yAxisLabelColor: "#374151"
      xAxisLineColor: "#d1d5db"
      yAxisLineColor: "#d1d5db"
---
xychart-beta horizontal
    title "LongMemEval-S accuracy by question type (%)"
    x-axis ["SS-user", "Know-update", "Abstain", "SS-assist", "Temporal", "SS-pref", "Multi-sess"]
    y-axis "Accuracy" 0 --> 100
    bar [98.6, 97.4, 96.7, 94.6, 89.5, 83.3, 80.5]
```

The 51 failures split as: 46 with the right session already retrieved — composer errors, not retrieval gaps. The remaining 5 are retrieval misses on questions that required connecting two unrelated conversation threads.

---

## LoCoMo-10 — 85.7%

LoCoMo (ACL 2024) uses 10 long-running synthetic two-person conversations spanning hundreds of sessions each. We evaluate categories 1–4 (1,540 questions); category 5 (adversarial) is excluded per the benchmark spec.

**Dataset**: 1,540 questions across 10 conversations  
**Composer & Judge**: `claude-sonnet-4-6`  
**Embeddings**: `all-MiniLM-L6-v2` (384-d, on-device)  
**Retrieval**: hybrid search + cross-encoder rerank + lineage traversal, session-diversified (max 4 turns/session), 60 memories/question  
**Graph depth**: 1

| Category | Accuracy | Raw |
|---|---|---|
| multi-hop | 87.9% | 248 / 282 |
| temporal | 86.9% | 279 / 321 |
| single-hop | 86.6% | 728 / 841 |
| open-domain | 67.7% | 65 / 96 |
| **Overall** | **85.7%** | **1,320 / 1,540** |

```mermaid
---
config:
  theme: base
  themeVariables:
    xyChart:
      backgroundColor: "transparent"
      plotColorPalette: "#7c3aed"
      xAxisLabelColor: "#374151"
      yAxisLabelColor: "#374151"
      xAxisLineColor: "#d1d5db"
      yAxisLineColor: "#d1d5db"
---
xychart-beta horizontal
    title "LoCoMo-10 accuracy by category (%)"
    x-axis ["Multi-hop", "Temporal", "Single-hop", "Open-domain"]
    y-axis "Accuracy" 0 --> 100
    bar [87.9, 86.9, 86.6, 67.7]
```

Open-domain (67.7%) is the honest weak spot. These questions ask about world knowledge that was never stored in the conversation — no retrieval optimization closes a gap in coverage.

---

## BEAM 1M — 79.6%

BEAM (ICLR 2026) is the hardest of the three. It tests ten distinct question types, including some that require Engram to do things raw retrieval systems fundamentally cannot: identify contradictions between turns, infer chronological event order, and produce full-span conversation summaries. We ran the full 1M-token split: 35 conversations × 20 questions = 700 total.

**Dataset**: 700 questions across 35 conversations (1M token scale)  
**Composer & Judge**: `claude-sonnet-4-6`  
**Embeddings**: `all-MiniLM-L6-v2` (384-d, on-device)  
**Retrieval**: hybrid search + cross-encoder rerank + lineage + graph traversal; candidate pool 500 pre-rerank, 100 post-rerank  
**Scoring**: rubric nugget scoring per question (0 / 0.5 / 1.0 per nugget, mean ≥ 0.5 = pass)  
**Graph depth**: 1

| Question type | Accuracy | Raw |
|---|---|---|
| abstention | 97.1% | 68 / 70 |
| preference_following | 92.9% | 65 / 70 |
| instruction_following | 90.0% | 63 / 70 |
| information_extraction | 84.3% | 59 / 70 |
| event_ordering | 81.4% | 57 / 70 |
| knowledge_update | 81.4% | 57 / 70 |
| multi_session_reasoning | 81.4% | 57 / 70 |
| contradiction_resolution | 80.0% | 56 / 70 |
| temporal_reasoning | 65.7% | 46 / 70 |
| summarization | 41.4% | 29 / 70 |
| **Overall** | **79.6%** | **557 / 700** |

Overall average nugget score: **0.711** (rubric nuggets scored 0 / 0.5 / 1.0; mean ≥ 0.5 = pass)

```mermaid
---
config:
  theme: base
  themeVariables:
    xyChart:
      backgroundColor: "transparent"
      plotColorPalette: "#059669"
      xAxisLabelColor: "#374151"
      yAxisLabelColor: "#374151"
      xAxisLineColor: "#d1d5db"
      yAxisLineColor: "#d1d5db"
---
xychart-beta horizontal
    title "BEAM 1M accuracy by question type (%)"
    x-axis ["Abstain", "Pref", "Instr", "InfoEx", "EventOrd", "KnowUpd", "MultiSess", "Contra", "Temporal", "Summ"]
    y-axis "Accuracy" 0 --> 100
    bar [97.1, 92.9, 90.0, 84.3, 81.4, 81.4, 81.4, 80.0, 65.7, 41.4]
```

### What the BEAM script does differently

The BEAM benchmark script applies several retrieval optimizations that are specific to its question types and worth being explicit about:

**Type-specific evidence budgets**: Single-fact question types (temporal_reasoning, information_extraction, knowledge_update, preference_following, abstention) are capped at 60 memories. All others use the full 100-memory budget. Giving single-fact types the full 100 increases noise without improving recall.

**Supplemental sub-queries for two hard types**:
- `event_ordering`: one targeted sub-query per rubric event, in addition to the broad query. Finds turns the general query misses because they describe a specific event, not the session topic.
- `contradiction_resolution`: five adversarial negation queries (`"never [topic]"`, `"not [topic]"`, etc.). The minority-opinion turn that creates a contradiction is semantically dominated by the majority-opinion turns and rarely appears in the top-500 reranked results. These queries surface it and are prepended to guarantee it falls within the evidence window.

**Question-type injection into the composer**: The question type is passed explicitly in the user prompt (`QUESTION TYPE: contradiction_resolution`) so type-specific rules in the composer system prompt fire reliably. Without this, the composer doesn't distinguish between "answer the question" and "report the contradiction without resolving it."

These are real engineering decisions that improve the relevant question types, but they're benchmark-tuned. A production agent doesn't know its question type in advance.

### Where BEAM still fails

**Summarization (41.4%)** is a hard architectural floor, not a tuning gap. The rubric checks for coverage across the entire conversation span — typically 6–8 distinct time periods and topic clusters. Relevance-ranked retrieval is precision-optimized: it returns the most similar turns, which cluster around the question topic and one or two recent sessions. Coverage-maximizing retrieval (returning representative samples from every session regardless of query similarity) doesn't exist in the current API surface. Session stratification helps at the margins but doesn't solve it.

**Temporal reasoning (65.7%)** misses when BEAM embeds date information implicitly in turn text (`"[May 15, 2023] USER: ..."`). The recall operator's temporal phrase resolution works correctly, but when BEAM questions ask for date arithmetic (`"how many days between X and Y"`), Engram must extract two dates from two separate turns and compute an interval in the composer pass. The temporal_chain recall intent (parallel search per event anchor, evidence merged chronologically) was added specifically for this and helps — but the gap remains when dates appear only as inline text rather than structured metadata.

---

## The shared pipeline

All three benchmarks run the same core pipeline:

```
add_batch() → search() + recall() + get_lineage() + traverse_many() → composer LLM
```

```mermaid
---
config:
  theme: base
  themeVariables:
    primaryColor: "#ede9fe"
    primaryBorderColor: "#7c3aed"
    primaryTextColor: "#1e1b4b"
    secondaryColor: "#f0f9ff"
    secondaryBorderColor: "#38bdf8"
    tertiaryColor: "#fefce8"
    tertiaryBorderColor: "#eab308"
    lineColor: "#94a3b8"
    textColor: "#1f2937"
    titleColor: "#111827"
    clusterBkg: "#f9fafb"
    clusterBorder: "#e2e8f0"
    edgeLabelBackground: "#ffffff"
---
flowchart LR
    IN(["Input"]):::pill

    subgraph INGEST["Ingest  ·  0 LLM calls"]
        A["add_batch()"] --> B[("pgvector")]
    end

    subgraph RETRIEVE["Retrieve  ·  4 surfaces"]
        C["search() + rerank"] --> D["recall() + lineage + traverse"]
    end

    subgraph EVALUATE["Evaluate"]
        E["Composer LLM"] --> F["Judge LLM"]
    end

    OUT(["Score"]):::pill

    IN --> A
    B  --> C
    D  --> E
    F  --> OUT

    classDef pill fill:#ede9fe,stroke:#7c3aed,color:#1e1b4b,font-weight:600

    style INGEST   fill:#f0f9ff,stroke:#38bdf8,color:#0c4a6e
    style RETRIEVE fill:#fefce8,stroke:#eab308,color:#713f12
    style EVALUATE fill:#fdf4ff,stroke:#c026d3,color:#701a75
```

**Ingest** (`add_batch()`): raw conversation turns are embedded on-device and written to pgvector. No LLM is called at this stage. Ingestion takes roughly 12 seconds per question on LongMemEval.

**Retrieve** — four surfaces, all called per question:

| API | What it does |
|---|---|
| `search(mode='hybrid', rerank=True)` | pgvector cosine + PostgreSQL full-text, fused with Reciprocal Rank Fusion, then cross-encoder reranked against the question |
| `recall(compose_answer=False)` | intent-classified retrieval (current / historical / event / lineage / temporal_chain); passes structured lineage evidence — current value, superseded predecessors, conflict notes — without generating a prose answer |
| `get_lineage()` | follows supersession chains so corrected values carry their history into the evidence block |
| `traverse_many()` | multi-hop graph traversal from the top-5 active search hits |

**Generate**: one composer LLM call assembles the evidence block into an answer. The judge runs separately on the same output.

> [!NOTE]
> **What this measures**: All three benchmarks bypass `add_conversation()` (Engram's full LLM-extraction pipeline). The scores reflect the retrieval layer as a raw substrate — episodic turns stored verbatim, with all reasoning deferred to query time. `add_conversation()` adds semantic extraction, fact deduplication, and conflict resolution at ingest; these are expected to improve structured fact types. The benchmark numbers are a floor, not a ceiling.

---

## What each component contributes (LongMemEval ablation)

| Configuration | Composer | Rerank | Accuracy |
|---|---|---|---|
| Hybrid search only | Haiku | no | 77.8% |
| + cross-encoder rerank | Haiku | yes | 87.0% |
| + stronger composer | Sonnet | yes | **89.8%** |

**Reranking is the biggest single lever.** The 9-point gap between no-rerank and rerank is retrieval quality: irrelevant turns are cut before the composer sees them. The additional 3 points from Haiku to Sonnet is reasoning quality over evidence that's already clean.

**Evidence budget interacts with question type.** Tightening below 60 memories regressed aggregation and multi-session questions. 60 memories over a reranked pool outperformed 30 memories with higher nominal precision, because counting and cross-session reasoning need every relevant turn in context.

### Retrieval vs. reader: how much is the prompt?

Two tools isolate where accuracy actually comes from, because end-to-end accuracy conflates retrieval, composer prompt, model, and judge:

- `--score-retrieval <traces.jsonl>` computes a **prompt-free retrieval hit-rate**: it joins the retrieved session ids against the dataset's gold `answer_session_ids`. No LLM, no judge. On the 89.8% run, retrieval surfaces the gold answer session **~100%** of the time (per question type, all ≥96.7%). Retrieval is not the bottleneck on LongMemEval — wherever an answer is wrong, the evidence was almost always present.
- `--dumb-reader` swaps the tuned composer for a neutral one-paragraph reader, holding ingest, retrieval, and judge identical. The accuracy delta isolates the prompt's contribution.

On a 100-question Sonnet slice: the **dumb reader scores 86%**, the tuned composer **91%** — a directional +5 points (not statistically significant at this sample, McNemar p≈0.23), concentrated entirely in hard multi-session questions. The same tuned prompt was net-*negative* on Haiku. Read together: the substrate (retrieval + the model reading clean evidence) carries ~95% of the result; the 300-line composer prompt is a model-specific top-up, not the engine, and should not be treated as portable accuracy. A caveat on the retrieval number: hit-rate is measured at session granularity, so it is an upper bound on evidence adequacy (the answer-bearing *turn* within a retrieved session may still be trimmed by the budget).

---

## Where each benchmark still fails

**LongMemEval** (51 failures): 46 had the right session in the retrieved evidence — composer errors, not retrieval. The remaining 5 are retrieval misses requiring two unrelated threads to be connected.

**LoCoMo open-domain** (31% miss rate): world knowledge the system never ingested. Retrieval cannot fill facts that were never stored.

**BEAM summarization** (59% miss rate): relevance-ranked search returns similar turns, not representative turns. A question requiring coverage of 6–8 distinct time periods will always undercount — the highest-scoring memories cluster around the question topic and the most recent sessions. This is an architectural gap in the current retrieval surface, not a prompt engineering problem.

**BEAM temporal reasoning** (34% miss rate): two-hop date arithmetic. Both event dates are usually in the evidence block, but computing the interval requires the composer to extract two dates from different turns and subtract. Accuracy here depends heavily on how explicitly dates are stated in the conversation. When dates appear only as inline text (`[May 15, 2023]`), the composer handles it. When they're implicit (`"that was three weeks after I started"`), the chain breaks.

---

## Reproduce it

All scripts are in `benchmark/`. Data files go in `data/`.

> [!WARNING]
> LLM API calls for composer and judge are billable. On-device embeddings are free. Set `ENGRAM_ANTHROPIC_API_KEY` in your `.env`.

### LongMemEval — 89.8% run

```bash
python benchmark/longmemeval_benchmark.py \
  --llm-model claude-sonnet-4-6 \
  --judge-model claude-sonnet-4-6 \
  --rerank \
  --search-limit 60 \
  --max-per-session 4 \
  --local-embedding --embedding-model all-MiniLM-L6-v2 --embedding-dimension 384 \
  --concurrency 8 \
  --graph-depth 0 \
  --clean-db \
  --output-dir benchmark/runs/lme-final-v4
```

### LongMemEval — cheaper run (Haiku composer, 87.0%)

```bash
python benchmark/longmemeval_benchmark.py \
  --rerank \
  --search-limit 60 \
  --max-per-session 4 \
  --judge-model claude-sonnet-4-6 \
  --local-embedding --embedding-model all-MiniLM-L6-v2 --embedding-dimension 384 \
  --concurrency 8 \
  --graph-depth 0 \
  --clean-db \
  --output-dir benchmark/runs/lme-cheap
```

### LoCoMo-10 — 85.7% run

```bash
python benchmark/locomo_benchmark.py \
  --conversations 0,1,2,3,4,5,6,7,8,9 \
  --search-limit 60 \
  --rerank \
  --concurrency 8 \
  --llm-model claude-sonnet-4-6 \
  --judge-model claude-sonnet-4-6 \
  --clean-db \
  --output-dir benchmark/runs/locomo-sonnet
```

### BEAM 1M — 79.6% run

```bash
python benchmark/beam_benchmark.py \
  --chat-sizes 1M \
  --llm-model claude-sonnet-4-6 \
  --judge-model claude-sonnet-4-6 \
  --rerank \
  --search-limit 100 \
  --candidate-limit 500 \
  --cutoffs 100 \
  --event-ordering-tau \
  --concurrency 8 \
  --judge-concurrency 10 \
  --clean-db \
  --output-dir benchmark/runs/beam-1m-v5
```

### Re-score without re-running

```bash
python benchmark/longmemeval_benchmark.py \
  --rejudge-only benchmark/runs/lme-final-v4/traces.jsonl \
  --judge-model claude-sonnet-4-6 \
  --output-dir benchmark/runs/lme-rejudge
```

### Output files

Each run writes three files to the output directory:

| File | Contents |
|---|---|
| `traces.jsonl` | Question, gold answer, retrieved evidence, composer answer, retrieval stats — one JSON object per question |
| `judgments.jsonl` | Per-question verdict with reasoning |
| `summary.json` | Overall and per-type accuracy, full configuration |

---

## Notes for the community

**Same-family judge**: all three headline runs use `claude-sonnet-4-6` for both composer and judge. Same-family judges are known to be lenient relative to cross-family evaluation. For a stricter score, run `--rejudge-only` with a different model family (e.g., composer `claude-sonnet-4-6`, judge `claude-opus-4-8`).

**BEAM is a newer and harder benchmark**: unlike LongMemEval and LoCoMo, BEAM includes question types that test the retrieval system's ability to surface contradictions, reconstruct event orderings, and summarize across full conversation spans. The 79.6% headline includes a 41.4% summarization score that pulls the average down significantly — the other eight question types average 84.0%.

**`add_batch()` vs `add_conversation()`**: these benchmarks deliberately use `add_batch()` (raw episodic turn storage, zero ingest LLM calls) to isolate the retrieval layer. Production use of `add_conversation()` performs LLM-based fact extraction, deduplication, and supersession at write time, which reduces retrieval noise for structured fact types. The benchmark scores are a lower bound on what the full Engram pipeline can achieve.

**Reproducibility**: given the same model versions and configuration, runs reproduce within ~1%. Accuracy changes meaningfully with embedding model choice, reranking, evidence budget, and composer strength — all exact parameters are stored in `summary.json` alongside the scores.
