## **AI Memory for Memory Decay Scoring and Graph Traversal Queries: A Comprehensive Research Report**

### **Executive Summary**

This research covers cutting-edge developments in AI memory systems with specific focus on two critical areas: memory decay scoring mechanisms and graph traversal query patterns. The field has rapidly matured in 2024-2025, moving from theoretical frameworks to production-ready implementations.

---

### **1. MEMORY DECAY SCORING**

#### **1.1 Foundational Theory: Ebbinghaus Forgetting Curve**

Memory decay in AI systems is inspired by the Ebbinghaus Forgetting Curve theory, which describes how memory retention decreases over time with a steeper decline initially and slower rate later. The forgetting curve is modeled using an exponential decay formula:

**R = e^(-t/S)**

Where:
- R = memory retention (fraction of information retained)
- t = time elapsed since memory formation
- S = memory strength (influenced by learning depth and repetition)

#### **1.2 Contemporary Implementation: MemoryBank**

MemoryBank implements a Weighted Memory Retrieval (WMR) system where the Recency score represents memory decay, decreasing hourly by a decay factor of 0.995. The implementation simplifies the model:

- S initialized at 1 upon first mention
- S increased by 1 each time memory is recalled
- t reset to 0 on recall
- This reduces forgetting probability for frequently accessed memories

The memory updating mechanism operates using principles inspired by the Ebbinghaus Forgetting Curve theory, allowing for realistic and human-like memory recall processes.

#### **1.3 Practical Memory Scoring Formulas**

**Combined Weighted Scoring:**

Modern systems calculate final scores using weighted combinations: final_score = (0.6 × relevance_score) + (0.25 × recency_score) + (0.15 × importance_score)

**Time Decay Implementations:**

Recency scoring typically uses: recency_score = 1.0 / (1.0 + 0.01 × time_delta_hours)

Alternative approaches decay over 30 days: recency_score = max(0, 1 - (days_old / 30))

**Exponential Moving Averages:**

At the heart of exponential moving average is applying exponential decay to weight assignments: if decay factor α = 0.5 and window = one day, an event now has weight 0.5^0 = 1.0, one day old has 0.5^1 = 0.5, two days old has 0.5^2 = 0.25

#### **1.4 Advanced Memory Dynamics**

**MemOS Framework:**

MemOS uses runtime metrics including access patterns (frequency and recency) to determine whether memory is "hot" or "cold" during inference, adjusting caching priority—for example, promoting high-frequency plaintext memory into activation vectors for faster decoding

**Memory State Transitions:**

Memory items transition through five states—Generated, Activated, Merged, Archived, and Expired—based on access patterns, time decay, and task labels

#### **1.5 Contextual Memory Intelligence**

Contextual Entropy is defined as the rate at which memory coherence deteriorates in distributed systems, operationalized by measuring decay in memory trace relevance, semantic divergence over time, or proportion of memory units lacking rationale linkage

---

### **2. GRAPH TRAVERSAL QUERIES**

#### **2.1 Knowledge Graph Memory Architecture**

**Graphiti Framework:**

Graphiti achieves extremely low-latency retrieval with P95 latency of 300ms enabled by hybrid search combining semantic embeddings, keyword (BM25) search, and direct graph traversal—avoiding any LLM calls during retrieval

Graphiti combines semantic embeddings, keyword (BM25), and graph traversal to achieve low-latency queries without reliance on LLM summarization

#### **2.2 Graph Traversal Patterns**

**Multi-Hop Reasoning:**

GraphRAG combines semantic similarity via vector search with structured, connected reasoning via graph queries, enabling LLMs to deliver answers that are not only relevant but also richer, deeper, and easier to trace back to source data

An agent can leverage graph traversal to perform logical inferences: for example, the agent might traverse "Policy ABC – ownedBy → Dept X – oversees → Project Alpha, and then find Dept X – flaggedRisk → Data Privacy Risk"

**Query Examples:**

Traversal examples include: "Find all people within 3 degrees of connection from this user who have expertise in machine learning and work at companies in the healthcare sector" or "ConceptY → discussed_in → Papers → written_by → Authors, ranked by citation count"

#### **2.3 Neo4j Cypher Implementation**

**Cypher 25 Enhancements:**

Cypher 25 incorporates stateful traversal with allReduce for aggregating traversal state (time, energy, cost) inlined during path expansion for early pruning; repeatable elements allowing traversal to revisit nodes or relationships; and conditional queries enabling branching logic

**Graph Data Model Patterns:**

A possible retrieval method is to perform similarity search against question embeddings in the database and then traverse to associated Cypher queries, returning top k question texts and Cypher query statements formatted as few-shot examples

#### **2.4 Hybrid Retrieval Strategies**

Graphiti enhances traditional hybrid search by adding graph traversal and temporal reasoning, allowing queries like finding entities with temporal context and relationship chains

During indexing, the knowledge graph is stored in both Vector DB and Graph DB: Milvus stores embeddings for fast similarity lookup at query time, and iGraph stores nodes and edges for fast traversal

#### **2.5 Multi-Hop Query Optimization**

**HopRAG System:**

HopRAG's reasoning-augmented graph traversal selectively hops to the most promising neighbor by leveraging LLM reasoning over edge questions, introducing a Helpfulness metric integrating textual similarity and logical importance through normalized arrival counts

**RT-RAG Framework:**

RT-RAG decomposes complex questions into consensus-validated tree structure with explicit entity analysis, retrieves evidence through bottom-up traversal with query refinement, and integrates information hierarchically to maintain coherence across multiple hops

---

### **3. INTEGRATION: MEMORY DECAY IN GRAPH SYSTEMS**

#### **3.1 Temporal Graph Traversal**

Graphiti stores episodes with temporal context and reference_time, enabling temporal reasoning during graph traversal alongside semantic and relationship-based queries

#### **3.2 Agent Memory Management**

Episodic memory is written in the background with receipt of user feedback to prevent inclusion of bad or unhelpful memories which would diminish performance, while procedural memory captures historical experiences and self-reflection insights

#### **3.3 Scalability and Performance**

Graph databases maintain consistent performance regardless of how many hops needed, while relational databases slow down exponentially as traversal depth increases

Using graph extension in Mem0 outperforms traditional memory systems by capturing complex relationships between entities and supporting advanced reasoning across interconnected facts, enabling multi-hop graph reasoning and hybrid retrieval across graph, vector, and keyword modalities

---

### **4. PRODUCTION-READY IMPLEMENTATIONS**

#### **4.1 Commercial Systems**

- **ChatGPT Memory**: User preference tracking with decay
- **Apple Intelligence**: Personal context with temporal awareness
- **Microsoft Recall**: Timeline-based memory retrieval

#### **4.2 Open-Source Frameworks**

- **MemoryScope**: Multi-level memory with decay mechanisms
- **Mem0**: Graph-based memory with temporal support
- **Graphiti**: Neo4j-based knowledge graph memory
- **Cognee**: Agentic systems with cognitive memory layers

#### **4.3 Key Technologies**

**Storage Backends:**
- Neo4j: Graph database with native traversal
- FalkorDB: High-performance graph queries
- Amazon Neptune Analytics: In-memory graph analytics
- Milvus: Vector similarity search

**Query Languages:**
- Cypher: Neo4j's declarative graph query language
- SPARQL: RDF triple store queries
- Gremlin: TinkerPop graph traversal language

---

### **5. BEST PRACTICES AND RECOMMENDATIONS**

#### **5.1 Memory Decay Scoring**

1. **Use hybrid scoring**: Combine relevance (0.6), recency (0.25), and importance (0.15)
2. **Implement exponential decay**: Factor of 0.995 per hour is effective
3. **Track access patterns**: Increment strength on recall, reset decay timer
4. **Consider context**: Adjust decay rates based on information type

#### **5.2 Graph Traversal**

1. **Hybrid retrieval**: Combine vector search + graph traversal + keyword search
2. **Limit hop depth**: 2-3 hops for most queries, adaptive based on confidence
3. **Use typed relationships**: Enable precise multi-hop reasoning
4. **Implement caching**: P95 latency under 300ms is achievable
5. **Bidirectional modeling**: Support queries from either direction

#### **5.3 Architecture Design**

1. **Separate hot/cold paths**: In-memory for active, disk for archived
2. **Progressive loading**: Stream results rather than blocking
3. **State management**: Track memory lifecycle (Generated → Activated → Merged → Archived → Expired)
4. **Error handling**: Graceful degradation when memory unavailable

---

### **6. EMERGING TRENDS AND FUTURE DIRECTIONS**

1. **Multimodal memory**: Extending beyond text to images, audio, video
2. **Streaming memory**: Real-time updates vs. batch processing
3. **Shared memory**: Cross-agent memory sharing and collaboration
4. **Automated evolution**: Self-optimizing memory systems
5. **Neurosymbolic integration**: Combining neural networks with symbolic reasoning

---

### **7. TECHNICAL SPECIFICATIONS**

#### **Memory Decay Formula (Production)**
```python
def calculate_memory_score(memory, current_time):
    # Time decay
    hours_elapsed = (current_time - memory.timestamp).total_seconds() / 3600
    recency_score = 0.995 ** hours_elapsed
    
    # Importance (LLM-generated or rule-based)
    importance_score = memory.importance_factor
    
    # Semantic relevance
    relevance_score = cosine_similarity(memory.embedding, query.embedding)
    
    # Weighted combination
    final_score = (0.6 * relevance_score + 
                  0.25 * recency_score + 
                  0.15 * importance_score)
    
    return final_score
```

#### **Graph Traversal (Cypher Example)**
```cypher
// Multi-hop reasoning with temporal constraints
MATCH path = (start:Entity {id: $entity_id})
  -[:RELATED_TO*1..3]->
  (end:Entity)
WHERE start.timestamp >= $start_time
  AND all(r in relationships(path) WHERE r.weight > 0.5)
RETURN path, 
       reduce(s = 0, r in relationships(path) | s + r.weight) as path_score
ORDER BY path_score DESC
LIMIT 10
```

---

This comprehensive research demonstrates that AI memory systems have matured significantly, with production-ready implementations combining sophisticated decay mechanisms with efficient graph traversal capabilities. The integration of these two approaches enables AI agents to maintain relevant, context-aware memory while efficiently navigating complex knowledge structures.