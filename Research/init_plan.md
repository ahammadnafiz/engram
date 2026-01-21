Converged cognitive Architecture- it means stop using 3 different tools like (files, vectordb, graphdb) and use 1 tool that does it all
which can be done by PostgreSQL.
what PostgreSQL. is capable of to do:
Regular data (SQL)
Documents (JSONB)
Vectors (pgvector)
Graphs (via relations)
when you put everything in the one place you get ACID(atomicity, consistency, isolation, durability) property. Which will make sure that your data doesn't gets corrupted when 2 people touch at the same time
sql
-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Agents Table: Identity and Configuration
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Memory Table: The Converged Store
-- Replaces both "File-Based" logs and "Vector Stores"
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id),
    session_id UUID NOT NULL,

    -- Content: The raw memory or fact
    content TEXT NOT NULL,

    -- Structured Metadata: For filtering (e.g., {"type": "preference", "topic": "coding"})
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Semantic Search: Dense Vector (e.g., 1536 dim for OpenAI)
    embedding VECTOR(1536),

    -- Lexical Search: Sparse Vector for keyword matching (BM25)
    -- Generated automatically from content
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- Temporal Context
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INT DEFAULT 0
);

-- Indices for Performance
-- HNSW Index for fast approximate nearest neighbor search (Vector)
CREATE INDEX idx_memory_embedding ON agent_memory USING hnsw (embedding vector_cosine_ops);

-- GIN Index for fast full-text search (Keyword)
CREATE INDEX idx_memory_text ON agent_memory USING GIN (text_search);

-- GIN Index for fast metadata filtering (JSONB)
CREATE INDEX idx_memory_metadata ON agent_memory USING GIN (metadata);

-- 3. Entity Graph Table: Lightweight Graph within SQL
-- Replaces the heavy "Graph Database" for simple relations
CREATE TABLE memory_relations (
    source_id UUID REFERENCES agent_memory(id),
    target_id UUID REFERENCES agent_memory(id),
    relation_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relation_type)
);-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Agents Table: Identity and Configuration
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Memory Table: The Converged Store
-- Replaces both "File-Based" logs and "Vector Stores"
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id),
    session_id UUID NOT NULL,

    -- Content: The raw memory or fact
    content TEXT NOT NULL,

    -- Structured Metadata: For filtering (e.g., {"type": "preference", "topic": "coding"})
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Semantic Search: Dense Vector (e.g., 1536 dim for OpenAI)
    embedding VECTOR(1536),

    -- Lexical Search: Sparse Vector for keyword matching (BM25)
    -- Generated automatically from content
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- Temporal Context
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INT DEFAULT 0
);

-- Indices for Performance
-- HNSW Index for fast approximate nearest neighbor search (Vector)
CREATE INDEX idx_memory_embedding ON agent_memory USING hnsw (embedding vector_cosine_ops);

-- GIN Index for fast full-text search (Keyword)
CREATE INDEX idx_memory_text ON agent_memory USING GIN (text_search);

-- GIN Index for fast metadata filtering (JSONB)
CREATE INDEX idx_memory_metadata ON agent_memory USING GIN (metadata);

-- 3. Entity Graph Table: Lightweight Graph within SQL
-- Replaces the heavy "Graph Database" for simple relations
CREATE TABLE memory_relations (
    source_id UUID REFERENCES agent_memory(id),
    target_id UUID REFERENCES agent_memory(id),
    relation_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relation_type)
);
4. Retrieval strategy: Hybrid search
after storing the data, now comes retrieval & how you get it out? if you just use Vector Search (embeddings), you have a problem as Keyword blindness.
E.g. Error 504 and Error 502 are basically the same thing conceptually.
So a vector search thinks they are identical. But to a developer, they are very different. We need Hybrid search.
This  mixes:
1. Dense retrieval: understanding the vibe/meaning.
2. Sparse retrieval: matching exact keywords (BM25).
To combine them, we use math called Reciprocal Rank Fusion  (RRF). This makes sure the results are actually relevant.
RRF_score
(
d
)
=
∑
r
∈
R
1
k
+
rank
(
d
,
r
)
it will take the rank from the vector search and the rank from the keyword search and fuses them.
python
import  asyncio
from  sqlalchemy.ext.asyncio  import  create_async_engine,  AsyncSession
from  sqlalchemy  import  text
from  typing  import  List,  Dict

DATABASE_URL  =  "postgresql+asyncpg://user:pass@localhost/agent_db"
engine  =  create_async_engine(DATABASE_URL,  pool_size=20)

class  CognitiveStore:
    def  __init__(self,  agent_id:  str):
        self.agent_id  =  agent_id

    async  def  hybrid_search(self,  query_text:  str,  query_vector:  List[float],  limit:  int  =  5):
        """
        Executes  Hybrid  Search  using  Reciprocal  Rank  Fusion  (RRF).
        Combines  pgvector  (Semantic)  and  tsvector  (Keyword).
        """
        async  with  AsyncSession(engine)  as  session:
            #  Complex  SQL  to  perform  both  searches  and  fuse  results
            rrf_query  =  text("""
                WITH  semantic  AS  (
                    SELECT  id,  content,  
                           RANK()  OVER  (ORDER  BY  embedding  <=>  :vec)  as  rank_dense
                    FROM  agent_memory
                    WHERE  agent_id  =  :aid
                    ORDER  BY  embedding  <=>  :vec
                    LIMIT  20
                ),
                keyword  AS  (
                    SELECT  id,  content,  
                           RANK()  OVER  (ORDER  BY  ts_rank_cd(text_search,  plainto_tsquery(:txt))  DESC)  as  rank_sparse
                    FROM  agent_memory
                    WHERE  agent_id  =  :aid  AND  text_search  @@  plainto_tsquery(:txt)
                    LIMIT  20
                )
                SELECT  
                    COALESCE(s.content,  k.content)  as  content,
                    COALESCE(s.id,  k.id)  as  id,
                    (COALESCE(1.0  /  (60  +  s.rank_dense),  0.0)  +  
                     COALESCE(1.0  /  (60  +  k.rank_sparse),  0.0))  as  rrf_score
                FROM  semantic  s
                FULL  OUTER  JOIN  keyword  k  ON  s.id  =  k.id
                ORDER  BY  rrf_score  DESC
                LIMIT  :limit;
            """)
            
            result  =  await  session.execute(rrf_query,  {
                "aid":  self.agent_id,
                "vec":  str(query_vector),  #  pgvector  requires  string  representation
                "txt":  query_text,
                "limit":  limit
            })
            
            return  [row._mapping  for  row  in  result]

    async  def  add_memory_atomic(self,  content:  str,  vector:  List[float],  metadata:  Dict):
        """
        Atomic  Write  guarantees  that  memory  is  never  partially  written.
        """
        async  with  AsyncSession(engine)  as  session:
            async  with  session.begin():  #  Starts  Transaction
                await  session.execute(
                    text("""
                        INSERT  INTO  agent_memory  (agent_id,  session_id,  content,  embedding,  metadata)
                        VALUES  (:aid,  :sid,  :cnt,  :vec,  :meta)
                    """),
                    {
                        "aid":  self.agent_id,
                        "sid":  "session_123",  #  Dynamic  in  prod
                        "cnt":  content,
                        "vec":  str(vector),
                        "meta":  json.dumps(metadata)
                    }
                )
5. Making Sure It Is Reliable
We talked about race conditions, but what about crashes?
if your agent says "I will send that email" & then crashes, you have a problem. If it updates the memory "I sent it" but didn't send it, which will be a False write which will be an issue. If it sends it but forgets to write to memory it might send it twice
The solution to this will be Transactional Outbox Pattern. You will save the memory & the action (like "send email") to the database in the same transaction. Then a background worker reads the DB and sends the email. Since they are in the same DB transaction it is impossible to have one without the other.
You need Idempotency. Give every action a unique ID. If the agent tries to do the Action, (suppose#12345)  twice the system sees the ID and says "Nope, the task has been executed earlier"
6. Security for your Memory
Agent that remembers everything is a security nightmare. We face two main threats for this:
Indirect Prompt Injection: imagine an agent which scrapes a webpage that has hidden white text. If the agent blindly saves this into its vector database, that memory becomes a Time Bomb, after a few time the user might ask a question that triggers the memory and the malicious instruction executes.
Solution- We can use the Input Sanitization to scan and clean text before it enters the memory.
PII Exposure: storing names & emails in a vector database can be a risky. Vectors are hard to read but the metadata attached to them is  just a plain text.
Solution: we will use Synthetic Replacement. Instead of storing "Himanshu" we can store "Person_A". We will keep a secure, encrypted key in a separate table that maps "Person_A" back to "Himanshu" only when the absolutely necessary. Alternatively, we use the Dynamic Masking to hide sensitive columns from the agent unless it has specific admin permissions
Operational Maintenance: The Decay of Memory
Memory is not a "Write Once, Read Forever" system. you have to maintain, otherwise it degrades. it remembers too much useless information, making retrieval slow and confusing. We need to implement biological style forgetting.
Memory Decay: we need a formula that values the recent information more than old information. When retrieving memories, we don't just look for the matches we weigh them by passing time
Score
=
(
VectorSimilarity
×
0.7
)
+
(
RecencyScore
×
0.3
)
This ensures that "I like Apple" (said today) will beat "I like strawberry"(said last week), even if the vector match is similar.
Summarization pipeline: we shouldn't store every "Hello" or "Okay" we use a Buffer (like Redis) to hold the last 10 turns of conversation. Once the buffer is full, trigger an LLM to summarize those 10 turns into one concise fact. We save the summary to Postgres and delete the raw logs. This reduces storage costs by 90% and removes noise.
Conclusion
I am thankful to that rejection which ultimately worked as a wake up call for me to shift from traditional ML world, where we are obsessed over benchmarks and weights. But in the Agentic world, intelligence is ephemeral without continuity. A model that cannot remember in Long horizon is just a sophisticated autocomplete; a system that can remember is an agent.
Finally we have to treat memory as a living biological lifecycle not just a static archive. Which can be achieved by Through Transactional Outbox Patterns where we ensure actions match memories. Through Synthetic replacement, we also secure the User privacy & through Time Based Decay & summarization, we prevent cognitive overload. Developing & building a system that can run for weeks without crashing, remember a user's preference from days ago without hallucinating & handle concurrent requests without corrupting the data. This can be the difference between a demo & a product.
Testing various local architectures to master the agentic workflows, while scouting for new high impact opportunities in AI.