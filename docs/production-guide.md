# Engram Production Guide

Complete guide for integrating Engram memory system into production applications.

## Table of Contents
1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Basic Integration](#basic-integration)
4. [FastAPI Integration](#fastapi-integration)
5. [LangChain Integration](#langchain-integration)
6. [Multi-Agent Systems](#multi-agent-systems)
7. [Scaling & Performance](#scaling--performance)
8. [Best Practices](#best-practices)

---

## Installation

### Option 1: Package Install (Recommended)
```bash
pip install engram
```

### Option 2: From Source
```bash
git clone https://github.com/your-org/engram.git
cd engram
pip install -e .
```

### Database Setup
```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Or use existing PostgreSQL (requires pgvector extension)
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## Configuration

### Environment Variables
```bash
# .env file
# Database
ENGRAM_DATABASE_URL=postgresql://user:pass@localhost:5432/engram

# Embedding Provider (choose one)
ENGRAM_EMBEDDING_PROVIDER=openai  # or sentence-transformers, cohere, ollama
ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
ENGRAM_EMBEDDING_DIMENSION=1536

# LLM Provider (for fact extraction)
ENGRAM_LLM_PROVIDER=openai
ENGRAM_LLM_MODEL=gpt-4o-mini

# API Keys
ENGRAM_OPENAI_API_KEY=sk-...
# ENGRAM_COHERE_API_KEY=...
# ENGRAM_ANTHROPIC_API_KEY=...

# Optional: Local models
# ENGRAM_OLLAMA_BASE_URL=http://localhost:11434
```

### Programmatic Configuration
```python
from engram import Engram, EmbeddingService, LLMService

# Create services with explicit configuration
embedding = EmbeddingService.from_provider(
    "openai",
    model="text-embedding-3-small",
    api_key="sk-...",
)

llm = LLMService.from_provider(
    "openai", 
    model="gpt-4o-mini",
    api_key="sk-...",
)

# Or use environment-based settings
embedding = EmbeddingService.from_settings()
llm = LLMService.from_settings()
```

---

## Basic Integration

### Simple Memory Storage
```python
import asyncio
from engram import Engram

async def main():
    # Initialize
    engram = Engram()
    await engram.connect()
    
    # Store a memory
    memory = await engram.add(
        content="User prefers dark mode",
        agent_id="my-agent",
        user_id="user-123",
        metadata={"category": "preference"},
    )
    
    # Search memories
    results = await engram.search(
        query="What theme does the user like?",
        agent_id="my-agent",
        user_id="user-123",
        limit=5,
    )
    
    for r in results:
        print(f"[{r.score:.2f}] {r.memory.content}")
    
    # Cleanup
    await engram.close()

asyncio.run(main())
```

### With Fact Extraction
```python
from engram import Engram, LLMService
from engram.llm import MemoryOperationType

async def process_conversation(user_msg: str, bot_response: str):
    engram = Engram()
    await engram.connect()
    
    llm = LLMService.from_provider("openai", model="gpt-4o-mini")
    
    # Extract facts from conversation
    facts = await llm.extract_facts(
        user_message=user_msg,
        assistant_response=bot_response,
    )
    
    # Store each fact
    for fact in facts:
        # Check for duplicates
        existing = await engram.search(
            query=fact, 
            agent_id="assistant",
            limit=1,
        )
        
        if not existing or existing[0].score < 0.85:
            await engram.add(
                content=fact,
                agent_id="assistant",
                user_id="user-123",
            )
    
    await engram.close()
```

---

## FastAPI Integration

### Complete API Example
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
from engram import Engram, LLMService, EmbeddingService

# Global instances
engram: Engram | None = None
llm: LLMService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engram, llm
    
    # Startup
    engram = Engram()
    await engram.connect()
    llm = LLMService.from_settings()
    
    yield
    
    # Shutdown
    if engram:
        await engram.close()

app = FastAPI(lifespan=lifespan)

# Request/Response Models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    memories_used: list[str]

class MemoryRequest(BaseModel):
    user_id: str
    content: str
    metadata: dict = {}

# Dependency
async def get_engram() -> Engram:
    if not engram:
        raise HTTPException(500, "Engram not initialized")
    return engram

# Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Engram = Depends(get_engram)):
    # Retrieve relevant memories
    results = await db.search(
        query=req.message,
        agent_id="assistant",
        user_id=req.user_id,
        limit=5,
    )
    
    memories = [r.memory.content for r in results if r.score > 0.3]
    
    # Build context for LLM
    context = "\n".join(f"- {m}" for m in memories)
    
    # Generate response (your LLM call here)
    response = await generate_response(req.message, context)
    
    # Extract and store facts in background
    asyncio.create_task(store_facts(req.user_id, req.message, response))
    
    return ChatResponse(response=response, memories_used=memories)

@app.post("/memories")
async def add_memory(req: MemoryRequest, db: Engram = Depends(get_engram)):
    memory = await db.add(
        content=req.content,
        agent_id="assistant",
        user_id=req.user_id,
        metadata=req.metadata,
    )
    return {"memory_id": memory.memory_id}

@app.get("/memories/{user_id}")
async def get_memories(user_id: str, db: Engram = Depends(get_engram)):
    memories = await db.list_recent(
        agent_id="assistant",
        user_id=user_id,
        limit=20,
    )
    return [{"id": m.memory_id, "content": m.content} for m in memories]

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, db: Engram = Depends(get_engram)):
    await db.forget(memory_id)
    return {"deleted": memory_id}

async def store_facts(user_id: str, user_msg: str, bot_msg: str):
    """Background task to extract and store facts."""
    if not llm or not engram:
        return
    
    facts = await llm.extract_facts(user_msg, bot_msg)
    for fact in facts:
        try:
            await engram.add(
                content=fact,
                agent_id="assistant", 
                user_id=user_id,
            )
        except Exception:
            pass  # Duplicate or other error
```

### Run with Uvicorn
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## LangChain Integration

### Custom Memory Class
```python
from langchain.memory import BaseMemory
from langchain.schema import BaseMessage
from engram import Engram
from typing import Any

class EngramMemory(BaseMemory):
    """LangChain memory backed by Engram."""
    
    engram: Engram
    agent_id: str
    user_id: str
    memory_key: str = "history"
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]
    
    async def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        # Get query from input
        query = inputs.get("input", inputs.get("question", ""))
        
        # Search relevant memories
        results = await self.engram.search(
            query=query,
            agent_id=self.agent_id,
            user_id=self.user_id,
            limit=5,
        )
        
        memories = [r.memory.content for r in results if r.score > 0.3]
        return {self.memory_key: "\n".join(memories)}
    
    async def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        # Extract user input and AI response
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", outputs.get("response", ""))
        
        # Store the exchange (you could also use LLM to extract facts)
        if user_input:
            await self.engram.add(
                content=f"User said: {user_input}",
                agent_id=self.agent_id,
                user_id=self.user_id,
            )
    
    def clear(self) -> None:
        # Sync wrapper for async purge
        import asyncio
        asyncio.run(self.engram.purge(agent_id=self.agent_id, user_id=self.user_id))

# Usage with LangChain
async def create_chain():
    from langchain.chains import ConversationChain
    from langchain.llms import OpenAI
    
    engram = Engram()
    await engram.connect()
    
    memory = EngramMemory(
        engram=engram,
        agent_id="langchain-agent",
        user_id="user-123",
    )
    
    chain = ConversationChain(
        llm=OpenAI(),
        memory=memory,
    )
    
    return chain
```

---

## Multi-Agent Systems

### Agent-Specific Memories
```python
from engram import Engram

class Agent:
    def __init__(self, agent_id: str, engram: Engram):
        self.agent_id = agent_id
        self.engram = engram
    
    async def remember(self, content: str, user_id: str = None):
        await self.engram.add(
            content=content,
            agent_id=self.agent_id,
            user_id=user_id,
        )
    
    async def recall(self, query: str, user_id: str = None) -> list[str]:
        results = await self.engram.search(
            query=query,
            agent_id=self.agent_id,
            user_id=user_id,
            limit=5,
        )
        return [r.memory.content for r in results]

# Usage
async def main():
    engram = Engram()
    await engram.connect()
    
    # Create specialized agents
    support_agent = Agent("support-agent", engram)
    sales_agent = Agent("sales-agent", engram)
    
    # Each agent has isolated memories
    await support_agent.remember("User reported login issues", user_id="user-1")
    await sales_agent.remember("User interested in premium plan", user_id="user-1")
    
    # Recall only relevant memories
    support_context = await support_agent.recall("login problem", user_id="user-1")
    sales_context = await sales_agent.recall("upgrade interest", user_id="user-1")
```

### Shared Knowledge Base
```python
# Use a shared agent_id for common knowledge
SHARED_KNOWLEDGE_AGENT = "shared-knowledge"

async def add_to_knowledge_base(content: str, category: str):
    await engram.add(
        content=content,
        agent_id=SHARED_KNOWLEDGE_AGENT,
        metadata={"category": category},
    )

async def query_knowledge_base(query: str, category: str = None):
    results = await engram.search(
        query=query,
        agent_id=SHARED_KNOWLEDGE_AGENT,
        limit=10,
    )
    
    if category:
        results = [r for r in results if r.memory.metadata.get("category") == category]
    
    return results
```

---

## Scaling & Performance

### Connection Pooling
```python
# Engram uses asyncpg with connection pooling by default
# Configure pool size via environment:
# ENGRAM_DATABASE_POOL_SIZE=20
# ENGRAM_DATABASE_MAX_OVERFLOW=10
```

### Caching Embeddings
```python
from engram import EmbeddingService

# EmbeddingService has built-in caching
embedding = EmbeddingService.from_provider(
    "openai",
    model="text-embedding-3-small",
    cache_size=10000,  # Cache up to 10k embeddings
)
```

### Batch Operations
```python
# Add multiple memories at once
memories = [
    {"content": "Fact 1", "agent_id": "agent", "user_id": "user"},
    {"content": "Fact 2", "agent_id": "agent", "user_id": "user"},
    {"content": "Fact 3", "agent_id": "agent", "user_id": "user"},
]

await engram.add_batch(memories)
```

### Read Replicas (PostgreSQL)
```python
# Use read replicas for search-heavy workloads
# Primary for writes, replica for reads

WRITE_DB_URL = "postgresql://user:pass@primary:5432/engram"
READ_DB_URL = "postgresql://user:pass@replica:5432/engram"

# Configure in your application layer
```

---

## Best Practices

### 1. **Structured Fact Extraction**
```python
# Always extract atomic facts, not raw messages
# Bad: "User said they like pizza and live in NYC"
# Good: ["User likes pizza", "User lives in NYC"]
```

### 2. **User Data Isolation**
```python
# Always filter by user_id to prevent data leakage
results = await engram.search(
    query=query,
    agent_id=agent_id,
    user_id=user_id,  # Critical for multi-tenant apps
    limit=5,
)
```

### 3. **Memory Importance**
```python
# Reinforce frequently accessed memories
for r in results:
    if r.score > 0.5:
        await engram.reinforce(r.memory.memory_id, delta=0.05)

# Decay old memories (optional cron job)
async def decay_old_memories():
    # Reduce importance of memories not accessed in 30 days
    pass
```

### 4. **Graceful Degradation**
```python
async def get_context_safe(query: str) -> str:
    try:
        results = await engram.search(query=query, ...)
        return "\n".join(r.memory.content for r in results)
    except Exception as e:
        logger.warning(f"Memory retrieval failed: {e}")
        return ""  # Continue without memory context
```

### 5. **Health Checks**
```python
@app.get("/health")
async def health():
    try:
        health = await engram.health_check()
        return health
    except Exception as e:
        raise HTTPException(503, f"Unhealthy: {e}")
```

### 6. **Monitoring**
```python
# Log memory operations for debugging
import logging
logging.getLogger("engram").setLevel(logging.INFO)

# Track metrics
from prometheus_client import Counter, Histogram

memory_ops = Counter("engram_operations", "Memory operations", ["operation"])
search_latency = Histogram("engram_search_latency", "Search latency")

@search_latency.time()
async def search_with_metrics(query: str):
    results = await engram.search(query=query, ...)
    memory_ops.labels(operation="search").inc()
    return results
```

---

## Docker Deployment

### docker-compose.yml
```yaml
version: '3.8'

services:
  app:
    build: .
    environment:
      - ENGRAM_DATABASE_URL=postgresql://engram:engram@db:5432/engram
      - ENGRAM_EMBEDDING_PROVIDER=openai
      - ENGRAM_OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      db:
        condition: service_healthy
    ports:
      - "8000:8000"

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: engram
      POSTGRES_PASSWORD: engram
      POSTGRES_DB: engram
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U engram"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: engram-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: your-app:latest
        env:
        - name: ENGRAM_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: engram-secrets
              key: database-url
        - name: ENGRAM_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: engram-secrets
              key: openai-api-key
```

---

## Summary

| Use Case | Key Components |
|----------|----------------|
| **Chatbot** | `Engram` + `LLMService.extract_facts()` |
| **RAG** | `Engram.search()` + your LLM |
| **Multi-agent** | Multiple `agent_id` values |
| **FastAPI** | Lifespan context + dependency injection |
| **LangChain** | Custom `BaseMemory` subclass |
| **Production** | Connection pooling + health checks + monitoring |

