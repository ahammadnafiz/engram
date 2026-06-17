# Quickstart: End-to-End Memory Loop

This guide will walk you through building a complete, memory-aware autonomous agent loop from scratch. 

By the end of this 10-minute guide, you will have a running PostgreSQL database, a configured Engram client, an active task ledger, and you will understand how to intelligently retrieve facts using the cognitive `recall` operator.

---

## 1. Prerequisites & Installation

You need Python 3.10+ and Docker installed on your machine.

First, clone the repository and install the library. We will install the `sentence-transformers` extra so we can run embedding models entirely locally without needing an API key.

```bash
git clone https://github.com/ahammadnafiz/engram.git
cd engram
pip install -e ".[dev,examples,sentence-transformers]"
```

> [!TIP]
> If you prefer to use OpenAI for both embeddings and LLM extraction, you can instead run `pip install -e ".[openai]"`.

---

## 2. Boot the Database

Engram requires PostgreSQL equipped with the `pgvector` and `pg_trgm` extensions. The easiest way to get this running locally is using the provided Docker compose file.

```bash
docker compose up -d postgres
```

Verify it is running:
```bash
docker compose ps postgres
```

---

## 3. Configure the Environment

Create a `.env` file in the root of your project or export these variables directly. We will point Engram to our local Docker database and instruct it to use local embeddings.

```bash
export ENGRAM_DATABASE_URL=postgresql://engram:engram_secret@localhost:5432/engram
export ENGRAM_EMBEDDING_PROVIDER=sentence-transformers
export ENGRAM_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

> [!NOTE]
> Engram handles database migrations automatically. The very first time your code calls `connect()`, Engram will securely create all the necessary extensions, tables, vector columns, and indexes.

---

## 4. Write Your First Agent Loop

Create a new file called `quickstart.py` and add the following code. We'll build the script step-by-step.

### Step A: Initialize the Client
We use the asynchronous context manager (`async with`) which automatically handles connection pooling and cleanup.

```py
import asyncio
from engram import Engram

async def main():
    # Engram will automatically create the database schema on boot
    async with Engram(memory_policy="default") as engram:
        
        health = await engram.health_check()
        print(f"✅ Engram Booted. Status: {health['status']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step B: The Task Ledger
If your agent is having a conversation or performing a long-running job, you should wrap the execution in a **Task**. This provides an immutable ledger of everything that happened.

Add this inside the `async with` block:

```py
        # 1. Start a long-running task
        task = await engram.start_task(
            goal="Help the user plan their vacation",
            agent_id="travel_assistant",
            user_id="alice_123"
        )
        print(f"📂 Created Task Run: {task.task_run_id}")
```

### Step C: Storing Critical Facts
Sometimes you learn something so important that you don't want to rely on background extraction—you want to store it immediately.

```py
        # 2. Explicitly store a critical constraint
        memory = await engram.add(
            fact="User is severely allergic to peanuts.",
            agent_id="travel_assistant",
            user_id="alice_123",
            main_content="[USER]: Please ensure my flights don't serve peanuts, I am highly allergic.",
            metadata={"critical": True, "critical_slot": "health:allergy"}
        )
        print(f"🧠 Stored Fact: {memory.fact}")
```

### Step D: Recording Conversational Turns
As your agent talks to the user, you should record the raw exchanges to the task ledger.

```py
        # 3. Record a conversational turn to the ledger
        user_message = "I want to visit Tokyo next Spring."
        assistant_reply = "Tokyo is beautiful in the Spring! I will start planning."
        
        await engram.record_turn(
            task_run_id=task.task_run_id,
            user_message=user_message,
            assistant_response=assistant_reply,
            agent_id="travel_assistant",
            user_id="alice_123",
            enqueue_processing=True  # Queues this turn for background extraction
        )
        print(f"📝 Recorded Turn to Ledger")
```

### Step E: Intelligent Recall
Now, imagine it is three days later. The user comes back and asks a question. Instead of manually writing vector search queries, we use the `recall()` cognitive operator.

*Note: The `recall()` operator requires an LLM to classify intent. If you don't have an LLM configured, it gracefully falls back to a standard hybrid search.*

```py
        # 4. Recall context intelligently based on the user's intent
        new_query = "What was that city I wanted to visit, and do I have any dietary restrictions?"
        
        trace = await engram.recall(
            query=new_query,
            agent_id="travel_assistant",
            user_id="alice_123",
            compose_answer=False  # Returns the raw prompt context, not an LLM auto-reply
        )
        
        print("\n🔍 Recall Trace Context Block:")
        print("-------------------------------")
        print(trace.context)
        print("-------------------------------")
```

---

## 5. Run the Script

Execute the file you just wrote:

```bash
python quickstart.py
```

### Expected Output
You should see output similar to this:
```text
✅ Engram Booted. Status: healthy
📂 Created Task Run: task_01J...
🧠 Stored Fact: User is severely allergic to peanuts.
📝 Recorded Turn to Ledger

🔍 Recall Trace Context Block:
-------------------------------
User is severely allergic to peanuts.
Source: [USER]: Please ensure my flights don't serve peanuts, I am highly allergic.
...
-------------------------------
```

> [!NOTE]  
> Notice that the context block automatically retrieved the peanut allergy constraint, even though the query (`"dietary restrictions"`) didn't explicitly use the word "allergy" or "peanuts". This is the power of hybrid vector retrieval.

---

## 6. Next Steps

You've successfully built a complete memory loop! Where to go from here?

- **[Concepts](concepts.md)**: Deep dive into how Lineages, Conflict Resolution, and Task Ledgers actually work under the hood.
- **[Architecture](architecture.md)**: Explore the visual flowcharts of how Engram routes data.
- **[API Reference](api-reference.md)**: View the exact method signatures and parameters for every tool you used today.
- **[Production Guide](production-guide.md)**: Learn how to split the `record_turn()` extraction into a separate background worker process for massive scale.
