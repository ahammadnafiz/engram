# Engram Two-Column Memory System Diagrams

## 1. ADD Flow - How Memory is Stored

```mermaid
flowchart TD
    subgraph Input["📝 Input"]
        Content["content='User likes coffee'"]
        MainContent["main_content='[USER]: I love coffee\n[AI]: Great taste!'"]
    end

    subgraph Embedding["🔢 Embedding (Cost)"]
        EmbedService["EmbeddingService.embed()"]
        Vector["Vector: [0.12, -0.34, 0.56, ...]"]
    end

    subgraph NoEmbed["💰 No Embedding (Free)"]
        Skip["Stored directly - NO API call"]
    end

    subgraph Database["🗄️ PostgreSQL"]
        FactCol["fact = 'User likes coffee'"]
        FactTSV["fact_tsv = auto-generated tsvector"]
        EmbedCol["embedding = [0.12, -0.34, ...]"]
        MainCol["main_content = '[USER]: I love...'"]
        MainTSV["main_content_tsv = auto-generated"]
    end

    Content -->|"$ Embed"| EmbedService
    EmbedService --> Vector
    Vector --> EmbedCol
    Content --> FactCol
    FactCol --> FactTSV

    MainContent -->|"FREE"| Skip
    Skip --> MainCol
    MainCol --> MainTSV

    style EmbedService fill:#ff9999
    style Skip fill:#99ff99
    style Vector fill:#ff9999
    style MainCol fill:#99ff99
```

## 2. SEARCH Flow - How Memory is Retrieved

```mermaid
flowchart TD
    subgraph Query["🔍 Search Query"]
        Q["query = 'coffee preferences'"]
    end

    subgraph Embed["🔢 Query Embedding"]
        QEmbed["EmbeddingService.embed(query)"]
        QVector["query_vector"]
    end

    subgraph HybridSearch["⚡ Hybrid Search (on FACT only)"]
        Semantic["Semantic Search\nembedding <=> query_vector"]
        Keyword["Keyword Search\nfact_tsv @@ query"]
        RRF["RRF Fusion + Decay + Importance"]
    end

    subgraph NotSearched["🚫 NOT Searched"]
        MainContentCol["main_content\nmain_content_tsv"]
    end

    subgraph Return["📤 Return BOTH Columns"]
        Fact["fact: 'User likes coffee'"]
        Main["main_content: '[USER]: I love coffee...'"]
        Score["score: 0.87"]
    end

    Q --> QEmbed
    QEmbed --> QVector
    QVector --> Semantic
    Q --> Keyword
    Semantic --> RRF
    Keyword --> RRF
    
    RRF -->|"Matched!"| Fact
    RRF -->|"Also return"| Main
    RRF --> Score

    MainContentCol -.->|"Not used for matching"| RRF

    style Semantic fill:#99ccff
    style Keyword fill:#99ccff
    style MainContentCol fill:#cccccc
    style Fact fill:#99ff99
    style Main fill:#99ff99
```

## 3. Complete Two-Column System

```mermaid
flowchart TB
    subgraph ADD["➕ engram.add()"]
        direction TB
        A1["content='User likes coffee'"]
        A2["main_content='[USER]: ...'"]
        
        A3["Embed content ✓"]
        A4["Skip main_content ✗"]
        
        A5[("PostgreSQL\n\nfact + embedding\nmain_content")]
    end

    subgraph SEARCH["🔍 engram.search()"]
        direction TB
        S1["query='coffee'"]
        S2["Embed query"]
        
        S3["Search on:\n• fact embedding\n• fact_tsv"]
        S4["NOT searched:\n• main_content"]
        
        S5["Return:\n• fact ✓\n• main_content ✓\n• score ✓"]
    end

    A1 --> A3
    A2 --> A4
    A3 --> A5
    A4 --> A5

    S1 --> S2
    S2 --> S3
    S3 --> S5
    S4 -.-> S5

    A5 -->|"Stored"| S3

    style A3 fill:#ff9999
    style A4 fill:#99ff99
    style S3 fill:#99ccff
    style S4 fill:#cccccc
    style S5 fill:#99ff99
```

## 4. Database Schema

```mermaid
erDiagram
    agent_memory {
        text memory_id PK
        text agent_id FK
        text user_id FK
        
        text fact "Extracted fact (EMBEDDED)"
        text main_content "Context (NOT embedded)"
        text content "Alias for fact"
        
        vector embedding "Vector of fact only"
        tsvector fact_tsv "Auto-generated"
        tsvector main_content_tsv "Auto-generated"
        
        float importance "0.0-1.0"
        int access_count
        timestamp created_at
        timestamp last_accessed_at
        jsonb metadata
    }
```

## 5. Cost Comparison

```mermaid
pie title Embedding API Costs
    "Fact (embedded)" : 20
    "Main Content (FREE)" : 80
```

## 6. Search vs Storage

```mermaid
quadrantChart
    title Two-Column Strategy
    x-axis Not Searched --> Searched
    y-axis Not Embedded --> Embedded
    quadrant-1 "❌ Never (waste)"
    quadrant-2 "✅ fact column"
    quadrant-3 "💰 main_content"
    quadrant-4 "❌ Never (expensive)"
    
    fact: [0.85, 0.85]
    main_content: [0.15, 0.15]
```

## 7. Chatbot Memory Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as Chatbot
    participant L as LLMService
    participant E as Engram
    participant DB as PostgreSQL

    U->>C: "I live near UIU in Dhaka"
    C->>L: extract_facts(user_msg, bot_msg)
    L-->>C: ["User lives near UIU in Dhaka"]
    
    C->>L: summarize(bot_msg)
    L-->>C: "Convenient for studies"
    
    Note over C: Build main_content:<br/>[USER]: I live near UIU...<br/>[AI]: Convenient for studies
    
    C->>E: add(content=fact, main_content=context)
    E->>E: embed(fact) only
    E->>DB: INSERT fact, main_content, embedding
    
    Note over DB: Stored:<br/>fact: "User lives near UIU in Dhaka"<br/>main_content: "[USER]: I live...[AI]: Convenient..."<br/>embedding: [0.12, -0.34, ...]

    U->>C: "Where do I live?"
    C->>E: search("Where do I live?")
    E->>E: embed(query)
    E->>DB: Hybrid search on fact + fact_tsv
    DB-->>E: fact + main_content + score
    E-->>C: SearchResult(fact, main_content, score=0.85)
    
    C->>L: complete_full(memories + query)
    L-->>C: "You live near UIU in Dhaka!"
    C-->>U: "You live near UIU in Dhaka!"
```

## Summary Table

| Column | Stored | Embedded | Searched | Returned | Cost |
|--------|--------|----------|----------|----------|------|
| `fact` | ✅ | ✅ | ✅ | ✅ | $ |
| `main_content` | ✅ | ❌ | ❌ | ✅ | Free |
| `embedding` | ✅ | - | ✅ | ❌ | - |
| `fact_tsv` | ✅ | - | ✅ | ❌ | - |
