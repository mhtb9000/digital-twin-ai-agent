# Architecture

```mermaid
flowchart TD
    U[User] --> S[Streamlit UI]
    S --> P[Prompt Builder]
    P --> R[RAG Retriever]
    P --> M[Short-Term Memory]
    P --> L[Long-Term Memory]
    R --> G[Gemini 2.5 Flash]
    M --> G
    L --> G
    G --> S

    subgraph Corpus
        A[Public text docs]
        B[Chunking]
        C[Embeddings]
        D[ChromaDB]
        A --> B --> C --> D
    end

    R --> D
    L --> D
```

## Flow

1. The user asks a question in Streamlit.
2. The app retrieves relevant document chunks from ChromaDB.
3. The app retrieves relevant durable memories for the same user.
4. The app builds a persona-rich prompt.
5. Gemini generates the response.
6. New durable memories are extracted and stored.
