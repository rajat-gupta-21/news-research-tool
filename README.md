# news-research-tool
          ┌────────────────────────────────────────────┐
          │              User Inputs URLs              │
          │  (e.g. BBC / Reuters tech news pages)      │
          └────────────────────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  1. WebBaseLoader           │
               │  → Downloads article text   │
               │  → Extracts clean content   │
               └─────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  2. RecursiveCharacterTextSplitter │
               │  → Breaks content into ~1000-char  │
               │    chunks with overlap            │
               └─────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  3. Gemini Embeddings       │
               │  → Calls Google GenAI API   │
               │  → Generates vector for each│
               │    text chunk               │
               └─────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  4. FAISS Vector Store      │
               │  → Stores all embeddings    │
               │    with original text       │
               │  → Saves locally to disk    │
               └─────────────────────────────┘
                              │
                              ▼
      ┌──────────────────────────────────────────────────────┐
      │              USER ASKS A QUESTION                    │
      └──────────────────────────────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  5. Query Embedding         │
               │  → Gemini Embeddings converts│
               │    user question into vector│
               └─────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  6. FAISS Similarity Search │
               │  → Finds top-K similar chunks│
               │    to the query              │
               └─────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  7. Context Construction    │
               │  → Combine top documents    │
               │    into one large context   │
               └─────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  8. Gemini LLM Call         │
               │  → Prompt = context + query │
               │  → model: gemini-2.5-flash  │
               │  → Returns natural-language │
               │    answer                   │
               └─────────────────────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │  9. Output Formatter        │
               │  → Print answer + sources   │
               └─────────────────────────────┘
