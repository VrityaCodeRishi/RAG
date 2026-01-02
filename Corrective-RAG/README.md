## Corrective RAG (Recipe Assistant)

### What this project is
This folder contains a **Corrective RAG** implementation for a **recipe + cooking assistant**:

- **Retrieval** from a local **Chroma** vector DB built from scraped cooking/recipe pages
- **Relevance grading** of retrieved documents
- **Query rewriting** (to improve web search)
- **Web search fallback** (Tavily) when retrieved docs are missing or not relevant
- **Answer generation** using retrieved context (vector DB + optional web results)

### Folder contents
- `create_recipe_vector_db.py`: scrapes URLs and builds the Chroma DB at `./chroma_db/recipes`
- `rag.py`: LangGraph workflow (retrieve → grade → decide → transform/web_search → generate)
- `chatbot.py`: Streamlit UI that imports and uses `builder` from `rag.py`
- `requirements.txt`: dependencies for this Corrective RAG project
- `chroma_db/recipes/`: persisted Chroma database (ignored by `.gitignore`)

### Prerequisites
- Python 3.12 (recommended; matches your local setup)
- API keys in `Corrective-RAG/.env`:

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

### Install dependencies
From repo root:

```bash
cd /Users/curious_techie/Desktop/RAG
source venv/bin/activate
pip install -r Corrective-RAG/requirements.txt
```

If web scraping fails with `No module named 'bs4'`, install:

```bash
pip install beautifulsoup4
```

### 1) Build / refresh the vector database
Run this from the `Corrective-RAG/` folder (important because paths are relative):

```bash
cd /Users/curious_techie/Desktop/RAG/Corrective-RAG
python create_recipe_vector_db.py
```

This creates/updates:
- `./chroma_db/recipes/chroma.sqlite3`
- `./chroma_db/recipes/<collection_id>/*`

### 2) Run the Corrective RAG workflow (CLI)

```bash
cd /Users/curious_techie/Desktop/RAG/Corrective-RAG
python rag.py
```

You’ll see the node-by-node execution path printed:
- `NODE: RETRIEVE`
- `NODE: GRADE DOCUMENTS`
- `NODE: DECIDE TO GENERATE`
- `NODE: TRANSFORM QUERY` (only if needed)
- `NODE: WEB SEARCH` (only if needed)
- `NODE: GENERATE`

### 3) Start the chatbot (Streamlit UI)

```bash
cd /Users/curious_techie/Desktop/RAG/Corrective-RAG
streamlit run chatbot.py
```

### How the Corrective RAG flow works
Given a user question:

- **Retrieve**: fetch top-k documents from Chroma
- **Grade**: LLM filters docs as relevant/not relevant
- **Decide**:
  - If **all docs relevant** → **Generate** using vector DB context
  - If **any doc not relevant** OR **no docs retrieved** → rewrite query → **Web search** → **Generate** using (filtered docs + web results)

### Architecture diagram

```mermaid
flowchart TD
  U[User Question] --> R[Retrieve (Chroma Vector DB)]
  R --> G[Grade Documents (LLM)]
  G --> D{Decision}

  D -->|All relevant| GEN[Generate Answer (LLM)]
  D -->|Missing/Not relevant| T[Transform Query (LLM)]
  T --> W[Web Search (Tavily)]
  W --> GEN

  subgraph Storage
    C[(Chroma DB: ./chroma_db/recipes)]
  end

  R --- C
  GEN --> OUT[Final Answer]
```


## Author

Anubhav Mandarwal ([Anubhav Mandarwal](https://www.linkedin.com/in/anubhav-mandarwal/))


