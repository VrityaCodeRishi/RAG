# Agentic RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot built with LangGraph that can answer questions about personal anime and gaming experiences. The system uses an agentic workflow with multiple tools including vector database retrieval and web search.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)

## Features

- **Agentic RAG Architecture**: Uses LangGraph to create a stateful agent that intelligently routes queries
- **Dual Vector Databases**: Separate ChromaDB collections for anime and game experiences
- **Smart Entity Classification**: Uses web search (Tavily) to identify if a query is about an anime or game
- **Question Type Detection**: Automatically detects general vs. specific questions
- **Beautiful Streamlit UI**: Interactive chat interface with conversation history
- **Contextual Responses**: Generates personalized responses based on stored experiences

## Architecture

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Agent     │────▶│   Tools     │
│ (Question   │     │ (Retriever/ │
│  Routing)   │◀────│  Web Search)│
└──────┬──────┘     └──────┬──────┘
       │                   │
       │            ┌──────┴──────┐
       │            ▼             ▼
       │     ┌───────────┐  ┌───────────┐
       │     │ Classify  │  │  Check    │
       │     │  Entity   │  │  Results  │
       │     └─────┬─────┘  └─────┬─────┘
       │           │              │
       │           ▼              ▼
       │     ┌───────────┐  ┌───────────┐
       │     │  Tools    │  │ Generate/ │
       │     │(Retriever)│  │ Not Found │
       │     └───────────┘  └───────────┘
       │                          │
       ▼                          ▼
┌─────────────────────────────────────┐
│                END                  │
└─────────────────────────────────────┘
```

### Workflow

1. **Agent Node**: Classifies the question as general or specific
   - **General questions** (e.g., "What's your favorite anime?") → Uses both anime and game retrievers
   - **Specific questions** (e.g., "What do you think about Attack on Titan?") → Uses web search first

2. **Tools Node**: Executes the appropriate tools (retrievers or web search)

3. **Classify Entity**: For specific questions, uses web search results to determine if the entity is an anime, game, or both

4. **Generate**: Creates the final response using retrieved context

5. **Not Found**: Handles cases where no relevant information is found

## Project Structure

```
Agentic-RAG/
├── chatbot.py          # Streamlit UI
├── rag.py              # LangGraph agent and workflow
├── create_vector_db.py # Vector database creation script
├── data.json           # Source data (anime & game experiences)
├── requirements.txt    # Python dependencies
├── chroma_db/          # Vector database storage
│   ├── anime/          # Anime experiences collection
│   └── games/          # Game experiences collection
└── README.md           # This file
```

## Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API Key
- Tavily API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Agentic-RAG
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

5. **Create the vector database**
   ```bash
   python create_vector_db.py
   ```

6. **Run the chatbot**
   ```bash
   streamlit run chatbot.py
   ```

7. **Open your browser** and navigate to `http://localhost:8501`

## Example Queries

### General Questions
- "What anime did you watch?"
- "What's your favorite game?"
- "What's your highest rated anime?"
- "What games did you play in 2022?"

### Specific Questions
- "What do you think about Attack on Titan?"
- "Tell me about your experience with Elden Ring"
- "Did you play The Witcher 3?"
- "What's your opinion on Death Note?"

### Edge Cases
- "What do you think about Naruto?" → Correctly says no info available (not in database)

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Agentic workflow orchestration |
| **LangChain** | LLM integration and tool creation |
| **OpenAI GPT-4o** | Language model for classification and generation |
| **ChromaDB** | Vector database for semantic search |
| **Tavily Search** | Web search for entity identification |
| **Streamlit** | Web UI framework |
| **Pydantic** | Data validation and structured outputs |

## Test Results

The chatbot has been tested with 15 different queries:

| Category | Pass Rate |
|----------|-----------|
| General Questions | 80% (4/5) |
| Specific Anime | 100% (3/3) |
| Specific Games | 100% (5/5) |
| Not in Database | 100% (2/2) |
| **Overall** | **93.3%** |

## Configuration

### Modifying the Data

Edit `data.json` to add your own anime and game experiences:

```json
{
  "anime_experiences": [
    {
      "title": "Your Anime Title",
      "experience": "Your experience description...",
      "rating": 9.0,
      "year_watched": 2024,
      "favorite_character": "Character Name",
      "thoughts": "Your thoughts..."
    }
  ],
  "game_experiences": [
    {
      "title": "Your Game Title",
      "experience": "Your experience description...",
      "rating": 9.5,
      "year_played": 2024,
      "platform": "PC",
      "playtime_hours": 50,
      "thoughts": "Your thoughts..."
    }
  ]
}
```

After modifying, re-run the vector database creation:
```bash
python create_vector_db.py
```

### Adjusting Retrieval

In `rag.py`, you can modify the number of documents retrieved:
```python
anime_retriever = anime_vectorstore.as_retriever(search_kwargs={"k": 10})
```


## Author

Anubhav Mandarwal ([Anubhav Mandarwal](https://www.linkedin.com/in/anubhav-mandarwal/))
