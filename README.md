# Nepali RAG Assistant

A Retrieval-Augmented Generation (RAG) system for Nepali legal documents with bilingual support (Nepali and English).

This project leverages advanced NLP techniques to provide intelligent, context-aware answers from legal documents using vector similarity search and a multilingual LLM. Built for the Nepal Company Act 2063 documentation.

## Features

- Bilingual Query Support: Ask questions in Nepali or English
- Hybrid Retrieval: Combines BM25 (keyword-based) + Vector (semantic) search for optimal results
- Multi-stage Ranking: Uses FlagEmbedding reranker for result refinement
- Hierarchical Chunking: Parent-child chunk relationships for contextual awareness
- Date Normalization: Handles Nepali calendar (Bikram Sambat) date conversions
- REST API Backend: FastAPI with CORS support for extensibility
- Modern Web UI: Streamlit frontend with responsive design
- Production-Ready: Docker containerization, health checks, persistent storage

## Tech Stack

### Backend
- FastAPI - High-performance REST API framework
- Google Gemini - LLM for context-aware responses
- Chroma - Vector database for semantic search
- LangChain - LLM orchestration and prompt management
- FlagEmbedding - Multi-lingual embeddings and reranking
- BM25 - Probabilistic relevance ranking (rank-bm25)

### Frontend
- Streamlit - Interactive web application
- Python Requests - Backend API communication

### Infrastructure
- Docker & Docker Compose - Container orchestration
- Python 3.12 - Runtime

## Project Structure

```
omnidata-rag/
├── app/
│   ├── backend/
│   │   └── main.py              # FastAPI application
│   └── frontend/
│       └── streamlit_app.py     # Streamlit UI
│
├── rag/                          # RAG Pipeline & LLM Logic
│   ├── pipeline.py               # Main RAG query engine
│   ├── retrieval.py              # BM25 + Vector hybrid search
│   ├── embeddings.py             # Embedding model initialization
│   ├── ingest.py                 # Database population script
│   ├── prompts.py                # Prompt templates
│   └── language.py               # Language detection & BS date normalization
│
├── evaluation/                   # Evaluation metrics
│
├── scripts/
│   ├── scrape.py                 # Web scraping utility
│   └── clear_chroma.py           # Vector DB reset utility
│
├── data/
│   └── company_act_ne.txt        # Nepal Company Act 2063 document
│
├── docker/
│   ├── docker-compose.yml        # Multi-container orchestration
│   ├── backend.Dockerfile        # Backend container image
│   └── frontend.Dockerfile       # Frontend container image
│
├── docs/
│   └── RUN.md                    # Detailed run/troubleshooting guide
│
├── chroma_db/                    # Vector database storage (generated, gitignored)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (for embedding models)
- Internet connection (for Google Gemini API)

### Installation & Running

```bash
# 1. Clone the repository
git clone <repo-url>
cd omnidata-rag

# 2. Setup environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY from https://aistudio.google.com/app/apikey

# 3. Initialize the vector database (first time only)
python -m rag.ingest

# 4. Start all services with Docker
docker compose -f docker/docker-compose.yml up

# 5. Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8080
# API Docs: http://localhost:8080/docs
```

## Configuration

### Environment Variables (.env)

```bash
# Backend Configuration
BACKEND_URL=http://backend:8080          # Used by frontend (Docker)

# Google Gemini API Key (Required)
GEMINI_API_KEY=your_api_key_here

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

Get your API key: https://aistudio.google.com/app/apikey

## API Documentation

### Endpoints

#### Health Check
```
GET /health
```
Response: `{"status": "ok"}`

#### Chat Endpoint
```
POST /chat
Content-Type: application/json

{
  "message": "What is the Nepal Company Act?"
}
```

Response:
```json
{
  "reply": "The Nepal Company Act 2063 is...",
  "sources": [
    {"id": "child_...", "source_type": "txt", "title": "...", "text": "..."}
  ],
  "status": "success"
}
```

## Development

### Local Development (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m rag.ingest

# Run backend
uvicorn app.backend.main:app --reload --port 8080

# In another terminal, run frontend
streamlit run app/frontend/streamlit_app.py
```

### Running Evaluation

```bash
python -m evaluation.retrieval_metrics
python -m evaluation.visualize
```

### Running Tests

Unit tests cover the pure logic (chunking, language/date handling, hybrid
retrieval scoring, backend intent routing) and don't require a GEMINI_API_KEY
or downloaded models, since those are now lazily loaded.

```bash
pip install -r requirements-dev.txt
pytest
```

## Security Notes

- API Keys: Never commit `.env` files to Git. Use `.env.example` as template.
- CORS: Backend CORS is configured for localhost. Update for production.
- Rate Limiting: Not implemented. Add before production deployment.

## System Architecture

```
User Query (Streamlit)
    ↓
[Frontend] → HTTP POST → [Backend API]
    ↓
Hybrid Retrieval (BM25 + Vector)
    ↓
Multilingual Reranking
    ↓
Context Building
    ↓
LLM Generation (Gemini)
    ↓
Response → [Frontend Display]
```

## License

This project is developed for the Nepal Company Act documentation system.

## Support

For issues or questions, please refer to the [Run.md](Run.md) troubleshooting section or create an issue in the repository.
