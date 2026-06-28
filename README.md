# Bidirectional RAG Pipeline

An agentic RAG system that gives a stateless LLM persistent external memory.
The LLM can both **store** new knowledge from user prompts and **retrieve** it in future conversations via a vector database.

## Architecture
- **LLM**: Groq (llama3) — reasoning and response generation
- **Embeddings**: HuggingFace MiniLM — local, no API cost
- **Vector DB**: ChromaDB — persistent knowledge store
- **Agent**: LangGraph ReAct loop with store, retrieve, summarize tools
- **API**: FastAPI — HTTP interface for deployment
- **Deployment**: Docker + Render

## Features
- Bidirectional RAG — store AND retrieve from vector DB
- Self-healing retrieval — HyDE + query rewriting fallback
- PII protection — Presidio masks sensitive data before storage
- Session memory — conversation history across turns
- Document ingestion — upload PDFs to knowledge base

## Setup
1. Clone the repo
2. Create `.env` with `GROQ_API_KEY`
3. Install deps: `pip install -r requirements.txt`
4. Run locally: `python main.py`
5. Run with Docker: `docker compose up`

## API Endpoints
- `POST /chat` — send a message to the RAG agent
- `POST /ingest` — upload a document to the knowledge base
- `GET /health` — health check
- `GET /count` — number of chunks in vector DB
- `GET /docs` — interactive API documentation