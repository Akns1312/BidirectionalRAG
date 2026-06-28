# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.12 slim — minimal Ubuntu-based image
# "slim" removes unnecessary packages keeping image size small
FROM python:3.12-slim

# ── Working directory ─────────────────────────────────────────────────────────
# All commands after this run inside /app
# All files are copied into /app
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential — needed to compile some Python packages (e.g. chromadb)
# curl            — useful for health checks inside container
# We clean up apt cache immediately to keep image size small
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first — separate layer from code.
# Docker caches layers — if requirements.txt hasn't changed,
# this layer is reused on rebuild (much faster builds).
# If you copy all code first, any code change rebuilds this slow layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── spaCy model ───────────────────────────────────────────────────────────────
# Presidio (PII guard) needs this NLP model.
# Downloaded once during build — baked into image permanently.
RUN python -m spacy download en_core_web_lg

# ── HuggingFace embedding model ───────────────────────────────────────────────
# Pre-download the MiniLM embedding model during build.
# Without this, the model downloads on first request — causing a
# 30-60 second delay and potential timeout on Render's health check.
# Baking it into the image means instant startup every time.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Embedding model downloaded successfully')"

# ── Application code ──────────────────────────────────────────────────────────
# Copy code AFTER installing dependencies.
# This way code changes don't invalidate the dependency cache layer.
# Order matters: slow layers first, fast-changing layers last.
COPY . .

# ── ChromaDB directory ────────────────────────────────────────────────────────
# Create the directory where ChromaDB will persist vector data.
# On Render free tier the filesystem resets on restart —
# data won't persist between deploys, but it works during a session.
RUN mkdir -p /app/chroma_store

# ── Port ──────────────────────────────────────────────────────────────────────
# Tell Docker this container listens on port 8000.
# Render reads this to route traffic to the right port.
EXPOSE 8000

# ── Startup command ───────────────────────────────────────────────────────────
# uvicorn    — the ASGI server that runs FastAPI
# api:app    — file: api.py, FastAPI instance: app
# --host 0.0.0.0  — listen on all network interfaces (required for Docker)
# --port 8000     — port to listen on
# --workers 1     — single worker (Render free tier has limited RAM)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]