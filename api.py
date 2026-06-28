import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from ingestion.prompt_ingestor import ingest_prompt
from ingestion.doc_ingestor import ingest_document
from agents.orchestrator import run
from embeddings.embedder import verify_embedder
from vectordb.db_client import get_collection_count

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bidirectional RAG Pipeline",
    description="An LLM with external persistent memory — store and retrieve knowledge.",
    version="1.0.0",
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows any frontend (web app, mobile) to call this API.
# In production you'd restrict this to your specific frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
# Pydantic models define the exact shape of data in and out.
# FastAPI auto-validates requests against these — wrong types = 422 error.

class ChatRequest(BaseModel):
    message   : str
    session_id: str  = None   # None on first message, reused after
    user_id   : str  = "user"

class ChatResponse(BaseModel):
    response  : str
    session_id: str            # always returned so client can reuse it

class HealthResponse(BaseModel):
    status    : str
    db_chunks : int            # how many chunks currently in vector DB
    message   : str


# ── Startup event ─────────────────────────────────────────────────────────────
# Runs once when the server starts — before accepting any requests.
# Verifies embedder is working so we fail fast if something is wrong.
@app.on_event("startup")
async def startup_event():
    print("[api] Starting up RAG pipeline...")
    try:
        verify_embedder()
        count = get_collection_count()
        print(f"[api] Ready — {count} chunks in vector DB")
    except Exception as e:
        print(f"[api] WARNING: Startup check failed: {e}")
        # Don't crash on startup — let health endpoint report the issue


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "name"   : "Bidirectional RAG Pipeline",
        "status" : "running",
        "docs"   : "/docs",    # FastAPI auto-generates docs at /docs
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Render uses this to verify your service is alive.
    Returns current DB chunk count — useful for monitoring.
    """
    try:
        count = get_collection_count()
        return HealthResponse(
            status   ="ok",
            db_chunks=count,
            message  =f"RAG pipeline healthy — {count} chunks stored"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {e}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint — the core of the RAG pipeline.

    Send a message → agent decides to store, retrieve, or both →
    returns grounded response + session_id for next message.

    Request body:
        message    : str  — the user's message
        session_id : str  — None for first message, reuse for conversation
        user_id    : str  — optional user identifier

    Response:
        response   : str  — agent's answer
        session_id : str  — pass this back in your next request
    """

    if not req.message or not req.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # Generate session_id on first message
    session_id = req.session_id or str(uuid.uuid4())

    try:
        # Build prompt envelope — sanitize, detect language, attach history
        envelope = ingest_prompt(req.message, session_id, req.user_id)

        # Run the agent — store, retrieve, or both
        response = run(envelope)

        return ChatResponse(
            response  =response,
            session_id=session_id,
        )

    except ValueError as e:
        # prompt_ingestor raises ValueError for invalid input
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Catch unexpected errors — don't expose internal details
        print(f"[api] ERROR in /chat: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again."
        )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Document ingestion endpoint.
    Upload a PDF or TXT file to store its contents in the vector DB.

    The file is temporarily saved, ingested, then deleted.
    Returns ingestion summary with chunk count.
    """

    # Validate file type
    allowed_types = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_types}"
        )

    # Save uploaded file temporarily
    tmp_path = f"/tmp/{uuid.uuid4()}{ext}"

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run ingestion pipeline
        result = ingest_document(tmp_path)

        # Rename for readable summary
        result["file"] = file.filename
        return result

    except Exception as e:
        print(f"[api] ERROR in /ingest: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {e}"
        )

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/count")
async def count():
    """Returns the number of chunks currently stored in the vector DB."""
    return {"chunks": get_collection_count()}