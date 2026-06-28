import uuid
from datetime import datetime

import chromadb

from config.settings import settings
from vectordb.schema import (
    COLLECTION_NAME,
    DISTANCE_METRIC,
    SOURCE_DOCUMENT,
    SOURCE_USER_PROMPT,
)

# ── ChromaDB client ───────────────────────────────────────────────────────────
# PersistentClient saves the DB to disk at CHROMA_PATH (set in .env).
# Data survives restarts — unlike the in-memory client which resets every run.
# On first run: creates the folder and an empty collection.
# On subsequent runs: loads existing data from disk automatically.
client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

# ── Collection ────────────────────────────────────────────────────────────────
# get_or_create_collection is safe to call every startup —
# it creates the collection if it doesn't exist, otherwise loads it.
# hnsw:space sets the distance metric for similarity search.
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": DISTANCE_METRIC},
)


# ── WRITE ─────────────────────────────────────────────────────────────────────

def upsert_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> int:
    """
    Store chunks with their embeddings and metadata in the vector DB.
    Uses upsert (not insert) — safe to call multiple times with the same
    content without creating duplicates, because each ID is freshly generated.

    Args:
        chunks     : Raw text of each chunk
        embeddings : Vector for each chunk (from embedder.py)
        metadatas  : Metadata dict for each chunk (source, session_id, etc.)

    Returns:
        Number of chunks successfully stored.
    """

    if not chunks:
        return 0

    # Validate all three lists are the same length.
    # A mismatch here means a bug upstream in the calling code.
    if not (len(chunks) == len(embeddings) == len(metadatas)):
        raise ValueError(
            f"Length mismatch: chunks={len(chunks)}, "
            f"embeddings={len(embeddings)}, "
            f"metadatas={len(metadatas)}"
        )

    # Generate a unique ID for each chunk.
    # UUID4 is random — no two chunks will ever collide even across sessions.
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Ensure every metadata dict has a timestamp.
    # If the caller forgot to set one, we add it here as a safety net.
    for meta in metadatas:
        if "timestamp" not in meta:
            meta["timestamp"] = datetime.utcnow().isoformat()

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


# ── READ ──────────────────────────────────────────────────────────────────────

def query_chunks(
    query_embedding: list[float],
    top_k: int = None,
    source_filter: str = None,
) -> tuple[list[str], list[dict]]:
    """
    Search the vector DB for chunks similar to the query embedding.

    Args:
        query_embedding : Vector of the user's question (from embedder.py)
        top_k           : Number of results to return (defaults to settings.TOP_K)
        source_filter   : Optional — filter by source type.
                          Pass SOURCE_USER_PROMPT or SOURCE_DOCUMENT to
                          restrict results to one source only.
                          Pass None to search across everything (recommended).

    Returns:
        Tuple of (list of chunk texts, list of metadata dicts)
        Both lists are ordered by similarity — most relevant first.
    """

    if top_k is None:
        top_k = settings.TOP_K

    # ── Build optional where filter ───────────────────────────────────────────
    # ChromaDB's where parameter filters by metadata before doing vector search.
    # Only apply it when the caller explicitly requests a specific source.
    where = None
    if source_filter in (SOURCE_USER_PROMPT, SOURCE_DOCUMENT):
        where = {"source": {"$eq": source_filter}}

    # ── Run similarity search ─────────────────────────────────────────────────
    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # ── Unpack results ────────────────────────────────────────────────────────
    # ChromaDB returns results nested in lists (one per query embedding).
    # Since we always send exactly one query embedding, we take index [0].
    documents = results["documents"][0]   # list of chunk texts
    metadatas = results["metadatas"][0]   # list of metadata dicts

    return documents, metadatas


# ── UTILS ─────────────────────────────────────────────────────────────────────

def get_collection_count() -> int:
    """
    Returns the total number of chunks currently stored in the DB.
    Useful for debugging and checking if ingestion worked.
    """
    return collection.count()


def delete_by_session(session_id: str) -> None:
    """
    Delete all chunks that were stored from a specific session.
    Useful for clearing a user's contributed knowledge if needed.
    """
    collection.delete(
        where={"session_id": {"$eq": session_id}}
    )