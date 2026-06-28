# ── Collection name ───────────────────────────────────────────────────────────
# This is the name of the "table" inside ChromaDB.
# All chunks — whether from user prompts or uploaded documents — go into
# this single collection. One collection, two sources, unified retrieval.
COLLECTION_NAME = "rag_knowledge"


# ── Metadata fields ───────────────────────────────────────────────────────────
# Every chunk stored in the DB carries these metadata fields alongside
# its vector and raw text. Metadata lets you filter searches later —
# e.g. "only retrieve chunks from documents, not from user prompts".
#
# Field breakdown:
#   source      → where the chunk came from: "user_prompt" or "document"
#   session_id  → which chat session produced this chunk (for user_prompt source)
#   timestamp   → when the chunk was stored (ISO format string)
#   chunk_index → position of this chunk within its original text (0-based)
#   file_name   → original filename (only set when source is "document")

METADATA_FIELDS = [
    "source",
    "session_id",
    "timestamp",
    "chunk_index",
    "file_name",
]


# ── Source type constants ──────────────────────────────────────────────────────
# Use these instead of raw strings when setting the "source" metadata field.
# Prevents bugs from typos like "user_prompts" vs "user_prompt".
SOURCE_USER_PROMPT = "user_prompt"
SOURCE_DOCUMENT    = "document"


# ── Distance metric ───────────────────────────────────────────────────────────
# ChromaDB supports "cosine", "l2", and "ip" (inner product).
# Cosine similarity is best for text — it measures the angle between vectors,
# not their magnitude, so chunk length doesn't affect similarity scores.
DISTANCE_METRIC = "cosine"