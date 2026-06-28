from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

# ── Embedding model instance ──────────────────────────────────────────────────
# nomic-embed-text is a dedicated embedding model — its only job is to
# convert text into vectors. It is NOT a chat model.
#
# Key properties of nomic-embed-text:
#   - Produces 768-dimensional vectors
#   - Max input: ~512 tokens (~380 words)
#   - Optimised for retrieval tasks (not generation)
#   - Runs locally via Ollama — no API key needed
#
# We create ONE instance here and reuse it everywhere.
# Creating a new OllamaEmbeddings object on every call wastes time
# re-establishing the connection to Ollama.
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)


# ── STORE PATH ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text chunks into vectors for storage in the DB.
    Called by store_tool and doc_ingestor before calling upsert_chunks().

    Args:
        texts: List of chunk strings (output of chunker.py)

    Returns:
        List of vectors. Each vector is a list of 768 floats.
        Order is preserved — texts[i] corresponds to vectors[i].

    Example:
        chunks = ["The Eiffel Tower is in Paris.", "It was built in 1889."]
        vectors = embed_texts(chunks)
        # vectors[0] is the embedding of the first chunk
        # vectors[1] is the embedding of the second chunk
    """

    if not texts:
        return []

    # Filter out empty strings before sending to Ollama.
    # An empty string would produce a meaningless zero vector
    # that could match anything in the DB.
    clean_texts = [t.strip() for t in texts if t.strip()]

    if not clean_texts:
        return []

    # embed_documents() handles batching internally —
    # you can pass 1 or 1000 chunks and it manages the calls to Ollama.
    vectors = embedder.embed_documents(clean_texts)

    return vectors


# ── RETRIEVE PATH ─────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Convert a single search query into a vector for similarity search.
    Called by retrieve_tool before calling query_chunks().

    This uses embed_query() instead of embed_documents() — a subtle but
    important distinction. Some embedding models (including nomic-embed-text)
    use slightly different internal representations for queries vs documents,
    optimised so that query vectors point toward relevant document vectors
    even when the wording is different.

    Args:
        query: The user's question or search string (clean text from envelope)

    Returns:
        A single vector — list of 768 floats.

    Example:
        vec = embed_query("Where is the Eiffel Tower?")
        docs, metas = query_chunks(vec, top_k=3)
        # Returns chunks most similar to the query vector
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty. Cannot embed a blank string.")

    vector = embedder.embed_query(query.strip())

    return vector


# ── UTILS ─────────────────────────────────────────────────────────────────────

def get_embedding_dimension() -> int:
    """
    Returns the dimension of vectors produced by the embedding model.
    nomic-embed-text produces 768-dimensional vectors.

    Useful for debugging — if your vectors are a different size than
    expected, something is wrong with the model or Ollama setup.
    """

    # Embed a test string and check the length of the resulting vector
    test_vector = embedder.embed_query("test")
    return len(test_vector)


def verify_embedder() -> bool:
    try:
        vec = embedder.embed_query("hello")
        assert len(vec) > 0
        print(f"[embedder] OK — model=all-MiniLM-L6-v2, dim={len(vec)}")
        return True
    except Exception as e:
        raise RuntimeError(f"HuggingFace embedder failed: {e}")