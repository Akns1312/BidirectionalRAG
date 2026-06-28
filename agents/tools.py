from datetime import datetime

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from config.settings import settings
from embeddings.chunker import chunk_text
from embeddings.embedder import embed_texts, embed_query
from vectordb.db_client import upsert_chunks, query_chunks
from vectordb.schema import SOURCE_USER_PROMPT
from agents.prompts import EXTRACTOR_SYSTEM, SUMMARIZER_SYSTEM

# ── LLM instance ──────────────────────────────────────────────────────────────
_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.LLM_MODEL,
    temperature=0.1,
)


# ── Self-healing helper functions ─────────────────────────────────────────────
# These are NOT tools — they are internal helpers called by retrieve_tool only.
# No @tool decorator — the agent never calls these directly.

def _format_chunks(docs: list[str], metas: list[dict]) -> str:
    """Format retrieved chunks with source labels for the agent to read."""
    formatted = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        source = meta.get("file_name") or meta.get("source", "unknown")
        formatted.append(f"[Chunk {i+1} — Source: {source}]\n{doc.strip()}")
    return "\n\n---\n\n".join(formatted)


def _check_relevance(query: str, chunks: list[str]) -> bool:
    """
    Ask the LLM if retrieved chunks actually answer the query.
    Returns True if relevant, False if retrieval failed.
    """
    if not chunks:
        return False

    context = "\n\n".join(chunks[:2])  # check top 2 chunks only

    prompt = f"""Does the following context contain information that answers this question?
Answer with only YES or NO.

Question: {query}
Context: {context}"""

    try:
        result = _llm.invoke(prompt).content.strip().upper()
        return result.startswith("YES")
    except Exception:
        return False  # if check fails, assume not relevant


def _hyde_query(query: str) -> list[float]:
    """
    HyDE — Hypothetical Document Embedding.
    Generate a hypothetical answer and embed that instead of the question.
    A hypothetical answer's vector is much closer to stored knowledge chunks
    than a question's vector — dramatically improves retrieval recall.

    Example:
      Query              : "Where does Akhil work?"
      Stored chunk       : "<PERSON> is an ML Engineer at Google."
      Hypothetical answer: "The person works at a tech company as an engineer."
      → hypothetical answer embedding is far closer to the stored chunk
    """
    prompt = f"""Write a short factual answer to this question as if you knew the answer.
Keep it to 2-3 sentences. Do not say you don't know — write a plausible answer.

Question: {query}"""

    hypothetical_answer = _llm.invoke(prompt).content.strip()
    return embed_query(hypothetical_answer)


def _rewrite_query(query: str) -> list[str]:
    """
    Generate alternative phrasings of the query.
    Different phrasings produce different embedding vectors which may
    match stored chunks that the original phrasing missed.

    Example:
      Original : "What does Akhil do for work?"
      Rewrites : ["What is Akhil's job?",
                  "What is Akhil's profession?",
                  "Where is Akhil employed?"]
    """
    prompt = f"""Generate 3 different ways to ask the following question.
Each version should use different words but mean the same thing.
Return only the questions, one per line, no numbering, no extra text.

Original question: {query}"""

    try:
        result = _llm.invoke(prompt).content.strip()
        rewrites = [q.strip() for q in result.split("\n") if q.strip()]
        return rewrites[:3]
    except Exception:
        return []


# ── Tool 1: Retrieve (self-healing) ───────────────────────────────────────────

@tool
def retrieve_tool(query: str) -> str:
    """
    Search the knowledge base for information relevant to the query.
    Use this whenever the user asks a question or needs information.
    Automatically tries multiple strategies if initial retrieval fails.
    Returns the most relevant text chunks found, or a not-found message.
    """

    if not query or not query.strip():
        return "No query provided — cannot search the knowledge base."

    # ── Attempt 1: Direct retrieval ───────────────────────────────────────────
    print(f"[retrieve] Attempt 1: direct query embedding")
    try:
        query_vector = embed_query(query.strip())
        docs, metas  = query_chunks(query_vector)
    except Exception as e:
        return f"Embedding failed: {e}"

    if docs and _check_relevance(query, docs):
        print(f"[retrieve] Attempt 1 succeeded — {len(docs)} relevant chunks")
        return _format_chunks(docs, metas)

    # ── Attempt 2: HyDE ───────────────────────────────────────────────────────
    print(f"[retrieve] Attempt 1 failed. Trying HyDE...")
    try:
        hyde_vector = _hyde_query(query)
        docs, metas = query_chunks(hyde_vector)
        if docs and _check_relevance(query, docs):
            print(f"[retrieve] HyDE succeeded — {len(docs)} relevant chunks")
            return _format_chunks(docs, metas)
    except Exception as e:
        print(f"[retrieve] HyDE failed: {e}")

    # ── Attempt 3: Query rewriting ────────────────────────────────────────────
    print(f"[retrieve] HyDE failed. Trying query rewrites...")
    rewrites = _rewrite_query(query)

    for i, rewrite in enumerate(rewrites):
        print(f"[retrieve] Rewrite {i+1}: '{rewrite}'")
        try:
            rewrite_vector = embed_query(rewrite)
            docs, metas    = query_chunks(rewrite_vector)
            if docs and _check_relevance(rewrite, docs):
                print(f"[retrieve] Rewrite {i+1} succeeded")
                return _format_chunks(docs, metas)
        except Exception:
            continue

    # ── All attempts failed ───────────────────────────────────────────────────
    print(f"[retrieve] All strategies exhausted — no relevant chunks found")
    return "No relevant information found in the knowledge base after multiple retrieval attempts."


# ── Tool 2: Store ─────────────────────────────────────────────────────────────

@tool
def store_tool(text: str, session_id: str = "unknown") -> str:
    """
    Extract knowledge from the given text and store it in the knowledge base.
    Use this when the user shares new facts, definitions, or information worth remembering.
    Always pass the current session_id when calling this tool.
    Returns a confirmation message with the number of facts stored.
    """

    if not text or not text.strip():
        return "No text provided — nothing to store."

    # ── Step 1: Extract clean facts ───────────────────────────────────────────
    extraction_prompt = f"{EXTRACTOR_SYSTEM}\n\nText to extract from:\n{text.strip()}"

    try:
        extracted = _llm.invoke(extraction_prompt).content.strip()
    except Exception as e:
        return f"Extraction failed: {e}"

    # ── Check for NO_FACTS sentinel ───────────────────────────────────────────
    if extracted.upper().strip() == "NO_FACTS":
        return "No new facts found in the message — nothing stored."

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    chunks = chunk_text(extracted)
    if not chunks:
        return "Extraction produced no storable content."

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        return f"Embedding failed: {e}"

    # ── Step 4: Build metadata ────────────────────────────────────────────────
    timestamp = datetime.utcnow().isoformat()
    metadatas = [
        {
            "source"     : SOURCE_USER_PROMPT,
            "session_id" : session_id,
            "chunk_index": i,
            "timestamp"  : timestamp,
            "file_name"  : "",
        }
        for i in range(len(chunks))
    ]

    # ── Step 5: Store ─────────────────────────────────────────────────────────
    try:
        stored = upsert_chunks(chunks, embeddings, metadatas)
    except Exception as e:
        return f"Storage failed: {e}"

    return f"Stored {stored} knowledge chunk(s) from your message."


# ── Tool 3: Summarize ─────────────────────────────────────────────────────────

@tool
def summarize_tool(text: str) -> str:
    """
    Summarize a long block of retrieved text into key points.
    Use this when retrieve_tool returns too much text or the user asks for a summary.
    Returns a concise 3-5 sentence summary.
    """

    if not text or not text.strip():
        return "No text provided to summarize."

    if len(text.strip()) < 500:
        return text.strip()

    summarization_prompt = f"{SUMMARIZER_SYSTEM}\n\nText to summarize:\n{text.strip()}"

    try:
        summary = _llm.invoke(summarization_prompt).content.strip()
    except Exception as e:
        return f"Summarization failed: {e}"

    return summary