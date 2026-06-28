from langchain_ollama import ChatOllama
from config.settings import settings


# ── System prompt ────────────────────────────────────────────────────────────
# Tells the LLM to stay strictly within the retrieved context.
# "ONLY the context below" is the key instruction that prevents hallucination.
RESPONDER_SYSTEM = """You are a precise assistant that answers questions using ONLY the context provided below.

Rules:
1. Answer ONLY from the context. Do not use outside knowledge.
2. If the context does not contain the answer, say: "I don't have enough information in my knowledge base to answer that."
3. Keep your answer concise and direct.
4. If multiple context chunks are relevant, synthesize them into one clear answer.
"""

# ── LLM instance ─────────────────────────────────────────────────────────────
# Low temperature = factual, deterministic responses
# We want the LLM to follow context, not be creative
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.LLM_MODEL,
    temperature=0.1,
)


def build_response(question: str, context_chunks: list[str]) -> str:
    """
    Build a grounded response from retrieved context chunks.

    Args:
        question      : The user's original question (clean text from envelope)
        context_chunks: List of text chunks retrieved from the vector DB

    Returns:
        A grounded answer string, or a fallback message if context is empty.
    """

    # ── Guard: no context retrieved ──────────────────────────────────────────
    # If the vector DB returned nothing, there's no point calling the LLM.
    # Return a clear message instead of letting the LLM hallucinate an answer.
    if not context_chunks:
        return "I don't have any relevant information in my knowledge base for that question."

    # ── Assemble context block ────────────────────────────────────────────────
    # Number each chunk so the LLM can reference them clearly.
    # Separator lines make chunk boundaries obvious to the model.
    context_block = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    )

    # ── Build the full prompt ─────────────────────────────────────────────────
    # Structure: system instructions → context → question
    # This ordering is important — context before question helps the LLM
    # "prime" itself with the evidence before reading what's being asked.
    prompt = f"""{RESPONDER_SYSTEM}

--- CONTEXT START ---
{context_block}
--- CONTEXT END ---

Question: {question}

Answer:"""

    # ── Call the LLM ──────────────────────────────────────────────────────────
    response = llm.invoke(prompt)

    return response.strip()


def build_conversational_response(question: str, context_chunks: list[str], history: list[dict]) -> str:
    """
    Extended version that also includes conversation history.
    Use this when the question is a follow-up that needs prior context.

    Args:
        question      : The user's current question
        context_chunks: Retrieved chunks from vector DB
        history       : Last N conversation turns from session_manager

    Returns:
        A grounded answer that is also aware of the conversation so far.
    """

    # ── Format conversation history ───────────────────────────────────────────
    # Only include last 3 turns to keep the prompt from growing too large.
    history_block = ""
    if history:
        recent = history[-3:]
        history_block = "--- CONVERSATION HISTORY ---\n"
        for turn in recent:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_block += f"{role}: {turn['content']}\n"
        history_block += "----------------------------\n\n"

    # ── Guard: no context ────────────────────────────────────────────────────
    if not context_chunks:
        if history:
            # Even without DB context, we can answer from conversation history
            prompt = f"""{RESPONDER_SYSTEM}

{history_block}Question: {question}

Answer (based on conversation history only):"""
            return llm.invoke(prompt).strip()
        return "I don't have any relevant information in my knowledge base for that question."

    # ── Assemble context block ────────────────────────────────────────────────
    context_block = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    )

    # ── Full prompt with history + context ───────────────────────────────────
    prompt = f"""{RESPONDER_SYSTEM}

{history_block}--- CONTEXT START ---
{context_block}
--- CONTEXT END ---

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.strip()