import uuid

from ingestion.prompt_ingestor import ingest_prompt
from ingestion.doc_ingestor import ingest_document, ingest_multiple
from agents.orchestrator import run, run_with_trace
from embeddings.embedder import verify_embedder
from vectordb.db_client import get_collection_count


# ── Startup checks ────────────────────────────────────────────────────────────
# Run these before accepting any user input.
# Catches the two most common startup failures:
#   1. Ollama not running
#   2. Models not pulled
# Better to fail loudly here than silently mid-conversation.

def startup_checks() -> bool:
    """
    Verify that Ollama is running and both models are loaded.
    Returns True if all checks pass, False otherwise.
    """
    print("\n[startup] Running checks...")

    try:
        verify_embedder()
    except RuntimeError as e:
        print(f"\n[startup] FAILED: {e}")
        print("[startup] Fix: make sure Ollama is running with 'ollama serve'")
        print("[startup]       and models are pulled:")
        print("[startup]       ollama pull qwen2.5:7b-instruct")
        print("[startup]       ollama pull nomic-embed-text")
        return False

    count = get_collection_count()
    print(f"[startup] Vector DB: {count} chunk(s) currently stored")
    print("[startup] All checks passed. System ready.\n")
    return True


# ── Document ingestion helper ─────────────────────────────────────────────────
# Call this before the chat loop to pre-load documents into the vector DB.
# Documents are stored permanently — they persist across restarts.

def load_documents(file_paths: list[str]) -> None:
    """
    Ingest one or more documents into the vector DB.

    Args:
        file_paths: List of paths to .pdf or .txt files

    Example:
        load_documents([
            "./docs/company_handbook.pdf",
            "./docs/product_manual.pdf",
        ])
    """
    if not file_paths:
        return

    print(f"[main] Loading {len(file_paths)} document(s)...\n")
    results = ingest_multiple(file_paths)

    print("\n[main] Ingestion summary:")
    for r in results:
        if "error" in r:
            print(f"  ✗ {r['file']} — ERROR: {r['error']}")
        else:
            print(f"  ✓ {r['file']} — {r['chunks']} chunks stored")
    print()


# ── Chat function ─────────────────────────────────────────────────────────────

def chat(
    user_input : str,
    session_id : str  = None,
    user_id    : str  = "user",
    debug_mode : bool = False,
) -> tuple[str, str]:
    """
    Process one user message and return the agent's response.

    Args:
        user_input : The raw string the user typed
        session_id : UUID for this session. If None, a new one is generated.
        user_id    : Identifier for the user
        debug_mode : If True, prints tool calls made by the agent

    Returns:
        Tuple of (response_string, session_id)
        session_id is returned so the caller can reuse it on the next turn.
    """

    # Generate session ID on first message
    session_id = session_id or str(uuid.uuid4())

    # Build the prompt envelope (sanitize, detect language, attach history)
    envelope = ingest_prompt(user_input, session_id, user_id)

    # Run the agent — normal or debug mode
    if debug_mode:
        trace    = run_with_trace(envelope)
        response = trace["response"]

        if trace["tool_calls"]:
            print("\n[debug] Tools called:")
            for tc in trace["tool_calls"]:
                print(f"  → {tc}")
        else:
            print("\n[debug] No tools called")
    else:
        response = run(envelope)

    return response, session_id


# ── Main chat loop ────────────────────────────────────────────────────────────

def main():
    """
    Interactive command-line chat loop.
    Runs until the user types 'exit' or 'quit'.

    Special commands during chat:
        exit / quit     → stop the program
        /debug          → toggle debug mode (shows tool calls)
        /load <path>    → ingest a document mid-conversation
        /count          → show how many chunks are in the vector DB
        /clear          → start a new session (fresh conversation history)
        /help           → show available commands
    """

    # ── Startup ───────────────────────────────────────────────────────────────
    if not startup_checks():
        return  # exit if Ollama isn't running

    # ── Optional: pre-load documents ─────────────────────────────────────────
    # Uncomment and add your document paths here to load them at startup.
    # Documents are stored permanently so you only need to do this once.
    #
    # load_documents([
    #     "./docs/your_document.pdf",
    # ])

    # ── Session state ─────────────────────────────────────────────────────────
    session_id = None   # generated on first message
    debug_mode = False  # toggle with /debug command

    print("=" * 55)
    print("  RAG Agent — Bidirectional Knowledge Pipeline")
    print("=" * 55)
    print("  Type your message to chat.")
    print("  Type /help to see available commands.")
    print("  Type exit to quit.")
    print("=" * 55 + "\n")

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:

        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D gracefully
            print("\n\n[main] Interrupted. Goodbye.")
            break

        # ── Empty input ───────────────────────────────────────────────────────
        if not user_input:
            continue

        # ── Special commands ──────────────────────────────────────────────────
        if user_input.lower() in ("exit", "quit"):
            print("\n[main] Goodbye.")
            break

        elif user_input.lower() == "/help":
            print("\nAvailable commands:")
            print("  /debug        — toggle debug mode (shows tool calls)")
            print("  /load <path>  — ingest a document into the knowledge base")
            print("  /count        — show number of chunks in the vector DB")
            print("  /clear        — start a fresh session")
            print("  exit / quit   — exit the program\n")
            continue

        elif user_input.lower() == "/debug":
            debug_mode = not debug_mode
            status = "ON" if debug_mode else "OFF"
            print(f"\n[main] Debug mode {status}\n")
            continue

        elif user_input.lower() == "/count":
            count = get_collection_count()
            print(f"\n[main] Vector DB contains {count} chunk(s)\n")
            continue

        elif user_input.lower() == "/clear":
            session_id = str(uuid.uuid4())
            print(f"\n[main] New session started. Conversation history cleared.\n")
            continue

        elif user_input.lower().startswith("/load "):
            file_path = user_input[6:].strip()
            if file_path:
                load_documents([file_path])
            else:
                print("\n[main] Usage: /load <path_to_file>\n")
            continue

        # ── Normal chat message ───────────────────────────────────────────────
        try:
            response, session_id = chat(
                user_input=user_input,
                session_id=session_id,
                debug_mode=debug_mode,
            )
            print(f"\nAgent: {response}\n")

        except ValueError as e:
            # prompt_ingestor raises ValueError for empty prompts
            print(f"\n[main] Input error: {e}\n")

        except Exception as e:
            # Catch unexpected errors without crashing the loop
            print(f"\n[main] Unexpected error: {e}")
            print("[main] The session is still active — please try again.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()