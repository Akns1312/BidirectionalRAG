from dataclasses import dataclass, field
from datetime import datetime

from langdetect import detect, LangDetectException

from ingestion.pii_guard import sanitize
from ingestion.session_manager import get_history, get_turn_count


# ── PromptEnvelope ────────────────────────────────────────────────────────────
# This is the central data object that flows through the entire pipeline.
# Every module downstream receives this object — nothing passes raw strings.
#
# Think of it as a "ticket" that carries:
#   - the clean, sanitized user message
#   - who sent it and when
#   - which conversation it belongs to
#   - the recent conversation history for context
#   - what language it's in
#
# Using a dataclass gives us:
#   - auto-generated __repr__ for easy debugging (just print the envelope)
#   - type hints on every field
#   - default values where sensible
@dataclass
class PromptEnvelope:
    text      : str          # sanitized user message (PII masked)
    raw_text  : str          # original unsanitized text (for reference only, never stored)
    session_id: str          # UUID identifying this conversation session
    user_id   : str          # identifier for the user
    turn      : int          # which turn number this is in the session (1-based)
    lang      : str          # ISO language code e.g. "en", "hi", "ta"
    history   : list[dict]   # last N turns from session_manager
    timestamp : str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __repr__(self) -> str:
        return (
            f"PromptEnvelope("
            f"turn={self.turn}, "
            f"lang={self.lang}, "
            f"session={self.session_id[:8]}..., "
            f"text='{self.text[:60]}{'...' if len(self.text) > 60 else ''}'"
            f")"
        )


# ── Language detection ────────────────────────────────────────────────────────
def _detect_language(text: str) -> str:
    """
    Detect the language of a text string.
    Returns ISO 639-1 code: "en", "hi", "ta", "fr", etc.
    Falls back to "en" if detection fails or text is too short.

    langdetect needs at least ~20 characters to be reliable.
    Very short prompts like "yes" or "ok" default to "en".
    """
    if len(text.strip()) < 20:
        return "en"

    try:
        return detect(text)
    except LangDetectException:
        # langdetect throws if it can't determine the language
        # (e.g. text is purely numeric or symbolic)
        return "en"


# ── Main function ─────────────────────────────────────────────────────────────

def ingest_prompt(
    raw_text  : str,
    session_id: str,
    user_id   : str = "user",
    history_n : int = 5,
) -> PromptEnvelope:
    """
    Process a raw user message into a structured PromptEnvelope.

    This is the single entry point for all user messages.
    Called by main.py on every user input before passing to the agent.

    Pipeline inside this function:
        1. Strip whitespace
        2. Sanitize PII (mask emails, phones, names)
        3. Detect language
        4. Fetch recent session history
        5. Get current turn number
        6. Assemble and return PromptEnvelope

    Args:
        raw_text  : The exact string the user typed
        session_id: UUID for this conversation (generated in main.py)
        user_id   : Identifier for the user (default "user")
        history_n : How many recent turns to include (default 5)

    Returns:
        A fully populated PromptEnvelope ready for the orchestrator.

    Raises:
        ValueError: If raw_text is empty or whitespace only.
    """

    # ── Step 1: Basic validation ──────────────────────────────────────────────
    if not raw_text or not raw_text.strip():
        raise ValueError("Prompt cannot be empty.")

    stripped = raw_text.strip()

    # ── Step 2: Sanitize PII ──────────────────────────────────────────────────
    # Run through pii_guard before anything else.
    # The sanitized version is what gets stored and processed.
    # We keep raw_text separately for reference but it is NEVER stored.
    clean_text = sanitize(stripped)

    # ── Step 3: Detect language ───────────────────────────────────────────────
    # Use the clean text for detection — PII masking doesn't affect language.
    lang = _detect_language(clean_text)

    # ── Step 4: Fetch session history ─────────────────────────────────────────
    # Get the last N turns so the agent understands conversational context.
    # Example: if user says "explain that further", the agent needs to know
    # what "that" refers to from previous turns.
    history = get_history(session_id, last_n=history_n)

    # ── Step 5: Get turn number ───────────────────────────────────────────────
    # Turn count BEFORE adding the current message.
    # get_turn_count returns how many turns exist, so +1 = this turn's number.
    turn = get_turn_count(session_id) + 1

    # ── Step 6: Assemble envelope ─────────────────────────────────────────────
    envelope = PromptEnvelope(
        text      = clean_text,
        raw_text  = stripped,       # kept for debugging, never passed downstream
        session_id= session_id,
        user_id   = user_id,
        turn      = turn,
        lang      = lang,
        history   = history,
    )

    return envelope