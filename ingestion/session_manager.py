from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field


# ── Turn dataclass ─────────────────────────────────────────────────────────────
# Represents one message in the conversation.
# role    : "user" or "assistant"
# content : the actual message text
# timestamp: when this turn was recorded (ISO format)
@dataclass
class Turn:
    role     : str
    content  : str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to plain dict for easy use in LangChain message lists."""
        return {
            "role"     : self.role,
            "content"  : self.content,
            "timestamp": self.timestamp,
        }


# ── In-memory session store ───────────────────────────────────────────────────
# A dict mapping session_id → list of Turn objects.
# defaultdict(list) means accessing a new session_id automatically
# creates an empty list — no need to initialise sessions manually.
#
# Important: this is in-memory only. Sessions are lost when the process
# restarts. For production you would persist this to Redis or a DB,
# but for local development in-memory is perfectly fine.
_sessions: dict[str, list[Turn]] = defaultdict(list)


# ── WRITE ─────────────────────────────────────────────────────────────────────

def add_turn(session_id: str, role: str, content: str) -> None:
    """
    Record a new message turn in the session history.
    Call this after every user message AND every assistant response.

    Args:
        session_id: Unique identifier for this conversation session
        role      : "user" or "assistant"
        content   : The message text
    """

    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got '{role}'")

    if not content or not content.strip():
        return  # don't store empty messages

    _sessions[session_id].append(
        Turn(role=role, content=content.strip())
    )


# ── READ ──────────────────────────────────────────────────────────────────────

def get_history(session_id: str, last_n: int = 5) -> list[dict]:
    """
    Retrieve the last N turns of a session as a list of dicts.
    Used by prompt_ingestor.py to attach history to the envelope,
    and by orchestrator.py to prepend context to the agent's messages.

    Args:
        session_id: The session to fetch history for
        last_n    : How many recent turns to return (default 5)
                    Keep this small — every turn adds tokens to the LLM call.
                    5 turns ≈ 2-3 back-and-forth exchanges, usually enough.

    Returns:
        List of dicts: [{"role": "user", "content": "...", "timestamp": "..."}]
        Returns empty list if session doesn't exist yet.
    """

    turns = _sessions.get(session_id, [])

    # Slice the last N turns — most recent conversation context
    recent = turns[-last_n:] if len(turns) > last_n else turns

    return [t.to_dict() for t in recent]


def get_turn_count(session_id: str) -> int:
    """
    Returns the total number of turns recorded in a session.
    Used by prompt_ingestor.py to set the turn number on the envelope.
    """
    return len(_sessions.get(session_id, []))


def get_all_sessions() -> list[str]:
    """
    Returns a list of all active session IDs.
    Useful for debugging — lets you see how many sessions are running.
    """
    return list(_sessions.keys())


# ── DELETE ────────────────────────────────────────────────────────────────────

def clear_session(session_id: str) -> None:
    """
    Wipe all history for a session.
    Call this when the user starts a fresh conversation or says
    something like "forget our conversation".

    Note: this only clears conversation history, NOT the vector DB.
    To also remove stored knowledge use db_client.delete_by_session().
    """
    if session_id in _sessions:
        del _sessions[session_id]


def clear_all_sessions() -> None:
    """
    Wipe all session history across all users.
    Only use during development/testing — destructive in production.
    """
    _sessions.clear()


# ── UTILS ─────────────────────────────────────────────────────────────────────

def format_history_as_string(session_id: str, last_n: int = 5) -> str:
    """
    Format conversation history as a readable string block.
    Used when injecting history directly into a prompt string
    rather than as a structured message list.

    Example output:
        User     : What is RAG?
        Assistant: RAG stands for Retrieval Augmented Generation...
        User     : How does it store data?
    """

    history = get_history(session_id, last_n)

    if not history:
        return ""

    lines = []
    for turn in history:
        role  = "User     " if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")

    return "\n".join(lines)