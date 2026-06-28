from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

# ── Why RecursiveCharacterTextSplitter ───────────────────────────────────────
# Instead of blindly splitting every N words, this splitter tries to find
# natural boundaries in order:
#   1. Paragraph breaks (\n\n) — most preferred split point
#   2. Line breaks (\n)
#   3. Spaces (word boundary)
#   4. Characters (last resort)
#
# This means chunks tend to end at paragraph or sentence boundaries,
# which preserves semantic meaning much better than fixed word windows.
#
# Note: chunk_size here is in CHARACTERS not words.
# 400 words ≈ 1600-2000 characters. We use 1500 as a safe middle ground
# that stays within nomic-embed-text's 512 token limit.


def get_splitter(
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> RecursiveCharacterTextSplitter:
    """
    Build and return a configured text splitter instance.
    Exposed separately so callers can reuse the same splitter
    across multiple texts without rebuilding it each time.
    """

    # Convert word-based settings to character-based
    # Average English word = ~5 characters + 1 space = ~6 chars
    # We multiply by 6 to convert word count → character count
    size    = (chunk_size    or settings.CHUNK_SIZE)    * 6
    overlap = (chunk_overlap or settings.CHUNK_OVERLAP) * 6

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,        # measure by character count
        is_separator_regex=False,   # treat separators as plain strings
    )


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[str]:
    """
    Split a single text string into overlapping chunks.

    Args:
        text         : Raw text to split
        chunk_size   : Max words per chunk (converted to chars internally)
        chunk_overlap: Overlap in words between consecutive chunks

    Returns:
        List of chunk strings. Empty list if input is blank.
    """

    if not text or not text.strip():
        return []

    splitter = get_splitter(chunk_size, chunk_overlap)
    chunks   = splitter.split_text(text)

    # Filter out chunks that are too short to be meaningful
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def chunk_documents(documents: list[str]) -> list[tuple[str, int]]:
    """
    Chunk multiple documents and track which doc each chunk came from.

    Args:
        documents: List of raw text strings (one per page or document)

    Returns:
        List of (chunk_text, doc_index) tuples.
        Used by doc_ingestor.py for multi-page PDFs.
    """

    splitter   = get_splitter()  # build once, reuse across all docs
    all_chunks = []

    for doc_index, doc_text in enumerate(documents):
        if not doc_text or not doc_text.strip():
            continue
        chunks = splitter.split_text(doc_text)
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 50:
                all_chunks.append((chunk, doc_index))

    return all_chunks


def estimate_chunk_count(text: str) -> int:
    """
    Estimate how many chunks a text will produce without chunking it.
    Useful for logging during large document ingestion.
    """

    if not text:
        return 0

    char_size  = settings.CHUNK_SIZE * 6
    char_step  = (settings.CHUNK_SIZE - settings.CHUNK_OVERLAP) * 6
    char_count = len(text)

    if char_count <= char_size:
        return 1

    return max(1, (char_count - (settings.CHUNK_OVERLAP * 6)) // char_step)