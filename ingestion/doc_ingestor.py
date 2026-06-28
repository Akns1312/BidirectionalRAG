import os
from datetime import datetime

from pypdf import PdfReader

from embeddings.chunker import chunk_text
from embeddings.embedder import embed_texts
from vectordb.db_client import upsert_chunks
from vectordb.schema import SOURCE_DOCUMENT


# ── Supported file types ──────────────────────────────────────────────────────
# Currently supports PDF. Plain text (.txt) is also handled.
# Extend SUPPORTED_EXTENSIONS and add a matching _extract_* function
# to support Word docs, HTML, markdown, etc. in the future.
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_from_pdf(file_path: str) -> list[str]:
    """
    Extract text from each page of a PDF.

    Returns:
        List of strings — one string per page.
        Empty pages are filtered out.
    """
    reader = PdfReader(file_path)
    pages  = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
        else:
            print(f"  [doc_ingestor] Page {page_num + 1} is empty or unreadable — skipped")

    return pages


def _extract_from_txt(file_path: str) -> list[str]:
    """
    Extract text from a plain .txt file.

    Returns:
        List with a single string — the entire file content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    return [content] if content else []


def _extract_text(file_path: str) -> list[str]:
    """
    Route to the correct extractor based on file extension.

    Args:
        file_path: Absolute or relative path to the file

    Returns:
        List of text strings (one per page for PDFs, one for txt)

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If the file doesn't exist
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        return _extract_from_pdf(file_path)
    elif ext == ".txt":
        return _extract_from_txt(file_path)


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_document(file_path: str) -> dict:
    """
    Full pipeline: extract → chunk → embed → store.
    Call this once per document before or during a chat session.

    Args:
        file_path: Path to the document file (.pdf or .txt)

    Returns:
        A summary dict with ingestion stats:
        {
            "file"      : "my_document.pdf",
            "pages"     : 12,
            "chunks"    : 47,
            "stored"    : 47,
            "timestamp" : "2026-06-13T10:30:00"
        }

    Example:
        result = ingest_document("./docs/company_handbook.pdf")
        print(f"Stored {result['chunks']} chunks from {result['file']}")
    """

    file_name = os.path.basename(file_path)
    print(f"\n[doc_ingestor] Starting ingestion: {file_name}")

    # ── Step 1: Extract text ──────────────────────────────────────────────────
    print(f"[doc_ingestor] Extracting text...")
    pages = _extract_text(file_path)

    if not pages:
        raise ValueError(f"No readable text found in {file_name}. "
                         f"The file may be scanned/image-based.")

    print(f"[doc_ingestor] Extracted {len(pages)} page(s)")

    # ── Step 2: Chunk all pages ───────────────────────────────────────────────
    # Process each page separately so chunk boundaries don't cross pages.
    # This preserves page-level context — chunk metadata records page number.
    print(f"[doc_ingestor] Chunking text...")

    all_chunks   = []
    all_metadatas = []

    for page_num, page_text in enumerate(pages):
        page_chunks = chunk_text(page_text)

        for chunk_idx, chunk in enumerate(page_chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source"     : SOURCE_DOCUMENT,
                "file_name"  : file_name,
                "file_path"  : file_path,
                "page_number": page_num + 1,      # 1-based for readability
                "chunk_index": chunk_idx,
                "session_id" : "external",        # not from a chat session
                "timestamp"  : datetime.utcnow().isoformat(),
            })

    if not all_chunks:
        raise ValueError(f"Text was extracted but produced no chunks. "
                         f"Check CHUNK_SIZE in .env — it may be too large.")

    print(f"[doc_ingestor] Produced {len(all_chunks)} chunk(s)")

    # ── Step 3: Embed all chunks ──────────────────────────────────────────────
    # embed_texts() sends all chunks to nomic-embed-text via Ollama.
    # This is the slowest step — large documents may take 30-60 seconds.
    print(f"[doc_ingestor] Embedding chunks (this may take a moment)...")
    embeddings = embed_texts(all_chunks)

    # ── Step 4: Store in vector DB ────────────────────────────────────────────
    print(f"[doc_ingestor] Storing in vector DB...")
    stored = upsert_chunks(all_chunks, embeddings, all_metadatas)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "file"     : file_name,
        "pages"    : len(pages),
        "chunks"   : len(all_chunks),
        "stored"   : stored,
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(f"[doc_ingestor] Done — {stored} chunks stored from {file_name}\n")
    return summary


def ingest_multiple(file_paths: list[str]) -> list[dict]:
    """
    Ingest multiple documents in sequence.
    Returns a list of summary dicts, one per document.

    Args:
        file_paths: List of file paths to ingest

    Example:
        results = ingest_multiple([
            "./docs/handbook.pdf",
            "./docs/policy.txt",
        ])
        for r in results:
            print(f"{r['file']}: {r['chunks']} chunks stored")
    """

    results = []

    for file_path in file_paths:
        try:
            result = ingest_document(file_path)
            results.append(result)
        except Exception as e:
            print(f"[doc_ingestor] ERROR ingesting {file_path}: {e}")
            results.append({
                "file" : os.path.basename(file_path),
                "error": str(e),
            })

    return results