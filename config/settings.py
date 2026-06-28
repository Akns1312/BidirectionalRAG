from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # Groq
    GROQ_API_KEY   : str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL      : str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    # Ollama — still used for embeddings only
    #OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    #OPENAI_API_KEY : str = os.getenv("OPENAI_API_KEY","")
    #EMBED_MODEL    : str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # ChromaDB
    CHROMA_PATH    : str = os.getenv("CHROMA_PATH", "./chroma_store")
    CHUNK_SIZE     : int = int(os.getenv("CHUNK_SIZE", 400))
    CHUNK_OVERLAP  : int = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K          : int = int(os.getenv("TOP_K", 5))

settings = Settings()