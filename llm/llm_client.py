from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from config.settings import settings
from langchain_ollama import ChatOllama

# Main LLM (Qwen2.5 7B) — used for reasoning, extraction, response
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.LLM_MODEL,
    temperature=0.2,       # low temp = more factual, less creative
)

# Embedding model (nomic-embed-text) — used only for vectorising
embedder = OllamaEmbeddings(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.EMBED_MODEL,
)
