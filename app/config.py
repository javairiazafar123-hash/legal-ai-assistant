"""
Configuration settings for the Legal AI Assistant RAG application.
All settings can be overridden via environment variables.
"""

import os


# ─── LLM Configuration ────────────────────────────────────────────────────────

LLM_API_URL: str = os.getenv("LLM_API_URL", "http://localhost:8000/v1")
"""OpenAI-compatible API base URL (vLLM, OpenAI, etc.)"""

MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
"""LLM model identifier passed to the API"""

LLM_API_KEY: str = os.getenv("LLM_API_KEY", "dummy-key")
"""API key; 'dummy-key' works fine with local vLLM deployments"""

# ─── Embedding Configuration ──────────────────────────────────────────────────

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
"""Sentence-transformers model used to embed document chunks"""

# ─── Vector Store Configuration ───────────────────────────────────────────────

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
"""Directory where ChromaDB persists its data between restarts"""

CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "legal_docs")
"""Name of the ChromaDB collection used to store document chunks"""

# ─── Retrieval Configuration ──────────────────────────────────────────────────

TOP_K: int = int(os.getenv("TOP_K", "5"))
"""Number of most-relevant chunks retrieved per query"""

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
"""Maximum character length of each document chunk"""

CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
"""Character overlap between consecutive chunks (preserves context at boundaries)"""

# ─── Generation Configuration ─────────────────────────────────────────────────

MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
"""Maximum tokens the LLM may generate in a single response"""

TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
"""Sampling temperature; lower = more deterministic / factual"""

# ─── Server Configuration ─────────────────────────────────────────────────────

HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8080"))
