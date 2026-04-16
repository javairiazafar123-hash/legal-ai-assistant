"""
RAG Pipeline — core logic for the Legal AI Assistant.

Responsibilities:
  - Chunk and embed uploaded documents
  - Store / retrieve chunks from ChromaDB
  - Build prompts and call the vLLM (OpenAI-compatible) API
  - Return structured answers with source attribution
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class SourceChunk:
    """A single retrieved document chunk with its metadata."""
    content: str
    filename: str
    chunk_index: int
    score: float  # cosine distance returned by ChromaDB (lower = more similar)


@dataclass
class QueryResult:
    """Structured answer returned to the API layer."""
    answer: str
    sources: List[SourceChunk]
    query: str
    elapsed_ms: float


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """
    Split *text* into overlapping windows of at most *chunk_size* characters.
    Simple character-level splitting is language-agnostic and dependency-free.
    """
    chunks: List[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ─── Pipeline ─────────────────────────────────────────────────────────────────


class RAGPipeline:
    """
    Full RAG pipeline:
      1. Embed documents → store in ChromaDB
      2. On query: retrieve top-k chunks → build prompt → call LLM → return answer
    """

    def __init__(self) -> None:
        logger.info("Initialising embedding model: %s", config.EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        logger.info("Connecting to ChromaDB at: %s", config.CHROMA_PERSIST_DIR)
        self._chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("Connecting to LLM API at: %s", config.LLM_API_URL)
        self._llm_client = OpenAI(
            base_url=config.LLM_API_URL,
            api_key=config.LLM_API_KEY,
        )

        logger.info("RAGPipeline ready.")

    # ── Document ingestion ────────────────────────────────────────────────────

    def add_document(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Chunk *text*, embed each chunk and upsert into ChromaDB.

        Returns a summary dict with chunk count and document filename.
        """
        chunks = _chunk_text(text)
        if not chunks:
            raise ValueError(f"Document '{filename}' produced no text chunks.")

        embeddings = self._embedder.encode(chunks, show_progress_bar=False).tolist()

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"filename": filename, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        logger.info("Added '%s': %d chunks stored.", filename, len(chunks))
        return {"filename": filename, "chunks": len(chunks)}

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> QueryResult:
        """
        Retrieve the top-k relevant chunks for *question*, build a prompt,
        call the configured LLM, and return a :class:`QueryResult`.
        """
        t0 = time.perf_counter()

        # 1. Embed the question
        q_embedding = self._embedder.encode([question], show_progress_bar=False).tolist()[0]

        # 2. Retrieve from ChromaDB
        results = self._collection.query(
            query_embeddings=[q_embedding],
            n_results=min(config.TOP_K, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        raw_docs = results["documents"][0] if results["documents"] else []
        raw_metas = results["metadatas"][0] if results["metadatas"] else []
        raw_dists = results["distances"][0] if results["distances"] else []

        sources: List[SourceChunk] = [
            SourceChunk(
                content=doc,
                filename=meta.get("filename", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=float(dist),
            )
            for doc, meta, dist in zip(raw_docs, raw_metas, raw_dists)
        ]

        # 3. Build the prompt
        context_text = "\n\n---\n\n".join(
            f"[Source: {s.filename}, chunk {s.chunk_index}]\n{s.content}"
            for s in sources
        )

        system_prompt = (
            "You are a highly knowledgeable legal assistant. "
            "Answer the user's question based ONLY on the provided legal document excerpts. "
            "If the answer cannot be found in the excerpts, say so clearly. "
            "Be precise, professional, and cite the source document where relevant."
        )

        user_prompt = (
            f"Context from legal documents:\n\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Please provide a clear and accurate answer based on the above context."
        )

        # 4. Call the LLM
        if not raw_docs:
            answer = (
                "No documents have been uploaded yet. "
                "Please upload a legal document before asking questions."
            )
        else:
            try:
                response = self._llm_client.chat.completions.create(
                    model=config.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                )
                answer = response.choices[0].message.content or "(empty response)"
            except Exception as exc:  # noqa: BLE001
                logger.error("LLM call failed: %s", exc)
                answer = (
                    f"⚠️ The LLM service is currently unavailable "
                    f"({config.LLM_API_URL}). "
                    f"Please ensure vLLM is running.\n\n"
                    f"Retrieved context is shown in the sources below."
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return QueryResult(
            answer=answer,
            sources=sources,
            query=question,
            elapsed_ms=round(elapsed_ms, 2),
        )

    # ── Document listing ──────────────────────────────────────────────────────

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Return a de-duplicated list of uploaded documents with their chunk counts.
        """
        total = self._collection.count()
        if total == 0:
            return []

        # Fetch all metadata (no document text needed)
        all_items = self._collection.get(include=["metadatas"])
        metas = all_items.get("metadatas") or []

        summary: Dict[str, Dict[str, Any]] = {}
        for m in metas:
            fn = m.get("filename", "unknown")
            if fn not in summary:
                summary[fn] = {"filename": fn, "chunks": 0}
            summary[fn]["chunks"] += 1

        return list(summary.values())

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear_all(self) -> Dict[str, str]:
        """Delete and recreate the ChromaDB collection, removing all stored data."""
        self._chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
        self._collection = self._chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared.")
        return {"status": "cleared"}
