"""
FastAPI entry point for the Legal AI Assistant.

Endpoints
─────────
POST   /upload      Upload a PDF or TXT file → add to RAG pipeline
POST   /query       Ask a question → get answer + source chunks
GET    /documents   List all uploaded documents
DELETE /clear       Wipe the vector store
GET    /health      Liveness / readiness probe
GET    /            Serve the frontend SPA
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Bootstrap logging before any local imports so config messages show up
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local imports (after logging so init messages are visible)
# ---------------------------------------------------------------------------
import config  # noqa: E402
from rag_pipeline import RAGPipeline, QueryResult  # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Legal AI Assistant",
    description="RAG-based legal document Q&A powered by LangChain + ChromaDB + vLLM",
    version="1.0.0",
)

# Initialise the RAG pipeline once at startup (loads embedding model, etc.)
_pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def _startup() -> None:
    global _pipeline
    logger.info("Starting Legal AI Assistant…")
    _pipeline = RAGPipeline()
    logger.info("Pipeline ready.")


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialised.")
    return _pipeline


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class ChunkOut(BaseModel):
    content: str
    filename: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[ChunkOut]
    query: str
    elapsed_ms: float


class DocumentOut(BaseModel):
    filename: str
    chunks: int


# ---------------------------------------------------------------------------
# Helper: extract text from an uploaded file
# ---------------------------------------------------------------------------

def _extract_text(filename: str, data: bytes) -> str:
    """Return plain text from a PDF or TXT upload."""
    lower = filename.lower()

    if lower.endswith(".txt"):
        return data.decode("utf-8", errors="replace")

    if lower.endswith(".pdf"):
        try:
            import PyPDF2  # type: ignore

            reader = PyPDF2.PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not parse PDF '{filename}': {exc}",
            ) from exc

    raise HTTPException(
        status_code=415,
        detail="Unsupported file type. Please upload a .pdf or .txt file.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health() -> JSONResponse:
    """Liveness probe — always returns 200 if the server is up."""
    docs = get_pipeline().list_documents()
    return JSONResponse({
        "status": "ok",
        "llm_api_url": config.LLM_API_URL,
        "model": config.MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL,
        "documents_loaded": len(docs),
    })


@app.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    summary="Upload a PDF or TXT legal document",
)
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a PDF or plain-text file, extract its text, chunk it,
    embed it, and store the embeddings in ChromaDB.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text = _extract_text(file.filename, data)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the file.")

    result = get_pipeline().add_document(text, file.filename)
    return JSONResponse(content=result, status_code=status.HTTP_201_CREATED)


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    summary="Ask a question about uploaded documents",
)
async def query_documents(req: QueryRequest) -> QueryResponse:
    """
    Embed *question*, retrieve relevant chunks from ChromaDB,
    build a prompt, and stream the answer from the configured LLM.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result: QueryResult = get_pipeline().query(req.question)

    return QueryResponse(
        answer=result.answer,
        sources=[
            ChunkOut(
                content=s.content,
                filename=s.filename,
                chunk_index=s.chunk_index,
                score=s.score,
            )
            for s in result.sources
        ],
        query=result.query,
        elapsed_ms=result.elapsed_ms,
    )


@app.get(
    "/documents",
    response_model=List[DocumentOut],
    tags=["documents"],
    summary="List uploaded documents",
)
async def list_documents() -> List[DocumentOut]:
    """Return a list of all documents currently stored in the vector store."""
    docs = get_pipeline().list_documents()
    return [DocumentOut(**d) for d in docs]


@app.delete(
    "/clear",
    tags=["documents"],
    summary="Clear all documents from the vector store",
)
async def clear_documents() -> JSONResponse:
    """Delete all document embeddings from ChromaDB."""
    result = get_pipeline().clear_all()
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Static file serving (frontend SPA) — must be mounted LAST
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
else:
    logger.warning("Static directory not found at %s — frontend will not be served.", _static_dir)
