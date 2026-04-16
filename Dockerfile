# ── Legal AI Assistant — Dockerfile ──────────────────────────────────────────
# Build:  docker build -t legal-ai-assistant .
# Run:    docker run -p 8080:8080 -e LLM_API_URL=http://host:8000/v1 legal-ai-assistant

FROM python:3.11-slim

# Metadata
LABEL maintainer="DL Project Team"
LABEL description="RAG-based Legal AI Assistant (FastAPI + ChromaDB + vLLM)"

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY app/ ./app/
COPY sample_docs/ ./sample_docs/

# Create directories the app may write to at runtime
RUN mkdir -p /app/chroma_db /app/uploads

# ── Environment defaults (override at runtime) ─────────────────────────────────
ENV LLM_API_URL=http://localhost:8000/v1
ENV MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV CHROMA_PERSIST_DIR=/app/chroma_db
ENV TOP_K=5
ENV HOST=0.0.0.0
ENV PORT=8080

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8080

# ── Healthcheck ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ── Start server ───────────────────────────────────────────────────────────────
WORKDIR /app/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
