# ⚖️ Legal AI Assistant — RAG Pipeline
### Deep Learning End-Semester Project | Track A: RAG

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)](https://trychroma.com)
[![vLLM](https://img.shields.io/badge/vLLM-Served-orange)](https://vllm.ai)

> Upload any legal document (PDF/TXT) and ask questions in plain English. Answers are grounded in your document using RAG + vLLM.

---

## 🚀 Quick Start (3 steps)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/dl-project.git
cd dl-project

# 2. Setup
bash scripts/setup.sh

# 3. Run
bash scripts/run.sh           # API on http://localhost:8080
bash scripts/run_streamlit.sh # UI  on http://localhost:8501
```

Open http://localhost:8501 in your browser. Done! ✅

---

## 📁 Project Structure

```
dl-project/
├── app/
│   ├── main.py              # FastAPI backend
│   ├── rag_pipeline.py      # RAG logic (embed, store, retrieve, generate)
│   ├── config.py            # All configuration (env-overridable)
│   ├── streamlit_app.py     # Streamlit UI
│   └── static/
│       └── index.html       # Web UI (dark theme)
├── notebooks/
│   └── Legal_AI_Assistant_Colab.ipynb  # Google Colab demo
├── sample_docs/
│   └── sample_legal.txt     # Sample NDA for testing
├── report/
│   └── project_report.md    # Full academic report
├── scripts/
│   ├── setup.sh             # Install dependencies
│   ├── run.sh               # Start FastAPI
│   └── run_streamlit.sh     # Start Streamlit
├── Dockerfile               # Container deployment
├── requirements.txt
└── .gitignore
```

---

## 🎯 Features

- 📄 **Upload PDF or TXT** legal documents
- 🔍 **Semantic search** via ChromaDB + sentence-transformers
- 🤖 **LLM answers** grounded in your document (no hallucination)
- 📚 **Source citations** — every answer shows which chunks it came from
- 🎨 **Two UIs** — Streamlit app + HTML/JS web interface
- 🐳 **Docker ready** — one command to containerize
- ☁️ **Google Colab** notebook for instant demo

---

## ⚙️ Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_API_URL` | `http://localhost:8000/v1` | vLLM / OpenAI-compatible endpoint |
| `LLM_API_KEY` | `dummy-key` | API key (not needed for local vLLM) |
| `MODEL_NAME` | `meta-llama/Llama-3.2-3B-Instruct` | LLM model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `500` | Characters per chunk |

---

## 🌐 Google Colab Demo

1. Open `notebooks/Legal_AI_Assistant_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Select **Runtime → Change runtime type → T4 GPU**
3. Get a free [Groq API key](https://console.groq.com) (takes 30 seconds)
4. Run all cells → get a **public ngrok URL** to share with your instructor!

---

## 🚀 Deploy vLLM on RunPod (GPU)

```bash
# 1. Go to runpod.io → Deploy → GPU Pod
# 2. Select RTX 3090 or A40 (24GB VRAM)
# 3. Template: vLLM OpenAI-Compatible Server
# 4. Set env: MODEL=meta-llama/Llama-3.2-3B-Instruct
# 5. Expose port 8000
# 6. Copy your pod URL

export LLM_API_URL=https://YOUR-POD-ID-8000.proxy.runpod.net/v1
bash scripts/run.sh
```

---

## 🐳 Docker

```bash
docker build -t legal-ai .
docker run -p 8080:8080 \
  -e LLM_API_URL=https://your-vllm-endpoint/v1 \
  legal-ai
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload PDF or TXT file |
| `POST` | `/query` | Ask a question |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/clear` | Clear all documents |
| `GET` | `/health` | Health check |

### Example
```bash
# Upload a document
curl -X POST http://localhost:8080/upload \
  -F "file=@sample_docs/sample_legal.txt"

# Ask a question
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the termination conditions?"}'
```

---

## 📊 Report

See [`report/project_report.md`](report/project_report.md) for the full academic report including methodology, architecture, evaluation metrics, and deployment guide.

---

## 👥 Authors

Deep Learning Course — End Semester Project  
Track A: Retrieval-Augmented Generation (RAG)
