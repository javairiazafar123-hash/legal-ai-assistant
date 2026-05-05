"""
Legal AI Assistant - Streamlit Interface
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import os

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(
    page_title="⚖️ Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0f1117; }
    .stApp { background: #0f1117; }
    .source-box {
        background: #1a1d27;
        border-left: 3px solid #6c63ff;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .source-meta { color: #a78bfa; font-size: 0.75rem; margin-bottom: 4px; }
    .answer-box {
        background: #1a1d27;
        border: 1px solid #2e3250;
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 0.95rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.ok, r.json() if r.ok else {}
    except Exception:
        return False, {}

def upload_file(file_bytes, filename):
    try:
        r = requests.post(
            f"{API_URL}/upload",
            files={"file": (filename, file_bytes)},
            timeout=60,
        )
        return r.ok, r.json() if r.ok else {"detail": "Upload failed"}
    except Exception as e:
        return False, {"detail": str(e)}

def query_rag(question):
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=120,
        )
        return r.ok, r.json() if r.ok else {"detail": "Query failed"}
    except Exception as e:
        return False, {"detail": str(e)}

def list_documents():
    try:
        r = requests.get(f"{API_URL}/documents", timeout=5)
        return r.json() if r.ok else []
    except Exception:
        return []

def clear_documents():
    try:
        r = requests.delete(f"{API_URL}/clear", timeout=10)
        return r.ok
    except Exception:
        return False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Legal AI Assistant")
    st.markdown("*Powered by RAG + vLLM*")
    st.divider()

    # Health status
    healthy, health_data = check_health()
    if healthy:
        model = health_data.get("model", "unknown").split("/")[-1]
        st.success(f"🟢 Online · `{model}`")
    else:
        st.error("🔴 Backend offline")
        st.info(f"Start the backend:\n```\ncd dl-project\nbash scripts/run.sh\n```")

    st.divider()

    # Upload
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload legal documents to query",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if st.button(f"📤 Upload: {uf.name}", key=f"up_{uf.name}"):
                with st.spinner(f"Processing {uf.name}…"):
                    ok, result = upload_file(uf.read(), uf.name)
                if ok:
                    st.success(f"✅ {result.get('chunks', '?')} chunks added")
                else:
                    st.error(f"❌ {result.get('detail', 'Error')}")

    st.divider()

    # Document list
    st.markdown("### 📋 Loaded Documents")
    docs = list_documents()
    if docs:
        for d in docs:
            icon = "📕" if d["filename"].endswith(".pdf") else "📄"
            st.markdown(f"{icon} **{d['filename']}** — `{d['chunks']} chunks`")
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if clear_documents():
                st.success("Cleared!")
                st.rerun()
            else:
                st.error("Failed to clear")
    else:
        st.caption("No documents loaded yet.")

    st.divider()
    st.caption("Deep Learning End-Semester Project\nTrack A: RAG\n\n[GitHub](https://github.com) · [Report](report/project_report.md)")

# ── Main Chat UI ──────────────────────────────────────────────────────────────
st.title("⚖️ Legal Document Q&A")
st.caption("Ask questions about your uploaded legal documents. Answers are grounded in the document text.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<div class="source-meta">📎 <strong>{s["filename"]}</strong> · chunk {s["chunk_index"]}</div>'
                        f'{s["content"][:300]}…'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Suggested questions (only when no history)
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    suggestions = [
        "What are the termination conditions?",
        "Summarise the confidentiality clause",
        "What is the governing law?",
        "Explain the non-compete terms",
    ]
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, key=f"sug_{i}"):
                st.session_state.pending_question = s
                st.rerun()

# Handle pending question from suggestion button
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
else:
    question = None

# Chat input
user_input = st.chat_input("Ask a question about your legal documents…")
if user_input:
    question = user_input

# Process question
if question:
    if not healthy:
        st.error("❌ Backend is offline. Please start the server first.")
    elif not docs:
        st.warning("⚠️ No documents loaded. Please upload a document first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.write(question)

        # Query backend
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Searching documents and generating answer…"):
                ok, result = query_rag(question)

            if ok:
                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])
                elapsed = result.get("elapsed_ms")

                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                if elapsed:
                    st.caption(f"⏱️ {elapsed/1000:.2f}s")

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        for s in sources:
                            st.markdown(
                                f'<div class="source-box">'
                                f'<div class="source-meta">📎 <strong>{s["filename"]}</strong> · chunk {s["chunk_index"]} · score {round(1-s.get("score",0), 3)}</div>'
                                f'{s["content"][:400]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            else:
                err = result.get("detail", "Unknown error")
                st.error(f"❌ {err}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {err}",
                })
