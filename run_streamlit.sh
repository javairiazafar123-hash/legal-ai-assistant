#!/bin/bash
# ── run_streamlit.sh — Start the Streamlit UI ────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "🎨 Starting Streamlit UI on http://localhost:8501"
streamlit run app/streamlit_app.py --server.port 8501
