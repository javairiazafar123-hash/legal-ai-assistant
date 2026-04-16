#!/bin/bash
# ── run.sh — Start the Legal AI Assistant backend ────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "🚀 Starting Legal AI Assistant backend on http://0.0.0.0:8080"
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
