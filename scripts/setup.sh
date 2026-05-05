#!/usr/bin/env bash
# ── Legal AI Assistant — One-Click Setup ──────────────────────────────────────
# Usage: bash scripts/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ⚖️  Legal AI Assistant — Setup Script       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_ROOT"

# ── Python version check ───────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')
echo "✔ Python: $PYTHON_VERSION"

PY_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    echo "❌ Python 3.9+ is required. Found: $PYTHON_VERSION"
    exit 1
fi

# ── Create virtual environment ─────────────────────────────────────────────────
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment at .venv …"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "✔ Virtual environment already exists at .venv"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "✔ Virtual environment activated"

# ── Upgrade pip ────────────────────────────────────────────────────────────────
echo "→ Upgrading pip …"
pip install --upgrade pip --quiet

# ── Install dependencies ───────────────────────────────────────────────────────
echo "→ Installing Python dependencies (this may take a few minutes) …"
pip install -r requirements.txt

# ── Create runtime directories ─────────────────────────────────────────────────
mkdir -p chroma_db uploads
echo "✔ Directories created: chroma_db/, uploads/"

# ── Pre-download embedding model ──────────────────────────────────────────────
echo "→ Pre-downloading embedding model (all-MiniLM-L6-v2) …"
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('✔ Embedding model cached')"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ✅  Setup complete!                         ║"
echo "║   Run: bash scripts/run.sh                   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
