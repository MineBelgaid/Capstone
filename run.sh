#!/usr/bin/env bash
# Launch the AI Project Workflow Intelligence dashboard, with a preflight that
# verifies everything the demo needs. Idempotent and safe to re-run.
#
# Checks / sets up, in order:
#   1. Python 3 available
#   2. Ollama installed
#   3. Ollama server running (starts it if not)
#   4. Required models pulled (pulls any missing)
#   5. Python virtualenv with dependencies
#   6. .env file (copied from .env.example)
# Then launches Streamlit.
#
# Usage:
#   ./run.sh           # preflight + launch
#   ./run.sh --check   # verify environment only, don't launch
#   PORT=8600 ./run.sh # custom port

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8501}"
MODELS=("qwen2.5:14b" "nomic-embed-text")
CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

info() { printf '\033[36m[*]  %s\033[0m\n' "$*"; }
good() { printf '\033[32m[OK] %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!]  %s\033[0m\n' "$*"; }
bad()  { printf '\033[31m[X]  %s\033[0m\n' "$*"; }

# 1. Python -------------------------------------------------------------------
PY="$(command -v python3 || command -v python || true)"
if [[ -z "$PY" ]]; then
  bad "Python 3 not found. Install Python 3.10+ and re-run."
  exit 1
fi
good "Python: $($PY --version 2>&1)"

# 2. Ollama installed ---------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  bad "Ollama is not installed."
  warn "Install with:  curl -fsSL https://ollama.com/install.sh | sh   (Linux)"
  warn "or download from: https://ollama.com/download                  (macOS)"
  exit 1
fi
good "Ollama: $(ollama --version 2>&1)"

# 3. Ollama server running ----------------------------------------------------
ollama_up() { curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; }
if ollama_up; then
  good "Ollama server is running"
else
  info "Starting the Ollama server…"
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
  if ollama_up; then good "Ollama server started"; else bad "Ollama server did not come up"; exit 1; fi
fi

# 4. Required models ----------------------------------------------------------
INSTALLED="$(ollama list 2>/dev/null || true)"
for m in "${MODELS[@]}"; do
  if grep -q -- "$m" <<<"$INSTALLED"; then
    good "model present: $m"
  else
    info "pulling model $m  (first time only; qwen2.5:14b is ~9GB)…"
    ollama pull "$m"
    good "pulled $m"
  fi
done

# 5. Virtualenv + deps --------------------------------------------------------
if [[ ! -x ".venv/bin/python" ]]; then
  info "creating virtualenv (.venv)…"
  "$PY" -m venv .venv
fi
if [[ ! -x ".venv/bin/streamlit" ]]; then
  info "installing dependencies (this can take a few minutes)…"
  .venv/bin/python -m pip install --upgrade pip --quiet
  .venv/bin/python -m pip install -r requirements.txt
  good "dependencies installed"
else
  good "dependencies present (.venv)"
fi

# 6. .env ---------------------------------------------------------------------
if [[ ! -f ".env" ]]; then
  cp .env.example .env
  good "created .env from .env.example (defaults to local Ollama)"
else
  good ".env present"
fi

echo
good "Preflight complete."

if [[ "$CHECK_ONLY" == "1" ]]; then
  info "--check specified; not launching the dashboard."
  exit 0
fi

echo
info "Launching dashboard at http://localhost:$PORT  (Ctrl+C to stop)"
exec .venv/bin/python -m streamlit run dashboard/app.py --server.port "$PORT"
