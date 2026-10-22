#!/usr/bin/env bash
# Start the MHEWS API with offline Ollama evaluation enabled.
# Ollama runs on your PC; forward it first (keep that session open):
#   ssh -R 11434:127.0.0.1:11434 <user>@<this-server>
set -u
cd "$(dirname "$0")/.."

[ -f .env ] && set -a && . ./.env && set +a
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
# Shared 256-core host: cap threads (see AGENTS.md 8.1)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

if curl -s --max-time 4 "$OLLAMA_URL/api/tags" | grep -q "\"$OLLAMA_MODEL\""; then
  echo "[ok] Ollama reachable at $OLLAMA_URL with model $OLLAMA_MODEL"
else
  echo "[warn] Ollama/model '$OLLAMA_MODEL' not reachable at $OLLAMA_URL."
  echo "       Start the SSH reverse tunnel from your PC; API will still run (physics only)."
fi

exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

