#!/usr/bin/env bash
# Start Himalayan-MHEWS Perpetual Ingestion & ML Optimization Engine in a persistent tmux session.
set -euo pipefail

SESSION_NAME="nepal-mhews-perpetual"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONPATH="${REPO_ROOT}/backend"

mkdir -p "${REPO_ROOT}/backend/logs"
LOG_FILE="${REPO_ROOT}/backend/logs/perpetual_engine.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[info] Tmux session '$SESSION_NAME' is already running."
  echo "       Attach via: tmux attach -t $SESSION_NAME"
  exit 0
fi

echo "[start] Launching perpetual engine in background tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" "bash -c '
  cd \"${REPO_ROOT}\"
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
  export PYTHONPATH=\"${REPO_ROOT}/backend\"
  while true; do
    echo \"=== Starting Perpetual Orchestrator: \$(date -u) ===\" >> \"${LOG_FILE}\"
    \"${REPO_ROOT}/backend/.venv/bin/python\" -u \"${REPO_ROOT}/backend/app/engine/perpetual_orchestrator.py\" >> \"${LOG_FILE}\" 2>&1 || true
    echo \"=== Perpetual engine crashed or exited. Restarting in 10s... ===\" >> \"${LOG_FILE}\"
    sleep 10
  done
'"

echo "[ok] Perpetual engine successfully started in tmux session '$SESSION_NAME'."
echo "     Logs: ${LOG_FILE}"
echo "     Commands to monitor:"
echo "       tmux attach -t $SESSION_NAME   (detach with Ctrl+b d)"
echo "       tail -f ${LOG_FILE}"
