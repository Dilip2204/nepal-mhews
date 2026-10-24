#!/usr/bin/env bash
#
# Install the Himalayan-MHEWS systemd USER units so the 6-month unattended run
# survives a host reboot or a maintenance window (today both long-running
# processes live in ad-hoc tmux sessions and die with the host).
#
# Idempotent and safe to re-run. INSTALL-ONLY by default: it copies the units,
# enables linger, enables the units for next boot, and prints what it did.
# Nothing is started unless you pass --start, and even then it refuses to start
# the orchestrator while the live tmux engine is still running - two concurrent
# orchestrators double-write perpetual_telemetry.db and corrupt the time series.
#
#   bash backend/scripts/install_systemd_services.sh              # install + enable
#   bash backend/scripts/install_systemd_services.sh --start      # also start now
#   bash backend/scripts/install_systemd_services.sh --dry-run    # show, change nothing
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_ROOT="/data-mqs/users/djena/avalanche_detector/nepal-mhews"
UNIT_SRC_DIR="${REPO_ROOT}/backend/deploy/systemd"
UNIT_DEST_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
UNITS=(nepal-mhews-perpetual.service nepal-mhews-api.service)
TMUX_SESSION="nepal-mhews-perpetual"
VENV_PY="${REPO_ROOT}/backend/.venv/bin/python"
ORCHESTRATOR="${REPO_ROOT}/backend/app/engine/perpetual_orchestrator.py"
THREAD_CAP_VARS=(OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS)

DO_START=0
DO_ENABLE=1
DRY_RUN=0

usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${BASH_SOURCE[0]}"
  cat <<'USAGE'
Options:
  --start       Start the units now (still refuses the orchestrator if tmux owns it).
  --no-enable   Copy the unit files only; do not enable them for boot.
  --dry-run     Print every action without changing anything.
  -h, --help    This message.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --start)     DO_START=1 ;;
    --no-enable) DO_ENABLE=0 ;;
    --dry-run)   DRY_RUN=1 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

say()  { echo "$*"; }
run()  { if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
fail() { echo "[error] $*" >&2; exit 1; }

say "=== Himalayan-MHEWS systemd user-unit installer ==="
say "repo root : ${REPO_ROOT}"
say "unit dest : ${UNIT_DEST_DIR}"
[ "$DRY_RUN" -eq 1 ] && say "mode      : DRY RUN (nothing will be changed)"
say ""

# ---------------------------------------------------------------- preflight --
say "--- Preflight ---"

if [ "${REPO_ROOT}" != "${EXPECTED_ROOT}" ]; then
  say "[warn] repo root is not ${EXPECTED_ROOT}."
  say "       The unit files carry ABSOLUTE paths to that location; edit"
  say "       ${UNIT_SRC_DIR}/*.service before installing from a different checkout."
fi

command -v systemctl >/dev/null 2>&1 || fail "systemctl not found; this host has no systemd."
command -v loginctl  >/dev/null 2>&1 || fail "loginctl not found."

if ! systemctl --user show --property=Version >/dev/null 2>&1; then
  fail "no systemd USER manager reachable (XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-unset}).
       Log in over SSH as ${USER} and re-run; do not use 'sudo' - these are user units."
fi
say "[ok] systemd user manager reachable (version $(systemctl --user show --property=Version --value 2>/dev/null))"

for unit in "${UNITS[@]}"; do
  [ -f "${UNIT_SRC_DIR}/${unit}" ] || fail "missing unit file: ${UNIT_SRC_DIR}/${unit}"
done
say "[ok] unit files present in ${UNIT_SRC_DIR}"

[ -x "${VENV_PY}" ]    || fail "venv interpreter not executable: ${VENV_PY}"
[ -f "${ORCHESTRATOR}" ] || fail "orchestrator not found: ${ORCHESTRATOR}"
say "[ok] interpreter $(${VENV_PY} -V 2>&1) at ${VENV_PY}"

# Shared 256-core host: EnvironmentFile= overrides Environment= in systemd
# (systemd.exec(5)), so a thread cap in backend/.env would silently defeat the
# caps baked into the units. Refuse to install in that state.
ENV_FILE="${REPO_ROOT}/backend/.env"
if [ -f "${ENV_FILE}" ]; then
  for var in "${THREAD_CAP_VARS[@]}"; do
    if grep -Eq "^[[:space:]]*${var}=" "${ENV_FILE}"; then
      fail "${ENV_FILE} sets ${var}. EnvironmentFile= overrides Environment=, so this
       would override the mandatory thread caps on this shared 256-core host.
       Remove that line from backend/.env and re-run."
    fi
  done
  say "[ok] backend/.env carries no thread-cap overrides"
else
  say "[info] no backend/.env (optional); units will run on their built-in defaults"
fi

if command -v systemd-analyze >/dev/null 2>&1; then
  if systemd-analyze verify "${UNIT_SRC_DIR}"/*.service >/dev/null 2>&1; then
    say "[ok] systemd-analyze verify passed"
  else
    say "[warn] systemd-analyze verify reported issues (often harmless for user units):"
    systemd-analyze verify "${UNIT_SRC_DIR}"/*.service 2>&1 | sed 's/^/       /' || true
  fi
fi
say ""

# ------------------------------------------------------------------ install --
say "--- Installing unit files ---"
run mkdir -p "${UNIT_DEST_DIR}"
for unit in "${UNITS[@]}"; do
  src="${UNIT_SRC_DIR}/${unit}"
  dest="${UNIT_DEST_DIR}/${unit}"
  if [ -f "${dest}" ] && cmp -s "${src}" "${dest}"; then
    say "[skip] ${unit} already up to date"
  else
    action="installed"; [ -f "${dest}" ] && action="updated"
    run install -m 0644 "${src}" "${dest}"
    say "[ok]   ${unit} ${action}"
  fi
done
run systemctl --user daemon-reload
say "[ok]   systemctl --user daemon-reload"
say ""

# ------------------------------------------------------------------- linger --
say "--- Linger (units must survive logout and reboot) ---"
LINGER="$(loginctl show-user "${USER}" --property=Linger --value 2>/dev/null || echo unknown)"
LINGER_OK=0
if [ "${LINGER}" = "yes" ]; then
  say "[skip] linger already enabled for ${USER}"
  LINGER_OK=1
else
  # `|| true` is load-bearing: enable-linger often needs admin rights, and under
  # `set -e` a non-zero exit would abort the script before the warning below ever
  # printed - installing the units while silently skipping the enable step.
  run loginctl enable-linger "${USER}" || true
  if [ "$DRY_RUN" -eq 0 ]; then
    LINGER_NOW="$(loginctl show-user "${USER}" --property=Linger --value 2>/dev/null || echo unknown)"
    if [ "${LINGER_NOW}" = "yes" ]; then
      say "[ok]   linger enabled for ${USER}"
      LINGER_OK=1
    else
      say "[warn] 'loginctl enable-linger ${USER}' did not take effect (Linger=${LINGER_NOW})."
      say "       Without linger the units stop at logout and do NOT start at boot."
      say "       Ask an admin to run: sudo loginctl enable-linger ${USER}"
    fi
  fi
fi
say ""

# ------------------------------------------------------------------- enable --
if [ "${DO_ENABLE}" -eq 1 ]; then
  say "--- Enabling units for boot ---"
  for unit in "${UNITS[@]}"; do
    run systemctl --user enable "${unit}"
    say "[ok]   enabled ${unit}"
  done
  say ""
fi

# --------------------------------------------------- live tmux engine guard --
say "--- Live orchestrator check ---"
TMUX_RUNNING=0
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  TMUX_RUNNING=1
fi
# -P excludes this script's own descendants and $$ excludes the script itself; without
# them pgrep -f matches the installer's own command line and reports a phantom engine.
PIDS="$(pgrep -u "${USER}" -f 'perpetual_orchestrator\.py' 2>/dev/null \
         | grep -v -w "$$" | tr '\n' ' ' || true)"
PIDS="$(echo ${PIDS})"

if [ "${TMUX_RUNNING}" -eq 1 ]; then
  say "[HOLD] tmux session '${TMUX_SESSION}' is RUNNING (matching PIDs: ${PIDS:-none found})."
  say "       Refusing to start nepal-mhews-perpetual.service: two orchestrators would"
  say "       double-write backend/data/perpetual_telemetry.db and corrupt the time series."
  say ""
  say "       Hand over cleanly, by hand, when you are ready:"
  say "         1) cd ${REPO_ROOT}"
  say "         2) tail -n 20 backend/logs/perpetual_engine.log   # wait for 'Sleeping' between cycles"
  say "         3) tmux kill-session -t ${TMUX_SESSION}           # also kills its restart while-loop"
  say "         4) pgrep -u ${USER} -fa perpetual_orchestrator.py # must print NOTHING"
  say "         5) systemctl --user start nepal-mhews-perpetual.service"
  say "         6) journalctl --user -u nepal-mhews-perpetual.service -f"
  say "            and confirm backend/data/perpetual_status.json updates within ~30 min."
  say "       Do not skip step 4. Never run the tmux engine and the unit at the same time."
elif [ -n "${PIDS}" ]; then
  say "[HOLD] an orchestrator is already running outside tmux (PIDs: ${PIDS})."
  say "       Refusing to start nepal-mhews-perpetual.service. Stop that process first:"
  say "         kill ${PIDS}   # then re-check with: pgrep -u ${USER} -fa perpetual_orchestrator.py"
else
  say "[ok]   no orchestrator currently running (tmux session absent, no matching process)."
fi

API_PORT="$(grep -E '^[[:space:]]*PORT=' "${ENV_FILE}" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"' | tr -d "'" || true)"
API_PORT="${API_PORT:-8000}"
API_BUSY=0
# Capture ss output into a variable FIRST. Piping straight into `grep -q` under
# `set -o pipefail` is a false-negative trap: grep exits on the first match, ss dies
# of SIGPIPE, the pipeline returns 141, and the guard concludes the port is free even
# when it is busy. Verified on this host: the same pattern missed a listening :22.
SS_LISTENERS="$(ss -ltn 2>/dev/null || true)"
if [ -n "${SS_LISTENERS}" ] && printf '%s\n' "${SS_LISTENERS}" | grep -Eq "[:.]${API_PORT}[[:space:]]"; then
  API_BUSY=1
  say "[HOLD] something is already listening on port ${API_PORT}; refusing to start"
  say "       nepal-mhews-api.service. Find and stop it first:  ss -ltnp | grep ':${API_PORT}'"
else
  say "[ok]   port ${API_PORT} is free for nepal-mhews-api.service"
fi
say ""

# -------------------------------------------------------------------- start --
say "--- Start ---"
if [ "${DO_START}" -eq 0 ]; then
  say "[info] install-only mode (no --start): nothing was started."
  if [ "${LINGER_OK}" -eq 1 ]; then
    say "       Linger is on and the units are enabled, so they start after a reboot."
  else
    say "[warn] Linger is NOT enabled, so the units will NOT start at boot and will stop"
    say "       at logout - reboot resilience is NOT yet in place. Fix it with:"
    say "         sudo loginctl enable-linger ${USER}"
    say "       then re-run this script to confirm."
  fi
  say "       To start them now, after the hand-over above:"
  say "         systemctl --user start nepal-mhews-perpetual.service"
  say "         systemctl --user start nepal-mhews-api.service"
else
  if [ "${TMUX_RUNNING}" -eq 1 ] || [ -n "${PIDS}" ]; then
    say "[skip] nepal-mhews-perpetual.service NOT started (see the HOLD above)."
  else
    run systemctl --user start nepal-mhews-perpetual.service
    say "[ok]   started nepal-mhews-perpetual.service"
  fi
  if [ "${API_BUSY}" -eq 1 ]; then
    say "[skip] nepal-mhews-api.service NOT started (port ${API_PORT} already in use)."
  else
    run systemctl --user start nepal-mhews-api.service
    say "[ok]   started nepal-mhews-api.service"
  fi
fi
say ""

# ------------------------------------------------------------------- verify --
say "--- Verification ---"
if [ "$DRY_RUN" -eq 1 ]; then
  say "[dry-run] skipping verification."
else
  for unit in "${UNITS[@]}"; do
    say "  ${unit}"
    say "    is-enabled : $(systemctl --user is-enabled "${unit}" 2>&1 || true)"
    say "    is-active  : $(systemctl --user is-active  "${unit}" 2>&1 || true)"
    systemctl --user --no-pager --lines=0 status "${unit}" 2>&1 | sed -n '1,6p' | sed 's/^/    /' || true
    say ""
  done
fi

cat <<NOTE
--- Note on 'systemctl --user is-system-running' ---
This host currently reports 'degraded' for the user manager. That is a pre-existing
condition caused by unrelated desktop units (xdg-desktop-portal*.service) failing on
a headless server; it does NOT by itself stop these units from starting or restarting.
Check whether it affects MHEWS:

  systemctl --user list-units --state=failed          # the MHEWS units must NOT appear here
  systemctl --user is-active nepal-mhews-perpetual.service
  systemctl --user is-active nepal-mhews-api.service
  journalctl --user -u nepal-mhews-perpetual.service --since '1 hour ago'

If a MHEWS unit does appear in the failed list, read its journal before re-enabling it;
'degraded' is only a problem when one of OUR units is the failing one.

--- Handy commands ---
  journalctl --user -u nepal-mhews-perpetual.service -f
  journalctl --user -u nepal-mhews-api.service -f
  systemctl --user restart nepal-mhews-api.service
  systemctl --user disable --now nepal-mhews-perpetual.service   # full rollback
NOTE
