#!/usr/bin/env bash
# Install the Himalayan-MHEWS unattended-operation crontab.
#
# Everything this system needs to survive 6 months without a human is driven from four
# cron jobs plus a log rotation. Nothing here touches the running orchestrator.
#
#   every 15 min   heartbeat_monitor.py      is the engine still producing cycles?
#   Sun 01:30      backup_mhews_data.sh      verified snapshot of the irreplaceable state
#   Sun 02:00      run_weekly_ai_audit.py    dual-model (Gemini + Claude) weekly audit
#   Sun 03:15      logrotate                 cap backend/logs/ growth
#   1st 04:30      db_maintenance.py         integrity + ANALYZE + compacted snapshot
#
# Idempotent: the entries live between marker comments, and re-running replaces that
# block rather than appending duplicates. Lines outside the markers are left untouched.
#
# SAFE BY DEFAULT: this prints the crontab it would install and exits. Pass --install
# to actually write it. The human installs; this script never does it by surprise.
#
# Usage
#   backend/scripts/install_mhews_cron.sh              # dry run (default)
#   backend/scripts/install_mhews_cron.sh --install    # write the crontab
#   backend/scripts/install_mhews_cron.sh --remove     # remove only the managed block
#
# Cron mails stdout/stderr to the local user, so the watchdogs deliberately stay silent
# while healthy and speak up (non-zero exit + output) only when something is wrong.
#
# Exit codes: 0 ok, 1 usage/environment error.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/backend/scripts"
LOG_DIR="${REPO_ROOT}/backend/logs"
LOGROTATE_CONF="${REPO_ROOT}/backend/deploy/logrotate/nepal-mhews"
LOGROTATE_STATE="${LOG_DIR}/.logrotate.state"
PYTHON_BIN="${REPO_ROOT}/backend/.venv/bin/python"

BEGIN_MARKER="# >>> nepal-mhews managed block (install_mhews_cron.sh) >>>"
END_MARKER="# <<< nepal-mhews managed block <<<"

# Shared 256-core host: every entry carries these (AGENTS.md 8.1).
CAPS="OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4"

MODE="dry-run"

log()  { printf '%s [cron-install] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
err()  { printf '%s [cron-install] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
warn() { printf '%s [cron-install] WARNING: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

usage() { sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --install)  MODE="install"; shift ;;
    --remove)   MODE="remove"; shift ;;
    --dry-run)  MODE="dry-run"; shift ;;
    -h|--help)  usage ;;
    *) err "unknown argument: $1"; exit 1 ;;
  esac
done

command -v crontab >/dev/null 2>&1 || { err "crontab not found on PATH"; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight. A cron entry pointing at a missing script fails silently forever.
# ---------------------------------------------------------------------------
PREFLIGHT_FAILED=0
for required in \
  "${PYTHON_BIN}" \
  "${SCRIPTS_DIR}/heartbeat_monitor.py" \
  "${SCRIPTS_DIR}/db_maintenance.py" \
  "${SCRIPTS_DIR}/backup_mhews_data.sh" \
  "${SCRIPTS_DIR}/run_weekly_ai_audit.py"
do
  if [ ! -e "${required}" ]; then
    err "missing: ${required}"
    PREFLIGHT_FAILED=1
  fi
done
if [ ! -e "${LOGROTATE_CONF}" ]; then
  warn "logrotate config missing: ${LOGROTATE_CONF} (its cron entry will be a no-op)"
fi
if ! command -v logrotate >/dev/null 2>&1 && [ ! -x /usr/sbin/logrotate ]; then
  warn "logrotate binary not found; backend/logs/ will not be rotated"
fi
if [ "${PREFLIGHT_FAILED}" -ne 0 ]; then
  err "pre-flight failed; refusing to build a crontab that points at missing files"
  exit 1
fi

# The audit's dual-engine mode is owned by another module and may not have landed yet.
if [ -f "${SCRIPTS_DIR}/run_weekly_ai_audit.py" ]; then
  if ! grep -q '"dual"' "${SCRIPTS_DIR}/run_weekly_ai_audit.py" 2>/dev/null; then
    warn "run_weekly_ai_audit.py does not yet advertise an '--engine dual' choice."
    warn "The Sunday 02:00 entry below uses it as the user asked; verify it by hand with:"
    warn "  cd ${REPO_ROOT} && PYTHONPATH=backend ${PYTHON_BIN} \\"
    warn "    backend/scripts/run_weekly_ai_audit.py --engine dual --dry-run --limit-bulletins 2 --limit-visual 1"
  fi
fi

mkdir -p "${LOG_DIR}"
log "log directory: ${LOG_DIR}"

LOGROTATE_BIN="$(command -v logrotate || echo /usr/sbin/logrotate)"

# ---------------------------------------------------------------------------
# The managed block.
# ---------------------------------------------------------------------------
build_block() {
  cat <<BLOCK
${BEGIN_MARKER}
# Managed by backend/scripts/install_mhews_cron.sh - edits inside this block are
# overwritten on the next run. Put your own entries outside the markers.
#
# CRON_TZ makes every schedule below UTC, because the user specified "Sunday 02:00 UTC"
# for the audit and this host runs in $(date +%Z). cronie applies CRON_TZ to all entries
# that FOLLOW it in the file, so any entry you add after this block inherits UTC too -
# add your own entries ABOVE the opening marker, or set CRON_TZ back explicitly.
CRON_TZ=UTC

# Liveness watchdog. Silent while healthy; non-zero exit + output (hence a cron mail)
# when perpetual_status.json goes stale or cycle_count stops advancing.
*/15 * * * * cd ${REPO_ROOT} && env ${CAPS} PYTHONPATH=backend ${PYTHON_BIN} backend/scripts/heartbeat_monitor.py --quiet

# Weekly verified backup, deliberately BEFORE the audit so the ground-truth catalog is
# captured as it stood before the audit appends to it.
30 1 * * 0 cd ${REPO_ROOT} && env ${CAPS} bash backend/scripts/backup_mhews_data.sh --keep 8 >> backend/logs/backup_cron.log 2>&1

# Weekly dual-model (Gemini + Claude) AI audit. Command specified by the user.
# NUMEXPR_NUM_THREADS is not listed here because run_weekly_ai_audit.py sets all four
# thread caps itself via os.environ.setdefault before importing numpy.
0 2 * * 0 cd ${REPO_ROOT} && env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=backend ${PYTHON_BIN} backend/scripts/run_weekly_ai_audit.py --engine dual --limit-bulletins 50 --limit-visual 20 >> backend/logs/weekly_audit_cron.log 2>&1

# Weekly log rotation. The -s state file is mandatory for a non-root logrotate.
15 3 * * 0 cd ${REPO_ROOT} && ${LOGROTATE_BIN} -s ${LOGROTATE_STATE} ${LOGROTATE_CONF} >> backend/logs/logrotate_cron.log 2>&1

# Monthly database maintenance: integrity check, row census, ANALYZE, and a compacted
# snapshot taken without locking the live engine (see db_maintenance.py docstring).
30 4 1 * * cd ${REPO_ROOT} && env ${CAPS} PYTHONPATH=backend ${PYTHON_BIN} backend/scripts/db_maintenance.py --mode compact-copy --keep 2 >> backend/logs/db_maintenance_cron.log 2>&1
${END_MARKER}
BLOCK
}

current_crontab() { crontab -l 2>/dev/null || true; }

# Everything outside the managed markers, preserved verbatim.
strip_managed_block() {
  awk -v b="${BEGIN_MARKER}" -v e="${END_MARKER}" '
    $0 == b { inblock = 1; next }
    $0 == e { inblock = 0; next }
    !inblock { print }
  '
}

EXISTING="$(current_crontab)"
PRESERVED="$(printf '%s\n' "${EXISTING}" | strip_managed_block | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}')"

case "${MODE}" in
  remove)
    log "removing the managed block (other entries are preserved)"
    if [ -z "${PRESERVED//[[:space:]]/}" ]; then
      crontab -r 2>/dev/null || true
      log "crontab is now empty"
    else
      printf '%s\n' "${PRESERVED}" | crontab -
      log "managed block removed"
    fi
    echo "----- resulting crontab -----"
    current_crontab
    exit 0
    ;;
  dry-run|install)
    NEW_CRONTAB="$(printf '%s\n%s\n' "${PRESERVED}" "$(build_block)" | sed -e '/./,$!d')"
    ;;
esac

if printf '%s\n' "${EXISTING}" | grep -qF "${BEGIN_MARKER}"; then
  log "an existing managed block was found; it will be REPLACED (not duplicated)"
else
  log "no existing managed block; a new one will be added"
fi

if [ "${MODE}" = "dry-run" ]; then
  log "DRY RUN - nothing installed. Re-run with --install to apply."
  echo "----- crontab that WOULD be installed -----"
  printf '%s\n' "${NEW_CRONTAB}"
  echo "-------------------------------------------"
  exit 0
fi

printf '%s\n' "${NEW_CRONTAB}" | crontab -
log "crontab installed"
echo "----- resulting crontab -----"
current_crontab
echo "-----------------------------"
log "note: cron mails stdout/stderr to the local user. Set MAILTO= at the top of the"
log "      crontab, or point the entries at a file, if that mailbox is not watched."
exit 0
