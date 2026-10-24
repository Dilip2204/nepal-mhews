#!/usr/bin/env bash
# Weekly backup of the Himalayan-MHEWS irreplaceable state.
#
# What is backed up and why
#   perpetual_telemetry.db            6 months of 30-minute telemetry and physics
#                                     calculations. Cannot be re-fetched: Open-Meteo
#                                     and USGS only serve a short history window.
#   live_ground_truth_feedback.csv    Verified landslide/avalanche labels. These are
#                                     the training signal for the whole 6-month run.
#   active_equation_parameters.json   The champion 7-factor equation weights that the
#                                     continuous optimizer promoted.
#
# Why not `cp` the database
#   The perpetual orchestrator writes every 30 minutes. Copying a live SQLite file with
#   cp/rsync/tar can capture a page-torn database - the copy opens fine and only fails
#   later, when you need it. This script uses the SQLite ONLINE BACKUP API through the
#   venv Python (sqlite3.Connection.backup), which takes a consistent snapshot while the
#   engine keeps writing, then verifies the snapshot with PRAGMA quick_check before it is
#   compressed. A backup that is not verified is just a hope.
#
#   The venv Python is used deliberately: the `sqlite3` binary first on PATH here belongs
#   to a Siemens toolchain (/tools/SIEMENS/...), not the system, so its behaviour and
#   version are not something this system should depend on.
#
# Usage
#   backend/scripts/backup_mhews_data.sh [--out DIR] [--keep N] [--dry-run]
#
#   --out DIR   backup root (default: backend/data/backups)
#   --keep N    backup sets to retain (default: 8, i.e. ~2 months of weeklies)
#   --dry-run   report what would happen, write nothing
#
# Exit codes: 0 ok, 1 usage/environment error, 2 a backup step failed.

set -euo pipefail

# Shared 256-core host: cap threads (AGENTS.md 8.1).
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/backend/data"
PYTHON_BIN="${REPO_ROOT}/backend/.venv/bin/python"

DB_PATH="${DATA_DIR}/perpetual_telemetry.db"
GT_CSV_PATH="${DATA_DIR}/processed/live_ground_truth_feedback.csv"
EQ_JSON_PATH="${DATA_DIR}/active_equation_parameters.json"

BACKUP_ROOT="${DATA_DIR}/backups"
KEEP=8
DRY_RUN=0
PREFIX="mhews-backup-"

log() { printf '%s [backup] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
err() { printf '%s [backup] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --out)     BACKUP_ROOT="${2:?--out needs a directory}"; shift 2 ;;
    --keep)    KEEP="${2:?--keep needs a number}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) err "unknown argument: $1"; exit 1 ;;
  esac
done

case "${KEEP}" in
  ''|*[!0-9]*) err "--keep must be a non-negative integer, got '${KEEP}'"; exit 1 ;;
esac
if [ "${KEEP}" -lt 1 ]; then err "--keep must be >= 1"; exit 1; fi

if [ ! -x "${PYTHON_BIN}" ]; then
  err "venv python not found or not executable: ${PYTHON_BIN}"
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SET_DIR="${BACKUP_ROOT}/${PREFIX}${STAMP}"
MANIFEST="${SET_DIR}/MANIFEST.txt"
FAILURES=0

log "repo        : ${REPO_ROOT}"
log "backup set  : ${SET_DIR}"
log "retention   : newest ${KEEP} sets"
if [ "${DRY_RUN}" -eq 1 ]; then
  log "DRY RUN - nothing will be written"
  log "would snapshot : ${DB_PATH}"
  log "would copy     : ${GT_CSV_PATH}"
  log "would copy     : ${EQ_JSON_PATH}"
  EXISTING=0
  if [ -d "${BACKUP_ROOT}" ]; then
    EXISTING=$(find "${BACKUP_ROOT}" -maxdepth 1 -type d -name "${PREFIX}*" | wc -l | tr -d ' ')
  fi
  log "existing sets  : ${EXISTING} (would prune to ${KEEP})"
  exit 0
fi

mkdir -p "${SET_DIR}"
{
  echo "Himalayan-MHEWS backup set"
  echo "created_utc : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host        : $(hostname)"
  echo "repo        : ${REPO_ROOT}"
  echo "method      : sqlite3 online backup API + gzip -9"
  echo ""
} > "${MANIFEST}"

record() {
  # record <label> <source> <artifact> <extra-note>
  local label="$1" src="$2" artifact="$3" note="${4:-}"
  local size sha
  size="$(stat -c '%s' "${artifact}" 2>/dev/null || echo 0)"
  sha="$(sha256sum "${artifact}" 2>/dev/null | cut -d' ' -f1 || echo unavailable)"
  {
    echo "[${label}]"
    echo "  source   : ${src}"
    echo "  artifact : $(basename "${artifact}")"
    echo "  bytes    : ${size}"
    echo "  sha256   : ${sha}"
    if [ -n "${note}" ]; then echo "  note     : ${note}"; fi
    echo ""
  } >> "${MANIFEST}"
}

# ---------------------------------------------------------------------------
# 1. SQLite telemetry database - online backup, verified, then compressed.
# ---------------------------------------------------------------------------
if [ -f "${DB_PATH}" ]; then
  SNAPSHOT="${SET_DIR}/perpetual_telemetry.db"
  log "snapshotting database via the SQLite online backup API..."
  DB_SUMMARY_FILE="${SET_DIR}/.db_summary"
  if "${PYTHON_BIN}" - "${DB_PATH}" "${SNAPSHOT}" > "${DB_SUMMARY_FILE}" <<'PYEOF'
import sqlite3
import sys

src_path, dest_path = sys.argv[1], sys.argv[2]

# Read-only source so a backup can never modify production data.
src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True, timeout=60.0)
src.execute("PRAGMA busy_timeout = 60000")
dest = sqlite3.connect(dest_path, timeout=60.0)
try:
    # pages/sleep keep the copy cooperative so the 30-minute engine write is
    # never starved; SQLite restarts the copy automatically if a writer commits.
    src.backup(dest, pages=512, sleep=0.05)
finally:
    dest.close()
    src.close()

check = sqlite3.connect(dest_path)
try:
    verdict = [row[0] for row in check.execute("PRAGMA quick_check")]
    rows = {}
    for (name,) in check.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ):
        rows[name] = check.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
finally:
    check.close()

if verdict != ["ok"]:
    print(f"quick_check failed: {verdict}", file=sys.stderr)
    sys.exit(1)

print("rows: " + ", ".join(f"{k}={v}" for k, v in rows.items()))
PYEOF
  then
    DB_SUMMARY="$(cat "${DB_SUMMARY_FILE}" 2>/dev/null || echo '')"
    rm -f "${DB_SUMMARY_FILE}"
    log "database snapshot verified (quick_check ok; ${DB_SUMMARY})"
    if gzip -9 "${SNAPSHOT}"; then
      record "telemetry_database" "${DB_PATH}" "${SNAPSHOT}.gz" \
             "online backup API, quick_check ok, ${DB_SUMMARY}"
      log "database archived: $(basename "${SNAPSHOT}.gz") ($(stat -c '%s' "${SNAPSHOT}.gz") bytes)"
    else
      err "gzip failed for ${SNAPSHOT}"
      FAILURES=$((FAILURES + 1))
    fi
  else
    err "SQLite online backup failed; the database was NOT archived this run"
    rm -f "${SNAPSHOT}" "${DB_SUMMARY_FILE}"
    FAILURES=$((FAILURES + 1))
  fi
else
  err "database not found: ${DB_PATH}"
  FAILURES=$((FAILURES + 1))
fi

# ---------------------------------------------------------------------------
# 2. Flat files - copy, validate, compress. Validation catches a mid-append read.
# ---------------------------------------------------------------------------
backup_flat_file() {
  local label="$1" src="$2" kind="$3"
  if [ ! -f "${src}" ]; then
    err "${label} not found: ${src}"
    FAILURES=$((FAILURES + 1))
    return
  fi
  local dest="${SET_DIR}/$(basename "${src}")"
  local check_out="${SET_DIR}/.validate_out"
  local attempt note=""
  for attempt in 1 2; do
    cp -p "${src}" "${dest}"
    if "${PYTHON_BIN}" - "${dest}" "${kind}" > "${check_out}" <<'PYEOF'
import csv
import json
import sys

path, kind = sys.argv[1], sys.argv[2]
if kind == "json":
    with open(path, "r", encoding="utf-8") as handle:
        json.load(handle)
    print("valid json")
else:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        print("empty csv", file=sys.stderr)
        sys.exit(1)
    width = len(rows[0])
    ragged = [i for i, r in enumerate(rows[1:], start=2) if r and len(r) != width]
    if ragged:
        print(f"ragged csv rows: {ragged[:5]}", file=sys.stderr)
        sys.exit(1)
    print(f"valid csv, {len(rows) - 1} data rows")
PYEOF
    then
      note="$(cat "${check_out}" 2>/dev/null || echo "validated ${kind}")"
      break
    fi
    if [ "${attempt}" -eq 1 ]; then
      err "${label} failed validation (possibly caught mid-write); retrying in 5s"
      sleep 5
    else
      err "${label} failed validation twice; archiving it anyway and flagging the set"
      note="VALIDATION FAILED - inspect before trusting"
      FAILURES=$((FAILURES + 1))
    fi
  done
  rm -f "${check_out}"
  if gzip -9 -f "${dest}"; then
    record "${label}" "${src}" "${dest}.gz" "${note}"
    log "${label} archived: $(basename "${dest}.gz")"
  else
    err "gzip failed for ${dest}"
    FAILURES=$((FAILURES + 1))
  fi
}

backup_flat_file "ground_truth_catalog" "${GT_CSV_PATH}" "csv"
backup_flat_file "active_equation_parameters" "${EQ_JSON_PATH}" "json"

# ---------------------------------------------------------------------------
# 3. Retention - keep the newest N sets. Timestamps sort lexicographically.
# ---------------------------------------------------------------------------
mapfile -t ALL_SETS < <(find "${BACKUP_ROOT}" -maxdepth 1 -type d -name "${PREFIX}*" | sort -r)
if [ "${#ALL_SETS[@]}" -gt "${KEEP}" ]; then
  for stale in "${ALL_SETS[@]:${KEEP}}"; do
    log "retention: removing ${stale}"
    rm -rf "${stale}"
  done
fi
log "retention: $(find "${BACKUP_ROOT}" -maxdepth 1 -type d -name "${PREFIX}*" | wc -l | tr -d ' ') set(s) on disk"

{
  echo "failures    : ${FAILURES}"
  echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${MANIFEST}"

if [ "${FAILURES}" -gt 0 ]; then
  err "${FAILURES} backup step(s) failed - see ${MANIFEST}"
  exit 2
fi

log "backup complete: ${SET_DIR}"
exit 0
