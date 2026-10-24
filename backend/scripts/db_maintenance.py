#!/usr/bin/env python3
"""Monthly SQLite maintenance for the Himalayan-MHEWS perpetual telemetry store.

Why this exists
---------------
The perpetual orchestrator appends telemetry for 491 sites every 30 minutes. Over a
6-month unattended run that is roughly 4.24M rows and 1.2-2.0 GB in
``backend/data/perpetual_telemetry.db``. Without periodic ANALYZE the query planner
works from stale statistics, and without periodic compaction deleted/churned pages
are never returned to the filesystem.

The locking problem (read this before changing the default mode)
----------------------------------------------------------------
A plain ``VACUUM`` rewrites the whole database and needs an EXCLUSIVE lock for the
entire rewrite. The live engine writes every 30 minutes, so a plain VACUUM either

  * fails immediately with "database is locked", or
  * blocks the engine's next write for however long the rewrite takes
    (minutes, once the file reaches gigabytes).

Either outcome is unacceptable for an unattended early-warning system: a blocked
write cycle is a cycle of missing telemetry.

``VACUUM INTO 'file'`` solves this upstream - it streams a compacted copy out while
holding only a read lock. **It is not available here.** This host's venv Python 3.12
is linked against SQLite 3.26.0; ``VACUUM INTO`` landed in SQLite 3.27.0. The
statement fails with ``near "INTO": syntax error``.

So the default mode (``compact-copy``) emulates it with two steps that are each safe:

  1. ``sqlite3.Connection.backup()`` - the C online-backup API. It copies the live DB
     page-by-page, yields between batches, and transparently restarts if a writer
     commits mid-copy. The live DB is never held hostage.
  2. plain ``VACUUM`` on that private copy - exclusive lock on a file nobody else has
     open, so it is instant and contention-free.

The result is a compacted snapshot plus an exact measurement of how many bytes a real
VACUUM would reclaim, with zero risk to the running engine. This script never swaps
the compacted copy in for the live database; doing that safely requires stopping the
orchestrator, which is a human decision.

``--mode vacuum`` performs the real in-place VACUUM and is deliberately behind a flag.
Run it only during a maintenance window with the orchestrator stopped.

Exit codes
----------
0 success, 1 unexpected error, 2 database locked / busy, 3 integrity check failed.

Examples
--------
    # default monthly run (safe against the live engine)
    PYTHONPATH=backend backend/.venv/bin/python backend/scripts/db_maintenance.py

    # see what it would do, touch nothing
    ... db_maintenance.py --dry-run

    # real in-place compaction, ONLY with the orchestrator stopped
    ... db_maintenance.py --mode vacuum
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "backend" / "data"
DEFAULT_DB_PATH = DATA_DIR / "perpetual_telemetry.db"
DEFAULT_OUT_DIR = DATA_DIR / "maintenance"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LOCKED = 2
EXIT_INTEGRITY = 3

# VACUUM INTO requires SQLite >= 3.27.0 (see module docstring).
_VACUUM_INTO_MIN = (3, 27, 0)


def _log(message: str, stream: Any = sys.stdout) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} [db_maintenance] {message}", file=stream, flush=True)


def _warn(message: str) -> None:
    _log(f"WARNING: {message}", stream=sys.stderr)


def _human_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:,.1f} PB"


def _sqlite_version_tuple() -> Tuple[int, ...]:
    return tuple(int(part) for part in sqlite3.sqlite_version.split("."))


def vacuum_into_supported() -> bool:
    """True when the linked SQLite can run ``VACUUM INTO`` natively."""
    return _sqlite_version_tuple() >= _VACUUM_INTO_MIN


def _is_lock_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def _connect(db_path: Path, busy_timeout_ms: int, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=busy_timeout_ms / 1000.0)
    else:
        conn = sqlite3.connect(str(db_path), timeout=busy_timeout_ms / 1000.0)
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    return conn


def _db_footprint(db_path: Path) -> Dict[str, int]:
    """On-disk bytes of the database and its journal/WAL sidecars."""
    sizes: Dict[str, int] = {}
    for suffix in ("", "-journal", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            sizes[candidate.name] = candidate.stat().st_size
    sizes["_total"] = sum(v for k, v in sizes.items() if not k.startswith("_"))
    return sizes


def collect_table_rows(conn: sqlite3.Connection) -> Dict[str, Optional[int]]:
    """Row count per user table. None means the count could not be taken."""
    rows: Dict[str, Optional[int]] = {}
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for name in names:
        try:
            rows[name] = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except sqlite3.Error as exc:
            _warn(f"could not count rows in {name}: {exc}")
            rows[name] = None
    return rows


def collect_space_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    return {
        "page_size": int(page_size),
        "page_count": int(page_count),
        "freelist_count": int(freelist),
        "logical_bytes": int(page_size) * int(page_count),
        "free_bytes": int(page_size) * int(freelist),
    }


def run_integrity_check(conn: sqlite3.Connection, quick: bool) -> List[str]:
    """Return the integrity findings. ``['ok']`` means the database is sound."""
    pragma = "quick_check" if quick else "integrity_check"
    return [str(row[0]) for row in conn.execute(f"PRAGMA {pragma}")]


def _backup_live_db(src: sqlite3.Connection, dest_path: Path, busy_timeout_ms: int) -> None:
    """Stream a consistent copy out via the online-backup API.

    ``pages``/``sleep`` keep the copy cooperative: SQLite drops its read lock between
    batches so the 30-minute engine write is never starved.
    """
    dest = sqlite3.connect(str(dest_path), timeout=busy_timeout_ms / 1000.0)
    try:
        dest.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        src.backup(dest, pages=512, sleep=0.05)
    finally:
        dest.close()


def _prune_old_copies(out_dir: Path, pattern: str, keep: int) -> List[Path]:
    copies = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    removed: List[Path] = []
    for stale in copies[keep:]:
        try:
            stale.unlink()
            removed.append(stale)
        except OSError as exc:
            _warn(f"could not remove old copy {stale}: {exc}")
    return removed


def perform_compact_copy(
    db_path: Path,
    out_dir: Path,
    busy_timeout_ms: int,
    keep: int,
) -> Dict[str, Any]:
    """Write a VACUUMed snapshot without ever locking the live database."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_path = out_dir / f"{db_path.stem}.compact-{stamp}.db"
    staging_path = out_dir / f".{db_path.stem}.staging-{stamp}.db"

    for leftover in (final_path, staging_path):
        if leftover.exists():
            leftover.unlink()

    started = time.time()
    src = _connect(db_path, busy_timeout_ms, read_only=True)
    try:
        _log("streaming online backup of the live database (engine stays writable)...")
        _backup_live_db(src, staging_path, busy_timeout_ms)
    finally:
        src.close()
    copied_bytes = staging_path.stat().st_size
    _log(f"online backup complete: {_human_bytes(copied_bytes)} in {time.time() - started:.1f}s")

    # VACUUM on the private copy: exclusive lock on a file nobody else has open.
    copy_conn = sqlite3.connect(str(staging_path))
    try:
        _log("compacting the private copy (plain VACUUM, no contention)...")
        copy_conn.execute("VACUUM")
        copy_conn.commit()
    finally:
        copy_conn.close()

    for sidecar in ("-journal", "-wal", "-shm"):
        leftover = Path(str(staging_path) + sidecar)
        if leftover.exists():
            leftover.unlink()

    staging_path.replace(final_path)
    compacted_bytes = final_path.stat().st_size
    removed = _prune_old_copies(out_dir, f"{db_path.stem}.compact-*.db", keep)
    for path in removed:
        _log(f"retention: removed old snapshot {path.name}")

    return {
        "snapshot_path": str(final_path),
        "snapshot_bytes": compacted_bytes,
        "pre_compaction_bytes": copied_bytes,
        "reclaimable_bytes": max(0, copied_bytes - compacted_bytes),
        "elapsed_sec": round(time.time() - started, 2),
        "retained_snapshots": keep,
        "removed_snapshots": [p.name for p in removed],
    }


def perform_inplace_vacuum(db_path: Path, busy_timeout_ms: int) -> Dict[str, Any]:
    """Real in-place VACUUM. Only safe with the orchestrator stopped."""
    before = db_path.stat().st_size
    started = time.time()
    conn = _connect(db_path, busy_timeout_ms)
    try:
        conn.isolation_level = None  # VACUUM cannot run inside a transaction
        conn.execute("VACUUM")
    finally:
        conn.close()
    after = db_path.stat().st_size
    return {
        "bytes_before": before,
        "bytes_after": after,
        "bytes_reclaimed": max(0, before - after),
        "elapsed_sec": round(time.time() - started, 2),
    }


def perform_analyze(db_path: Path, busy_timeout_ms: int) -> Dict[str, Any]:
    """Refresh planner statistics. Takes a short write lock."""
    started = time.time()
    conn = _connect(db_path, busy_timeout_ms)
    try:
        conn.execute("ANALYZE")
        conn.execute("PRAGMA optimize")
        conn.commit()
    finally:
        conn.close()
    return {"elapsed_sec": round(time.time() - started, 2)}


def _print_report(report: Dict[str, Any]) -> None:
    space = report["space_before"]
    _log(f"database        : {report['db_path']}")
    _log(f"sqlite library  : {report['sqlite_version']} "
         f"(VACUUM INTO native: {report['vacuum_into_native']})")
    _log(f"file on disk    : {_human_bytes(report['footprint_before']['_total'])}")
    _log(f"pages           : {space['page_count']:,} x {space['page_size']} B "
         f"= {_human_bytes(space['logical_bytes'])}")
    _log(f"free pages      : {space['freelist_count']:,} "
         f"({_human_bytes(space['free_bytes'])} reclaimable in-place)")
    _log(f"integrity       : {report['integrity_mode']} -> {', '.join(report['integrity'])}")
    total_rows = 0
    for table, count in report["tables"].items():
        if count is None:
            _log(f"  rows {table:<32} unavailable")
        else:
            total_rows += count
            _log(f"  rows {table:<32} {count:,}")
    _log(f"  rows {'TOTAL':<32} {total_rows:,}")


def build_report(
    db_path: Path,
    busy_timeout_ms: int,
    quick: bool,
) -> Dict[str, Any]:
    footprint_before = _db_footprint(db_path)
    conn = _connect(db_path, busy_timeout_ms, read_only=True)
    try:
        space_before = collect_space_stats(conn)
        tables = collect_table_rows(conn)
        integrity = run_integrity_check(conn, quick=quick)
    finally:
        conn.close()
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "sqlite_version": sqlite3.sqlite_version,
        "vacuum_into_native": vacuum_into_supported(),
        "footprint_before": footprint_before,
        "space_before": space_before,
        "tables": tables,
        "integrity_mode": "quick_check" if quick else "integrity_check",
        "integrity": integrity,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monthly SQLite maintenance for the MHEWS perpetual telemetry store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  compact-copy  (default) online backup + VACUUM on the copy. Safe while the\n"
            "                perpetual engine is running. Never touches the live file.\n"
            "  report        read-only: sizes, row counts, integrity check. No writes.\n"
            "  vacuum        real in-place VACUUM. Needs an exclusive lock - run ONLY with\n"
            "                the orchestrator stopped.\n"
        ),
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument(
        "--mode",
        choices=["compact-copy", "report", "vacuum"],
        default="compact-copy",
        help="maintenance strategy (default: compact-copy)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="where compacted snapshots are written")
    parser.add_argument("--keep", type=int, default=2,
                        help="compacted snapshots to retain (default: 2)")
    parser.add_argument("--busy-timeout-ms", type=int, default=30000,
                        help="SQLite busy timeout before giving up (default: 30000)")
    parser.add_argument("--quick", action="store_true",
                        help="use PRAGMA quick_check instead of the full integrity_check")
    parser.add_argument("--no-analyze", action="store_true",
                        help="skip ANALYZE / PRAGMA optimize on the live database")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only; make no writes of any kind")
    parser.add_argument("--json", action="store_true",
                        help="emit the machine-readable report on stdout")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    db_path: Path = args.db.resolve()

    if not db_path.exists():
        _log(f"ERROR: database not found: {db_path}", stream=sys.stderr)
        return EXIT_ERROR

    if args.keep < 1:
        _log("ERROR: --keep must be >= 1", stream=sys.stderr)
        return EXIT_ERROR

    try:
        report = build_report(db_path, args.busy_timeout_ms, quick=args.quick)
    except sqlite3.OperationalError as exc:
        if _is_lock_error(exc):
            _log(f"ERROR: database is locked ({exc}). The perpetual engine is probably "
                 f"mid-write. Re-run later; not waiting.", stream=sys.stderr)
            return EXIT_LOCKED
        _log(f"ERROR: {exc}", stream=sys.stderr)
        return EXIT_ERROR

    if not args.json:
        _print_report(report)

    integrity_ok = report["integrity"] == ["ok"]
    if not integrity_ok:
        _log(f"ERROR: integrity check FAILED: {report['integrity']}", stream=sys.stderr)
        report["result"] = "integrity_failed"
        if args.json:
            print(json.dumps(report, indent=2), flush=True)
        return EXIT_INTEGRITY

    effective_mode = "report" if args.dry_run else args.mode
    report["mode"] = args.mode
    report["dry_run"] = bool(args.dry_run)

    if args.dry_run:
        plan = [f"mode={args.mode}"]
        if args.mode == "compact-copy":
            plan.append(f"would write a compacted snapshot under {args.out_dir} "
                        f"(retain newest {args.keep})")
        elif args.mode == "vacuum":
            plan.append("would run an in-place VACUUM (exclusive lock on the live DB)")
        if not args.no_analyze and args.mode != "report":
            plan.append("would run ANALYZE + PRAGMA optimize on the live DB")
        _log("DRY RUN - no writes performed. Plan: " + "; ".join(plan))
        report["result"] = "dry_run"
        if args.json:
            print(json.dumps(report, indent=2), flush=True)
        return EXIT_OK

    try:
        if effective_mode == "compact-copy":
            report["compact_copy"] = perform_compact_copy(
                db_path, args.out_dir.resolve(), args.busy_timeout_ms, args.keep
            )
            cc = report["compact_copy"]
            _log(f"snapshot        : {cc['snapshot_path']}")
            _log(f"snapshot size   : {_human_bytes(cc['snapshot_bytes'])} "
                 f"(VACUUM would reclaim {_human_bytes(cc['reclaimable_bytes'])} "
                 f"from the live file)")
        elif effective_mode == "vacuum":
            if not vacuum_into_supported():
                _warn("this SQLite build has no VACUUM INTO; the in-place VACUUM below "
                      "takes an EXCLUSIVE lock and will block the perpetual engine.")
            report["vacuum"] = perform_inplace_vacuum(db_path, args.busy_timeout_ms)
            vac = report["vacuum"]
            _log(f"vacuum          : {_human_bytes(vac['bytes_before'])} -> "
                 f"{_human_bytes(vac['bytes_after'])} "
                 f"(reclaimed {_human_bytes(vac['bytes_reclaimed'])}) "
                 f"in {vac['elapsed_sec']}s")

        if not args.no_analyze and effective_mode != "report":
            report["analyze"] = perform_analyze(db_path, args.busy_timeout_ms)
            _log(f"analyze         : ANALYZE + PRAGMA optimize in "
                 f"{report['analyze']['elapsed_sec']}s")
    except sqlite3.OperationalError as exc:
        if _is_lock_error(exc):
            _log(f"ERROR: database is locked ({exc}). The perpetual engine holds it; "
                 f"maintenance aborted rather than blocking the live write cycle. "
                 f"Re-run outside the 30-minute cycle, or use --mode compact-copy.",
                 stream=sys.stderr)
            return EXIT_LOCKED
        _log(f"ERROR: {exc}", stream=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        _log(f"ERROR: filesystem failure during maintenance: {exc}", stream=sys.stderr)
        return EXIT_ERROR

    report["footprint_after"] = _db_footprint(db_path)
    report["result"] = "ok"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if args.json:
        print(json.dumps(report, indent=2), flush=True)
    else:
        _log("maintenance complete")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
