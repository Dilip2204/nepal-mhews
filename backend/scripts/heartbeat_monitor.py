#!/usr/bin/env python3
"""Liveness watchdog for the Himalayan-MHEWS perpetual orchestrator.

Why this exists
---------------
The orchestrator rewrites ``backend/data/perpetual_status.json`` at the end of every
cycle (normally every 30 minutes, 15 under the monsoon throttle). During a 6-month
unattended run that file is the only cheap, external proof that the engine is still
alive: the tmux session can die, the loop can wedge on a network call, or a restart
can crash on import and be swallowed by the orchestrator's broad ``except``.

A silently dead early-warning system is worse than no system, because the empty
dashboard still looks calm. This script turns that silence into a loud, non-zero
exit so cron mails the operator.

Checks performed
----------------
1. The status file exists and parses as JSON.         -> exit 2 on failure
2. Its mtime is newer than ``--max-age-minutes``.     -> exit 1 on failure
3. ``cycle_count`` has advanced since the previous    -> exit 1 on failure
   run (optional, needs the small state file).
4. The embedded ``last_cycle_timestamp_utc`` roughly
   agrees with the file mtime.                        -> warning only

Notification
------------
When a check fails the script tries ``app.alerting.notifier`` and falls back to stderr
if it is not importable or exposes no usable entry point. That module is built for
ORANGE/RED *hazard* alerts, so anything sent from here is tagged
``event=SYSTEM_HEARTBEAT_FAILURE`` / ``alert_level=SYSTEM``: an engine outage is an
operations failure, never a hazard level. Only the physics engine sets hazard levels.

Outbound delivery honours ``settings.alerts_enabled`` so this script cannot start
messaging people the operator has deliberately muted; ``--force-notify`` overrides
that for the dead-engine case. The watchdog's non-zero exit (and the resulting cron
mail) is the primary signal and never depends on any of this.

Exit codes
----------
0 healthy, 1 stale / stalled, 2 status file missing or unreadable.

Examples
--------
    # human-readable one-shot check
    PYTHONPATH=backend backend/.venv/bin/python backend/scripts/heartbeat_monitor.py

    # cron form: silent while healthy, mails only on trouble
    ... heartbeat_monitor.py --quiet
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "backend" / "data"
DEFAULT_STATUS_FILE = DATA_DIR / "perpetual_status.json"
DEFAULT_STATE_FILE = DATA_DIR / "heartbeat_state.json"

EXIT_OK = 0
EXIT_STALE = 1
EXIT_MISSING = 2

# Tolerated disagreement between the file mtime and the timestamp written inside it.
_TIMESTAMP_SKEW_WARN_SEC = 900.0

# Entry points we will accept from app.alerting.notifier, most specific first.
_NOTIFIER_FUNCTIONS: Tuple[str, ...] = (
    "send_ops_notification",
    "send_operational_notification",
    "send_system_notification",
    "dispatch_notification",
    "send_notification",
    "dispatch_alert",
    "send_alert",
    "notify",
)
_SUBJECT_ALIASES = ("subject", "title", "summary", "headline", "name")
_BODY_ALIASES = ("message", "body", "text", "detail", "details", "content")
_SEVERITY_ALIASES = ("severity", "level", "alert_level")


def _now() -> float:
    return time.time()


def _log(message: str, stream: Any = sys.stdout) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} [heartbeat] {message}", file=stream, flush=True)


def _fmt_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.1f} h"


def _try_generic_entry_point(notifier: Any, subject: str, body: str, severity: str) -> Optional[str]:
    """Call a text-shaped notification function if the module happens to expose one."""
    for name in _NOTIFIER_FUNCTIONS:
        func = getattr(notifier, name, None)
        if not callable(func):
            continue
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            continue
        kwargs: Dict[str, Any] = {}
        for alias in _SUBJECT_ALIASES:
            if alias in params:
                kwargs[alias] = subject
                break
        body_param = next((alias for alias in _BODY_ALIASES if alias in params), None)
        if body_param is None:
            # Never guess: a lone positional parameter that is not plainly a string
            # could be a sequence of site evaluations, and passing text would corrupt it.
            continue
        kwargs[body_param] = body
        for alias in _SEVERITY_ALIASES:
            if alias in params:
                kwargs[alias] = severity
                break
        try:
            func(**kwargs)
            _log(f"notification dispatched via app.alerting.notifier.{name}()")
            return f"notifier.{name}"
        except Exception as exc:
            _log(f"notifier.{name}() failed: {type(exc).__name__}: {exc}", stream=sys.stderr)
    return None


def _try_configured_channels(notifier: Any, subject: str, body: str) -> Optional[str]:
    """Reuse the alerting module's configured transports for a SYSTEM-level message.

    ``_configured_channels`` is private to that module, so this is strictly
    best-effort: if it is renamed or removed, the getattr below yields None and the
    caller falls back to stderr.
    """
    resolver = getattr(notifier, "_configured_channels", None)
    if not callable(resolver):
        return None
    try:
        channels = resolver()
    except Exception as exc:
        _log(f"could not resolve alert channels: {type(exc).__name__}: {exc}",
             stream=sys.stderr)
        return None
    if not channels:
        return None

    payload = {
        "event": "SYSTEM_HEARTBEAT_FAILURE",
        "alert_level": "SYSTEM",  # not a hazard level; the physics engine owns those
        "site_id": "perpetual_engine",
        "site_name": "MHEWS Perpetual Orchestrator",
        "subject": subject,
        "text": f"{subject}\n\n{body}",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    delivered: List[str] = []
    for channel_name, sender in channels:
        try:
            if sender(payload):
                delivered.append(channel_name)
        except Exception as exc:
            _log(f"{channel_name} delivery failed: {type(exc).__name__}: {exc}",
                 stream=sys.stderr)
    if delivered:
        _log(f"notification delivered via {', '.join(delivered)}")
        return ",".join(delivered)
    return None


def dispatch_notification(
    subject: str,
    body: str,
    severity: str = "CRITICAL",
    force: bool = False,
) -> str:
    """Best-effort hand-off to the alerting module; stderr is the guaranteed floor.

    ``app.alerting.notifier`` is owned elsewhere, so both the import and every entry
    point are probed defensively rather than assumed.
    """
    try:
        from app.alerting import notifier  # type: ignore
    except Exception as exc:  # ImportError, or a config error raised at import time
        _log(f"alerting module unavailable ({type(exc).__name__}: {exc}); "
             f"falling back to stderr", stream=sys.stderr)
        _log(f"undelivered: {subject} (details below)", stream=sys.stderr)
        return "stderr"

    alerts_enabled = True
    try:
        from app.core.config import settings  # type: ignore

        alerts_enabled = bool(getattr(settings, "alerts_enabled", False))
    except Exception:
        alerts_enabled = False

    if not alerts_enabled and not force:
        _log("outbound alerting is disabled (settings.alerts_enabled is false); "
             "reporting to stderr only. Use --force-notify to override.",
             stream=sys.stderr)
        _log(f"undelivered: {subject} (details below)", stream=sys.stderr)
        return "stderr"

    channel = _try_generic_entry_point(notifier, subject, body, severity)
    if channel is None:
        channel = _try_configured_channels(notifier, subject, body)
    if channel is not None:
        return channel

    _log("app.alerting.notifier has no usable entry point or no configured channel; "
         "falling back to stderr", stream=sys.stderr)
    _log(f"undelivered: {subject} (details below)", stream=sys.stderr)
    return "stderr"


def load_state(state_file: Path) -> Dict[str, Any]:
    try:
        with state_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state_file: Path, state: Dict[str, Any]) -> None:
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_suffix(state_file.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        tmp.replace(state_file)
    except OSError as exc:
        _log(f"could not persist heartbeat state to {state_file}: {exc}", stream=sys.stderr)


def evaluate_heartbeat(
    status_file: Path,
    max_age_sec: float,
    state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Inspect the status file and return a verdict dictionary."""
    now = _now()
    result: Dict[str, Any] = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status_file": str(status_file),
        "max_age_sec": max_age_sec,
        "healthy": False,
        "problems": [],
        "warnings": [],
    }

    if not status_file.exists():
        result["problems"].append(f"status file does not exist: {status_file}")
        result["exit_code"] = EXIT_MISSING
        return result

    mtime = status_file.stat().st_mtime
    age_sec = now - mtime
    result["mtime_utc"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    result["age_sec"] = round(age_sec, 1)

    try:
        with status_file.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    except (OSError, ValueError) as exc:
        result["problems"].append(f"status file is unreadable or not valid JSON: {exc}")
        result["exit_code"] = EXIT_MISSING
        return result

    cycle_count = status.get("cycle_count")
    result["cycle_count"] = cycle_count
    result["uptime_seconds"] = status.get("uptime_seconds")
    result["last_cycle_duration_sec"] = status.get("last_cycle_duration_sec")
    result["monitored_sites_total"] = status.get("monitored_sites_total")
    alert_summary = status.get("alert_summary") or {}
    result["alert_summary"] = alert_summary
    result["elevated_sites"] = int(alert_summary.get("ORANGE", 0) or 0) + int(
        alert_summary.get("RED", 0) or 0
    )
    optimizer = status.get("optimizer") or {}
    result["optimizer_generation"] = optimizer.get("current_generation")
    # Written by the adaptive monsoon throttle; absent on older engine builds.
    result["cadence"] = status.get("cadence")

    embedded_ts = status.get("last_cycle_timestamp_utc")
    if isinstance(embedded_ts, (int, float)):
        result["embedded_age_sec"] = round(now - float(embedded_ts), 1)
        if abs(float(embedded_ts) - mtime) > _TIMESTAMP_SKEW_WARN_SEC:
            result["warnings"].append(
                f"last_cycle_timestamp_utc and the file mtime disagree by "
                f"{_fmt_age(abs(float(embedded_ts) - mtime))}; the file may have been "
                f"touched by something other than the engine, or the clock has drifted"
            )

    if age_sec > max_age_sec:
        result["problems"].append(
            f"status file is {_fmt_age(age_sec)} old (threshold {_fmt_age(max_age_sec)}); "
            f"the perpetual orchestrator has not completed a cycle"
        )

    if state is not None and isinstance(cycle_count, int):
        prev_cycle = state.get("cycle_count")
        prev_checked = state.get("checked_at")
        if (
            isinstance(prev_cycle, int)
            and isinstance(prev_checked, (int, float))
            and cycle_count <= prev_cycle
            and (now - float(prev_checked)) > max_age_sec
        ):
            result["problems"].append(
                f"cycle_count has not advanced (still {cycle_count}) in "
                f"{_fmt_age(now - float(prev_checked))}; the engine process may be "
                f"wedged even though the status file looks recent"
            )

    result["healthy"] = not result["problems"]
    result["exit_code"] = EXIT_OK if result["healthy"] else EXIT_STALE
    return result


def format_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    verdict = "HEALTHY" if result["healthy"] else "UNHEALTHY"
    lines.append(f"Himalayan-MHEWS perpetual engine heartbeat: {verdict}")
    lines.append(f"  status file : {result['status_file']}")
    if "age_sec" in result:
        lines.append(
            f"  last update : {_fmt_age(result['age_sec'])} ago "
            f"({result.get('mtime_utc', 'unknown')})"
        )
    if result.get("cycle_count") is not None:
        uptime = result.get("uptime_seconds")
        uptime_text = _fmt_age(float(uptime)) if isinstance(uptime, (int, float)) else "unknown"
        lines.append(
            f"  cycles      : {result['cycle_count']} "
            f"(uptime {uptime_text}, last cycle "
            f"{result.get('last_cycle_duration_sec', '?')}s)"
        )
    if result.get("alert_summary"):
        summary = result["alert_summary"]
        parts = " | ".join(
            f"{level}={summary.get(level, 0)}" for level in ("GREEN", "YELLOW", "ORANGE", "RED")
        )
        lines.append(f"  sites       : {result.get('monitored_sites_total', '?')} monitored")
        lines.append(f"  alerts      : {parts}")
    if result.get("optimizer_generation") is not None:
        lines.append(f"  optimizer   : generation {result['optimizer_generation']}")
    if result.get("cadence"):
        lines.append(f"  cadence     : {result['cadence']}")
    for warning in result.get("warnings", []):
        lines.append(f"  WARNING     : {warning}")
    for problem in result.get("problems", []):
        lines.append(f"  PROBLEM     : {problem}")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alert when the MHEWS perpetual orchestrator stops producing cycles.",
    )
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE,
                        help="path to perpetual_status.json")
    parser.add_argument("--max-age-minutes", type=float, default=45.0,
                        help="staleness threshold in minutes (default: 45)")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE,
                        help="where to remember the last seen cycle_count")
    parser.add_argument("--no-state", action="store_true",
                        help="disable the state file and the cycle-stall check")
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing while healthy (cron-friendly)")
    parser.add_argument("--json", action="store_true",
                        help="emit the machine-readable verdict on stdout")
    parser.add_argument("--no-notify", action="store_true",
                        help="do not dispatch a notification on failure")
    parser.add_argument("--force-notify", action="store_true",
                        help="notify even when settings.alerts_enabled is false")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    status_file: Path = args.status_file.resolve()
    max_age_sec = max(1.0, args.max_age_minutes * 60.0)

    state: Optional[Dict[str, Any]] = None
    if not args.no_state:
        state = load_state(args.state_file)

    result = evaluate_heartbeat(status_file, max_age_sec, state)

    if not args.no_state and isinstance(result.get("cycle_count"), int):
        save_state(
            args.state_file,
            {
                "cycle_count": result["cycle_count"],
                "checked_at": _now(),
                "checked_utc": result["checked_utc"],
                "healthy": result["healthy"],
            },
        )

    report = format_report(result)
    exit_code = int(result["exit_code"])

    if not result["healthy"] and not args.no_notify:
        result["notification_channel"] = dispatch_notification(
            subject="MHEWS perpetual engine heartbeat FAILED",
            body=report,
            severity="CRITICAL",
            force=bool(args.force_notify),
        )

    if args.json:
        print(json.dumps(result, indent=2), flush=True)
    elif result["healthy"]:
        if not args.quiet:
            print(report, flush=True)
    else:
        print(report, file=sys.stderr, flush=True)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
