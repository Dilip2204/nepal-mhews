#!/usr/bin/env python3
"""Decontaminate the live ground-truth training catalog before an unattended learning run.

`backend/data/processed/live_ground_truth_feedback.csv` is the only *observed* evidence the
ContinuousEquationOptimizer and the DifferentialPhysicsCorrector learn from, and the headline
"Live-GT accuracy" metric is computed over it. It currently holds rows that were never observed
anywhere: rows derived from the hardcoded `SEED_DISASTER_BULLETINS` demo fixtures by a live
(non-dry-run) weekly audit, and rows created by pressing the dashboard's ground-truth submit
button against a monitored site. Because the catalog is tiny, those rows dominate the metric.

This tool finds them, explains *why* each row is suspect, and (only with an explicit `--apply`)
removes the unambiguous ones after taking a verified timestamped backup. It never removes a row
it is merely unsure about: deleting real field evidence is worse than keeping a questionable row,
so ambiguous rows are reported as SUSPECT for a human to decide.

Usage (dry run is the default and never touches the catalog):

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
      PYTHONPATH=backend backend/.venv/bin/python backend/scripts/clean_training_catalog.py

    ... clean_training_catalog.py --apply      # mutates, after backup + verification
"""
import os

# Enforce thread caps on shared 256-core server before importing pandas/numpy
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
import argparse
import csv
import hashlib
import inspect
import io
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ingestion.ground_truth_feedback import (  # noqa: E402
    CSV_FIELDNAMES,
    FEEDBACK_CSV_PATH,
    INITIAL_VERIFIED_FEEDBACK_EVENTS,
)

DATA_DIR = _BACKEND_DIR / "data"
REPORT_DIR = DATA_DIR / "audit_reports"
BACKUP_DIR = DATA_DIR / "backups" / "catalog_cleaning"

# Nepal bounding box used for plausibility screening (generous; the real country box is
# ~80.06-88.20E / 26.35-30.45N).
NEPAL_LON_MIN, NEPAL_LON_MAX = 80.0, 89.0
NEPAL_LAT_MIN, NEPAL_LAT_MAX = 26.0, 31.0
# Beyond this margin a coordinate cannot plausibly be a mis-keyed Nepal location.
BBOX_GROSS_MARGIN_DEG = 2.0

# Two rows are treated as the same observation at this separation (matches the dedup
# tolerance already used by append_ground_truth_event()).
DUP_COORD_TOL_DEG = 0.002
# Tolerance when matching a catalog row against coordinates quoted inside a seed bulletin.
SEED_COORD_TOL_DEG = 0.01

STATUS_CLEAN = "CLEAN"
STATUS_SUSPECT = "SUSPECT"
STATUS_CONTAMINATED = "CONTAMINATED"
_STATUS_RANK = {STATUS_CLEAN: 0, STATUS_SUSPECT: 1, STATUS_CONTAMINATED: 2}

# Machine-generated note text that carries no field evidence whatsoever.
_BOILERPLATE_NOTE_RE = re.compile(
    r"^ingested via [a-z0-9_]+ live ground-truth feedback pipeline\.?$", re.IGNORECASE
)
# Dashboard test-submission naming pattern: dashboard.html builds
#   event_name = `${site.name} (${source.split("_")[0]})`
_SOURCE_TAG_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<tag>[A-Z][A-Z0-9_]{1,30})\)\s*$")
# Catalog site identifiers such as "Nepal Landslide Site #427 (Kali_Gandaki Corridor)".
_SITE_NUMBER_RE = re.compile(r"\bsite\s*#\s*\d+", re.IGNORECASE)
_PLACEHOLDER_TOKEN_RE = re.compile(
    r"\b(test|tests|testing|demo|dummy|sample|placeholder|example|lorem|ipsum|foo|bar|baz|asdf|xxx|tbd|todo)\b",
    re.IGNORECASE,
)
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DECIMAL_RE = re.compile(r"\d{2}\.\d{2,6}")

_SECONDS_PER_DAY = 86400.0


@dataclass
class Signal:
    """One piece of evidence about a row."""

    code: str
    level: str  # STATUS_SUSPECT or STATUS_CONTAMINATED
    detail: str


@dataclass
class RowFinding:
    line_no: int  # 1-based data row number (excludes the header)
    raw: Dict[str, str]
    event_id: str
    event_name: str
    source: str
    latitude: Optional[float]
    longitude: Optional[float]
    label: Optional[int]
    verified_utc: Optional[float]
    notes: str
    protected: bool = False
    signals: List[Signal] = field(default_factory=list)

    def add(self, code: str, level: str, detail: str) -> None:
        self.signals.append(Signal(code, level, detail))

    @property
    def status(self) -> str:
        if self.protected:
            return STATUS_CLEAN
        worst = STATUS_CLEAN
        for s in self.signals:
            if _STATUS_RANK[s.level] > _STATUS_RANK[worst]:
                worst = s.level
        return worst

    @property
    def reason_text(self) -> str:
        if self.protected:
            return "baseline verified record (ground_truth_feedback.INITIAL_VERIFIED_FEEDBACK_EVENTS)"
        if not self.signals:
            return "no contamination signals"
        return "; ".join(f"{s.code}: {s.detail}" for s in self.signals)


# --------------------------------------------------------------------------------------
# Reference material (read at runtime -- nothing about the contamination is hardcoded here)
# --------------------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> Set[str]:
    return {t for t in _normalize(text).split() if len(t) > 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _extract_coords(text: str) -> List[Tuple[float, float]]:
    """Pull (lat, lon) pairs out of free-text bulletins, including Devanagari numerals."""
    ascii_text = (text or "").translate(_DEVANAGARI_DIGITS)
    lats: List[float] = []
    lons: List[float] = []
    for match in _DECIMAL_RE.finditer(ascii_text):
        value = float(match.group(0))
        if NEPAL_LAT_MIN <= value <= NEPAL_LAT_MAX:
            lats.append(value)
        elif NEPAL_LON_MIN <= value <= NEPAL_LON_MAX:
            lons.append(value)
    return list(zip(lats, lons))


def load_seed_signatures() -> Dict[str, Any]:
    """Read the actual demo fixtures from the collector module (never a guessed list)."""
    info: Dict[str, Any] = {
        "available": False,
        "error": "",
        "count": 0,
        "titles": [],
        "norm_titles": [],
        "title_tokens": [],
        "coords": [],
    }
    try:
        from app.ingestion.unstructured_collector import SEED_DISASTER_BULLETINS
    except Exception as exc:  # pragma: no cover - import guard
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    for bulletin in SEED_DISASTER_BULLETINS:
        title = str(bulletin.get("title", ""))
        raw_text = str(bulletin.get("raw_text", ""))
        info["titles"].append(title)
        info["norm_titles"].append(_normalize(title))
        info["title_tokens"].append(_tokens(title))
        info["coords"].extend(_extract_coords(f"{title} {raw_text}"))
    info["available"] = True
    info["count"] = len(SEED_DISASTER_BULLETINS)
    return info


def load_site_names() -> Dict[str, Any]:
    """Names of the 491 monitored sites. A ground-truth row named after a monitoring station
    (rather than a reported incident) was almost certainly created from the dashboard."""
    info: Dict[str, Any] = {"available": False, "error": "", "names": set(), "count": 0}
    try:
        from app.engine.sites_catalog import load_sites_of_interest

        names = {_normalize(s.name) for s in load_sites_of_interest()}
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["available"] = True
    info["names"] = names
    info["count"] = len(names)
    return info


def inspect_collector_seed_gate() -> Dict[str, Any]:
    """Report (do not fix) whether re-running the collector re-stages the demo bulletins."""
    info: Dict[str, Any] = {"available": False, "gated": False, "detail": ""}
    try:
        from app.ingestion import unstructured_collector as collector

        source = inspect.getsource(collector.stage_unprocessed_bulletins)
    except Exception as exc:
        info["detail"] = f"could not inspect collector: {type(exc).__name__}: {exc}"
        return info

    info["available"] = True
    gated = ("seed_demo_bulletins_enabled" in source) or ("MHEWS_SEED_DEMO" in source)
    info["gated"] = gated
    if gated:
        info["detail"] = (
            "stage_unprocessed_bulletins() references the seed-demo gate; confirm "
            "MHEWS_SEED_DEMO=0 in the production environment."
        )
    else:
        info["detail"] = (
            "stage_unprocessed_bulletins() stages SEED_DISASTER_BULLETINS unconditionally "
            "(no reference to settings.seed_demo_bulletins_enabled / MHEWS_SEED_DEMO). Every "
            "collector run re-stages the same demo bulletins, so cleaning the catalog is not "
            "durable until that gate is added by the owner of unstructured_collector.py."
        )
    return info


# --------------------------------------------------------------------------------------
# CSV reading
# --------------------------------------------------------------------------------------


def _to_float(value: Optional[str]) -> Optional[float]:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _to_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def read_catalog(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def classify_rows(
    rows: Sequence[Dict[str, str]],
    seeds: Dict[str, Any],
    sites: Dict[str, Any],
    now_utc: float,
) -> List[RowFinding]:
    baseline_ids = {str(e["event_id"]) for e in INITIAL_VERIFIED_FEEDBACK_EVENTS}
    findings: List[RowFinding] = []

    for idx, raw in enumerate(rows, start=1):
        event_id = (raw.get("event_id") or "").strip()
        event_name = (raw.get("event_name") or "").strip()
        source = (raw.get("source") or "").strip()
        notes = (raw.get("notes") or "").strip()
        f = RowFinding(
            line_no=idx,
            raw=raw,
            event_id=event_id,
            event_name=event_name,
            source=source,
            latitude=_to_float(raw.get("latitude")),
            longitude=_to_float(raw.get("longitude")),
            label=_to_int(raw.get("label")),
            verified_utc=_to_float(raw.get("verified_timestamp_utc")),
            notes=notes,
            protected=event_id in baseline_ids,
        )
        _screen_seed_derivation(f, seeds)
        _screen_evidence_free_submission(f, sites)
        _screen_coordinates(f)
        _screen_timestamp(f, now_utc)
        _screen_schema(f)
        findings.append(f)

    _screen_duplicates(findings)
    return findings


def _screen_seed_derivation(f: RowFinding, seeds: Dict[str, Any]) -> None:
    if not seeds.get("available"):
        return

    machine_extracted = f.source.upper().startswith(("AI_AUDIT_", "MULTIMODAL_VISION_"))
    name_norm = _normalize(f.event_name)
    notes_norm = _normalize(f.notes)
    name_tokens = _tokens(f.event_name)

    for title, title_norm, title_tokens in zip(
        seeds["titles"], seeds["norm_titles"], seeds["title_tokens"]
    ):
        if not title_norm:
            continue
        exact = name_norm == title_norm or title_norm in notes_norm
        if exact:
            f.add(
                "SEED_BULLETIN_DERIVED",
                STATUS_CONTAMINATED,
                f"content matches hardcoded SEED_DISASTER_BULLETINS fixture '{title}'",
            )
            return
        if _jaccard(name_tokens, title_tokens) >= 0.6:
            f.add(
                "SEED_TITLE_SIMILAR",
                STATUS_SUSPECT,
                f"event name closely resembles seed fixture '{title}' (token overlap >= 0.6)",
            )
            break

    if f.latitude is not None and f.longitude is not None:
        for lat, lon in seeds["coords"]:
            if (
                abs(lat - f.latitude) <= SEED_COORD_TOL_DEG
                and abs(lon - f.longitude) <= SEED_COORD_TOL_DEG
            ):
                f.add(
                    "SEED_COORD_MATCH",
                    STATUS_SUSPECT,
                    f"coordinates match a location quoted inside a seed bulletin "
                    f"({lat:.3f}N, {lon:.3f}E); may still be a genuine event at the same place",
                )
                break

    if machine_extracted and f.status != STATUS_CONTAMINATED:
        f.add(
            "UNREVIEWED_MACHINE_LABEL",
            STATUS_SUSPECT,
            f"source='{f.source}' -- label was extracted by an audit engine and never "
            f"human-reviewed",
        )


def _screen_evidence_free_submission(f: RowFinding, sites: Dict[str, Any]) -> None:
    """A row whose notes are auto-generated boilerplate carries no observational evidence.
    On its own that is only suspicious; combined with a name that identifies a *monitoring
    site* or a UI source tag rather than a reported incident it is a dashboard test row."""
    explicit_placeholder = _PLACEHOLDER_TOKEN_RE.search(
        f"{f.event_id} {f.event_name} {f.notes}"
    )
    if explicit_placeholder:
        f.add(
            "PLACEHOLDER_TOKEN",
            STATUS_CONTAMINATED,
            f"test/demo placeholder word '{explicit_placeholder.group(0)}' in the record",
        )

    no_evidence = (not f.notes) or bool(_BOILERPLATE_NOTE_RE.match(f.notes))

    name_signals: List[str] = []
    suffix = _SOURCE_TAG_SUFFIX_RE.match(f.event_name)
    base_name = suffix.group("base").strip() if suffix else f.event_name
    if suffix and f.source.upper().startswith(suffix.group("tag").upper()):
        name_signals.append(
            f"name ends with '({suffix.group('tag')})', the dashboard submit-button pattern "
            f"`${{site.name}} (${{source.split('_')[0]}})`"
        )
    if sites.get("available") and _normalize(base_name) in sites["names"]:
        name_signals.append(
            "event name is a monitored site in sites_catalog, not a reported incident"
        )
    if _SITE_NUMBER_RE.search(f.event_name):
        name_signals.append("name is a catalog site identifier of the form 'Site #NNN'")

    if name_signals and no_evidence:
        f.add(
            "DASHBOARD_TEST_SUBMISSION",
            STATUS_CONTAMINATED,
            "auto-generated boilerplate notes (no field evidence) + "
            + "; ".join(name_signals),
        )
    elif name_signals:
        f.add(
            "SITE_NAMED_ROW",
            STATUS_SUSPECT,
            "; ".join(name_signals) + " -- but the notes carry specific evidence, so kept",
        )
    elif no_evidence:
        f.add(
            "NO_EVIDENCE_NOTES",
            STATUS_SUSPECT,
            "notes are empty or auto-generated boilerplate; nothing records how this event "
            "was verified",
        )


def _screen_coordinates(f: RowFinding) -> None:
    if f.latitude is None or f.longitude is None:
        f.add(
            "COORD_UNPARSEABLE",
            STATUS_CONTAMINATED,
            f"latitude/longitude missing or non-numeric "
            f"(lat={f.raw.get('latitude')!r}, lon={f.raw.get('longitude')!r})",
        )
        return

    if abs(f.latitude) < 1e-3 and abs(f.longitude) < 1e-3:
        f.add("COORD_NULL_ISLAND", STATUS_CONTAMINATED, "coordinates are (0, 0)")
        return

    lat_over = max(NEPAL_LAT_MIN - f.latitude, f.latitude - NEPAL_LAT_MAX, 0.0)
    lon_over = max(NEPAL_LON_MIN - f.longitude, f.longitude - NEPAL_LON_MAX, 0.0)
    overshoot = max(lat_over, lon_over)
    if overshoot <= 0.0:
        return
    where = f"({f.latitude:.4f}N, {f.longitude:.4f}E)"
    if overshoot > BBOX_GROSS_MARGIN_DEG:
        f.add(
            "COORD_FAR_OUTSIDE_NEPAL",
            STATUS_CONTAMINATED,
            f"{where} is {overshoot:.2f} deg outside the Nepal bbox "
            f"({NEPAL_LON_MIN}-{NEPAL_LON_MAX}E, {NEPAL_LAT_MIN}-{NEPAL_LAT_MAX}N)",
        )
    else:
        f.add(
            "COORD_OUTSIDE_NEPAL",
            STATUS_SUSPECT,
            f"{where} is {overshoot:.2f} deg outside the Nepal bbox; could be a border site "
            "or a digit error -- human decision",
        )


def _screen_timestamp(f: RowFinding, now_utc: float) -> None:
    ts = f.verified_utc
    if ts is None or ts <= 0:
        f.add(
            "TIMESTAMP_INVALID",
            STATUS_SUSPECT,
            f"verified_timestamp_utc={f.raw.get('verified_timestamp_utc')!r} is missing or <= 0",
        )
        return
    ahead_days = (ts - now_utc) / _SECONDS_PER_DAY
    if ahead_days > 365.0:
        f.add(
            "TIMESTAMP_IMPOSSIBLE_FUTURE",
            STATUS_CONTAMINATED,
            f"verified {ahead_days:.0f} days in the future ({_fmt_ts(ts)})",
        )
    elif ahead_days > 1.0:
        f.add(
            "TIMESTAMP_FUTURE",
            STATUS_SUSPECT,
            f"verified {ahead_days:.1f} days in the future ({_fmt_ts(ts)}); could be clock skew",
        )


def _screen_schema(f: RowFinding) -> None:
    if f.label not in (0, 1):
        f.add(
            "LABEL_INVALID",
            STATUS_SUSPECT,
            f"label={f.raw.get('label')!r} is not 0 or 1",
        )
    unparseable = [
        name
        for name in (
            "elevation_m",
            "slope_deg",
            "profile_curvature",
            "local_relief_m",
            "fault_distance_km",
            "ndvi_Drop",
            "rainfall_72h_mm",
        )
        if _to_float(f.raw.get(name)) is None
    ]
    if unparseable:
        f.add(
            "LOADER_WOULD_FAIL",
            STATUS_SUSPECT,
            "non-numeric field(s) "
            + ", ".join(unparseable)
            + " -- load_live_ground_truth_events() raises ValueError on this row",
        )


def _screen_duplicates(findings: List[RowFinding]) -> None:
    content_fields = [c for c in CSV_FIELDNAMES if c not in ("event_id", "verified_timestamp_utc")]

    seen_content: Dict[Tuple[str, ...], RowFinding] = {}
    for f in findings:
        key = tuple((f.raw.get(c) or "").strip() for c in content_fields)
        first = seen_content.get(key)
        if first is None:
            seen_content[key] = f
            continue
        f.add(
            "EXACT_DUPLICATE",
            STATUS_CONTAMINATED,
            f"every field except event_id/timestamp is identical to row {first.line_no} "
            f"({first.event_id})",
        )

    seen_ids: Dict[str, RowFinding] = {}
    for f in findings:
        if not f.event_id:
            continue
        first = seen_ids.get(f.event_id)
        if first is None:
            seen_ids[f.event_id] = f
        else:
            f.add(
                "DUPLICATE_EVENT_ID",
                STATUS_SUSPECT,
                f"event_id collides with row {first.line_no}",
            )

    # Near-duplicates: same place, same day, same label.
    groups: Dict[Tuple[int, int, str, Optional[int]], List[RowFinding]] = {}
    for f in findings:
        if f.latitude is None or f.longitude is None or f.verified_utc is None:
            continue
        key = (
            int(round(f.latitude / DUP_COORD_TOL_DEG)),
            int(round(f.longitude / DUP_COORD_TOL_DEG)),
            _fmt_ts(f.verified_utc)[:10],
            f.label,
        )
        groups.setdefault(key, []).append(f)
    for key, members in groups.items():
        if len(members) < 2:
            continue
        others = ", ".join(str(m.line_no) for m in members)
        for f in members:
            f.add(
                "NEAR_DUPLICATE",
                STATUS_SUSPECT,
                f"same location/date/label as rows {others} "
                f"({len(members)} rows at {key[2]}) -- one real event may be counted "
                f"{len(members)} times",
            )


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "n/a"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return f"invalid({ts})"


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _truncate(text: str, width: int) -> str:
    text = text or ""
    return text if len(text) <= width else text[: width - 1] + "…"


def render_table(findings: Sequence[RowFinding]) -> str:
    out = io.StringIO()
    header = f"{'#':>3}  {'STATUS':<13} {'EVENT NAME':<52} {'SOURCE':<34} {'LABEL':>5}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    for f in findings:
        out.write(
            f"{f.line_no:>3}  {f.status:<13} {_truncate(f.event_name, 52):<52} "
            f"{_truncate(f.source, 34):<34} {str(f.label):>5}\n"
        )
        for s in f.signals:
            out.write(f"       - [{s.level[:4]}] {s.code}: {s.detail}\n")
        if f.protected and f.signals:
            out.write(
                "       - [KEEP] PROTECTED_BASELINE: listed in "
                "ground_truth_feedback.INITIAL_VERIFIED_FEEDBACK_EVENTS; never auto-removed\n"
            )
    return out.getvalue()


def _label_counts(findings: Sequence[RowFinding]) -> Tuple[int, int, int]:
    pos = sum(1 for f in findings if f.label == 1)
    neg = sum(1 for f in findings if f.label == 0)
    other = len(findings) - pos - neg
    return pos, neg, other


def build_report(
    findings: Sequence[RowFinding],
    csv_path: Path,
    md5_before: str,
    md5_after: Optional[str],
    applied: bool,
    backup_path: Optional[Path],
    seeds: Dict[str, Any],
    sites: Dict[str, Any],
    gate: Dict[str, Any],
    started_utc: float,
) -> str:
    contaminated = [f for f in findings if f.status == STATUS_CONTAMINATED]
    suspect = [f for f in findings if f.status == STATUS_SUSPECT]
    clean = [f for f in findings if f.status == STATUS_CLEAN]
    kept = [f for f in findings if f.status != STATUS_CONTAMINATED]
    pos_b, neg_b, other_b = _label_counts(findings)
    pos_a, neg_a, other_a = _label_counts(kept)

    o = io.StringIO()
    o.write("# Training Catalog Decontamination Report\n\n")
    o.write(f"- **Generated (UTC):** {_fmt_ts(started_utc)}\n")
    o.write(f"- **Mode:** {'APPLY (file mutated)' if applied else 'DRY RUN (no file written)'}\n")
    o.write(f"- **Catalog:** `{csv_path}`\n")
    o.write(f"- **MD5 before:** `{md5_before}`\n")
    o.write(f"- **MD5 after:** `{md5_after or md5_before + '  (unchanged -- dry run)'}`\n")
    o.write(f"- **Backup:** {f'`{backup_path}`' if backup_path else 'not taken (dry run)'}\n\n")

    o.write("## Counts\n\n")
    o.write("| | rows | label=1 | label=0 | other |\n|---|---:|---:|---:|---:|\n")
    o.write(f"| before | {len(findings)} | {pos_b} | {neg_b} | {other_b} |\n")
    o.write(
        f"| after {'(applied)' if applied else '(would be)'} | {len(kept)} | {pos_a} "
        f"| {neg_a} | {other_a} |\n\n"
    )
    o.write(
        f"- CLEAN: **{len(clean)}** · SUSPECT (kept, human decision): **{len(suspect)}** · "
        f"CONTAMINATED {'(removed)' if applied else '(would be removed)'}: "
        f"**{len(contaminated)}**\n\n"
    )
    if len(kept) < 30:
        o.write(
            f"> **Honesty note:** the cleaned catalog holds only {len(kept)} rows "
            f"({pos_a} positive / {neg_a} negative). Any \"Live-GT accuracy\" figure computed "
            f"over it -- including the reported 100% -- is a statement about "
            f"{len(kept)} observations and must not be presented as a validated skill score.\n\n"
        )

    seed_desc = (
        f"read {seeds.get('count', 0)} bulletins from "
        "unstructured_collector.SEED_DISASTER_BULLETINS"
        if seeds.get("available")
        else f"UNAVAILABLE: {seeds.get('error', '')}"
    )
    site_desc = (
        f"{sites.get('count', 0)} names from engine.sites_catalog"
        if sites.get("available")
        else f"UNAVAILABLE: {sites.get('error', '')}"
    )
    o.write("## Reference material used\n\n")
    o.write(f"- Seed fixtures: {seed_desc}\n")
    o.write(f"- Monitored site names: {site_desc}\n")
    o.write(
        f"- Baseline protected rows: {len(INITIAL_VERIFIED_FEEDBACK_EVENTS)} event_ids from "
        "ground_truth_feedback.INITIAL_VERIFIED_FEEDBACK_EVENTS\n\n"
    )

    o.write("## Rows removed" if applied else "## Rows that WOULD be removed (--apply)")
    o.write("\n\n")
    if not contaminated:
        o.write("_None._\n\n")
    for f in contaminated:
        o.write(f"### Row {f.line_no} — `{f.event_id}`\n\n")
        o.write(f"- **event_name:** {f.event_name}\n")
        o.write(f"- **source:** {f.source}\n")
        o.write(
            f"- **coords/label/verified:** {f.latitude}, {f.longitude} · label={f.label} · "
            f"{_fmt_ts(f.verified_utc)}\n"
        )
        o.write(f"- **notes:** {f.notes or '(empty)'}\n")
        o.write("- **Reasons:**\n")
        for s in f.signals:
            o.write(f"  - `{s.code}` ({s.level}): {s.detail}\n")
        o.write(f"- **Verbatim row:** `{_verbatim(f)}`\n\n")

    o.write("## SUSPECT rows kept for human decision\n\n")
    if not suspect:
        o.write("_None._\n\n")
    else:
        o.write(
            "These were NOT removed. Deleting real field evidence is worse than keeping a "
            "questionable row, so each needs a human call.\n\n"
        )
        for f in suspect:
            o.write(f"- **Row {f.line_no}** `{f.event_id}` — {f.event_name}\n")
            for s in f.signals:
                o.write(f"  - `{s.code}`: {s.detail}\n")
        o.write("\n")

    o.write("## Rows retained as CLEAN\n\n")
    for f in clean:
        tag = " _(protected baseline)_" if f.protected else ""
        o.write(f"- Row {f.line_no} `{f.event_id}` — {f.event_name}{tag}\n")
    o.write("\n")

    o.write("## Will the contamination come back?\n\n")
    o.write(f"- {gate.get('detail', 'unknown')}\n")
    o.write(
        "- The gate that must stay off in production is `settings.seed_demo_bulletins_enabled` "
        "(env `MHEWS_SEED_DEMO`, default `0`). Cleaning the catalog is pointless if a live "
        "(non-`--dry-run`) weekly audit can re-extract the same demo bulletins from the "
        "staging queue.\n"
    )
    o.write(
        "- Rows already staged in the `unprocessed_bulletins` queue from earlier runs are "
        "**not** touched by this tool; they live in the telemetry DB and are owned elsewhere.\n"
    )
    o.write(
        "- `append_ground_truth_event()` dedups only on (lat, lon, label, exact event_name), "
        "so the same event under a different name still appends. That is why the "
        "NEAR_DUPLICATE groups above exist.\n"
    )
    return o.getvalue()


def _verbatim(f: RowFinding) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(f.raw.keys()), lineterminator="")
    writer.writerow(f.raw)
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# Mutation (only under --apply)
# --------------------------------------------------------------------------------------


def take_backup(csv_path: Path, backup_dir: Path, stamp: str) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{csv_path.stem}.{stamp}.pre-clean.csv"
    shutil.copy2(csv_path, target)
    if file_md5(target) != file_md5(csv_path):
        raise RuntimeError(f"backup verification failed: {target} differs from {csv_path}")
    return target


def write_cleaned(
    csv_path: Path, fieldnames: Sequence[str], kept_rows: Sequence[Dict[str, str]]
) -> None:
    """Rewrite the catalog atomically, byte-compatible with csv.DictWriter defaults."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in kept_rows:
        writer.writerow(row)
    content = buf.getvalue()

    # Verify the serialized text round-trips to exactly the rows we meant to keep.
    reparsed = list(csv.DictReader(io.StringIO(content, newline="")))
    if len(reparsed) != len(kept_rows):
        raise RuntimeError("refusing to write: round-trip row count mismatch")
    for original, parsed in zip(kept_rows, reparsed):
        for key in fieldnames:
            if (original.get(key) or "") != (parsed.get(key) or ""):
                raise RuntimeError(
                    f"refusing to write: round-trip mismatch on field '{key}' "
                    f"of event_id={original.get('event_id')!r}"
                )

    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, csv_path)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and (optionally) clean the live ground-truth training catalog. "
            "Dry run by default; --apply is required to modify the file."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually remove CONTAMINATED rows (takes a verified backup first). "
        "Without this flag nothing is written to the catalog.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="explicit no-op default; kept so cron entries can state intent",
    )
    parser.add_argument("--csv", default=str(FEEDBACK_CSV_PATH), help="catalog path")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--backup-dir", default=str(BACKUP_DIR))
    parser.add_argument("--no-report", action="store_true", help="skip the markdown report file")
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive.", flush=True)
        return 1

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"ERROR: catalog not found: {csv_path}", flush=True)
        return 1

    started = time.time()
    stamp = datetime.fromtimestamp(started, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    md5_before = file_md5(csv_path)

    fieldnames, rows = read_catalog(csv_path)
    if fieldnames != list(CSV_FIELDNAMES):
        print(
            "WARNING: catalog header differs from ground_truth_feedback.CSV_FIELDNAMES.\n"
            f"  file:   {fieldnames}\n  module: {list(CSV_FIELDNAMES)}",
            flush=True,
        )

    seeds = load_seed_signatures()
    sites = load_site_names()
    gate = inspect_collector_seed_gate()
    findings = classify_rows(rows, seeds, sites, started)

    contaminated = [f for f in findings if f.status == STATUS_CONTAMINATED]
    kept = [f for f in findings if f.status != STATUS_CONTAMINATED]

    print("=" * 110, flush=True)
    print(
        f"TRAINING CATALOG DECONTAMINATION  [{'APPLY' if args.apply else 'DRY RUN'}]  {_fmt_ts(started)}",
        flush=True,
    )
    print(f"catalog : {csv_path}", flush=True)
    print(f"md5     : {md5_before}", flush=True)
    if not seeds.get("available"):
        print(f"WARNING : seed fixtures unavailable ({seeds.get('error')})", flush=True)
    if not sites.get("available"):
        print(f"WARNING : site catalog unavailable ({sites.get('error')})", flush=True)
    print("=" * 110, flush=True)
    print(render_table(findings), end="", flush=True)
    print("-" * 110, flush=True)
    print(
        f"rows={len(findings)}  CLEAN={len(findings) - len(contaminated) - sum(1 for f in findings if f.status == STATUS_SUSPECT)}"
        f"  SUSPECT={sum(1 for f in findings if f.status == STATUS_SUSPECT)}"
        f"  CONTAMINATED={len(contaminated)}  ->  would keep {len(kept)}",
        flush=True,
    )
    print(f"seed gate: {gate.get('detail')}", flush=True)

    backup_path: Optional[Path] = None
    md5_after: Optional[str] = None
    if args.apply:
        if not contaminated:
            print("Nothing classified CONTAMINATED; catalog left untouched.", flush=True)
        else:
            backup_path = take_backup(csv_path, Path(args.backup_dir).resolve(), stamp)
            print(f"backup  : {backup_path} (md5 verified)", flush=True)
            write_cleaned(csv_path, fieldnames, [f.raw for f in kept])
            md5_after = file_md5(csv_path)
            _, rows_after = read_catalog(csv_path)
            if len(rows_after) != len(kept):
                print(
                    f"ERROR: post-write row count {len(rows_after)} != expected {len(kept)}; "
                    f"restore from {backup_path}",
                    flush=True,
                )
                return 1
            print(
                f"applied : removed {len(contaminated)} rows, {len(rows_after)} remain. "
                f"md5 {md5_before} -> {md5_after}",
                flush=True,
            )
    else:
        md5_now = file_md5(csv_path)
        print(
            f"dry run : catalog untouched (md5 still {md5_now}, "
            f"{'unchanged' if md5_now == md5_before else 'CHANGED -- investigate'}). "
            f"Re-run with --apply to remove the {len(contaminated)} CONTAMINATED rows.",
            flush=True,
        )

    if not args.no_report:
        report_dir = Path(args.report_dir).resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        suffix = "applied" if (args.apply and contaminated) else "dryrun"
        report_path = report_dir / f"catalog_cleaning_{stamp}_{suffix}.md"
        report_path.write_text(
            build_report(
                findings,
                csv_path,
                md5_before,
                md5_after,
                bool(args.apply and contaminated),
                backup_path,
                seeds,
                sites,
                gate,
                started,
            ),
            encoding="utf-8",
        )
        print(f"report  : {report_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
