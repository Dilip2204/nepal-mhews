"""NDRRMA BIPAD Portal Harvester — official Nepal government disaster incident records.

BIPAD (https://bipadportal.gov.np) is NDRRMA's national incident portal. Incidents are
filed by Nepal Police and district disaster authorities, which makes it the only
large-volume source of *officially attributed* landslide / flood / avalanche labels for
Nepal. The ML training catalog currently holds ~19 live ground-truth rows, several of
them demo contamination, so without a real labelled stream six months of telemetry
cannot improve model accuracy.

WHAT THIS MODULE DOES NOT DO
    It never writes to the training catalog. Every harvested incident is staged into the
    `unprocessed_bulletins` queue with status PENDING, so it must still pass the weekly
    dual-model (Gemini + Claude) consensus gate and human approval before it can become a
    training label. The physics engine, not this module and not an LLM, sets alert levels.

API SHAPE — VERIFIED, NOT ASSUMED
    `bipadportal.gov.np` resolves from this host (202.45.146.3) but refuses/times out on
    TLS :443, so the live response could not be exercised here. The schema below was
    instead confirmed against 6,735 real response records captured from
    `https://bipadportal.gov.np/api/v1/incident/` on 2026-10-13 and published at
    github.com/chalise-suraj/UrbanPulse-Nepal (data/raw/bipad_incidents/).

    Envelope (Django REST Framework LimitOffsetPagination):
        {"count": int, "next": str|null, "previous": str|null, "results": [ ... ]}
    Record fields actually present in every sampled record:
        id, title, titleNe, wards[], point{type,coordinates:[lon,lat]}, polygon,
        incidentOn, reportedOn, createdOn, modifiedOn, description, detail, cause,
        streetAddress, verified, approved, verificationMessage, needFollowup,
        region, regionId, dataSource, dataSourceId, source, event, hazard, loss,
        createdBy, updatedBy
    `hazard` and `loss` are integer foreign keys, not labels. Pagination observed in the
    wild: `?limit=1000&offset=N`. The portal did not report a usable `count` (it returned
    sys.maxsize), so pagination must follow `next` until results are empty.

    Server-side date filtering and ordering are NOT confirmed. This module sends
    `incident_on__gt` and `ordering=-incidentOn` opportunistically and then *re-filters
    every record client-side*, so an API that silently ignores unknown query parameters
    (DRF's default) still yields a correct result, just over more pages.

KNOWN DATA-QUALITY CAVEAT — READ BEFORE TRUSTING A COORDINATE
    BIPAD `point` is frequently a ward / municipality administrative centroid rather than
    the actual failure location: the 2026-10-13 sample contained only 571 distinct
    coordinates across 2,072 geohazard incidents, one centroid repeating 46 times. A
    centroid is a poor label location for 30 m terrain derivatives. Records whose
    coordinate repeats within a batch are therefore tagged
    `coordinate_precision='ADMIN_CENTROID_SUSPECTED'` and have their confidence reduced.
    Confidence is a provenance heuristic, never a model probability, and never reaches 1.0.

DEFAULT IS OFF
    Ships disabled (`BIPAD_ENABLED=0`). With the portal unreachable from this host that is
    the honest default; enabling it is a deliberate operator decision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.core.config import settings
from app.core.database import BaseTimeSeriesStore, get_time_series_store
from app.ingestion.ground_truth_feedback import load_live_ground_truth_events
from app.ingestion.unstructured_collector import extract_district_hint

logger = logging.getLogger("nepal_mhews.bipad_scraper")

SOURCE_TAG = "BIPAD_NDRRMA"

# Nepal Standard Time is UTC+05:45 year round (no DST), so a fixed offset is exact.
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))

# Same bounding box the weekly audit uses to validate extracted coordinates.
NEPAL_BBOX: Tuple[float, float, float, float] = (80.0, 26.2, 88.3, 30.6)

USER_AGENT = (
    "Himalayan-MHEWS/1.0 (Nepal multi-hazard early warning research system; "
    "weekly NDRRMA BIPAD ground-truth harvest; contact dilip.jena@marqueesemi.com)"
)

# Empirical `hazard` id -> label, derived by cross-tabulating the integer id against the
# incident title prefix over 6,735 real captured records (e.g. every hazard==17 record was
# titled "Landslide at ..."). The portal's own /hazard/ endpoint was unreachable, so this
# table is evidence-based but may be incomplete: unseen ids fall through to text matching
# and, failing that, the record is dropped rather than guessed.
BIPAD_HAZARD_ID_LABELS: Dict[int, str] = {
    2: "Animal Incidents",
    3: "Avalanche",
    4: "Boat Capsize",
    7: "Drowning",
    8: "Earthquake",
    9: "Epidemic",
    10: "Fire",
    11: "Flood",
    12: "Forest Fire",
    14: "Heavy Rainfall",
    16: "High Altitude",
    17: "Landslide",
    20: "Snake Bite",
    22: "Storm",
    23: "Thunderbolt",
    24: "Wind Storm",
    45: "Other",
}

# BIPAD vocabulary -> this project's hazard types. Ids with no slope/flood relevance map to
# None so they are discarded instead of polluting the queue.
BIPAD_HAZARD_ID_TO_TYPE: Dict[int, Optional[str]] = {
    2: None,
    3: "rock_ice_avalanche",
    4: None,
    7: None,
    8: "earthquake",
    9: None,
    10: None,
    11: "flood",
    12: None,
    14: "heavy_rainfall",
    16: None,
    17: "landslide",
    20: None,
    22: None,
    23: None,
    24: None,
    45: None,
}

# Hazard types worth staging as candidate ground truth for a slope-stability model.
GEOHAZARD_TYPES: Tuple[str, ...] = (
    "landslide",
    "debris_flow",
    "rockfall",
    "rock_ice_avalanche",
    "glof",
    "flood",
)

# Text refinements applied on top of the id mapping. A match may only narrow a hazard the
# portal already classified as a mass-movement/flood; it never overrides an id the portal
# explicitly filed as something else.
#
# Keywords must be unambiguous substrings. Devanagari has no word boundaries, so a short
# token matches inside unrelated words: "ढुङ्गा" (stone) was dropped because it matches
# inside the district name "ओखलढुङ्गा" (Okhaldhunga) and promoted Fire records to rockfall.
_TEXT_REFINEMENTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("glof", ("glof", "glacial lake outburst", "moraine", "outburst flood", "हिमताल")),
    ("rock_ice_avalanche", ("avalanche", "serac", "icefall", "rock-ice", "हिमपहिरो")),
    ("debris_flow", ("debris flow", "mudflow", "लेदो")),
    ("rockfall", ("rockfall", "rock fall")),
)

# Hazard classes a text refinement is allowed to narrow.
_REFINABLE_TYPES = frozenset({"landslide", "flood", "heavy_rainfall"})


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def map_bipad_hazard(hazard_id: Any, text: str = "") -> Optional[str]:
    """Map a BIPAD `hazard` id (plus free text) onto this project's hazard vocabulary.

    Returns None when the incident is not a slope/flood hazard or cannot be classified,
    which causes the caller to drop the record.
    """
    base: Optional[str] = None
    known = False
    try:
        key = int(hazard_id)
        known = key in BIPAD_HAZARD_ID_TO_TYPE
        base = BIPAD_HAZARD_ID_TO_TYPE.get(key)
    except (TypeError, ValueError):
        known = False

    # The portal filed this incident under a hazard it tracks and that hazard is not a
    # slope/flood process (Fire, Snake Bite, Thunderbolt...). Free text must not override
    # an official classification, or a fire whose district is spelled "Okhaldhunga"
    # becomes a rockfall.
    if known and base is None:
        return None

    lowered = (text or "").lower()
    for refined, keywords in _TEXT_REFINEMENTS:
        if any(kw in lowered for kw in keywords):
            if base is None or base in _REFINABLE_TYPES:
                return refined

    if base is None and "landslide" in lowered:
        return "landslide"
    return base


def _parse_event_date(raw: Any) -> Optional[str]:
    """Parse a BIPAD ISO-8601 timestamp into a Nepal-local 'YYYY-MM-DD' date string."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.fromisoformat(text[:19])
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NEPAL_TZ)
    return dt.astimezone(NEPAL_TZ).date().isoformat()


def _extract_point(raw: Any) -> Optional[Tuple[float, float]]:
    """Pull (latitude, longitude) out of a GeoJSON Point, validating the Nepal envelope."""
    if not isinstance(raw, dict):
        return None
    coords = raw.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    try:
        lon = float(coords[0])
        lat = float(coords[1])
    except (TypeError, ValueError):
        return None
    if lat != lat or lon != lon:  # NaN
        return None
    min_lon, min_lat, max_lon, max_lat = NEPAL_BBOX
    if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
        return None
    return round(lat, 4), round(lon, 4)


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def normalize_incident(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize one raw BIPAD incident record, or return None if it is unusable.

    Any field that cannot be read from the payload is omitted or set to None. Nothing is
    synthesized: a record without a usable coordinate, date or hazard class is dropped.
    """
    if not isinstance(raw, dict):
        return None

    point = _extract_point(raw.get("point"))
    if point is None:
        return None
    latitude, longitude = point

    event_date = (
        _parse_event_date(raw.get("incidentOn"))
        or _parse_event_date(raw.get("reportedOn"))
        or _parse_event_date(raw.get("createdOn"))
    )
    if event_date is None:
        return None

    title = _clean(raw.get("title"))
    title_ne = _clean(raw.get("titleNe"))
    street = _clean(raw.get("streetAddress"))
    detail = _clean(raw.get("description")) or _clean(raw.get("detail"))
    cause = _clean(raw.get("cause"))

    text_blob = " ".join([title, title_ne, street, detail, cause])
    hazard_type = map_bipad_hazard(raw.get("hazard"), text_blob)
    if hazard_type is None:
        return None

    external_id = raw.get("id")
    if external_id is None:
        return None
    external_id = f"bipad-{external_id}"

    district = extract_district_hint(text_blob) or None

    verified = raw.get("verified") is True
    approved = raw.get("approved") is True
    reporter = _clean(raw.get("source")) or "unknown"

    confidence = 0.40
    if verified:
        confidence += 0.25
    if approved:
        confidence += 0.10
    if reporter == "nepal_police":
        confidence += 0.10

    hazard_label = BIPAD_HAZARD_ID_LABELS.get(
        raw.get("hazard") if isinstance(raw.get("hazard"), int) else -1, "Unclassified"
    )
    description_parts = [p for p in (title, title_ne, street, detail, cause) if p]

    return {
        "latitude": latitude,
        "longitude": longitude,
        "event_date": event_date,
        "hazard_type": hazard_type,
        "district": district,
        "source": SOURCE_TAG,
        "external_id": external_id,
        "description": " | ".join(description_parts) or f"{hazard_label} incident",
        "confidence": round(min(0.90, max(0.10, confidence)), 2),
        # Provenance extras consumed by the staging text and the weekly consensus gate.
        "bipad_hazard_label": hazard_label,
        "reported_by": reporter,
        "verified": verified,
        "approved": approved,
        "coordinate_precision": "REPORTED_POINT",
        "url": f"https://bipadportal.gov.np/incidents/{raw.get('id')}",
    }


def _flag_repeated_coordinates(events: List[Dict[str, Any]]) -> None:
    """Downgrade events whose coordinate repeats in the batch (likely a ward centroid)."""
    counts: Dict[Tuple[float, float], int] = {}
    for ev in events:
        key = (ev["latitude"], ev["longitude"])
        counts[key] = counts.get(key, 0) + 1
    for ev in events:
        if counts[(ev["latitude"], ev["longitude"])] > 1:
            ev["coordinate_precision"] = "ADMIN_CENTROID_SUSPECTED"
            ev["confidence"] = round(max(0.10, ev["confidence"] - 0.15), 2)


# ---------------------------------------------------------------------------
# Polite HTTP
# ---------------------------------------------------------------------------
def _request_json(
    url: str,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> Optional[Any]:
    """GET a JSON document with retry/backoff. Returns None on any failure."""
    attempt = 0
    while attempt < max(1, max_retries):
        attempt += 1
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                payload = response.read()
            return json.loads(payload.decode("utf-8", errors="replace"))
        except Exception as exc:  # URLError, HTTPError, socket.timeout, JSONDecodeError
            logger.warning(
                "[BIPAD] request attempt %d/%d failed for %s: %s",
                attempt,
                max(1, max_retries),
                url.split("?")[0],
                exc,
            )
            if attempt >= max(1, max_retries):
                return None
            if retry_backoff_s > 0:
                time.sleep(retry_backoff_s * (2 ** (attempt - 1)))
    return None


def _same_host(candidate: str, reference: str) -> bool:
    try:
        return (
            urllib.parse.urlparse(candidate).netloc.lower()
            == urllib.parse.urlparse(reference).netloc.lower()
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_recent_incidents(
    since_days: int = 7,
    *,
    api_url: Optional[str] = None,
    limit_per_page: int = 1000,
    max_pages: int = 80,
    timeout_s: float = 25.0,
    page_delay_s: float = 1.0,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
    hazard_types: Optional[Iterable[str]] = None,
    enabled: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Harvest recent official BIPAD incidents, normalized to this project's vocabulary.

    Returns a list of dicts with keys: latitude, longitude, event_date, hazard_type,
    district, source, external_id, description, confidence (plus provenance extras).

    Never raises: a disabled flag, DNS failure, refused connection, HTTP error, malformed
    JSON or unexpected schema all degrade to an empty list plus a warning.
    """
    if enabled is None:
        enabled = bool(getattr(settings, "bipad_enabled", False))
    if not enabled:
        logger.info("[BIPAD] disabled (BIPAD_ENABLED=0); skipping harvest, no request sent.")
        return []

    try:
        return _fetch_recent_incidents_inner(
            since_days=since_days,
            api_url=api_url or getattr(settings, "bipad_api_url", ""),
            limit_per_page=limit_per_page,
            max_pages=max_pages,
            timeout_s=timeout_s,
            page_delay_s=page_delay_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            hazard_types=tuple(hazard_types) if hazard_types else GEOHAZARD_TYPES,
        )
    except Exception as exc:
        logger.warning("[BIPAD] harvest aborted, returning no events: %s", exc)
        return []


def _fetch_recent_incidents_inner(
    *,
    since_days: int,
    api_url: str,
    limit_per_page: int,
    max_pages: int,
    timeout_s: float,
    page_delay_s: float,
    max_retries: int,
    retry_backoff_s: float,
    hazard_types: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    if not api_url:
        logger.warning("[BIPAD] no api url configured; skipping harvest.")
        return []

    today_npt = datetime.now(NEPAL_TZ).date()
    cutoff: date = today_npt - timedelta(days=max(0, int(since_days)))
    # Guard against clock-skewed / mistyped future incident dates in the portal.
    horizon: date = today_npt + timedelta(days=1)

    params = {
        "limit": str(max(1, int(limit_per_page))),
        # Opportunistic server-side narrowing; ignored safely by a DRF view that does not
        # declare these filters, because every record is re-filtered below regardless.
        "incident_on__gt": cutoff.isoformat(),
        "ordering": "-incidentOn",
    }
    next_url: Optional[str] = f"{api_url}?{urllib.parse.urlencode(params)}"

    collected: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()
    seen_external_ids: Set[str] = set()
    pages_read = 0
    records_seen = 0
    server_filtered: Optional[bool] = None
    truncated = False

    while next_url:
        if pages_read >= max(1, int(max_pages)):
            truncated = True
            break
        if next_url in seen_urls:
            logger.warning("[BIPAD] pagination loop detected; stopping.")
            truncated = True
            break
        if pages_read and not _same_host(next_url, api_url):
            logger.warning("[BIPAD] pagination link left the BIPAD host; stopping.")
            truncated = True
            break
        seen_urls.add(next_url)

        payload = _request_json(next_url, timeout_s, max_retries, retry_backoff_s)
        if payload is None:
            logger.warning("[BIPAD] page %d unavailable; returning what was harvested.", pages_read + 1)
            truncated = True
            break
        pages_read += 1

        if isinstance(payload, dict):
            results = payload.get("results")
            candidate_next = payload.get("next")
        elif isinstance(payload, list):
            results = payload
            candidate_next = None
        else:
            logger.warning(
                "[BIPAD] unexpected payload type %s; treating as no events.", type(payload).__name__
            )
            break

        if not isinstance(results, list):
            logger.warning("[BIPAD] payload has no 'results' list; treating as no events.")
            break
        if not results:
            break

        records_seen += len(results)
        # Window statistics are computed from the raw dates, independent of hazard class,
        # so a page dominated by irrelevant hazards (fire, snakebite) is not mistaken for
        # evidence that the server ignored the date filter.
        page_dates = [
            _parse_event_date(r.get("incidentOn")) or _parse_event_date(r.get("reportedOn"))
            for r in results
            if isinstance(r, dict)
        ]
        dated = [d for d in page_dates if d]
        in_window = sum(1 for d in dated if d >= cutoff.isoformat())

        for raw in results:
            event = normalize_incident(raw)
            if event is None:
                continue
            if not (cutoff.isoformat() <= event["event_date"] <= horizon.isoformat()):
                continue
            if event["hazard_type"] not in hazard_types:
                continue
            if event["external_id"] in seen_external_ids:
                continue
            seen_external_ids.add(event["external_id"])
            collected.append(event)

        if server_filtered is None:
            # If nearly everything on page 1 is inside the window, the portal honoured the
            # date filter and pagination will be short. Otherwise we are scanning the whole
            # catalog, which may need every page.
            server_filtered = bool(dated) and (in_window / len(dated)) >= 0.9
            logger.info(
                "[BIPAD] server-side date filter %s (%d/%d dated records on page 1 within "
                "the %d-day window).",
                "appears honoured"
                if server_filtered
                else "appears IGNORED; falling back to a full scan",
                in_window,
                len(dated),
                since_days,
            )

        # If the portal also honoured `ordering=-incidentOn`, a page that ends older than
        # the cutoff means every later page is older too.
        descending = len(dated) > 1 and all(
            dated[i] >= dated[i + 1] for i in range(len(dated) - 1)
        )
        if descending and dated and dated[-1] < cutoff.isoformat():
            logger.info("[BIPAD] reached records older than the window on an ordered feed; stopping.")
            break

        if not isinstance(candidate_next, str) or not candidate_next:
            break
        next_url = urllib.parse.urljoin(next_url, candidate_next)
        if page_delay_s > 0:
            time.sleep(page_delay_s)

    if truncated:
        logger.warning(
            "[BIPAD] scan incomplete after %d page(s); harvest may be missing events.", pages_read
        )

    _flag_repeated_coordinates(collected)
    logger.info(
        "[BIPAD] harvested %d candidate event(s) from %d record(s) across %d page(s).",
        len(collected),
        records_seen,
        pages_read,
    )
    return collected


def build_staging_key(latitude: float, longitude: float, event_date: str) -> str:
    """Deterministic staging-queue primary key for one (lat, lon, event_date) triple.

    Because the key is derived from the event itself rather than from a timestamp,
    re-running the weekly harvest re-inserts the same id, and both the SQLite
    (INSERT OR IGNORE) and PostGIS (ON CONFLICT DO NOTHING) stores drop it. Two distinct
    BIPAD incidents sharing a ward centroid on the same day therefore collapse into one
    queue entry, which is deliberate: duplicate training rows silently bias the model.
    """
    digest = hashlib.md5(
        f"{latitude:.4f}|{longitude:.4f}|{event_date}".encode("utf-8")
    ).hexdigest()[:12]
    return f"bip_{digest}"


def _catalog_coordinate_keys() -> Set[Tuple[float, float]]:
    """Coordinates already present in the training catalog, rounded to 4 decimals.

    The catalog schema records `verified_timestamp_utc` (ingestion time), not the event
    date, so a (lat, lon, date) match against it is not possible; matching on coordinate
    alone is the conservative choice.
    """
    try:
        return {
            (round(float(ev["latitude"]), 4), round(float(ev["longitude"]), 4))
            for ev in load_live_ground_truth_events()
        }
    except Exception as exc:
        logger.warning("[BIPAD] could not read training catalog for dedup: %s", exc)
        return set()


def _pending_staging_keys(store: BaseTimeSeriesStore, limit: int = 500) -> Set[str]:
    try:
        return {
            b.get("bulletin_id")
            for b in store.get_pending_bulletins(limit=limit)
            if b.get("bulletin_id")
        }
    except Exception as exc:
        logger.warning("[BIPAD] could not read staging queue for dedup: %s", exc)
        return set()


def _build_bulletin_text(event: Dict[str, Any]) -> str:
    """Compose the staged bulletin body, including machine-readable coordinates."""
    return (
        f"NDRRMA BIPAD Portal official incident record {event['external_id']}. "
        f"Coordinates: {event['latitude']:.4f}N, {event['longitude']:.4f}E. "
        f"Hazard: {event['bipad_hazard_label']} (normalized type: {event['hazard_type']}). "
        f"Date of incident: {event['event_date']}. "
        f"District: {event['district'] or 'not resolved from record text'}. "
        f"Reported by: {event['reported_by']}; verified={event['verified']}; "
        f"approved={event['approved']}. "
        f"Harvester provenance confidence {event['confidence']:.2f}; "
        f"coordinate precision {event['coordinate_precision']} — BIPAD points are often "
        f"ward/municipality administrative centroids rather than the failure location, so "
        f"this coordinate must be confirmed before it is used as a training label. "
        f"Record text: {event['description']}"
    )


def stage_bipad_incidents(
    store: Optional[BaseTimeSeriesStore] = None,
    since_days: int = 7,
    **fetch_kwargs: Any,
) -> Dict[str, Any]:
    """Harvest BIPAD incidents and stage them in the PENDING bulletin queue.

    Nothing reaches `live_ground_truth_feedback.csv` here. The weekly audit's dual-model
    consensus gate plus human approval remain the only path into the training catalog.

    Returns a summary dict: {'fetched', 'queued', 'suppressed_duplicates', 'errors'}.
    Never raises.
    """
    summary: Dict[str, Any] = {
        "fetched": 0,
        "queued": 0,
        "suppressed_duplicates": 0,
        "errors": 0,
    }

    events = fetch_recent_incidents(since_days=since_days, **fetch_kwargs)
    summary["fetched"] = len(events)
    if not events:
        return summary

    try:
        if store is None:
            store = get_time_series_store()
    except Exception as exc:
        logger.warning("[BIPAD] no store available; nothing staged: %s", exc)
        summary["errors"] = len(events)
        return summary

    catalog_coords = _catalog_coordinate_keys()
    pending_keys = _pending_staging_keys(store)
    batch_keys: Set[str] = set()

    for event in events:
        key = build_staging_key(event["latitude"], event["longitude"], event["event_date"])
        if key in batch_keys or key in pending_keys:
            summary["suppressed_duplicates"] += 1
            continue
        if (event["latitude"], event["longitude"]) in catalog_coords:
            summary["suppressed_duplicates"] += 1
            continue
        batch_keys.add(key)

        try:
            store.insert_unprocessed_bulletin(
                {
                    "bulletin_id": key,
                    "received_utc": time.time(),
                    "source": SOURCE_TAG,
                    "title": (
                        f"{event['bipad_hazard_label']} at "
                        f"{event['district'] or 'unresolved district'} "
                        f"({event['event_date']}) [{event['external_id']}]"
                    ),
                    "raw_text": _build_bulletin_text(event),
                    "url": event["url"],
                    "district_hint": event["district"],
                    "status": "PENDING",
                    "parsed_json": event,
                }
            )
            summary["queued"] += 1
        except Exception as exc:
            logger.warning("[BIPAD] failed to stage %s: %s", event["external_id"], exc)
            summary["errors"] += 1

    logger.info(
        "[BIPAD] staged %d/%d incident(s); %d duplicate(s) suppressed, %d error(s).",
        summary["queued"],
        summary["fetched"],
        summary["suppressed_duplicates"],
        summary["errors"],
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(stage_bipad_incidents(since_days=7), indent=2), flush=True)
