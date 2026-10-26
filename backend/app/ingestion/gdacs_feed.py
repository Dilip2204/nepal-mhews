"""GDACS official multi-hazard event ingestion — the reachable stand-in for BIPAD.

WHY THIS EXISTS
    NDRRMA's BIPAD portal was the intended source of *official, coordinate-bearing*
    disaster records, but `bipadportal.gov.np` times out on both TCP 443 and 80 from this
    host (a network-level block, not DNS), so `app.ingestion.bipad_scraper` ships
    disabled. GDACS — the Global Disaster Alert and Coordination System, operated by the
    EU Joint Research Centre with UN OCHA — is reachable, free, needs no key, and every
    item in its feed carries explicit `<geo:lat>` / `<geo:long>` coordinates. That is the
    capability BIPAD was wanted for.

WHAT THIS MODULE DOES NOT DO
    It never writes to the training catalog. Every event is staged into the
    `unprocessed_bulletins` queue with status PENDING, so it still has to pass the weekly
    dual-model consensus gate and human approval before it can become a training label.
    The physics engine, not this module and not an LLM, sets alert levels.

READ THIS BEFORE TRUSTING A COORDINATE  (the whole point of the module)
    A GDACS coordinate is NOT the location of a slope failure. Verified against the live
    feed and the live event archive on 2026-10-24:

      * Earthquakes (`EQ`) carry a seismological **epicentre**. A co-seismic landslide can
        occur tens of km from it, and the epicentre itself has its own location error.
        -> coordinate_precision='EPICENTRE'
      * Floods (`FL`) come from GloFAS and the archive labels them explicitly:
        `"polygonlabel": "Centroid"`, `"Class": "Point_Centroid"`. The Red Nepal flood of
        2026-08-26 is filed as a single point (85.3649, 27.2953) for an event spanning a
        whole river system.
        -> coordinate_precision='BASIN_CENTROID'
      * Multi-country events (droughts, regional floods) are a centroid over an area that
        can exceed 10^6 km2.
        -> coordinate_precision='REGION_CENTROID'

    `gdacs:bbox` is NOT a location-uncertainty estimate: in every sampled item it is a
    fixed +/-4 degree box around the point, so it is deliberately ignored here.
    `location_uncertainty_hint_km` is an order-of-magnitude provenance judgment attached
    to the precision class — it is NOT a measured error bar, and the lead's distance-based
    correlation step must treat it as such. `confidence` is likewise a provenance
    heuristic, never a model probability, and never reaches 1.0.

    If an item has no parseable coordinate, date or hazard class it is DROPPED, never
    guessed or back-filled from a default.

FEED COVERAGE — MEASURED, NOT ASSUMED
    The live `rss.xml` snapshot of 2026-10-24 held 315 items, all with coordinates, but
    **zero inside the Himalaya window**: 282 wildfires, 12 droughts, 9 floods, 8
    earthquakes, 4 cyclones, scattered worldwide. The RSS is a *current-events* feed. For
    Nepal specifically it will be empty most weeks and will carry an event only while one
    is active, so an empty result is the normal case and not a fault.

    The GDACS event-list archive (`fetch_gdacs_archive_events`) is the historical
    counterpart: a live query for Nepal, 2024-01-01 to 2026-10-24, returned 33 events
    (19 floods, 8 earthquakes, 4 wildfires, 2 droughts) with real coordinates, including
    the Red flood above. That is the backfill path for seeding the review queue with real
    labels; the RSS is the weekly incremental path.

NAMESPACES
    geo    http://www.w3.org/2003/01/geo/wgs84_pos#
    gdacs  http://www.gdacs.org            (no trailing slash)
    georss http://www.georss.org/georss
    dc     http://purl.org/dc/elements/1.1/
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.config import settings
from app.core.database import BaseTimeSeriesStore, get_time_series_store
from app.ingestion.ground_truth_feedback import load_live_ground_truth_events
from app.ingestion.unstructured_collector import extract_district_hint

logger = logging.getLogger("nepal_mhews.gdacs_feed")

SOURCE_TAG = "GDACS"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
# Dedup state: a weekly cron must not re-stage the same GDACS event forever.
GDACS_STATE_PATH = DATA_DIR / "gdacs_ingest_state.json"
STATE_VERSION = 1
STATE_RETENTION_DAYS = 540

# Nepal Standard Time is UTC+05:45 year round (no DST), so a fixed offset is exact.
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))

# Nepal + the surrounding Himalayan arc (lon 80-89, lat 26-31). The filter is spatial,
# not political: a failure 20 km across the Indian or Chinese border is still inside the
# monitored corridors' hazard envelope, so `country` is recorded rather than used to
# reject. GDACS is global and unfiltered would flood the queue with Pacific earthquakes.
HIMALAYA_BBOX: Tuple[float, float, float, float] = (80.0, 26.0, 89.0, 31.0)

USER_AGENT = (
    "Himalayan-MHEWS/1.0 (Nepal multi-hazard early warning research system; "
    "weekly GDACS official event harvest; contact dilip.jena@marqueesemi.com)"
)

# Historical counterpart to the RSS feed. Not in settings (config.py is owned by the lead
# and only carries `gdacs_rss_url`); promote it there if the backfill becomes routine.
GDACS_ARCHIVE_SEARCH_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"

RSS_NAMESPACES: Dict[str, str] = {
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "gdacs": "http://www.gdacs.org",
    "georss": "http://www.georss.org/georss",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# GDACS event type -> this project's hazard vocabulary (mirrors bipad_scraper's terms).
# Types with no slope/flood relevance map to None so they are discarded rather than
# polluting the queue. 'LS'/'MS' are accepted defensively: no landslide item appeared in
# the 2026-10-24 snapshot, so that branch is coded to the documented GDACS vocabulary and
# is NOT confirmed against a live item.
GDACS_EVENT_TYPE_TO_HAZARD: Dict[str, Optional[str]] = {
    "EQ": "earthquake",
    "FL": "flood",
    "FF": "flood",          # flash flood
    "LS": "landslide",
    "MS": "landslide",      # mass movement
    "AV": "rock_ice_avalanche",
    "TC": None,             # tropical cyclone
    "DR": None,             # drought
    "WF": None,             # wildfire
    "VO": None,             # volcano
}

GDACS_EVENT_TYPE_LABELS: Dict[str, str] = {
    "EQ": "Earthquake",
    "FL": "Flood",
    "FF": "Flash flood",
    "LS": "Landslide",
    "MS": "Mass movement",
    "AV": "Avalanche",
    "TC": "Tropical cyclone",
    "DR": "Drought",
    "WF": "Wildfire",
    "VO": "Volcano",
}

# Hazard types worth staging as candidate ground truth for a slope-stability model.
GEOHAZARD_TYPES: Tuple[str, ...] = (
    "landslide",
    "debris_flow",
    "rockfall",
    "rock_ice_avalanche",
    "glof",
    "flood",
    "earthquake",
)

# Text refinements, applied the same way as in bipad_scraper: a keyword may only NARROW a
# hazard GDACS already filed as a flood or a mass movement. It never overrides an official
# classification (an earthquake bulletin that mentions "landslides triggered" stays an
# earthquake, because its coordinate is still the epicentre).
_TEXT_REFINEMENTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("glof", ("glof", "glacial lake outburst", "moraine", "outburst flood")),
    ("rock_ice_avalanche", ("avalanche", "serac", "icefall", "rock-ice", "rock ice")),
    ("debris_flow", ("debris flow", "mudflow", "mud flow", "debris-flow")),
    ("rockfall", ("rockfall", "rock fall")),
)
_REFINABLE_TYPES = frozenset({"flood", "landslide"})

# GDACS alert level -> the project's alert vocabulary.
ALERT_LEVEL_MAP: Dict[str, str] = {
    "green": "GREEN",
    "orange": "ORANGE",
    "red": "RED",
}
_ALERT_RANK: Dict[str, int] = {"UNKNOWN": 0, "GREEN": 1, "ORANGE": 2, "RED": 3}

# Order-of-magnitude provenance hints, NOT measured error bars. See the module docstring.
COORDINATE_PRECISION_SCALE_KM: Dict[str, float] = {
    "REPORTED_POINT": 5.0,
    "EPICENTRE": 15.0,
    "BASIN_CENTROID": 50.0,
    "REGION_CENTROID": 200.0,
}

_PRECISION_BASE_CONFIDENCE: Dict[str, float] = {
    "REPORTED_POINT": 0.60,
    "EPICENTRE": 0.55,
    "BASIN_CENTROID": 0.35,
    "REGION_CENTROID": 0.20,
}

MAX_CONFIDENCE = 0.75


# ---------------------------------------------------------------------------
# Vocabulary mapping
# ---------------------------------------------------------------------------
def map_alert_level(raw: Any) -> str:
    """Map a GDACS alert level (Green/Orange/Red) onto the project's vocabulary.

    An unrecognized or missing level becomes 'UNKNOWN' rather than a cheerful default.
    """
    if not isinstance(raw, str):
        return "UNKNOWN"
    return ALERT_LEVEL_MAP.get(raw.strip().lower(), "UNKNOWN")


def map_gdacs_event_type(event_type: Any, text: str = "") -> Optional[str]:
    """Map a GDACS event type code (plus free text) onto this project's hazard vocabulary.

    Returns None when the event is not a slope/flood process or cannot be classified,
    which causes the caller to drop the item.
    """
    code = event_type.strip().upper() if isinstance(event_type, str) else ""
    known = code in GDACS_EVENT_TYPE_TO_HAZARD
    base = GDACS_EVENT_TYPE_TO_HAZARD.get(code)

    # GDACS filed this under a hazard it tracks and that hazard is irrelevant to slope
    # stability (wildfire, drought, cyclone). Free text must not override that.
    if known and base is None:
        return None

    lowered = (text or "").lower()
    if base is None or base in _REFINABLE_TYPES:
        for refined, keywords in _TEXT_REFINEMENTS:
            if any(kw in lowered for kw in keywords):
                return refined

    if base is None and not known and "landslide" in lowered:
        return "landslide"
    return base


def _coordinate_precision(hazard_type: str, event_code: str, hints: str = "") -> str:
    """Classify how far a GDACS point can plausibly sit from the actual failure."""
    lowered = (hints or "").lower()
    if "region" in lowered or lowered.count(",") >= 2:
        # Multi-country `country` strings mark a regional centroid.
        return "REGION_CENTROID"
    # Checked BEFORE the polygon label: the archive labels every point 'Centroid' /
    # 'Point_Centroid', earthquakes included (verified on eventid 1564287), so that label
    # is generic. An EQ coordinate is a seismological epicentre whatever GDACS calls it.
    if event_code == "EQ":
        return "EPICENTRE"
    if "centroid" in lowered:
        return "BASIN_CENTROID"
    if event_code in ("FL", "FF"):
        return "BASIN_CENTROID"
    if hazard_type in ("landslide", "debris_flow", "rockfall", "rock_ice_avalanche", "glof"):
        return "REPORTED_POINT"
    return "BASIN_CENTROID"


def _provenance_confidence(precision: str, severity: str, country: str) -> float:
    """Provenance heuristic in [0.10, 0.75]. Never a model probability, never 1.0."""
    score = _PRECISION_BASE_CONFIDENCE.get(precision, 0.25)
    if severity == "RED":
        score += 0.10
    elif severity == "ORANGE":
        score += 0.05
    if "nepal" in (country or "").lower():
        score += 0.05
    return round(min(MAX_CONFIDENCE, max(0.10, score)), 2)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _clean(value: Any) -> str:
    return " ".join(str(value).split()) if value not in (None, "") else ""


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return out


def _parse_rfc822(raw: Any) -> Optional[datetime]:
    """Parse an RSS pubDate such as 'Sat, 24 Oct 2026 05:23:40 GMT'."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dt = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_iso(raw: Any) -> Optional[datetime]:
    """Parse a GDACS API timestamp such as '2026-08-26T01:00:00' (UTC, often naive)."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _in_bbox(lat: float, lon: float, bbox: Sequence[float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def _build_event(
    *,
    latitude: float,
    longitude: float,
    when: datetime,
    event_code: str,
    hazard_type: str,
    title: str,
    description: str,
    external_id: str,
    source_url: str,
    alert_level: str,
    country: str,
    iso3: str,
    episode_id: str,
    severity_text: str,
    severity_value: Optional[float],
    severity_unit: str,
    feed: str,
    precision_hints: str,
    event_date_authoritative: bool = True,
) -> Dict[str, Any]:
    precision = _coordinate_precision(hazard_type, event_code, precision_hints)
    text_blob = " ".join([title, description, country])
    return {
        # --- Core project event shape ---
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "event_date": when.astimezone(NEPAL_TZ).date().isoformat(),
        "event_time_utc": round(when.timestamp(), 2),
        # False => the date came from a FEED timestamp, not the event itself.
        # Temporal correlation must not treat it as the time of occurrence.
        "event_date_authoritative": bool(event_date_authoritative),
        "hazard_type": hazard_type,
        "title": title or f"{GDACS_EVENT_TYPE_LABELS.get(event_code, 'Event')} ({external_id})",
        "source": SOURCE_TAG,
        "source_url": source_url,
        "external_id": external_id,
        "severity": alert_level,
        "country": country,
        "confidence": _provenance_confidence(precision, alert_level, country),
        # --- Provenance extras consumed by staging and the correlation step ---
        "description": description,
        "district": extract_district_hint(text_blob) or None,
        "gdacs_event_type": event_code,
        "gdacs_event_label": GDACS_EVENT_TYPE_LABELS.get(event_code, "Unclassified"),
        "gdacs_alert_level": alert_level,
        "episode_id": episode_id,
        "iso3": iso3,
        "severity_text": severity_text,
        "severity_value": severity_value,
        "severity_unit": severity_unit,
        "coordinate_precision": precision,
        # Order-of-magnitude judgment from the precision class; NOT a measured error.
        "location_uncertainty_hint_km": COORDINATE_PRECISION_SCALE_KM.get(precision, 50.0),
        "feed": feed,
    }


def normalize_rss_item(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize one `<item>` element of the GDACS RSS feed, or None if unusable.

    Nothing is synthesized: an item without a parseable coordinate, date or relevant
    hazard class returns None so the caller drops it.
    """
    try:
        if item is None:
            return None

        point = item.find("geo:Point", RSS_NAMESPACES)
        latitude = longitude = None
        if point is not None:
            latitude = _to_float(point.findtext("geo:lat", namespaces=RSS_NAMESPACES))
            longitude = _to_float(point.findtext("geo:long", namespaces=RSS_NAMESPACES))
        if latitude is None or longitude is None:
            # georss:point is 'lat lon' in a single element.
            parts = _clean(item.findtext("georss:point", namespaces=RSS_NAMESPACES)).split()
            if len(parts) == 2:
                latitude, longitude = _to_float(parts[0]), _to_float(parts[1])
        # gdacs:bbox is a fixed +/-4 degree box, never a real extent: not used as a fallback.
        if latitude is None or longitude is None:
            return None
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            return None

        # Only gdacs:fromdate is the event's own start time. pubDate/dateadded are FEED
        # timestamps: a 2024 flood re-published today would be stamped today, sail through
        # a since_days filter, and post-date a live RED alert - manufacturing a
        # "prediction confirmed". The substitution is kept (the event is still a useful
        # candidate) but flagged, so temporal correlation can refuse to rely on it.
        authoritative = _parse_rfc822(
            item.findtext("gdacs:fromdate", namespaces=RSS_NAMESPACES)
        )
        when = (
            authoritative
            or _parse_rfc822(item.findtext("pubDate"))
            or _parse_rfc822(item.findtext("gdacs:dateadded", namespaces=RSS_NAMESPACES))
        )
        if when is None:
            return None
        event_date_authoritative = authoritative is not None

        title = _clean(item.findtext("title"))
        description = _clean(item.findtext("description"))
        event_name = _clean(item.findtext("gdacs:eventname", namespaces=RSS_NAMESPACES))
        country = _clean(item.findtext("gdacs:country", namespaces=RSS_NAMESPACES))

        event_code = _clean(item.findtext("gdacs:eventtype", namespaces=RSS_NAMESPACES)).upper()
        hazard_type = map_gdacs_event_type(
            event_code, " ".join([title, description, event_name])
        )
        if hazard_type is None:
            return None

        event_id = _clean(item.findtext("gdacs:eventid", namespaces=RSS_NAMESPACES))
        if not event_id:
            guid = _clean(item.findtext("guid"))
            event_id = guid[2:] if guid[:2].upper() == event_code and len(guid) > 2 else guid
        if not event_id:
            return None

        severity_el = item.find("gdacs:severity", RSS_NAMESPACES)
        severity_text = _clean(severity_el.text) if severity_el is not None else ""
        severity_value = _to_float(severity_el.get("value")) if severity_el is not None else None
        severity_unit = _clean(severity_el.get("unit")) if severity_el is not None else ""

        link = _clean(item.findtext("link")) or (
            f"https://www.gdacs.org/report.aspx?eventtype={event_code}&eventid={event_id}"
        )

        return _build_event(
            latitude=latitude,
            longitude=longitude,
            when=when,
            event_code=event_code,
            hazard_type=hazard_type,
            title=title,
            description=" | ".join(p for p in (description, event_name) if p),
            external_id=f"gdacs-{event_code or 'XX'}-{event_id}",
            source_url=link,
            alert_level=map_alert_level(
                item.findtext("gdacs:alertlevel", namespaces=RSS_NAMESPACES)
            ),
            country=country,
            iso3=_clean(item.findtext("gdacs:iso3", namespaces=RSS_NAMESPACES)),
            episode_id=_clean(item.findtext("gdacs:episodeid", namespaces=RSS_NAMESPACES)),
            severity_text=severity_text,
            severity_value=severity_value,
            severity_unit=severity_unit,
            feed="RSS",
            precision_hints=country,
            event_date_authoritative=event_date_authoritative,
        )
    except Exception as exc:  # a single malformed item must not kill the batch
        logger.warning("[GDACS] skipped an unparseable RSS item: %s", exc)
        return None


def normalize_archive_feature(feature: Any) -> Optional[Dict[str, Any]]:
    """Normalize one GeoJSON feature of the GDACS event-list archive, or None if unusable."""
    try:
        if not isinstance(feature, dict):
            return None
        props = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(props, dict) or not isinstance(geometry, dict):
            return None
        coords = geometry.get("coordinates")
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            return None
        longitude, latitude = _to_float(coords[0]), _to_float(coords[1])
        if latitude is None or longitude is None:
            return None
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            return None

        # See normalize_rss_item: datemodified is when GDACS last touched the record, not
        # when the event happened. Flag the substitution instead of hiding it.
        authoritative = _parse_iso(props.get("fromdate"))
        when = authoritative or _parse_iso(props.get("datemodified"))
        if when is None:
            return None
        event_date_authoritative = authoritative is not None

        event_code = _clean(props.get("eventtype")).upper()
        title = _clean(props.get("htmldescription")) or _clean(props.get("name"))
        description = _clean(props.get("description"))
        country = _clean(props.get("country"))
        hazard_type = map_gdacs_event_type(event_code, " ".join([title, description]))
        if hazard_type is None:
            return None

        event_id = _clean(props.get("eventid"))
        if not event_id:
            return None

        severity = props.get("severitydata") if isinstance(props.get("severitydata"), dict) else {}
        urls = props.get("url") if isinstance(props.get("url"), dict) else {}
        report_url = _clean(urls.get("report")) or (
            f"https://www.gdacs.org/report.aspx?eventtype={event_code}&eventid={event_id}"
        )

        # The archive states its own geometry class, which beats any guess by event type.
        precision_hints = " ".join(
            [_clean(props.get("polygonlabel")), _clean(props.get("Class")), country]
        )

        return _build_event(
            latitude=latitude,
            longitude=longitude,
            when=when,
            event_code=event_code,
            hazard_type=hazard_type,
            title=title,
            description=description,
            external_id=f"gdacs-{event_code or 'XX'}-{event_id}",
            source_url=report_url,
            alert_level=map_alert_level(props.get("alertlevel")),
            country=country,
            iso3=_clean(props.get("iso3")),
            episode_id=_clean(props.get("episodeid")),
            severity_text=_clean(severity.get("severitytext")),
            severity_value=_to_float(severity.get("severity")),
            severity_unit=_clean(severity.get("severityunit")),
            feed="ARCHIVE_SEARCH",
            precision_hints=precision_hints,
            event_date_authoritative=event_date_authoritative,
        )
    except Exception as exc:
        logger.warning("[GDACS] skipped an unparseable archive feature: %s", exc)
        return None


def parse_gdacs_rss(payload: Any) -> List[Dict[str, Any]]:
    """Parse a GDACS RSS document into normalized events. Never raises.

    Malformed or empty XML yields an empty list. No spatial or recency filtering happens
    here; see `filter_events`.
    """
    if payload is None:
        return []
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if not isinstance(payload, (bytes, bytearray)):
        return []
    body = bytes(payload).lstrip(b"\xef\xbb\xbf").strip()
    if not body:
        return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        logger.warning("[GDACS] feed is not well-formed XML: %s", exc)
        return []

    items = root.findall("./channel/item") or root.findall(".//item")
    events: List[Dict[str, Any]] = []
    for item in items:
        normalized = normalize_rss_item(item)
        if normalized is not None:
            events.append(normalized)
    return events


def filter_events(
    events: Iterable[Dict[str, Any]],
    bbox: Sequence[float] = HIMALAYA_BBOX,
    hazard_types: Sequence[str] = GEOHAZARD_TYPES,
    since_days: Optional[int] = None,
    min_earthquake_magnitude: float = 0.0,
    max_items: int = 200,
) -> List[Dict[str, Any]]:
    """Keep only Himalayan, slope-relevant, recent-enough events, newest first."""
    allowed = set(hazard_types)
    cutoff: Optional[float] = None
    if since_days is not None and since_days > 0:
        cutoff = time.time() - since_days * 86400.0

    kept: List[Dict[str, Any]] = []
    for ev in events:
        try:
            if not _in_bbox(ev["latitude"], ev["longitude"], bbox):
                continue
            if ev["hazard_type"] not in allowed:
                continue
            if cutoff is not None and float(ev["event_time_utc"]) < cutoff:
                continue
            if (
                min_earthquake_magnitude > 0.0
                and ev["gdacs_event_type"] == "EQ"
                and (ev.get("severity_value") or 0.0) < min_earthquake_magnitude
            ):
                continue
            kept.append(ev)
        except (KeyError, TypeError, ValueError):
            continue

    kept.sort(key=lambda e: e.get("event_time_utc", 0.0), reverse=True)
    return kept[: max(0, int(max_items))]


# ---------------------------------------------------------------------------
# Polite HTTP
# ---------------------------------------------------------------------------
def _request_bytes(
    url: str,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    warnings: Optional[List[str]],
) -> Optional[bytes]:
    """GET a URL, returning None (plus a recorded warning) on any failure. Never raises."""
    attempts = max(1, int(max_retries))
    last_error: Optional[str] = None
    for attempt in range(attempts):
        response = None
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/xml, text/xml, application/json;q=0.9, */*;q=0.1",
                },
            )
            response = urllib.request.urlopen(request, timeout=timeout_s)
            return response.read()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < attempts and retry_backoff_s > 0:
                time.sleep(retry_backoff_s * (attempt + 1))
        finally:
            try:
                if response is not None:
                    response.close()
            except Exception:
                pass

    message = f"GDACS fetch failed after {attempts} attempt(s) for {url}: {last_error}"
    logger.warning("[GDACS] %s", message)
    if warnings is not None:
        warnings.append(message)
    return None


def fetch_gdacs_events(
    enabled: Optional[bool] = None,
    rss_url: Optional[str] = None,
    bbox: Sequence[float] = HIMALAYA_BBOX,
    hazard_types: Sequence[str] = GEOHAZARD_TYPES,
    since_days: Optional[int] = None,
    min_earthquake_magnitude: float = 0.0,
    max_items: int = 200,
    timeout_s: float = 20.0,
    max_retries: int = 2,
    retry_backoff_s: float = 2.0,
    warnings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch the GDACS RSS feed and return normalized Himalayan geohazard events.

    Never raises. A disabled setting, a dead feed, malformed XML or a feed with no
    Himalayan items all return `[]`; failures are logged and appended to `warnings` when
    a list is supplied. An empty result is the normal weekly case for Nepal (see the
    module docstring) and must not be read as an error.
    """
    is_enabled = settings.gdacs_enabled if enabled is None else bool(enabled)
    if not is_enabled:
        logger.info("[GDACS] disabled (GDACS_ENABLED=0); no request made.")
        return []

    url = rss_url or settings.gdacs_rss_url
    raw = _request_bytes(url, timeout_s, max_retries, retry_backoff_s, warnings)
    if raw is None:
        return []

    parsed = parse_gdacs_rss(raw)
    if not parsed and warnings is not None:
        warnings.append(f"GDACS feed at {url} yielded no parseable items.")

    events = filter_events(
        parsed,
        bbox=bbox,
        hazard_types=hazard_types,
        since_days=since_days,
        min_earthquake_magnitude=min_earthquake_magnitude,
        max_items=max_items,
    )
    logger.info(
        "[GDACS] %d/%d feed item(s) fell inside the Himalaya window and the hazard filter.",
        len(events),
        len(parsed),
    )
    return events


def fetch_gdacs_archive_events(
    from_date: str,
    to_date: str,
    country: str = "Nepal",
    alert_levels: str = "Green;Orange;Red",
    enabled: Optional[bool] = None,
    archive_url: Optional[str] = None,
    bbox: Sequence[float] = HIMALAYA_BBOX,
    hazard_types: Sequence[str] = GEOHAZARD_TYPES,
    min_earthquake_magnitude: float = 0.0,
    max_items: int = 500,
    timeout_s: float = 40.0,
    max_retries: int = 2,
    retry_backoff_s: float = 2.0,
    warnings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Backfill historical GDACS events (GeoJSON event-list archive). Never raises.

    `from_date` / `to_date` are 'YYYY-MM-DD'. This is the one-off seeding path for the
    review queue; `fetch_gdacs_events` remains the weekly incremental path. Verified live
    on 2026-10-24: Nepal 2024-01-01..2026-10-24 returned 33 events with real coordinates.

    `alert_levels` is NOT optional in practice: the same query without it returned only
    3 of those 33 events (the endpoint silently narrows its own default), so the levels
    are always sent explicitly.
    """
    is_enabled = settings.gdacs_enabled if enabled is None else bool(enabled)
    if not is_enabled:
        logger.info("[GDACS] disabled (GDACS_ENABLED=0); no archive request made.")
        return []

    base = archive_url or GDACS_ARCHIVE_SEARCH_URL
    query = urllib.parse.urlencode(
        {
            "country": country,
            "fromDate": from_date,
            "toDate": to_date,
            "alertlevel": alert_levels,
        },
        safe=";",
    )
    raw = _request_bytes(
        f"{base}?{query}", timeout_s, max_retries, retry_backoff_s, warnings
    )
    if raw is None:
        return []

    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
        features = payload.get("features") if isinstance(payload, dict) else None
        features = features if isinstance(features, list) else []
    except (ValueError, AttributeError) as exc:
        message = f"GDACS archive returned non-JSON content: {exc}"
        logger.warning("[GDACS] %s", message)
        if warnings is not None:
            warnings.append(message)
        return []

    parsed = [ev for ev in (normalize_archive_feature(f) for f in features) if ev]
    return filter_events(
        parsed,
        bbox=bbox,
        hazard_types=hazard_types,
        since_days=None,
        min_earthquake_magnitude=min_earthquake_magnitude,
        max_items=max_items,
    )


# ---------------------------------------------------------------------------
# Cross-run dedup state
# ---------------------------------------------------------------------------
def build_staging_key(external_id: str) -> str:
    """Deterministic staging-queue primary key for one GDACS external event id.

    Derived from the event id rather than a timestamp, so a repeated weekly harvest
    re-inserts the same id and both stores (SQLite INSERT OR IGNORE / PostGIS ON CONFLICT
    DO NOTHING) drop it even if the JSON state file is lost.
    """
    digest = hashlib.md5(str(external_id).encode("utf-8")).hexdigest()[:12]
    return f"gdc_{digest}"


def load_ingest_state(state_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the persisted dedup state. A missing or corrupt file yields an empty state."""
    path = Path(state_path) if state_path else GDACS_STATE_PATH
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("events"), dict):
            return data
        logger.warning("[GDACS] state file %s has an unexpected shape; starting empty.", path)
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        logger.warning("[GDACS] could not read state file %s: %s", path, exc)
    return {"version": STATE_VERSION, "updated_utc": 0.0, "events": {}}


def save_ingest_state(state: Dict[str, Any], state_path: Optional[Path] = None) -> bool:
    """Persist the dedup state atomically, pruning entries older than the retention window."""
    path = Path(state_path) if state_path else GDACS_STATE_PATH
    cutoff = time.time() - STATE_RETENTION_DAYS * 86400.0
    events = {
        key: value
        for key, value in (state.get("events") or {}).items()
        if isinstance(value, dict) and float(value.get("last_seen_utc") or 0.0) >= cutoff
    }
    payload = {"version": STATE_VERSION, "updated_utc": round(time.time(), 2), "events": events}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        os.replace(tmp_path, path)
        return True
    except OSError as exc:
        logger.warning("[GDACS] could not write state file %s: %s", path, exc)
        return False


def _catalog_coordinate_keys() -> Set[Tuple[float, float]]:
    """Coordinates already in the training catalog, rounded to 4 decimals (same as BIPAD)."""
    try:
        return {
            (round(float(ev["latitude"]), 4), round(float(ev["longitude"]), 4))
            for ev in load_live_ground_truth_events()
        }
    except Exception as exc:
        logger.warning("[GDACS] could not read training catalog for dedup: %s", exc)
        return set()


def _pending_staging_keys(store: BaseTimeSeriesStore, limit: int = 500) -> Set[str]:
    try:
        return {
            b.get("bulletin_id")
            for b in store.get_pending_bulletins(limit=limit)
            if b.get("bulletin_id")
        }
    except Exception as exc:
        logger.warning("[GDACS] could not read staging queue for dedup: %s", exc)
        return set()


def _build_bulletin_text(event: Dict[str, Any]) -> str:
    """Compose the staged bulletin body, including machine-readable coordinates."""
    severity_bits = event.get("severity_text") or "not reported"
    return (
        f"GDACS official event record {event['external_id']} "
        f"(EU JRC / UN OCHA Global Disaster Alert and Coordination System). "
        f"Coordinates: {event['latitude']:.4f}N, {event['longitude']:.4f}E. "
        f"Hazard: {event['gdacs_event_label']} (normalized type: {event['hazard_type']}). "
        f"GDACS alert level: {event['severity']}. Severity: {severity_bits}. "
        f"Date of event: {event['event_date']}. Country: {event['country'] or 'not stated'}. "
        f"District: {event['district'] or 'not resolved from the bulletin text'}. "
        f"Harvester provenance confidence {event['confidence']:.2f}; coordinate precision "
        f"{event['coordinate_precision']} (indicative scale ~"
        f"{event['location_uncertainty_hint_km']:.0f} km, an order-of-magnitude judgment, "
        f"NOT a measured error). A GDACS point is an epicentre or a model/basin centroid, "
        f"not a surveyed failure location, so this coordinate MUST be confirmed before it "
        f"is used as a training label. Feed: {event['feed']}. "
        f"Report: {event['source_url']}. Text: {event['description'] or event['title']}"
    )


def stage_gdacs_events(
    store: Optional[BaseTimeSeriesStore] = None,
    since_days: Optional[int] = 30,
    state_path: Optional[Path] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    restage_on_escalation: bool = True,
    check_training_catalog: bool = True,
    **fetch_kwargs: Any,
) -> Dict[str, Any]:
    """Harvest GDACS events and stage them PENDING for the dual-model audit. Never raises.

    Nothing reaches `live_ground_truth_feedback.csv` here: the weekly consensus gate plus
    human approval remain the only path into the training catalog.

    Dedup has three independent layers, so losing any one of them does not cause repeats:
      1. the persisted state file, keyed by GDACS external event id (survives restarts);
      2. the pending staging queue;
      3. the store's own primary key, via a deterministic `bulletin_id`.
    An already-ingested event is re-staged only when its GDACS alert level ESCALATES
    (GREEN -> ORANGE -> RED), which is materially new information and can happen at most
    twice per event.

    Pass `events=` to stage an already-fetched batch (e.g. from
    `fetch_gdacs_archive_events`) instead of fetching the RSS feed.

    Returns {'fetched', 'queued', 'suppressed_duplicates', 'restaged_escalations',
    'errors', 'warnings', 'state_saved'}.
    """
    summary: Dict[str, Any] = {
        "fetched": 0,
        "queued": 0,
        "suppressed_duplicates": 0,
        "restaged_escalations": 0,
        "errors": 0,
        "warnings": [],
        "state_saved": False,
    }

    try:
        warnings: List[str] = summary["warnings"]
        batch = (
            list(events)
            if events is not None
            else fetch_gdacs_events(since_days=since_days, warnings=warnings, **fetch_kwargs)
        )
        summary["fetched"] = len(batch)
        if not batch:
            return summary

        try:
            if store is None:
                store = get_time_series_store()
        except Exception as exc:
            message = f"no store available; nothing staged: {exc}"
            logger.warning("[GDACS] %s", message)
            warnings.append(message)
            summary["errors"] = len(batch)
            return summary

        state = load_ingest_state(state_path)
        known: Dict[str, Any] = state.get("events") or {}
        pending_keys = _pending_staging_keys(store)
        catalog_coords = _catalog_coordinate_keys() if check_training_catalog else set()
        seen_this_run: Set[str] = set()
        now = round(time.time(), 2)
        state_dirty = False

        for event in batch:
            external_id = event.get("external_id")
            if not external_id or external_id in seen_this_run:
                summary["suppressed_duplicates"] += 1
                continue
            seen_this_run.add(external_id)

            key = build_staging_key(external_id)
            prior = known.get(external_id) if isinstance(known.get(external_id), dict) else None
            escalated = False
            if prior is not None:
                prior_rank = _ALERT_RANK.get(str(prior.get("alert_level", "UNKNOWN")), 0)
                escalated = (
                    restage_on_escalation
                    and _ALERT_RANK.get(event.get("severity", "UNKNOWN"), 0) > prior_rank
                )
                prior["last_seen_utc"] = now
                state_dirty = True
                if not escalated:
                    summary["suppressed_duplicates"] += 1
                    continue
                # A re-staged escalation needs its own queue row, keyed by the new level.
                key = f"{key}_{event.get('severity', 'UNKNOWN').lower()}"

            if key in pending_keys:
                summary["suppressed_duplicates"] += 1
                continue
            if (event["latitude"], event["longitude"]) in catalog_coords:
                summary["suppressed_duplicates"] += 1
                continue

            try:
                store.insert_unprocessed_bulletin(
                    {
                        "bulletin_id": key,
                        "received_utc": now,
                        "source": SOURCE_TAG,
                        "title": (
                            f"{event['gdacs_event_label']} "
                            f"[{event['severity']}] in "
                            f"{event['country'] or 'the Himalaya window'} "
                            f"({event['event_date']}) [{external_id}]"
                        ),
                        "raw_text": _build_bulletin_text(event),
                        "url": event["source_url"],
                        "district_hint": event["district"],
                        "status": "PENDING",
                        "parsed_json": event,
                    }
                )
                pending_keys.add(key)
                summary["queued"] += 1
                if escalated:
                    summary["restaged_escalations"] += 1
                known[external_id] = {
                    "first_seen_utc": (prior or {}).get("first_seen_utc", now),
                    "last_seen_utc": now,
                    "alert_level": event.get("severity", "UNKNOWN"),
                    "episode_id": event.get("episode_id", ""),
                    "event_date": event.get("event_date", ""),
                    "staged_bulletin_id": key,
                }
                state_dirty = True
            except Exception as exc:
                logger.warning("[GDACS] failed to stage %s: %s", external_id, exc)
                warnings.append(f"failed to stage {external_id}: {exc}")
                summary["errors"] += 1

        if state_dirty:
            state["events"] = known
            summary["state_saved"] = save_ingest_state(state, state_path)

        logger.info(
            "[GDACS] staged %d/%d event(s); %d duplicate(s) suppressed, %d escalation(s), "
            "%d error(s).",
            summary["queued"],
            summary["fetched"],
            summary["suppressed_duplicates"],
            summary["restaged_escalations"],
            summary["errors"],
        )
        return summary
    except Exception as exc:  # the staging entry point must never raise into a caller
        logger.warning("[GDACS] staging aborted: %s", exc)
        summary["warnings"].append(f"staging aborted: {exc}")
        summary["errors"] += 1
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(stage_gdacs_events(since_days=30), indent=2, default=str), flush=True)
