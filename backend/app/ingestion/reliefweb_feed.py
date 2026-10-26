"""ReliefWeb (UN OCHA) situation reports - the one feed here that carries article prose.

Why this source exists in the pipeline
--------------------------------------
Measured against the live feeds on 2026-10-26: **Google News RSS carries no article body.**
Across 100 items not one ``<description>`` exceeded its ``<title>`` by 60 characters - it is
the headline as a hyperlink plus the source name - and its ``<link>`` is an opaque redirect
that does not resolve server-side to the publisher. So the two text-reading components in
this system (``event_time.extract_event_time`` and ``geocoding.extract_toponym``) had only
headlines to work with, and roughly 24 of every 25 items fell back to the publication date.

ReliefWeb v2 returns a real ``body`` field: full situation-report text, usually naming both
the district and the date of the event. That is what makes those two components bite.

Access
------
v1 is decommissioned (HTTP 410). v2 refuses with HTTP 403 unless an **approved appname** is
supplied; it is free, from https://apidoc.reliefweb.int/parameters#appname. Set
``RELIEFWEB_APPNAME`` once it is granted. Until then this module reports itself unavailable
and returns nothing - it never fabricates, and it never fails a collection cycle.

Dates
-----
ReliefWeb distinguishes ``date.original`` (when the report refers to) from ``date.created``
(when it was posted). ``date.original`` is preferred, which already avoids a chunk of the
publication-date-as-event-date error. The body is still passed through
``event_time.extract_event_time`` because a situation report routinely covers an event from
days earlier.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("nepal_mhews.reliefweb")

REQUEST_TIMEOUT_SEC = 25

# ReliefWeb's own country id for Nepal. Filtering server-side on this is far cheaper and
# more reliable than matching the string "Nepal" in free text, which also catches reports
# that merely mention Nepal in passing.
NEPAL_COUNTRY_ISO3 = "npl"

HAZARD_TERMS = (
    "landslide", "landslip", "mudslide", "debris flow", "rockfall", "avalanche",
    "flood", "flash flood", "glacial lake outburst", "GLOF", "inundation",
)


def describe_capability() -> Dict[str, Any]:
    """Can this source be used right now, and if not, exactly what is missing?"""
    appname = (settings.reliefweb_appname or "").strip()
    return {
        "enabled": bool(settings.reliefweb_enabled),
        "appname_present": bool(appname),
        "api_url": settings.reliefweb_api_url,
        "usable": bool(settings.reliefweb_enabled and appname),
        "blocker": (
            None if appname else
            "No RELIEFWEB_APPNAME. v2 returns HTTP 403 without an approved appname; "
            "request one free at https://apidoc.reliefweb.int/parameters#appname"
        ),
    }


def _strip_html(raw: Optional[str]) -> str:
    if not raw:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'"))
    return re.sub(r"\s+", " ", text).strip()


def _parse_iso(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _build_query(lookback_days: int, limit: int) -> Dict[str, Any]:
    """A POST body for /v2/reports, filtered server-side to Nepal + hazard terms."""
    since = datetime.now(tz=timezone.utc).timestamp() - lookback_days * 86400.0
    since_iso = datetime.fromtimestamp(since, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return {
        "appname": (settings.reliefweb_appname or "").strip(),
        "limit": int(limit),
        "sort": ["date.created:desc"],
        "fields": {
            "include": [
                "title", "body", "url", "date.created", "date.original",
                "country.iso3", "disaster_type.name", "source.name",
            ]
        },
        "query": {
            "value": " OR ".join(f'"{t}"' for t in HAZARD_TERMS),
            "fields": ["title", "body"],
            "operator": "OR",
        },
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "country.iso3", "value": NEPAL_COUNTRY_ISO3},
                {"field": "date.created", "value": {"from": since_iso}},
            ],
        },
    }


def _post(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    url = settings.reliefweb_api_url
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "nepal-mhews/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "ignore")[:300]
        except Exception:
            pass
        if exc.code == 403:
            logger.warning(
                "ReliefWeb refused the request: the appname %r is not approved. "
                "Request one free at https://apidoc.reliefweb.int/parameters#appname. %s",
                (settings.reliefweb_appname or "")[:40], detail,
            )
        elif exc.code == 410:
            logger.error("ReliefWeb says this API version is decommissioned: %s", detail)
        else:
            logger.warning("ReliefWeb HTTP %s: %s", exc.code, detail)
    except Exception as exc:
        logger.warning("ReliefWeb request failed: %s: %s", type(exc).__name__, exc)
    return None


def fetch_reports(
    *, lookback_days: Optional[int] = None, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Recent Nepal hazard situation reports, in the collector's bulletin shape.

    Returns ``[]`` - never raises and never invents - when the source is unavailable.
    Each item carries real ``raw_text`` body prose, which is the whole point of it.
    """
    cap = describe_capability()
    if not cap["usable"]:
        logger.debug("ReliefWeb unavailable: %s", cap["blocker"])
        return []

    payload = _build_query(
        lookback_days if lookback_days is not None else settings.reliefweb_lookback_days,
        limit if limit is not None else settings.reliefweb_max_items,
    )
    body = _post(payload)
    if not body or not isinstance(body.get("data"), list):
        return []

    out: List[Dict[str, Any]] = []
    for entry in body["data"]:
        fields = entry.get("fields") or {}
        title = _strip_html(fields.get("title"))
        prose = _strip_html(fields.get("body"))
        if not title:
            continue

        dates = fields.get("date") or {}
        # date.original is what the report is ABOUT; date.created is when it was posted.
        published_utc = _parse_iso(dates.get("original")) or _parse_iso(dates.get("created"))
        if published_utc is None:
            continue

        # Now that there is real prose, read it for when the event actually happened.
        event_time: Dict[str, Any] = {}
        try:
            from app.ingestion.event_time import extract_event_time

            event_time = extract_event_time(title, prose, published_utc) or {}
        except Exception as exc:
            logger.debug("event-time extraction failed for a ReliefWeb item: %s", exc)

        sources = fields.get("source") or []
        source_name = (sources[0].get("name") if sources and isinstance(sources[0], dict)
                       else "ReliefWeb")
        out.append({
            "source": f"ReliefWeb / {source_name}",
            "title": title,
            # Trimmed: the collector stores this and some reports run to tens of kB.
            "raw_text": f"{title}. {prose}"[:20000],
            "url": fields.get("url") or "",
            "district_hint": None,
            "received_utc": time.time(),
            "published_utc": published_utc,
            "published_basis": "date.original" if dates.get("original") else "date.created",
            "event_utc": event_time.get("event_utc"),
            "event_time_confidence": event_time.get("confidence"),
            "event_time_evidence": event_time.get("evidence_phrase"),
            "event_time_method": event_time.get("resolution_method"),
            "event_time_ambiguity_days": event_time.get("ambiguity_days"),
            "disaster_types": [
                d.get("name") for d in (fields.get("disaster_type") or [])
                if isinstance(d, dict) and d.get("name")
            ],
        })
    logger.info("ReliefWeb returned %d Nepal hazard report(s) with body text", len(out))
    return out
