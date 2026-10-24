"""Tests for the NDRRMA BIPAD portal incident harvester.

Fully offline: `urllib.request.urlopen` is monkeypatched in every test that exercises the
network path, so no test needs the live portal (which is unreachable from this host) or
any credential. Payload fixtures follow the schema verified against real BIPAD responses
captured on 2026-10-13; see the module docstring of `app.ingestion.bipad_scraper`.
"""

import io
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import pytest

from app.core.database import SQLiteTimeSeriesStore
from app.ingestion import bipad_scraper
from app.ingestion.bipad_scraper import (
    NEPAL_TZ,
    build_staging_key,
    fetch_recent_incidents,
    map_bipad_hazard,
    normalize_incident,
    stage_bipad_incidents,
)

API_URL = "https://bipadportal.gov.np/api/v1/incident/"

# Keyword arguments that keep every test fast and silent: no sleeping between pages or
# between retries.
FAST = {
    "enabled": True,
    "api_url": API_URL,
    "page_delay_s": 0.0,
    "retry_backoff_s": 0.0,
    "max_retries": 1,
    "timeout_s": 1.0,
}


def _today_npt(offset_days: int = 0) -> str:
    return (datetime.now(NEPAL_TZ) + timedelta(days=offset_days)).date().isoformat()


def _incident(
    incident_id: int = 95019,
    hazard: int = 17,
    title: str = "Landslide at Bhumikasthan Municipality-7",
    lon: float = 83.0423,
    lat: float = 27.9439,
    days_ago: int = 1,
    **overrides,
):
    """Build one record in the verified BIPAD response shape."""
    record = {
        "id": incident_id,
        "title": title,
        "titleNe": None,
        "wards": [2164],
        "point": {"type": "Point", "coordinates": [lon, lat]},
        "polygon": None,
        "createdOn": f"{_today_npt(-days_ago)}T10:32:17.764099+05:45",
        "modifiedOn": f"{_today_npt(-days_ago)}T15:53:36.332145+05:45",
        "incidentOn": f"{_today_npt(-days_ago)}T00:00:00+05:45",
        "reportedOn": f"{_today_npt(-days_ago)}T10:20:05+05:45",
        "description": None,
        "detail": None,
        "cause": None,
        "streetAddress": "Panchabuddha",
        "verified": True,
        "approved": True,
        "verificationMessage": None,
        "needFollowup": False,
        "region": None,
        "regionId": None,
        "dataSource": "drr_api",
        "dataSourceId": 60500,
        "source": "nepal_police",
        "event": None,
        "hazard": hazard,
        "loss": 211946,
        "createdBy": None,
        "updatedBy": None,
    }
    record.update(overrides)
    return record


def _page(results, next_url=None):
    return {"count": 9223372036854775807, "next": next_url, "previous": None, "results": results}


def _install_fake_urlopen(monkeypatch, pages_by_call, calls=None):
    """Serve a canned JSON payload per call; `pages_by_call` may also be an exception."""
    sequence = list(pages_by_call)

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if calls is not None:
            calls.append(url)
        item = sequence.pop(0) if len(sequence) > 1 else sequence[0]
        if isinstance(item, Exception):
            raise item
        body = json.dumps(item).encode("utf-8") if not isinstance(item, (bytes, str)) else item
        if isinstance(body, str):
            body = body.encode("utf-8")
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------
def test_disabled_harvester_makes_no_request_and_returns_empty(monkeypatch):
    """Shipping default is BIPAD_ENABLED=0: no socket is opened, no events are produced."""
    calls = []

    def exploding_urlopen(req, timeout=None):
        calls.append(req)
        raise AssertionError("A disabled harvester must never touch the network.")

    monkeypatch.setattr(urllib.request, "urlopen", exploding_urlopen)
    monkeypatch.setattr(bipad_scraper.settings, "bipad_enabled", False, raising=False)

    assert fetch_recent_incidents(since_days=7) == []
    assert calls == []


def test_disabled_staging_reports_zero_counts(monkeypatch):
    monkeypatch.setattr(bipad_scraper.settings, "bipad_enabled", False, raising=False)
    summary = stage_bipad_incidents(store=None, since_days=7)
    assert summary == {"fetched": 0, "queued": 0, "suppressed_duplicates": 0, "errors": 0}


# ---------------------------------------------------------------------------
# A dead portal must degrade, never raise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("[Errno -2] Name or service not known"),
        urllib.error.URLError("[Errno 111] Connection refused"),
        TimeoutError("timed out"),
        urllib.error.HTTPError(API_URL, 503, "Service Unavailable", {}, None),
    ],
)
def test_network_failure_degrades_to_empty_list(monkeypatch, failure):
    """DNS failure, refused connection, timeout and HTTP 5xx all yield no events."""
    _install_fake_urlopen(monkeypatch, [failure])
    assert fetch_recent_incidents(since_days=7, **FAST) == []


def test_non_json_response_yields_zero_events(monkeypatch):
    _install_fake_urlopen(monkeypatch, [b"<html>503 upstream error</html>"])
    assert fetch_recent_incidents(since_days=7, **FAST) == []


# ---------------------------------------------------------------------------
# Normalization of a realistic payload
# ---------------------------------------------------------------------------
def test_mocked_payload_normalizes_correctly(monkeypatch):
    _install_fake_urlopen(monkeypatch, [_page([_incident()])])

    events = fetch_recent_incidents(since_days=7, **FAST)
    assert len(events) == 1
    ev = events[0]

    for key in (
        "latitude",
        "longitude",
        "event_date",
        "hazard_type",
        "district",
        "source",
        "external_id",
        "description",
        "confidence",
    ):
        assert key in ev

    assert ev["latitude"] == pytest.approx(27.9439)
    assert ev["longitude"] == pytest.approx(83.0423)
    assert ev["event_date"] == _today_npt(-1)
    assert ev["hazard_type"] == "landslide"
    assert ev["source"] == "BIPAD_NDRRMA"
    assert ev["external_id"] == "bipad-95019"
    assert "Landslide" in ev["description"]
    assert 0.0 < ev["confidence"] <= 0.90


def test_latitude_and_longitude_are_not_swapped(monkeypatch):
    """BIPAD GeoJSON is [lon, lat]; reading it in order would put Nepal in the ocean."""
    _install_fake_urlopen(monkeypatch, [_page([_incident(lon=85.0880, lat=27.4427)])])
    ev = fetch_recent_incidents(since_days=7, **FAST)[0]
    assert 26.2 <= ev["latitude"] <= 30.6
    assert 80.0 <= ev["longitude"] <= 88.3
    assert ev["latitude"] == pytest.approx(27.4427)


def test_district_resolved_from_devanagari_title(monkeypatch):
    record = _incident(titleNe="बागमती, सिन्धुपाल्चोक, भोटेकोशी-२ मा पहिरो")
    _install_fake_urlopen(monkeypatch, [_page([record])])
    assert fetch_recent_incidents(since_days=7, **FAST)[0]["district"] == "Sindhupalchok"


def test_unverified_record_scores_lower_confidence(monkeypatch):
    _install_fake_urlopen(
        monkeypatch,
        [_page([_incident(verified=True, approved=True), _incident(incident_id=95020, lon=84.1, lat=28.1, verified=False, approved=False, source="other")])],
    )
    events = {e["external_id"]: e for e in fetch_recent_incidents(since_days=7, **FAST)}
    assert events["bipad-95019"]["confidence"] > events["bipad-95020"]["confidence"]


def test_events_outside_the_time_window_are_excluded(monkeypatch):
    _install_fake_urlopen(
        monkeypatch,
        [
            _page(
                [
                    _incident(incident_id=1, days_ago=2),
                    _incident(incident_id=2, lon=84.1, lat=28.1, days_ago=400),
                    _incident(incident_id=3, lon=84.2, lat=28.2, days_ago=-30),
                ]
            )
        ],
    )
    ids = {e["external_id"] for e in fetch_recent_incidents(since_days=7, **FAST)}
    assert ids == {"bipad-1"}


# ---------------------------------------------------------------------------
# Hazard vocabulary mapping
# ---------------------------------------------------------------------------
def test_hazard_type_mapping_from_bipad_ids():
    assert map_bipad_hazard(17, "Landslide at Bakaiya Rural Municipality-11") == "landslide"
    assert map_bipad_hazard(3, "Avalanche at Khumbupasanglahmu Rural Municipality-3") == "rock_ice_avalanche"
    assert map_bipad_hazard(11, "Flood at Rajapur Municipality-4") == "flood"
    # Hazards the portal tracks that are not slope/flood processes are dropped.
    assert map_bipad_hazard(10, "Fire at Ram Gaau") is None
    assert map_bipad_hazard(20, "Snake Bite at Barahathawa-6") is None
    assert map_bipad_hazard(23, "Thunderbolt at Chandragiri-3") is None


def test_hazard_text_refinement_narrows_but_never_overrides():
    # Free text narrows a generic landslide into a mechanism the project models.
    assert map_bipad_hazard(17, "Landslide: glacial lake outburst scoured the valley") == "glof"
    assert map_bipad_hazard(17, "Landslide with large debris flow down the gully") == "debris_flow"
    assert map_bipad_hazard(11, "Flood caused by moraine dam outburst flood") == "glof"
    # Regression: the district name "Okhaldhunga" in Devanagari contains the stone token,
    # and must not turn an officially filed Fire into a rockfall.
    assert map_bipad_hazard(10, "Fire at Molung Rural Municipality-1 | कोशी, ओखलढुङ्गा, मोलुङ-१ मा आगलागी") is None
    # An id absent from the empirical table falls back to text, then to nothing.
    assert map_bipad_hazard(999, "Landslide blocked the Araniko highway") == "landslide"
    assert map_bipad_hazard(999, "Cattle shed collapsed") is None
    assert map_bipad_hazard(None, "") is None


# ---------------------------------------------------------------------------
# Malformed / unexpected payloads produce nothing, never garbage rows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": None},
        {"results": "not-a-list"},
        {"detail": "Authentication credentials were not provided."},
        [],
        "a bare string",
        123,
        {"results": [None, 42, "x", [], {}]},
    ],
)
def test_malformed_payload_yields_zero_events(monkeypatch, payload):
    _install_fake_urlopen(monkeypatch, [payload])
    assert fetch_recent_incidents(since_days=7, **FAST) == []


@pytest.mark.parametrize(
    "broken",
    [
        {"point": None},
        {"point": {"type": "Point"}},
        {"point": {"type": "Point", "coordinates": []}},
        {"point": {"type": "Point", "coordinates": ["x", "y"]}},
        # Outside the Nepal envelope (these are Berlin coordinates).
        {"point": {"type": "Point", "coordinates": [13.4, 52.5]}},
        # Latitude/longitude order accidentally swapped by the upstream writer.
        {"point": {"type": "Point", "coordinates": [27.9439, 83.0423]}},
        {"incidentOn": None, "reportedOn": None, "createdOn": None},
        {"incidentOn": "not-a-date", "reportedOn": "", "createdOn": None},
        {"id": None},
    ],
)
def test_unusable_record_is_dropped_rather_than_guessed(broken):
    assert normalize_incident(_incident(**broken)) is None


def test_normalize_incident_tolerates_non_dict_input():
    for junk in (None, 42, "x", [], ()):
        assert normalize_incident(junk) is None


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
def test_pagination_follows_next_until_exhausted(monkeypatch):
    calls = []
    page2 = f"{API_URL}?limit=1000&offset=1000"
    _install_fake_urlopen(
        monkeypatch,
        [
            _page([_incident(incident_id=1)], next_url=page2),
            _page([_incident(incident_id=2, lon=84.1, lat=28.1)], next_url=None),
        ],
        calls=calls,
    )
    ids = {e["external_id"] for e in fetch_recent_incidents(since_days=7, **FAST)}
    assert ids == {"bipad-1", "bipad-2"}
    assert len(calls) == 2


def test_max_pages_guard_stops_a_runaway_feed(monkeypatch):
    calls = []
    _install_fake_urlopen(
        monkeypatch,
        [_page([_incident(incident_id=1)], next_url=f"{API_URL}?offset={n}") for n in range(1, 12)],
        calls=calls,
    )
    kwargs = dict(FAST, max_pages=3)
    fetch_recent_incidents(since_days=7, **kwargs)
    assert len(calls) == 3


def test_pagination_link_to_another_host_is_rejected(monkeypatch):
    calls = []
    _install_fake_urlopen(
        monkeypatch,
        [
            _page([_incident(incident_id=1)], next_url="https://evil.example.com/api/v1/incident/"),
            _page([_incident(incident_id=666, lon=84.1, lat=28.1)]),
        ],
        calls=calls,
    )
    ids = {e["external_id"] for e in fetch_recent_incidents(since_days=7, **FAST)}
    assert ids == {"bipad-1"}
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Staging queue + dedup
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path):
    return SQLiteTimeSeriesStore(db_path=tmp_path / "bipad_test.db")


@pytest.fixture()
def empty_catalog(monkeypatch):
    """Isolate dedup tests from the production training catalog CSV."""
    monkeypatch.setattr(bipad_scraper, "load_live_ground_truth_events", lambda: [])


def test_incidents_are_staged_pending_and_never_written_to_the_catalog(
    monkeypatch, store, empty_catalog
):
    _install_fake_urlopen(monkeypatch, [_page([_incident()])])

    def forbidden(*args, **kwargs):
        raise AssertionError("The harvester must never write the training catalog directly.")

    monkeypatch.setattr(bipad_scraper, "load_live_ground_truth_events", lambda: [])
    import app.ingestion.ground_truth_feedback as gtf

    monkeypatch.setattr(gtf, "append_ground_truth_event", forbidden)

    summary = stage_bipad_incidents(store=store, since_days=7, **FAST)
    assert summary["fetched"] == 1
    assert summary["queued"] == 1
    assert summary["errors"] == 0

    pending = store.get_pending_bulletins(limit=10)
    assert len(pending) == 1
    bulletin = pending[0]
    assert bulletin["status"] == "PENDING"
    assert bulletin["source"] == "BIPAD_NDRRMA"
    # The staged text carries machine-readable coordinates for the weekly NLP pillar.
    assert "27.9439N, 83.0423E" in bulletin["raw_text"]


def test_dedup_suppresses_a_repeat_of_the_same_lat_lon_date(monkeypatch, store, empty_catalog):
    _install_fake_urlopen(monkeypatch, [_page([_incident()])])
    first = stage_bipad_incidents(store=store, since_days=7, **FAST)
    assert first["queued"] == 1

    # Re-running the weekly harvest must not create a second queue entry.
    _install_fake_urlopen(monkeypatch, [_page([_incident()])])
    second = stage_bipad_incidents(store=store, since_days=7, **FAST)
    assert second["fetched"] == 1
    assert second["queued"] == 0
    assert second["suppressed_duplicates"] == 1
    assert len(store.get_pending_bulletins(limit=10)) == 1


def test_dedup_collapses_two_incident_ids_sharing_a_ward_centroid(
    monkeypatch, store, empty_catalog
):
    """Two BIPAD rows at one administrative centroid on one day are a single label."""
    _install_fake_urlopen(
        monkeypatch,
        [_page([_incident(incident_id=1), _incident(incident_id=2, title="Landslide at the same ward")])],
    )
    summary = stage_bipad_incidents(store=store, since_days=7, **FAST)
    assert summary["fetched"] == 2
    assert summary["queued"] == 1
    assert summary["suppressed_duplicates"] == 1


def test_dedup_against_the_existing_training_catalog(monkeypatch, store):
    """A coordinate already labelled in the catalog is not re-queued."""
    monkeypatch.setattr(
        bipad_scraper,
        "load_live_ground_truth_events",
        lambda: [{"latitude": 27.9439, "longitude": 83.0423}],
    )
    _install_fake_urlopen(monkeypatch, [_page([_incident()])])
    summary = stage_bipad_incidents(store=store, since_days=7, **FAST)
    assert summary["queued"] == 0
    assert summary["suppressed_duplicates"] == 1


def test_distinct_events_are_both_staged(monkeypatch, store, empty_catalog):
    _install_fake_urlopen(
        monkeypatch,
        [
            _page(
                [
                    _incident(incident_id=1),
                    _incident(incident_id=2, lon=85.515, lat=28.215, title="Landslide at Kyangjin"),
                ]
            )
        ],
    )
    summary = stage_bipad_incidents(store=store, since_days=7, **FAST)
    assert summary["queued"] == 2
    assert len(store.get_pending_bulletins(limit=10)) == 2


def test_staging_key_is_stable_and_coordinate_scoped():
    key = build_staging_key(27.9439, 83.0423, "2026-10-20")
    assert key == build_staging_key(27.9439, 83.0423, "2026-10-20")
    assert key != build_staging_key(27.9439, 83.0423, "2026-10-21")
    assert key != build_staging_key(27.9440, 83.0423, "2026-10-20")
    assert key.startswith("bip_")


def test_repeated_coordinates_are_flagged_as_suspected_centroids(monkeypatch):
    """Identical coordinates on different dates signal an administrative centroid."""
    _install_fake_urlopen(
        monkeypatch,
        [_page([_incident(incident_id=1, days_ago=1), _incident(incident_id=2, days_ago=3)])],
    )
    events = fetch_recent_incidents(since_days=7, **FAST)
    assert len(events) == 2
    assert all(e["coordinate_precision"] == "ADMIN_CENTROID_SUSPECTED" for e in events)
    assert all(e["confidence"] <= 0.70 for e in events)


def test_staging_degrades_quietly_when_the_portal_is_down(monkeypatch, store, empty_catalog):
    _install_fake_urlopen(monkeypatch, [urllib.error.URLError("Name or service not known")])
    summary = stage_bipad_incidents(store=store, since_days=7, **FAST)
    assert summary == {"fetched": 0, "queued": 0, "suppressed_duplicates": 0, "errors": 0}
    assert store.get_pending_bulletins(limit=10) == []
