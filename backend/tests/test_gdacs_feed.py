"""Tests for the GDACS official multi-hazard event harvester.

Fully offline: `urllib.request.urlopen` is monkeypatched in every test that touches the
network path, so nothing here needs the live GDACS service or any credential. The RSS
fixture below reproduces the exact element names, namespaces and attribute shapes of the
live `https://www.gdacs.org/xml/rss.xml` document captured on 2026-10-24 (315 items), and
the archive fixture reproduces one real GeoJSON feature from the GDACS event-list API
(the Red Nepal flood of 2026-08-26, eventid 1104124).
"""

import io
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from app.core.database import SQLiteTimeSeriesStore
from app.ingestion import gdacs_feed
from app.ingestion.gdacs_feed import (
    HIMALAYA_BBOX,
    build_staging_key,
    fetch_gdacs_archive_events,
    fetch_gdacs_events,
    filter_events,
    load_ingest_state,
    map_alert_level,
    map_gdacs_event_type,
    normalize_archive_feature,
    parse_gdacs_rss,
    stage_gdacs_events,
)

RSS_URL = "https://www.gdacs.org/xml/rss.xml"

# Keyword arguments that keep every test fast and silent: one attempt, no backoff sleep.
FAST = {
    "enabled": True,
    "rss_url": RSS_URL,
    "timeout_s": 1.0,
    "max_retries": 1,
    "retry_backoff_s": 0.0,
}


def _rfc822(days_ago: float = 1.0) -> str:
    return format_datetime(datetime.now(timezone.utc) - timedelta(days=days_ago))


def _item(
    event_type: str = "FL",
    event_id: str = "1104124",
    lat: float = 27.2953,
    lon: float = 85.3649,
    alert: str = "Red",
    country: str = "Nepal",
    iso3: str = "NPL",
    title: str = "Red flood alert in Nepal",
    description: str = "On 26/08/2026, a flood started in Nepal.",
    days_ago: float = 1.0,
    severity_value: str = "0",
    severity_unit: str = "",
    severity_text: str = "Magnitude 0",
    with_geo_point: bool = True,
    with_georss: bool = True,
    extra: str = "",
) -> str:
    """One `<item>` in the exact shape of the live GDACS RSS feed."""
    when = _rfc822(days_ago)
    geo_point = (
        f"<geo:Point><geo:lat>{lat}</geo:lat><geo:long>{lon}</geo:long></geo:Point>"
        if with_geo_point
        else ""
    )
    georss = f"<georss:point>{lat} {lon}</georss:point>" if with_georss else ""
    return f"""
    <item>
      <title>{title}</title>
      <description>{description}</description>
      <link>http://www.gdacs.org/report.aspx?eventtype={event_type}&amp;eventid={event_id}</link>
      <pubDate>{when}</pubDate>
      <gdacs:dateadded>{when}</gdacs:dateadded>
      <gdacs:fromdate>{when}</gdacs:fromdate>
      <gdacs:todate>{when}</gdacs:todate>
      <dc:subject>{event_type}1</dc:subject>
      <guid isPermaLink="false">{event_type}{event_id}</guid>
      {geo_point}
      <gdacs:bbox>{lon - 4} {lon + 4} {lat - 4} {lat + 4}</gdacs:bbox>
      {georss}
      <gdacs:eventtype>{event_type}</gdacs:eventtype>
      <gdacs:alertlevel>{alert}</gdacs:alertlevel>
      <gdacs:alertscore>3</gdacs:alertscore>
      <gdacs:eventname />
      <gdacs:eventid>{event_id}</gdacs:eventid>
      <gdacs:episodeid>1</gdacs:episodeid>
      <gdacs:severity unit="{severity_unit}" value="{severity_value}">{severity_text}</gdacs:severity>
      <gdacs:population unit="Population Affected" value="0">0 deaths</gdacs:population>
      <gdacs:iso3>{iso3}</gdacs:iso3>
      <gdacs:country>{country}</gdacs:country>
      {extra}
    </item>"""


def _rss(*items: str) -> bytes:
    """Wrap items in the live feed's channel envelope, BOM included."""
    body = f"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:geo="http://www.w3.org/2003/01/geo/wgs84_pos#"
     xmlns:gdacs="http://www.gdacs.org"
     xmlns:glide="http://glidenumber.net"
     xmlns:georss="http://www.georss.org/georss"
     xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>GDACS RSS information</title>
    <link>http://www.gdacs.org/</link>
    <pubDate>{_rfc822(0)}</pubDate>
    {''.join(items)}
  </channel>
</rss>"""
    return b"\xef\xbb\xbf" + body.encode("utf-8")


# Real items from the 2026-10-24 snapshot, used as the "must be filtered" cases.
VANUATU_EARTHQUAKE = _item(
    event_type="EQ",
    event_id="1573536",
    lat=-14.0833,
    lon=165.6982,
    alert="Green",
    country="Vanuatu",
    iso3="VUT",
    title="Green earthquake (Magnitude 5.8M, Depth:10km) in Vanuatu",
    description="An earthquake occurred in Vanuatu.",
    severity_value="5.8",
    severity_unit="M",
    severity_text="Magnitude 5.8M, Depth:10km",
)
HIMALAYAN_WILDFIRE = _item(
    event_type="WF",
    event_id="1021001",
    lat=28.3,
    lon=84.1,
    alert="Green",
    country="Nepal",
    title="Green Forest Fire alert in Nepal",
    description="A forest fire is burning in Nepal.",
)


def _install_fake_urlopen(monkeypatch, responses, calls=None):
    """Serve a canned body per call; an entry may also be an exception to raise."""
    sequence = list(responses)

    def fake_urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if calls is not None:
            calls.append(url)
        item = sequence.pop(0) if len(sequence) > 1 else sequence[0]
        if isinstance(item, Exception):
            raise item
        body = item if isinstance(item, bytes) else str(item).encode("utf-8")
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


@pytest.fixture()
def store(tmp_path):
    return SQLiteTimeSeriesStore(db_path=tmp_path / "gdacs_test.db")


@pytest.fixture()
def empty_catalog(monkeypatch):
    """Isolate dedup tests from the production training catalog CSV."""
    monkeypatch.setattr(gdacs_feed, "load_live_ground_truth_events", lambda: [])


@pytest.fixture()
def state_file(tmp_path):
    return tmp_path / "gdacs_ingest_state.json"


# ---------------------------------------------------------------------------
# Namespace-correct parsing
# ---------------------------------------------------------------------------
def test_representative_item_parses_with_correct_namespaces(monkeypatch):
    """The real Red Nepal flood item must come out fully normalized."""
    _install_fake_urlopen(monkeypatch, [_rss(_item())])
    events = fetch_gdacs_events(since_days=None, **FAST)

    assert len(events) == 1
    event = events[0]
    assert event["latitude"] == 27.2953
    assert event["longitude"] == 85.3649
    assert event["hazard_type"] == "flood"
    assert event["source"] == "GDACS"
    assert event["external_id"] == "gdacs-FL-1104124"
    assert event["severity"] == "RED"
    assert event["country"] == "Nepal"
    assert event["source_url"].endswith("eventtype=FL&eventid=1104124")
    assert event["event_date"] == (
        datetime.now(gdacs_feed.NEPAL_TZ) - timedelta(days=1)
    ).date().isoformat()
    assert isinstance(event["event_time_utc"], float)
    # A provenance heuristic, never a certainty.
    assert 0.10 <= event["confidence"] < 1.0
    # A GloFAS flood point is a basin centroid, not a surveyed failure location.
    assert event["coordinate_precision"] == "BASIN_CENTROID"
    assert event["location_uncertainty_hint_km"] > 0


def test_latitude_and_longitude_are_not_swapped(monkeypatch):
    """A swap would file a Gorkha event in the Terai; both axes are asserted apart."""
    _install_fake_urlopen(monkeypatch, [_rss(_item(lat=28.2657, lon=83.5973))])
    event = fetch_gdacs_events(since_days=None, **FAST)[0]
    assert event["latitude"] == 28.2657
    assert event["longitude"] == 83.5973


def test_georss_point_is_used_when_geo_point_is_absent(monkeypatch):
    _install_fake_urlopen(
        monkeypatch, [_rss(_item(with_geo_point=False, lat=28.8698, lon=82.1688))]
    )
    event = fetch_gdacs_events(since_days=None, **FAST)[0]
    assert (event["latitude"], event["longitude"]) == (28.8698, 82.1688)


def test_item_without_any_coordinate_is_dropped_not_guessed(monkeypatch):
    """No coordinate means no event. A default lat/lon would poison the catalog."""
    _install_fake_urlopen(
        monkeypatch, [_rss(_item(with_geo_point=False, with_georss=False))]
    )
    assert fetch_gdacs_events(since_days=None, **FAST) == []


def test_bbox_is_never_used_as_a_coordinate_fallback():
    """gdacs:bbox is a fixed +/-4 degree box, so it must not stand in for a point."""
    events = parse_gdacs_rss(_rss(_item(with_geo_point=False, with_georss=False)))
    assert events == []


def test_severity_fields_are_carried_through(monkeypatch):
    _install_fake_urlopen(monkeypatch, [_rss(_item(lat=29.2076, lon=83.9584, event_type="EQ",
                                                   event_id="1564287", alert="Green",
                                                   severity_value="4.9", severity_unit="M",
                                                   severity_text="Magnitude 4.9M"))])
    event = fetch_gdacs_events(since_days=None, **FAST)[0]
    assert event["hazard_type"] == "earthquake"
    assert event["severity"] == "GREEN"
    assert event["severity_value"] == 4.9
    assert event["severity_unit"] == "M"
    # An epicentre is not where the slope failed.
    assert event["coordinate_precision"] == "EPICENTRE"


# ---------------------------------------------------------------------------
# Filtering: geography and hazard type
# ---------------------------------------------------------------------------
def test_events_outside_the_himalaya_window_are_filtered_out(monkeypatch):
    """The live feed is 90% Pacific wildfires and quakes; none may reach the queue."""
    _install_fake_urlopen(monkeypatch, [_rss(VANUATU_EARTHQUAKE, _item())])
    events = fetch_gdacs_events(since_days=None, **FAST)
    assert [e["external_id"] for e in events] == ["gdacs-FL-1104124"]


def test_irrelevant_hazard_types_are_filtered_out(monkeypatch):
    """A wildfire inside the window is still irrelevant to slope stability."""
    _install_fake_urlopen(monkeypatch, [_rss(HIMALAYAN_WILDFIRE, _item())])
    events = fetch_gdacs_events(since_days=None, **FAST)
    assert [e["hazard_type"] for e in events] == ["flood"]


def test_whole_feed_of_irrelevant_events_yields_nothing(monkeypatch):
    """An empty result is the normal weekly case for Nepal, not an error."""
    _install_fake_urlopen(monkeypatch, [_rss(VANUATU_EARTHQUAKE, HIMALAYAN_WILDFIRE)])
    assert fetch_gdacs_events(since_days=None, **FAST) == []


def test_since_days_excludes_stale_events(monkeypatch):
    _install_fake_urlopen(monkeypatch, [_rss(_item(days_ago=90), _item(event_id="2", days_ago=2))])
    events = fetch_gdacs_events(since_days=30, **FAST)
    assert [e["external_id"] for e in events] == ["gdacs-FL-2"]


def test_results_are_sorted_newest_first(monkeypatch):
    _install_fake_urlopen(
        monkeypatch,
        [_rss(_item(event_id="old", days_ago=9), _item(event_id="new", days_ago=1))],
    )
    events = fetch_gdacs_events(since_days=None, **FAST)
    assert [e["external_id"] for e in events] == ["gdacs-FL-new", "gdacs-FL-old"]


def test_event_type_and_alert_level_vocabulary_mapping():
    assert map_gdacs_event_type("EQ") == "earthquake"
    assert map_gdacs_event_type("FL") == "flood"
    assert map_gdacs_event_type("LS") == "landslide"
    assert map_gdacs_event_type("MS") == "landslide"
    # Tracked but irrelevant hazards are dropped, not reinterpreted from free text.
    assert map_gdacs_event_type("WF", "landslide blocked the highway") is None
    assert map_gdacs_event_type("DR") is None
    assert map_gdacs_event_type("", "") is None

    assert map_alert_level("Green") == "GREEN"
    assert map_alert_level("orange") == "ORANGE"
    assert map_alert_level("Red") == "RED"
    # Never a cheerful default for an unreadable level.
    assert map_alert_level(None) == "UNKNOWN"
    assert map_alert_level("Catastrophic") == "UNKNOWN"


def test_text_refinement_narrows_a_flood_to_glof_but_never_overrides_an_official_type():
    assert map_gdacs_event_type("FL", "Glacial lake outburst flood in Solukhumbu") == "glof"
    assert map_gdacs_event_type("LS", "debris flow buried the road") == "debris_flow"
    # An earthquake bulletin mentioning landslides keeps its epicentre semantics.
    assert map_gdacs_event_type("EQ", "earthquake triggered landslides") == "earthquake"


def test_filter_events_tolerates_malformed_records():
    assert filter_events([{}, {"latitude": None}, None], bbox=HIMALAYA_BBOX) == []


# ---------------------------------------------------------------------------
# Failure modes: never raise, never invent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"   ",
        b"<html><body>503 upstream error</body></html>",
        b"<?xml version='1.0'?><rss><channel><item><title>no namespaces</title></item></channel></rss>",
        b"<rss><channel><item>",  # truncated mid-document
    ],
)
def test_malformed_or_empty_xml_yields_zero_events_and_no_exception(monkeypatch, payload):
    _install_fake_urlopen(monkeypatch, [payload])
    assert fetch_gdacs_events(since_days=None, **FAST) == []


def test_parse_gdacs_rss_accepts_odd_input_without_raising():
    assert parse_gdacs_rss(None) == []
    assert parse_gdacs_rss(123) == []
    assert parse_gdacs_rss("") == []


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("Name or service not known"),
        urllib.error.HTTPError(RSS_URL, 503, "Service Unavailable", {}, None),
        TimeoutError("timed out"),
    ],
)
def test_dead_feed_returns_empty_list_plus_a_warning(monkeypatch, failure):
    _install_fake_urlopen(monkeypatch, [failure])
    warnings = []
    assert fetch_gdacs_events(since_days=None, warnings=warnings, **FAST) == []
    assert warnings and "GDACS fetch failed" in warnings[0]


def test_disabled_setting_makes_no_network_call(monkeypatch):
    """GDACS_ENABLED=0 must not open a socket."""

    def exploding_urlopen(*args, **kwargs):
        raise AssertionError("No request may be made while GDACS is disabled.")

    monkeypatch.setattr(urllib.request, "urlopen", exploding_urlopen)
    monkeypatch.setattr(gdacs_feed.settings, "gdacs_enabled", False, raising=False)

    assert fetch_gdacs_events(since_days=None) == []
    assert fetch_gdacs_archive_events("2026-01-01", "2026-10-24") == []
    summary = stage_gdacs_events(since_days=None)
    assert summary["fetched"] == 0 and summary["queued"] == 0


def test_staging_never_raises_even_when_the_store_explodes(monkeypatch, empty_catalog, state_file):
    _install_fake_urlopen(monkeypatch, [_rss(_item())])

    class BrokenStore:
        def get_pending_bulletins(self, limit=50):
            raise RuntimeError("database is locked")

        def insert_unprocessed_bulletin(self, bulletin):
            raise RuntimeError("disk I/O error")

    summary = stage_gdacs_events(
        store=BrokenStore(), since_days=None, state_path=state_file, **FAST
    )
    assert summary["queued"] == 0
    assert summary["errors"] == 1
    assert summary["warnings"]


# ---------------------------------------------------------------------------
# Staging queue + cross-run dedup
# ---------------------------------------------------------------------------
def test_events_are_staged_pending_and_never_written_to_the_catalog(
    monkeypatch, store, empty_catalog, state_file
):
    _install_fake_urlopen(monkeypatch, [_rss(_item())])

    def forbidden(*args, **kwargs):
        raise AssertionError("The harvester must never write the training catalog directly.")

    import app.ingestion.ground_truth_feedback as gtf

    monkeypatch.setattr(gtf, "append_ground_truth_event", forbidden)

    summary = stage_gdacs_events(
        store=store, since_days=None, state_path=state_file, **FAST
    )
    assert summary == {
        "fetched": 1,
        "queued": 1,
        "suppressed_duplicates": 0,
        "restaged_escalations": 0,
        "errors": 0,
        "warnings": [],
        "state_saved": True,
    }

    pending = store.get_pending_bulletins(limit=10)
    assert len(pending) == 1
    bulletin = pending[0]
    assert bulletin["status"] == "PENDING"
    assert bulletin["source"] == "GDACS"
    assert bulletin["bulletin_id"] == build_staging_key("gdacs-FL-1104124")
    # Machine-readable coordinates for the weekly NLP pillar...
    assert "27.2953N, 85.3649E" in bulletin["raw_text"]
    # ...and the precision caveat travels with them to the human reviewer.
    assert "BASIN_CENTROID" in bulletin["raw_text"]
    assert "NOT a measured error" in bulletin["raw_text"]


def test_dedup_suppresses_a_repeat_across_two_runs(monkeypatch, store, empty_catalog, state_file):
    _install_fake_urlopen(monkeypatch, [_rss(_item())])
    first = stage_gdacs_events(store=store, since_days=None, state_path=state_file, **FAST)
    assert first["queued"] == 1

    _install_fake_urlopen(monkeypatch, [_rss(_item())])
    second = stage_gdacs_events(store=store, since_days=None, state_path=state_file, **FAST)
    assert second["fetched"] == 1
    assert second["queued"] == 0
    assert second["suppressed_duplicates"] == 1
    assert len(store.get_pending_bulletins(limit=10)) == 1


def test_dedup_survives_a_restart_with_a_fresh_store(monkeypatch, tmp_path, empty_catalog, state_file):
    """The weekly cron restarts the process and may face a drained queue; the persisted
    state file alone must still suppress an already-ingested GDACS event id."""
    first_store = SQLiteTimeSeriesStore(db_path=tmp_path / "run1.db")
    _install_fake_urlopen(monkeypatch, [_rss(_item())])
    assert stage_gdacs_events(
        store=first_store, since_days=None, state_path=state_file, **FAST
    )["queued"] == 1
    assert state_file.exists()

    # Fresh process, fresh database: only the state file carries the memory.
    second_store = SQLiteTimeSeriesStore(db_path=tmp_path / "run2.db")
    _install_fake_urlopen(monkeypatch, [_rss(_item())])
    second = stage_gdacs_events(
        store=second_store, since_days=None, state_path=state_file, **FAST
    )
    assert second["queued"] == 0
    assert second["suppressed_duplicates"] == 1
    assert second_store.get_pending_bulletins(limit=10) == []

    persisted = load_ingest_state(state_file)
    assert "gdacs-FL-1104124" in persisted["events"]
    assert persisted["events"]["gdacs-FL-1104124"]["alert_level"] == "RED"


def test_duplicate_ids_within_one_batch_are_collapsed(monkeypatch, store, empty_catalog, state_file):
    _install_fake_urlopen(monkeypatch, [_rss(_item(), _item(title="Same event, new episode"))])
    summary = stage_gdacs_events(store=store, since_days=None, state_path=state_file, **FAST)
    assert summary["fetched"] == 2
    assert summary["queued"] == 1
    assert summary["suppressed_duplicates"] == 1


def test_an_escalating_alert_level_is_restaged_once(monkeypatch, store, empty_catalog, state_file):
    """GREEN -> RED on the same event id is materially new information."""
    _install_fake_urlopen(monkeypatch, [_rss(_item(alert="Green"))])
    assert stage_gdacs_events(
        store=store, since_days=None, state_path=state_file, **FAST
    )["queued"] == 1

    _install_fake_urlopen(monkeypatch, [_rss(_item(alert="Red"))])
    escalated = stage_gdacs_events(store=store, since_days=None, state_path=state_file, **FAST)
    assert escalated["queued"] == 1
    assert escalated["restaged_escalations"] == 1

    # A third run at the same level adds nothing further.
    _install_fake_urlopen(monkeypatch, [_rss(_item(alert="Red"))])
    steady = stage_gdacs_events(store=store, since_days=None, state_path=state_file, **FAST)
    assert steady["queued"] == 0
    assert steady["suppressed_duplicates"] == 1
    assert len(store.get_pending_bulletins(limit=10)) == 2


def test_de_escalation_is_not_restaged(monkeypatch, store, empty_catalog, state_file):
    _install_fake_urlopen(monkeypatch, [_rss(_item(alert="Red"))])
    stage_gdacs_events(store=store, since_days=None, state_path=state_file, **FAST)
    _install_fake_urlopen(monkeypatch, [_rss(_item(alert="Green"))])
    assert stage_gdacs_events(
        store=store, since_days=None, state_path=state_file, **FAST
    )["queued"] == 0


def test_an_event_already_in_the_training_catalog_is_suppressed(
    monkeypatch, store, state_file
):
    monkeypatch.setattr(
        gdacs_feed,
        "load_live_ground_truth_events",
        lambda: [{"latitude": 27.2953, "longitude": 85.3649}],
    )
    _install_fake_urlopen(monkeypatch, [_rss(_item())])
    summary = stage_gdacs_events(store=store, since_days=None, state_path=state_file, **FAST)
    assert summary["queued"] == 0
    assert summary["suppressed_duplicates"] == 1


def test_corrupt_state_file_is_tolerated(monkeypatch, store, empty_catalog, state_file):
    state_file.write_text("{not json at all", encoding="utf-8")
    _install_fake_urlopen(monkeypatch, [_rss(_item())])
    summary = stage_gdacs_events(store=store, since_days=None, state_path=state_file, **FAST)
    assert summary["queued"] == 1
    assert load_ingest_state(state_file)["events"]


def test_prefetched_events_can_be_staged_without_any_network(store, empty_catalog, state_file):
    """The archive backfill path hands `stage_gdacs_events` an already-fetched batch."""
    events = parse_gdacs_rss(_rss(_item()))
    summary = stage_gdacs_events(
        store=store, events=events, state_path=state_file
    )
    assert summary["fetched"] == 1 and summary["queued"] == 1


# ---------------------------------------------------------------------------
# Archive (GeoJSON) backfill path
# ---------------------------------------------------------------------------
ARCHIVE_FEATURE = {
    "type": "Feature",
    "bbox": [85.3649, 27.2953, 85.3649, 27.2953],
    "geometry": {"type": "Point", "coordinates": [85.3649, 27.2953]},
    "properties": {
        "eventtype": "FL",
        "eventid": 1104124,
        "episodeid": 1,
        "name": "Flood in Nepal",
        "description": "Flood in Nepal",
        "htmldescription": "Red Flood in Nepal from: 26 Aug 2026 01 to: 01 Sep 2026 14.",
        "url": {
            "report": "https://www.gdacs.org/report.aspx?eventid=1104124&episodeid=1&eventtype=FL"
        },
        "alertlevel": "Red",
        "country": "Nepal",
        "fromdate": "2026-08-26T01:00:00",
        "todate": "2026-09-01T14:00:00",
        "iso3": "NPL",
        "source": "GLOFAS",
        "polygonlabel": "Centroid",
        "Class": "Point_Centroid",
        "severitydata": {"severity": 0.0, "severitytext": "Magnitude 0 ", "severityunit": ""},
    },
}


def test_archive_feature_normalizes_and_records_its_centroid_class(monkeypatch):
    _install_fake_urlopen(
        monkeypatch, [json.dumps({"type": "FeatureCollection", "features": [ARCHIVE_FEATURE]})]
    )
    events = fetch_gdacs_archive_events(
        "2026-08-01", "2026-09-01", enabled=True, max_retries=1, retry_backoff_s=0.0
    )
    assert len(events) == 1
    event = events[0]
    assert event["external_id"] == "gdacs-FL-1104124"
    assert event["event_date"] == "2026-08-26"
    assert event["severity"] == "RED"
    assert event["feed"] == "ARCHIVE_SEARCH"
    # GDACS states the geometry class itself; it beats any guess from the event type.
    assert event["coordinate_precision"] == "BASIN_CENTROID"


def test_archive_earthquake_stays_an_epicentre_despite_the_generic_centroid_label():
    """GDACS labels every archive point 'Centroid', earthquakes included (eventid
    1564287). An EQ coordinate is still an epicentre, and must be reported as one."""
    feature = json.loads(json.dumps(ARCHIVE_FEATURE))
    feature["properties"].update(
        {
            "eventtype": "EQ",
            "eventid": 1564287,
            "alertlevel": "Green",
            "fromdate": "2026-09-08T16:08:34",
            "htmldescription": "Green M 4.9 Earthquake in Nepal at: 08 Sep 2026 16:08:34.",
            "severitydata": {
                "severity": 4.9,
                "severitytext": "Magnitude 4.9M, Depth:10km",
                "severityunit": "M",
            },
        }
    )
    event = normalize_archive_feature(feature)
    assert event["hazard_type"] == "earthquake"
    assert event["coordinate_precision"] == "EPICENTRE"
    assert event["severity_value"] == 4.9


def test_multi_country_event_is_flagged_as_a_regional_centroid():
    """A point covering several countries can be hundreds of km from any one slope."""
    feature = json.loads(json.dumps(ARCHIVE_FEATURE))
    feature["properties"]["country"] = "India, Nepal, Bangladesh"
    event = normalize_archive_feature(feature)
    assert event["coordinate_precision"] == "REGION_CENTROID"
    assert event["location_uncertainty_hint_km"] >= 100


def test_archive_non_json_and_broken_features_yield_nothing(monkeypatch):
    _install_fake_urlopen(monkeypatch, [b"<html>gateway timeout</html>"])
    warnings = []
    assert (
        fetch_gdacs_archive_events(
            "2026-08-01", "2026-09-01", enabled=True, max_retries=1,
            retry_backoff_s=0.0, warnings=warnings,
        )
        == []
    )
    assert warnings

    assert normalize_archive_feature(None) is None
    assert normalize_archive_feature({"properties": {}}) is None
    assert normalize_archive_feature({"geometry": {"coordinates": []}, "properties": {}}) is None
