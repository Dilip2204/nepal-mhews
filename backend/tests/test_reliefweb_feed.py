"""ReliefWeb v2 client: the source that supplies real article prose.

The API cannot be exercised live until an approved appname is granted (v2 returns 403
without one), so the HTTP layer is stubbed here. What is actually tested is everything
around it: the unavailable-without-appname contract, the date-field preference, HTML
stripping, and that prose reaches the event-time extractor.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.config import settings
from app.ingestion import reliefweb_feed as rw


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    # No test here may reach the network. The event-time extractor calls Gemini when it
    # sees prose with no parseable date, which one of these fixtures deliberately has.
    from app.ingestion import event_time
    monkeypatch.setattr(event_time, "_gemini_extract_phrase", lambda text: None)

    before = (settings.reliefweb_appname, settings.reliefweb_enabled)
    yield
    settings.reliefweb_appname, settings.reliefweb_enabled = before


def _report(title, body, original=None, created="2026-10-21T06:00:00+00:00"):
    date = {"created": created}
    if original:
        date["original"] = original
    return {"fields": {"title": title, "body": body, "date": date,
                       "url": "https://reliefweb.int/node/1",
                       "source": [{"name": "OCHA"}],
                       "disaster_type": [{"name": "Flash Flood"}]}}


# ------------------------------------------------------------------ availability
def test_without_an_appname_the_source_is_unavailable_not_broken():
    settings.reliefweb_appname = ""
    cap = rw.describe_capability()
    assert cap["usable"] is False
    assert "appname" in cap["blocker"]
    assert rw.fetch_reports() == []          # returns nothing, raises nothing


def test_disabled_means_disabled_even_with_an_appname():
    settings.reliefweb_appname = "nepal-mhews"
    settings.reliefweb_enabled = False
    assert rw.describe_capability()["usable"] is False
    assert rw.fetch_reports() == []


def test_a_403_is_reported_and_yields_nothing(monkeypatch, caplog):
    """The expected failure before registration - it must be loud but not fatal."""
    import urllib.error

    settings.reliefweb_appname = "unapproved"
    settings.reliefweb_enabled = True

    def _raise(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(rw.urllib.request, "urlopen", _raise)
    with caplog.at_level("WARNING"):
        assert rw.fetch_reports() == []
    assert "not approved" in caplog.text


# ------------------------------------------------------------------ parsing
def _stub(monkeypatch, reports):
    settings.reliefweb_appname = "nepal-mhews"
    settings.reliefweb_enabled = True
    monkeypatch.setattr(rw, "_post", lambda payload: {"data": reports})


def test_body_prose_survives_into_raw_text(monkeypatch):
    _stub(monkeypatch, [_report(
        "Nepal: Gorkha Flash Flood",
        "<p>A landslide dam breached on <b>11 October 2026</b>, sending debris down the "
        "Budhi Gandaki. District authorities evacuated several wards.</p>",
    )])
    items = rw.fetch_reports()
    assert len(items) == 1
    assert "<p>" not in items[0]["raw_text"]
    assert "landslide dam breached" in items[0]["raw_text"]


def test_date_original_is_preferred_over_date_created(monkeypatch):
    """date.original is what the report is ABOUT; date.created is when it was posted."""
    _stub(monkeypatch, [_report("Report", "body text",
                                original="2026-10-12T00:00:00+00:00",
                                created="2026-10-21T06:00:00+00:00")])
    item = rw.fetch_reports()[0]
    assert item["published_basis"] == "date.original"
    assert datetime.fromtimestamp(item["published_utc"], tz=timezone.utc).date() \
        == datetime(2026, 10, 12).date()


def test_date_created_is_the_documented_fallback(monkeypatch):
    _stub(monkeypatch, [_report("Report", "body", created="2026-10-21T06:00:00+00:00")])
    item = rw.fetch_reports()[0]
    assert item["published_basis"] == "date.created"


def test_an_item_with_no_usable_date_is_dropped_not_guessed(monkeypatch):
    _stub(monkeypatch, [{"fields": {"title": "No date", "body": "x", "date": {}}}])
    assert rw.fetch_reports() == []


def test_prose_reaches_the_event_time_extractor(monkeypatch):
    """The entire reason this source exists: text for the extractor to read."""
    _stub(monkeypatch, [_report(
        "Nepal: Gorkha-Budhi Gandaki Flash Flood - Humanitarian Snapshot",
        "Rescue operations continued through the week. The flash flood occurred on "
        "11 October 2026 after a landslide dam upstream breached, and district "
        "authorities have since relocated affected families to higher ground.",
        original="2026-10-21T06:00:00+00:00",
    )])
    item = rw.fetch_reports()[0]
    assert item["event_time_confidence"] == "explicit_date"
    assert "11 October 2026" in item["event_time_evidence"]
    # a 10-day gap between the report and the event it describes, now visible
    assert datetime.fromtimestamp(item["event_utc"], tz=timezone.utc).date() \
        == datetime(2026, 10, 11).date()


def test_the_query_filters_to_nepal_server_side():
    settings.reliefweb_appname = "nepal-mhews"
    q = rw._build_query(30, 40)
    conditions = q["filter"]["conditions"]
    assert {"field": "country.iso3", "value": "npl"} in conditions
    assert q["appname"] == "nepal-mhews"
    assert "body" in q["fields"]["include"]


def test_oversized_reports_are_trimmed(monkeypatch):
    _stub(monkeypatch, [_report("Big report", "word " * 20000)])
    assert len(rw.fetch_reports()[0]["raw_text"]) <= 20000


def test_a_malformed_response_yields_nothing(monkeypatch):
    settings.reliefweb_appname = "nepal-mhews"
    settings.reliefweb_enabled = True
    monkeypatch.setattr(rw, "_post", lambda payload: {"unexpected": True})
    assert rw.fetch_reports() == []
