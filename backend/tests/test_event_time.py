"""Event-time extraction: does the article say when it actually happened?

The regression that motivates this file is :func:`test_the_october_false_alarm`. Every
other test exists to stop a fix for that case from breaking something else.

No live LLM is used anywhere here - the resolver is the part that must be deterministic,
and the model's only job (producing a phrase) is stubbed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.ingestion import event_time as et

NPT = et.NPT


def _utc(y, m, d, hh=7, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp()


def _local_date(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(NPT).date()


# Enough prose to clear MIN_BODY_CHARS_FOR_LLM. Real Google News descriptions never do -
# see test_headline_only_items_never_reach_the_model.
_PROSE = (
    "Rescue teams from the Nepal Army and Armed Police Force were deployed to the "
    "affected ward after local officials reported that several houses had been buried. "
    "District authorities said the search would continue, and that displaced families "
    "were being moved to a school building on higher ground while the road remains cut."
)


# ------------------------------------------------------------------ the regression
def test_the_october_false_alarm():
    """A real article: published 12 Oct 2026, event 11 Oct 2026.

    Scraped on 25 October, the old pipeline saw only the pubDate, called it fresh, and
    emailed an URGENT 'prediction confirmed' alert about a two-week-old event.
    """
    title = ("The 11 October 2026 landslide and debris flow/flood on the Budhi Gandaki "
             "River in Gorkha district, Nepal")
    out = et.extract_event_time(title, "", _utc(2026, 10, 12), allow_llm=False)

    assert out["confidence"] == "explicit_date"
    assert _local_date(out["event_utc"]) == datetime(2026, 10, 11).date()
    assert "11 October 2026" in out["evidence_phrase"]

    # and the staleness guard it feeds must refuse to call this fresh
    verdict = et.is_stale_for_alerting(out, max_age_days=3.0)
    assert verdict["stale"] is True
    assert verdict["weak_basis"] is False


def test_publication_date_alone_is_not_treated_as_an_event_time():
    out = et.extract_event_time(
        "Rescue teams continue search operation in Gorkha", "", _utc(2026, 10, 12),
        allow_llm=False,
    )
    assert out["confidence"] == "publication_fallback"
    assert out["event_utc"] == _utc(2026, 10, 12)
    assert out["evidence_phrase"] is None
    verdict = et.is_stale_for_alerting(out, max_age_days=3.0)
    assert verdict["weak_basis"] is True


# ------------------------------------------------------------------ real phrasings
@pytest.mark.parametrize("text,pub,expected", [
    # verbatim from the live Google News feed
    ("Raging floodwaters repeatedly battered a bridge in central Nepal on Sunday, Oct. 11, "
     "before the structure finally gave way.", _utc(2026, 10, 12), (2026, 10, 11)),
    ("Another major flash flood struck Nepal on the morning of October 11th.",
     _utc(2026, 10, 11), (2026, 10, 11)),
    ("Aerial footage released by the Nepali Army on Sunday, Oct. 11, showed a village "
     "inundated.", _utc(2026, 10, 11), (2026, 10, 11)),
    # other forms Nepali English-language press uses
    ("A landslide swept away four houses in Sindhupalchok on 2026-09-28.",
     _utc(2026, 9, 30), (2026, 9, 28)),
    ("The landslide occurred on September 28, 2026, police said.",
     _utc(2026, 10, 2), (2026, 9, 28)),
    ("Three died when a slope collapsed yesterday in Myagdi.",
     _utc(2026, 8, 14), (2026, 8, 13)),
    ("The disaster struck two days ago, locals said.",
     _utc(2026, 8, 14), (2026, 8, 12)),
])
def test_real_phrasings_resolve_to_the_right_local_day(text, pub, expected):
    out = et.extract_event_time(text, "", pub, allow_llm=False)
    assert out["confidence"] in ("explicit_date", "relative_resolved")
    assert _local_date(out["event_utc"]) == datetime(*expected).date()


def test_weekday_resolves_backwards_from_publication():
    # Thu 15 Oct 2026 publication; "on Monday" is the 12th, not a future Monday.
    pub = _utc(2026, 10, 15)
    assert datetime.fromtimestamp(pub, tz=timezone.utc).astimezone(NPT).weekday() == 3
    out = et.extract_event_time("Landslide blocked the highway on Monday", "", pub,
                                allow_llm=False)
    assert _local_date(out["event_utc"]) == datetime(2026, 10, 12).date()


def test_last_weekday_never_resolves_to_the_same_day():
    pub = _utc(2026, 10, 15)  # a Thursday
    out = et.extract_event_time("The slope failed last Thursday", "", pub, allow_llm=False)
    assert _local_date(out["event_utc"]) == datetime(2026, 10, 8).date()


# ------------------------------------------------------------------ the hard edges
def test_year_boundary_rolls_back_not_forward():
    """'December 30' in a January article is LAST December."""
    out = et.extract_event_time(
        "Rescuers recovered the last body from the December 30 landslide.",
        "", _utc(2027, 1, 4), allow_llm=False,
    )
    assert _local_date(out["event_utc"]) == datetime(2026, 12, 30).date()


def test_a_date_after_publication_is_rejected_not_reported():
    """An article cannot report an event that has not happened yet."""
    assert et.resolve_phrase("December 30", _utc(2026, 7, 1)) is not None  # prior year is fine
    r = et.resolve_phrase("December 30", _utc(2026, 7, 1))
    assert _local_date(r["event_utc"]) == datetime(2025, 12, 30).date()


def test_an_ancient_date_is_rejected_as_background_context():
    """'the 2015 earthquake' is context, not this article's event."""
    assert et.resolve_phrase("25 April 2015", _utc(2026, 10, 12)) is None


def test_bikram_sambat_is_detected_and_refused_not_guessed():
    out = et.extract_event_time("Kartik 9 ko pahiro ma tin jana ko mrityu", "",
                                _utc(2026, 10, 26), allow_llm=False)
    assert out["confidence"] == "detected_unresolvable"
    assert out["matched_rule"] == "bikram_sambat"
    assert out.get("fell_back_to_publication") is True
    assert "not attempted" in out["note"]


def test_an_impossible_calendar_date_does_not_crash():
    assert et.resolve_phrase("February 31", _utc(2026, 10, 12)) is None


# ------------------------------------------------------------------ precision
def test_a_clock_time_narrows_the_uncertainty_window():
    loose = et.extract_event_time("The landslide hit on October 11.", "",
                                  _utc(2026, 10, 12), allow_llm=False)
    tight = et.extract_event_time("The landslide hit at 3:30 am on October 11.", "",
                                  _utc(2026, 10, 12), allow_llm=False)
    assert tight["ambiguity_days"] < loose["ambiguity_days"]
    ev = datetime.fromtimestamp(tight["event_utc"], tz=timezone.utc).astimezone(NPT)
    assert (ev.hour, ev.minute) == (3, 30)


def test_a_daypart_narrows_it_too():
    out = et.extract_event_time("Flooding struck on the morning of October 11.", "",
                                _utc(2026, 10, 12), allow_llm=False)
    assert out["ambiguity_days"] < 1.0
    ev = datetime.fromtimestamp(out["event_utc"], tz=timezone.utc).astimezone(NPT)
    assert 6 <= ev.hour <= 12


def test_pm_clock_times_convert_correctly():
    out = et.extract_event_time("The slope gave way at 7 pm on October 11.", "",
                                _utc(2026, 10, 12), allow_llm=False)
    ev = datetime.fromtimestamp(out["event_utc"], tz=timezone.utc).astimezone(NPT)
    assert ev.hour == 19


def test_results_are_reported_in_nepal_local_time_not_utc():
    """NPT is UTC+5:45; resolving in UTC would shift events across the day boundary."""
    out = et.extract_event_time("A landslide at 2:00 am on October 11 killed three.", "",
                                _utc(2026, 10, 12), allow_llm=False)
    ev_utc = datetime.fromtimestamp(out["event_utc"], tz=timezone.utc)
    assert ev_utc.date() == datetime(2026, 10, 10).date()   # 02:00 NPT == 20:15 UTC prev day
    assert _local_date(out["event_utc"]) == datetime(2026, 10, 11).date()


# ------------------------------------------------------------------ the LLM boundary
def test_the_model_may_never_supply_a_timestamp():
    """The hard line: a phrase, never a resolved time."""
    payload = {"phrase": "on Sunday", "event_utc": 1760000000, "iso_date": "2026-10-11",
               "nested": {"timestamp": 123, "sentence": "ok"}}
    assert et._strip_llm_timestamps(payload) is True
    assert "event_utc" not in payload and "iso_date" not in payload
    assert "timestamp" not in payload["nested"]
    assert payload["phrase"] == "on Sunday"          # the phrase survives
    assert payload["nested"]["sentence"] == "ok"


def test_an_llm_phrase_is_resolved_by_the_same_deterministic_code(monkeypatch):
    monkeypatch.setattr(et, "_gemini_extract_phrase", lambda text: {
        "phrase": "on Sunday", "sentence": "It happened on Sunday evening.",
        "refers_to_event": True, "confidence": 0.9,
    })
    pub = _utc(2026, 10, 15)  # Thursday
    out = et.extract_event_time("Slope collapse in Gorkha", _PROSE, pub)
    assert out["extractor"] == "llm_phrase+deterministic_resolve"
    assert _local_date(out["event_utc"]) == datetime(2026, 10, 11).date()  # prior Sunday


def test_a_phrase_the_model_says_is_not_the_event_is_ignored(monkeypatch):
    monkeypatch.setattr(et, "_gemini_extract_phrase", lambda text: {
        "phrase": "on Monday", "sentence": "The minister spoke on Monday.",
        "refers_to_event": False, "confidence": 0.9,
    })
    out = et.extract_event_time("Minister visits site", _PROSE, _utc(2026, 10, 15))
    assert out["confidence"] == "publication_fallback"


def test_the_deterministic_path_wins_and_the_model_is_not_called(monkeypatch):
    def _boom(text):
        raise AssertionError("LLM called even though an explicit date was present")

    monkeypatch.setattr(et, "_gemini_extract_phrase", _boom)
    out = et.extract_event_time("The 11 October 2026 landslide in Gorkha", "",
                                _utc(2026, 10, 12))
    assert out["confidence"] == "explicit_date"


def test_it_degrades_to_regex_when_no_llm_is_reachable(monkeypatch):
    """The system runs unattended for six months; an unreachable model is normal."""
    monkeypatch.setattr(et, "_gemini_extract_phrase", lambda text: None)
    out = et.extract_event_time("Landslide struck yesterday in Myagdi", _PROSE,
                                _utc(2026, 8, 14))
    assert out["confidence"] == "relative_resolved"
    assert _local_date(out["event_utc"]) == datetime(2026, 8, 13).date()


# ------------------------------------------------------------------ staleness guard
def test_staleness_uses_the_youngest_edge_of_the_uncertainty_window():
    now = datetime.now(tz=timezone.utc).timestamp()
    recent = {"event_utc": now - 3.4 * 86400, "ambiguity_days": 1.0,
              "confidence": "explicit_date", "resolution_method": "x"}
    # 3.4 d old, +/- 0.5 d -> could be as young as 2.9 d, so a 3 d window keeps it
    assert et.is_stale_for_alerting(recent, max_age_days=3.0)["stale"] is False


def test_an_unresolvable_event_is_stale_by_default():
    assert et.is_stale_for_alerting(
        {"event_utc": None, "confidence": "detected_unresolvable"}, max_age_days=3.0
    )["stale"] is True


def test_empty_input_does_not_crash():
    out = et.extract_event_time(None, None, _utc(2026, 10, 12), allow_llm=False)
    assert out["confidence"] == "publication_fallback"
    assert out["event_utc"] == _utc(2026, 10, 12)


def test_html_markup_is_stripped_before_matching():
    out = et.extract_event_time(
        "<p>The landslide hit on <b>October 11</b>.</p>", "", _utc(2026, 10, 12),
        allow_llm=False,
    )
    assert _local_date(out["event_utc"]) == datetime(2026, 10, 11).date()



def test_headline_only_items_never_reach_the_model(monkeypatch):
    """Google News RSS ships the headline as the description and nothing else - verified
    across 100 live items, not one description exceeded its title by 60 characters. An
    LLM call per item to re-read a headline the regex already swept is pure spend."""
    def _boom(text):
        raise AssertionError("LLM called on a headline-only item")

    monkeypatch.setattr(et, "_gemini_extract_phrase", _boom)
    out = et.extract_event_time(
        "Nepal urges villagers to evacuate after flood triggered by landslide, 5 missing",
        "Nepal urges villagers to evacuate after flood triggered by landslide, 5 missing   Reuters",
        _utc(2026, 10, 12),
    )
    assert out["confidence"] == "publication_fallback"
