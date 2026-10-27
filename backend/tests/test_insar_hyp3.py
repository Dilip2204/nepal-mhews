"""Tests for ASF HyP3 on-demand InSAR.

The governing property: a velocity may be reported as MEASURED only when a real HyP3
product was downloaded and sampled. Every other path must yield None so the caller falls
back to the clearly-labelled physics proxy. Several tests exist specifically to prove a
fabricated number cannot escape.

All offline - no credentials, no network. Rasters are written to tmp_path and read back
through real rasterio, so the pixel/geometry maths is genuinely exercised.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.ingestion import insar_hyp3 as h


REF = "S1_039571_IW1_20261023T001859_VV_33FF-BURST"
SEC = "S1_039571_IW1_20261011T001859_VV_A700-BURST"
LAT, LON = 28.22, 85.52


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Never touch real state, real products, or the network."""
    monkeypatch.setattr(h, "STATE_PATH", tmp_path / "hyp3_state.json")
    monkeypatch.setattr(h, "PRODUCT_DIR", tmp_path / "products")
    monkeypatch.setattr(h.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(h.settings, "earthdata_token", "test-token", raising=False)

    def _no_network(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(h.urllib.request, "urlopen", _no_network)
    return tmp_path


# ---------------------------------------------------------------- burst parsing
def _umm_granule(name, direction="DESCENDING", path="19", burst_id_full=None):
    """One CMR umm_json item, shaped like the live response.

    Verified against cmr.earthdata.nasa.gov on 2026-10-26: the response is
    {"items": [{"umm": {...}, "meta": {...}}]}, the granule name lives at
    umm["GranuleUR"], and AdditionalAttributes is a list of {"Name":.., "Values":[..]}.
    Passing direction=None or path=None omits that attribute entirely, which is how a
    granule that simply does not declare it is simulated.
    """
    attributes = []
    if direction is not None:
        attributes.append({"Name": "ASCENDING_DESCENDING", "Values": [direction]})
    if path is not None:
        attributes.append({"Name": "PATH_NUMBER", "Values": [path]})
    if burst_id_full is not None:
        attributes.append({"Name": "BURST_ID_FULL", "Values": [burst_id_full]})
    return {"umm": {"GranuleUR": name, "AdditionalAttributes": attributes}, "meta": {}}


def _cmr_payload(names, **kwargs):
    return {"items": [_umm_granule(n, **kwargs) for n in names]}


def test_cross_polarized_bursts_are_excluded(monkeypatch):
    """The live API rejects VH/HV outright - submitting them wastes a round trip."""
    names = [
        REF, SEC,
        "S1_039571_IW1_20261023T001859_VH_33FF-BURST",
        "S1_039571_IW1_20261011T001859_VH_A700-BURST",
    ]
    monkeypatch.setattr(h, "_http_json", lambda *a, **k: _cmr_payload(names))
    stacks = h.find_burst_stacks(LAT, LON)
    assert len(stacks) == 1
    assert stacks[0].polarization == "VV"
    assert stacks[0].depth == 2


def test_stacks_group_by_burst_swath_and_polarization(monkeypatch):
    names = [
        REF, SEC,
        "S1_039572_IW2_20261023T001900_VV_33FF-BURST",
        "S1_039572_IW2_20261011T001900_VV_A700-BURST",
        "S1_039572_IW2_20260929T001900_VV_B100-BURST",
    ]
    monkeypatch.setattr(h, "_http_json", lambda *a, **k: _cmr_payload(names))
    stacks = h.find_burst_stacks(LAT, LON)
    assert [s.depth for s in stacks] == [3, 2]  # deepest first
    assert stacks[0].burst_id == "039572"


def test_malformed_granules_are_ignored(monkeypatch):
    monkeypatch.setattr(
        h, "_http_json", lambda *a, **k: _cmr_payload(["not-a-burst", "", REF, SEC])
    )
    stacks = h.find_burst_stacks(LAT, LON)
    assert stacks and stacks[0].depth == 2


def test_cmr_failure_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(h, "_http_json", lambda *a, **k: None)
    assert h.find_burst_stacks(LAT, LON) == []


# ---------------------------------------------------------------- pair selection
def _stack(dates):
    acqs = []
    for d in dates:
        acqs.append((datetime.strptime(d, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc),
                     f"S1_039571_IW1_{d}_VV_33FF-BURST"))
    acqs.sort(key=lambda a: a[0], reverse=True)
    return h.BurstStack("039571", "IW1", "VV", acqs)


def test_pair_selection_prefers_the_shortest_usable_baseline():
    """Coherence decays with temporal baseline, badly in vegetated Himalayan terrain."""
    pair = h.select_pair(_stack(["20261023T001859", "20261011T001859", "20260801T001859"]))
    assert pair is not None
    _, secondary, days = pair
    assert "20261011" in secondary
    assert days == pytest.approx(12.0, abs=0.1)


def test_pair_selection_rejects_an_over_long_baseline():
    assert h.select_pair(_stack(["20261023T001859", "20260101T001859"])) is None


def test_pair_selection_needs_two_acquisitions():
    assert h.select_pair(_stack(["20261023T001859"])) is None


def test_before_restricts_the_reference_for_retrospective_work():
    stack = _stack(["20261023T001859", "20261011T001859", "20260929T001859"])
    cutoff = datetime(2026, 10, 15, tzinfo=timezone.utc)
    pair = h.select_pair(stack, before=cutoff)
    assert pair is not None and "20261011" in pair[0]


# ---------------------------------------------------------------- submission guards
def test_disabled_never_submits(monkeypatch):
    monkeypatch.setattr(h.settings, "hyp3_enabled", False, raising=False)
    assert h.request_measurement(LAT, LON)["status"] == "DISABLED"


def test_missing_credentials_never_submits(monkeypatch):
    monkeypatch.setattr(h.settings, "earthdata_token", "", raising=False)
    assert h.request_measurement(LAT, LON)["status"] == "NO_CREDENTIALS"


def test_malformed_granule_is_refused_before_any_request(monkeypatch):
    called = []
    monkeypatch.setattr(h, "_http_json", lambda *a, **k: called.append(1))
    assert h.submit_insar_job("garbage", SEC) is None
    assert not called


def test_budget_exhaustion_stops_submission(monkeypatch):
    monkeypatch.setattr(h.settings, "hyp3_monthly_credit_budget", 2, raising=False)
    monkeypatch.setattr(h, "remaining_credits", lambda: 8000.0)
    state = h._load_state()
    h._record_credit_spend(state, 2)
    h._save_state(state)
    assert h.request_measurement(LAT, LON)["status"] == "BUDGET_EXHAUSTED"


def test_remote_credit_exhaustion_also_stops_submission(monkeypatch):
    """HyP3's own balance overrides optimistic local accounting."""
    monkeypatch.setattr(h, "remaining_credits", lambda: 0.0)
    assert h._budget_allows(h._load_state(), 5) == 0


def test_submission_records_state_that_survives_a_restart(monkeypatch):
    monkeypatch.setattr(h, "_http_json", lambda *a, **k: _cmr_payload([REF, SEC]))
    monkeypatch.setattr(h, "remaining_credits", lambda: 8000.0)
    monkeypatch.setattr(
        h, "submit_insar_job",
        lambda *a, **k: {"job_id": "job-1", "status_code": "PENDING"},
    )
    result = h.request_measurement(LAT, LON)
    assert result["status"] == "SUBMITTED"
    # fresh read from disk == a new process
    reloaded = h._load_state()
    assert reloaded["jobs"][h.site_key(LAT, LON)]["job_id"] == "job-1"
    assert h.credits_spent_this_period(reloaded) == h.JOB_CREDIT_COST


def test_a_pending_job_is_not_resubmitted(monkeypatch):
    state = h._load_state()
    state["jobs"][h.site_key(LAT, LON)] = {"job_id": "job-1", "status_code": "RUNNING"}
    h._save_state(state)
    assert h.request_measurement(LAT, LON)["status"] == "ALREADY_PENDING"


# ---------------------------------------------------------------- measurement maths
def _write_product(tmp_path, phase_value, coherence=0.8, size=41, nodata=None, conncomp=1):
    """A HyP3-shaped product: unwrapped phase + coherence + conncomp, as the real one ships.

    Band names mirror a genuine INSAR_ISCE_BURST product inspected on 2026-10-26
    (*_unw_phase.tif, *_corr.tif, *_conncomp.tif).
    """
    d = tmp_path / "prod"
    d.mkdir(exist_ok=True)
    transform = from_origin(LON - 0.02, LAT + 0.02, 0.001, 0.001)
    prof = dict(driver="GTiff", height=size, width=size, count=1, dtype="float32",
                crs="EPSG:4326", transform=transform)
    phase = np.full((size, size), phase_value, dtype="float32")
    with rasterio.open(d / "S1_x_unw_phase.tif", "w", nodata=nodata, **prof) as dst:
        dst.write(phase, 1)
    coh = np.full((size, size), coherence, dtype="float32")
    with rasterio.open(d / "S1_x_corr.tif", "w", **prof) as dst:
        dst.write(coh, 1)
    cc = np.full((size, size), conncomp, dtype="float32")
    with rasterio.open(d / "S1_x_conncomp.tif", "w", **prof) as dst:
        dst.write(cc, 1)
    return d


def test_failed_unwrapping_connected_component_yields_no_measurement(isolate):
    """conncomp == 0 is ISCE's authoritative "could not unwrap" flag.

    Measured on the real Langtang product: both langtang_lirung and the 28.22/85.52 test
    point had coherence 0.000 and conncomp 0 - genuinely unmeasurable terrain. Reporting a
    velocity there would be inventing one.
    """
    d = _write_product(isolate, 2.5, coherence=0.9, conncomp=0)
    assert h.measure_los_velocity_from_product(d, LAT, LON, days_apart=12.0) is None


def test_velocity_matches_the_phase_to_displacement_formula(isolate):
    """d = -(lambda/4pi) * dphi, scaled to mm per 30 days."""
    phase = 2.0
    d = _write_product(isolate, phase)
    m = h.measure_los_velocity_from_product(d, LAT, LON, days_apart=12.0)
    assert m is not None
    # HyP3's raster is already sign-corrected: positive phase = away from sensor.
    expected_m = (h.SENTINEL1_WAVELENGTH_M / (4 * math.pi)) * phase
    assert m["los_displacement_mm"] == pytest.approx(expected_m * 1000.0, abs=1e-3)
    assert m["los_velocity_mm_month"] == pytest.approx(
        expected_m * 1000.0 * (30.0 / 12.0), abs=1e-3
    )


def test_sign_follows_hyp3_positive_phase_is_motion_away_from_the_sensor(isolate):
    """Pinned to HyP3's documented raster convention, not the textbook phase formula.

    The product README states: "Positive values indicate movement away from the sensor...
    negative values indicate movement towards the sensor." HyP3 has already applied the
    sign flip, so negating again would invert every velocity - downslope creep would be
    reported as uplift, and the number would look perfectly plausible.
    """
    away = h.measure_los_velocity_from_product(
        _write_product(isolate, 3.0), LAT, LON, days_apart=12.0
    )
    assert away["los_velocity_mm_month"] > 0, "positive phase must mean AWAY from sensor"
    toward = h.measure_los_velocity_from_product(
        _write_product(isolate, -3.0), LAT, LON, days_apart=12.0
    )
    assert toward["los_velocity_mm_month"] < 0


def test_low_coherence_yields_no_measurement_not_a_number(isolate):
    d = _write_product(isolate, 2.0, coherence=0.05)
    assert h.measure_los_velocity_from_product(d, LAT, LON, days_apart=12.0) is None


def test_failed_unwrapping_zeros_are_not_read_as_a_stable_slope(isolate):
    """HyP3 writes exact 0.0 where unwrapping failed; that is no-data, not 'no motion'."""
    d = _write_product(isolate, 0.0)
    assert h.measure_los_velocity_from_product(d, LAT, LON, days_apart=12.0) is None


def test_point_outside_the_interferogram_yields_none(isolate):
    d = _write_product(isolate, 2.0)
    assert h.measure_los_velocity_from_product(d, 10.0, 10.0, days_apart=12.0) is None


def test_missing_product_file_yields_none(isolate):
    empty = isolate / "empty"
    empty.mkdir()
    assert h.measure_los_velocity_from_product(empty, LAT, LON, days_apart=12.0) is None


def test_zero_days_apart_cannot_divide_into_a_velocity(isolate):
    d = _write_product(isolate, 2.0)
    assert h.measure_los_velocity_from_product(d, LAT, LON, days_apart=0.0) is None


# ---------------------------------------------------------------- provider contract
def test_provider_returns_none_without_a_measurement():
    """No measurement on file must mean no number - the core anti-fabrication property."""
    assert h.hyp3_measured_velocity_provider(LAT, LON) is None


def test_provider_returns_none_when_disabled(monkeypatch):
    state = h._load_state()
    state["baselines"][h.site_key(LAT, LON)] = {
        "los_velocity_mm_month": 12.0, "job_id": "j", "measured_utc": time.time()
    }
    h._save_state(state)
    monkeypatch.setattr(h.settings, "hyp3_enabled", False, raising=False)
    assert h.hyp3_measured_velocity_provider(LAT, LON) is None


def test_provider_refuses_a_stale_baseline(monkeypatch):
    monkeypatch.setattr(h.settings, "hyp3_baseline_max_age_days", 30, raising=False)
    state = h._load_state()
    state["baselines"][h.site_key(LAT, LON)] = {
        "los_velocity_mm_month": 12.0,
        "job_id": "j",
        "measured_utc": time.time() - 60 * 86400,
    }
    h._save_state(state)
    assert h.hyp3_measured_velocity_provider(LAT, LON) is None


def test_provider_surfaces_a_real_measurement_with_its_product_reference():
    state = h._load_state()
    state["baselines"][h.site_key(LAT, LON)] = {
        "los_velocity_mm_month": -7.5,
        "job_id": "abc-123",
        "reference_granule": REF,
        "secondary_granule": SEC,
        "days_apart": 12.0,
        "mean_coherence": 0.71,
        "pixels_valid": 90,
        "measured_utc": time.time(),
    }
    h._save_state(state)
    out = h.hyp3_measured_velocity_provider(LAT, LON)
    assert out is not None
    assert out["los_velocity_mm_month"] == -7.5
    assert "abc-123" in out["product_reference"]
    assert REF in out["product_reference"]


def test_provider_satisfies_the_insar_cdse_validator(monkeypatch):
    """The real cross-module contract, exercised for real rather than assumed."""
    from app.ingestion import insar_cdse

    state = h._load_state()
    state["baselines"][h.site_key(LAT, LON)] = {
        "los_velocity_mm_month": -7.5,
        "job_id": "abc-123",
        "reference_granule": REF,
        "secondary_granule": SEC,
        "days_apart": 12.0,
        "measured_utc": time.time(),
    }
    h._save_state(state)

    insar_cdse.register_measured_velocity_provider(h.hyp3_measured_velocity_provider)
    try:
        accepted = insar_cdse._invoke_measured_provider(LAT, LON)
        assert accepted is not None, "insar_cdse rejected a genuine measurement"
    finally:
        insar_cdse.register_measured_velocity_provider(None)


def test_an_empty_provider_result_is_rejected_by_insar_cdse():
    """Symmetry check: the validator must refuse a provider that supplies nothing."""
    from app.ingestion import insar_cdse

    insar_cdse.register_measured_velocity_provider(lambda lat, lon: None)
    try:
        assert insar_cdse._invoke_measured_provider(LAT, LON) is None
    finally:
        insar_cdse.register_measured_velocity_provider(None)


def test_describe_capability_reports_honestly(monkeypatch):
    cap = h.describe_capability()
    assert cap["job_type"] == "INSAR_ISCE_BURST"
    assert cap["credit_cost_per_job"] == 1
    assert cap["credentials_present"] is True


# ------------------------------------------------- flight direction from CMR metadata
"""Flight direction exists so ascending and descending geometries can be combined.

Two look vectors constrain two components of motion; one look vector twice over
constrains nothing extra while looking like it does. So the only acceptable source for
the direction is the authoritative per-granule attribute, and the only acceptable answer
when that attribute is missing is "UNKNOWN".

The granule names and attribute values below are taken from a live CMR query at glc_105
(29.5023, 82.2504) on 2026-10-26: 337397/IW1 ASCENDING path 158, 196362/IW3 DESCENDING
path 92, 118306/IW3 ASCENDING path 56.
"""

ASC = "S1_337397_IW1_20261020T001859_VV_33FF-BURST"
ASC_OLDER = "S1_337397_IW1_20261008T001859_VV_A701-BURST"
DESC = "S1_196362_IW3_20261016T123833_VV_8EF7-BURST"
DESC_OLDER = "S1_196362_IW3_20261004T123833_VV_B102-BURST"


def _items(*granules):
    return {"items": list(granules)}


def _stub_http(payload, *, hits=None, search_after=None, calls=None):
    """Stand in for _http_json, optionally populating the CMR response headers.

    A stub that ignores headers_out models the realistic failure of not knowing the hit
    count at all - which must read as "total unknown", never as "nothing was truncated".
    """
    def _stub(url, *, headers=None, timeout=None, headers_out=None, **kwargs):
        if calls is not None:
            calls.append({"url": url, "headers": headers})
        if headers_out is not None:
            if hits is not None:
                headers_out["CMR-Hits"] = str(hits)
            if search_after is not None:
                headers_out["CMR-Search-After"] = search_after
        return payload
    return _stub


def test_umm_json_parsing_yields_direction_and_path_per_stack(monkeypatch):
    """Both geometries are kept and labelled, not collapsed onto the deepest stack."""
    payload = _items(
        _umm_granule(ASC, direction="ASCENDING", path="158"),
        _umm_granule(ASC_OLDER, direction="ASCENDING", path="158"),
        _umm_granule(DESC, direction="DESCENDING", path="92"),
        _umm_granule(DESC_OLDER, direction="DESCENDING", path="92"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    stacks = h.find_burst_stacks(29.5023, 82.2504)

    by_burst = {s.burst_id: s for s in stacks}
    assert set(by_burst) == {"337397", "196362"}
    assert by_burst["337397"].flight_direction == "ASCENDING"
    assert by_burst["337397"].path_number == "158"
    assert by_burst["196362"].flight_direction == "DESCENDING"
    assert by_burst["196362"].path_number == "92"
    # Both directions are available to a combiner - the whole point of the change.
    assert {s.flight_direction for s in stacks} == {"ASCENDING", "DESCENDING"}


def test_missing_direction_attribute_is_unknown_and_never_guessed(monkeypatch):
    """No ASCENDING_DESCENDING means UNKNOWN - not inferred from orbit, time or geometry.

    DESC is a real descending granule; with the attribute stripped the only honest answer
    is UNKNOWN, so a test that merely checked "not ASCENDING" would pass on a lucky guess.
    """
    payload = _items(
        _umm_granule(DESC, direction=None, path="92"),
        _umm_granule(DESC_OLDER, direction=None, path="92"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    stack = h.find_burst_stacks(29.5023, 82.2504)[0]
    assert stack.flight_direction == "UNKNOWN"
    assert stack.flight_direction not in ("ASCENDING", "DESCENDING")
    # The rest of the metadata still parses; only the absent field is unknown.
    assert stack.path_number == "92"
    assert stack.depth == 2


def test_an_unrecognised_direction_value_is_unknown_not_passed_through(monkeypatch):
    payload = _items(_umm_granule(DESC, direction="SIDEWAYS", path="92"))
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    assert h.find_burst_stacks(29.5023, 82.2504)[0].flight_direction == "UNKNOWN"


def test_conflicting_directions_in_one_stack_are_refused(monkeypatch, caplog):
    """Impossible by construction (burst ids are track-bound) - so it must never be voted on.

    Picking the majority would hand the combiner two "independent" look vectors that are
    really one, manufacturing agreement between a measurement and itself.
    """
    payload = _items(
        _umm_granule(DESC, direction="DESCENDING", path="92"),
        _umm_granule(DESC_OLDER, direction="ASCENDING", path="92"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    with caplog.at_level("WARNING", logger="nepal_mhews.insar_hyp3"):
        stack = h.find_burst_stacks(29.5023, 82.2504)[0]
    assert stack.flight_direction == "UNKNOWN"
    assert any("conflicting flight direction" in r.message.lower() for r in caplog.records)


def test_cross_polarized_granules_cannot_contribute_direction(monkeypatch):
    """VH is dropped before metadata is read, so it cannot poison a VV stack's direction."""
    payload = _items(
        _umm_granule(ASC, direction="ASCENDING", path="158"),
        _umm_granule(ASC_OLDER, direction="ASCENDING", path="158"),
        _umm_granule("S1_337397_IW1_20261020T001859_VH_33FF-BURST",
                     direction="DESCENDING", path="999"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    stacks = h.find_burst_stacks(29.5023, 82.2504)
    assert len(stacks) == 1
    assert stacks[0].polarization == "VV"
    assert stacks[0].flight_direction == "ASCENDING"
    assert stacks[0].path_number == "158"


# ------------------------------------------------- track number adjudication
def test_a_single_mis_stamped_path_is_settled_by_burst_id_full(monkeypatch):
    """A measured CMR anomaly, reproduced exactly rather than imagined.

    Live at glc_105 on 2026-10-26: 339 of 340 granules of burst 196362/IW3 report
    PATH_NUMBER 92 and one 2026-05-30 granule reports 167, while all 340 agree on
    BURST_ID_FULL '092_196362_IW3'. BURST_ID_FULL embeds the track, so reading it is
    parsing a second authoritative field - not taking a majority vote.
    """
    payload = _items(
        _umm_granule(DESC, direction="DESCENDING", path="92",
                     burst_id_full="092_196362_IW3"),
        _umm_granule(DESC_OLDER, direction="DESCENDING", path="167",
                     burst_id_full="092_196362_IW3"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    stack = h.find_burst_stacks(29.5023, 82.2504)[0]
    assert stack.path_number == "92"
    # The mis-stamp is confined to the track; direction is unaffected.
    assert stack.flight_direction == "DESCENDING"


def test_irreconcilable_track_metadata_leaves_path_unset(monkeypatch):
    """When even BURST_ID_FULL is split there is no authoritative track, so: None."""
    payload = _items(
        _umm_granule(DESC, direction="DESCENDING", path="92",
                     burst_id_full="092_196362_IW3"),
        _umm_granule(DESC_OLDER, direction="DESCENDING", path="167",
                     burst_id_full="167_196362_IW3"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    stack = h.find_burst_stacks(29.5023, 82.2504)[0]
    assert stack.path_number is None
    assert stack.flight_direction == "DESCENDING"


def test_absent_path_attribute_leaves_path_unset(monkeypatch):
    payload = _items(_umm_granule(DESC, direction="DESCENDING", path=None))
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    assert h.find_burst_stacks(29.5023, 82.2504)[0].path_number is None


def test_track_is_parsed_from_burst_id_full_without_zero_padding():
    assert h._track_from_burst_id_full("092_196362_IW3") == "92"
    assert h._track_from_burst_id_full("158_337397_IW1") == "158"
    assert h._track_from_burst_id_full("garbage") is None
    assert h._track_from_burst_id_full(None) is None


# ------------------------------------------------- page-size truncation
def test_hitting_the_page_limit_mid_result_is_flagged_and_logged(monkeypatch, caplog):
    """Depth must be reported as a lower bound, not as the whole archive.

    Truncation is the structural fact "we stopped while CMR was still handing over
    granules", so the stub keeps offering a CMR-Search-After token until the page budget
    is spent - exactly how Langtang behaves at max_pages=2 (1973 + 1222 granules, more
    still queued).
    """
    payload = _items(
        _umm_granule(ASC, direction="ASCENDING", path="158"),
        _umm_granule(ASC_OLDER, direction="ASCENDING", path="158"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload, hits=3231, search_after="tok"))
    with caplog.at_level("INFO", logger="nepal_mhews.insar_hyp3"):
        stack = h.find_burst_stacks(29.5023, 82.2504, max_pages=2)[0]
    assert stack.truncated is True
    # The measured numbers have to reach the log, not just the boolean - an operator needs
    # to see what was collected against what CMR claimed before judging the cut.
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "3231" in message and "2-page limit" in message


def test_a_result_read_to_exhaustion_is_not_flagged_as_truncated(monkeypatch):
    """No continuation token means CMR has nothing more - that is the only 'complete' signal."""
    payload = _items(
        _umm_granule(ASC, direction="ASCENDING", path="158"),
        _umm_granule(ASC_OLDER, direction="ASCENDING", path="158"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload, hits=2))
    assert h.find_burst_stacks(29.5023, 82.2504)[0].truncated is False


def test_an_over_counted_hit_header_does_not_fake_a_truncation(monkeypatch):
    """CMR-Hits over-counts, so it must not drive the flag.

    Measured at glc_105: a complete walk yields 1871 granules against a reported 1897,
    and the next page is empty. Believing the header would mark every site in the network
    truncated and the flag would stop carrying information.
    """
    payload = _items(_umm_granule(ASC, direction="ASCENDING", path="158"))
    monkeypatch.setattr(h, "_http_json", _stub_http(payload, hits=1897))  # no token
    assert h.find_burst_stacks(29.5023, 82.2504)[0].truncated is False


def test_an_empty_follow_up_page_ends_paging_without_flagging_truncation(monkeypatch):
    """glc_105 exactly: page 1 is short of the hit count, page 2 is empty. Complete."""
    pages = [[_umm_granule(ASC, direction="ASCENDING", path="158")], []]
    calls = []

    def _stub(url, *, headers=None, timeout=None, headers_out=None, **kwargs):
        index = len(calls)
        calls.append(headers)
        if headers_out is not None:
            headers_out["CMR-Hits"] = "1897"
            headers_out["CMR-Search-After"] = f"token-{index}"  # CMR sends one even at the end
        return {"items": pages[index]}

    monkeypatch.setattr(h, "_http_json", _stub)
    stack = h.find_burst_stacks(29.5023, 82.2504, max_pages=2)[0]
    assert len(calls) == 2
    assert stack.truncated is False


def test_overlapping_pages_cannot_inflate_stack_depth(monkeypatch):
    """Depth is what the planner's redundancy budget is spent against; it must not double."""
    payload = _items(
        _umm_granule(ASC, direction="ASCENDING", path="158"),
        _umm_granule(ASC_OLDER, direction="ASCENDING", path="158"),
    )
    # Same two granules returned on every page, with a token that never clears.
    monkeypatch.setattr(h, "_http_json", _stub_http(payload, search_after="tok"))
    stack = h.find_burst_stacks(29.5023, 82.2504, max_pages=3)[0]
    assert stack.depth == 2


def test_the_default_page_size_is_cmrs_documented_maximum(monkeypatch):
    """200 sampled ~9% of glc_105's 1897 granules across three competing stacks."""
    calls = []
    monkeypatch.setattr(
        h, "_http_json",
        _stub_http(_items(_umm_granule(ASC, direction="ASCENDING")), calls=calls),
    )
    h.find_burst_stacks(29.5023, 82.2504)
    assert h.CMR_MAX_PAGE_SIZE == 2000
    assert "page_size=2000" in calls[0]["url"]


def test_pagination_follows_cmr_search_after_when_asked(monkeypatch):
    """max_pages > 1 lifts the ceiling instead of silently dropping acquisitions.

    Measured live at Langtang: 3231 hits, 1973 granules on page 1 and 1222 on page 2.
    """
    pages = [
        [_umm_granule(ASC, direction="ASCENDING", path="158")],
        [_umm_granule(ASC_OLDER, direction="ASCENDING", path="158")],
    ]
    seen = []

    def _stub(url, *, headers=None, timeout=None, headers_out=None, **kwargs):
        index = len(seen)
        seen.append(headers)
        if headers_out is not None:
            headers_out["CMR-Hits"] = "2"
            if index < len(pages) - 1:
                headers_out["CMR-Search-After"] = f"token-{index}"
        return {"items": pages[index]} if index < len(pages) else {"items": []}

    monkeypatch.setattr(h, "_http_json", _stub)
    stack = h.find_burst_stacks(29.5023, 82.2504, max_pages=2)[0]
    assert stack.depth == 2, "second page was not merged into the stack"
    assert stack.truncated is False, "both pages were collected, so nothing was cut"
    assert seen[0] is None
    assert seen[1] == {"CMR-Search-After": "token-0"}


def test_the_default_budget_is_two_pages(monkeypatch):
    """One page cannot answer "is this complete?", because CMR always sends a token.

    The second request costs ~0.3 s when the result was already complete (glc_105: an
    empty page 2, 1.38 s -> 1.66 s) and is the only thing that makes `truncated` mean
    anything.
    """
    calls = []
    monkeypatch.setattr(
        h, "_http_json",
        _stub_http(_items(_umm_granule(ASC)), hits=1897, search_after="tok", calls=calls),
    )
    h.find_burst_stacks(29.5023, 82.2504)
    assert len(calls) == 2
    assert calls[1]["headers"] == {"CMR-Search-After": "tok"}


def test_paging_stops_early_when_cmr_offers_no_continuation(monkeypatch):
    """The common case: one page exhausts the result set, so the budget is not spent."""
    calls = []
    monkeypatch.setattr(
        h, "_http_json", _stub_http(_items(_umm_granule(ASC)), hits=1, calls=calls)
    )
    h.find_burst_stacks(29.5023, 82.2504, max_pages=5)
    assert len(calls) == 1


# ------------------------------------------------- endpoint and header plumbing
def test_the_umm_json_endpoint_is_queried_not_the_legacy_json_one(monkeypatch):
    """granules.json carries no AdditionalAttributes at all - direction is absent from it."""
    calls = []
    monkeypatch.setattr(
        h, "_http_json", _stub_http(_items(_umm_granule(ASC)), calls=calls)
    )
    h.find_burst_stacks(29.5023, 82.2504)
    assert calls[0]["url"].split("?")[0].endswith("granules.umm_json")


def test_umm_json_url_is_derived_from_whatever_format_settings_name(monkeypatch):
    for configured in (
        "https://cmr.earthdata.nasa.gov/search/granules.json",
        "https://cmr.earthdata.nasa.gov/search/granules.umm_json",
        "https://cmr.earthdata.nasa.gov/search/granules",
    ):
        monkeypatch.setattr(h.settings, "cmr_granule_search_url", configured, raising=False)
        assert h._cmr_umm_json_url().endswith("/search/granules.umm_json")


def test_http_json_exposes_response_headers_for_the_hit_count(monkeypatch):
    """The truncation check depends on this out-parameter actually being filled."""
    class _Resp:
        headers = {"CMR-Hits": "1897", "Content-Type": "application/json"}

        def read(self):
            return b'{"items": []}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(h.urllib.request, "urlopen", lambda *a, **k: _Resp())
    out = {}
    assert h._http_json("https://example.invalid/x", headers_out=out) == {"items": []}
    assert out["CMR-Hits"] == "1897"


def test_response_header_lookup_is_case_insensitive(monkeypatch):
    """HTTP header names are case-insensitive; CMR's casing is not a contract.

    Paging would silently stop after one page if the continuation token were missed.
    """
    calls = []

    def _stub(url, *, headers=None, timeout=None, headers_out=None, **kwargs):
        calls.append(headers)
        if headers_out is not None:
            headers_out["cmr-hits"] = "1897"
            headers_out["cmr-search-after"] = "tok"
        return _items(_umm_granule(ASC))

    monkeypatch.setattr(h, "_http_json", _stub)
    stack = h.find_burst_stacks(29.5023, 82.2504, max_pages=2)[0]
    assert calls[1] == {"CMR-Search-After": "tok"}
    assert stack.truncated is True


# ------------------------------------------------- unchanged contract
def test_grouping_sorting_and_newest_first_order_are_unchanged(monkeypatch):
    """Deepest stack first, acquisitions newest first - exactly as before umm_json."""
    payload = _items(
        _umm_granule(DESC, direction="DESCENDING", path="92"),
        _umm_granule(DESC_OLDER, direction="DESCENDING", path="92"),
        _umm_granule("S1_196362_IW3_20260922T123833_VV_C303-BURST",
                     direction="DESCENDING", path="92"),
        _umm_granule(ASC, direction="ASCENDING", path="158"),
        _umm_granule(ASC_OLDER, direction="ASCENDING", path="158"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    stacks = h.find_burst_stacks(29.5023, 82.2504)
    assert [s.depth for s in stacks] == [3, 2]
    assert stacks[0].burst_id == "196362"
    times = [t for t, _ in stacks[0].acquisitions]
    assert times == sorted(times, reverse=True)


def test_a_malformed_umm_item_cannot_crash_discovery(monkeypatch):
    payload = {"items": [
        {"umm": {"GranuleUR": "not-a-burst"}},
        {"umm": {}},
        {"umm": "not-even-an-object"},
        {"no_umm_at_all": True},
        "garbage",
        {"umm": {"GranuleUR": ASC, "AdditionalAttributes": [{"Name": "PATH_NUMBER"}]}},
    ]}
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    stacks = h.find_burst_stacks(29.5023, 82.2504)
    assert len(stacks) == 1
    assert stacks[0].flight_direction == "UNKNOWN"
    assert stacks[0].path_number is None


def test_select_pair_is_indifferent_to_the_new_fields():
    """The pair chooser's contract did not change; direction only labels the stack."""
    stack = h.BurstStack(
        "337397", "IW1", "VV",
        [(datetime(2026, 10, 20, tzinfo=timezone.utc), ASC),
         (datetime(2026, 10, 8, tzinfo=timezone.utc), ASC_OLDER)],
        flight_direction="ASCENDING", path_number="158",
    )
    pair = h.select_pair(stack)
    assert pair is not None and pair[0] == ASC and pair[2] == pytest.approx(12.0, abs=0.1)


def test_burst_stack_defaults_keep_positional_construction_working():
    """Existing callers build BurstStack with four positional arguments."""
    stack = h.BurstStack("039571", "IW1", "VV", [])
    assert stack.flight_direction == "UNKNOWN"
    assert stack.path_number is None
    assert stack.truncated is False


# ------------------------------------------------- backfill helper
def test_burst_direction_backfills_an_existing_networks_geometry(monkeypatch):
    """The one solved production network (glc_105, 337397/IW1/VV) stores no direction.

    Ascending/descending combination has to know which geometry that solution already
    covers, and re-solving it to find out would spend its whole credit budget again.
    """
    payload = _items(
        _umm_granule(ASC, direction="ASCENDING", path="158"),
        _umm_granule(DESC, direction="DESCENDING", path="92"),
    )
    monkeypatch.setattr(h, "_http_json", _stub_http(payload))
    assert h.burst_direction("337397", "IW1", "VV", 29.5023, 82.2504) == "ASCENDING"
    assert h.burst_direction("196362", "IW3", "VV", 29.5023, 82.2504) == "DESCENDING"


def test_burst_direction_is_unknown_when_cmr_is_unreachable(monkeypatch):
    monkeypatch.setattr(h, "_http_json", lambda *a, **k: None)
    assert h.burst_direction("337397", "IW1", "VV", 29.5023, 82.2504) == "UNKNOWN"


def test_burst_direction_is_unknown_for_a_stack_that_is_not_there(monkeypatch):
    monkeypatch.setattr(
        h, "_http_json", _stub_http(_items(_umm_granule(ASC, direction="ASCENDING")))
    )
    assert h.burst_direction("999999", "IW2", "VV", 29.5023, 82.2504) == "UNKNOWN"


def test_burst_direction_swallows_an_unexpected_failure(monkeypatch):
    """A backfill must degrade to UNKNOWN, never crash the caller into a retry loop."""
    def _boom(*a, **k):
        raise RuntimeError("cmr exploded")

    monkeypatch.setattr(h, "find_burst_stacks", _boom)
    assert h.burst_direction("337397", "IW1", "VV", 29.5023, 82.2504) == "UNKNOWN"
