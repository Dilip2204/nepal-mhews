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
def _cmr_payload(names):
    return {"feed": {"entry": [{"producer_granule_id": n} for n in names]}}


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
