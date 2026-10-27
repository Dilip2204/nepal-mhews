"""Tests for SBAS-style InSAR time-series inversion.

The centrepiece is a ground-truth recovery test: a synthetic network is built from a KNOWN
velocity plus realistic per-epoch atmospheric noise, and the inversion must recover that
velocity. The same test also demonstrates why the feature exists at all, by showing that a
single pair from the identical data is wrong by an order of magnitude.

Offline, deterministic (fixed seed), no network.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.ingestion import insar_timeseries as ts
from app.ingestion import insar_hyp3 as hyp3


T0 = datetime(2025, 10, 1, tzinfo=timezone.utc)
LAT, LON = 28.13, 85.44


def _build_network(
    n_epochs=30, connections=3, velocity_mm_year=20.0, aps_sigma_mm=8.0, seed=7,
    unwrap_error_at=None,
):
    """Synthesize a small-baseline network from a known velocity + per-epoch atmosphere.

    Atmospheric delay is modelled per-acquisition because that is how it behaves: every
    interferogram touching a given date inherits that date's atmospheric state.
    """
    rng = np.random.default_rng(seed)
    epochs = [T0 + timedelta(days=12 * i) for i in range(n_epochs)]
    aps = rng.normal(0.0, aps_sigma_mm, size=n_epochs)

    def truth(i):
        days = (epochs[i] - epochs[0]).total_seconds() / 86400.0
        return velocity_mm_year * days / 365.25

    observations = []
    for i in range(n_epochs):
        for j in range(i + 1, min(i + 1 + connections, n_epochs)):
            value = (truth(j) + aps[j]) - (truth(i) + aps[i])
            if unwrap_error_at is not None and len(observations) == unwrap_error_at:
                # a 2-pi unwrapping jump, the classic corrupting outlier
                value += (hyp3.SENTINEL1_WAVELENGTH_M / 2.0) * 1000.0
            observations.append(
                {"t_early": epochs[i], "t_late": epochs[j], "displacement_mm": value}
            )
    return epochs, observations


# ------------------------------------------------------------------ the main claim
def test_inversion_recovers_a_known_velocity_through_atmospheric_noise():
    """The whole justification for stacking, stated as an assertion."""
    _, obs = _build_network(velocity_mm_year=20.0, aps_sigma_mm=8.0)
    result = ts.invert_timeseries(obs)
    assert result is not None
    assert result["velocity_mm_year"] == pytest.approx(20.0, abs=6.0)
    # the reported uncertainty must be honest about the remaining spread
    assert result["velocity_stderr_mm_year"] > 0
    assert abs(result["velocity_mm_year"] - 20.0) < 3 * result["velocity_stderr_mm_year"]


def test_a_single_pair_from_the_same_data_is_wildly_wrong():
    """Why a single-pair number must never be presented as a velocity.

    With 8 mm per-epoch atmosphere, one 12-day pair extrapolates its noise by ~30x to reach
    an annual rate. The time series from the same data lands near the truth.
    """
    _, obs = _build_network(velocity_mm_year=20.0, aps_sigma_mm=8.0)
    first = obs[0]
    days = (first["t_late"] - first["t_early"]).total_seconds() / 86400.0
    single_pair_velocity = first["displacement_mm"] * 365.25 / days

    series = ts.invert_timeseries(obs)
    assert series is not None
    single_err = abs(single_pair_velocity - 20.0)
    series_err = abs(series["velocity_mm_year"] - 20.0)
    assert single_err > 50.0, "the single-pair estimate should be badly contaminated"
    assert series_err < single_err / 5, "stacking must be substantially better"


def test_zero_velocity_is_recovered_as_approximately_zero():
    """A stable slope must not acquire a velocity out of noise."""
    _, obs = _build_network(velocity_mm_year=0.0, aps_sigma_mm=8.0, seed=11)
    result = ts.invert_timeseries(obs)
    assert result is not None
    assert abs(result["velocity_mm_year"]) < 3 * result["velocity_stderr_mm_year"]


# ------------------------------------------------------------------ refusal paths
def test_too_few_pairs_yields_no_velocity():
    _, obs = _build_network(n_epochs=4, connections=1)
    assert ts.invert_timeseries(obs) is None


def test_a_bare_chain_has_no_redundancy_and_is_refused():
    """connections=1 gives exactly one observation per unknown - nothing averages out."""
    _, obs = _build_network(n_epochs=20, connections=1)
    assert ts.invert_timeseries(obs) is None


def test_unconstrained_epochs_are_excluded_never_minimum_normed():
    """The safety invariant behind the rank check.

    lstsq would invent minimum-norm values for epochs no observation constrains. Those
    epochs must be EXCLUDED from the solution (and counted), not silently assigned a
    number. The solved epoch count must match the connected block actually used.
    """
    _, left = _build_network(n_epochs=10, connections=3, seed=3)
    epochs_right = [T0 + timedelta(days=900 + 12 * i) for i in range(10)]
    right = [
        {"t_early": epochs_right[i], "t_late": epochs_right[j], "displacement_mm": 1.0}
        for i in range(10) for j in range(i + 1, min(i + 4, 10))
    ]
    result = ts.invert_timeseries(left + right)
    assert result is not None
    # exactly one of the two blocks is solved; the other is dropped, not fabricated
    assert result["n_epochs"] == 10
    assert result["epochs_dropped_disconnected"] == 10
    assert len(result["displacements_mm"]) == result["n_epochs"]


def test_an_unwrapping_error_is_detectable_in_the_residual():
    """A 2-pi jump must show up as network self-inconsistency, not vanish silently.

    Measured: a clean network residuals at ~0.00 mm; injecting one 2-pi jump into 84 pairs
    lifts it to ~2.3 mm and shifts the velocity by only ~3 mm/yr. That partial absorption
    is exactly what redundancy is for - so the honest assertion is that the residual rises
    detectably and a strict gate refuses it, NOT that a lone jump destroys the solution.
    """
    _, clean = _build_network(aps_sigma_mm=1.0)
    _, broken = _build_network(aps_sigma_mm=1.0, unwrap_error_at=5)

    clean_fit = ts.invert_timeseries(clean, max_residual_rms_mm=1e9)
    broken_fit = ts.invert_timeseries(broken, max_residual_rms_mm=1e9)
    assert broken_fit["residual_rms_mm"] > clean_fit["residual_rms_mm"] + 1.0

    # a gate tightened to the clean network's noise floor rejects the corrupted one
    assert ts.invert_timeseries(broken, max_residual_rms_mm=1.0) is None
    assert ts.invert_timeseries(clean, max_residual_rms_mm=1.0) is not None


def test_consistent_network_passes_the_same_residual_gate():
    """Control for the test above: without the jump, the identical gate is satisfied."""
    _, obs = _build_network(aps_sigma_mm=1.0)
    assert ts.invert_timeseries(obs, max_residual_rms_mm=3.0) is not None


def test_empty_observations_yield_none():
    assert ts.invert_timeseries([]) is None


def test_observations_missing_a_displacement_are_dropped():
    _, obs = _build_network()
    obs[0]["displacement_mm"] = None
    result = ts.invert_timeseries(obs)
    assert result is not None
    assert result["n_pairs"] == len(obs) - 1


# ------------------------------------------------------------------ double differencing
def _write_interferogram(tmp_path, target_phase, ref_phase, offset=0.0, size=171, res_deg=0.0008):
    """A synthetic interferogram: a central target patch on a different phase level.

    `offset` simulates SNAPHU's arbitrary per-product datum, which double differencing
    must cancel.
    """
    d = tmp_path / f"ifg_{abs(hash((target_phase, ref_phase, offset))) % 99999}"
    d.mkdir(exist_ok=True)
    transform = from_origin(LON - size * res_deg / 2, LAT + size * res_deg / 2, res_deg, res_deg)
    prof = dict(driver="GTiff", height=size, width=size, count=1, dtype="float32",
                crs="EPSG:4326", transform=transform)
    phase = np.full((size, size), ref_phase + offset, dtype="float32")
    c = size // 2
    # The target block must be at least as wide as the code's sampling patch
    # (PATCH_PX, 11 px), or the median samples mostly reference-level pixels.
    hw = hyp3.PATCH_PX // 2 + 2
    phase[c - hw:c + hw + 1, c - hw:c + hw + 1] = target_phase + offset
    with rasterio.open(d / "x_unw_phase.tif", "w", **prof) as dst:
        dst.write(phase, 1)
    with rasterio.open(d / "x_corr.tif", "w", **prof) as dst:
        dst.write(np.full((size, size), 0.8, dtype="float32"), 1)
    with rasterio.open(d / "x_conncomp.tif", "w", **prof) as dst:
        dst.write(np.ones((size, size), dtype="float32"), 1)
    return d


def test_double_difference_cancels_the_arbitrary_product_datum(tmp_path):
    """HyP3 says SNAPHU picks its own reference point per product.

    Two interferograms with identical real signal but different arbitrary offsets must
    produce the SAME double-differenced observation. Without this, stacking sums unrelated
    constants into a confident, meaningless number.
    """
    a = ts.measure_double_difference(
        _write_interferogram(tmp_path, 1.5, 0.5, offset=0.0), LAT, LON
    )
    b = ts.measure_double_difference(
        _write_interferogram(tmp_path, 1.5, 0.5, offset=37.0), LAT, LON
    )
    assert a is not None and b is not None
    assert a["displacement_mm"] == pytest.approx(b["displacement_mm"], abs=1e-6)


def test_double_difference_sign_follows_the_target_minus_reference(tmp_path):
    # Reference phase must be non-zero: exact 0.0 is HyP3's no-data sentinel and is
    # (correctly) masked out, which would leave no reference annulus at all.
    d = _write_interferogram(tmp_path, 2.5, 0.5)
    out = ts.measure_double_difference(d, LAT, LON)
    assert out is not None
    expected_mm = hyp3._phase_to_displacement_m(2.0) * 1000.0
    assert out["displacement_mm"] == pytest.approx(expected_mm, abs=1e-4)
    assert out["displacement_mm"] > 0  # positive phase => away from sensor


def test_double_difference_without_a_usable_reference_annulus_yields_none(tmp_path):
    """No reference means the arbitrary datum cannot be cancelled - so, no observation."""
    d = _write_interferogram(tmp_path, 1.5, 0.5)
    # demand an annulus far outside the raster
    out = ts.measure_double_difference(d, LAT, LON, inner_m=50_000, outer_m=90_000)
    assert out is None


def test_double_difference_rejects_an_incoherent_target(tmp_path):
    d = _write_interferogram(tmp_path, 1.5, 0.5)
    with rasterio.open(d / "x_corr.tif") as src:
        prof = src.profile
    with rasterio.open(d / "x_corr.tif", "w", **prof) as dst:
        dst.write(np.full((prof["height"], prof["width"]), 0.02, dtype="float32"), 1)
    assert ts.measure_double_difference(d, LAT, LON) is None


# ------------------------------------------------------------------ network planning
@pytest.fixture(autouse=True)
def _asf_resolves_everything(monkeypatch):
    """plan_network now checks burst availability against ASF before planning. No test
    may reach the network, so by default every granule resolves; the tests that exercise
    the check override this."""
    monkeypatch.setattr(hyp3, "resolve_granules", lambda names, **k: set(names))



def _stack(n, start=T0, step_days=12):
    acqs = [
        (start + timedelta(days=step_days * i),
         f"S1_039571_IW1_{(start + timedelta(days=step_days*i)).strftime('%Y%m%dT%H%M%S')}_VV_33FF-BURST")
        for i in range(n)
    ]
    acqs.sort(key=lambda a: a[0], reverse=True)
    return hyp3.BurstStack("039571", "IW1", "VV", acqs)


def test_plan_network_builds_a_redundant_network(monkeypatch):
    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [_stack(20)])
    plan = ts.plan_network(LAT, LON, since_days=None)
    assert plan is not None
    assert plan.redundancy > ts.MIN_REDUNDANCY
    # every pair must run earlier -> later, which the inversion's sign depends on
    assert all(t_a < t_b for _, _, t_a, t_b in plan.pairs)


def test_plan_network_respects_the_baseline_limit(monkeypatch):
    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [_stack(20)])
    plan = ts.plan_network(LAT, LON, max_baseline_days=24, since_days=None)
    assert plan is not None
    assert all((t_b - t_a).days <= 24 for _, _, t_a, t_b in plan.pairs)


def test_plan_network_caps_total_pairs_preferring_short_baselines(monkeypatch):
    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [_stack(30)])
    plan = ts.plan_network(LAT, LON, max_pairs=10, since_days=None)
    assert plan is not None and len(plan.pairs) == 10
    assert max((t_b - t_a).days for _, _, t_a, t_b in plan.pairs) <= 24


def test_plan_network_needs_a_stack(monkeypatch):
    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [])
    assert ts.plan_network(LAT, LON) is None


def test_plan_network_needs_enough_epochs(monkeypatch):
    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [_stack(3)])
    assert ts.plan_network(LAT, LON, since_days=None) is None


def test_summarize_is_honest_about_nothing():
    assert "no time-series velocity" in ts.summarize(None)


def test_summarize_reports_uncertainty_and_network_size():
    _, obs = _build_network()
    line = ts.summarize(ts.invert_timeseries(obs))
    assert "mm/yr" in line and "pairs" in line and "residual RMS" in line


# ------------------------------------------------------------------ connectivity recovery
def test_largest_connected_component_is_solved_when_decorrelation_splits_the_network():
    """Real behaviour from glc_105: 36 of 90 pairs decorrelated and split the graph.

    The whole set has unconstrained epochs, so inverting it would let lstsq invent
    minimum-norm values for them. Solving the largest fully-connected block instead
    recovers a valid velocity and reports how many epochs were dropped.
    """
    _, left = _build_network(n_epochs=14, connections=3, velocity_mm_year=20.0,
                             aps_sigma_mm=1.0, seed=5)
    # a second block, far later in time, with no edge joining it to the first
    far = [T0 + timedelta(days=900 + 12 * i) for i in range(8)]
    right = [
        {"t_early": far[i], "t_late": far[j], "displacement_mm": 0.0}
        for i in range(8) for j in range(i + 1, min(i + 4, 8))
    ]
    result = ts.invert_timeseries(left + right)
    assert result is not None, "the connected majority should still be solvable"
    assert result["epochs_dropped_disconnected"] == 8
    assert result["velocity_mm_year"] == pytest.approx(20.0, abs=6.0)


def test_a_fully_connected_network_drops_nothing():
    _, obs = _build_network(aps_sigma_mm=1.0)
    result = ts.invert_timeseries(obs)
    assert result is not None
    assert result["epochs_dropped_disconnected"] == 0


# ============================================================================
# Enhancements: atmospheric correction, reference stability, slope-parallel
# projection, incremental growth. Each verified against either hand-computed
# geometry or synthetic ground truth with injected noise.
# ============================================================================

def _write_interferogram_with_atmosphere(
    tmp_path, target_phase, ref_phase, atmo_coeff_rad_per_m=0.0005,
    offset=0.0, size=171, res_deg=0.0008,
):
    """A synthetic interferogram with a REAL elevation gradient across the raster and an
    injected elevation-correlated phase trend - the actual failure mode the per-pair
    detrending in measure_double_difference exists to remove. The target block is flat at
    `target_phase`; everywhere else is `ref_phase` with `offset`; a DEM ramping with row
    index provides genuine elevation spread for the regression to fit against.
    """
    d = tmp_path / f"ifgatmo_{abs(hash((target_phase, ref_phase, atmo_coeff_rad_per_m))) % 99999}"
    d.mkdir(exist_ok=True)
    transform = from_origin(LON - size * res_deg / 2, LAT + size * res_deg / 2, res_deg, res_deg)
    prof = dict(driver="GTiff", height=size, width=size, count=1, dtype="float32",
                crs="EPSG:4326", transform=transform)
    c = size // 2
    hw = hyp3.PATCH_PX // 2 + 2

    dem = 2000.0 + 20.0 * np.arange(size)[:, None] * np.ones((1, size))  # 0..~3400m, varies by row
    phase = np.full((size, size), ref_phase + offset, dtype="float64")
    phase[c - hw:c + hw + 1, c - hw:c + hw + 1] = target_phase + offset
    # Inject the SAME elevation-correlated trend everywhere (this is what a real
    # atmospheric delay screen looks like - common-mode, correlated with altitude).
    phase += atmo_coeff_rad_per_m * (dem - dem.mean())

    with rasterio.open(d / "x_unw_phase.tif", "w", **prof) as dst:
        dst.write(phase.astype("float32"), 1)
    with rasterio.open(d / "x_corr.tif", "w", **prof) as dst:
        dst.write(np.full((size, size), 0.8, dtype="float32"), 1)
    with rasterio.open(d / "x_conncomp.tif", "w", **prof) as dst:
        dst.write(np.ones((size, size), dtype="float32"), 1)
    with rasterio.open(d / "x_dem.tif", "w", **prof) as dst:
        dst.write(dem.astype("float32"), 1)
    return d


def test_per_interferogram_detrending_removes_injected_atmosphere(tmp_path):
    """The actual, physically-sound mechanism: fit phase-vs-elevation across the WHOLE
    window (not two fixed points), subtract it, THEN take target-minus-reference.
    """
    true_signal = 2.5  # rad, the real target-vs-reference difference with NO atmosphere
    d = _write_interferogram_with_atmosphere(
        tmp_path, target_phase=true_signal, ref_phase=0.0, atmo_coeff_rad_per_m=0.0008,
    )
    corrected = ts.measure_double_difference(d, LAT, LON)
    assert corrected is not None
    assert corrected["atmospheric_correction_applied"] is True
    # recovered dd_phase should be close to the TRUE signal, with the injected
    # atmosphere (which affects target and reference differently since they sit at
    # different mean elevations within their respective masks) mostly removed
    assert corrected["dd_phase_rad"] == pytest.approx(true_signal, abs=0.3)


def test_no_detrending_without_a_dem_band(tmp_path):
    """Confirms the correction is additive: no DEM, no correction, core value unaffected."""
    d = _write_interferogram_with_atmosphere(tmp_path, target_phase=2.5, ref_phase=0.0,
                                              atmo_coeff_rad_per_m=0.0)
    (d / "x_dem.tif").unlink()
    result = ts.measure_double_difference(d, LAT, LON)
    assert result is not None
    assert result["atmospheric_correction_applied"] is False
    assert result["atmospheric_coefficient_mm_per_m"] is None


def test_detrending_skipped_when_elevation_range_too_small(tmp_path):
    """A near-flat local DEM gives an unidentifiable slope - must skip, not force-fit noise."""
    d = tmp_path / "flat_dem"
    d.mkdir()
    size, res_deg = 171, 0.0008
    transform = from_origin(LON - size * res_deg / 2, LAT + size * res_deg / 2, res_deg, res_deg)
    prof = dict(driver="GTiff", height=size, width=size, count=1, dtype="float32",
                crs="EPSG:4326", transform=transform)
    c = size // 2
    hw = hyp3.PATCH_PX // 2 + 2
    phase = np.full((size, size), 0.0, dtype="float32")
    phase[c - hw:c + hw + 1, c - hw:c + hw + 1] = 2.5
    with rasterio.open(d / "x_unw_phase.tif", "w", **prof) as dst:
        dst.write(phase, 1)
    with rasterio.open(d / "x_corr.tif", "w", **prof) as dst:
        dst.write(np.full((size, size), 0.8, dtype="float32"), 1)
    with rasterio.open(d / "x_conncomp.tif", "w", **prof) as dst:
        dst.write(np.ones((size, size), dtype="float32"), 1)
    with rasterio.open(d / "x_dem.tif", "w", **prof) as dst:
        dst.write(np.full((size, size), 3000.0, dtype="float32"), 1)  # perfectly flat
    result = ts.measure_double_difference(d, LAT, LON)
    assert result is not None
    assert result["atmospheric_correction_applied"] is False


def test_network_level_correction_would_be_ill_posed_and_is_not_attempted():
    """Documents WHY the correction lives in measure_double_difference, not here.

    target_elevation minus reference_elevation is the SAME two fixed geographic points on
    every pair - treating that as one joint network-wide unknown is mathematically
    ill-posed (it was tried; it drove the design matrix rank-deficient). invert_timeseries
    therefore only AGGREGATES what per-pair detrending already decided upstream.
    """
    _, obs = _build_network(aps_sigma_mm=1.0)
    for o in obs:
        o["atmospheric_correction_applied"] = True
        o["atmospheric_coefficient_mm_per_m"] = 0.02
    result = ts.invert_timeseries(obs)
    assert result is not None
    assert result["n_pairs_atmo_corrected"] == len(obs)
    assert result["mean_atmo_coefficient_mm_per_m"] == pytest.approx(0.02, abs=1e-6)
    assert "elevation-detrended" in result["method"]


# ------------------------------------------------------------------ reference stability
def test_collect_network_flags_an_unstable_reference(tmp_path, monkeypatch):
    """The reference annulus itself is checked against a further-out ring.

    A real site is unstable relative to its surroundings here - the check must surface
    that as a caveat, not silently report a velocity as if the zero-point were solid.
    """
    monkeypatch.setattr(hyp3, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(hyp3.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(hyp3, "_auth_headers", lambda: {"Authorization": "Bearer x"})

    epochs, base_obs = _build_network(n_epochs=20, connections=3, velocity_mm_year=0.0, aps_sigma_mm=0.5)
    pairs = []
    for i, o in enumerate(base_obs):
        pairs.append({
            "job_id": f"job{i}", "status": "SUCCEEDED",
            "t_early": o["t_early"].isoformat(), "t_late": o["t_late"].isoformat(),
            "reference": "r", "secondary": "s",
            "observation": {
                "displacement_mm": o["displacement_mm"],
                # reference drifts ~8 mm/yr relative to the outer ring - should be flagged
                "ref_vs_outer_displacement_mm": o["displacement_mm"] + 8.0 * (
                    (o["t_late"] - o["t_early"]).days / 365.25
                ),
            },
        })
    key = ts.network_key(1.0, 2.0, "ASCENDING")
    state = {"version": 1, "networks": {
        key: {"site_id": "s1", "lat": 1.0, "lon": 2.0, "burst_id": "b", "swath": "IW1",
              "polarization": "VV", "flight_direction": "ASCENDING",
              "pairs": pairs, "solution": None}
    }}
    hyp3._save_state(state)
    monkeypatch.setattr(hyp3, "get_job", lambda jid: None)  # pairs already SUCCEEDED+observed

    res = ts.collect_network(1.0, 2.0)
    assert res["reference_stable"] is False

    saved = hyp3._load_state()
    stability = saved["networks"][key]["solution"]["reference_stability"]
    assert stability["checked"] is True
    assert stability["stable"] is False
    assert abs(stability["drift_mm_year"]) > ts.REFERENCE_DRIFT_FLOOR_MM_YEAR


def test_collect_network_passes_a_stable_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(hyp3, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(hyp3.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(hyp3, "_auth_headers", lambda: {"Authorization": "Bearer x"})

    _, base_obs = _build_network(n_epochs=20, connections=3, velocity_mm_year=0.0, aps_sigma_mm=0.5, seed=4)
    pairs = []
    for i, o in enumerate(base_obs):
        pairs.append({
            "job_id": f"job{i}", "status": "SUCCEEDED",
            "t_early": o["t_early"].isoformat(), "t_late": o["t_late"].isoformat(),
            "reference": "r", "secondary": "s",
            "observation": {
                "displacement_mm": o["displacement_mm"],
                "ref_vs_outer_displacement_mm": o["displacement_mm"] * 0.1,  # ~noise, no real drift
            },
        })
    key = ts.network_key(1.0, 2.0, "ASCENDING")
    state = {"version": 1, "networks": {
        key: {"site_id": "s1", "lat": 1.0, "lon": 2.0, "burst_id": "b", "swath": "IW1",
              "polarization": "VV", "flight_direction": "ASCENDING",
              "pairs": pairs, "solution": None}
    }}
    hyp3._save_state(state)
    monkeypatch.setattr(hyp3, "get_job", lambda jid: None)

    res = ts.collect_network(1.0, 2.0)
    assert res["reference_stable"] is True


# ------------------------------------------------------------------ slope-parallel projection
def test_slope_parallel_projection_matches_hand_computed_geometry():
    """Verified by hand against the real Langtang product's look vector (theta=0.97,
    phi=-0.19 rad): a north-facing 40deg slope gives dot=-0.612, an east-facing one
    (same steepness) gives dot=-0.105 - LOWER despite naive "alignment" intuition,
    because the look vector's large vertical component already explains most of a
    downslope-moving target's range change, and the horizontal terms partially cancel
    for an east-facing slope given this specific (ascending, east-looking) geometry.
    """
    theta, phi = 0.97, -0.19
    north = ts.project_los_to_slope_parallel(10.0, theta, phi, slope_deg=40, aspect_deg=0)
    assert north["sensitivity"] == pytest.approx(-0.612, abs=0.005)
    assert north["reliable"] is True
    assert north["slope_parallel_value"] == pytest.approx(10.0 / -0.612, abs=0.05)

    east = ts.project_los_to_slope_parallel(10.0, theta, phi, slope_deg=40, aspect_deg=90)
    assert east["sensitivity"] == pytest.approx(-0.105, abs=0.005)
    assert east["reliable"] is False
    assert east["slope_parallel_value"] is None


def test_slope_parallel_projection_vector_identity():
    """A slope facing exactly where the look vector's horizontal projection points, with
    a vertical look component of zero, must have sensitivity -cos(slope) (pure horizontal
    alignment, satellite directly overhead is excluded by theta=0 here for a clean check).
    """
    # theta=0 -> look vector purely horizontal, pointing East (phi=0)
    out = ts.project_los_to_slope_parallel(1.0, lv_theta_rad=0.0, lv_phi_rad=0.0,
                                            slope_deg=30, aspect_deg=90)  # slope faces East too
    # look=(1,0,0), down=(cos(30),0,-sin(30)) -> dot = cos(30)
    # the function rounds sensitivity to 3dp for display, so the tolerance must match that
    assert out["sensitivity"] == pytest.approx(math.cos(math.radians(30)), abs=5e-4)


def test_aspect_estimation_degrades_cleanly_without_network(monkeypatch):
    monkeypatch.setattr(
        ts.rasterio if False else __import__("rasterio"), "open",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no network")),
    )
    assert ts.estimate_aspect_deg(28.2, 85.5) is None


def test_lookup_slope_deg_returns_none_outside_catalog(monkeypatch):
    class _FakeSite:
        latitude, longitude, slope_deg = 10.0, 10.0, 99.0

    monkeypatch.setattr(
        "app.engine.sites_catalog.load_sites_of_interest", lambda: [_FakeSite()]
    )
    assert ts._lookup_slope_deg(28.2, 85.5) is None
    assert ts._lookup_slope_deg(10.0, 10.0) == 99.0


# ------------------------------------------------------------------ creep input resolution
def _measured(v_year, sigma, **extra):
    rec = {"los_velocity_mm_year": v_year, "uncertainty_mm_year": sigma}
    rec.update(extra)
    return rec


def test_no_measurement_leaves_the_proxy_untouched():
    out = ts.resolve_creep_input(8.0, None)
    assert out["value_mm_month"] == 8.0
    assert out["provenance"] == "MODELLED_PROXY"


def test_detected_motion_replaces_the_proxy():
    out = ts.resolve_creep_input(8.0, _measured(120.0, 5.0))
    assert out["provenance"] == "MEASURED_INSAR"
    assert out["value_mm_month"] == pytest.approx(10.0)


def test_motion_toward_the_sensor_is_used_as_a_magnitude():
    """A negative LOS rate means the ground moved toward the satellite, not that the
    slope is safer. The softening term is a strain-rate proxy and takes the magnitude."""
    out = ts.resolve_creep_input(8.0, _measured(-120.0, 5.0))
    assert out["provenance"] == "MEASURED_INSAR"
    assert out["value_mm_month"] == pytest.approx(10.0)


def test_a_precise_null_rules_the_proxy_out():
    """The real glc_105 result: -0.3 +/- 6.7 mm/yr. Indistinguishable from zero, but tight
    enough to exclude the 8 mm/month the heuristic was asserting."""
    out = ts.resolve_creep_input(8.0, _measured(-0.3, 6.7))
    assert out["provenance"] == "MEASURED_INSAR_NULL"
    assert out["value_mm_month"] == pytest.approx(13.4 / 12.0)
    assert out["value_mm_month"] < 8.0


def test_an_imprecise_null_must_not_silence_the_proxy():
    """The failure mode that matters: a noisy measurement reading as 'slope is still'."""
    out = ts.resolve_creep_input(0.5, _measured(-0.3, 6.7))
    assert out["provenance"] == "MODELLED_PROXY"
    assert out["value_mm_month"] == 0.5
    assert "inconclusive" in out["reason"]


def test_an_unstable_reference_annulus_disqualifies_the_measurement():
    """The velocity is measured against that annulus. If it is drifting, the number is
    describing the annulus, not the slope."""
    out = ts.resolve_creep_input(
        8.0,
        _measured(120.0, 5.0, reference_stability={
            "checked": True, "stable": False, "drift_mm_year": 9.0,
        }),
    )
    assert out["provenance"] == "MODELLED_PROXY"
    assert out["value_mm_month"] == 8.0
    assert "disqualified" in out["reason"]


def test_a_stable_reference_annulus_does_not_block_the_measurement():
    out = ts.resolve_creep_input(
        8.0,
        _measured(120.0, 5.0, reference_stability={
            "checked": True, "stable": True, "drift_mm_year": 0.4,
        }),
    )
    assert out["provenance"] == "MEASURED_INSAR"


def test_slope_parallel_rate_is_preferred_over_raw_los():
    out = ts.resolve_creep_input(
        8.0,
        _measured(120.0, 5.0,
                  slope_parallel_reliable=True,
                  slope_parallel_velocity_mm_year=240.0,
                  slope_parallel_uncertainty_mm_year=10.0),
    )
    assert out["geometry"] == "slope-parallel"
    assert out["value_mm_month"] == pytest.approx(20.0)
    # the sigma used must be the slope-parallel one, not the line-of-sight 5.0
    assert out["uncertainty_mm_year"] == 10.0


def test_a_downslope_velocity_without_a_downslope_sigma_is_refused():
    """Pairing a slope-parallel rate with the raw LOS sigma understates the error by
    1/|s| - up to 6x at the sensitivity gate - and that ratio is exactly what decides
    MEASURED vs PROXY. Refuse rather than mix the two quantities."""
    out = ts.resolve_creep_input(
        8.0,
        _measured(120.0, 5.0,
                  slope_parallel_reliable=True,
                  slope_parallel_velocity_mm_year=240.0),   # no matching sigma
    )
    assert out["provenance"] == "MODELLED_PROXY"
    assert "without a matching slope-parallel uncertainty" in out["reason"]


def test_unreliable_slope_parallel_projection_falls_back_to_los():
    out = ts.resolve_creep_input(
        8.0,
        _measured(120.0, 5.0,
                  slope_parallel_reliable=False,
                  slope_parallel_velocity_mm_year=9999.0),
    )
    assert out["geometry"] == "line-of-sight"
    assert out["value_mm_month"] == pytest.approx(10.0)


def test_a_measurement_without_an_uncertainty_is_not_usable():
    out = ts.resolve_creep_input(8.0, {"los_velocity_mm_year": 120.0})
    assert out["provenance"] == "MODELLED_PROXY"


def test_load_measured_velocities_makes_no_network_call(tmp_path, monkeypatch):
    """The 30-minute loop runs this over ~491 sites. It must never reach the internet."""
    state = {"version": hyp3.STATE_VERSION, "networks": {hyp3.site_key(28.0, 85.0): {
        "lat": 28.0, "lon": 85.0, "burst_id": "b", "swath": "IW2", "polarization": "VV",
        "pairs": [], "solved_utc": time.time(),
        "solution": {
            "velocity_mm_month": 1.0, "velocity_mm_year": 12.0,
            "velocity_stderr_mm_year": 2.0, "n_pairs": 30, "n_epochs": 12,
            "span_days": 300.0, "residual_rms_mm": 3.0, "redundancy": 2.5,
        },
    }}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    monkeypatch.setattr(hyp3, "STATE_PATH", path)
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)

    def _boom(*a, **k):
        raise AssertionError("estimate_aspect_deg reached the network from the hot path")

    monkeypatch.setattr(ts, "estimate_aspect_deg", _boom)
    ts._VELOCITY_CACHE["mtime"] = None

    values = ts.load_measured_velocities()
    assert set(values) == {hyp3.site_key(28.0, 85.0)}
    assert values[hyp3.site_key(28.0, 85.0)]["los_velocity_mm_year"] == 12.0


def test_load_measured_velocities_caches_against_state_mtime(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": hyp3.STATE_VERSION, "networks": {}}))
    monkeypatch.setattr(hyp3, "STATE_PATH", path)
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    ts._VELOCITY_CACHE["mtime"] = None

    calls = {"n": 0}
    real_load = hyp3._load_state

    def counting_load():
        calls["n"] += 1
        return real_load()

    monkeypatch.setattr(hyp3, "_load_state", counting_load)
    ts.load_measured_velocities()
    ts.load_measured_velocities()
    ts.load_measured_velocities()
    assert calls["n"] == 1


# ------------------------------------------------------------------ direction-scoped keys
def test_network_key_is_scoped_by_flight_direction():
    asc = ts.network_key(29.5023, 82.2504, "ASCENDING")
    desc = ts.network_key(29.5023, 82.2504, "DESCENDING")
    assert asc != desc
    assert asc.startswith(hyp3.site_key(29.5023, 82.2504))
    assert ts.split_network_key(asc) == (hyp3.site_key(29.5023, 82.2504), "ASCENDING")


def test_network_key_normalises_case_and_missing_direction():
    assert ts.network_key(1.0, 2.0, "ascending") == ts.network_key(1.0, 2.0, "ASCENDING")
    assert ts.split_network_key(ts.network_key(1.0, 2.0, ""))[1] == ts.DIRECTION_UNKNOWN


def test_a_bare_legacy_key_still_parses():
    """Keys written before direction scoping have no separator at all."""
    assert ts.split_network_key("29.5023,82.2504") == ("29.5023,82.2504", "UNKNOWN")


def test_legacy_migration_preserves_the_solved_network():
    """The production state file holds one solved network under a bare key. Losing it
    would throw away 90 spent HyP3 credits and the only measured velocity in the system."""
    solution = {"velocity_mm_year": -0.31, "velocity_stderr_mm_year": 6.71, "n_pairs": 39}
    state = {"version": hyp3.STATE_VERSION, "networks": {
        "29.5023,82.2504": {"lat": 29.5023, "lon": 82.2504, "site_id": "glc_105",
                            "pairs": [{"job_id": "a"}], "solution": solution},
    }}
    assert ts.migrate_legacy_network_keys(state) == 1
    assert list(state["networks"]) == ["29.5023,82.2504@UNKNOWN"]
    migrated = state["networks"]["29.5023,82.2504@UNKNOWN"]
    assert migrated["solution"] == solution
    assert migrated["flight_direction"] == "UNKNOWN"


def test_legacy_migration_is_idempotent():
    state = {"version": hyp3.STATE_VERSION, "networks": {
        "1.0000,2.0000": {"lat": 1.0, "lon": 2.0, "solution": {"velocity_mm_year": 1.0}},
    }}
    assert ts.migrate_legacy_network_keys(state) == 1
    assert ts.migrate_legacy_network_keys(state) == 0
    assert ts.migrate_legacy_network_keys(state) == 0
    assert len(state["networks"]) == 1


def test_migration_never_guesses_a_direction():
    """A guessed direction would silently corrupt the asc/desc combination. The legacy
    record has nothing to infer from, so it must say UNKNOWN."""
    state = {"version": hyp3.STATE_VERSION, "networks": {
        "1.0000,2.0000": {"lat": 1.0, "lon": 2.0, "burst_id": "337397", "swath": "IW1"},
    }}
    ts.migrate_legacy_network_keys(state)
    assert state["networks"]["1.0000,2.0000@UNKNOWN"]["flight_direction"] == "UNKNOWN"


def test_migration_keeps_a_solved_network_over_an_unsolved_collision():
    state = {"version": hyp3.STATE_VERSION, "networks": {
        "1.0000,2.0000@UNKNOWN": {"lat": 1.0, "lon": 2.0, "solution": {"velocity_mm_year": 5.0}},
        "1.0000,2.0000": {"lat": 1.0, "lon": 2.0, "solution": None},
    }}
    ts.migrate_legacy_network_keys(state)
    kept = state["networks"]["1.0000,2.0000@UNKNOWN"]
    assert kept["solution"] == {"velocity_mm_year": 5.0}


def test_networks_for_site_returns_every_direction():
    state = {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(29.5, 82.25, "ASCENDING"): {"lat": 29.5, "lon": 82.25, "solution": {}},
        ts.network_key(29.5, 82.25, "DESCENDING"): {"lat": 29.5, "lon": 82.25, "solution": {}},
        ts.network_key(28.0, 85.0, "ASCENDING"): {"lat": 28.0, "lon": 85.0},
    }}
    found = ts.networks_for_site(state, 29.5, 82.25)
    assert set(found) == {"ASCENDING", "DESCENDING"}
    assert set(ts.networks_for_site(state, 28.0, 85.0)) == {"ASCENDING"}
    assert ts.networks_for_site(state, 1.0, 1.0) == {}


def test_networks_for_site_does_not_match_a_different_site_by_prefix():
    """site_key is a fixed 4-dp format, but guard the prefix match anyway."""
    state = {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(29.5, 82.25, "ASCENDING"): {"lat": 29.5, "lon": 82.25},
    }}
    assert ts.networks_for_site(state, 29.5, 82.2) == {}


def test_collect_network_migrates_a_legacy_key_in_place(tmp_path, monkeypatch):
    """A production state file written before direction scoping must still collect."""
    monkeypatch.setattr(hyp3, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(hyp3, "_auth_headers", lambda: {"Authorization": "x"})
    hyp3._save_state({"version": hyp3.STATE_VERSION, "networks": {
        "1.0000,2.0000": {"site_id": "s1", "lat": 1.0, "lon": 2.0, "pairs": [],
                          "solution": None},
    }})
    res = ts.collect_network(1.0, 2.0)
    assert res["status"] != "NO_NETWORK"          # the legacy network was found
    assert "1.0000,2.0000@UNKNOWN" in hyp3._load_state()["networks"]


def test_collect_network_refuses_to_guess_between_two_geometries(tmp_path, monkeypatch):
    """With both directions recorded, 'the' network is no longer a well-defined request."""
    monkeypatch.setattr(hyp3, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(hyp3, "_auth_headers", lambda: {"Authorization": "x"})
    hyp3._save_state({"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(1.0, 2.0, "ASCENDING"): {"lat": 1.0, "lon": 2.0, "pairs": []},
        ts.network_key(1.0, 2.0, "DESCENDING"): {"lat": 1.0, "lon": 2.0, "pairs": []},
    }})
    res = ts.collect_network(1.0, 2.0)
    assert res["status"] == "AMBIGUOUS_DIRECTION"
    assert res["available_directions"] == ["ASCENDING", "DESCENDING"]
    # naming one resolves it
    assert ts.collect_network(1.0, 2.0, "DESCENDING")["status"] != "AMBIGUOUS_DIRECTION"


def _solved_net(lat, lon, stderr, direction="ASCENDING", v=12.0):
    return {"lat": lat, "lon": lon, "burst_id": "b", "swath": "IW1", "polarization": "VV",
            "flight_direction": direction, "pairs": [], "solved_utc": time.time(),
            "solution": {"velocity_mm_month": v / 12.0, "velocity_mm_year": v,
                         "velocity_stderr_mm_year": stderr, "n_pairs": 30, "n_epochs": 12,
                         "span_days": 300.0, "residual_rms_mm": 3.0, "redundancy": 2.5}}


def test_a_migrated_network_is_still_found_by_the_provider(tmp_path, monkeypatch):
    """The regression that would have fired on the next Sunday cron: once collect_network
    migrates the key, a bare-site_key lookup returns nothing and the only measured
    velocity in the system silently reverts to MODELLED_PROXY."""
    path = tmp_path / "state.json"
    state = {"version": hyp3.STATE_VERSION,
             "networks": {"29.5023,82.2504": _solved_net(29.5023, 82.2504, 6.71)}}
    ts.migrate_legacy_network_keys(state)
    assert list(state["networks"]) == ["29.5023,82.2504@ASCENDING"]
    path.write_text(json.dumps(state))
    monkeypatch.setattr(hyp3, "STATE_PATH", path)
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(ts, "estimate_aspect_deg", lambda *a, **k: None)
    ts._VELOCITY_CACHE["mtime"] = None

    assert ts.timeseries_velocity_provider(29.5023, 82.2504, allow_network=False) is not None
    assert set(ts.load_measured_velocities(force=True)) == {hyp3.site_key(29.5023, 82.2504)}


def test_an_unmigrated_legacy_state_still_resolves(tmp_path, monkeypatch):
    """Belt and braces: the on-disk file may not have been migrated yet."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": hyp3.STATE_VERSION,
                                "networks": {"29.5023,82.2504": _solved_net(29.5023, 82.2504, 6.71)}}))
    monkeypatch.setattr(hyp3, "STATE_PATH", path)
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(ts, "estimate_aspect_deg", lambda *a, **k: None)
    ts._VELOCITY_CACHE["mtime"] = None
    assert ts.timeseries_velocity_provider(29.5023, 82.2504, allow_network=False) is not None


def test_load_measured_velocities_keys_by_bare_site_not_by_direction(tmp_path, monkeypatch):
    """The physics loop looks sites up by hyp3.site_key. A direction-scoped key here
    would miss every one of them."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(28.0, 85.0, "ASCENDING"): _solved_net(28.0, 85.0, 4.0, "ASCENDING"),
        ts.network_key(28.0, 85.0, "DESCENDING"): _solved_net(28.0, 85.0, 3.0, "DESCENDING"),
    }}))
    monkeypatch.setattr(hyp3, "STATE_PATH", path)
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(ts, "estimate_aspect_deg", lambda *a, **k: None)
    ts._VELOCITY_CACHE["mtime"] = None

    values = ts.load_measured_velocities(force=True)
    # ONE record for the site, not one per geometry
    assert set(values) == {hyp3.site_key(28.0, 85.0)}


def test_the_provider_picks_the_tightest_geometry_when_not_told_which(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(28.0, 85.0, "ASCENDING"): _solved_net(28.0, 85.0, 9.0, "ASCENDING"),
        ts.network_key(28.0, 85.0, "DESCENDING"): _solved_net(28.0, 85.0, 2.0, "DESCENDING"),
    }}))
    monkeypatch.setattr(hyp3, "STATE_PATH", path)
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(ts, "estimate_aspect_deg", lambda *a, **k: None)

    best = ts.timeseries_velocity_provider(28.0, 85.0, allow_network=False)
    assert best["flight_direction"] == "DESCENDING"
    assert best["uncertainty_mm_year"] == 2.0
    asc = ts.timeseries_velocity_provider(28.0, 85.0, allow_network=False, direction="ASCENDING")
    assert asc["flight_direction"] == "ASCENDING"
    assert ts.timeseries_velocity_provider(
        28.0, 85.0, allow_network=False, direction="NONSENSE") is None


# ------------------------------------------------------------------ asc + desc end to end
# Look vectors MEASURED from real HyP3 products (64 local products carrying lv_phi/lv_theta;
# 63 ascending at glc_105 burst 337397/IW1, one descending at 039571/IW1). Not assumed.
# 30 degrees off north. At exactly 0/180 the two look vectors respond identically to
# vertical motion, so the consistency test has no power and correctly reports None rather
# than True - pinned separately by test_a_blind_geometry_reports_untested_not_agreed.
_ASPECT = 30.0

MEASURED_ASC = {"lv_theta_rad": 0.9696, "lv_phi_rad": -2.9541}
MEASURED_DESC = {"lv_theta_rad": 0.9719, "lv_phi_rad": -0.1883}


def _geom_net(lat, lon, direction, v, stderr, lv, span_days=300.0):
    now = time.time()
    obs = {"displacement_mm": 1.0, "lv_theta_rad": lv["lv_theta_rad"],
           "lv_phi_rad": lv["lv_phi_rad"]}
    return {"lat": lat, "lon": lon, "burst_id": "b", "swath": "IW1", "polarization": "VV",
            "flight_direction": direction, "slope_deg": 35.0, "aspect_deg": _ASPECT,
            "solved_utc": now,
            "pairs": [{"job_id": "j1", "status": "SUCCEEDED", "observation": obs}],
            "solution": {"velocity_mm_month": v / 12.0, "velocity_mm_year": v,
                         "velocity_stderr_mm_year": stderr, "n_pairs": 30, "n_epochs": 12,
                         "span_days": span_days, "residual_rms_mm": 3.0, "redundancy": 2.5}}


def _two_geometry_state(v_asc, v_desc, s_asc=6.0, s_desc=6.0):
    return {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(29.5, 82.25, "ASCENDING"):
            _geom_net(29.5, 82.25, "ASCENDING", v_asc, s_asc, MEASURED_ASC),
        ts.network_key(29.5, 82.25, "DESCENDING"):
            _geom_net(29.5, 82.25, "DESCENDING", v_desc, s_desc, MEASURED_DESC),
    }}


def _install(state, tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    monkeypatch.setattr(hyp3, "STATE_PATH", path)
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(ts, "estimate_aspect_deg", lambda *a, **k: None)
    ts._VELOCITY_CACHE["mtime"] = None


def test_two_geometries_combine_into_one_record(tmp_path, monkeypatch):
    from app.ingestion import insar_combine
    truth = 30.0
    s_a = insar_combine.projection_sensitivity(
        MEASURED_ASC["lv_theta_rad"], MEASURED_ASC["lv_phi_rad"], 35.0, _ASPECT)
    s_d = insar_combine.projection_sensitivity(
        MEASURED_DESC["lv_theta_rad"], MEASURED_DESC["lv_phi_rad"], 35.0, _ASPECT)
    _install(_two_geometry_state(s_a * truth, s_d * truth), tmp_path, monkeypatch)

    rec = ts.combined_velocity_for_site(29.5, 82.25)
    assert rec is not None
    assert rec["geometries_available"] == ["ASCENDING", "DESCENDING"]
    assert rec["combination"]["status"] == "OK"
    assert rec["combination"]["n_tracks_used"] == 2
    # the combined downslope rate recovers the truth both tracks were built from
    assert rec["slope_parallel_velocity_mm_year"] == pytest.approx(truth, abs=0.5)
    assert rec["combination_consistent"] is True
    assert "ASCENDING+DESCENDING" in rec["source"]


def test_combining_is_tighter_than_either_track_alone(tmp_path, monkeypatch):
    """The precision half of the claim, at an aspect where both tracks see the slope."""
    from app.ingestion import insar_combine
    truth, sigma = 30.0, 6.0
    s_a = insar_combine.projection_sensitivity(
        MEASURED_ASC["lv_theta_rad"], MEASURED_ASC["lv_phi_rad"], 35.0, _ASPECT)
    s_d = insar_combine.projection_sensitivity(
        MEASURED_DESC["lv_theta_rad"], MEASURED_DESC["lv_phi_rad"], 35.0, _ASPECT)
    _install(_two_geometry_state(s_a * truth, s_d * truth, sigma, sigma),
             tmp_path, monkeypatch)

    combined = ts.combined_velocity_for_site(29.5, 82.25)["combination"]
    best_single = min(sigma / abs(s_a), sigma / abs(s_d))
    assert combined["uncertainty_mm_year"] < best_single
    improvement = best_single / combined["uncertainty_mm_year"]
    assert 1.0 < improvement <= math.sqrt(2) + 1e-6


def test_disagreeing_geometries_do_not_report_a_tight_number(tmp_path, monkeypatch):
    """The honesty half. Two tracks that cannot both come from one downslope rate must
    degrade the uncertainty, not keep the tight combined value."""
    _install(_two_geometry_state(+40.0, -40.0, 1.0, 1.0), tmp_path, monkeypatch)
    combined = ts.combined_velocity_for_site(29.5, 82.25)["combination"]
    assert combined["consistent"] is False
    sp = combined["slope_parallel"]
    assert sp["uncertainty_mm_year"] > sp["formal_uncertainty_mm_year"]


def test_a_single_geometry_site_is_unchanged_by_the_combination_path(tmp_path, monkeypatch):
    """The common case today: one network, no combination, same record as before."""
    state = {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(29.5, 82.25, "ASCENDING"):
            _geom_net(29.5, 82.25, "ASCENDING", 12.0, 4.0, MEASURED_ASC)}}
    _install(state, tmp_path, monkeypatch)
    rec = ts.combined_velocity_for_site(29.5, 82.25)
    assert rec["geometries_available"] == ["ASCENDING"]
    assert "combination" not in rec
    assert rec["los_velocity_mm_year"] == 12.0


def test_the_hot_path_serves_the_combined_record(tmp_path, monkeypatch):
    from app.ingestion import insar_combine
    s_a = insar_combine.projection_sensitivity(
        MEASURED_ASC["lv_theta_rad"], MEASURED_ASC["lv_phi_rad"], 35.0, _ASPECT)
    s_d = insar_combine.projection_sensitivity(
        MEASURED_DESC["lv_theta_rad"], MEASURED_DESC["lv_phi_rad"], 35.0, _ASPECT)
    _install(_two_geometry_state(s_a * 30.0, s_d * 30.0), tmp_path, monkeypatch)
    values = ts.load_measured_velocities(force=True)
    assert set(values) == {hyp3.site_key(29.5, 82.25)}
    assert values[hyp3.site_key(29.5, 82.25)]["combination"]["n_tracks_used"] == 2


def test_the_measured_look_vectors_classify_to_the_right_directions():
    """Guards the sign convention the whole decomposition rests on, using the values
    measured from real products rather than the ones derived from orbital geometry."""
    from app.ingestion import insar_combine
    assert insar_combine.classify_flight_direction(MEASURED_ASC["lv_phi_rad"]) == "ASCENDING"
    assert insar_combine.classify_flight_direction(MEASURED_DESC["lv_phi_rad"]) == "DESCENDING"
    assert math.cos(MEASURED_ASC["lv_phi_rad"]) < 0    # looks West -> ascending
    assert math.cos(MEASURED_DESC["lv_phi_rad"]) > 0   # looks East -> descending


# ------------------------------------------------------------------ direction backfill
def test_backfill_resolves_an_unknown_network_and_rekeys_it(monkeypatch):
    """An UNKNOWN network can never be combined - its geometry is the thing a combination
    needs. The burst id was stored, and CMR maps it to one authoritative direction."""
    monkeypatch.setattr(hyp3, "burst_direction", lambda *a, **k: "ASCENDING")
    solution = {"velocity_mm_year": -0.31, "velocity_stderr_mm_year": 6.71}
    state = {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(29.5023, 82.2504, "UNKNOWN"): {
            "lat": 29.5023, "lon": 82.2504, "burst_id": "337397", "swath": "IW1",
            "polarization": "VV", "flight_direction": "UNKNOWN",
            "pairs": [{"job_id": "a"}], "solution": solution},
    }}
    assert ts.backfill_network_direction(state, 29.5023, 82.2504) == "ASCENDING"
    assert list(state["networks"]) == ["29.5023,82.2504@ASCENDING"]
    net = state["networks"]["29.5023,82.2504@ASCENDING"]
    assert net["flight_direction"] == "ASCENDING"
    assert net["solution"] == solution and len(net["pairs"]) == 1


def test_backfill_leaves_the_network_alone_when_cmr_cannot_say(monkeypatch):
    monkeypatch.setattr(hyp3, "burst_direction", lambda *a, **k: "UNKNOWN")
    state = {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(1.0, 2.0, "UNKNOWN"): {"lat": 1.0, "lon": 2.0, "burst_id": "b",
                                              "swath": "IW1", "polarization": "VV"},
    }}
    assert ts.backfill_network_direction(state, 1.0, 2.0) is None
    assert list(state["networks"]) == ["1.0000,2.0000@UNKNOWN"]


def test_backfill_refuses_to_merge_onto_an_occupied_direction(monkeypatch):
    """Two networks built from different stacks must not be silently merged."""
    monkeypatch.setattr(hyp3, "burst_direction", lambda *a, **k: "ASCENDING")
    state = {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(1.0, 2.0, "UNKNOWN"): {"lat": 1.0, "lon": 2.0, "burst_id": "b1",
                                              "swath": "IW1", "polarization": "VV"},
        ts.network_key(1.0, 2.0, "ASCENDING"): {"lat": 1.0, "lon": 2.0, "burst_id": "b2",
                                                "swath": "IW2", "polarization": "VV",
                                                "solution": {"velocity_mm_year": 3.0}},
    }}
    assert ts.backfill_network_direction(state, 1.0, 2.0) is None
    assert state["networks"][ts.network_key(1.0, 2.0, "ASCENDING")]["burst_id"] == "b2"
    assert ts.network_key(1.0, 2.0, "UNKNOWN") in state["networks"]


def test_backfill_needs_a_burst_id(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("CMR queried with no burst id to query for")

    monkeypatch.setattr(hyp3, "burst_direction", _boom)
    state = {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(1.0, 2.0, "UNKNOWN"): {"lat": 1.0, "lon": 2.0},
    }}
    assert ts.backfill_network_direction(state, 1.0, 2.0) is None


def test_backfill_is_not_reachable_from_the_hot_path(tmp_path, monkeypatch):
    """The 30-minute loop must never make a network call. Backfill is a weekly-path
    operation; if load_measured_velocities could reach it, the loop would depend on CMR."""
    def _boom(*a, **k):
        raise AssertionError("burst_direction reached from the hot path")

    monkeypatch.setattr(hyp3, "burst_direction", _boom)
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(29.5, 82.25, "UNKNOWN"): _solved_net(29.5, 82.25, 6.0, "UNKNOWN"),
    }}))
    monkeypatch.setattr(hyp3, "STATE_PATH", path)
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(ts, "estimate_aspect_deg", lambda *a, **k: None)
    ts._VELOCITY_CACHE["mtime"] = None
    assert ts.load_measured_velocities(force=True)      # still serves the velocity


def test_east_up_is_still_recovered_when_the_aspect_is_unknown(tmp_path, monkeypatch):
    """Separating vertical from horizontal motion needs only the two look vectors - no
    slope, no aspect. Withholding it when the aspect is unresolved would discard the one
    thing no single track can do, in exactly the case that is normal today."""
    _install(_two_geometry_state(-12.0, +8.0, 3.0, 3.0), tmp_path, monkeypatch)
    # same state, but with the aspect genuinely unknown
    state = json.loads((tmp_path / "state.json").read_text())
    for net in state["networks"].values():
        net.pop("aspect_deg", None)
    (tmp_path / "state.json").write_text(json.dumps(state))
    ts._VELOCITY_CACHE["mtime"] = None

    rec = ts.combined_velocity_for_site(29.5, 82.25)
    assert rec["combination"]["status"] == "NO_SLOPE_GEOMETRY"   # downslope withheld
    # no downslope rate is offered at all, because the downslope direction is unknown
    assert rec.get("slope_parallel_velocity_mm_year") is None
    # but the decomposition that needs no slope model still runs
    assert rec["east_velocity_mm_year"] is not None
    assert rec["up_velocity_mm_year"] is not None
    # the assumption that makes it possible must always be stated
    assert rec["east_up_assumes_no_north_motion"] is True


def test_east_up_is_not_claimed_from_two_same_direction_tracks(tmp_path, monkeypatch):
    """Two ascending tracks are not an asc/desc pair; decomposing them would amplify
    noise into a confident-looking number."""
    state = {"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(29.5, 82.25, "ASCENDING"):
            _geom_net(29.5, 82.25, "ASCENDING", -12.0, 3.0, MEASURED_ASC),
        ts.network_key(29.5, 82.25, "UNKNOWN"):
            _geom_net(29.5, 82.25, "UNKNOWN", -11.0, 3.0, MEASURED_ASC),
    }}
    _install(state, tmp_path, monkeypatch)
    rec = ts.combined_velocity_for_site(29.5, 82.25)
    eu = (rec.get("combination") or {}).get("east_up")
    assert rec.get("east_velocity_mm_year") is None
    assert eu is None or eu.get("refused") is True


def test_a_blind_geometry_reports_untested_not_agreed(tmp_path, monkeypatch):
    """End-to-end guard on the worst failure the review found: on a due-north slope the
    two geometries respond identically to vertical motion, so pure subsidence came back
    as a multi-sigma downslope creep rate labelled "tracks agree"."""
    from app.ingestion import insar_combine
    state = _two_geometry_state(-5.5, +5.5)
    for net in state["networks"].values():
        net["aspect_deg"] = 0.0           # due north: the blind spot
    _install(state, tmp_path, monkeypatch)

    rec = ts.combined_velocity_for_site(29.5, 82.25)
    combo = rec["combination"]
    assert combo["consistent"] is None            # never True
    sp = combo["slope_parallel"]
    assert sp["vertical_blind_spot"]["blind_to_vertical"] is True
    assert "UNTESTABLE" in combo["summary"]
    # and the uncertainty that travels with the velocity is the combination's own
    assert rec["slope_parallel_uncertainty_mm_year"] == combo["uncertainty_mm_year"]


# ------------------------------------------------------------------ burst availability
def test_unavailable_bursts_are_dropped_before_they_can_burn_credits(monkeypatch):
    """Measured at glc_105: five of 34 acquisitions were listed by CMR but missing from
    ASF's burst index. Each sits in ~6 pairs, so they killed 27 of 90 jobs - every one
    failing inside HyP3 with "Failed to find granule(s)", never starting, 27 credits
    charged and not refunded, and the gap severed the network across the monsoon."""
    stack = _stack(20)
    doomed = {stack.acquisitions[3][1], stack.acquisitions[7][1]}
    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [stack])
    monkeypatch.setattr(hyp3, "resolve_granules",
                        lambda names, **k: {n for n in names if n not in doomed})
    plan = ts.plan_network(LAT, LON, since_days=None)
    assert plan is not None
    planned = {g for p in plan.pairs for g in (p[0], p[1])}
    assert not (planned & doomed)          # not one credit spent on an unprocessable burst
    assert len(plan.epochs) == 18          # 20 minus the two that ASF cannot serve


def test_a_failed_availability_check_does_not_block_planning(monkeypatch):
    """ASF being briefly unreachable must not stop the weekly run. 'We do not know' is
    not the same as 'nothing is available', so planning proceeds with a warning."""
    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [_stack(20)])
    monkeypatch.setattr(hyp3, "resolve_granules", lambda names, **k: None)
    plan = ts.plan_network(LAT, LON, since_days=None)
    assert plan is not None
    assert len(plan.epochs) == 20          # nothing dropped on an inconclusive check


def test_planning_refuses_when_too_few_bursts_survive_the_check(monkeypatch):
    stack = _stack(20)
    keep = {a[1] for a in stack.acquisitions[:3]}
    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [stack])
    monkeypatch.setattr(hyp3, "resolve_granules", lambda names, **k: keep)
    assert ts.plan_network(LAT, LON, since_days=None) is None


def test_the_check_can_be_turned_off(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("ASF queried despite validate_against_asf=False")

    monkeypatch.setattr(hyp3, "find_burst_stacks", lambda *a, **k: [_stack(20)])
    monkeypatch.setattr(hyp3, "resolve_granules", _boom)
    assert ts.plan_network(LAT, LON, since_days=None, validate_against_asf=False) is not None


# ------------------------------------------------------------------ measurement versioning
def test_a_stale_observation_is_re_measured_from_the_local_product(tmp_path, monkeypatch):
    """The bug this closes cost three shipped features at once. collect_network skipped
    any pair that already had an observation, so per-interferogram atmospheric
    detrending, look-vector extraction (which the slope-parallel projection and the whole
    asc+desc combination depend on) and the reference-stability check had all been
    written, tested and deployed without ever touching the real data. Re-measuring the
    production products turned a null (-0.3 +/- 6.7) into a 3.1-sigma detection
    (+8.4 +/- 2.7)."""
    monkeypatch.setattr(hyp3, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(hyp3, "_auth_headers", lambda: {"Authorization": "x"})
    monkeypatch.setattr(hyp3, "burst_direction", lambda *a, **k: "ASCENDING")
    monkeypatch.setattr(hyp3, "product_dir_for_job", lambda jid, **k: tmp_path / "prod")
    fresh = {"displacement_mm": 7.0, "measurement_version": ts.MEASUREMENT_VERSION,
             "lv_theta_rad": 0.97, "lv_phi_rad": -2.95,
             "atmospheric_correction_applied": True,
             "atmospheric_coefficient_mm_per_m": 0.004}
    monkeypatch.setattr(ts, "measure_double_difference", lambda *a, **k: dict(fresh))

    hyp3._save_state({"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(1.0, 2.0, "ASCENDING"): {
            "lat": 1.0, "lon": 2.0, "flight_direction": "ASCENDING", "pairs": [{
                "job_id": "j1", "status": "SUCCEEDED",
                "t_early": "2025-01-01T00:00:00+00:00", "t_late": "2025-01-13T00:00:00+00:00",
                # a v1 observation: only the five original keys, no version stamp
                "observation": {"displacement_mm": 1.0, "dd_phase_rad": 0.2,
                                "mean_coherence": 0.5, "target_pixels": 100,
                                "reference_pixels": 900},
            }], "solution": None}}})

    res = ts.collect_network(1.0, 2.0, "ASCENDING")
    assert res["remeasured"] == 1
    saved = hyp3._load_state()["networks"][ts.network_key(1.0, 2.0, "ASCENDING")]
    obs = saved["pairs"][0]["observation"]
    assert obs["measurement_version"] == ts.MEASUREMENT_VERSION
    assert obs["lv_theta_rad"] == 0.97            # look vectors now present
    assert obs["atmospheric_correction_applied"] is True


def test_a_current_observation_is_not_re_measured(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("re-measured an observation that was already current")

    monkeypatch.setattr(hyp3, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(hyp3, "_auth_headers", lambda: {"Authorization": "x"})
    monkeypatch.setattr(hyp3, "burst_direction", lambda *a, **k: "ASCENDING")
    monkeypatch.setattr(ts, "measure_double_difference", _boom)
    hyp3._save_state({"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(1.0, 2.0, "ASCENDING"): {
            "lat": 1.0, "lon": 2.0, "flight_direction": "ASCENDING", "pairs": [{
                "job_id": "j1", "status": "SUCCEEDED",
                "t_early": "2025-01-01T00:00:00+00:00", "t_late": "2025-01-13T00:00:00+00:00",
                "observation": {"displacement_mm": 1.0,
                                "measurement_version": ts.MEASUREMENT_VERSION},
            }], "solution": None}}})
    assert ts.collect_network(1.0, 2.0, "ASCENDING")["remeasured"] == 0


def test_a_failed_re_measurement_keeps_the_old_observation(tmp_path, monkeypatch):
    """A transient raster read error must not turn a usable pair into a lost one."""
    monkeypatch.setattr(hyp3, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(hyp3, "_auth_headers", lambda: {"Authorization": "x"})
    monkeypatch.setattr(hyp3, "burst_direction", lambda *a, **k: "ASCENDING")
    monkeypatch.setattr(hyp3, "product_dir_for_job", lambda jid, **k: tmp_path / "prod")
    monkeypatch.setattr(ts, "measure_double_difference", lambda *a, **k: None)
    hyp3._save_state({"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(1.0, 2.0, "ASCENDING"): {
            "lat": 1.0, "lon": 2.0, "flight_direction": "ASCENDING", "pairs": [{
                "job_id": "j1", "status": "SUCCEEDED",
                "t_early": "2025-01-01T00:00:00+00:00", "t_late": "2025-01-13T00:00:00+00:00",
                "observation": {"displacement_mm": 1.0},
            }], "solution": None}}})
    res = ts.collect_network(1.0, 2.0, "ASCENDING")
    assert res["remeasured"] == 0
    saved = hyp3._load_state()["networks"][ts.network_key(1.0, 2.0, "ASCENDING")]
    assert saved["pairs"][0]["observation"]["displacement_mm"] == 1.0


def test_a_stale_observation_with_no_local_product_is_kept(tmp_path, monkeypatch):
    """Products are prunable. Losing the raster must not lose the measurement."""
    monkeypatch.setattr(hyp3, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ts.settings, "hyp3_enabled", True, raising=False)
    monkeypatch.setattr(hyp3, "_auth_headers", lambda: {"Authorization": "x"})
    monkeypatch.setattr(hyp3, "burst_direction", lambda *a, **k: "ASCENDING")
    monkeypatch.setattr(hyp3, "product_dir_for_job", lambda jid, **k: None)
    hyp3._save_state({"version": hyp3.STATE_VERSION, "networks": {
        ts.network_key(1.0, 2.0, "ASCENDING"): {
            "lat": 1.0, "lon": 2.0, "flight_direction": "ASCENDING", "pairs": [{
                "job_id": "j1", "status": "SUCCEEDED",
                "t_early": "2025-01-01T00:00:00+00:00", "t_late": "2025-01-13T00:00:00+00:00",
                "observation": {"displacement_mm": 1.0},
            }], "solution": None}}})
    res = ts.collect_network(1.0, 2.0, "ASCENDING")
    assert res["measured"] == 1 and res["remeasured"] == 0


def test_new_measurements_carry_the_current_version(tmp_path):
    out = ts.measure_double_difference(
        _write_interferogram_with_atmosphere(tmp_path, 2.5, 0.5),
        LAT, LON)
    assert out["measurement_version"] == ts.MEASUREMENT_VERSION
    # two of the three things the stale-observation bug had been suppressing (the
    # third, look vectors, needs lv_theta/lv_phi bands this synthetic fixture omits -
    # the real products do carry them, verified: 54/54 re-measurable from local disk)
    assert out["atmospheric_correction_applied"] is True
    assert "ref_vs_outer_displacement_mm" in out
