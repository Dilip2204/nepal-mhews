"""NISAR L-band GUNW reader: the measurement C-band cannot make.

No test here touches the network. What is pinned is the arithmetic, the geolocation
logic, the refusal paths, and above all the honesty flags - the sign convention is NOT
yet verified against a product README and nothing may quietly claim otherwise.
"""

from __future__ import annotations

import math

import pytest

from app.ingestion import nisar_gunw as ng


# ------------------------------------------------------------------ the physics
def test_one_phase_cycle_is_the_l_band_half_wavelength():
    """119.2 mm per cycle at L-band against 27.7 mm at Sentinel-1 C-band. NISAR is ~4.3x
    coarser per fringe - the trade for staying coherent where C-band does not."""
    assert ng.phase_to_displacement_mm(2.0 * math.pi) == pytest.approx(119.25, abs=0.1)
    c_band_mm_per_cycle = 0.0555 / 2.0 * 1000.0
    ratio = (ng.NISAR_WAVELENGTH_M / 2.0 * 1000.0) / c_band_mm_per_cycle
    assert 4.0 < ratio < 4.6


def test_phase_to_displacement_is_linear_and_signed():
    assert ng.phase_to_displacement_mm(0.0) == 0.0
    a = ng.phase_to_displacement_mm(1.0)
    assert ng.phase_to_displacement_mm(-1.0) == pytest.approx(-a)
    assert ng.phase_to_displacement_mm(2.0) == pytest.approx(2 * a)


def test_the_unverified_sign_convention_is_declared_everywhere():
    """The C-band path once applied a textbook sign to a raster HyP3 had already
    sign-corrected, inverting every velocity at a perfectly plausible magnitude. Until a
    NISAR README is quoted the same way, nothing may present this as settled."""
    assert ng.describe_capability()["sign_convention_verified"] is False
    assert "NOT YET VERIFIED" in ng.phase_to_displacement_mm.__doc__.upper()


# ------------------------------------------------------------------ capability
def test_capability_reports_the_blocker_when_credentials_are_absent(monkeypatch):
    monkeypatch.setattr(ng, "_auth_token", lambda: "")
    cap = ng.describe_capability()
    assert cap["credentials_present"] is False
    assert "EARTHDATA_TOKEN" in cap["blocker"]


def test_nisar_costs_no_credits():
    """The point of GUNW over HyP3: NASA ships the unwrapped interferogram already made,
    so there is no job, no hour-long wait and no 8000-credit monthly ceiling."""
    assert ng.describe_capability()["costs_credits"] is False


# ------------------------------------------------------------------ geolocation
def _axes(x0=126760.0, dx=80.0, nx=4311, y0=3347240.0, dy=-80.0, ny=4230):
    import numpy as np
    return np.arange(nx) * dx + x0, np.arange(ny) * dy + y0


def test_the_projection_is_derived_from_the_file_not_the_longitude(monkeypatch):
    """Taking the UTM zone from longitude would be an assumption, and a granule
    straddling a zone boundary would land in the wrong one silently. Measured on real
    granules: langtang (85.5 E) resolves to EPSG:32645 and glc_105 (82.25 E) to 32644."""
    xs, ys = _axes()
    monkeypatch.setattr(ng, "_read_axis",
                        lambda url, layer: xs if layer == ng.LAYER_X else ys)
    where = ng.locate("http://x", 28.232, 85.518)
    assert where is not None
    assert where["epsg"] == 32645
    assert where["res_m"] == 80.0
    assert 0 <= where["row"] < 4230 and 0 <= where["col"] < 4311


def test_a_point_outside_every_candidate_zone_is_refused(monkeypatch):
    xs, ys = _axes()
    monkeypatch.setattr(ng, "_read_axis",
                        lambda url, layer: xs if layer == ng.LAYER_X else ys)
    assert ng.locate("http://x", -33.9, 151.2) is None      # Sydney


def test_unreadable_coordinate_arrays_yield_none_not_a_guess(monkeypatch):
    def _boom(url, layer):
        raise OSError("range read failed")

    monkeypatch.setattr(ng, "_read_axis", _boom)
    assert ng.locate("http://x", 28.232, 85.518) is None


# ------------------------------------------------------------------ refusal paths
def test_measurement_refuses_when_the_point_cannot_be_located(monkeypatch):
    monkeypatch.setattr(ng, "locate", lambda *a, **k: None)
    assert ng.measure_double_difference("http://x", 28.0, 85.0) is None


def test_measurement_refuses_when_the_window_leaves_the_granule(monkeypatch):
    # a point one pixel from the edge cannot carry a 2500 m reference annulus
    monkeypatch.setattr(ng, "locate", lambda *a, **k: {
        "epsg": 32645, "row": 1, "col": 1, "res_m": 80.0, "height": 4230, "width": 4311})
    assert ng.measure_double_difference("http://x", 28.0, 85.0) is None


def test_search_returns_nothing_rather_than_raising_when_cmr_fails(monkeypatch):
    def _boom(*a, **k):
        raise OSError("CMR unreachable")

    monkeypatch.setattr(ng.urllib.request, "urlopen", _boom)
    assert ng.search_gunw(28.232, 85.518) == []
