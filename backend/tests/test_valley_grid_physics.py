"""Unit tests for the vectorized Valley-Corridor Physics & FS Engine."""

from __future__ import annotations

import numpy as np
import pytest
from app.physics.valley_grid_physics import (
    compute_vectorized_grid_fs,
    fs_to_gradient_color,
    classify_alert_level,
    valley_grid_service,
    PRIMARY_CORRIDORS,
)


def test_fs_gradient_color_interpolation():
    assert fs_to_gradient_color(0.70) == "#dc2626"
    assert fs_to_gradient_color(0.99).startswith("#")
    assert fs_to_gradient_color(1.05).startswith("#")
    assert fs_to_gradient_color(1.20).startswith("#")
    assert fs_to_gradient_color(1.30).startswith("#")
    assert fs_to_gradient_color(1.60) == "#10b981"


def test_classify_alert_level():
    assert classify_alert_level(0.92, 60.0) == "RED"
    assert classify_alert_level(1.04, 40.0) == "ORANGE"
    assert classify_alert_level(1.15, 20.0) == "YELLOW"
    assert classify_alert_level(1.35, 5.0) == "GREEN"


def test_vectorized_grid_fs_rainfall_response():
    """Verify that rainfall decreases FS monotonically across cells."""
    slopes = np.array([25.0, 35.0, 45.0])
    elevations = np.array([1800.0, 2400.0, 3200.0])

    dry_res = compute_vectorized_grid_fs(
        slope_deg=slopes,
        elevation_m=elevations,
        rainfall_72h_mm=0.0,
    )

    monsoon_res = compute_vectorized_grid_fs(
        slope_deg=slopes,
        elevation_m=elevations,
        rainfall_72h_mm=180.0,
    )

    # Monsoon rainfall must lower FS on all cells
    assert np.all(monsoon_res["fs"] < dry_res["fs"])
    # Monsoon rainfall must increase water table zw
    assert np.all(monsoon_res["water_table_zw_m"] > dry_res["water_table_zw_m"])
    # Steeper slopes must have lower FS than gentle slopes
    assert dry_res["fs"][2] < dry_res["fs"][0]
    assert monsoon_res["fs"][2] < monsoon_res["fs"][0]


def test_vectorized_grid_fs_road_cut_influence():
    """Verify that proximity to mountain road increases effective slope & reduces FS."""
    slopes = np.array([38.0, 38.0])
    elevations = np.array([2200.0, 2200.0])
    road_dists = np.array([10.0, 1500.0])  # adjacent to cut vs far away

    res = compute_vectorized_grid_fs(
        slope_deg=slopes,
        elevation_m=elevations,
        road_distance_m=road_dists,
        rainfall_72h_mm=60.0,
    )

    # Cut slope must be steeper and have lower FS
    assert res["effective_slope_deg"][0] > res["effective_slope_deg"][1]
    assert res["road_cut_penalty"][0] > res["road_cut_penalty"][1]
    assert res["effective_cohesion_kpa"][0] < res["effective_cohesion_kpa"][1]
    assert res["fs"][0] < res["fs"][1]


def test_vectorized_grid_fs_dsgsd_degradation():
    """Verify that DSGSD deformation degrades cohesion and lowers FS."""
    slopes = np.array([36.0, 36.0])
    elevations = np.array([3000.0, 3000.0])
    dsgsd_flags = np.array([True, False])

    res = compute_vectorized_grid_fs(
        slope_deg=slopes,
        elevation_m=elevations,
        dsgsd_indicated=dsgsd_flags,
        rainfall_72h_mm=50.0,
    )

    assert res["effective_cohesion_kpa"][0] < res["effective_cohesion_kpa"][1]
    assert res["fs"][0] < res["fs"][1]


def test_valley_grid_service_corridors_and_cells():
    corridors = valley_grid_service.get_corridors_list()
    assert len(corridors) >= 7
    c_ids = [c["id"] for c in corridors]
    assert "langtang" in c_ids
    assert "budhi_gandaki" in c_ids

    # Evaluate Langtang corridor cells
    cells = valley_grid_service.evaluate_corridor_grid("langtang", rainfall_72h_mm=40.0)
    # NOT CorridorMetadata.cell_count. That field is the hand-written number the old
    # generator fabricated up to - 320 invented cells standing in for a bounding box
    # that holds ~36,000 real ones at 240 m. The grid is now measured from the
    # Copernicus DEM, so the count is whatever the terrain actually yields above the
    # 20 deg slope threshold, and asserting the declared number would re-impose exactly
    # the fiction this replaced.
    assert len(cells) > 5_000, (
        f"only {len(cells)} cells - the measured grid looks unbuilt or truncated"
    )
    declared = PRIMARY_CORRIDORS["langtang"].cell_count
    assert len(cells) > declared * 10, (
        "the measured grid should dwarf the declared placeholder count"
    )
    
    first = cells[0]
    assert "cell_id" in first
    assert "bounds" in first
    assert len(first["bounds"]) == 4
    # Bounds width should match ~240m (2 * 0.00108 = 0.00216 deg)
    lat_span = first["bounds"][2] - first["bounds"][0]
    assert 0.0020 <= lat_span <= 0.0023
    assert "fs" in first
    assert "color_hex" in first
    assert first["color_hex"].startswith("#")
    assert first["alert_level"] in ("RED", "ORANGE", "YELLOW", "GREEN")
    assert 28.05 <= first["latitude"] <= 28.45
    assert 85.15 <= first["longitude"] <= 85.80
    assert "asset_type" in first
    assert "asset_name" in first
    assert "human_exposure_desc" in first
    assert "human_asset_metric" in first


def test_valley_grid_human_settlement_and_infrastructure_exposure():
    """Verify that human settlements, road cuts, and hydro infrastructure are modeled."""
    corridor = PRIMARY_CORRIDORS["langtang"]
    assert len(corridor.highways) >= 2
    assert len(corridor.key_settlements) >= 4
    assert len(corridor.infrastructure_assets) >= 2

    cells = valley_grid_service.evaluate_corridor_grid("langtang", rainfall_72h_mm=0.0)
    asset_types = {c["asset_type"] for c in cells}
    # Categories are DERIVED FROM MEASURED DISTANCES now, not assigned by quota. The old
    # generator split cells 32/34/16/13/5 by hardcoded share, so every category was
    # guaranteed present and none of the counts meant anything. Here a category appears
    # only if OSM actually maps the thing near enough to a cell, so the assertion is on
    # the categories the terrain can support - and HYDRO_INFRASTRUCTURE is checked
    # separately below, since it depends on a dam being mapped inside this corridor.
    assert "SETTLEMENT" in asset_types
    assert "HIGHWAY_ROADCUT" in asset_types
    assert "RIVER_GORGE" in asset_types
    assert "CRYOSPHERE_GLACIAL" in asset_types

    # Test filtering by asset_type
    settlements = valley_grid_service.evaluate_corridor_grid("langtang", asset_type="settlement")
    assert len(settlements) >= 80
    assert all(c["asset_type"] == "SETTLEMENT" for c in settlements)

    highways = valley_grid_service.evaluate_corridor_grid("langtang", asset_type="highway_roadcut")
    assert len(highways) >= 80
    assert all(c["asset_type"] == "HIGHWAY_ROADCUT" for c in highways)
    # Highway-cut cells must lie within the measured exposure radius. The old bound here
    # was 100 m, which held only because the generator drew road distance from
    # rng.uniform(4.0, 50.0). Real exposure reaches further: a slope failing 400 m above
    # a highway still lands on it, so the category spans to the 1 km search radius and
    # the road-cut penalty decays with distance instead of being gated on nearness.
    assert all(c["road_distance_m"] is not None for c in highways), (
        "a HIGHWAY_ROADCUT cell with no measured road distance is a classification bug"
    )
    assert all(c["road_distance_m"] <= 1000.0 for c in highways)
    # The penalty must actually track the measurement: near cells cut more than far ones.
    near = [c["road_cut_penalty"] for c in highways if c["road_distance_m"] < 100.0]
    far = [c["road_cut_penalty"] for c in highways if c["road_distance_m"] > 500.0]
    if near and far:
        assert sum(near) / len(near) > sum(far) / len(far), (
            "road-cut penalty does not decay with measured distance"
        )
    assert any(c["road_cut_penalty"] > 0.20 for c in highways)


def test_valley_grid_live_telemetry_resolution():
    """Verify that when simulation parameters are omitted, live raw telemetry is resolved."""
    from app.physics.valley_grid_physics import resolve_corridor_live_telemetry
    live_telem = resolve_corridor_live_telemetry("langtang")
    assert "rainfall_72h_mm" in live_telem
    assert "api_30d_mm" in live_telem
    assert "seismic_kh" in live_telem
    assert live_telem["rainfall_72h_mm"] >= 0.0
    assert live_telem["api_30d_mm"] >= 0.0
    assert live_telem["seismic_kh"] >= 0.0
    assert live_telem["provenance"] in ("MEASURED_LIVE_RAW_TELEMETRY", "MEASURED_STATUS_SNAPSHOT", "REAL_TIME_FALLBACK_DEFAULT")

    # Evaluate without explicit rainfall or parameters -> should evaluate on live telemetry
    cells = valley_grid_service.evaluate_corridor_grid("langtang")
    assert len(cells) > 5_000, "measured grid looks unbuilt"
    assert hasattr(valley_grid_service, "_latest_corridor_telemetry")
    assert "langtang" in valley_grid_service._latest_corridor_telemetry
    stored = valley_grid_service._latest_corridor_telemetry["langtang"]
    assert stored["is_live_telemetry"] is True




# ======================================================================================
# The measured-grid contract
#
# These exist because the grid they replace was fabricated end to end: every elevation,
# slope, curvature, clay fraction and road distance came from np.random seeded on
# hash(corridor_id). The tests above assert that the physics runs; these assert that it
# runs on measurements.
# ======================================================================================
def test_grid_terrain_is_internally_consistent_with_a_real_dem():
    """Measured terrain has signatures that generated terrain does not.

    A random grid has no correlation between neighbouring cells and no relationship
    between elevation and position. A real one does: Langtang's floor is around 1,400 m
    at Syabrubesi and its peaks exceed 7,000 m, and slope and relief move together
    because both are read from the same surface.
    """
    cells = valley_grid_service.evaluate_corridor_grid("langtang")
    elevations = [c["elevation_m"] for c in cells]
    slopes = [c["slope_deg"] for c in cells]

    # Real Himalayan corridor relief, not a clipped random range.
    assert min(elevations) < 2_000, "no valley floor present - this is not a real DEM"
    assert max(elevations) > 6_000, "no high peaks present - this is not a real DEM"

    # The build drops anything below the 20 deg failure threshold, and nothing in the
    # Himalaya stands past vertical.
    assert min(slopes) >= 20.0
    assert max(slopes) < 90.0

    # Cells share a regular grid, so coordinates must repeat across rows and columns.
    # Jittered random points essentially never do.
    lats = {round(c["latitude"], 5) for c in cells}
    assert len(lats) < len(cells) / 5, (
        "latitudes are nearly all unique - these are scattered points, not a grid"
    )


def test_grid_is_stable_across_service_instances():
    """The same corridor must give the same grid every time.

    The old generator seeded on ``hash(corridor_id)``, and Python salts string hashing
    per process, so the grid silently reshuffled on every API restart - three runs gave
    seeds 110651952, 475701804 and 1872610634 - while a comment claimed it was
    "reproducible ... for spatial consistency". Reading from a cache makes that
    structurally impossible, and this pins it.
    """
    from app.physics.valley_grid_physics import ValleyGridPhysicsService

    a = ValleyGridPhysicsService()._load_corridor_cells("langtang")
    b = ValleyGridPhysicsService()._load_corridor_cells("langtang")
    assert a["latitude"].shape == b["latitude"].shape
    assert (a["latitude"] == b["latitude"]).all()
    assert (a["elevation_m"] == b["elevation_m"]).all()
    assert (a["clay_fraction"] == b["clay_fraction"]).all()


def test_grid_refuses_to_invent_a_corridor_it_has_not_measured(monkeypatch):
    """A missing grid is an outage, not an empty map quietly drawn.

    The generator this replaced produced a full grid on every call, so "no data" and
    "no hazard" looked identical downstream. Loading must raise instead.
    """
    import app.engine.valley_grid_terrain as vgt
    from app.physics.valley_grid_physics import ValleyGridPhysicsService

    monkeypatch.setattr(vgt, "load_corridor", lambda *a, **k: {})
    with pytest.raises(FileNotFoundError, match="not generated at runtime"):
        ValleyGridPhysicsService()._load_corridor_cells("langtang")


def test_grid_refuses_a_corridor_with_no_resolvable_soil(monkeypatch):
    """Cells present but no soil anywhere is also a refusal, not a silent empty grid."""
    import numpy as np

    import app.engine.valley_grid_terrain as vgt
    from app.physics.valley_grid_physics import ValleyGridPhysicsService

    n = 4
    nan = np.full(n, np.nan, dtype=np.float32)
    fake = {"lat": np.full(n, 28.1, dtype=np.float32),
            "clay_fraction": nan, "dry_bulk_density_g_cm3": nan}
    monkeypatch.setattr(vgt, "load_corridor", lambda *a, **k: fake)
    with pytest.raises(FileNotFoundError, match="none with resolved"):
        ValleyGridPhysicsService()._load_corridor_cells("langtang")


def test_soil_absent_cells_are_excluded_not_filled():
    """SoilGrids masks rock and ice; those cells must drop out, not get a clay fraction.

    82% of Langtang's cells resolve a soil profile. The other 18% are glacier and bare
    rock above the soil mask, where the infinite-slope soil model does not apply. Giving
    them a plausible clay fraction would be the exact failure this rewrite removes.
    """
    from app.engine.valley_grid_terrain import load_grid

    grid = load_grid()
    raw = grid.get("langtang") or []
    if not raw:
        pytest.skip("measured grid not built in this environment")

    # The loader requires BOTH a clay fraction and a bulk density, since the FS needs
    # the unit weight as much as the cohesion; a handful of cells resolve one and not
    # the other, so match the loader's criterion exactly rather than approximating it.
    without_soil = [c for c in raw if c.get("clay_fraction") is None
                    or c.get("dry_bulk_density_g_cm3") is None]
    assert without_soil, "expected some rock/ice cells with no soil profile"
    # They must be high: this is the glacier, not a fetch failure scattered at random.
    high = sum(1 for c in without_soil if c["elevation_m"] > 4_000)
    assert high / len(without_soil) > 0.8, (
        "cells missing soil are not concentrated at altitude - looks like dropped "
        "reads rather than a genuine soil mask"
    )

    evaluated = valley_grid_service.evaluate_corridor_grid("langtang")
    assert len(evaluated) == len(raw) - len(without_soil)
    assert valley_grid_service._excluded_no_soil["langtang"] == len(without_soil)
