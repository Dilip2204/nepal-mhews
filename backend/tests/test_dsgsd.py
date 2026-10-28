"""Unit tests for Deep-Seated Gravitational Slope Deformation (DSGSD) physics and indicators."""

from __future__ import annotations

import pytest
from app.core.config import settings
from app.physics import dsgsd
from app.physics.feature_synthesizer import (
    RawEarthObservationInputs,
    synthesize_physics_variables,
)


def test_morphometric_indicator_passes_when_all_criteria_met():
    res = dsgsd.morphometric_indicator(
        valley_relief_m=650.0,
        slope_deg=35.0,
        valley_profile_curvature=-0.005,
        scarp_relief_above_m=420.0,
    )
    assert res["indicator"] == 1.0
    assert res["criteria_met"] == 4
    assert res["criteria_total"] == 4
    assert all(res["checks"].values())


def test_morphometric_indicator_fails_when_any_input_is_none():
    # Never guesses or fabricates missing measurements
    res = dsgsd.morphometric_indicator(
        valley_relief_m=None,
        slope_deg=35.0,
        valley_profile_curvature=-0.005,
        scarp_relief_above_m=420.0,
    )
    assert res["indicator"] == 0.0
    assert res["criteria_met"] == 3
    assert res["checks"]["valley_relief"] is False


def test_morphometric_indicator_fails_on_stable_alpine_plateau_controls():
    # Negative control anchor: gentle alpine plateau (<15 deg slope)
    res = dsgsd.morphometric_indicator(
        valley_relief_m=600.0,
        slope_deg=12.0,  # Below 32 deg threshold
        valley_profile_curvature=-0.002,
        scarp_relief_above_m=400.0,
    )
    assert res["indicator"] == 0.0
    assert res["checks"]["slope"] is False


def test_morphometric_indicator_fails_on_convex_ridge():
    # Convex profile (curvature > 0) is a ridge crest, not a concave toe / scarp base
    res = dsgsd.morphometric_indicator(
        valley_relief_m=700.0,
        slope_deg=38.0,
        valley_profile_curvature=0.008,
        scarp_relief_above_m=350.0,
    )
    assert res["indicator"] == 0.0
    assert res["checks"]["concave_flank"] is False


def test_dsgsd_physics_modifiers_at_zero_and_one():
    zero = dsgsd.dsgsd_physics_modifiers(0.0)
    assert zero["cohesion_multiplier"] == 1.0
    assert zero["friction_reduction_deg"] == 0.0
    assert zero["infiltration_multiplier"] == 1.0

    full = dsgsd.dsgsd_physics_modifiers(1.0)
    assert full["cohesion_multiplier"] == pytest.approx(1.0 - dsgsd.DSGSD_COHESION_REDUCTION)
    assert full["friction_reduction_deg"] == pytest.approx(dsgsd.DSGSD_FRICTION_REDUCTION_DEG)
    assert full["infiltration_multiplier"] == pytest.approx(1.0 + dsgsd.DSGSD_COUNTERSCARP_CAPTURE)


def test_resolve_dsgsd_unmapped_aoi_with_missing_terrain_is_absent():
    # Outside mapped AOI and no terrain raster
    out = dsgsd.resolve_dsgsd(lat=28.232, lon=85.518, slope_deg=36.0, valley_terrain=None)
    assert out["indicator"] == 0.0
    assert out["provenance"] == dsgsd.PROV_ABSENT
    assert out["detail"] == "TERRAIN_UNAVAILABLE"


def test_resolve_dsgsd_unmapped_aoi_inference_disabled():
    terrain = {
        "status": "MEASURED",
        "valley_relief_m": 800.0,
        "valley_profile_curvature": -0.01,
        "scarp_relief_above_m": 500.0,
    }
    out = dsgsd.resolve_dsgsd(
        lat=28.232,
        lon=85.518,
        slope_deg=36.0,
        valley_terrain=terrain,
        allow_inference=False,
    )
    assert out["indicator"] == 0.0
    assert out["provenance"] == dsgsd.PROV_ABSENT
    assert out["detail"] == "INFERENCE_DISABLED"


def test_resolve_dsgsd_unmapped_aoi_with_valid_morphometry_is_inferred():
    terrain = {
        "status": "MEASURED",
        "valley_relief_m": 800.0,
        "valley_profile_curvature": -0.01,
        "scarp_relief_above_m": 500.0,
        "source": "Copernicus_GLO_30",
        "window_radius_m": 2000.0,
    }
    out = dsgsd.resolve_dsgsd(
        lat=28.232,
        lon=85.518,
        slope_deg=36.0,
        valley_terrain=terrain,
        allow_inference=True,
    )
    assert out["indicator"] == 1.0
    assert out["provenance"] == dsgsd.PROV_INFERRED
    assert out["detail"] == "MORPHOMETRIC_RULE"
    assert "caveat" in out


def test_feature_synthesizer_respects_dsgsd_disabled_flag(monkeypatch):
    monkeypatch.setattr(settings, "dsgsd_enabled", False)
    raw = RawEarthObservationInputs(
        elevation_m=2500.0,
        slope_deg=36.0,
        dsgsd_indicator=1.0,
        dsgsd_provenance="MAPPED",
    )
    syn = synthesize_physics_variables(raw)
    assert syn["dsgsd_applied"] is False
    assert syn["dsgsd_cohesion_multiplier"] == 1.0
    assert syn["dsgsd_friction_reduction_deg"] == 0.0
    assert syn["dsgsd_infiltration_multiplier"] == 1.0


def test_feature_synthesizer_applies_dsgsd_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "dsgsd_enabled", True)
    raw_off = RawEarthObservationInputs(
        elevation_m=2500.0,
        slope_deg=36.0,
        dsgsd_indicator=0.0,
        dsgsd_provenance="ABSENT",
    )
    syn_off = synthesize_physics_variables(raw_off)

    raw_on = RawEarthObservationInputs(
        elevation_m=2500.0,
        slope_deg=36.0,
        dsgsd_indicator=1.0,
        dsgsd_provenance="MAPPED",
    )
    syn_on = synthesize_physics_variables(raw_on)

    assert syn_on["dsgsd_applied"] is True
    assert syn_on["dsgsd_cohesion_multiplier"] == pytest.approx(0.65)
    assert syn_on["dsgsd_friction_reduction_deg"] == pytest.approx(4.0)
    assert syn_on["dsgsd_infiltration_multiplier"] == pytest.approx(1.60)
    # Cohesion and friction angle should be degraded
    assert syn_on["c_prime_kpa"] < syn_off["c_prime_kpa"]
    assert syn_on["phi_prime_deg"] < syn_off["phi_prime_deg"]



# ---------------------------------------------------------------------------
# Mapped-inventory provenance.
#
# The coordinates below are REAL points in the Far-Western Nepal inventory
# (doi:10.5281/zenodo.4290100, CC-BY-4.0), not invented ones:
#
#   MAPPED_POINT        representative point of deep-seated polygon object_id 30
#                       (137,051 m2, vertical drop 376 m, interpreted 2011)
#   IN_AOI_NO_POLYGON   inside the mapped AOI, >2 km from any deep-seated polygon
#                       - the mappers looked here and marked nothing
#
# The inventory GeoJSON is a derived data product and is not committed, so these
# skip rather than fail where it is absent.
# ---------------------------------------------------------------------------

MAPPED_POINT = (28.937805, 80.876690)
IN_AOI_NO_POLYGON = (29.257534, 81.112142)
OUTSIDE_AOI = (28.232, 85.518)  # Langtang; far east of the Far-Western AOI

#: Terrain that satisfies all four morphometric criteria. Used to prove that the
#: morphometric rule is what fires outside the AOI, and that it is what gets
#: *suppressed* inside it.
PASSING_TERRAIN = {
    "status": "MEASURED",
    "valley_relief_m": 800.0,
    "valley_profile_curvature": -0.01,
    "scarp_relief_above_m": 500.0,
    "source": "Copernicus_GLO_30",
    "window_radius_m": 2000.0,
}


def _inventory_available() -> bool:
    from app.ingestion.dsgsd_inventory import inventory_status

    try:
        return bool(inventory_status().get("available"))
    except Exception:
        return False


requires_inventory = pytest.mark.skipif(
    not _inventory_available(),
    reason="mapped deep-seated inventory (zenodo.4290100) not present in this checkout",
)


@requires_inventory
def test_inside_mapped_polygon_is_observed_not_inferred():
    """A point inside a mapped polygon is MAPPED, and says which polygon."""
    lat, lon = MAPPED_POINT
    out = dsgsd.resolve_dsgsd(lat=lat, lon=lon, slope_deg=36.0, valley_terrain=None)

    assert out["indicator"] == 1.0
    assert out["provenance"] == dsgsd.PROV_MAPPED
    assert out["detail"] == "MAPPED_DEEP_SEATED_POLYGON"
    assert out["mapping_coverage"] == "IN_MAPPED_AOI"
    assert out["mapped_doi"] == "10.5281/zenodo.4290100"
    assert out["mapped_object_id"] is not None
    assert out["mapped_distance_m"] == 0.0
    # The class is 'deep-seated landslide', which is NOT the same landform as a
    # DSGSD. That difference must travel with the record, not be dropped.
    assert "deep-seated" in out["caveat"]
    # MAPPED terrain needs no DEM: the observation stands on its own.
    assert out["indicator"] == 1.0


@requires_inventory
def test_mapped_negative_is_absent_even_when_morphometry_would_fire():
    """Inside a mapped AOI the inventory wins; the terrain rule cannot overrule it.

    This is the asymmetry the module is built around. The SAME terrain dict that
    produces INFERRED outside the AOI must produce ABSENT inside it, because in
    there somebody actually looked.
    """
    lat, lon = IN_AOI_NO_POLYGON
    inside = dsgsd.resolve_dsgsd(
        lat=lat, lon=lon, slope_deg=36.0, valley_terrain=PASSING_TERRAIN
    )
    assert inside["mapping_coverage"] == "IN_MAPPED_AOI"
    assert inside["indicator"] == 0.0
    assert inside["provenance"] == dsgsd.PROV_ABSENT
    assert inside["detail"] == "MAPPED_NEGATIVE"

    # Same terrain, no mapping coverage -> the rule is allowed to speak.
    out_lat, out_lon = OUTSIDE_AOI
    outside = dsgsd.resolve_dsgsd(
        lat=out_lat, lon=out_lon, slope_deg=36.0, valley_terrain=PASSING_TERRAIN
    )
    assert outside["mapping_coverage"] != "IN_MAPPED_AOI"
    assert outside["provenance"] == dsgsd.PROV_INFERRED

    # The terrain was identical. Only the evidence differed.
    assert inside["indicator"] != outside["indicator"]


def test_provenance_labels_are_exactly_the_three_declared():
    """No fourth label may leak out of resolve_dsgsd()."""
    assert {dsgsd.PROV_MAPPED, dsgsd.PROV_INFERRED, dsgsd.PROV_ABSENT} == {
        "MAPPED",
        "INFERRED",
        "ABSENT",
    }


# ---------------------------------------------------------------------------
# Safety invariants.
# ---------------------------------------------------------------------------


def test_dsgsd_defaults_to_disabled():
    """Production default is OFF until the consequence numbers have been reviewed."""
    from app.core.config import Settings

    assert Settings().dsgsd_enabled is False


def test_modifiers_clamp_an_out_of_range_indicator():
    """An indicator outside [0, 1] must not produce an out-of-range strength cut."""
    assert dsgsd.dsgsd_physics_modifiers(5.0) == dsgsd.dsgsd_physics_modifiers(1.0)
    assert dsgsd.dsgsd_physics_modifiers(-3.0) == dsgsd.dsgsd_physics_modifiers(0.0)


def test_modifiers_only_ever_move_toward_instability():
    """Across the whole indicator range, every modifier points one way: weaker."""
    prev = dsgsd.dsgsd_physics_modifiers(0.0)
    for step in range(1, 11):
        cur = dsgsd.dsgsd_physics_modifiers(step / 10.0)
        assert cur["cohesion_multiplier"] <= prev["cohesion_multiplier"]
        assert cur["cohesion_multiplier"] <= 1.0
        assert cur["friction_reduction_deg"] >= prev["friction_reduction_deg"]
        assert cur["infiltration_multiplier"] >= prev["infiltration_multiplier"]
        prev = cur


def test_unlabelled_indicator_is_refused_not_applied(monkeypatch):
    """A strength reduction with no provenance is rejected, not applied quietly."""
    monkeypatch.setattr(settings, "dsgsd_enabled", True)
    syn = synthesize_physics_variables(
        RawEarthObservationInputs(
            elevation_m=2500.0,
            slope_deg=36.0,
            dsgsd_indicator=1.0,
            dsgsd_provenance="GUESS",
        )
    )
    assert syn["dsgsd_applied"] is False
    assert syn["dsgsd_provenance"] == "INVALID_PROVENANCE"
    assert syn["dsgsd_cohesion_multiplier"] == 1.0


def test_enabling_dsgsd_never_raises_the_factor_of_safety(monkeypatch):
    """The one invariant that matters for a warning system.

    Measured across the 491-site catalogue, enabling this factor changed the alert
    level at 22 of the 52 sites where it fires, and **zero** of them moved toward
    GREEN. That is a property of the physics, not an accident of the catalogue, so
    it is pinned here as a property: over a grid of slope / rain / elevation, FS
    with DSGSD applied is never greater than FS without it.

    If this test ever fails, the factor has acquired a path that makes a slope look
    safer than it did before - the one direction an early-warning system must not
    move on its own.
    """
    from app.physics.infinite_slope import compute_infinite_slope_fs

    def fs_for(raw: RawEarthObservationInputs) -> float:
        syn = synthesize_physics_variables(raw)
        return float(
            compute_infinite_slope_fs(
                c_prime=syn["c_prime_kpa"],
                gamma=syn["gamma_kn_m3"],
                z=syn["z_m"],
                z_w=syn["z_w_m"],
                beta_deg=syn["effective_slope_deg"],
                phi_prime_deg=syn["phi_prime_deg"],
                seismic_kh=syn["seismic_kh"],
                surface_surcharge_q_kpa=syn["surface_surcharge_q_kpa"],
                matric_suction_kpa=syn["matric_suction_kpa"],
            )
        )

    checked = 0
    for slope in (18.0, 27.0, 36.0, 45.0, 55.0):
        for rain in (0.0, 40.0, 120.0, 250.0):
            for elevation in (1200.0, 2500.0, 4200.0):
                kwargs = dict(
                    elevation_m=elevation,
                    slope_deg=slope,
                    imerg_rain_72h_mm=rain,
                    antecedent_rain_30d_api_mm=rain * 0.6,
                )
                monkeypatch.setattr(settings, "dsgsd_enabled", False)
                fs_off = fs_for(RawEarthObservationInputs(**kwargs))

                monkeypatch.setattr(settings, "dsgsd_enabled", True)
                fs_on = fs_for(
                    RawEarthObservationInputs(
                        **kwargs, dsgsd_indicator=1.0, dsgsd_provenance="MAPPED"
                    )
                )

                assert fs_on <= fs_off + 1e-9, (
                    f"DSGSD made the slope look SAFER at slope={slope} rain={rain} "
                    f"elev={elevation}: FS {fs_off:.4f} -> {fs_on:.4f}"
                )
                checked += 1

    assert checked == 60
