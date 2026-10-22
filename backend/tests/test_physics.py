"""Unit tests for Infinite Slope Stability FS and Voellmy debris-flow runout routing."""

import numpy as np

from app.physics.feature_synthesizer import (
    RawEarthObservationInputs,
    synthesize_physics_variables,
)
from app.physics.historical_replay import replay_historical_cascade
from app.physics.infinite_slope import (
    classify_stability,
    compute_infinite_slope_fs,
    compute_permafrost_effective_cohesion,
    compute_probabilistic_fs,
    estimate_regolith_depth,
)
from app.physics.runout_routing import simulate_debris_runout


def test_infinite_slope_exact_equation():
    """Verify exact mathematical match with the AGENTS.md Infinite Slope FS formula."""
    c_prime = 12.0
    gamma = 20.0
    gamma_w = 9.81
    z = 2.5
    z_w = 1.0
    beta_deg = 32.0
    phi_deg = 30.0

    beta = np.deg2rad(beta_deg)
    phi = np.deg2rad(phi_deg)
    expected_fs = (
        c_prime + (gamma * z - gamma_w * z_w) * (np.cos(beta) ** 2) * np.tan(phi)
    ) / (gamma * z * np.sin(beta) * np.cos(beta))

    computed_fs = compute_infinite_slope_fs(
        c_prime=c_prime,
        gamma=gamma,
        z=z,
        z_w=z_w,
        beta_deg=beta_deg,
        phi_prime_deg=phi_deg,
        gamma_w=gamma_w,
    )
    assert np.isclose(computed_fs, expected_fs, rtol=1e-7)


def test_infinite_slope_saturation_and_seismic_degradation():
    """Increasing water table z_w or seismic acceleration k_h must decrease FS."""
    dry_fs = compute_infinite_slope_fs(
        c_prime=12.0, gamma=19.0, z=3.0, z_w=0.0, beta_deg=38.0, phi_prime_deg=31.0
    )
    saturated_fs = compute_infinite_slope_fs(
        c_prime=12.0, gamma=19.0, z=3.0, z_w=3.0, beta_deg=38.0, phi_prime_deg=31.0
    )
    seismic_fs = compute_infinite_slope_fs(
        c_prime=12.0,
        gamma=19.0,
        z=3.0,
        z_w=3.0,
        beta_deg=38.0,
        phi_prime_deg=31.0,
        seismic_kh=0.18,
    )
    assert dry_fs > 1.0
    assert saturated_fs < dry_fs
    assert seismic_fs < saturated_fs
    assert seismic_fs < 1.0

    summary = classify_stability(np.array([dry_fs, saturated_fs, seismic_fs]))
    assert summary["stability_regime"] == "CRITICAL_FAILURE"
    assert summary["critical_cell_count"] >= 1


def test_invalid_slab_depth_raises():
    """Zero or negative slab thickness z must raise ValueError."""
    try:
        compute_infinite_slope_fs(
            c_prime=10.0, gamma=20.0, z=0.0, z_w=0.0, beta_deg=30.0, phi_prime_deg=30.0
        )
        raise AssertionError("Expected ValueError for z=0.0")
    except ValueError as exc:
        assert "Slab thickness" in str(exc)


def test_debris_runout_routing_down_valley():
    """Verify debris flow propagates downslope from FS < 1.0 release cells."""
    rows, cols = 15, 15
    r_grid, c_grid = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    # Steep inclined valley sloping toward higher row index
    dem = 4500.0 - 25.0 * r_grid + 5.0 * np.abs(c_grid - 7)
    source = np.zeros((rows, cols), dtype=bool)
    source[1, 7] = True

    result = simulate_debris_runout(
        dem_m=dem,
        source_mask=source,
        cell_size_m=30.0,
        basal_friction_mu=0.12,
    )
    assert result["runout_length_m"] > 100.0
    assert result["peak_velocity_m_s"] > 2.0
    assert result["affected_cells"] > 5


def test_permafrost_cohesion_and_historical_replay():
    """Verify high-altitude permafrost thaw reduces cohesion and historical replays escalate to RED."""
    frozen_c = compute_permafrost_effective_cohesion(8.0, elevation_m=5500.0, pdd_c_days=0.0)
    thawed_c = compute_permafrost_effective_cohesion(8.0, elevation_m=5500.0, pdd_c_days=12.0)
    assert frozen_c > thawed_c

    gentle_z = estimate_regolith_depth(10.0)
    steep_z = estimate_regolith_depth(55.0)
    assert gentle_z > steep_z

    prob = compute_probabilistic_fs(
        c_prime=5.0, gamma=20.0, z=2.2, z_w=2.0, beta_deg=42.0, phi_prime_deg=29.0, seismic_kh=0.20
    )
    assert prob["failure_probability"] > 0.80

    langtang = replay_historical_cascade("langtang_2015")
    assert langtang["timeline"][0]["alert_level"] == "GREEN"
    assert langtang["timeline"][-1]["alert_level"] == "RED"
    assert langtang["timeline"][-1]["deterministic_fs"] < 1.0

    monsoon = replay_historical_cascade("budhi_gandaki_monsoon")
    assert monsoon["timeline"][0]["alert_level"] == "GREEN"
    assert monsoon["timeline"][-1]["alert_level"] == "RED"


def test_multisource_feature_synthesizer_and_generalized_fs():
    """Verify 18-variable raw EO synthesizer feeds Generalized Infinite Slope FS accurately."""
    # Forested mid-elevation slope vs barren fault-damaged high-altitude thawing slope
    forested_raw = RawEarthObservationInputs(
        elevation_m=2800.0,
        slope_deg=32.0,
        sentinel2_ndvi=0.68,
        gedi_canopy_height_m=18.0,
        distance_to_fault_km=12.0,
        imerg_rain_72h_mm=30.0,
        antecedent_rain_30d_api_mm=20.0,
        bedrock_pga_g=0.0,
    )
    barren_fault_raw = RawEarthObservationInputs(
        elevation_m=5200.0,
        slope_deg=41.0,
        sentinel2_ndvi=0.02,
        gedi_canopy_height_m=0.0,
        distance_to_fault_km=0.4,
        diurnal_freeze_thaw_cycles_30d=25,
        pdd_c_days_7d=14.0,
        imerg_rain_72h_mm=160.0,
        antecedent_rain_30d_api_mm=140.0,
        snow_water_equivalent_mm=320.0,
        bedrock_pga_g=0.14,
    )

    f_syn = synthesize_physics_variables(forested_raw)
    b_syn = synthesize_physics_variables(barren_fault_raw)

    assert f_syn["c_root_kpa"] > 4.0
    assert b_syn["c_root_kpa"] == 0.0
    assert b_syn["seismic_kh"] > barren_fault_raw.bedrock_pga_g  # Topographic ridge amplification

    fs_forested = float(
        compute_infinite_slope_fs(
            c_prime=f_syn["c_prime_kpa"],
            gamma=f_syn["gamma_kn_m3"],
            z=f_syn["z_m"],
            z_w=f_syn["z_w_m"],
            beta_deg=forested_raw.slope_deg,
            phi_prime_deg=f_syn["phi_prime_deg"],
            seismic_kh=f_syn["seismic_kh"],
            surface_surcharge_q_kpa=f_syn["surface_surcharge_q_kpa"],
            matric_suction_kpa=f_syn["matric_suction_kpa"],
        )
    )
    fs_barren = float(
        compute_infinite_slope_fs(
            c_prime=b_syn["c_prime_kpa"],
            gamma=b_syn["gamma_kn_m3"],
            z=b_syn["z_m"],
            z_w=b_syn["z_w_m"],
            beta_deg=barren_fault_raw.slope_deg,
            phi_prime_deg=b_syn["phi_prime_deg"],
            seismic_kh=b_syn["seismic_kh"],
            surface_surcharge_q_kpa=b_syn["surface_surcharge_q_kpa"],
            matric_suction_kpa=b_syn["matric_suction_kpa"],
        )
    )
    assert fs_forested > 1.25
    assert fs_barren < 1.0
