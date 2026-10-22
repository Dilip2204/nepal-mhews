"""Unit tests for Tier A ML Susceptibility and Tier B Temporal Runoff Anomaly."""

import numpy as np

from app.ml.moisture_downscaler import downscale_smap_moisture_and_regolith_to_30m
from app.ml.pinn_inversion import PINNCohesionDepthInverter
from app.ml.runoff_anomaly import TemporalRunoffAnomalyDetector
from app.ml.susceptibility import (
    LandslideSusceptibilityModel,
    generate_synthetic_basin_training_data,
)


def test_susceptibility_model_strict_split_and_missing_imputation():
    """Verify susceptibility pipeline handles NaNs and achieves strong ROC-AUC on held-out set."""
    X, y = generate_synthetic_basin_training_data(n_samples=400, random_state=42)
    # Inject missing values to verify imputation pipeline
    X[0, 0] = np.nan
    X[15, 3] = np.nan

    model = LandslideSusceptibilityModel(random_state=42)
    report = model.fit_and_evaluate(X, y)

    assert report["missing_value_rates"]["slope_deg"] > 0.0
    assert report["metrics"]["roc_auc"] > 0.75

    preds = model.predict_susceptibility(X[:5])
    assert preds.shape == (5,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_temporal_runoff_anomaly_detector():
    """Verify chronological fit and anomaly detection on extreme monsoon + melt surge."""
    detector = TemporalRunoffAnomalyDetector()
    fit_stats = detector.fit_chronological()
    assert fit_stats["train_steps"] > fit_stats["val_steps"]

    normal_obs = detector.score_observation(
        rainfall_mm=12.0, pdd_c_days=3.5, discharge_m3s=85.0
    )
    extreme_obs = detector.score_observation(
        rainfall_mm=185.0, pdd_c_days=16.0, discharge_m3s=950.0
    )
    assert extreme_obs["anomaly_z_score"] > normal_obs["anomaly_z_score"]
    assert extreme_obs["is_anomaly"] is True


def test_real_nepal_hybrid_differential_and_neighbor_granulation():
    """Verify real 949-point dataset training, differential block calibration, and scarce-block granulation."""
    model = LandslideSusceptibilityModel(random_state=42)
    report = model.fit_and_evaluate()

    assert model.trained_on_real_data is True
    assert report["metrics"]["roc_auc"] > 0.80
    assert report["metrics"]["calibrated_blocks_count"] > 40

    # 1. Evaluate a block with real data present (Annapurna/Kaski block (94, 279) with 23 real points)
    rich = model.evaluate_point_hybrid(
        latitude=28.25,
        longitude=83.85,
        elevation_m=1450.0,
        slope_deg=34.0,
        profile_curvature=-0.65,
        local_relief_m=245.0,
        fault_distance_km=12.0,
    )
    assert rich["differential_calibration"]["supporting_real_samples"] >= 4
    assert (
        rich["differential_calibration"]["data_coverage_mode"]
        == "DIRECT_REAL_DATA_BLOCK_CALIBRATION"
    )

    # 2. Verify high-mountain Langtang block (28.232, 85.518) is now anchored by cryosphere inventory
    langtang_high = model.evaluate_point_hybrid(
        latitude=28.232,
        longitude=85.518,
        elevation_m=5680.0,
        slope_deg=46.5,
        profile_curvature=-0.62,
        local_relief_m=640.0,
        fault_distance_km=3.8,
    )
    assert (
        langtang_high["differential_calibration"]["data_coverage_mode"]
        == "DIRECT_REAL_DATA_BLOCK_CALIBRATION"
    )

    # 3. Verify unmapped high-mountain block (30.40, 81.90) uses neighbor granulation
    scarce = model.evaluate_point_hybrid(
        latitude=30.40,
        longitude=81.90,
        elevation_m=5650.0,
        slope_deg=44.0,
        profile_curvature=-0.50,
        local_relief_m=510.0,
        fault_distance_km=3.5,
    )
    assert scarce["differential_calibration"]["data_coverage_mode"] in (
        "NEIGHBOR_BLOCK_GRANULATED_SIMULATION",
        "SCARCE_DATA_HYBRID_GRANULATION",
    )
    assert 0.0 <= scarce["hybrid_susceptibility_score"] <= 1.0


def test_pinn_cohesion_depth_inverter_and_smap_downscaler():
    """Verify PINN inverts high InSAR creep into lower c'/FS and SMAP downscaler preserves mass."""
    inverter = PINNCohesionDepthInverter(n_epochs=45, random_state=42)
    train_stats = inverter.fit_pinn(n_collocation_points=120)
    assert train_stats["final_pinn_loss"] <= train_stats["initial_pinn_loss"]

    # High observed InSAR velocity (28 mm/month) implies slope is closer to failure than naive prior
    inv = inverter.invert_subsurface_parameters(
        slope_deg=38.0,
        phi_prime_deg=31.0,
        saturation_ratio=0.55,
        c_prior_kpa=16.0,
        z_prior_m=2.2,
        observed_insar_velocity_mm_month=28.0,
    )
    assert inv["pinn_inverted_fs"] < inv["prior_fs"]
    assert inv["inverted_c_prime_kpa"] < inv["prior_c_prime_kpa"]
    assert inv["creep_physics_residual_mm_month"] < 2.0

    # Test 9km -> 30m SMAP Soil Moisture & Regolith Downscaler mass conservation
    slope_grid = np.linspace(15.0, 45.0, 25).reshape(5, 5)
    aspect_grid = np.linspace(0.0, 180.0, 25).reshape(5, 5)
    twi_grid = np.linspace(11.0, 3.5, 25).reshape(5, 5)
    ndvi_grid = np.linspace(0.65, 0.10, 25).reshape(5, 5)
    dt_lst_grid = np.linspace(10.0, 22.0, 25).reshape(5, 5)

    ds_res = downscale_smap_moisture_and_regolith_to_30m(
        coarse_smap_vwc=0.28,
        coarse_regolith_depth_m=2.4,
        slope_deg_grid=slope_grid,
        aspect_deg_grid=aspect_grid,
        twi_grid=twi_grid,
        ndvi_grid=ndvi_grid,
        diurnal_lst_range_c_grid=dt_lst_grid,
    )
    assert np.isclose(ds_res["mean_downscaled_vwc"], 0.28, atol=0.02)
    assert ds_res["max_vwc_gully"] > ds_res["min_vwc_ridge"]


def test_live_ground_truth_feedback_loop_and_evolutionary_optimizer():
    """Verify Live Ground-Truth Event Feedback ingestion and Rule 8.4 Evolutionary Optimizer."""
    from app.ingestion.ground_truth_feedback import (
        ensure_feedback_catalog_initialized,
        load_live_ground_truth_events,
    )
    from app.ml.continuous_equation_optimizer import ContinuousEquationOptimizer

    ensure_feedback_catalog_initialized()
    events = load_live_ground_truth_events()
    assert len(events) >= 6

    opt = ContinuousEquationOptimizer()
    acc, rec, auc, f1, cryo_acc, fb_acc, avg_dev = opt.evaluate_parameter_set(opt.active_params)
    assert rec >= 0.9136
    assert cryo_acc >= 0.999
    assert fb_acc >= 0.85
    assert avg_dev <= 5.0

    step_res = opt.step_optimization_generation(n_candidates=2)
    assert step_res.dataset_total_points >= 979
    assert step_res.live_feedback_points >= 6
    assert step_res.recall >= 0.9136
    assert step_res.cryo_accuracy >= 0.999
