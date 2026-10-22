"""Train & Evaluate the Hybrid Real-Data + Physics-Synthesized + Differential Corrector Model.

Combines:
  1. Raw Real Data (949 points from NASA GLC + Copernicus 30m DEM + GEM Active Faults)
  2. Physics-Synthesized Variables (`soil_depth_z_m`, `c_prime_kpa`, `twi`, `static_fs`, `physics_failure_prob`)
  3. Elevation-Stratum Inverse-Density Weighting (counters human reporting bias at high elevations)
  4. Differential Physics Offset Corrector & Neighbor-Block Granulation for data-scarce blocks.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.susceptibility import LandslideSusceptibilityModel


def main() -> None:
    model = LandslideSusceptibilityModel(random_state=42)
    report = model.fit_and_evaluate()
    m = report["metrics"]

    print("=== Hybrid Real-Data + Physics + Differential Calibration (0.3° Spatial Holdout) ===")
    print(f"Trained on Real Data:     {model.trained_on_real_data}")
    print(f"Train / Test Points:      {m['train_samples']} / {m['test_samples']}")
    print(f"Calibrated 0.3° Blocks:   {m.get('calibrated_blocks_count', 0)}")
    print(f"Significant Anomaly Blks: {m.get('significant_anomaly_blocks', 0)}")
    print(f"Accuracy:                 {m['accuracy']:.3f}")
    print(f"Precision:                {m['precision']:.3f}")
    print(f"Recall:                   {m['recall']:.3f}")
    print(f"F1 Score:                 {m['f1_score']:.3f}")
    print(f"ROC-AUC:                  {m['roc_auc']:.3f}")

    print("\n=== Hybrid Feature Importances (Raw + Physics + Differential) ===")
    for name, imp in sorted(report.get("feature_importances", {}).items(), key=lambda x: -x[1]):
        print(f"  {name:30s} {imp:.4f}")

    # Demonstrate Differential Calibration on a Data-Rich Block vs Data-Scarce High-Mountain Block
    rich_eval = model.evaluate_point_hybrid(
        latitude=27.95,
        longitude=84.22,
        elevation_m=1450.0,
        slope_deg=34.0,
        profile_curvature=-0.65,
        local_relief_m=245.0,
        fault_distance_km=12.0,
    )
    scarce_high_mountain = model.evaluate_point_hybrid(
        latitude=28.23,
        longitude=85.55,
        elevation_m=5450.0,
        slope_deg=43.0,
        profile_curvature=-0.45,
        local_relief_m=480.0,
        fault_distance_km=4.2,
    )

    print("\n=== Case 1: Data-Rich Mid-Hill Block (Direct Real-Data Differential Calibration) ===")
    print(f"  Mode:            {rich_eval['differential_calibration']['data_coverage_mode']}")
    print(f"  Real Samples:    {rich_eval['differential_calibration']['supporting_real_samples']}")
    print(f"  Diff Anomaly:    {rich_eval['differential_calibration']['block_differential_anomaly']:+.4f}")
    print(f"  Raw Physics FS -> Calibrated FS: {rich_eval['raw_physics_fs']:.3f} -> {rich_eval['calibrated_physics_fs']:.3f}")
    print(f"  Hybrid Risk:     {rich_eval['hybrid_susceptibility_score']:.3f}")

    print("\n=== Case 2: Data-Scarce High-Mountain Block (Neighbor-Block Granulated Simulation) ===")
    print(f"  Mode:            {scarce_high_mountain['differential_calibration']['data_coverage_mode']}")
    print(f"  Real Samples:    {scarce_high_mountain['differential_calibration']['supporting_real_samples']}")
    print(f"  Borrowed Anom:   {scarce_high_mountain['differential_calibration']['block_differential_anomaly']:+.4f}")
    print(f"  Raw Physics FS -> Calibrated FS: {scarce_high_mountain['raw_physics_fs']:.3f} -> {scarce_high_mountain['calibrated_physics_fs']:.3f}")
    print(f"  Calibrated z/c': z={scarce_high_mountain['differential_calibration']['calibrated_z_m']}m, c'={scarce_high_mountain['differential_calibration']['calibrated_c_prime_kpa']} kPa")
    print(f"  Hybrid Risk:     {scarce_high_mountain['hybrid_susceptibility_score']:.3f}")


if __name__ == "__main__":
    main()