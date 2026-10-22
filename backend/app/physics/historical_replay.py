"""Historical Disaster Replay & Backtesting Engine for Himalayan-MHEWS.

Replays step-by-step telemetry timelines for two signature Himalayan hazard cascades:
1. `langtang_2015`: April 2015 Langtang Lirung co-seismic rock-ice avalanche cascade.
2. `budhi_gandaki_monsoon`: Compound monsoon rain-on-snow + Thulagi glacial lake surge.
"""

from typing import Dict, List
from app.ingestion.hydro_imerg import evaluate_hydromet_saturation
from app.ingestion.seismic_usgs import evaluate_seismic_trigger
from app.ingestion.stac_sentinel import evaluate_sar_creep_acceleration
from app.physics.infinite_slope import (
    compute_permafrost_effective_cohesion,
    compute_probabilistic_fs,
    estimate_regolith_depth,
)


def replay_historical_cascade(scenario_id: str = "langtang_2015") -> Dict[str, object]:
    """Run a chronological backtest replay showing how FS and alerts evolve step-by-step."""
    if scenario_id == "langtang_2015":
        basin_id = "langtang"
        title = "2015 Langtang Lirung Rock-Ice Avalanche & Co-Seismic Cascade"
        slope_deg = 45.0
        elevation_m = 5650.0
        base_cohesion_kpa = 4.0
        phi_deg = 32.0
        gamma = 20.5
        z_slab = 3.2  # Deep rock-ice detachment slab thickness (m)
        steps_raw = [
            {
                "timestamp": "T-14 Days (Mid-April Baseline)",
                "layman_summary": "Frozen high-altitude rock-ice face; permafrost ice-glue keeps FS safely above 1.25.",
                "pdd_c_days": 0.5,
                "rain_72h_mm": 12.0,
                "api_30d_mm": 15.0,
                "sar_disp_mm": [0.0, 1.5, 3.1],
                "seismic_mw": 2.1,
                "lake_exp_pct": 1.2,
            },
            {
                "timestamp": "T-3 Days (Pre-Event Thermal Thaw & Creep)",
                "layman_summary": "Warm spring days above 5000m partially thaw ice in rock cracks; Sentinel-1 radar spots creep acceleration.",
                "pdd_c_days": 6.5,
                "rain_72h_mm": 25.0,
                "api_30d_mm": 20.0,
                "sar_disp_mm": [0.0, 2.0, 8.8],
                "seismic_mw": 3.2,
                "lake_exp_pct": 2.4,
            },
            {
                "timestamp": "T-0 (25 April 2015 11:56 NPT — M7.8 Gorkha Earthquake)",
                "layman_summary": "Massive seismic shaking (PGA/g = 0.35) slashes effective grip; FS drops below 1.0 and triggers catastrophic rock-ice avalanche.",
                "pdd_c_days": 14.0,
                "rain_72h_mm": 40.0,
                "api_30d_mm": 30.0,
                "sar_disp_mm": [0.0, 2.0, 10.5],
                "seismic_mw": 7.8,
                "lake_exp_pct": 3.1,
            },
        ]
    elif scenario_id == "budhi_gandaki_monsoon":
        basin_id = "budhi_gandaki"
        title = "Budhi Gandaki Monsoon Rain-on-Snow & Glacial Lake Surge"
        slope_deg = 36.0
        elevation_m = 4400.0
        base_cohesion_kpa = 11.5
        phi_deg = 31.0
        gamma = 19.8
        z_slab = 2.6
        steps_raw = [
            {
                "timestamp": "Day 1 (Early Monsoon Dry Soil)",
                "layman_summary": "Dry pre-monsoon soil sponge; moderate rain drains easily and slope stays GREEN.",
                "pdd_c_days": 1.0,
                "rain_72h_mm": 25.0,
                "api_30d_mm": 10.0,
                "sar_disp_mm": [0.0, 1.8, 3.6],
                "seismic_mw": 1.8,
                "lake_exp_pct": 2.1,
            },
            {
                "timestamp": "Day 15 (Sponge Filling + Glacial Melt)",
                "layman_summary": "Two weeks of monsoon rain fill soil pores (high 30-day API); lake swells past +5% threshold -> ORANGE.",
                "pdd_c_days": 9.0,
                "rain_72h_mm": 98.0,
                "api_30d_mm": 110.0,
                "sar_disp_mm": [0.0, 2.5, 9.8],
                "seismic_mw": 2.5,
                "lake_exp_pct": 5.8,
            },
            {
                "timestamp": "Day 24 (Extreme Cloudburst + >5000m Melt Surge)",
                "layman_summary": "240mm cloudburst falls on already-saturated soil + high snowmelt + M4.2 tremor; FS drops < 1.0 -> RED Evacuation.",
                "pdd_c_days": 16.5,
                "rain_72h_mm": 240.0,
                "api_30d_mm": 220.0,
                "sar_disp_mm": [0.0, 3.0, 13.5],
                "seismic_mw": 4.2,
                "lake_exp_pct": 8.4,
            },
        ]
    else:
        raise ValueError(f"Unknown scenario_id '{scenario_id}'")

    timeline_results: List[Dict[str, object]] = []

    for step in steps_raw:
        c_eff = float(
            compute_permafrost_effective_cohesion(
                base_cohesion_kpa=base_cohesion_kpa,
                elevation_m=elevation_m,
                pdd_c_days=float(step["pdd_c_days"]),
            )
        )
        hydro = evaluate_hydromet_saturation(
            hourly_rainfall_mm_72h=[float(step["rain_72h_mm"]) / 72.0] * 72,
            daily_temps_c_above_5000m=[float(step["pdd_c_days"])],
            elevations_m=[5300.0],
            antecedent_daily_rain_30d_mm=[float(step["api_30d_mm"]) / 6.5] * 30,
            slab_porosity=0.20,
            slab_depth_m=z_slab,
        )
        sar = evaluate_sar_creep_acceleration(step["sar_disp_mm"])
        seis = evaluate_seismic_trigger([{"magnitude": float(step["seismic_mw"])}])

        prob_fs = compute_probabilistic_fs(
            c_prime=c_eff,
            gamma=gamma,
            z=z_slab,
            z_w=float(hydro["estimated_zw_m"]),
            beta_deg=slope_deg,
            phi_prime_deg=phi_deg,
            seismic_kh=float(seis["estimated_seismic_kh"]),
        )
        fs_val = prob_fs["deterministic_fs"]

        triggers: List[str] = []
        if sar["trigger_active"]:
            triggers.append("SAR_CREEP")
        if float(step["lake_exp_pct"]) > 5.0:
            triggers.append("LAKE_EXPANSION")
        if hydro["trigger_active"]:
            triggers.append("HYDROMET_SATURATION")
        if seis["trigger_active"]:
            triggers.append("SEISMIC_TRIGGER")

        if fs_val < 1.0 and len(triggers) >= 1:
            alert = "RED"
        elif fs_val < 1.05 or len(triggers) >= 2:
            alert = "ORANGE"
        elif fs_val < 1.25 or len(triggers) == 1:
            alert = "YELLOW"
        else:
            alert = "GREEN"

        timeline_results.append(
            {
                "timestamp": step["timestamp"],
                "layman_summary": step["layman_summary"],
                "alert_level": alert,
                "effective_cohesion_kpa": round(c_eff, 2),
                "water_table_zw_m": hydro["estimated_zw_m"],
                "slab_depth_z_m": z_slab,
                "deterministic_fs": fs_val,
                "failure_probability_pct": round(prob_fs["failure_probability"] * 100.0, 1),
                "active_triggers": triggers,
            }
        )

    return {
        "scenario_id": scenario_id,
        "basin_id": basin_id,
        "title": title,
        "timeline": timeline_results,
    }
