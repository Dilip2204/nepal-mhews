"""Hydrometeorological Saturation Ingestion (NASA IMERG + 5000m+ PDD Snowmelt + 30d API).

Implements Dynamic Precursor Trigger 3 from AGENTS.md:
  3. Hydrometeorological Saturation: 72-hour cumulative rainfall (NASA IMERG) +
     Positive Degree Day (PDD) snowmelt index above 5000m.

Enhanced with 30-day Antecedent Precipitation Index (`API` "sponge memory") so the
model distinguishes storms falling on dry pre-monsoon soil from storms falling on
already-saturated late-monsoon slopes.
"""

from typing import Dict, Optional, Sequence
import numpy as np
from app.core.config import settings


def compute_antecedent_precipitation_index(
    daily_rainfall_mm_30d: Sequence[float],
    decay_k: float = 0.88,
) -> float:
    """Compute 30-day Antecedent Precipitation Index (API) with daily drainage decay k.

    Layman intuition: Soil acts like a sponge that slowly drains by ~12% each day
    (`decay_k = 0.88`). Rain from yesterday counts almost fully (88%), while rain
    from 20 days ago has mostly drained away (0.88^20 = 8%).

    Args:
        daily_rainfall_mm_30d: Daily rainfall totals (mm) ordered oldest -> most recent.
        decay_k: Daily soil moisture retention factor (typically 0.85 - 0.92).
    """
    rain = np.maximum(np.asarray(daily_rainfall_mm_30d, dtype=np.float64), 0.0)
    if rain.size == 0:
        return 0.0
    # Exponents: oldest day gets k^(N), most recent prior day gets k^1
    n = rain.size
    exponents = np.arange(n, 0, -1, dtype=np.float64)
    weights = decay_k ** exponents
    return round(float(np.sum(rain * weights)), 2)


def compute_pdd_snowmelt_above_5000m(
    temperatures_c: np.ndarray,
    elevations_m: np.ndarray,
    elevation_threshold_m: float = settings.pdd_elevation_threshold_m,
    degree_day_factor_mm_c_day: float = settings.degree_day_melt_factor_mm_c_day,
    base_temp_c: float = 0.0,
) -> Dict[str, float]:
    """Compute Positive Degree Day (PDD) thermal snow/ice melt above 5000m."""
    temp_arr = np.asarray(temperatures_c, dtype=np.float64)
    elev_arr = np.asarray(elevations_m, dtype=np.float64)

    high_mask = elev_arr >= elevation_threshold_m
    if not np.any(high_mask):
        return {
            "pdd_c_days": 0.0,
            "snowmelt_mm_we": 0.0,
            "high_altitude_fraction": 0.0,
        }

    positive_temps = np.maximum(temp_arr - base_temp_c, 0.0)
    if positive_temps.shape == elev_arr.shape:
        high_pos_temps = positive_temps[high_mask]
        pdd = float(np.sum(high_pos_temps))
    else:
        high_pos_temps = positive_temps[..., high_mask]
        pdd = float(np.sum(np.mean(high_pos_temps, axis=-1)))

    snowmelt_mm = pdd * degree_day_factor_mm_c_day
    return {
        "pdd_c_days": round(pdd, 3),
        "snowmelt_mm_we": round(snowmelt_mm, 3),
        "high_altitude_fraction": round(float(np.mean(high_mask)), 4),
    }


def evaluate_hydromet_saturation(
    hourly_rainfall_mm_72h: Sequence[float],
    daily_temps_c_above_5000m: Sequence[float],
    elevations_m: Sequence[float],
    antecedent_daily_rain_30d_mm: Optional[Sequence[float]] = None,
    rain_threshold_mm: float = settings.rain_72h_threshold_mm,
    slab_porosity: float = 0.35,
    slab_depth_m: float = 2.5,
) -> Dict[str, object]:
    """Combine 72-hour NASA IMERG rainfall, 30-day sponge memory (API), and >5000m PDD snowmelt."""
    rain_arr = np.asarray(hourly_rainfall_mm_72h, dtype=np.float64)
    if rain_arr.size > 72:
        rain_arr = rain_arr[-72:]
    cumulative_rain_72h_mm = float(np.sum(np.maximum(rain_arr, 0.0)))

    api_30d_mm = (
        compute_antecedent_precipitation_index(antecedent_daily_rain_30d_mm)
        if antecedent_daily_rain_30d_mm is not None
        else 0.0
    )

    pdd_result = compute_pdd_snowmelt_above_5000m(
        temperatures_c=np.asarray(daily_temps_c_above_5000m, dtype=np.float64),
        elevations_m=np.asarray(elevations_m, dtype=np.float64),
    )
    snowmelt_mm = pdd_result["snowmelt_mm_we"]
    # 35% of retained antecedent moisture contributes directly to perched water table
    effective_api_mm = 0.35 * api_30d_mm
    total_water_equivalent_mm = cumulative_rain_72h_mm + snowmelt_mm + effective_api_mm

    infiltration_depth_m = (total_water_equivalent_mm / 1000.0) / max(
        slab_porosity, 0.05
    )
    estimated_zw_m = float(np.clip(infiltration_depth_m, 0.0, slab_depth_m))
    saturation_ratio = estimated_zw_m / max(slab_depth_m, 1e-6)

    trigger_active = bool(
        total_water_equivalent_mm >= rain_threshold_mm
        or (cumulative_rain_72h_mm >= 80.0 and pdd_result["pdd_c_days"] >= 6.0)
    )

    return {
        "cumulative_rain_72h_mm": round(cumulative_rain_72h_mm, 2),
        "antecedent_api_30d_mm": round(api_30d_mm, 2),
        "pdd_c_days_above_5000m": pdd_result["pdd_c_days"],
        "snowmelt_mm_we": snowmelt_mm,
        "total_water_input_mm": round(total_water_equivalent_mm, 2),
        "estimated_zw_m": round(estimated_zw_m, 3),
        "saturation_ratio": round(saturation_ratio, 3),
        "trigger_active": trigger_active,
    }
