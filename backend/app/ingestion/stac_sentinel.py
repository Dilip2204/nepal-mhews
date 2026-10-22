"""Copernicus Sentinel-1 SAR Creep & Sentinel-2 NDWI Glacial Lake Expansion.

Implements Dynamic Precursor Triggers 1 & 2 from AGENTS.md:
  1. SAR Slope/Glacier Creep: Acceleration > 10 mm/month over consecutive 12-day
     Sentinel-1 passes.
  2. Glacial Lake Expansion: Rapid positive anomaly in NDWI = (B03 - B08) / (B03 + B08)
     on Sentinel-2.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from app.core.config import PILOT_BASINS, settings


def compute_ndwi(b03_green: np.ndarray, b08_nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index: (B03 - B08) / (B03 + B08).

    Handles zero-denominator pixels cleanly without raising runtime warnings.
    """
    g = np.asarray(b03_green, dtype=np.float64)
    n = np.asarray(b08_nir, dtype=np.float64)
    denom = g + n
    ndwi = np.where(np.abs(denom) > 1e-9, (g - n) / denom, 0.0)
    return np.clip(ndwi, -1.0, 1.0)


def evaluate_glacial_lake_expansion(
    b03_baseline: np.ndarray,
    b08_baseline: np.ndarray,
    b03_current: np.ndarray,
    b08_current: np.ndarray,
    pixel_area_m2: float = 100.0,
    ndwi_threshold: float = settings.ndwi_water_threshold,
    expansion_pct_threshold: float = settings.glof_area_anomaly_pct_threshold,
) -> Dict[str, object]:
    """Evaluate Sentinel-2 NDWI glacial lake surface expansion anomaly."""
    ndwi_base = compute_ndwi(b03_baseline, b08_baseline)
    ndwi_curr = compute_ndwi(b03_current, b08_current)

    water_base = ndwi_base > ndwi_threshold
    water_curr = ndwi_curr > ndwi_threshold

    area_base_ha = float(np.sum(water_base) * pixel_area_m2 / 10000.0)
    area_curr_ha = float(np.sum(water_curr) * pixel_area_m2 / 10000.0)

    if area_base_ha > 0:
        area_change_pct = ((area_curr_ha - area_base_ha) / area_base_ha) * 100.0
    else:
        area_change_pct = 100.0 if area_curr_ha > 0 else 0.0

    mean_ndwi_delta = float(np.mean(ndwi_curr - ndwi_base))
    trigger_active = bool(area_change_pct > expansion_pct_threshold)

    return {
        "baseline_area_ha": round(area_base_ha, 3),
        "current_area_ha": round(area_curr_ha, 3),
        "expansion_pct": round(area_change_pct, 2),
        "mean_ndwi_anomaly": round(mean_ndwi_delta, 4),
        "trigger_active": trigger_active,
        "threshold_pct": expansion_pct_threshold,
    }


def evaluate_cloud_resilient_lake_expansion(
    b03_baseline: np.ndarray,
    b08_baseline: np.ndarray,
    b03_current: np.ndarray,
    b08_current: np.ndarray,
    cloud_cover_pct: float = 0.0,
    sar_vv_db_current: Optional[np.ndarray] = None,
    sar_water_threshold_db: float = -16.0,
    pixel_area_m2: float = 100.0,
    ndwi_threshold: float = settings.ndwi_water_threshold,
    expansion_pct_threshold: float = settings.glof_area_anomaly_pct_threshold,
) -> Dict[str, object]:
    """Monsoon countermeasure: fuse Sentinel-2 NDWI with Sentinel-1 SAR backscatter.

    Layman intuition: During Nepal's monsoon (June-September), thick clouds block
    Sentinel-2 optical cameras. However, Sentinel-1 radar pierces right through clouds,
    and calm lake water acts like a mirror that bounces radar away (producing dark pixels
    below -16 dB). When cloud cover exceeds 50%, we use SAR water masks automatically.
    """
    ndwi_base = compute_ndwi(b03_baseline, b08_baseline)
    water_base = ndwi_base > ndwi_threshold

    if cloud_cover_pct > 50.0 and sar_vv_db_current is not None:
        sensor_used = "SENTINEL1_SAR_BACKSCATTER_FALLBACK"
        water_curr = np.asarray(sar_vv_db_current, dtype=np.float64) < sar_water_threshold_db
    else:
        sensor_used = "SENTINEL2_OPTICAL_NDWI"
        ndwi_curr = compute_ndwi(b03_current, b08_current)
        water_curr = ndwi_curr > ndwi_threshold

    area_base_ha = float(np.sum(water_base) * pixel_area_m2 / 10000.0)
    area_curr_ha = float(np.sum(water_curr) * pixel_area_m2 / 10000.0)
    area_change_pct = (
        ((area_curr_ha - area_base_ha) / area_base_ha) * 100.0
        if area_base_ha > 0
        else (100.0 if area_curr_ha > 0 else 0.0)
    )

    return {
        "sensor_mode": sensor_used,
        "cloud_cover_pct": round(float(cloud_cover_pct), 1),
        "baseline_area_ha": round(area_base_ha, 3),
        "current_area_ha": round(area_curr_ha, 3),
        "expansion_pct": round(area_change_pct, 2),
        "trigger_active": bool(area_change_pct > expansion_pct_threshold),
    }


def evaluate_sar_creep_acceleration(
    displacement_mm_series: Sequence[float],
    pass_interval_days: float = 12.0,
    accel_threshold_mm_month: float = settings.sar_creep_accel_threshold_mm_month,
) -> Dict[str, object]:
    """Evaluate Sentinel-1 InSAR Line-of-Sight (LOS) creep acceleration.

    Requires at least 3 consecutive Sentinel-1 epochs (i.e., 2 consecutive 12-day
    intervals) to compute velocity and acceleration normalized to mm/month (30 days).

    Args:
        displacement_mm_series: Cumulative LOS displacement (mm) at consecutive passes.
        pass_interval_days: Repeat cycle in days (12.0 for Sentinel-1A).
        accel_threshold_mm_month: Critical acceleration threshold (10.0 mm/month).
    """
    disp = np.asarray(displacement_mm_series, dtype=np.float64)
    if disp.size < 3:
        raise ValueError(
            "At least 3 Sentinel-1 passes (2 consecutive intervals) are required."
        )
    if pass_interval_days <= 0:
        raise ValueError("pass_interval_days must be positive.")

    month_scale = 30.0 / pass_interval_days
    # Velocity (mm/month) over each 12-day pass interval
    velocities_mm_month = np.diff(disp) * month_scale
    # Acceleration (mm/month change per pass interval)
    accelerations_mm_month = np.diff(velocities_mm_month)

    latest_velocity = float(velocities_mm_month[-1])
    latest_accel = float(accelerations_mm_month[-1])
    max_accel = float(np.max(accelerations_mm_month))
    trigger_active = bool(latest_accel > accel_threshold_mm_month)

    return {
        "latest_velocity_mm_month": round(latest_velocity, 2),
        "latest_acceleration_mm_month": round(latest_accel, 2),
        "max_acceleration_mm_month": round(max_accel, 2),
        "pass_interval_days": pass_interval_days,
        "trigger_active": trigger_active,
        "threshold_mm_month": accel_threshold_mm_month,
    }


def search_basin_stac_scenes(
    basin_id: str,
    collections: Optional[List[str]] = None,
    datetime_range: str = "2026-10-01/2026-10-22",
    live_query: bool = False,
) -> List[Dict[str, object]]:
    """Query STAC catalog for Sentinel-1/2 scenes over a pilot basin.

    Uses deterministic synthetic STAC metadata when `live_query=False` or offline.
    """
    basin = PILOT_BASINS.get(basin_id, PILOT_BASINS["langtang"])
    target_collections = collections or ["sentinel-2-l2a", "sentinel-1-grd"]

    if live_query:
        try:
            from pystac_client import Client

            client = Client.open(settings.earth_search_stac_url)
            search = client.search(
                collections=target_collections,
                bbox=list(basin.bbox),
                datetime=datetime_range,
                max_items=5,
            )
            items = [
                {
                    "id": item.id,
                    "collection": item.collection_id,
                    "datetime": item.datetime.isoformat() if item.datetime else None,
                    "bbox": item.bbox,
                }
                for item in search.items()
            ]
            if items:
                return items
        except Exception:
            pass

    return [
        {
            "id": f"S1A_IW_SLC_{basin.id.upper()}_20261020T001245",
            "collection": "sentinel-1-grd",
            "datetime": "2026-10-20T00:12:45Z",
            "bbox": list(basin.bbox),
            "orbit_state": "DESCENDING",
            "repeat_cycle_days": 12,
        },
        {
            "id": f"S2B_MSIL2A_{basin.id.upper()}_20261021T050659",
            "collection": "sentinel-2-l2a",
            "datetime": "2026-10-21T05:06:59Z",
            "bbox": list(basin.bbox),
            "cloud_cover_pct": 8.4,
        },
    ]
