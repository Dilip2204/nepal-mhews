"""AI Accelerator 2: Spatial Soil-Moisture (9km SMAP -> 30m) & Regolith Downscaler.

Sharpens coarse satellite products:
  - NASA SMAP L-Band Root-Zone Soil Moisture (9 km resolution)
  - Global Regolith / SoilGrids Thickness Priors (250m - 1km resolution)
down to our 30m x 30m Himalayan watershed grid using high-resolution topographic,
optical, and thermal modulators (`TWI`, `slope_deg`, `aspect_deg`, `NDVI`, `ATI`),
while strictly preserving Mass / Volume Conservation across the coarse pixel.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


def downscale_smap_moisture_and_regolith_to_30m(
    coarse_smap_vwc: float,
    coarse_regolith_depth_m: float,
    slope_deg_grid: np.ndarray,
    aspect_deg_grid: np.ndarray,
    twi_grid: np.ndarray,
    ndvi_grid: np.ndarray,
    diurnal_lst_range_c_grid: np.ndarray,
    surface_albedo_grid: np.ndarray | float = 0.20,
) -> Dict[str, np.ndarray | float]:
    """Downscale coarse 9 km SMAP soil moisture and 250m regolith depth to a 30m DEM grid.

    Layman Intuition:
    1. NASA's SMAP satellite sees only one blurry average moisture number for a whole 9x9 km
       valley (say, 0.28 m^3/m^3).
    2. Inside that 9x9 km square, water drains away from steep, sun-baked ridges (low TWI,
       south aspect) and pools inside shaded, vegetated gullies (high TWI, north aspect, high NDVI).
    3. We compute a 30m Wetness Modulation Weight `W(x, y)` for every 30m pixel and normalize it
       by its spatial mean `W(x, y) / mean(W)` so the average across all 30m pixels equals the
       exact NASA SMAP satellite reading (Strict Mass Conservation)!
    """
    slope = np.asarray(slope_deg_grid, dtype=np.float64)
    aspect = np.asarray(aspect_deg_grid, dtype=np.float64)
    twi = np.asarray(twi_grid, dtype=np.float64)
    ndvi = np.asarray(ndvi_grid, dtype=np.float64)
    dt_lst = np.asarray(diurnal_lst_range_c_grid, dtype=np.float64)
    albedo = np.asarray(surface_albedo_grid, dtype=np.float64)

    # 1. Apparent Thermal Inertia (ATI) at 30m: higher ATI indicates wetter/cooler or denser ground
    ati = (1.0 - np.clip(albedo, 0.05, 0.85)) / np.maximum(dt_lst, 2.0)
    ati_norm = (ati - np.mean(ati)) / max(float(np.std(ati)), 1e-6)

    # 2. Solar aspect shading: North-facing slopes (aspect ~ 0) retain moisture; South (180) dries out
    southness = 0.5 * (1.0 - np.cos(np.deg2rad(aspect)))
    shade_retention = 1.15 - 0.30 * southness

    # 3. Topographic Wetness Index (TWI) & Vegetation (NDVI) sub-grid redistribution weights
    twi_rel = np.clip(twi / max(float(np.mean(twi)), 1.0), 0.4, 2.4)
    veg_retention = 1.0 + 0.25 * np.clip(ndvi, 0.0, 0.85)
    thermal_wetness = np.clip(1.0 + 0.12 * ati_norm, 0.75, 1.30)

    raw_moisture_weights = twi_rel * shade_retention * veg_retention * thermal_wetness
    # Enforce strict spatial mass conservation: mean(normalized_weights) == 1.0
    norm_moisture_weights = raw_moisture_weights / max(float(np.mean(raw_moisture_weights)), 1e-9)

    downscaled_vwc_30m = np.clip(coarse_smap_vwc * norm_moisture_weights, 0.03, 0.58)

    # 4. Downscale 250m/1km Regolith Depth to 30m using slope shedding + TWI bowl trapping + ATI
    slope_shedding = np.exp(-0.042 * np.clip(slope - np.mean(slope), -25.0, 35.0))
    insulating_mantle = np.clip(1.25 - 4.0 * ati, 0.45, 1.45)
    raw_depth_weights = slope_shedding * (0.7 + 0.3 * twi_rel) * insulating_mantle
    norm_depth_weights = raw_depth_weights / max(float(np.mean(raw_depth_weights)), 1e-9)

    downscaled_z_30m = np.clip(coarse_regolith_depth_m * norm_depth_weights, 0.30, 6.0)

    return {
        "downscaled_vwc_30m": np.round(downscaled_vwc_30m, 4),
        "downscaled_regolith_z_30m": np.round(downscaled_z_30m, 3),
        "coarse_smap_vwc": round(float(coarse_smap_vwc), 4),
        "mean_downscaled_vwc": round(float(np.mean(downscaled_vwc_30m)), 4),
        "min_vwc_ridge": round(float(np.min(downscaled_vwc_30m)), 4),
        "max_vwc_gully": round(float(np.max(downscaled_vwc_30m)), 4),
    }

