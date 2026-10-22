"""Multi-Source Raw Data to Physics Input Variable Synthesizer & AI Accelerator Pipeline.

Transforms raw satellite, geophysical, soil, and atmospheric datasets into the
physical parameters required by the Generalized Infinite Slope Stability equation:
  - Total Effective Cohesion `c_prime` = c_mineral + c_root(NDVI, GEDI) + c_ice(LST, PDD, freeze_thaw) * fault_damage_factor
  - Effective Internal Friction `phi_prime_deg` = f(sand_pct, clay_pct, fault_distance_km, weathering_index)
  - Moist Bulk Unit Weight `gamma` = f(dry_bulk_density, porosity, saturation_ratio)
  - Regolith / Slab Thickness `z` = f(slope_deg, profile_curvature, twi)
  - Perched Water Table `z_w` = f(IMERG_72h, API_30d, SMAP_soil_moisture, TWI, solar_aspect_evap, PDD_melt)
  - Surface Surcharge Load `q_s` (kPa) = snow_water_equivalent_load + biomass_load
  - Amplified Seismic Coefficient `k_h` = base_pga_g * topographic_ridge_amplification
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
from app.core.config import settings


@dataclass
class RawEarthObservationInputs:
    """Pool of 18 raw satellite, geophysical, and atmospheric sensor inputs for a 30m cell."""

    # 1. DEM & Morphometry (Copernicus / ALOS PALSAR / HMA Stereo DEM)
    elevation_m: float = 4650.0
    slope_deg: float = 36.0
    aspect_deg: float = 195.0  # 0=N, 90=E, 180=S (south-facing receives max sun)
    profile_curvature: float = -0.35  # Negative = concave hollow (collects water/debris)
    plan_curvature: float = -0.25  # Negative = convergent gully channel
    upslope_contributing_area_m2: float = 4500.0

    # 2. Optical & LiDAR Ecology (Sentinel-2 L2A + NASA GEDI Spaceborne LiDAR)
    sentinel2_ndvi: float = 0.32  # [-1, 1] vegetation greenness
    gedi_canopy_height_m: float = 6.5  # Forest canopy height (m)

    # 3. Lithology, Tectonics & SoilGrids 250m (ISRIC SoilGrids + DMG Nepal Faults)
    sand_fraction: float = 0.48
    clay_fraction: float = 0.16
    coarse_rock_fragment_fraction: float = 0.35
    dry_bulk_density_g_cm3: float = 1.55
    distance_to_fault_km: float = 2.4  # Distance to Main Central Thrust (MCT) or active fault

    # 4. Cryosphere & Thermal (MODIS/VIIRS/ECOSTRESS LST + ERA5-Land + Sentinel-1 SAR Snow)
    modis_lst_mean_c: float = -1.5  # Land Surface Temperature (C)
    diurnal_lst_range_c: float = 16.0  # Day-minus-Night LST (C) -> Apparent Thermal Inertia proxy
    surface_albedo: float = 0.22  # Optical broadband albedo [0, 1]
    diurnal_freeze_thaw_cycles_30d: int = 14  # Count of 0C crossings in past 30 days
    pdd_c_days_7d: float = 6.0  # Positive Degree Days (C-days)
    snow_water_equivalent_mm: float = 180.0  # Weight of snowpack pressing on slope

    # 5. Hydrometeorology & Microwave Soil Moisture (NASA IMERG + SMAP L-Band + NISAR/ALOS-2 L-band)
    imerg_rain_72h_mm: float = 115.0
    antecedent_rain_30d_api_mm: float = 95.0
    smap_rootzone_moisture_vwc: float = 0.31  # Volumetric water content (m^3/m^3)
    sar_l_to_c_band_penetration_ratio: float = 1.15  # ALOS-2/NISAR L-band vs Sentinel-1 C-band ratio

    # 6. Seismology & Radar Deformation (USGS ShakeMap + Sentinel-1 InSAR)
    bedrock_pga_g: float = 0.08  # Peak Ground Acceleration (g) at flat bedrock
    insar_los_velocity_mm_month: float = 14.5  # Line-of-Sight radar creep speed


def estimate_satellite_soil_depth_z(
    slope_deg: float,
    profile_curvature: float,
    plan_curvature: float,
    twi: float,
    southness_index: float,
    diurnal_lst_range_c: float = 16.0,
    surface_albedo: float = 0.22,
    gedi_canopy_height_m: float = 0.0,
    sar_l_to_c_band_ratio: float = 1.0,
) -> Dict[str, float]:
    """Estimate Himalayan soil/regolith thickness z (m) without ground boreholes.

    Fuses 6 astronomical, thermal, radar, LiDAR, and geomorphic satellite signals:
    1. Nonlinear Critical-Slope Mass Balance: Above ~50 deg, loose soil sheds off (z -> bedrock).
    2. Geomorphic Hollows & TWI Accumulation: Concave bowls (curvature < 0) trap deep colluvium.
    3. Astronomical Solar Aspect Weathering: South-facing slopes experience intense daily
       solar thermal cycling (`southness_index`), accelerating bedrock breakdown into regolith.
    4. Apparent Thermal Inertia (ATI) from MODIS/ECOSTRESS Day-Night Temperature Swing:
       High ATI (low diurnal swing) indicates dense exposed bedrock (thin z); low ATI
       (large diurnal swing) indicates porous, insulating dry regolith/talus mantle.
    5. NASA GEDI LiDAR Canopy Root-Anchoring Floor: Tall trees require a minimum rooting depth.
    6. L-Band (NISAR/ALOS-2) vs. C-Band (Sentinel-1) Radar Penetration Ratio.
    """
    # 1. Apparent Thermal Inertia: ATI = (1 - albedo) / max(Delta_T, 2.0)
    ati = (1.0 - np.clip(surface_albedo, 0.05, 0.85)) / max(diurnal_lst_range_c, 2.0)
    # High ATI (>0.12) = bare rock outcrop; Low ATI (<0.06) = thick porous debris mantle
    thermal_mantle_factor = float(np.clip(1.35 - 5.5 * ati, 0.35, 1.35))

    # 2. Hillslope mass-balance baseline with critical cliff cutoff
    hillslope_base_m = (0.40 + 3.1 * np.exp(-0.042 * slope_deg)) * thermal_mantle_factor

    # 3. Geomorphic concavity (bowl trapping) + TWI valley deposition
    curvature_accumulation_m = float(
        np.clip(-0.35 * profile_curvature - 0.25 * plan_curvature, -0.45, 1.10)
    )
    twi_deposition_m = 0.08 * max(twi - 4.0, 0.0)

    # 4. Astronomical solar insolation weathering bonus (south-facing thermal stress)
    solar_weathering_m = 0.22 * southness_index

    # 5. L-band / C-band SAR subsurface volume scattering adjustment
    sar_penetration_m = 0.25 * np.clip(sar_l_to_c_band_ratio - 1.0, -0.4, 0.8)

    # 6. Spaceborne LiDAR (GEDI) biomechanical root-depth floor (z >= 0.35 * H^0.55)
    lidar_root_floor_m = (
        float(0.35 * (gedi_canopy_height_m ** 0.55)) if gedi_canopy_height_m > 1.0 else 0.30
    )

    z_raw = (
        hillslope_base_m
        + curvature_accumulation_m
        + twi_deposition_m
        + solar_weathering_m
        + sar_penetration_m
    )
    z_final = float(np.clip(max(z_raw, lidar_root_floor_m), 0.30, 6.0))
    return {
        "z_m": round(z_final, 3),
        "apparent_thermal_inertia": round(float(ati), 4),
        "lidar_root_floor_m": round(lidar_root_floor_m, 3),
    }


def synthesize_physics_variables(raw: RawEarthObservationInputs) -> Dict[str, float]:
    """Mathematical + AI-accelerated feature synthesis converting raw inputs into slope physics.

    Returns all synthesized parameters plus a breakdown of each physical mechanism.
    """
    slope_rad = np.deg2rad(np.clip(raw.slope_deg, 0.5, 85.0))

    # --- 1. Topographic Wetness Index (TWI) & Solar Aspect Evapotranspiration Factor ---
    # TWI = ln(a / tan(beta)), where 'a' is specific catchment area per unit contour width (30m)
    specific_catchment_area = max(raw.upslope_contributing_area_m2 / 30.0, 1.0)
    twi = float(np.log(specific_catchment_area / max(np.tan(slope_rad), 0.01)))

    # South-facing slopes in Nepal (aspect ~ 180 deg) receive stronger solar radiation -> more drying & faster snowmelt
    aspect_rad = np.deg2rad(raw.aspect_deg)
    southness_index = float(0.5 * (1.0 - np.cos(aspect_rad)))  # 0.0 = North, 1.0 = South

    # --- 2. Regolith / Sliding Slab Depth z (m) from Multi-Satellite Signals ---
    depth_info = estimate_satellite_soil_depth_z(
        slope_deg=raw.slope_deg,
        profile_curvature=raw.profile_curvature,
        plan_curvature=raw.plan_curvature,
        twi=twi,
        southness_index=southness_index,
        diurnal_lst_range_c=raw.diurnal_lst_range_c,
        surface_albedo=raw.surface_albedo,
        gedi_canopy_height_m=raw.gedi_canopy_height_m,
        sar_l_to_c_band_ratio=raw.sar_l_to_c_band_penetration_ratio,
    )
    z_m = depth_info["z_m"]

    # --- 3. Composite Effective Cohesion c' (kPa) ---
    # (a) Mineral soil matrix cohesion from clay & coarse interlocking
    c_mineral_kpa = 3.5 + 14.0 * raw.clay_fraction + 4.0 * raw.coarse_rock_fragment_fraction

    # (b) Wu-Waldron Vegetation Root Cohesion from Sentinel-2 NDVI + NASA GEDI Canopy Height
    # Active only below ~4200m alpine treeline where NDVI > 0.15
    if raw.elevation_m < 4300.0 and raw.sentinel2_ndvi > 0.15:
        canopy_factor = np.clip(raw.gedi_canopy_height_m / 15.0, 0.1, 1.5)
        c_root_kpa = float(12.0 * (raw.sentinel2_ndvi ** 1.4) * canopy_factor)
    else:
        c_root_kpa = 0.0

    # (c) Permafrost Ice-Cement Cohesion & Freeze-Thaw Frost-Cracking Degradation
    if raw.elevation_m >= 4500.0:
        freeze_strength = max(0.0, 18.0 - 1.15 * max(0.0, raw.modis_lst_mean_c + 2.0))
        thaw_loss = 0.95 * raw.pdd_c_days_7d
        # Repeated diurnal freeze-thaw cycles wedge rock cracks open (frost fatigue)
        frost_fatigue_loss = 0.22 * raw.diurnal_freeze_thaw_cycles_30d
        c_ice_kpa = float(max(0.0, freeze_strength - thaw_loss - frost_fatigue_loss))
    else:
        c_ice_kpa = 0.0

    # (d) Tectonic Fault-Zone Fracturing Penalty & InSAR Micro-Cracking Softening
    # Rock within 3 km of the Main Central Thrust (MCT) is heavily shattered
    fault_integrity_factor = float(1.0 - 0.40 * np.exp(-raw.distance_to_fault_km / 2.5))
    creep_softening_factor = float(1.0 / (1.0 + 0.018 * max(0.0, raw.insar_los_velocity_mm_month - 5.0)))

    c_prime_kpa = float(
        max(
            0.5,
            (c_mineral_kpa * fault_integrity_factor + c_root_kpa + c_ice_kpa)
            * creep_softening_factor,
        )
    )

    # --- 4. Effective Internal Friction Angle phi' (degrees) ---
    # Angular sand and coarse rock fragments increase friction; clay and fault gouge reduce it
    base_phi_deg = (
        24.0
        + 12.0 * raw.sand_fraction
        + 9.5 * raw.coarse_rock_fragment_fraction
        - 14.0 * raw.clay_fraction
    )
    phi_prime_deg = float(
        np.clip(base_phi_deg * (0.85 + 0.15 * fault_integrity_factor), 16.0, 44.0)
    )

    # --- 5. Hydrology, SMAP Soil Moisture & Perched Water Table z_w (m) ---
    # Porosity from dry bulk density (assuming mineral particle density 2.65 g/cm^3)
    porosity = float(np.clip(1.0 - (raw.dry_bulk_density_g_cm3 / 2.65), 0.18, 0.52))
    snowmelt_mm = raw.pdd_c_days_7d * settings.degree_day_melt_factor_mm_c_day * (1.0 + 0.2 * southness_index)
    effective_water_mm = (
        raw.imerg_rain_72h_mm
        + 0.35 * raw.antecedent_rain_30d_api_mm
        + snowmelt_mm
    )
    # Topographic convergence multiplier: concave gullies (high TWI) concentrate subsurface flow
    twi_concentration = float(np.clip(twi / 6.0, 0.6, 2.2))
    infiltration_zw = ((effective_water_mm / 1000.0) / porosity) * twi_concentration
    smap_baseline_zw = float(np.clip((raw.smap_rootzone_moisture_vwc / porosity) * z_m * 0.6, 0.0, z_m))
    z_w_m = float(np.clip(max(infiltration_zw, smap_baseline_zw), 0.0, z_m))
    saturation_ratio = z_w_m / max(z_m, 1e-6)

    # Unsaturated Matric Suction Bonus (capillary surface tension in partially wet soil)
    # Peaks around 40% saturation, vanishes at 100% saturation (z_w == z)
    matric_suction_kpa = float(
        6.0 * saturation_ratio * ((1.0 - saturation_ratio) ** 1.5) * (1.0 + 2.0 * raw.clay_fraction)
    )

    # --- 6. Dynamic Moist Bulk Unit Weight gamma (kN/m^3) & Surface Surcharge q_s (kPa) ---
    gamma_dry_kn_m3 = raw.dry_bulk_density_g_cm3 * 9.81
    gamma_moist_kn_m3 = float(
        gamma_dry_kn_m3 + porosity * saturation_ratio * settings.water_unit_weight_kn_m3
    )

    # Snowpack weight (1 mm SWE = 1 kg/m^2 = 0.00981 kPa) + forest canopy surcharge
    snow_load_kpa = (raw.snow_water_equivalent_mm / 1000.0) * settings.water_unit_weight_kn_m3
    canopy_load_kpa = 0.08 * raw.gedi_canopy_height_m if raw.elevation_m < 4300.0 else 0.0
    surface_surcharge_q_kpa = float(snow_load_kpa + canopy_load_kpa)

    # --- 7. Topographic Ridge Amplification of Seismic Shaking k_h ---
    # Convex mountain ridges (profile_curvature > 0) and high elevations trap seismic waves
    ridge_amplification = float(
        np.clip(
            1.0
            + 0.25 * max(0.0, raw.profile_curvature)
            + 0.20 * max(0.0, (raw.elevation_m - 2500.0) / 4000.0),
            1.0,
            1.75,
        )
    )
    seismic_kh = float(np.clip(raw.bedrock_pga_g * ridge_amplification, 0.0, 0.65))

    return {
        "c_prime_kpa": round(c_prime_kpa, 3),
        "c_mineral_kpa": round(c_mineral_kpa, 3),
        "c_root_kpa": round(c_root_kpa, 3),
        "c_ice_kpa": round(c_ice_kpa, 3),
        "matric_suction_kpa": round(matric_suction_kpa, 3),
        "phi_prime_deg": round(phi_prime_deg, 2),
        "gamma_kn_m3": round(gamma_moist_kn_m3, 2),
        "z_m": round(z_m, 3),
        "apparent_thermal_inertia": depth_info["apparent_thermal_inertia"],
        "lidar_root_floor_m": depth_info["lidar_root_floor_m"],
        "z_w_m": round(z_w_m, 3),
        "saturation_ratio": round(saturation_ratio, 3),
        "surface_surcharge_q_kpa": round(surface_surcharge_q_kpa, 3),
        "seismic_kh": round(seismic_kh, 4),
        "ridge_amplification_factor": round(ridge_amplification, 3),
        "twi": round(twi, 2),
        "porosity": round(porosity, 3),
    }
