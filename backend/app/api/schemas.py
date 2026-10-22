"""Pydantic v2 request and response schemas for Himalayan-MHEWS endpoints."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class InfiniteSlopeRequest(BaseModel):
    """Input payload for Infinite Slope Factor of Safety calculation."""

    c_prime: float = Field(12.0, ge=0.0, description="Effective cohesion c' (kPa)")
    gamma: float = Field(19.5, gt=0.0, description="Slope material unit weight (kN/m^3)")
    z: float = Field(2.5, gt=0.0, description="Slab depth z (m)")
    z_w: float = Field(1.2, ge=0.0, description="Saturated water table height z_w (m)")
    beta_deg: float = Field(34.0, gt=0.0, lt=89.5, description="Slope angle beta (degrees)")
    phi_prime_deg: float = Field(32.0, gt=0.0, lt=85.0, description="Friction angle phi' (degrees)")
    seismic_kh: float = Field(0.0, ge=0.0, le=1.0, description="Horizontal seismic coeff k_h (PGA/g)")


class InfiniteSlopeResponse(BaseModel):
    """Computed Factor of Safety and stability regime."""

    fs: float
    critical_failure: bool
    stability_regime: str


class CascadeEvaluationRequest(BaseModel):
    """Multi-hazard Tier A + Tier B cascade evaluation request for a pilot basin."""

    basin_id: str = Field("langtang", description="Pilot basin ID: langtang or budhi_gandaki")
    c_prime: float = Field(10.0, ge=0.0)
    gamma: float = Field(20.0, gt=0.0)
    z: float = Field(2.5, gt=0.0)
    beta_deg: float = Field(36.0, gt=0.0, lt=89.0)
    phi_prime_deg: float = Field(31.0, gt=0.0, lt=85.0)
    sar_displacement_mm: List[float] = Field(
        default_factory=lambda: [0.0, 3.2, 11.8, 26.4],
        description="Cumulative LOS displacement (mm) across consecutive 12-day Sentinel-1 passes",
    )
    hourly_rainfall_mm_72h: List[float] = Field(
        default_factory=lambda: [2.0] * 72,
        description="Hourly NASA IMERG precipitation (mm) over past 72h",
    )
    daily_temps_c_above_5000m: List[float] = Field(
        default_factory=lambda: [3.2, 4.1, 3.8],
        description="Daily mean temperatures (C) at >5000m stations",
    )
    station_elevations_m: List[float] = Field(
        default_factory=lambda: [5150.0, 5420.0, 5680.0],
        description="Station elevations (m)",
    )
    max_seismic_magnitude: float = Field(2.4, ge=0.0, description="Peak local earthquake magnitude")
    lake_expansion_pct: float = Field(6.2, description="30-day Sentinel-2 NDWI lake area change (%)")


class CascadeAlertResponse(BaseModel):
    """Comprehensive Tier A + Tier B multi-hazard alert response."""

    basin_id: str
    basin_name: str
    alert_level: str
    static_fs: float
    dynamic_fs: float
    ml_susceptibility_score: float
    active_triggers: List[str]
    precursor_details: Dict[str, object]
    runout_summary: Optional[Dict[str, object]] = None


class SatelliteSynthesisRequest(BaseModel):
    """Raw satellite, astronomical, LiDAR, radar, and weather inputs for a 30m mountain cell."""

    latitude: float = Field(28.215, ge=26.0, le=31.0, description="WGS84 Latitude for spatial block lookup")
    longitude: float = Field(85.515, ge=80.0, le=89.0, description="WGS84 Longitude for spatial block lookup")
    local_relief_m: float = Field(240.0, ge=0.0, le=3000.0, description="330m window relief from DEM (m)")
    elevation_m: float = Field(4650.0, ge=500.0, le=8848.0)
    slope_deg: float = Field(36.0, gt=0.5, lt=85.0)
    aspect_deg: float = Field(195.0, ge=0.0, le=360.0)
    profile_curvature: float = Field(-0.35, ge=-2.5, le=2.5)
    plan_curvature: float = Field(-0.25, ge=-2.5, le=2.5)
    upslope_contributing_area_m2: float = Field(4500.0, ge=30.0)
    sentinel2_ndvi: float = Field(0.32, ge=-1.0, le=1.0)
    gedi_canopy_height_m: float = Field(6.5, ge=0.0, le=50.0)
    sand_fraction: float = Field(0.48, ge=0.0, le=1.0)
    clay_fraction: float = Field(0.16, ge=0.0, le=1.0)
    coarse_rock_fragment_fraction: float = Field(0.35, ge=0.0, le=1.0)
    dry_bulk_density_g_cm3: float = Field(1.55, gt=0.8, le=2.6)
    distance_to_fault_km: float = Field(2.4, ge=0.0)
    modis_lst_mean_c: float = Field(-1.5, ge=-45.0, le=45.0)
    diurnal_lst_range_c: float = Field(16.0, gt=1.0, le=40.0)
    surface_albedo: float = Field(0.22, ge=0.02, le=0.95)
    diurnal_freeze_thaw_cycles_30d: int = Field(14, ge=0, le=60)
    pdd_c_days_7d: float = Field(6.0, ge=0.0, le=80.0)
    snow_water_equivalent_mm: float = Field(180.0, ge=0.0, le=3000.0)
    imerg_rain_72h_mm: float = Field(115.0, ge=0.0, le=1000.0)
    antecedent_rain_30d_api_mm: float = Field(95.0, ge=0.0, le=1000.0)
    smap_rootzone_moisture_vwc: float = Field(0.31, ge=0.0, le=0.65)
    sar_l_to_c_band_penetration_ratio: float = Field(1.15, ge=0.4, le=2.5)
    bedrock_pga_g: float = Field(0.08, ge=0.0, le=1.0)
    insar_los_velocity_mm_month: float = Field(14.5, ge=0.0, le=200.0)


class SatelliteSynthesisResponse(BaseModel):
    """Synthesized soil depth, cohesion components, water table, and Factor of Safety."""

    synthesized_variables: Dict[str, float]
    deterministic_fs: float
    calibrated_fs: float
    hybrid_susceptibility_score: float
    differential_calibration: Dict[str, object]
    failure_probability_pct: float
    stability_regime: str
