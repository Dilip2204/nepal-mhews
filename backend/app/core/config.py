"""Application configuration and domain constants for Himalayan-MHEWS."""

from dataclasses import dataclass, field
import os
from typing import Dict, Tuple


@dataclass(frozen=True)
class BasinConfig:
    """Spatial and geomorphological metadata for a pilot Himalayan watershed."""

    id: str
    name: str
    district: str
    bbox: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    center: Tuple[float, float]  # (lon, lat)
    max_elevation_m: float
    key_hotspots: Tuple[str, ...]


PILOT_BASINS: Dict[str, BasinConfig] = {
    "langtang": BasinConfig(
        id="langtang",
        name="Trishuli / Langtang Watershed",
        district="Rasuwa",
        bbox=(85.15, 28.05, 85.80, 28.45),
        center=(85.515, 28.215),
        max_elevation_m=7227.0,
        key_hotspots=(
            "Langtang Lirung Rock-Ice Face (7227m)",
            "Kyangjin Gompa Debris Cone",
            "Shalbachum Glacial Lake",
        ),
    ),
    "budhi_gandaki": BasinConfig(
        id="budhi_gandaki",
        name="Budhi Gandaki Watershed",
        district="Gorkha",
        bbox=(84.50, 28.10, 85.10, 28.75),
        center=(84.630, 28.540),
        max_elevation_m=8163.0,
        key_hotspots=(
            "Thulagi Glacial Lake (Dona Lake)",
            "Birendra Tal (Manaslu Glacier)",
            "Philim Gorge Landslide Dam Choke Point",
        ),
    ),
}


@dataclass
class Settings:
    """Runtime settings for Himalayan-MHEWS backend."""

    app_name: str = "Nepal Multi-Hazard Early Warning System (Himalayan-MHEWS)"
    version: str = "0.1.0"
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://mhews:mhews_secret@localhost:5432/nepal_mhews",
        )
    )
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    copernicus_stac_url: str = "https://catalogue.dataspace.copernicus.eu/stac"
    earth_search_stac_url: str = "https://earth-search.aws.element84.com/v1"
    usgs_fdsn_url: str = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # Domain Physics & Precursor Thresholds
    critical_fs_threshold: float = 1.0
    watch_fs_threshold: float = 1.25
    water_unit_weight_kn_m3: float = 9.81
    sar_creep_accel_threshold_mm_month: float = 10.0
    ndwi_water_threshold: float = 0.20
    glof_area_anomaly_pct_threshold: float = 5.0
    rain_72h_threshold_mm: float = 120.0
    pdd_elevation_threshold_m: float = 5000.0
    degree_day_melt_factor_mm_c_day: float = 4.5
    seismic_min_magnitude: float = 4.0


settings = Settings()
