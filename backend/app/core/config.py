"""Application configuration and domain constants for Himalayan-MHEWS."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Tuple

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


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

    # --- Weekly AI Audit: dual-model (Gemini + Claude) consensus ---
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    )
    # Gemini returns 503 UNAVAILABLE ("high demand") on popular models; without a
    # fallback the weekly audit silently loses one of its two independent votes and
    # the consensus gate admits nothing. Tried in order after a 503/429.
    gemini_fallback_models: Tuple[str, ...] = (
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    )
    # Org-scoped Anthropic keys must name a workspace or every call 400s. Leave blank
    # when using a workspace-scoped key.
    anthropic_workspace_id: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_WORKSPACE_ID", "")
    )
    # A ground-truth label is admitted only when both models agree (see audit Pillar 1).
    audit_require_dual_consensus: bool = field(
        default_factory=lambda: os.getenv("AUDIT_REQUIRE_DUAL_CONSENSUS", "1") == "1"
    )
    # Seed/demo bulletins must stay out of the production training catalog.
    seed_demo_bulletins_enabled: bool = field(
        default_factory=lambda: os.getenv("MHEWS_SEED_DEMO", "0") == "1"
    )

    # --- Emergency alert dispatch (ORANGE/RED transitions) ---
    alerts_enabled: bool = field(
        default_factory=lambda: os.getenv("MHEWS_ALERTS_ENABLED", "0") == "1"
    )
    alert_min_level: str = field(
        default_factory=lambda: os.getenv("MHEWS_ALERT_MIN_LEVEL", "ORANGE")
    )
    # Suppress repeat alerts for the same site+level within this window.
    alert_cooldown_minutes: int = field(
        default_factory=lambda: int(os.getenv("MHEWS_ALERT_COOLDOWN_MIN", "180"))
    )
    alert_webhook_url: str = field(
        default_factory=lambda: os.getenv("MHEWS_ALERT_WEBHOOK_URL", "")
    )
    telegram_bot_token: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", "")
    )
    alert_email_to: str = field(default_factory=lambda: os.getenv("MHEWS_ALERT_EMAIL_TO", ""))
    alert_email_from: str = field(
        default_factory=lambda: os.getenv("MHEWS_ALERT_EMAIL_FROM", "mhews@localhost")
    )
    alert_smtp_host: str = field(default_factory=lambda: os.getenv("MHEWS_SMTP_HOST", ""))
    alert_smtp_port: int = field(default_factory=lambda: int(os.getenv("MHEWS_SMTP_PORT", "587")))
    alert_smtp_user: str = field(default_factory=lambda: os.getenv("MHEWS_SMTP_USER", ""))
    alert_smtp_password: str = field(
        default_factory=lambda: os.getenv("MHEWS_SMTP_PASSWORD", "")
    )

    # --- Copernicus Data Space Ecosystem (real Sentinel-1 InSAR ingestion) ---
    cdse_enabled: bool = field(
        default_factory=lambda: os.getenv("CDSE_ENABLED", "0") == "1"
    )
    cdse_client_id: str = field(default_factory=lambda: os.getenv("CDSE_CLIENT_ID", ""))
    cdse_client_secret: str = field(
        default_factory=lambda: os.getenv("CDSE_CLIENT_SECRET", "")
    )
    # OData is the live catalogue API; the older resto/OpenSearch path now returns 404.
    cdse_catalogue_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    cdse_token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )

    # --- ASF HyP3 on-demand Sentinel-1 InSAR (the only path to real LOS velocity here) ---
    # CDSE publishes SLC scenes only and LiCSAR's public tree 404s on its interferogram
    # directories, so a genuinely MEASURED velocity has to be processed on demand.
    hyp3_enabled: bool = field(default_factory=lambda: os.getenv("HYP3_ENABLED", "0") == "1")
    hyp3_api_url: str = field(
        default_factory=lambda: os.getenv("HYP3_API_URL", "https://hyp3-api.asf.alaska.edu")
    )
    # Earthdata Login. A bearer token is preferred; user/password is the fallback.
    earthdata_token: str = field(default_factory=lambda: os.getenv("EARTHDATA_TOKEN", ""))
    earthdata_username: str = field(default_factory=lambda: os.getenv("EARTHDATA_USERNAME", ""))
    earthdata_password: str = field(default_factory=lambda: os.getenv("EARTHDATA_PASSWORD", ""))
    # NASA CMR granule search is open (no credentials) and is used to discover which real
    # Sentinel-1 acquisitions cover a site before any job is submitted.
    cmr_granule_search_url: str = (
        "https://cmr.earthdata.nasa.gov/search/granules.json"
    )
    # HyP3 Basic grants 8000 free credits/month; exceeding it silently stops processing,
    # so the refresh run budgets against it rather than discovering the cap at runtime.
    hyp3_monthly_credit_budget: int = field(
        default_factory=lambda: int(os.getenv("HYP3_MONTHLY_CREDIT_BUDGET", "8000"))
    )
    hyp3_max_jobs_per_run: int = field(
        default_factory=lambda: int(os.getenv("HYP3_MAX_JOBS_PER_RUN", "20"))
    )
    # InSAR baselines refresh on a weekly cadence (jobs take ~1h), never in the 30-min loop.
    hyp3_baseline_max_age_days: int = field(
        default_factory=lambda: int(os.getenv("HYP3_BASELINE_MAX_AGE_DAYS", "30"))
    )

    # --- NDRRMA BIPAD incident portal (real ground-truth harvesting) ---
    bipad_enabled: bool = field(
        default_factory=lambda: os.getenv("BIPAD_ENABLED", "0") == "1"
    )
    bipad_api_url: str = field(
        default_factory=lambda: os.getenv(
            "BIPAD_API_URL", "https://bipadportal.gov.np/api/v1/incident/"
        )
    )

    # --- Adaptive monsoon cadence throttle ---
    adaptive_throttle_enabled: bool = field(
        default_factory=lambda: os.getenv("MHEWS_ADAPTIVE_THROTTLE", "1") == "1"
    )
    normal_cycle_interval_sec: int = field(
        default_factory=lambda: int(os.getenv("CYCLE_INTERVAL_SEC", "1800"))
    )
    monsoon_cycle_interval_sec: int = field(
        default_factory=lambda: int(os.getenv("MONSOON_CYCLE_INTERVAL_SEC", "900"))
    )
    # Any site exceeding this 72h rainfall switches the whole loop to fast cadence.
    monsoon_throttle_rain_72h_mm: float = field(
        default_factory=lambda: float(os.getenv("MONSOON_THROTTLE_RAIN_MM", "150.0"))
    )


settings = Settings()
