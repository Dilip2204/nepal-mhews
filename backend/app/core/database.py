"""PostGIS + TimescaleDB schema definitions and SQL helper utilities."""

from typing import Dict, List
from app.core.config import settings

POSTGIS_TIMESCALEDB_DDL: List[str] = [
    "CREATE EXTENSION IF NOT EXISTS postgis;",
    "CREATE EXTENSION IF NOT EXISTS postgis_raster;",
    "CREATE EXTENSION IF NOT EXISTS timescaledb;",
    """
    CREATE TABLE IF NOT EXISTS susceptibility_grid_30m (
        cell_id BIGSERIAL PRIMARY KEY,
        basin_id VARCHAR(64) NOT NULL,
        slope_deg DOUBLE PRECISION NOT NULL,
        elevation_m DOUBLE PRECISION NOT NULL,
        cohesion_kpa DOUBLE PRECISION NOT NULL,
        friction_deg DOUBLE PRECISION NOT NULL,
        static_fs DOUBLE PRECISION NOT NULL,
        ml_susceptibility_score DOUBLE PRECISION NOT NULL,
        geom GEOMETRY(Polygon, 4326) NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS precursor_telemetry (
        ts TIMESTAMPTZ NOT NULL,
        basin_id VARCHAR(64) NOT NULL,
        sensor_type VARCHAR(32) NOT NULL,
        metric_name VARCHAR(64) NOT NULL,
        metric_value DOUBLE PRECISION NOT NULL,
        is_anomaly BOOLEAN DEFAULT FALSE,
        location GEOMETRY(Point, 4326)
    );
    """,
    """
    SELECT create_hypertable('precursor_telemetry', 'ts', if_not_exists => TRUE);
    """,
    """
    CREATE TABLE IF NOT EXISTS hazard_alerts (
        alert_id VARCHAR(64) PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        basin_id VARCHAR(64) NOT NULL,
        alert_level VARCHAR(16) NOT NULL,
        min_fs DOUBLE PRECISION NOT NULL,
        active_triggers JSONB NOT NULL,
        runout_geojson JSONB
    );
    """,
]


def get_database_metadata() -> Dict[str, object]:
    """Return configured database connection and DDL status metadata."""
    return {
        "database_url_configured": bool(settings.database_url),
        "extensions": ["postgis", "postgis_raster", "timescaledb"],
        "tables": [
            "susceptibility_grid_30m",
            "precursor_telemetry",
            "hazard_alerts",
        ],
        "ddl_statements": POSTGIS_TIMESCALEDB_DDL,
    }
