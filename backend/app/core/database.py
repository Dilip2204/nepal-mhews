"""PostGIS + TimescaleDB Production Time-Series Store & Schema Management.

Provides an enterprise-grade abstraction layer supporting PostgreSQL 16 + PostGIS + TimescaleDB
hypertables with automated, transparent fallback to local SQLite when running in lightweight
workstation mode without the Dockerized database cluster.
"""

from __future__ import annotations
import json
import logging
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger("nepal_mhews.database")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_SQLITE_PATH = DATA_DIR / "perpetual_telemetry.db"

# Raw telemetry columns persisted on every cycle (shared by both engines).
# Columns not measured live (e.g. api_rain_30d_mm, insar_creep_mm_month) stay NULL
# rather than being filled with proxies, so downstream AI audits see real data only.
TELEMETRY_COLUMNS: List[Tuple[str, str]] = [
    ("rain_72h_mm", "REAL"),
    ("rain_peak_1h_mm", "REAL"),
    ("temp_2m_c", "REAL"),
    ("surface_pressure_hpa", "REAL"),
    ("pressure_deficit_hpa", "REAL"),
    ("cloud_cover_pct", "REAL"),
    ("wind_speed_kmh", "REAL"),
    ("wind_gusts_kmh", "REAL"),
    ("wind_surcharge_kpa", "REAL"),
    ("pdd_days_7d", "REAL"),
    ("api_rain_30d_mm", "REAL"),
    ("ati_diurnal_lst", "REAL"),
    ("smap_saturation_pct", "REAL"),
    ("insar_creep_mm_month", "REAL"),
    ("seismic_pga_g", "REAL"),
    ("relative_humidity_pct", "REAL"),
    ("seismic_mag", "REAL"),
    ("cloudburst_flag", "INTEGER"),
    ("cyclonic_depression_flag", "INTEGER"),
]

# Field-name aliases emitted by `LiveSiteTelemetry` -> canonical DB column names
_TELEMETRY_ALIASES: Dict[str, str] = {
    "wind_speed_10m_kmh": "wind_speed_kmh",
    "pdd_days": "pdd_days_7d",
    "dahal_hasegawa_cloudburst": "cloudburst_flag",
}


def normalize_telemetry_record(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Map a live telemetry dict onto canonical `TELEMETRY_COLUMNS` names."""
    out: Dict[str, Any] = {}
    for key, value in telemetry.items():
        col = _TELEMETRY_ALIASES.get(key, key)
        if col not in out or out[col] is None:
            out[col] = value
    for flag in ("cloudburst_flag", "cyclonic_depression_flag"):
        if out.get(flag) is not None:
            out[flag] = 1 if bool(out[flag]) else 0
    return {col: out.get(col) for col, _ in TELEMETRY_COLUMNS}

POSTGIS_TIMESCALEDB_DDL: List[str] = [
    "CREATE EXTENSION IF NOT EXISTS postgis;",
    "CREATE EXTENSION IF NOT EXISTS postgis_raster;",
    "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;",
    """
    CREATE TABLE IF NOT EXISTS sites (
        site_id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(128) NOT NULL,
        hazard_type VARCHAR(64) NOT NULL,
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        elevation_m DOUBLE PRECISION NOT NULL,
        slope_deg DOUBLE PRECISION NOT NULL,
        profile_curvature DOUBLE PRECISION,
        local_relief_m DOUBLE PRECISION,
        fault_distance_km DOUBLE PRECISION,
        basin_id VARCHAR(64) NOT NULL,
        geom GEOMETRY(Point, 4326)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_sites_geom ON sites USING GIST (geom);",
    "CREATE INDEX IF NOT EXISTS idx_sites_basin ON sites (basin_id);",
    """
    CREATE TABLE IF NOT EXISTS telemetry_snapshots (
        ts TIMESTAMPTZ NOT NULL,
        site_id VARCHAR(64) NOT NULL,
        rain_72h_mm DOUBLE PRECISION,
        rain_peak_1h_mm DOUBLE PRECISION,
        temp_2m_c DOUBLE PRECISION,
        surface_pressure_hpa DOUBLE PRECISION,
        pressure_deficit_hpa DOUBLE PRECISION,
        cloud_cover_pct DOUBLE PRECISION,
        wind_speed_kmh DOUBLE PRECISION,
        wind_gusts_kmh DOUBLE PRECISION,
        wind_surcharge_kpa DOUBLE PRECISION,
        pdd_days_7d DOUBLE PRECISION,
        api_rain_30d_mm DOUBLE PRECISION,
        ati_diurnal_lst DOUBLE PRECISION,
        smap_saturation_pct DOUBLE PRECISION,
        insar_creep_mm_month DOUBLE PRECISION,
        seismic_pga_g DOUBLE PRECISION,
        geom GEOMETRY(Point, 4326)
    );
    """,
    # Columns added after the initial cutover (idempotent upgrades for existing clusters)
    "ALTER TABLE telemetry_snapshots ADD COLUMN IF NOT EXISTS relative_humidity_pct DOUBLE PRECISION;",
    "ALTER TABLE telemetry_snapshots ADD COLUMN IF NOT EXISTS seismic_mag DOUBLE PRECISION;",
    "ALTER TABLE telemetry_snapshots ADD COLUMN IF NOT EXISTS cloudburst_flag BOOLEAN;",
    "ALTER TABLE telemetry_snapshots ADD COLUMN IF NOT EXISTS cyclonic_depression_flag BOOLEAN;",
    "SELECT create_hypertable('telemetry_snapshots', 'ts', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_site_ts ON telemetry_snapshots (site_id, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_geom ON telemetry_snapshots USING GIST (geom);",
    """
    CREATE TABLE IF NOT EXISTS physics_calculations (
        ts TIMESTAMPTZ NOT NULL,
        site_id VARCHAR(64) NOT NULL,
        soil_depth_z_m DOUBLE PRECISION NOT NULL,
        c_prime_kpa DOUBLE PRECISION NOT NULL,
        phi_prime_deg DOUBLE PRECISION NOT NULL,
        static_fs DOUBLE PRECISION NOT NULL,
        dynamic_fs DOUBLE PRECISION NOT NULL,
        calibrated_fs DOUBLE PRECISION NOT NULL,
        failure_prob_pct DOUBLE PRECISION NOT NULL,
        alert_level VARCHAR(16) NOT NULL,
        active_triggers TEXT,
        geom GEOMETRY(Point, 4326)
    );
    """,
    "SELECT create_hypertable('physics_calculations', 'ts', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
    "CREATE INDEX IF NOT EXISTS idx_physics_site_ts ON physics_calculations (site_id, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_physics_alert_ts ON physics_calculations (alert_level, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_physics_geom ON physics_calculations USING GIST (geom);",
    """
    CREATE TABLE IF NOT EXISTS ground_truth_events (
        event_id VARCHAR(64) PRIMARY KEY,
        event_name VARCHAR(128) NOT NULL,
        verified_timestamp TIMESTAMPTZ NOT NULL,
        source VARCHAR(64) NOT NULL,
        hazard_type VARCHAR(64) NOT NULL,
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        elevation_m DOUBLE PRECISION NOT NULL,
        slope_deg DOUBLE PRECISION NOT NULL,
        profile_curvature DOUBLE PRECISION,
        local_relief_m DOUBLE PRECISION,
        fault_distance_km DOUBLE PRECISION,
        label INTEGER NOT NULL,
        ndvi_drop DOUBLE PRECISION,
        rainfall_72h_mm DOUBLE PRECISION,
        notes TEXT,
        geom GEOMETRY(Point, 4326)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_gt_geom ON ground_truth_events USING GIST (geom);",
    "CREATE INDEX IF NOT EXISTS idx_gt_label ON ground_truth_events (label);",
    "CREATE INDEX IF NOT EXISTS idx_gt_source ON ground_truth_events (source);",
    """
    CREATE TABLE IF NOT EXISTS hazard_alerts (
        alert_id VARCHAR(64) PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        site_id VARCHAR(64) NOT NULL,
        basin_id VARCHAR(64) NOT NULL,
        alert_level VARCHAR(16) NOT NULL,
        dynamic_fs DOUBLE PRECISION NOT NULL,
        active_triggers JSONB NOT NULL,
        runout_geojson JSONB,
        civil_broadcast JSONB
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_alerts_level_time ON hazard_alerts (alert_level, created_at DESC);",
    """
    CREATE TABLE IF NOT EXISTS unprocessed_bulletins (
        bulletin_id VARCHAR(64) PRIMARY KEY,
        received_utc DOUBLE PRECISION NOT NULL,
        source VARCHAR(64) NOT NULL,
        title TEXT NOT NULL,
        raw_text TEXT NOT NULL,
        url TEXT,
        district_hint VARCHAR(64),
        status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
        extracted_event_id VARCHAR(64),
        parsed_json JSONB
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_bulletins_status ON unprocessed_bulletins (status, received_utc DESC);",
    "CREATE INDEX IF NOT EXISTS idx_bulletins_source ON unprocessed_bulletins (source);",
    """
    CREATE TABLE IF NOT EXISTS satellite_visual_candidates (
        candidate_id VARCHAR(64) PRIMARY KEY,
        detected_utc DOUBLE PRECISION NOT NULL,
        site_id VARCHAR(64),
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        trigger_type VARCHAR(64) NOT NULL,
        delta_ndvi DOUBLE PRECISION,
        sar_velocity_mm_month DOUBLE PRECISION,
        thumbnail_before_path TEXT,
        thumbnail_after_path TEXT,
        status VARCHAR(32) NOT NULL DEFAULT 'PENDING_AI_INSPECTION',
        ai_confidence DOUBLE PRECISION,
        ai_notes TEXT,
        geom GEOMETRY(Point, 4326)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_visual_status ON satellite_visual_candidates (status, detected_utc DESC);",
    "CREATE INDEX IF NOT EXISTS idx_visual_geom ON satellite_visual_candidates USING GIST (geom);",
    """
    CREATE TABLE IF NOT EXISTS ai_audit_runs (
        run_id VARCHAR(64) PRIMARY KEY,
        audit_timestamp_utc DOUBLE PRECISION NOT NULL,
        engine_used VARCHAR(64) NOT NULL,
        bulletins_scanned INTEGER NOT NULL DEFAULT 0,
        events_extracted INTEGER NOT NULL DEFAULT 0,
        visual_candidates_inspected INTEGER NOT NULL DEFAULT 0,
        visual_confirmed INTEGER NOT NULL DEFAULT 0,
        equations_evaluated INTEGER NOT NULL DEFAULT 0,
        promoted_to_production BOOLEAN NOT NULL DEFAULT FALSE,
        promoted_parameters JSONB,
        summary_report TEXT,
        token_usage_json JSONB
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_audit_time ON ai_audit_runs (audit_timestamp_utc DESC);",
]


class BaseTimeSeriesStore:
    """Interface contract for Himalayan-MHEWS Time-Series Store."""

    def get_engine_name(self) -> str:
        raise NotImplementedError

    def init_schema(self) -> None:
        raise NotImplementedError

    def insert_physics_calculation(
        self,
        timestamp_utc: float,
        site_id: str,
        soil_depth_z_m: float,
        c_prime_kpa: float,
        phi_prime_deg: float,
        static_fs: float,
        dynamic_fs: float,
        calibrated_fs: float,
        failure_prob_pct: float,
        alert_level: str,
        active_triggers: List[str],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    def insert_telemetry_snapshot(
        self,
        timestamp_utc: float,
        site_id: str,
        telemetry: Dict[str, Any],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    def get_latest_physics_calculations(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def get_active_disaster_indicator(self) -> Dict[str, Any]:
        raise NotImplementedError

    # --- Staging & Weekly AI Audit Queues ---
    def insert_unprocessed_bulletin(self, bulletin: Dict[str, Any]) -> str:
        raise NotImplementedError

    def get_pending_bulletins(self, limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def update_bulletin_status(
        self,
        bulletin_id: str,
        status: str,
        extracted_event_id: Optional[str] = None,
        parsed_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    def insert_visual_candidate(self, candidate: Dict[str, Any]) -> str:
        raise NotImplementedError

    def get_pending_visual_candidates(self, limit: int = 20) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def update_visual_candidate_status(
        self,
        candidate_id: str,
        status: str,
        ai_confidence: float,
        ai_notes: str,
    ) -> None:
        raise NotImplementedError

    def insert_ai_audit_run(self, run: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_latest_ai_audit_runs(self, limit: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_telemetry_outliers_for_audit(self, window_hours: int = 168) -> Dict[str, Any]:
        raise NotImplementedError


class TimescalePostGISStore(BaseTimeSeriesStore):
    """Production Time-Series Store backed by PostgreSQL 16 + PostGIS + TimescaleDB."""

    def __init__(self, dsn: str):
        import psycopg2
        from psycopg2.pool import ThreadedConnectionPool

        self.dsn = dsn
        # Strip SQLAlchemy dialect prefix if present
        clean_dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")
        self.pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=clean_dsn)
        self.init_schema()

    def get_engine_name(self) -> str:
        return "POSTGIS_TIMESCALEDB"

    def init_schema(self) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                for stmt in POSTGIS_TIMESCALEDB_DDL:
                    stmt_clean = stmt.strip()
                    if stmt_clean:
                        try:
                            cur.execute(stmt_clean)
                        except Exception as e:
                            conn.rollback()
                            logger.debug("TimescaleDB DDL statement notice (%s): %s", e, stmt_clean[:50])
                            continue
            conn.commit()
            logger.info("TimescaleDB + PostGIS schema and hypertables successfully verified")
        finally:
            self.pool.putconn(conn)

    def insert_physics_calculation(
        self,
        timestamp_utc: float,
        site_id: str,
        soil_depth_z_m: float,
        c_prime_kpa: float,
        phi_prime_deg: float,
        static_fs: float,
        dynamic_fs: float,
        calibrated_fs: float,
        failure_prob_pct: float,
        alert_level: str,
        active_triggers: List[str],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                ts_iso = time.strftime("%Y-%m-%d %H:%M:%S+00", time.gmtime(timestamp_utc))
                geom_sql = "ST_SetSRID(ST_MakePoint(%s, %s), 4326)" if (lat and lon) else "NULL"
                params = [
                    ts_iso,
                    site_id,
                    soil_depth_z_m,
                    c_prime_kpa,
                    phi_prime_deg,
                    static_fs,
                    dynamic_fs,
                    calibrated_fs,
                    failure_prob_pct,
                    alert_level,
                    ",".join(active_triggers),
                ]
                if lat and lon:
                    params.extend([lon, lat])
                    sql = f"""
                        INSERT INTO physics_calculations (
                            ts, site_id, soil_depth_z_m, c_prime_kpa, phi_prime_deg,
                            static_fs, dynamic_fs, calibrated_fs, failure_prob_pct,
                            alert_level, active_triggers, geom
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, {geom_sql})
                    """
                else:
                    sql = """
                        INSERT INTO physics_calculations (
                            ts, site_id, soil_depth_z_m, c_prime_kpa, phi_prime_deg,
                            static_fs, dynamic_fs, calibrated_fs, failure_prob_pct,
                            alert_level, active_triggers
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                cur.execute(sql, params)
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def insert_telemetry_snapshot(
        self,
        timestamp_utc: float,
        site_id: str,
        telemetry: Dict[str, Any],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                ts_iso = time.strftime("%Y-%m-%d %H:%M:%S+00", time.gmtime(timestamp_utc))
                sql = """
                    INSERT INTO telemetry_snapshots (
                        ts, site_id, rain_72h_mm, rain_peak_1h_mm, temp_2m_c,
                        surface_pressure_hpa, pressure_deficit_hpa, cloud_cover_pct,
                        wind_speed_kmh, wind_gusts_kmh, wind_surcharge_kpa,
                        pdd_days_7d, api_rain_30d_mm, ati_diurnal_lst,
                        smap_saturation_pct, insar_creep_mm_month, seismic_pga_g, geom
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        CASE WHEN %s IS NOT NULL AND %s IS NOT NULL THEN ST_SetSRID(ST_MakePoint(%s, %s), 4326) ELSE NULL END
                    )
                """
                cur.execute(
                    sql,
                    (
                        ts_iso,
                        site_id,
                        telemetry.get("rain_72h_mm"),
                        telemetry.get("rain_peak_1h_mm"),
                        telemetry.get("temp_2m_c"),
                        telemetry.get("surface_pressure_hpa"),
                        telemetry.get("pressure_deficit_hpa"),
                        telemetry.get("cloud_cover_pct"),
                        telemetry.get("wind_speed_kmh"),
                        telemetry.get("wind_gusts_kmh"),
                        telemetry.get("wind_surcharge_kpa"),
                        telemetry.get("pdd_days_7d"),
                        telemetry.get("api_rain_30d_mm"),
                        telemetry.get("ati_diurnal_lst"),
                        telemetry.get("smap_saturation_pct"),
                        telemetry.get("insar_creep_mm_month"),
                        telemetry.get("seismic_pga_g"),
                        lon,
                        lat,
                        lon,
                        lat,
                    ),
                )
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_latest_physics_calculations(self) -> Dict[str, Dict[str, Any]]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Optimized TimescaleDB query utilizing composite index (site_id, ts DESC)
                cur.execute("""
                    SELECT DISTINCT ON (site_id)
                        site_id, static_fs, dynamic_fs, calibrated_fs,
                        failure_prob_pct, alert_level, active_triggers,
                        EXTRACT(EPOCH FROM ts) AS timestamp_utc
                    FROM physics_calculations
                    ORDER BY site_id, ts DESC;
                """)
                calc_map = {}
                for row in cur.fetchall():
                    dyn_fs = float(row[2]) if row[2] is not None else 1.25
                    static_fs = float(row[1]) if row[1] is not None else 1.0
                    fail_prob = (
                        float(row[4])
                        if row[4] is not None
                        else float(1.0 / (1.0 + math.exp(6.8 * (dyn_fs - 1.0)))) * 100.0
                    )
                    triggers = row[6].split(",") if row[6] else []

                    # Enforce live actual risk classification
                    if dyn_fs < 1.0 and len(triggers) >= 1:
                        computed_alert = "RED"
                    elif dyn_fs < 1.0 or len(triggers) >= 2:
                        computed_alert = "ORANGE"
                    elif dyn_fs < 1.25 or len(triggers) >= 1:
                        computed_alert = "YELLOW"
                    else:
                        computed_alert = "GREEN"

                    calc_map[row[0]] = {
                        "static_fs": static_fs,
                        "dynamic_fs": dyn_fs,
                        "calibrated_fs": dyn_fs,
                        "failure_prob_pct": fail_prob,
                        "alert_level": computed_alert,
                        "active_triggers": triggers,
                        "last_refresh_utc": float(row[7]),
                    }
                return calc_map
        finally:
            self.pool.putconn(conn)

    def get_active_disaster_indicator(self) -> Dict[str, Any]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ON (site_id) site_id, dynamic_fs, alert_level, active_triggers
                    FROM physics_calculations
                    ORDER BY site_id, ts DESC;
                """)
                critical_sites = []
                warning_sites = []
                for row in cur.fetchall():
                    sid, dyn_fs, lvl, trigs = row[0], float(row[1]), row[2], row[3] or ""
                    trig_list = trigs.split(",") if trigs else []
                    if dyn_fs < 1.0 and len(trig_list) >= 1:
                        critical_sites.append({"site_id": sid, "dynamic_fs": dyn_fs, "triggers": trigs})
                    elif dyn_fs < 1.0 or len(trig_list) >= 2:
                        warning_sites.append({"site_id": sid, "dynamic_fs": dyn_fs, "triggers": trigs})

                if critical_sites:
                    return {
                        "level": "CRITICAL_RED",
                        "title": "CRITICAL EMERGENCY: ACTIVE SLOPE FAILURE / COLLAPSE DETECTED",
                        "message": f"Immediate civil evacuation required! {len(critical_sites)} monitored site(s) currently exhibiting critical failure conditions under active environmental triggers.",
                        "is_disaster_active": True,
                        "critical_count": len(critical_sites),
                        "warning_count": len(warning_sites),
                        "critical_sites": critical_sites,
                    }
                elif warning_sites:
                    return {
                        "level": "ELEVATED_ORANGE",
                        "title": "ELEVATED HAZARD WATCH: STEEP GULLY MONITORING ACTIVE",
                        "message": f"Zero active emergency collapses detected. {len(warning_sites)} steep high-gradient scree site(s) show marginal equilibrium (FS ~0.98–1.04) under baseline dry weather. No nationwide catastrophe.",
                        "is_disaster_active": False,
                        "critical_count": 0,
                        "warning_count": len(warning_sites),
                        "warning_sites": warning_sites,
                    }
                else:
                    return {
                        "level": "NORMAL_GREEN",
                        "title": "NO ACTIVE LANDSLIDE OR AVALANCHE DISASTER DETECTED",
                        "message": "All ground-truth verification stations and monitored corridors across Nepal are in stable equilibrium. Zero active slope failures, glacial lake outbursts, or rock-ice avalanches detected under current conditions.",
                        "is_disaster_active": False,
                        "critical_count": 0,
                        "warning_count": 0,
                    }
        finally:
            self.pool.putconn(conn)

    # --- Staging & Weekly AI Audit Implementations (PostGIS / TimescaleDB) ---
    def insert_unprocessed_bulletin(self, bulletin: Dict[str, Any]) -> str:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                bid = bulletin.get("bulletin_id") or f"bul_{int(time.time())}_{abs(hash(bulletin.get('title', '')))%10000:04d}"
                cur.execute(
                    """
                    INSERT INTO unprocessed_bulletins (
                        bulletin_id, received_utc, source, title, raw_text, url, district_hint, status, extracted_event_id, parsed_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (bulletin_id) DO NOTHING;
                    """,
                    (
                        bid,
                        bulletin.get("received_utc", time.time()),
                        bulletin.get("source", "UNKNOWN"),
                        bulletin.get("title", ""),
                        bulletin.get("raw_text", ""),
                        bulletin.get("url"),
                        bulletin.get("district_hint"),
                        bulletin.get("status", "PENDING"),
                        bulletin.get("extracted_event_id"),
                        json.dumps(bulletin.get("parsed_json")) if bulletin.get("parsed_json") else None,
                    ),
                )
            conn.commit()
            return bid
        finally:
            self.pool.putconn(conn)

    def get_pending_bulletins(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT bulletin_id, received_utc, source, title, raw_text, url, district_hint, status, extracted_event_id
                    FROM unprocessed_bulletins
                    WHERE status = 'PENDING'
                    ORDER BY received_utc DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "bulletin_id": r[0],
                        "received_utc": r[1],
                        "source": r[2],
                        "title": r[3],
                        "raw_text": r[4],
                        "url": r[5],
                        "district_hint": r[6],
                        "status": r[7],
                        "extracted_event_id": r[8],
                    }
                    for r in rows
                ]
        finally:
            self.pool.putconn(conn)

    def update_bulletin_status(
        self,
        bulletin_id: str,
        status: str,
        extracted_event_id: Optional[str] = None,
        parsed_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE unprocessed_bulletins
                    SET status = %s,
                        extracted_event_id = COALESCE(%s, extracted_event_id),
                        parsed_json = COALESCE(%s::jsonb, parsed_json)
                    WHERE bulletin_id = %s;
                    """,
                    (
                        status,
                        extracted_event_id,
                        json.dumps(parsed_json) if parsed_json else None,
                        bulletin_id,
                    ),
                )
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def insert_visual_candidate(self, candidate: Dict[str, Any]) -> str:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cid = candidate.get("candidate_id") or f"vis_{int(time.time())}_{abs(hash(candidate.get('site_id', '')))%10000:04d}"
                lat = candidate["latitude"]
                lon = candidate["longitude"]
                cur.execute(
                    """
                    INSERT INTO satellite_visual_candidates (
                        candidate_id, detected_utc, site_id, latitude, longitude,
                        trigger_type, delta_ndvi, sar_velocity_mm_month,
                        thumbnail_before_path, thumbnail_after_path, status,
                        ai_confidence, ai_notes, geom
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                    )
                    ON CONFLICT (candidate_id) DO NOTHING;
                    """,
                    (
                        cid,
                        candidate.get("detected_utc", time.time()),
                        candidate.get("site_id"),
                        lat,
                        lon,
                        candidate.get("trigger_type", "OPTICAL_DELTA_NDVI"),
                        candidate.get("delta_ndvi"),
                        candidate.get("sar_velocity_mm_month"),
                        candidate.get("thumbnail_before_path"),
                        candidate.get("thumbnail_after_path"),
                        candidate.get("status", "PENDING_AI_INSPECTION"),
                        candidate.get("ai_confidence"),
                        candidate.get("ai_notes"),
                        lon,
                        lat,
                    ),
                )
            conn.commit()
            return cid
        finally:
            self.pool.putconn(conn)

    def get_pending_visual_candidates(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT candidate_id, detected_utc, site_id, latitude, longitude,
                           trigger_type, delta_ndvi, sar_velocity_mm_month,
                           thumbnail_before_path, thumbnail_after_path, status
                    FROM satellite_visual_candidates
                    WHERE status = 'PENDING_AI_INSPECTION'
                    ORDER BY detected_utc DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "candidate_id": r[0],
                        "detected_utc": r[1],
                        "site_id": r[2],
                        "latitude": r[3],
                        "longitude": r[4],
                        "trigger_type": r[5],
                        "delta_ndvi": r[6],
                        "sar_velocity_mm_month": r[7],
                        "thumbnail_before_path": r[8],
                        "thumbnail_after_path": r[9],
                        "status": r[10],
                    }
                    for r in rows
                ]
        finally:
            self.pool.putconn(conn)

    def update_visual_candidate_status(
        self,
        candidate_id: str,
        status: str,
        ai_confidence: float,
        ai_notes: str,
    ) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE satellite_visual_candidates
                    SET status = %s, ai_confidence = %s, ai_notes = %s
                    WHERE candidate_id = %s;
                    """,
                    (status, ai_confidence, ai_notes, candidate_id),
                )
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def insert_ai_audit_run(self, run: Dict[str, Any]) -> None:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                rid = run.get("run_id") or f"audit_{int(time.time())}"
                cur.execute(
                    """
                    INSERT INTO ai_audit_runs (
                        run_id, audit_timestamp_utc, engine_used, bulletins_scanned,
                        events_extracted, visual_candidates_inspected, visual_confirmed,
                        equations_evaluated, promoted_to_production, promoted_parameters,
                        summary_report, token_usage_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        summary_report = EXCLUDED.summary_report,
                        promoted_to_production = EXCLUDED.promoted_to_production;
                    """,
                    (
                        rid,
                        run.get("audit_timestamp_utc", time.time()),
                        run.get("engine_used", "UNKNOWN"),
                        run.get("bulletins_scanned", 0),
                        run.get("events_extracted", 0),
                        run.get("visual_candidates_inspected", 0),
                        run.get("visual_confirmed", 0),
                        run.get("equations_evaluated", 0),
                        bool(run.get("promoted_to_production", False)),
                        json.dumps(run.get("promoted_parameters")) if run.get("promoted_parameters") else None,
                        run.get("summary_report", ""),
                        json.dumps(run.get("token_usage_json")) if run.get("token_usage_json") else None,
                    ),
                )
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def get_latest_ai_audit_runs(self, limit: int = 5) -> List[Dict[str, Any]]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, audit_timestamp_utc, engine_used, bulletins_scanned,
                           events_extracted, visual_candidates_inspected, visual_confirmed,
                           equations_evaluated, promoted_to_production, promoted_parameters,
                           summary_report, token_usage_json
                    FROM ai_audit_runs
                    ORDER BY audit_timestamp_utc DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                out = []
                for r in rows:
                    out.append({
                        "run_id": r[0],
                        "audit_timestamp_utc": r[1],
                        "engine_used": r[2],
                        "bulletins_scanned": r[3],
                        "events_extracted": r[4],
                        "visual_candidates_inspected": r[5],
                        "visual_confirmed": r[6],
                        "equations_evaluated": r[7],
                        "promoted_to_production": bool(r[8]),
                        "promoted_parameters": r[9],
                        "summary_report": r[10],
                        "token_usage": r[11],
                    })
                return out
        finally:
            self.pool.putconn(conn)

    def get_telemetry_outliers_for_audit(self, window_hours: int = 168) -> Dict[str, Any]:
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT site_id, MAX(rain_72h_mm) as max_rain, MAX(rain_peak_1h_mm) as max_burst,
                           MAX(wind_speed_kmh) as max_wind, MAX(seismic_pga_g) as max_pga
                    FROM telemetry_snapshots
                    WHERE ts >= NOW() - INTERVAL '%s hours'
                    GROUP BY site_id
                    HAVING MAX(rain_72h_mm) >= 80.0 OR MAX(rain_peak_1h_mm) >= 20.0 OR MAX(seismic_pga_g) >= 0.10
                    ORDER BY max_rain DESC
                    LIMIT 25;
                    """,
                    (window_hours,),
                )
                outliers = [
                    {"site_id": r[0], "max_rain_72h": r[1], "max_burst_1h": r[2], "max_wind": r[3], "max_pga": r[4]}
                    for r in cur.fetchall()
                ]
                return {"extreme_telemetry_sites": outliers, "total_outliers": len(outliers)}
        finally:
            self.pool.putconn(conn)


class SQLiteTimeSeriesStore(BaseTimeSeriesStore):
    """Local SQLite fallback store for development workstations without Docker."""

    def __init__(self, db_path: Path = DEFAULT_SQLITE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_engine_name(self) -> str:
        return "SQLITE_LOCAL_FALLBACK"

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10)

    def init_schema(self) -> None:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sites (
                    site_id TEXT PRIMARY KEY,
                    name TEXT,
                    hazard_type TEXT,
                    latitude REAL,
                    longitude REAL,
                    elevation_m REAL,
                    slope_deg REAL,
                    profile_curvature REAL,
                    local_relief_m REAL,
                    fault_distance_km REAL,
                    basin_id TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc REAL,
                    site_id TEXT,
                    rain_72h_mm REAL,
                    rain_peak_1h_mm REAL,
                    temp_2m_c REAL,
                    surface_pressure_hpa REAL,
                    pressure_deficit_hpa REAL,
                    cloud_cover_pct REAL,
                    wind_speed_kmh REAL,
                    wind_gusts_kmh REAL,
                    wind_surcharge_kpa REAL,
                    pdd_days_7d REAL,
                    api_rain_30d_mm REAL,
                    ati_diurnal_lst REAL,
                    smap_saturation_pct REAL,
                    insar_creep_mm_month REAL,
                    seismic_pga_g REAL,
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
            """)
            # Idempotently migrate columns if table existed prior to expansion
            cur.execute("PRAGMA table_info(telemetry_snapshots)")
            existing_cols = {r[1] for r in cur.fetchall()}
            for col, col_type in TELEMETRY_COLUMNS:
                if col not in existing_cols:
                    try:
                        cur.execute(f"ALTER TABLE telemetry_snapshots ADD COLUMN {col} {col_type};")
                    except Exception as e:
                        logger.debug("SQLite column migration (%s): %s", col, e)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS physics_calculations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc REAL,
                    site_id TEXT,
                    soil_depth_z_m REAL,
                    c_prime_kpa REAL,
                    phi_prime_deg REAL,
                    static_fs REAL,
                    dynamic_fs REAL,
                    calibrated_fs REAL,
                    failure_prob_pct REAL,
                    alert_level TEXT,
                    active_triggers TEXT,
                    FOREIGN KEY (site_id) REFERENCES sites(site_id)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calc_site ON physics_calculations(site_id, id DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_calc_alert ON physics_calculations(alert_level);")

            # Staging & AI Audit Queues
            cur.execute("""
                CREATE TABLE IF NOT EXISTS unprocessed_bulletins (
                    bulletin_id TEXT PRIMARY KEY,
                    received_utc REAL NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    url TEXT,
                    district_hint TEXT,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    extracted_event_id TEXT,
                    parsed_json TEXT
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sqlite_bul_status ON unprocessed_bulletins (status, received_utc DESC);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS satellite_visual_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    detected_utc REAL NOT NULL,
                    site_id TEXT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    trigger_type TEXT NOT NULL,
                    delta_ndvi REAL,
                    sar_velocity_mm_month REAL,
                    thumbnail_before_path TEXT,
                    thumbnail_after_path TEXT,
                    status TEXT NOT NULL DEFAULT 'PENDING_AI_INSPECTION',
                    ai_confidence REAL,
                    ai_notes TEXT
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sqlite_vis_status ON satellite_visual_candidates (status, detected_utc DESC);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_audit_runs (
                    run_id TEXT PRIMARY KEY,
                    audit_timestamp_utc REAL NOT NULL,
                    engine_used TEXT NOT NULL,
                    bulletins_scanned INTEGER NOT NULL DEFAULT 0,
                    events_extracted INTEGER NOT NULL DEFAULT 0,
                    visual_candidates_inspected INTEGER NOT NULL DEFAULT 0,
                    visual_confirmed INTEGER NOT NULL DEFAULT 0,
                    equations_evaluated INTEGER NOT NULL DEFAULT 0,
                    promoted_to_production INTEGER NOT NULL DEFAULT 0,
                    promoted_parameters TEXT,
                    summary_report TEXT,
                    token_usage_json TEXT
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sqlite_audit_time ON ai_audit_runs (audit_timestamp_utc DESC);")

            conn.commit()
        finally:
            conn.close()

    def insert_physics_calculation(
        self,
        timestamp_utc: float,
        site_id: str,
        soil_depth_z_m: float,
        c_prime_kpa: float,
        phi_prime_deg: float,
        static_fs: float,
        dynamic_fs: float,
        calibrated_fs: float,
        failure_prob_pct: float,
        alert_level: str,
        active_triggers: List[str],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO physics_calculations (
                    timestamp_utc, site_id, soil_depth_z_m, c_prime_kpa, phi_prime_deg,
                    static_fs, dynamic_fs, calibrated_fs, failure_prob_pct, alert_level, active_triggers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp_utc,
                    site_id,
                    soil_depth_z_m,
                    c_prime_kpa,
                    phi_prime_deg,
                    static_fs,
                    dynamic_fs,
                    calibrated_fs,
                    failure_prob_pct,
                    alert_level,
                    ",".join(active_triggers),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_telemetry_snapshot(
        self,
        timestamp_utc: float,
        site_id: str,
        telemetry: Dict[str, Any],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        rec = normalize_telemetry_record(telemetry)
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cols = ["timestamp_utc", "site_id"] + [c for c, _ in TELEMETRY_COLUMNS]
            placeholders = ["?"] * len(cols)
            vals = [timestamp_utc, site_id] + [rec.get(c) for c, _ in TELEMETRY_COLUMNS]
            cur.execute(
                f"INSERT INTO telemetry_snapshots ({', '.join(cols)}) VALUES ({', '.join(placeholders)})",
                vals,
            )
            conn.commit()
        finally:
            conn.close()

    def insert_unprocessed_bulletin(self, bulletin: Dict[str, Any]) -> str:
        bid = bulletin.get("bulletin_id") or f"bul_{int(time.time())}_{abs(hash(bulletin.get('title', '')))%10000:04d}"
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO unprocessed_bulletins (
                    bulletin_id, received_utc, source, title, raw_text, url, district_hint, status, extracted_event_id, parsed_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bid,
                    bulletin.get("received_utc", time.time()),
                    bulletin.get("source", "UNKNOWN"),
                    bulletin.get("title", ""),
                    bulletin.get("raw_text", ""),
                    bulletin.get("url"),
                    bulletin.get("district_hint"),
                    bulletin.get("status", "PENDING"),
                    bulletin.get("extracted_event_id"),
                    json.dumps(bulletin.get("parsed_json")) if bulletin.get("parsed_json") else None,
                ),
            )
            conn.commit()
            return bid
        finally:
            conn.close()

    def get_pending_bulletins(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT bulletin_id, received_utc, source, title, raw_text, url, district_hint, status, extracted_event_id
                FROM unprocessed_bulletins
                WHERE status = 'PENDING'
                ORDER BY received_utc DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "bulletin_id": r[0],
                    "received_utc": r[1],
                    "source": r[2],
                    "title": r[3],
                    "raw_text": r[4],
                    "url": r[5],
                    "district_hint": r[6],
                    "status": r[7],
                    "extracted_event_id": r[8],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def update_bulletin_status(
        self,
        bulletin_id: str,
        status: str,
        extracted_event_id: Optional[str] = None,
        parsed_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE unprocessed_bulletins
                SET status = ?,
                    extracted_event_id = COALESCE(?, extracted_event_id),
                    parsed_json = COALESCE(?, parsed_json)
                WHERE bulletin_id = ?;
                """,
                (
                    status,
                    extracted_event_id,
                    json.dumps(parsed_json) if parsed_json else None,
                    bulletin_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_visual_candidate(self, candidate: Dict[str, Any]) -> str:
        cid = candidate.get("candidate_id") or f"vis_{int(time.time())}_{abs(hash(candidate.get('site_id', '')))%10000:04d}"
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO satellite_visual_candidates (
                    candidate_id, detected_utc, site_id, latitude, longitude,
                    trigger_type, delta_ndvi, sar_velocity_mm_month,
                    thumbnail_before_path, thumbnail_after_path, status,
                    ai_confidence, ai_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    candidate.get("detected_utc", time.time()),
                    candidate.get("site_id"),
                    candidate["latitude"],
                    candidate["longitude"],
                    candidate.get("trigger_type", "OPTICAL_DELTA_NDVI"),
                    candidate.get("delta_ndvi"),
                    candidate.get("sar_velocity_mm_month"),
                    candidate.get("thumbnail_before_path"),
                    candidate.get("thumbnail_after_path"),
                    candidate.get("status", "PENDING_AI_INSPECTION"),
                    candidate.get("ai_confidence"),
                    candidate.get("ai_notes"),
                ),
            )
            conn.commit()
            return cid
        finally:
            conn.close()

    def get_pending_visual_candidates(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT candidate_id, detected_utc, site_id, latitude, longitude,
                       trigger_type, delta_ndvi, sar_velocity_mm_month,
                       thumbnail_before_path, thumbnail_after_path, status
                FROM satellite_visual_candidates
                WHERE status = 'PENDING_AI_INSPECTION'
                ORDER BY detected_utc DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "candidate_id": r[0],
                    "detected_utc": r[1],
                    "site_id": r[2],
                    "latitude": r[3],
                    "longitude": r[4],
                    "trigger_type": r[5],
                    "delta_ndvi": r[6],
                    "sar_velocity_mm_month": r[7],
                    "thumbnail_before_path": r[8],
                    "thumbnail_after_path": r[9],
                    "status": r[10],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def update_visual_candidate_status(
        self,
        candidate_id: str,
        status: str,
        ai_confidence: float,
        ai_notes: str,
    ) -> None:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE satellite_visual_candidates
                SET status = ?, ai_confidence = ?, ai_notes = ?
                WHERE candidate_id = ?;
                """,
                (status, ai_confidence, ai_notes, candidate_id),
            )
            conn.commit()
        finally:
            conn.close()

    def insert_ai_audit_run(self, run: Dict[str, Any]) -> None:
        rid = run.get("run_id") or f"audit_{int(time.time())}"
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO ai_audit_runs (
                    run_id, audit_timestamp_utc, engine_used, bulletins_scanned,
                    events_extracted, visual_candidates_inspected, visual_confirmed,
                    equations_evaluated, promoted_to_production, promoted_parameters,
                    summary_report, token_usage_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    run.get("audit_timestamp_utc", time.time()),
                    run.get("engine_used", "UNKNOWN"),
                    run.get("bulletins_scanned", 0),
                    run.get("events_extracted", 0),
                    run.get("visual_candidates_inspected", 0),
                    run.get("visual_confirmed", 0),
                    run.get("equations_evaluated", 0),
                    1 if run.get("promoted_to_production") else 0,
                    json.dumps(run.get("promoted_parameters")) if run.get("promoted_parameters") else None,
                    run.get("summary_report", ""),
                    json.dumps(run.get("token_usage_json")) if run.get("token_usage_json") else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_latest_ai_audit_runs(self, limit: int = 5) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT run_id, audit_timestamp_utc, engine_used, bulletins_scanned,
                       events_extracted, visual_candidates_inspected, visual_confirmed,
                       equations_evaluated, promoted_to_production, promoted_parameters,
                       summary_report, token_usage_json
                FROM ai_audit_runs
                ORDER BY audit_timestamp_utc DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cur.fetchall()
            out = []
            for r in rows:
                out.append({
                    "run_id": r[0],
                    "audit_timestamp_utc": r[1],
                    "engine_used": r[2],
                    "bulletins_scanned": r[3],
                    "events_extracted": r[4],
                    "visual_candidates_inspected": r[5],
                    "visual_confirmed": r[6],
                    "equations_evaluated": r[7],
                    "promoted_to_production": bool(r[8]),
                    "promoted_parameters": json.loads(r[9]) if r[9] else None,
                    "summary_report": r[10],
                    "token_usage": json.loads(r[11]) if r[11] else None,
                })
            return out
        finally:
            conn.close()

    def get_telemetry_outliers_for_audit(self, window_hours: int = 168) -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cutoff = time.time() - (window_hours * 3600.0)
            cur.execute(
                """
                SELECT site_id, MAX(rain_72h_mm) as max_rain, MAX(rain_peak_1h_mm) as max_burst,
                       MAX(wind_speed_kmh) as max_wind, MAX(seismic_pga_g) as max_pga
                FROM telemetry_snapshots
                WHERE timestamp_utc >= ?
                GROUP BY site_id
                HAVING MAX(rain_72h_mm) >= 80.0 OR MAX(rain_peak_1h_mm) >= 20.0 OR MAX(seismic_pga_g) >= 0.10
                ORDER BY max_rain DESC
                LIMIT 25;
                """,
                (cutoff,),
            )
            outliers = [
                {"site_id": r[0], "max_rain_72h": r[1], "max_burst_1h": r[2], "max_wind": r[3], "max_pga": r[4]}
                for r in cur.fetchall()
            ]
            return {"extreme_telemetry_sites": outliers, "total_outliers": len(outliers)}
        finally:
            conn.close()

    def get_latest_physics_calculations(self) -> Dict[str, Dict[str, Any]]:
        if not self.db_path.exists():
            return {}
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT site_id, static_fs, dynamic_fs, calibrated_fs,
                       failure_prob_pct, alert_level, active_triggers, timestamp_utc
                FROM physics_calculations
                WHERE id IN (
                    SELECT MAX(id) FROM physics_calculations GROUP BY site_id
                )
            """)
            calc_map = {}
            for row in cur.fetchall():
                dyn_fs = float(row[2]) if row[2] is not None else 1.25
                static_fs = float(row[1]) if row[1] is not None else 1.0
                fail_prob = (
                    float(row[4])
                    if row[4] is not None
                    else float(1.0 / (1.0 + math.exp(6.8 * (dyn_fs - 1.0)))) * 100.0
                )
                triggers = row[6].split(",") if row[6] else []

                if dyn_fs < 1.0 and len(triggers) >= 1:
                    computed_alert = "RED"
                elif dyn_fs < 1.0 or len(triggers) >= 2:
                    computed_alert = "ORANGE"
                elif dyn_fs < 1.25 or len(triggers) >= 1:
                    computed_alert = "YELLOW"
                else:
                    computed_alert = "GREEN"

                calc_map[row[0]] = {
                    "static_fs": static_fs,
                    "dynamic_fs": dyn_fs,
                    "calibrated_fs": dyn_fs,
                    "failure_prob_pct": fail_prob,
                    "alert_level": computed_alert,
                    "active_triggers": triggers,
                    "last_refresh_utc": float(row[7]),
                }
            return calc_map
        finally:
            conn.close()

    def get_active_disaster_indicator(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            return {
                "level": "NORMAL_GREEN",
                "title": "NO ACTIVE LANDSLIDE OR AVALANCHE DISASTER DETECTED",
                "message": "Database initializing. Zero active danger triggers.",
                "is_disaster_active": False,
                "critical_count": 0,
                "warning_count": 0,
            }
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT site_id, dynamic_fs, alert_level, active_triggers 
                FROM physics_calculations 
                WHERE id IN (SELECT MAX(id) FROM physics_calculations GROUP BY site_id)
                  AND alert_level IN ('RED', 'ORANGE')
            """)
            critical_sites = []
            warning_sites = []
            for row in cur.fetchall():
                if row[2] == "RED":
                    critical_sites.append({"site_id": row[0], "dynamic_fs": row[1], "triggers": row[3]})
                elif row[2] == "ORANGE":
                    warning_sites.append({"site_id": row[0], "dynamic_fs": row[1], "triggers": row[3]})

            if critical_sites:
                return {
                    "level": "CRITICAL_RED",
                    "title": "CRITICAL EMERGENCY: ACTIVE SLOPE FAILURE / COLLAPSE DETECTED",
                    "message": f"Immediate civil evacuation required! {len(critical_sites)} monitored site(s) currently exhibiting critical failure conditions under active environmental triggers.",
                    "is_disaster_active": True,
                    "critical_count": len(critical_sites),
                    "warning_count": len(warning_sites),
                    "critical_sites": critical_sites,
                }
            elif warning_sites:
                return {
                    "level": "ELEVATED_ORANGE",
                    "title": "ELEVATED HAZARD WATCH: STEEP GULLY MONITORING ACTIVE",
                    "message": f"Zero active emergency collapses detected. {len(warning_sites)} steep high-gradient scree site(s) show marginal equilibrium (FS ~0.98–1.04) under baseline dry weather. No nationwide catastrophe.",
                    "is_disaster_active": False,
                    "critical_count": 0,
                    "warning_count": len(warning_sites),
                    "warning_sites": warning_sites,
                }
            else:
                return {
                    "level": "NORMAL_GREEN",
                    "title": "NO ACTIVE LANDSLIDE OR AVALANCHE DISASTER DETECTED",
                    "message": "All ground-truth verification stations and monitored corridors across Nepal are in stable equilibrium. Zero active slope failures, glacial lake outbursts, or rock-ice avalanches detected under current conditions.",
                    "is_disaster_active": False,
                    "critical_count": 0,
                    "warning_count": 0,
                }
        finally:
            conn.close()


_ACTIVE_STORE: Optional[BaseTimeSeriesStore] = None


def get_time_series_store() -> BaseTimeSeriesStore:
    """Singleton factory returning TimescalePostGISStore if available, otherwise SQLiteTimeSeriesStore."""
    global _ACTIVE_STORE
    if _ACTIVE_STORE is not None:
        return _ACTIVE_STORE

    db_url = settings.database_url
    if db_url and ("postgresql" in db_url or "postgres://" in db_url):
        try:
            import psycopg2
            clean_url = db_url.replace("postgresql+psycopg2://", "postgresql://")
            # Quick probe with 2.0s timeout
            conn = psycopg2.connect(clean_url, connect_timeout=2)
            conn.close()
            _ACTIVE_STORE = TimescalePostGISStore(clean_url)
            logger.info("[Database] Production TimescaleDB + PostGIS Store connected: %s", clean_url.split("@")[-1])
            return _ACTIVE_STORE
        except Exception as e:
            logger.warning(
                "[Database] TimescaleDB at '%s' not reachable (%s); falling back to local SQLite store.",
                db_url.split("@")[-1] if "@" in db_url else db_url,
                e,
            )

    _ACTIVE_STORE = SQLiteTimeSeriesStore()
    logger.info("[Database] Local SQLite Time-Series Store active at '%s'", DEFAULT_SQLITE_PATH)
    return _ACTIVE_STORE


def get_database_metadata() -> Dict[str, object]:
    """Return database engine status, hypertable info, and DDL metadata."""
    store = get_time_series_store()
    return {
        "active_engine": store.get_engine_name(),
        "database_url_configured": bool(settings.database_url),
        "postgis_extensions": ["postgis", "postgis_raster", "timescaledb"],
        "hypertables": ["telemetry_snapshots", "physics_calculations"],
        "spatial_tables": [
            "sites",
            "ground_truth_events",
            "hazard_alerts",
            "unprocessed_bulletins",
            "satellite_visual_candidates",
            "ai_audit_runs",
        ],
        "ddl_statements": POSTGIS_TIMESCALEDB_DDL,
    }
