-- Himalayan-MHEWS PostGIS + TimescaleDB Production DDL
-- Multi-Node Spatial & Time-Series Engine for Cascading Mountain Hazards

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 1. Sites of Interest Master Table (491+ Points across Nepal)
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

CREATE INDEX IF NOT EXISTS idx_sites_geom ON sites USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_sites_basin ON sites (basin_id);

-- 2. 22-Sensor Ingestion Telemetry Snapshots Hypertable
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

SELECT create_hypertable('telemetry_snapshots', 'ts', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
CREATE INDEX IF NOT EXISTS idx_telemetry_site_ts ON telemetry_snapshots (site_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_geom ON telemetry_snapshots USING GIST (geom);

-- 3. Physics & Dynamic Stability Calculations Hypertable
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

SELECT create_hypertable('physics_calculations', 'ts', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
CREATE INDEX IF NOT EXISTS idx_physics_site_ts ON physics_calculations (site_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_physics_alert_ts ON physics_calculations (alert_level, ts DESC);
CREATE INDEX IF NOT EXISTS idx_physics_geom ON physics_calculations USING GIST (geom);

-- 4. Live Ground-Truth Feedback Events Catalog
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

CREATE INDEX IF NOT EXISTS idx_gt_geom ON ground_truth_events USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_gt_label ON ground_truth_events (label);
CREATE INDEX IF NOT EXISTS idx_gt_source ON ground_truth_events (source);

-- 5. Cascade Alerts & Civil Protection Broadcasts
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

CREATE INDEX IF NOT EXISTS idx_alerts_level_time ON hazard_alerts (alert_level, created_at DESC);
