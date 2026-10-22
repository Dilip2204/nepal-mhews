"""Integration tests for FastAPI routes (basins, FS calculator, cascade alert engine)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_and_basins_endpoints():
    """Verify /health and /api/v1/basins return Langtang and Budhi Gandaki configs."""
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_basins = client.get("/api/v1/basins")
    assert r_basins.status_code == 200
    basin_ids = [b["id"] for b in r_basins.json()["basins"]]
    assert "langtang" in basin_ids
    assert "budhi_gandaki" in basin_ids


def test_physics_fs_and_cascade_alert_endpoints():
    """Verify /api/v1/physics/fs and /api/v1/alerts/evaluate endpoints."""
    r_fs = client.post(
        "/api/v1/physics/fs",
        json={
            "c_prime": 4.0,
            "gamma": 20.0,
            "z": 3.0,
            "z_w": 2.8,
            "beta_deg": 42.0,
            "phi_prime_deg": 28.0,
            "seismic_kh": 0.15,
        },
    )
    assert r_fs.status_code == 200
    data_fs = r_fs.json()
    assert data_fs["fs"] < 1.0
    assert data_fs["critical_failure"] is True

    r_eval = client.post(
        "/api/v1/alerts/evaluate",
        json={
            "basin_id": "langtang",
            "c_prime": 5.0,
            "gamma": 20.0,
            "z": 2.8,
            "beta_deg": 40.0,
            "phi_prime_deg": 29.0,
            "sar_displacement_mm": [0.0, 2.5, 12.0, 29.0],
            "hourly_rainfall_mm_72h": [2.5] * 72,
            "daily_temps_c_above_5000m": [4.5, 5.0, 4.2],
            "station_elevations_m": [5200.0, 5450.0, 5700.0],
            "max_seismic_magnitude": 4.5,
            "lake_expansion_pct": 8.4,
        },
    )
    assert r_eval.status_code == 200
    alert_data = r_eval.json()
    assert alert_data["alert_level"] == "RED"
    assert alert_data["dynamic_fs"] < 1.0
    assert len(alert_data["active_triggers"]) == 4
    assert alert_data["runout_summary"] is not None
