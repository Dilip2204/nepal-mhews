"""Unit tests for Dynamic Precursor Triggers 1-4 (Sentinel-1/2, IMERG+PDD, Seismic)."""

import numpy as np

from app.ingestion.dem_processor import (
    fetch_and_process_copernicus_dem_patch,
    process_dem_grid_derivatives,
)
from app.ingestion.hydro_imerg import (
    compute_pdd_snowmelt_above_5000m,
    evaluate_hydromet_saturation,
)
from app.ingestion.seismic_usgs import (
    detect_rock_ice_avalanche_signature,
    evaluate_seismic_trigger,
)
from app.ingestion.stac_sentinel import (
    compute_ndwi,
    evaluate_cloud_resilient_lake_expansion,
    evaluate_glacial_lake_expansion,
    evaluate_sar_creep_acceleration,
)


def test_sar_creep_acceleration_trigger():
    """Trigger 1: Acceleration > 10 mm/month across consecutive 12-day passes."""
    # Pass 0->1: +2 mm in 12d (5 mm/mo); Pass 1->2: +8 mm in 12d (20 mm/mo) -> accel = +15 mm/mo
    res = evaluate_sar_creep_acceleration([0.0, 2.0, 10.0], pass_interval_days=12.0)
    assert res["latest_acceleration_mm_month"] == 15.0
    assert res["trigger_active"] is True

    # Steady creep: +2 mm every 12d -> 0 acceleration
    steady = evaluate_sar_creep_acceleration([0.0, 2.0, 4.0, 6.0], pass_interval_days=12.0)
    assert steady["latest_acceleration_mm_month"] == 0.0
    assert steady["trigger_active"] is False


def test_sentinel2_ndwi_glacial_lake_expansion():
    """Trigger 2: Rapid positive anomaly in NDWI = (B03 - B08) / (B03 + B08)."""
    b03_base = np.full((10, 10), 0.15)
    b08_base = np.full((10, 10), 0.35)
    # 20 water pixels in baseline
    b03_base[:2, :] = 0.45
    b08_base[:2, :] = 0.10

    b03_curr = b03_base.copy()
    b08_curr = b08_base.copy()
    # Expand lake to 30 water pixels (+50% expansion)
    b03_curr[2, :] = 0.45
    b08_curr[2, :] = 0.10

    ndwi = compute_ndwi(b03_curr, b08_curr)
    assert np.all(ndwi[:3, :] > 0.25)

    expansion = evaluate_glacial_lake_expansion(b03_base, b08_base, b03_curr, b08_curr)
    assert expansion["expansion_pct"] == 50.0
    assert expansion["trigger_active"] is True

    # Test monsoon cloud fallback (>50% clouds switches to Sentinel-1 SAR < -16 dB)
    sar_vv_db = np.full((10, 10), -8.0)
    sar_vv_db[:3, :] = -19.5  # 30 water pixels detected through clouds
    sar_fallback = evaluate_cloud_resilient_lake_expansion(
        b03_base,
        b08_base,
        b03_curr,
        b08_curr,
        cloud_cover_pct=85.0,
        sar_vv_db_current=sar_vv_db,
    )
    assert sar_fallback["sensor_mode"] == "SENTINEL1_SAR_BACKSCATTER_FALLBACK"
    assert sar_fallback["expansion_pct"] == 50.0
    assert sar_fallback["trigger_active"] is True


def test_hydromet_72h_rainfall_and_5000m_pdd():
    """Trigger 3: 72-hour NASA IMERG rainfall + >5000m PDD snowmelt index."""
    pdd = compute_pdd_snowmelt_above_5000m(
        temperatures_c=np.array([4.0, 3.0, -2.0, 6.0]),
        elevations_m=np.array([5200.0, 5400.0, 5600.0, 4200.0]),  # last station < 5000m ignored
    )
    # Only first two stations >= 5000m have T > 0: 4.0 + 3.0 = 7.0 C-days
    assert pdd["pdd_c_days"] == 7.0
    assert pdd["snowmelt_mm_we"] == 7.0 * 4.5

    sat = evaluate_hydromet_saturation(
        hourly_rainfall_mm_72h=[2.0] * 72,  # 144 mm
        daily_temps_c_above_5000m=[4.0, 3.5, 2.5],
        elevations_m=[5100.0, 5350.0, 5600.0],
    )
    assert sat["cumulative_rain_72h_mm"] == 144.0
    assert sat["trigger_active"] is True


def test_seismic_tectonic_and_rock_ice_waveform():
    """Trigger 4: M4.0+ tectonic event or 2-15 Hz spindle rock-ice detachment waveform."""
    rng = np.random.default_rng(7)
    fs = 100.0
    t = np.arange(0, 20.0, 1.0 / fs)
    # Spindle-shaped 6 Hz rock-ice avalanche signal with low background noise
    spindle = np.sin(np.pi * t / 20.0) ** 2
    carrier = np.sin(2 * np.pi * 6.0 * t) + 0.5 * np.sin(2 * np.pi * 9.5 * t)
    waveform = 0.05 * rng.normal(size=t.size) + 1.5 * spindle * carrier

    wf_diag = detect_rock_ice_avalanche_signature(waveform, sampling_rate_hz=fs)
    assert wf_diag["high_freq_energy_ratio"] > 0.45
    assert wf_diag["is_rock_ice_detachment"] is True

    trig = evaluate_seismic_trigger(
        events=[{"magnitude": 4.6, "place": "Langtang Valley"}],
        waveform=waveform,
        sampling_rate_hz=fs,
    )
    assert trig["tectonic_trigger_active"] is True
    assert trig["trigger_active"] is True


def test_dem_raster_processor_derivatives_and_d8_twi():
    """Verify 30m DEM processor computes slope, aspect, curvature, D8 flow area, and TWI."""
    patch = fetch_and_process_copernicus_dem_patch(
        center_lat=28.215, center_lon=85.515, window_size_px=15, live_cog_read=False
    )
    grids = patch["grids"]
    assert grids["elevation_m"].shape == (15, 15)
    assert np.all(grids["slope_deg"] >= 0.0)
    assert np.all((grids["aspect_deg"] >= 0.0) & (grids["aspect_deg"] <= 360.0))
    # Valley center column (index 7) must accumulate more D8 upslope area than ridge edges (index 0)
    assert np.mean(grids["upslope_contributing_area_m2"][:, 7]) > np.mean(
        grids["upslope_contributing_area_m2"][:, 0]
    )
    assert np.mean(grids["twi"][:, 7]) > np.mean(grids["twi"][:, 0])
