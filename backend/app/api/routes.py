"""FastAPI routes for basins, Infinite Slope physics, ML susceptibility, and cascade alerts."""

from typing import Dict, List
import numpy as np
from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    CascadeAlertResponse,
    CascadeEvaluationRequest,
    InfiniteSlopeRequest,
    InfiniteSlopeResponse,
    SatelliteSynthesisRequest,
    SatelliteSynthesisResponse,
)
from app.core.config import PILOT_BASINS, settings
from app.core.database import get_database_metadata
from app.ingestion.dem_processor import fetch_and_process_copernicus_dem_patch
from app.ingestion.hydro_imerg import evaluate_hydromet_saturation
from app.ingestion.seismic_usgs import evaluate_seismic_trigger
from app.ingestion.stac_sentinel import (
    evaluate_sar_creep_acceleration,
    search_basin_stac_scenes,
)
from app.ml.moisture_downscaler import downscale_smap_moisture_and_regolith_to_30m
from app.ml.pinn_inversion import PINNCohesionDepthInverter
from app.ml.runoff_anomaly import TemporalRunoffAnomalyDetector
from app.ml.susceptibility import LandslideSusceptibilityModel
from app.physics.feature_synthesizer import (
    RawEarthObservationInputs,
    synthesize_physics_variables,
)
from app.physics.historical_replay import replay_historical_cascade
from app.physics.infinite_slope import (
    classify_stability,
    compute_infinite_slope_fs,
    compute_probabilistic_fs,
)
from app.physics.runout_routing import simulate_debris_runout

router = APIRouter(prefix="/api/v1", tags=["Himalayan-MHEWS"])

_susceptibility_model = LandslideSusceptibilityModel(random_state=42)
_anomaly_detector = TemporalRunoffAnomalyDetector()
_pinn_inverter = PINNCohesionDepthInverter(random_state=42)


@router.get("/basins")
def list_pilot_basins() -> Dict[str, object]:
    """Return metadata and bounding boxes for Langtang and Budhi Gandaki pilot basins."""
    return {
        "basins": [
            {
                "id": b.id,
                "name": b.name,
                "district": b.district,
                "bbox": list(b.bbox),
                "center": list(b.center),
                "max_elevation_m": b.max_elevation_m,
                "key_hotspots": list(b.key_hotspots),
            }
            for b in PILOT_BASINS.values()
        ],
        "database": get_database_metadata(),
    }


@router.post("/physics/fs", response_model=InfiniteSlopeResponse)
def calculate_fs_endpoint(payload: InfiniteSlopeRequest) -> InfiniteSlopeResponse:
    """Compute Infinite Slope Factor of Safety (FS) for a given slope cell."""
    fs_val = float(
        compute_infinite_slope_fs(
            c_prime=payload.c_prime,
            gamma=payload.gamma,
            z=payload.z,
            z_w=payload.z_w,
            beta_deg=payload.beta_deg,
            phi_prime_deg=payload.phi_prime_deg,
            seismic_kh=payload.seismic_kh,
        )
    )
    summary = classify_stability(fs_val)
    return InfiniteSlopeResponse(
        fs=round(fs_val, 4),
        critical_failure=bool(fs_val < settings.critical_fs_threshold),
        stability_regime=str(summary["stability_regime"]),
    )


@router.post("/physics/synthesize", response_model=SatelliteSynthesisResponse)
def synthesize_satellite_physics_endpoint(
    payload: SatelliteSynthesisRequest,
) -> SatelliteSynthesisResponse:
    """Synthesize soil depth z, cohesion c', water table z_w, and FS from 18+ raw satellite signals,
    and apply Differential Physics Offset & Neighbor-Block Granulation."""
    raw_dict = payload.model_dump()
    lat = float(raw_dict.pop("latitude", 28.215))
    lon = float(raw_dict.pop("longitude", 85.515))
    local_relief_m = float(raw_dict.pop("local_relief_m", 240.0))

    raw = RawEarthObservationInputs(**raw_dict)
    syn = synthesize_physics_variables(raw)
    fs_val = float(
        compute_infinite_slope_fs(
            c_prime=syn["c_prime_kpa"],
            gamma=syn["gamma_kn_m3"],
            z=syn["z_m"],
            z_w=syn["z_w_m"],
            beta_deg=raw.slope_deg,
            phi_prime_deg=syn["phi_prime_deg"],
            seismic_kh=syn["seismic_kh"],
            surface_surcharge_q_kpa=syn["surface_surcharge_q_kpa"],
            matric_suction_kpa=syn["matric_suction_kpa"],
        )
    )
    hybrid_eval = _susceptibility_model.evaluate_point_hybrid(
        latitude=lat,
        longitude=lon,
        elevation_m=raw.elevation_m,
        slope_deg=raw.slope_deg,
        profile_curvature=raw.profile_curvature,
        local_relief_m=local_relief_m,
        fault_distance_km=raw.distance_to_fault_km,
        raw_synthesized={
            "static_fs": round(fs_val, 4),
            "c_prime_kpa": syn["c_prime_kpa"],
            "soil_depth_z_m": syn["z_m"],
            "twi": syn["twi"],
            "physics_failure_prob": float(1.0 / (1.0 + np.exp(4.5 * (fs_val - 1.15)))),
        },
    )
    calibrated_fs = float(hybrid_eval["calibrated_physics_fs"])
    prob = compute_probabilistic_fs(
        c_prime=float(hybrid_eval["differential_calibration"]["calibrated_c_prime_kpa"])
        + syn["matric_suction_kpa"],
        gamma=syn["gamma_kn_m3"],
        z=float(hybrid_eval["differential_calibration"]["calibrated_z_m"]),
        z_w=syn["z_w_m"],
        beta_deg=raw.slope_deg,
        phi_prime_deg=syn["phi_prime_deg"],
        seismic_kh=syn["seismic_kh"],
    )
    summary = classify_stability(calibrated_fs)
    return SatelliteSynthesisResponse(
        synthesized_variables=syn,
        deterministic_fs=round(fs_val, 4),
        calibrated_fs=round(calibrated_fs, 4),
        hybrid_susceptibility_score=float(hybrid_eval["hybrid_susceptibility_score"]),
        differential_calibration=hybrid_eval["differential_calibration"],
        failure_probability_pct=round(prob["failure_probability"] * 100.0, 1),
        stability_regime=str(summary["stability_regime"]),
    )


@router.get("/ml/differential-blocks")
def get_differential_calibration_blocks() -> Dict[str, object]:
    """Return real-data 0.3° calibration blocks, differential anomalies, and model metrics."""
    if _susceptibility_model.pipeline is None:
        _susceptibility_model.fit_and_evaluate()
    blocks_list = list(_susceptibility_model.corrector.blocks.values())
    blocks_list.sort(key=lambda b: -abs(b["differential_anomaly"]))
    return {
        "trained_on_real_data": _susceptibility_model.trained_on_real_data,
        "metrics": _susceptibility_model.metrics,
        "feature_importances": _susceptibility_model.feature_importances,
        "total_calibrated_blocks": len(blocks_list),
        "top_anomaly_blocks": blocks_list[:15],
    }


@router.get("/accelerators/dem-and-pinn/{basin_id}")
def run_dem_downscaler_and_pinn_pipeline(
    basin_id: str,
    observed_insar_velocity_mm_month: float = 22.0,
    coarse_smap_vwc: float = 0.28,
    live_cog: bool = False,
) -> Dict[str, object]:
    """Run the 30m DEM Raster Processor, 9km->30m SMAP Downscaler, and PINN (c', z) Inverter."""
    if basin_id not in PILOT_BASINS:
        raise HTTPException(status_code=404, detail=f"Unknown basin '{basin_id}'")
    basin = PILOT_BASINS[basin_id]
    lon, lat = basin.center

    dem_patch = fetch_and_process_copernicus_dem_patch(
        center_lat=lat, center_lon=lon, window_size_px=15, live_cog_read=live_cog
    )
    grids = dem_patch["grids"]
    ndvi_grid = np.clip(0.68 - (grids["elevation_m"] - 1500.0) / 6000.0, 0.02, 0.75)
    dt_lst_grid = 14.0 + 4.0 * np.sin(np.deg2rad(grids["aspect_deg"]))

    downscaled = downscale_smap_moisture_and_regolith_to_30m(
        coarse_smap_vwc=coarse_smap_vwc,
        coarse_regolith_depth_m=2.1,
        slope_deg_grid=grids["slope_deg"],
        aspect_deg_grid=grids["aspect_deg"],
        twi_grid=grids["twi"],
        ndvi_grid=ndvi_grid,
        diurnal_lst_range_c_grid=dt_lst_grid,
    )

    center_px = dem_patch["center_pixel"]
    mid = dem_patch["window_shape"][0] // 2
    center_z_prior = float(downscaled["downscaled_regolith_z_30m"][mid, mid])
    center_vwc = float(downscaled["downscaled_vwc_30m"][mid, mid])

    pinn_res = _pinn_inverter.invert_subsurface_parameters(
        slope_deg=max(float(center_px["slope_deg"]), 15.0),
        phi_prime_deg=31.0,
        saturation_ratio=min(center_vwc / 0.42, 0.98),
        c_prior_kpa=11.0,
        z_prior_m=center_z_prior,
        observed_insar_velocity_mm_month=observed_insar_velocity_mm_month,
    )

    return {
        "basin_id": basin.id,
        "dem_patch_source": dem_patch["source"],
        "dem_center_pixel": center_px,
        "dem_grid_summary": dem_patch["grid_summary"],
        "smap_30m_downscaler": {
            "coarse_smap_vwc": downscaled["coarse_smap_vwc"],
            "mean_downscaled_vwc": downscaled["mean_downscaled_vwc"],
            "min_vwc_ridge": downscaled["min_vwc_ridge"],
            "max_vwc_gully": downscaled["max_vwc_gully"],
            "center_downscaled_z_m": round(center_z_prior, 3),
        },
        "pinn_subsurface_inversion": pinn_res,
    }


@router.post("/alerts/evaluate", response_model=CascadeAlertResponse)
def evaluate_cascade_hazard(payload: CascadeEvaluationRequest) -> CascadeAlertResponse:
    """Fuse Tier A static susceptibility with Tier B dynamic precursors into an alert level."""
    if payload.basin_id not in PILOT_BASINS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown basin '{payload.basin_id}'. Valid basins: {list(PILOT_BASINS.keys())}",
        )
    basin = PILOT_BASINS[payload.basin_id]

    # 1. Tier A Dry/Baseline Static FS (z_w = 0.2 * z, no seismic)
    static_fs = float(
        compute_infinite_slope_fs(
            c_prime=payload.c_prime,
            gamma=payload.gamma,
            z=payload.z,
            z_w=0.2 * payload.z,
            beta_deg=payload.beta_deg,
            phi_prime_deg=payload.phi_prime_deg,
        )
    )

    # 2. Tier B Precursor 1: Sentinel-1 SAR Slope/Glacier Creep
    sar_eval = evaluate_sar_creep_acceleration(payload.sar_displacement_mm)

    # 3. Tier B Precursor 2: Sentinel-2 NDWI Glacial Lake Expansion
    glof_trigger = bool(
        payload.lake_expansion_pct > settings.glof_area_anomaly_pct_threshold
    )

    # 4. Tier B Precursor 3: Hydrometeorological Saturation (IMERG 72h + >5000m PDD)
    hydro_eval = evaluate_hydromet_saturation(
        hourly_rainfall_mm_72h=payload.hourly_rainfall_mm_72h,
        daily_temps_c_above_5000m=payload.daily_temps_c_above_5000m,
        elevations_m=payload.station_elevations_m,
        slab_depth_m=payload.z,
    )

    # 5. Tier B Precursor 4: Seismic Trigger
    seismic_eval = evaluate_seismic_trigger(
        events=[{"magnitude": payload.max_seismic_magnitude}]
    )

    # 6. Dynamic FS incorporating hydromet water table zw and seismic kh
    dynamic_fs = float(
        compute_infinite_slope_fs(
            c_prime=payload.c_prime,
            gamma=payload.gamma,
            z=payload.z,
            z_w=float(hydro_eval["estimated_zw_m"]),
            beta_deg=payload.beta_deg,
            phi_prime_deg=payload.phi_prime_deg,
            seismic_kh=float(seismic_eval["estimated_seismic_kh"]),
        )
    )

    # 7. ML Susceptibility & Temporal Runoff Anomaly Score
    feature_vec = np.array(
        [[payload.beta_deg, 4800.0, -0.4, 8.5, 0.18, 3.2, dynamic_fs]],
        dtype=np.float64,
    )
    ml_score = float(_susceptibility_model.predict_susceptibility(feature_vec)[0])
    runoff_anom = _anomaly_detector.score_observation(
        rainfall_mm=float(hydro_eval["cumulative_rain_72h_mm"]) / 3.0,
        pdd_c_days=float(hydro_eval["pdd_c_days_above_5000m"]),
        discharge_m3s=85.0 + 1.8 * float(hydro_eval["total_water_input_mm"]),
    )

    active_triggers: List[str] = []
    if sar_eval["trigger_active"]:
        active_triggers.append("SAR_SLOPE_CREEP_ACCELERATION")
    if glof_trigger:
        active_triggers.append("SENTINEL2_NDWI_GLACIAL_LAKE_EXPANSION")
    if hydro_eval["trigger_active"]:
        active_triggers.append("IMERG_72H_PLUS_5000M_PDD_SATURATION")
    if seismic_eval["trigger_active"]:
        active_triggers.append("SEISMIC_TRIGGER_M4_PLUS")

    # Determine alert level per AGENTS.md matrix
    if dynamic_fs < settings.critical_fs_threshold and len(active_triggers) >= 1:
        alert_level = "RED"
    elif dynamic_fs < 1.05 or len(active_triggers) >= 2:
        alert_level = "ORANGE"
    elif dynamic_fs < settings.watch_fs_threshold or len(active_triggers) == 1:
        alert_level = "YELLOW"
    else:
        alert_level = "GREEN"

    # Simulate 30m DEM debris-flow runout when FS < 1.05
    runout_summary = None
    if dynamic_fs < 1.05:
        grid_r, grid_c = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20), indexing="ij")
        synthetic_dem = 5200.0 - 1600.0 * grid_r + 120.0 * np.abs(grid_c - 0.5)
        source_mask = np.zeros_like(synthetic_dem, dtype=bool)
        source_mask[1, 9:11] = True
        ro = simulate_debris_runout(synthetic_dem, source_mask, cell_size_m=30.0)
        runout_summary = {
            "runout_length_m": ro["runout_length_m"],
            "peak_velocity_m_s": ro["peak_velocity_m_s"],
            "affected_cells_30m": ro["affected_cells"],
        }

    return CascadeAlertResponse(
        basin_id=basin.id,
        basin_name=basin.name,
        alert_level=alert_level,
        static_fs=round(static_fs, 4),
        dynamic_fs=round(dynamic_fs, 4),
        ml_susceptibility_score=ml_score,
        active_triggers=active_triggers,
        precursor_details={
            "sar_creep": sar_eval,
            "glacial_lake_expansion_pct": payload.lake_expansion_pct,
            "hydromet_saturation": hydro_eval,
            "seismic": seismic_eval,
            "temporal_runoff_anomaly": runoff_anom,
        },
        runout_summary=runout_summary,
    )


@router.get("/basins/{basin_id}/grid-geojson")
def get_basin_susceptibility_geojson(basin_id: str) -> Dict[str, object]:
    """Generate 30m-style GeoJSON susceptibility & runout cells for Deck.gl 3D visualization."""
    if basin_id not in PILOT_BASINS:
        raise HTTPException(status_code=404, detail=f"Unknown basin '{basin_id}'")

    basin = PILOT_BASINS[basin_id]
    min_lon, min_lat, max_lon, max_lat = basin.bbox
    lons = np.linspace(min_lon + 0.08, max_lon - 0.08, 8)
    lats = np.linspace(min_lat + 0.06, max_lat - 0.06, 8)

    features: List[Dict[str, object]] = []
    cell_idx = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            cell_idx += 1
            slope_deg = 22.0 + 3.2 * ((i + j) % 7)
            elev_m = 2100.0 + 480.0 * i + 120.0 * (j % 3)
            zw_ratio = 0.35 + 0.08 * ((i * j) % 6)
            fs_val = float(
                compute_infinite_slope_fs(
                    c_prime=9.5,
                    gamma=19.8,
                    z=2.4,
                    z_w=2.4 * zw_ratio,
                    beta_deg=slope_deg,
                    phi_prime_deg=31.0,
                )
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(float(lon), 5), round(float(lat), 5), round(elev_m, 1)],
                    },
                    "properties": {
                        "cell_id": f"{basin.id}_{cell_idx}",
                        "elevation_m": round(elev_m, 1),
                        "slope_deg": round(slope_deg, 1),
                        "fs": round(fs_val, 3),
                        "critical": bool(fs_val < 1.0),
                    },
                }
            )

    return {
        "type": "FeatureCollection",
        "basin_id": basin.id,
        "stac_scenes": search_basin_stac_scenes(basin.id, live_query=False),
        "features": features,
    }


@router.get("/alerts/replay/{scenario_id}")
def get_historical_cascade_replay(scenario_id: str) -> Dict[str, object]:
    """Replay historical Himalayan hazard cascades (`langtang_2015` or `budhi_gandaki_monsoon`)."""
    try:
        return replay_historical_cascade(scenario_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
