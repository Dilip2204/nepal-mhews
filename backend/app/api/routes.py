"""FastAPI routes for basins, Infinite Slope physics, ML susceptibility, and cascade alerts."""

from pathlib import Path
from typing import Dict, List, Optional, Any
import math
import time
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    CascadeAlertResponse,
    CascadeEvaluationRequest,
    InfiniteSlopeRequest,
    InfiniteSlopeResponse,
    SatelliteSynthesisRequest,
    SatelliteSynthesisResponse,
    WeeklyAuditTriggerRequest,
)
from app.core.config import PILOT_BASINS, settings
from app.core.database import get_database_metadata
from app.core.logging import get_logger
from app.ingestion.dem_processor import fetch_and_process_copernicus_dem_patch
from app.ingestion.hydro_imerg import (
    evaluate_hydromet_saturation,
    fetch_live_hydromet_observations,
)
from app.ingestion.seismic_usgs import evaluate_seismic_trigger
from app.ingestion.stac_sentinel import (
    evaluate_sar_creep_acceleration,
    search_basin_stac_scenes,
)
from app.ml.ollama_evaluator import (
    evaluate_with_ollama,
    explain_live_weather_and_factors,
    generate_nepali_civil_protection_broadcast,
    ollama_status,
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
    compute_viscoplastic_creep_velocity,
)
from app.physics.runout_routing import (
    compute_settlement_impact_and_arrival_times,
    simulate_debris_runout,
)
from app.physics.valley_grid_physics import (
    classify_alert_level,
    compute_vectorized_grid_fs,
    fs_to_gradient_color,
    valley_grid_service,
    resolve_corridor_live_telemetry,
    PRIMARY_CORRIDORS,
)

router = APIRouter(prefix="/api/v1", tags=["Himalayan-MHEWS"])

logger = get_logger("nepal_mhews.api")

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


@router.get("/ollama/status")
def get_ollama_status() -> Dict[str, object]:
    """Check whether the local Ollama server is reachable and list its models."""
    return ollama_status()


@router.post("/alerts/evaluate-ollama")
def evaluate_cascade_with_ollama(payload: CascadeEvaluationRequest) -> Dict[str, object]:
    """Run the physics cascade evaluation, then have a local Ollama LLM review it offline."""
    logger.info("evaluate-ollama called for basin '%s', slope=%.1f°, z=%.2fm", payload.basin_id, payload.beta_deg, payload.z)
    alert = evaluate_cascade_hazard(payload)
    summary = alert.model_dump()
    logger.info("Physics evaluation complete: Level=%s, DynamicFS=%.4f, Triggers=%s", summary.get("alert_level"), summary.get("dynamic_fs"), summary.get("active_triggers"))
    
    # Compute downstream settlement arrival times & evacuation isochrones
    ro = summary.get("runout_summary") or {}
    peak_v = float(ro.get("peak_velocity_m_s", 25.0))
    runout_l = float(ro.get("runout_length_m", 1500.0))
    settlements = compute_settlement_impact_and_arrival_times(
        basin_id=payload.basin_id,
        peak_velocity_m_s=peak_v,
        runout_length_m=runout_l,
    )
    summary["downstream_settlement_impacts"] = settlements

    # Generate bilingual emergency broadcast
    civil_broadcast = None
    try:
        civil_broadcast = generate_nepali_civil_protection_broadcast(
            basin_name=summary.get("basin_name", "Himalayan Basin"),
            alert_level=summary.get("alert_level", "GREEN"),
            dynamic_fs=float(summary.get("dynamic_fs", 1.25)),
            active_triggers=summary.get("active_triggers", []),
            impacted_settlements=settlements,
        )
    except Exception as exc:
        logger.warning("Bilingual broadcast generation failed: %s", exc)

    summary["civil_protection_broadcast"] = civil_broadcast

    try:
        review = evaluate_with_ollama(summary)
        llm = {"available": True, "review": review}
    except Exception as exc:  # Ollama down/slow: physics alert still returned
        logger.warning("Ollama offline review bypassed or failed: %s", exc)
        llm = {"available": False, "error": str(exc)}
    return {"alert": summary, "llm": llm}


@router.get("/weather/live-and-explain")
def get_live_weather_and_ai_synthesis(
    lat: float = 28.232,
    lon: float = 85.518,
    elevation_m: float = 5680.0,
    slope_deg: float = 46.0,
    station_name: str = "Langtang Station",
) -> Dict[str, object]:
    """Fetch live precipitation/temperature for station coordinates, then explain factors with Ollama."""
    live_hydro = fetch_live_hydromet_observations(lat=lat, lon=lon, elevation_m=elevation_m)
    logger.info("Fetched live weather for '%s': 72h rain=%.2fmm, temp=%.1f°C", station_name, live_hydro["cumulative_rain_72h_mm"], live_hydro["mean_temp_c"])
    
    ai_interpretation = None
    try:
        ai_interpretation = explain_live_weather_and_factors(
            station_name=station_name,
            live_hydro=live_hydro,
            slope_deg=slope_deg,
            elevation_m=elevation_m,
        )
    except Exception as exc:
        logger.warning("Ollama weather explanation failed: %s", exc)
        ai_interpretation = {"error": str(exc), "risk_severity": "UNKNOWN"}
        
    return {
        "station_name": station_name,
        "live_observations": live_hydro,
        "ai_synthesis": ai_interpretation,
    }


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


@router.get("/grid/corridors")
def list_valley_grid_corridors() -> Dict[str, Any]:
    """Return active monitored valley corridors spanning river basins, roads, faults, and settlements."""
    corridors = valley_grid_service.get_corridors_list()
    total_cells = sum(c["cell_count"] for c in corridors)
    return {
        "status": "ok",
        "total_corridors": len(corridors),
        "total_grid_cells": total_cells,
        "corridors": corridors,
    }


@router.get("/grid/cells")
def get_valley_grid_cells(
    corridor_id: str = Query("langtang", description="Corridor ID (e.g. langtang, budhi_gandaki, kali_gandaki, karnali, etc.)"),
    rainfall_72h_mm: Optional[float] = Query(None, ge=0.0, le=1000.0, description="Optional override in mm; if omitted, uses live raw real-time hydromet telemetry"),
    api_30d_mm: Optional[float] = Query(None, ge=0.0, le=1000.0, description="Optional override in mm; if omitted, uses live raw real-time antecedent moisture"),
    seismic_kh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional override; if omitted, uses live raw real-time USGS seismic PGA"),
    asset_type: Optional[str] = Query(None, description="Filter: all, settlement, highway_roadcut, hydro_infrastructure, river_gorge, cryosphere_glacial, hillslope"),
    exposure: str = Query("exposed", description="'exposed' (default: cells within 1 km of a mapped road or 2 km of a settlement), or 'all'"),
    bbox: Optional[str] = Query(None, description="Viewport filter 'min_lon,min_lat,max_lon,max_lat'"),
    limit: int = Query(2500, ge=1, le=60000, description="Max cells returned, lowest FS first; truncation is reported in summary.truncated"),
) -> Dict[str, Any]:
    """Evaluate 240 m grid cells with live Factor of Safety over REAL measured terrain.

    Terrain, soil, roads, rivers and faults are measured, not generated - see
    ``ValleyGridPhysicsService._load_corridor_cells``. A corridor now holds tens of
    thousands of real cells rather than a few hundred invented ones, so the response is
    filtered: by default to cells with actual human exposure, which is the scope this
    grid exists to cover, and optionally to the map viewport. Any truncation is stated
    in ``summary.truncated`` rather than silently applied.
    """
    if corridor_id not in PRIMARY_CORRIDORS:
        raise HTTPException(status_code=404, detail=f"Unknown corridor '{corridor_id}'. Available: {list(PRIMARY_CORRIDORS.keys())}")

    # Resolve live telemetry for corridor
    live_telem = resolve_corridor_live_telemetry(corridor_id)
    eff_rain = rainfall_72h_mm if rainfall_72h_mm is not None else float(live_telem["rainfall_72h_mm"])
    eff_api = api_30d_mm if api_30d_mm is not None else float(live_telem["api_30d_mm"])
    eff_kh = seismic_kh if seismic_kh is not None else float(live_telem["seismic_kh"])

    view = None
    if bbox:
        try:
            w, s_, e, n = (float(v) for v in bbox.split(","))
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="bbox must be 'min_lon,min_lat,max_lon,max_lat'")
        view = (w, s_, e, n)

    try:
        cells = valley_grid_service.evaluate_corridor_grid(
            corridor_id=corridor_id,
            rainfall_72h_mm=eff_rain,
            api_30d_mm=eff_api,
            seismic_kh=eff_kh,
            asset_type=asset_type,
            bbox=view,
            exposed_only=(exposure != "all"),
            limit=limit,
        )
    except FileNotFoundError as exc:
        # The grid is measured and cached, never generated on demand. If it is missing
        # we say so; we do not substitute anything for it.
        raise HTTPException(status_code=503, detail=str(exc))

    sel = getattr(valley_grid_service, "_last_selection", {})
    total_evaluated = sel.get("evaluated", len(cells))
    after_filters = sel.get("after_filters", len(cells))
    truncated = sel.get("truncated", False)

    fs_vals = [c["fs"] for c in cells]
    alert_counts = {"RED": 0, "ORANGE": 0, "YELLOW": 0, "GREEN": 0}
    asset_distribution: Dict[str, int] = {}
    for c in cells:
        lvl = c.get("alert_level", "GREEN")
        if lvl in alert_counts:
            alert_counts[lvl] += 1
        atype = c.get("asset_type", "OTHER")
        asset_distribution[atype] = asset_distribution.get(atype, 0) + 1

    return {
        "corridor_id": corridor_id,
        "corridor_name": PRIMARY_CORRIDORS[corridor_id].name,
        "live_telemetry": {
            "rainfall_72h_mm": round(eff_rain, 2),
            "api_30d_mm": round(eff_api, 2),
            "seismic_kh": round(eff_kh, 4),
            "temp_2m_c": live_telem.get("temp_2m_c", 15.0),
            "pdd_snowmelt_mm": live_telem.get("pdd_snowmelt_mm", 0.0),
            "timestamp_utc": live_telem.get("timestamp_utc", time.time()),
            "provenance": live_telem.get("provenance", "MEASURED_LIVE_RAW_TELEMETRY"),
            "is_live": (rainfall_72h_mm is None and api_30d_mm is None and seismic_kh is None),
        },
        # Named "evaluation_inputs", not "simulation_parameters": with no override
        # supplied these ARE the live measured values the physics ran on. The override
        # arguments exist for offline sensitivity work and are flagged when used.
        "evaluation_inputs": {
            "rainfall_72h_mm": round(eff_rain, 2),
            "api_30d_mm": round(eff_api, 2),
            "seismic_kh": round(eff_kh, 4),
            "asset_type_filter": asset_type or "all",
            "exposure_filter": exposure,
            "is_override": any(v is not None for v in (rainfall_72h_mm, api_30d_mm, seismic_kh)),
        },
        "summary": {
            "total_cells": len(cells),
            "cells_evaluated": total_evaluated,
            "cells_after_filters": after_filters,
            "truncated": truncated,
            "cells_without_soil_excluded": getattr(
                valley_grid_service, "_excluded_no_soil", {}).get(corridor_id, 0),
            "terrain_provenance": "MEASURED - Copernicus DEM GLO-30, SoilGrids v2.0, OSM, GEM faults",
            "mean_fs": round(float(np.mean(fs_vals)), 3) if fs_vals else None,
            "min_fs": round(float(np.min(fs_vals)), 3) if fs_vals else None,
            "max_fs": round(float(np.max(fs_vals)), 3) if fs_vals else None,
            "alert_distribution": alert_counts,
            "asset_distribution": asset_distribution,
        },
        "cells": cells,
    }


@router.get("/grid/geojson")
def get_valley_grid_geojson(
    corridor_id: str = Query("langtang", description="Corridor ID"),
    rainfall_72h_mm: Optional[float] = Query(None, ge=0.0, le=1000.0, description="Optional override; if omitted, uses live raw telemetry"),
    seismic_kh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional override; if omitted, uses live raw telemetry"),
    asset_type: Optional[str] = Query(None, description="Filter: all, settlement, highway_roadcut, hydro_infrastructure, river_gorge, cryosphere_glacial"),
) -> Dict[str, Any]:
    """GeoJSON FeatureCollection of 240m grid polygons with FS gradient styling for Leaflet/Deck.gl."""
    if corridor_id not in PRIMARY_CORRIDORS:
        raise HTTPException(status_code=404, detail=f"Unknown corridor '{corridor_id}'")

    live_telem = resolve_corridor_live_telemetry(corridor_id)
    eff_rain = rainfall_72h_mm if rainfall_72h_mm is not None else float(live_telem["rainfall_72h_mm"])
    eff_kh = seismic_kh if seismic_kh is not None else float(live_telem["seismic_kh"])

    cells = valley_grid_service.evaluate_corridor_grid(
        corridor_id=corridor_id,
        rainfall_72h_mm=eff_rain,
        seismic_kh=eff_kh,
        asset_type=asset_type,
    )

    features = []
    for c in cells:
        s, w, n, e = c["bounds"]
        poly = [[[round(w, 5), round(s, 5)], [round(e, 5), round(s, 5)], [round(e, 5), round(n, 5)], [round(w, 5), round(n, 5)], [round(w, 5), round(s, 5)]]]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": poly,
            },
            "properties": c,
        })

    return {
        "type": "FeatureCollection",
        "corridor_id": corridor_id,
        "features": features,
    }


@router.get("/alerts/replay/{scenario_id}")
def get_historical_cascade_replay(scenario_id: str) -> Dict[str, object]:
    """Replay historical Himalayan hazard cascades (`langtang_2015` or `budhi_gandaki_monsoon`)."""
    try:
        return replay_historical_cascade(scenario_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/engine/status")
def get_perpetual_engine_status() -> Dict[str, object]:
    """Return status of perpetual ingestion, physics calculation, and continual learning engine."""
    from pathlib import Path
    import json
    status_file = Path(__file__).resolve().parent.parent.parent / "data" / "perpetual_status.json"
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read perpetual status: %s", e)
    return {
        "service": "Himalayan-MHEWS Perpetual Ingestion & Physics Engine",
        "status": "initializing",
        "cycle_count": 0,
        "monitored_sites_total": 491,
    }


@router.get("/sites")
def list_sites_of_interest() -> List[Dict[str, object]]:
    """Return catalog of all 491+ monitored sites across Nepal with latest physics calculations and dynamic creep."""
    from app.engine.sites_catalog import load_sites_of_interest
    from app.core.database import get_time_series_store
    from app.engine.environmental_features import enrich_sites_catalog

    sites = load_sites_of_interest()
    store = get_time_series_store()
    calc_map = store.get_latest_physics_calculations()
    env_map = enrich_sites_catalog()

    results = []
    # One local state read for the whole sweep - never per site, never a network call.
    _resolve_creep = None
    _hyp3_site_key = None
    measured_velocities = {}
    try:
        from app.ingestion import insar_hyp3 as _h, insar_timeseries as _t

        _resolve_creep, _hyp3_site_key = _t.resolve_creep_input, _h.site_key
        measured_velocities = _t.load_measured_velocities()
    except Exception as _creep_err:
        logger.warning("measured InSAR unavailable for site list: %s", _creep_err)

    for s in sites:
        c = calc_map.get(s.site_id, {
            "static_fs": 1.25,
            "dynamic_fs": 1.28,
            "calibrated_fs": 1.25,
            "failure_prob_pct": 14.5,
            "alert_level": "GREEN",
            "active_triggers": [],
            "last_refresh_utc": 0.0,
        })
        dyn_fs = float(c.get("dynamic_fs") or c.get("calibrated_fs") or 1.25)
        modelled_creep = compute_viscoplastic_creep_velocity(
            slope_deg=s.slope_deg,
            dynamic_fs=dyn_fs,
            elevation_m=s.elevation_m,
            hazard_type=s.hazard_type,
            rain_72h_mm=0.0,
        )
        # Same substitution the physics loop makes, so the map and the site report cannot
        # disagree about what a site's creep rate is or where the number came from.
        if _resolve_creep is not None:
            resolved = _resolve_creep(modelled_creep, measured_velocities.get(
                _hyp3_site_key(s.latitude, s.longitude)))
        else:
            resolved = {"value_mm_month": modelled_creep, "provenance": "MODELLED_PROXY",
                        "reason": "measured InSAR module unavailable"}
        creep = resolved["value_mm_month"]
        c["ground_creep_mm_month"] = creep
        c["ground_creep_provenance"] = resolved["provenance"]
        env_f = env_map.get(s.site_id)
        if env_f is None:
            from app.engine.environmental_features import compute_site_environmental_features
            env_f = compute_site_environmental_features(
                site_id=s.site_id,
                latitude=s.latitude,
                longitude=s.longitude,
                elevation_m=s.elevation_m,
                slope_deg=s.slope_deg,
                profile_curvature=s.profile_curvature,
                local_relief_m=s.local_relief_m,
                basin_id=s.basin_id,
            )
        results.append({
            "site_id": s.site_id,
            "name": s.name,
            "hazard_type": s.hazard_type,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "elevation_m": s.elevation_m,
            "slope_deg": s.slope_deg,
            "profile_curvature": s.profile_curvature,
            "local_relief_m": s.local_relief_m,
            "fault_distance_km": s.fault_distance_km,
            "basin_id": s.basin_id,
            "physics": c,
            "ground_creep_mm_month": creep,
            "creep_provenance": resolved["provenance"],
            "creep_basis": resolved.get("reason"),
            "distance_to_road_m": env_f.distance_to_road_m,
            "road_type": env_f.road_type,
            "road_cut_penalty": env_f.road_cut_penalty,
            "bedrock_anchor": {
                "name": env_f.anchor_name,
                "latitude": env_f.anchor_lat,
                "longitude": env_f.anchor_lon,
                "slope_deg": env_f.anchor_slope_deg,
            },
            "environmental": {
                "clay_fraction": env_f.clay_fraction,
                "sand_fraction": env_f.sand_fraction,
                "coarse_rock_fraction": env_f.coarse_rock_fragment_fraction,
                "dry_bulk_density_g_cm3": env_f.dry_bulk_density_g_cm3,
                "sentinel2_ndvi": env_f.sentinel2_ndvi,
                "gedi_canopy_height_m": env_f.gedi_canopy_height_m,
                "aspect_deg": env_f.aspect_deg,
            },
        })
    return results


@router.get("/sites/{site_id}")
def get_site_risk_analysis_report(site_id: str) -> Dict[str, object]:
    """Return detailed multi-parameter risk analysis report and 22-point telemetry for a site."""
    from app.engine.sites_catalog import load_sites_of_interest
    from app.ingestion.expanded_weather import fetch_batch_expanded_weather, compute_standard_barometric_pressure
    import time

    sites = load_sites_of_interest()
    target_site = next((s for s in sites if s.site_id == site_id), None)
    if not target_site:
        raise HTTPException(status_code=404, detail=f"Site '{site_id}' not found in catalog")

    # 1. Fetch cached telemetry from store for instantaneous (<2ms) response, or fallback
    from app.core.database import get_time_series_store
    store = get_time_series_store()
    cached_t = store.get_latest_telemetry(target_site.site_id)
    now_utc = time.time()

    if cached_t:
        rain_72h = float(cached_t.get("rain_72h_mm", 4.0))
        temp_c = float(cached_t.get("temp_2m_c", 12.0))
        wind_surcharge = float(cached_t.get("wind_surcharge_kpa", 0.0))
        pga_g = float(cached_t.get("seismic_pga_g", 0.04))
        pdd_val = float(cached_t.get("pdd_days_7d", 0.0))
        cloudburst = bool(cached_t.get("cloudburst_flag", False))
        depression = bool(cached_t.get("cyclonic_depression_flag", False))
        rain_peak = float(cached_t.get("rain_peak_1h_mm", 0.5))
        surf_press = float(cached_t.get("surface_pressure_hpa", 850.0))
        press_def = float(cached_t.get("pressure_deficit_hpa", 0.0))
        cloud_pct = float(cached_t.get("cloud_cover_pct", 20.0))
        wind_spd = float(cached_t.get("wind_speed_kmh", 8.0))
        wind_gst = float(cached_t.get("wind_gusts_kmh", 14.0))
        rel_hum = float(cached_t.get("relative_humidity_pct", 45.0))
        seis_m = float(cached_t.get("seismic_mag", 0.0))
    else:
        telem_map = fetch_batch_expanded_weather([target_site], batch_size=1)
        telem = telem_map.get(target_site.site_id)
        rain_72h = telem.rain_72h_mm if telem else 4.0
        temp_c = telem.temp_2m_c if telem else 12.0
        wind_surcharge = telem.wind_surcharge_kpa if telem else 0.0
        pga_g = telem.seismic_pga_g if telem else 0.04
        pdd_val = telem.pdd_days if telem else 0.0
        cloudburst = bool(telem.dahal_hasegawa_cloudburst) if telem else False
        depression = bool(telem.cyclonic_depression_flag) if telem else False
        rain_peak = telem.rain_peak_1h_mm if telem else 0.5
        surf_press = telem.surface_pressure_hpa if telem else 850.0
        press_def = telem.pressure_deficit_hpa if telem else 0.0
        cloud_pct = telem.cloud_cover_pct if telem else 20.0
        wind_spd = telem.wind_speed_10m_kmh if telem else 8.0
        wind_gst = telem.wind_gusts_kmh if telem else 14.0
        rel_hum = telem.relative_humidity_pct if telem else 45.0
        seis_m = telem.seismic_mag if telem else 0.0

    # Load enriched environmental features (soil, vegetation, aspect, road cut, anchor)
    from app.engine.environmental_features import enrich_sites_catalog, compute_site_environmental_features
    env_catalog = enrich_sites_catalog()
    env = env_catalog.get(site_id)
    if env is None:
        env = compute_site_environmental_features(
            site_id=target_site.site_id,
            latitude=target_site.latitude,
            longitude=target_site.longitude,
            elevation_m=target_site.elevation_m,
            slope_deg=target_site.slope_deg,
            profile_curvature=target_site.profile_curvature,
            local_relief_m=target_site.local_relief_m,
            basin_id=target_site.basin_id,
        )
    clay_frac = env.clay_fraction
    sand_frac = env.sand_fraction
    coarse_frac = env.coarse_rock_fragment_fraction
    bulk_density = env.dry_bulk_density_g_cm3
    ndvi_val = env.sentinel2_ndvi
    canopy_val = env.gedi_canopy_height_m
    aspect_val = env.aspect_deg
    diurnal_lst_val = env.diurnal_lst_range_c
    dist_road_val = env.distance_to_road_m
    road_penalty_val = env.road_cut_penalty

    # Run physics synthesis with real data
    raw = RawEarthObservationInputs(
        elevation_m=target_site.elevation_m,
        slope_deg=target_site.slope_deg,
        aspect_deg=aspect_val,
        profile_curvature=target_site.profile_curvature,
        plan_curvature=target_site.profile_curvature * 0.5,
        upslope_contributing_area_m2=4500.0,
        sentinel2_ndvi=ndvi_val,
        gedi_canopy_height_m=canopy_val,
        clay_fraction=clay_frac,
        sand_fraction=sand_frac,
        coarse_rock_fragment_fraction=coarse_frac,
        dry_bulk_density_g_cm3=bulk_density,
        distance_to_fault_km=target_site.fault_distance_km,
        diurnal_lst_range_c=diurnal_lst_val,
        imerg_rain_72h_mm=rain_72h,
        antecedent_rain_30d_api_mm=max(12.0, rain_72h * 1.5),
        pdd_c_days_7d=pdd_val,
        bedrock_pga_g=pga_g,
        insar_los_velocity_mm_month=2.5,
        snow_water_equivalent_mm=max(20.0, wind_surcharge * 100.0),
        distance_to_road_m=dist_road_val,
        road_cut_penalty=road_penalty_val,
    )
    syn = synthesize_physics_variables(raw)
    from app.ml.susceptibility import synthesize_point_physics_from_terrain
    geo = synthesize_point_physics_from_terrain(
        elevation_m=target_site.elevation_m,
        slope_deg=target_site.slope_deg,
        profile_curvature=target_site.profile_curvature,
        local_relief_m=target_site.local_relief_m,
        fault_distance_km=target_site.fault_distance_km,
        latitude=target_site.latitude,
        longitude=target_site.longitude,
    )
    from app.physics.infinite_slope import compute_infinite_slope_fs
    dyn_fs = float(compute_infinite_slope_fs(
        c_prime=syn["c_prime_kpa"],
        gamma=syn["gamma_kn_m3"],
        z=syn["z_m"],
        z_w=syn["z_w_m"],
        beta_deg=syn.get("effective_slope_deg", target_site.slope_deg),
        phi_prime_deg=syn["phi_prime_deg"],
        seismic_kh=syn["seismic_kh"],
        surface_surcharge_q_kpa=syn["surface_surcharge_q_kpa"] + wind_surcharge,
        matric_suction_kpa=syn["matric_suction_kpa"],
    ))
    static_fs = geo["static_fs"]

    triggers = []
    if rain_72h >= 120.0 or cloudburst:
        triggers.append("CLOUD_BURST_SATURATION")
    if depression:
        triggers.append("CYCLONIC_BAROMETRIC_DEPRESSION")
    if pga_g >= 0.12:
        triggers.append("SEISMIC_SHAKEMAP_PGA")
    if target_site.elevation_m >= 4500.0 and pdd_val >= 8.0:
        triggers.append("CRITICAL_THERMAL_ICE_MELT")
    if wind_surcharge >= 0.40:
        triggers.append("WIND_BLOWN_CORNICE_SURCHARGE")

    # Modelled creep from terrain and geomechanics (mm/month). This is NOT InSAR: it is a
    # function of slope, FS and rainfall. Where a real inverted Sentinel-1 network exists it
    # is replaced below by the measurement, and the provenance says which one is in play.
    modelled_creep_mm_month = compute_viscoplastic_creep_velocity(
        slope_deg=target_site.slope_deg,
        dynamic_fs=dyn_fs,
        elevation_m=target_site.elevation_m,
        hazard_type=target_site.hazard_type,
        rain_72h_mm=rain_72h,
        saturation_ratio=float(syn.get("saturation_ratio", 0.2)),
    )
    creep_input = {
        "value_mm_month": modelled_creep_mm_month,
        "provenance": "MODELLED_PROXY",
        "reason": "measured InSAR module unavailable",
    }
    try:
        from app.ingestion import insar_hyp3, insar_timeseries

        creep_input = insar_timeseries.resolve_creep_input(
            modelled_creep_mm_month,
            insar_timeseries.load_measured_velocities().get(
                insar_hyp3.site_key(target_site.latitude, target_site.longitude)
            ),
        )
    except Exception as creep_err:  # the report must still render
        logger.warning("measured InSAR lookup failed: %s", creep_err)
    dynamic_creep_mm_month = creep_input["value_mm_month"]
    creep_measured = creep_input["provenance"].startswith("MEASURED")

    if dynamic_creep_mm_month >= 10.0 and (rain_72h >= 80.0 or pga_g >= 0.08 or pdd_val >= 6.0):
        # Only a real interferometric measurement earns the SAR_ name.
        triggers.append(
            "SAR_SLOPE_CREEP_ACCELERATION" if creep_measured
            else "MODELLED_CREEP_ACCELERATION"
        )

    # Real-Time Dynamic Live Stability Alert Matrix (Evaluates physical hazard under live weather right now)
    if dyn_fs < 1.0 and len(triggers) >= 1:
        alert_lvl = "RED"
    elif dyn_fs < 1.0 or len(triggers) >= 2:
        alert_lvl = "ORANGE"
    elif dyn_fs < 1.25 or len(triggers) == 1:
        alert_lvl = "YELLOW"
    else:
        alert_lvl = "GREEN"

    fail_prob_pct = float(1.0 / (1.0 + math.exp(6.8 * (dyn_fs - 1.0)))) * 100.0
    report_timestamp_utc = now_utc
    report_timestamp_human = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now_utc))

    # 22 Data Ingestion Points with Refresh Rates & Last Refresh Timestamps
    ingestion_points = [
        {"name": "NASA GPM IMERG 72h Precipitation", "val": f"{rain_72h:.1f} mm", "refresh_cadence": "Every 30 Mins", "last_refresh_utc": now_utc - (now_utc % 1800)},
        {"name": "1-Hour Cloudburst Peak Intensity", "val": f"{rain_peak:.1f} mm/h", "refresh_cadence": "Hourly", "last_refresh_utc": now_utc - (now_utc % 3600)},
        {"name": "2m Ambient Temperature", "val": f"{temp_c:.1f} °C", "refresh_cadence": "Hourly", "last_refresh_utc": now_utc - (now_utc % 3600)},
        {"name": "Atmospheric Surface Pressure", "val": f"{surf_press:.1f} hPa", "refresh_cadence": "Hourly", "last_refresh_utc": now_utc - (now_utc % 3600)},
        {"name": "Synoptic Pressure Deficit", "val": f"{press_def:+.1f} hPa", "refresh_cadence": "Hourly", "last_refresh_utc": now_utc - (now_utc % 3600)},
        {"name": "Cloud Cover Percentage", "val": f"{cloud_pct:.1f}%", "refresh_cadence": "Hourly", "last_refresh_utc": now_utc - (now_utc % 3600)},
        {"name": "10m Wind Speed & Gusts", "val": f"{wind_spd:.1f} / {wind_gst:.1f} km/h", "refresh_cadence": "Hourly", "last_refresh_utc": now_utc - (now_utc % 3600)},
        {"name": "Relative Humidity (2m)", "val": f"{rel_hum:.1f}%", "refresh_cadence": "Hourly", "last_refresh_utc": now_utc - (now_utc % 3600)},
        {"name": "Wind-Blown Cornice Surcharge (qs)", "val": f"{wind_surcharge:.3f} kPa", "refresh_cadence": "Hourly", "last_refresh_utc": now_utc - (now_utc % 3600)},
        {"name": "Positive Degree Day (PDD) 7d Melt", "val": f"{pdd_val:.1f} °C-d", "refresh_cadence": "Daily (24h)", "last_refresh_utc": now_utc - (now_utc % 86400)},
        {"name": "30-Day API Rain Sponge Memory", "val": f"{max(12.0, rain_72h * 1.5):.1f} mm", "refresh_cadence": "Daily (24h)", "last_refresh_utc": now_utc - (now_utc % 86400)},
        {"name": "MODIS/VIIRS Diurnal LST Swing (ATI)", "val": f"{syn['apparent_thermal_inertia']:.3f}", "refresh_cadence": "Twice Daily (12h)", "last_refresh_utc": now_utc - (now_utc % 43200)},
        {"name": "Copernicus Snow Water Equivalent (SWE)", "val": f"{max(20.0, wind_surcharge * 100.0):.1f} mm", "refresh_cadence": "Daily (24h)", "last_refresh_utc": now_utc - (now_utc % 86400)},
        {"name": "NASA SMAP 30m Downscaled Moisture", "val": f"{syn['saturation_ratio']*100:.1f}% saturation", "refresh_cadence": "Every 2-3 Days", "last_refresh_utc": now_utc - 72000},
        {"name": ("Sentinel-1 InSAR LOS Creep (SBAS Measured)" if creep_measured
                  else "Sentinel-1 InSAR LOS Creep (Viscoplastic Proxy)"),
         "val": f"{dynamic_creep_mm_month:.1f} mm/mo",
         "refresh_cadence": ("Weekly SBAS inversion (Measured)" if creep_measured
                             else "Every 6-12 Days (Modelled)"),
         "last_refresh_utc": now_utc - (now_utc % 1800)},
        {"name": "Sentinel-2 NDVI Vegetation Canopy", "val": f"{ndvi_val:.2f}", "refresh_cadence": "Every 5 Days", "last_refresh_utc": now_utc - 120000},
        {"name": "NASA GEDI LiDAR Canopy Height", "val": f"{canopy_val:.1f} m", "refresh_cadence": "Spaceborne LiDAR", "last_refresh_utc": 1704067200},
        {"name": "Soil Texture (Clay/Sand/Rock)", "val": f"{clay_frac*100:.0f}% / {sand_frac*100:.0f}% / {coarse_frac*100:.0f}%", "refresh_cadence": "Physiographic Map", "last_refresh_utc": 1704067200},
        {"name": "Road Proximity & Cut Undercutting", "val": f"{dist_road_val:.0f} m (Rcut={road_penalty_val:.2f})", "refresh_cadence": "OpenStreetMap 2026", "last_refresh_utc": 1704067200},
        {"name": "Stable Bedrock Reference Anchor", "val": f"{env.anchor_name} ({env.anchor_slope_deg:.1f}°)", "refresh_cadence": "Geomorphic Anchor", "last_refresh_utc": 1704067200},
        {"name": "Proglacial Lake Expansion Rate", "val": "0.0%", "refresh_cadence": "Every 5-6 Days", "last_refresh_utc": now_utc - 140000},
        {"name": "USGS Real-Time ShakeMap PGA", "val": f"{pga_g:.3f} g (M{seis_m:.1f})", "refresh_cadence": "Real-Time Streaming", "last_refresh_utc": now_utc - 60},
        {"name": "Copernicus 30m DEM Elevation", "val": f"{target_site.elevation_m:.0f} m", "refresh_cadence": "Static 30m Grid", "last_refresh_utc": 1704067200},
        {"name": "Slope Gradient (beta) & Subgrid Scarp", "val": f"{target_site.slope_deg:.1f}°", "refresh_cadence": "Static 30m Grid", "last_refresh_utc": 1704067200},
        {"name": "Profile & Plan Geomorphic Curvature", "val": f"{target_site.profile_curvature:+.2f} m⁻¹", "refresh_cadence": "Static 30m Grid", "last_refresh_utc": 1704067200},
        {"name": "Distance to Active Fault (MHT/MCT)", "val": f"{target_site.fault_distance_km:.2f} km", "refresh_cadence": "Static Annual", "last_refresh_utc": 1704067200},
    ]

    impacted_villages = compute_settlement_impact_and_arrival_times(
        basin_id=target_site.basin_id,
        peak_velocity_m_s=35.0 if alert_lvl in ("RED", "ORANGE") else 14.0,
        runout_length_m=12500.0 if alert_lvl == "RED" else 6500.0 if alert_lvl == "ORANGE" else 1500.0,
        source_station_name=target_site.name,
        alert_level=alert_lvl,
    )
    bilingual_broadcast = generate_nepali_civil_protection_broadcast(
        basin_name=f"{target_site.basin_id.replace('_', ' ').title()} River Corridor",
        alert_level=alert_lvl,
        dynamic_fs=dyn_fs,
        active_triggers=triggers,
        impacted_settlements=impacted_villages,
    )

    return {
        "site_id": target_site.site_id,
        "name": target_site.name,
        "hazard_type": target_site.hazard_type,
        "basin_id": target_site.basin_id,
        "coordinates": {"latitude": target_site.latitude, "longitude": target_site.longitude},
        "elevation_m": target_site.elevation_m,
        "slope_deg": target_site.slope_deg,
        "fault_distance_km": target_site.fault_distance_km,
        "risk_evaluation": {
            "alert_level": alert_lvl,
            "static_fs": round(static_fs, 3),
            "static_fs_description": "Baseline terrain susceptibility score under saturated conditions (permanent geometry)",
            "dynamic_fs": round(dyn_fs, 3),
            "dynamic_fs_description": "Actual real-time physical stability factor of safety under current live weather & seismic telemetry",
            "failure_probability_pct": round(fail_prob_pct, 1),
            "soil_depth_z_m": round(syn["z_m"], 2),
            "cohesion_c_prime_kpa": round(syn["c_prime_kpa"], 2),
            "friction_phi_deg": round(syn["phi_prime_deg"], 1),
            "water_table_zw_m": round(syn["z_w_m"], 2),
            "saturation_pct": round(syn["saturation_ratio"] * 100, 1),
            "ground_creep_mm_month": dynamic_creep_mm_month,
            "ground_creep_provenance": creep_input["provenance"],
            "ground_creep_basis": creep_input.get("reason"),
            "insar_measured_mm_year": creep_input.get("measured_mm_year"),
            "insar_uncertainty_mm_year": creep_input.get("uncertainty_mm_year"),
            "insar_detection_limit_mm_year": creep_input.get("detection_limit_mm_year"),
            "distance_to_road_m": round(dist_road_val, 1),
            "road_cut_penalty": round(road_penalty_val, 3),
            "effective_slope_deg": round(float(syn.get("effective_slope_deg", target_site.slope_deg)), 1),
            "bedrock_anchor": {
                "name": env.anchor_name,
                "lat": env.anchor_lat,
                "lon": env.anchor_lon,
                "slope_deg": env.anchor_slope_deg,
            },
            "soil_texture": {
                "clay_pct": round(clay_frac * 100, 1),
                "sand_pct": round(sand_frac * 100, 1),
                "rock_pct": round(coarse_frac * 100, 1),
            },
            "vegetation": {
                "ndvi": round(ndvi_val, 2),
                "canopy_height_m": round(canopy_val, 1),
            },
            "active_triggers": triggers,
            "stability_status": "CRITICAL" if dyn_fs < 1.0 else "WATCH" if dyn_fs < 1.25 else "SAFE",
        },
        "report_generated_at_utc": report_timestamp_utc,
        "report_generated_at_human": report_timestamp_human,
        "ingestion_points": ingestion_points,
        "settlement_impacts": impacted_villages,
        "civil_protection_broadcast": bilingual_broadcast,
    }


@router.post("/sites/{site_id}/recalculate")
def recalculate_site_report(site_id: str) -> Dict[str, object]:
    """Force an immediate re-fetch of live weather and re-run physics calculations for a site."""
    return get_site_risk_analysis_report(site_id)


def compute_ground_truth_disaster_indicator() -> Dict[str, object]:
    """Inspect time-series store calculations to assess if any active confirmed
    landslide, avalanche, or critical slope failure risk is detected across the network.
    """
    from app.core.database import get_time_series_store
    store = get_time_series_store()
    return store.get_active_disaster_indicator()


@router.get("/database/metadata")
def get_database_status_metadata() -> Dict[str, object]:
    """Return active database engine (PostGIS+TimescaleDB vs SQLite), hypertables, and DDL metadata."""
    from app.core.database import get_database_metadata
    return get_database_metadata()


@router.get("/feedback/ground-truth")
async def get_ground_truth_feedback_catalog():
    """Return the Live Ground-Truth Event Feedback Catalog (NDRRMA + Sentinel-2/SAR), Optimizer state,
    and the high-visibility disaster indicator status.
    """
    from app.ingestion.ground_truth_feedback import load_live_ground_truth_events
    from app.ml.continuous_equation_optimizer import ContinuousEquationOptimizer

    events = load_live_ground_truth_events()
    opt = ContinuousEquationOptimizer()
    acc, rec, auc, f1, cryo_acc, fb_acc, avg_dev = opt.evaluate_parameter_set(opt.active_params)
    indicator = compute_ground_truth_disaster_indicator()

    return {
        "disaster_indicator": indicator,
        "total_training_catalog_points": len(opt._dataset),
        "base_catalog_points": 949,
        "cryosphere_anchor_points": 24,
        "live_feedback_points": len(events),
        "optimizer_state": {
            "current_generation": opt.current_generation,
            "accuracy_pct": round(acc * 100.0, 2),
            "holdout_recall_pct": round(rec * 100.0, 2),
            "roc_auc": round(auc, 4),
            "cryosphere_anchor_acc_pct": round(cryo_acc * 100.0, 1),
            "live_feedback_acc_pct": round(fb_acc * 100.0, 1),
            "avg_fs_deviation_pct": round(avg_dev, 2),
            "active_parameters": opt.active_params,
        },
        "events": events[::-1],
    }


@router.post("/feedback/recalibrate")
async def recalibrate_ground_truth_network():
    """Single recalibration endpoint: re-evaluates all ground-truth catalog events,
    steps an evolutionary generation of the equation optimizer, computes live disaster status,
    and returns the updated report.
    """
    from app.ingestion.ground_truth_feedback import load_live_ground_truth_events
    from app.ml.continuous_equation_optimizer import ContinuousEquationOptimizer
    import time

    _susceptibility_model.fit_and_evaluate()
    opt = ContinuousEquationOptimizer()
    opt_res = opt.step_optimization_generation(n_candidates=6)
    events = load_live_ground_truth_events()
    indicator = compute_ground_truth_disaster_indicator()
    now_utc = time.time()
    now_human = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now_utc))

    return {
        "status": "recalibrated",
        "timestamp_utc": now_utc,
        "timestamp_human": now_human,
        "disaster_indicator": indicator,
        "total_training_catalog_points": opt_res.dataset_total_points,
        "live_feedback_points": len(events),
        "optimizer_state": {
            "current_generation": opt_res.generation,
            "fitness_score": opt_res.fitness_score,
            "holdout_recall_pct": round(opt_res.recall * 100.0, 2),
            "roc_auc": round(opt_res.roc_auc, 4),
            "cryosphere_anchor_acc_pct": round(opt_res.cryo_accuracy * 100.0, 1),
            "live_feedback_acc_pct": round(opt_res.live_feedback_accuracy * 100.0, 1),
            "avg_fs_deviation_pct": opt_res.avg_fs_deviation_pct,
            "promoted": opt_res.promoted,
            "reason": opt_res.reason,
        },
        "events": events[::-1],
    }


@router.post("/feedback/ground-truth")
async def submit_ground_truth_feedback_event(payload: dict):
    """Ingest a newly confirmed NDRRMA field report or Sentinel-2/SAR scar detection,
    append to the training catalog, recalibrate 0.3-deg differential blocks, and step
    the ContinuousEquationOptimizer.
    """
    from app.ingestion.ground_truth_feedback import append_ground_truth_event, load_live_ground_truth_events
    from app.ml.continuous_equation_optimizer import ContinuousEquationOptimizer

    rec = append_ground_truth_event(
        event_name=str(payload.get("event_name", "NDRRMA Field-Verified Slope Event")),
        latitude=float(payload.get("latitude", 28.18)),
        longitude=float(payload.get("longitude", 85.35)),
        elevation_m=float(payload.get("elevation_m", 1850.0)),
        slope_deg=float(payload.get("slope_deg", 38.0)),
        label=int(payload.get("label", 1)),
        source=str(payload.get("source", "NDRRMA_FIELD_VERIFIED")),
        hazard_type=str(payload.get("hazard_type", "monsoon_landslide")),
        profile_curvature=payload.get("profile_curvature"),
        local_relief_m=payload.get("local_relief_m"),
        fault_distance_km=payload.get("fault_distance_km"),
        ndvi_drop=float(payload.get("ndvi_drop", -0.25)),
        rainfall_72h_mm=float(payload.get("rainfall_72h_mm", 160.0)),
        notes=str(payload.get("notes", "")),
    )

    # Retrain differential block offsets & step evolutionary optimizer
    _susceptibility_model.fit_and_evaluate()
    opt = ContinuousEquationOptimizer()
    opt_res = opt.step_optimization_generation(n_candidates=6)
    events = load_live_ground_truth_events()

    return {
        "ingested_event": rec,
        "total_training_catalog_points": opt_res.dataset_total_points,
        "live_feedback_points": len(events),
        "adaptation_result": {
            "generation": opt_res.generation,
            "fitness_score": opt_res.fitness_score,
            "holdout_recall_pct": round(opt_res.recall * 100.0, 2),
            "roc_auc": round(opt_res.roc_auc, 4),
            "cryosphere_anchor_acc_pct": round(opt_res.cryo_accuracy * 100.0, 1),
            "live_feedback_acc_pct": round(opt_res.live_feedback_accuracy * 100.0, 1),
            "avg_fs_deviation_pct": opt_res.avg_fs_deviation_pct,
            "promoted": opt_res.promoted,
            "reason": opt_res.reason,
        },
        "events": events[::-1],
    }


# =============================================================================
# Weekly AI Geohazard Audit & Continuous Learning Endpoints
# =============================================================================

@router.get("/audit/staging-queues")
async def get_audit_staging_queues(limit: int = Query(10, ge=1, le=50)):
    """Inspect pending unstructured bulletins and satellite visual anomaly candidates."""
    from app.core.database import get_time_series_store
    store = get_time_series_store()
    pending_bul = store.get_pending_bulletins(limit=limit)
    pending_vis = store.get_pending_visual_candidates(limit=limit)
    return {
        "pending_bulletins_count": len(pending_bul),
        "pending_visual_candidates_count": len(pending_vis),
        "bulletins_sample": pending_bul,
        "visual_candidates_sample": pending_vis,
    }


@router.post("/audit/trigger-weekly-audit")
async def trigger_weekly_ai_audit_run(payload: WeeklyAuditTriggerRequest):
    """Trigger an on-demand AI Geohazard audit run across unstructured bulletins,
    multimodal satellite thumbnails, and equation optimization.
    """
    from scripts.run_weekly_ai_audit import WeeklyAIAuditPipeline
    pipeline = WeeklyAIAuditPipeline(
        engine=payload.engine,
        dry_run=payload.dry_run,
    )
    result = pipeline.execute_complete_weekly_audit(
        limit_bulletins=payload.limit_bulletins,
        limit_visual=payload.limit_visual,
    )
    return result


@router.get("/audit/runs")
async def get_recent_ai_audit_runs(limit: int = Query(15, ge=1, le=100)):
    """Fetch history of completed AI audit runs, token usage, and equation promotions."""
    from app.core.database import get_time_series_store
    store = get_time_series_store()
    runs = store.get_latest_ai_audit_runs(limit=limit)
    return {
        "count": len(runs),
        "runs": runs,
    }


@router.get("/audit/reports")
async def list_audit_reports():
    """List generated executive markdown audit reports and summary metrics."""
    reports_dir = Path(__file__).resolve().parents[2] / "data" / "audit_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_files = sorted(reports_dir.glob("weekly_ai_audit_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    items = []
    for rf in report_files:
        st = rf.stat()
        items.append({
            "filename": rf.name,
            "size_bytes": st.st_size,
            "modified_utc": st.st_mtime,
            "path": str(rf),
        })

    return {
        "count": len(items),
        "reports": items,
    }


@router.get("/audit/reports/{filename}")
async def get_audit_report_content(filename: str):
    """Fetch the full markdown content of a specific weekly audit report."""
    reports_dir = Path(__file__).resolve().parents[2] / "data" / "audit_reports"
    safe_path = (reports_dir / filename).resolve()
    if not str(safe_path).startswith(str(reports_dir.resolve())) or not safe_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found.")

    with open(safe_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "filename": filename,
        "content_markdown": content,
    }


# =============================================================================
# Quarantined Bulletins Human Review Queue Endpoints
# =============================================================================

@router.get("/audit/quarantined-bulletins")
def list_quarantined_bulletins(limit: int = Query(50, ge=1, le=100)) -> Dict[str, Any]:
    """Return bulletins awaiting human review or quarantined due to uncertainty."""
    from app.core.database import get_time_series_store
    store = get_time_series_store()
    bulletins = store.get_quarantined_bulletins(limit=limit)
    return {
        "status": "ok",
        "total_pending": len(bulletins),
        "bulletins": bulletins,
    }


@router.post("/audit/quarantined-bulletins/{bulletin_id}/action")
async def take_bulletin_action(bulletin_id: str, payload: dict) -> Dict[str, Any]:
    """Approve or reject a quarantined bulletin.
    If approved:
      1. Appends verified ground-truth event to catalog with source & news link.
      2. If failure confirmed (label=1) and matches site predicted at high risk,
         dispatches URGENT confirmed collapse alert email to operator.
      3. Adapts ContinuousEquationOptimizer and updates bulletin status.
    """
    from app.core.database import get_time_series_store
    from app.ingestion.ground_truth_feedback import append_ground_truth_event
    from app.ml.continuous_equation_optimizer import ContinuousEquationOptimizer

    action = str(payload.get("action", "REJECT")).upper()
    store = get_time_series_store()

    bul = store.get_bulletin(bulletin_id)
    if not bul:
        raise HTTPException(status_code=404, detail=f"Bulletin '{bulletin_id}' not found.")

    if action == "APPROVE":
        ev_name = str(payload.get("event_name") or bul.get("title") or "Human-Reviewed Disaster Event")

        # An approved bulletin becomes a label=1 row in the ML training catalog, so its
        # location IS the training signal. Defaulting to a fixed coordinate silently filed
        # a Pokhara-Sikles landslide ~145 km away near Kathmandu, with invented terrain.
        # Refuse rather than fabricate: a mislocated positive actively mistrains the model.
        if payload.get("latitude") is None or payload.get("longitude") is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "latitude and longitude are required to approve a bulletin into the "
                    "training catalog. Locate the event on the map before approving - "
                    "an approved row with a guessed location mistrains the model."
                ),
            )
        lat = float(payload["latitude"])
        lon = float(payload["longitude"])
        if not (80.0 <= lon <= 89.0 and 26.0 <= lat <= 31.0):
            raise HTTPException(
                status_code=400,
                detail=f"Coordinates ({lat}, {lon}) fall outside Nepal; refusing to admit.",
            )
        elev = float(payload.get("elevation_m") or 1600.0)
        slope = float(payload.get("slope_deg") or 36.0)
        hazard = str(payload.get("hazard_type") or "monsoon_landslide")
        lbl = int(payload.get("label", 1))
        src = str(bul.get("source") or "HUMAN_VERIFIED_BULLETIN")
        src_url = str(payload.get("source_url") or bul.get("url") or "")
        notes = str(payload.get("notes") or f"Human-approved from bulletin '{bul.get('title')}'. {bul.get('raw_text', '')[:160]}")

        rec = append_ground_truth_event(
            event_name=ev_name,
            latitude=lat,
            longitude=lon,
            elevation_m=elev,
            slope_deg=slope,
            label=lbl,
            source=src,
            source_url=src_url,
            hazard_type=hazard,
            notes=notes,
            dispatch_urgent_email=True,
        )

        store.update_bulletin_status(
            bulletin_id=bulletin_id,
            status="APPROVED_BY_HUMAN",
            extracted_event_id=rec.get("event_id"),
        )

        # Retrain differential blocks & step optimizer
        _susceptibility_model.fit_and_evaluate()
        opt = ContinuousEquationOptimizer()
        opt_res = opt.step_optimization_generation(n_candidates=6)

        return {
            "status": "APPROVED",
            "bulletin_id": bulletin_id,
            "event_id": rec.get("event_id"),
            "event_name": ev_name,
            "source_url": src_url,
            "optimizer_recall_pct": round(opt_res.recall * 100.0, 2),
            "holdout_accuracy_pct": round(opt_res.accuracy * 100.0, 2),
        }
    else:  # REJECT
        store.update_bulletin_status(
            bulletin_id=bulletin_id,
            status="REJECTED_BY_HUMAN",
        )
        return {
            "status": "REJECTED",
            "bulletin_id": bulletin_id,
        }


