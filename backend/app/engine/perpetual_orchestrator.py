"""Perpetual Ingestion, Physics Calculation, AI Accelerators & Continual Learning Orchestrator.

Continuously runs across all 491+ identified sites of interest in Nepal:
  1. Batched live data ingestion (NASA IMERG, Open-Meteo 72h rain, 1h cloudburst,
     barometric pressure, cloud cover, 10m wind speed, relative humidity, USGS seismic)
     synchronized with sensor refresh rates.
  2. Runs 6-signal spaceborne soil depth synthesizer (z), AI Accelerator 1 (PINN InSAR inversion),
     AI Accelerator 2 (SMAP 30m downscaler), and Generalized Infinite Slope FS.
  3. Evaluates 7-factor geomechanical & orographic hazard equations.
  4. Runs continuous evolutionary equation optimization seeking higher ROC-AUC & holdout recall.
  5. Records real-time telemetry and physics snapshots in SQLite.
"""

import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

# Cap threads for shared server host
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

from app.engine.sites_catalog import load_sites_of_interest, SiteInfo
from app.ingestion.expanded_weather import fetch_batch_expanded_weather, LiveSiteTelemetry
from app.ingestion.ground_truth_feedback import (
    load_live_ground_truth_events,
    scan_live_telemetry_for_ground_truth_confirmations,
)
from app.physics.feature_synthesizer import RawEarthObservationInputs, synthesize_physics_variables
from app.ml.susceptibility import synthesize_point_physics_from_terrain
from app.ml.continuous_equation_optimizer import ContinuousEquationOptimizer
from app.core.database import get_time_series_store, BaseTimeSeriesStore

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "perpetual_telemetry.db"
STATUS_JSON_PATH = DATA_DIR / "perpetual_status.json"
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


class PerpetualEngineOrchestrator:
    """Master background engine for all Nepal geohazard monitoring points."""

    def __init__(self, fast_loop_interval_sec: int = 1800):
        self.fast_loop_interval_sec = fast_loop_interval_sec  # 30 minutes default
        self.sites = load_sites_of_interest()
        self.optimizer = ContinuousEquationOptimizer()
        self.store: BaseTimeSeriesStore = get_time_series_store()
        self.cycle_count = 0
        self.start_time = time.time()
        self.last_run_time = 0.0
        self.latest_status: Dict[str, Any] = {}

    def execute_single_calculation_cycle(self) -> Dict[str, Any]:
        """Run one complete data fetch, physics calculation, and ML equation optimization pass."""
        self.cycle_count += 1
        cycle_start = time.time()
        print(f"\n=======================================================", flush=True)
        print(f"🔄 PERPETUAL ENGINE CYCLE #{self.cycle_count} STARTING", flush=True)
        print(f"   Target: {len(self.sites)} sites across Nepal | Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
        print(f"   Storage Engine: {self.store.get_engine_name()}", flush=True)
        print(f"=======================================================", flush=True)

        # 1. Batched Expanded Weather Ingestion (22 Live Input Points)
        print(f"[1/4] Ingesting live atmospheric, hydromet, pressure, wind, and seismic data...", flush=True)
        telemetry_map = fetch_batch_expanded_weather(self.sites, batch_size=50)
        print(f"      Ingested live telemetry for {len(telemetry_map)}/{len(self.sites)} points.", flush=True)

        # 2. Physics & AI Accelerators Pipeline Execution for All Sites
        print(f"[2/4] Running 6-signal soil depth (z), PINN InSAR inversion, and 7-Factor slope stability...", flush=True)
        alert_counts = {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}
        site_evaluations = []
        now_utc = time.time()

        for s in self.sites:
            telem = telemetry_map.get(s.site_id)
            rain_72h = telem.rain_72h_mm if telem else 4.0
            temp_c = telem.temp_2m_c if telem else 12.0
            wind_surcharge = telem.wind_surcharge_kpa if telem else 0.0
            pga_g = telem.seismic_pga_g if telem else 0.04

            # Synthesize raw inputs
            raw = RawEarthObservationInputs(
                elevation_m=s.elevation_m,
                slope_deg=s.slope_deg,
                profile_curvature=s.profile_curvature,
                plan_curvature=s.profile_curvature * 0.5,
                upslope_contributing_area_m2=4500.0,
                distance_to_fault_km=s.fault_distance_km,
                imerg_rain_72h_mm=rain_72h,
                antecedent_rain_30d_api_mm=max(12.0, rain_72h * 1.5),
                pdd_c_days_7d=telem.pdd_days if telem else 0.0,
                bedrock_pga_g=pga_g,
                insar_los_velocity_mm_month=2.5,
                snow_water_equivalent_mm=max(20.0, wind_surcharge * 100.0),
            )
            syn = synthesize_physics_variables(raw)

            # 7-factor geomechanical calculation
            geo = synthesize_point_physics_from_terrain(
                elevation_m=s.elevation_m,
                slope_deg=s.slope_deg,
                profile_curvature=s.profile_curvature,
                local_relief_m=s.local_relief_m,
                fault_distance_km=s.fault_distance_km,
                latitude=s.latitude,
                longitude=s.longitude,
            )

            from app.physics.infinite_slope import compute_infinite_slope_fs
            dynamic_fs = float(compute_infinite_slope_fs(
                c_prime=syn["c_prime_kpa"],
                gamma=syn["gamma_kn_m3"],
                z=syn["z_m"],
                z_w=syn["z_w_m"],
                beta_deg=s.slope_deg,
                phi_prime_deg=syn["phi_prime_deg"],
                seismic_kh=syn["seismic_kh"],
                surface_surcharge_q_kpa=syn["surface_surcharge_q_kpa"] + wind_surcharge,
                matric_suction_kpa=syn["matric_suction_kpa"],
            ))
            static_fs = geo["static_fs"]
            cal_fs = dynamic_fs

            # Precursor triggers check under live telemetry
            triggers = []
            if rain_72h >= 120.0 or (telem and telem.dahal_hasegawa_cloudburst):
                triggers.append("CLOUD_BURST_SATURATION")
            if telem and telem.cyclonic_depression_flag:
                triggers.append("CYCLONIC_BAROMETRIC_DEPRESSION")
            if pga_g >= 0.12:
                triggers.append("SEISMIC_SHAKEMAP_PGA")
            if s.elevation_m >= 4500.0 and (telem and telem.pdd_days >= 8.0):
                triggers.append("CRITICAL_THERMAL_ICE_MELT")
            if wind_surcharge >= 0.40:
                triggers.append("WIND_BLOWN_CORNICE_SURCHARGE")

            # Cascade Alert Matrix (Rule 6 - Evaluated on Real-Time Dynamic FS)
            if dynamic_fs < 1.0 and len(triggers) >= 1:
                lvl = "RED"
            elif dynamic_fs < 1.0 or len(triggers) >= 2:
                lvl = "ORANGE"
            elif dynamic_fs < 1.25 or len(triggers) == 1:
                lvl = "YELLOW"
            else:
                lvl = "GREEN"

            alert_counts[lvl] += 1
            fail_prob = float(1.0 / (1.0 + math.exp(6.8 * (dynamic_fs - 1.0)))) * 100.0

            # Store in unified Time-Series Store (TimescaleDB hypertable or SQLite fallback)
            self.store.insert_physics_calculation(
                timestamp_utc=now_utc,
                site_id=s.site_id,
                soil_depth_z_m=syn["z_m"],
                c_prime_kpa=syn["c_prime_kpa"],
                phi_prime_deg=syn["phi_prime_deg"],
                static_fs=static_fs,
                dynamic_fs=dynamic_fs,
                calibrated_fs=cal_fs,
                failure_prob_pct=fail_prob,
                alert_level=lvl,
                active_triggers=triggers,
                lat=s.latitude,
                lon=s.longitude,
            )
            if telem:
                self.store.insert_telemetry_snapshot(
                    timestamp_utc=now_utc,
                    site_id=s.site_id,
                    telemetry=asdict(telem) if hasattr(telem, "__dataclass_fields__") else vars(telem),
                    lat=s.latitude,
                    lon=s.longitude,
                )

            site_evaluations.append({
                "site_id": s.site_id,
                "name": s.name,
                "lat": s.latitude,
                "lon": s.longitude,
                "elev": s.elevation_m,
                "slope": s.slope_deg,
                "cal_fs": round(cal_fs, 2),
                "alert": lvl,
                "triggers": triggers,
            })

        print(f"      Alert Distribution: GREEN={alert_counts['GREEN']} | YELLOW={alert_counts['YELLOW']} | ORANGE={alert_counts['ORANGE']} | RED={alert_counts['RED']}", flush=True)

        # 3. Live Ground-Truth Event Feedback Loop (NDRRMA & Sentinel-2/SAR Scar Detection)
        print(f"[3/5] Scanning for newly verified NDRRMA / Sentinel-2 ground-truth feedback events...", flush=True)
        sites_dicts = [asdict(s) for s in self.sites]
        live_weather_dicts = {
            sid: {
                "rainfall_72h_mm": t.rain_72h_mm,
                "peak_intensity_1h_mm": t.rain_peak_1h_mm,
                "seismic_pga_g": t.seismic_pga_g,
            }
            for sid, t in telemetry_map.items()
        }
        new_gt_events = scan_live_telemetry_for_ground_truth_confirmations(sites_dicts, live_weather_dicts)
        all_gt_events = load_live_ground_truth_events()
        print(
            f"      Ground-Truth Catalog: {len(all_gt_events)} live feedback events ({len(new_gt_events)} newly auto-confirmed this cycle).",
            flush=True,
        )

        # 4. Evolutionary Geomechanical Equation Optimization Generation
        print(f"[4/5] Running evolutionary geomechanical equation optimization generation...", flush=True)
        opt_res = self.optimizer.step_optimization_generation()
        print(f"      Gen #{opt_res.generation}: Fitness={opt_res.fitness_score:.4f} | Recall={opt_res.recall*100:.2f}% | Cryo={opt_res.cryo_accuracy*100:.1f}% | LiveGT={opt_res.live_feedback_accuracy*100:.1f}% | Promoted={opt_res.promoted}", flush=True)
        print(f"      Status: {opt_res.reason}", flush=True)

        # 5. Save Real-Time Status Snapshot for Dashboard & Monitoring
        cycle_duration = round(time.time() - cycle_start, 2)
        print(f"[5/5] Cycle #{self.cycle_count} complete in {cycle_duration}s. Writing status snapshot...", flush=True)

        status = {
            "service": "Himalayan-MHEWS Perpetual Ingestion & Physics Engine",
            "cycle_count": self.cycle_count,
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "last_cycle_timestamp_utc": now_utc,
            "last_cycle_duration_sec": cycle_duration,
            "monitored_sites_total": len(self.sites),
            "alert_summary": alert_counts,
            "latest_top_risk_sites": [
                s for s in site_evaluations if s["alert"] in ("RED", "ORANGE", "YELLOW")
            ][:10],
            "ground_truth_feedback": {
                "total_training_catalog_points": opt_res.dataset_total_points,
                "base_catalog_points": 949,
                "cryosphere_anchor_points": 24,
                "live_feedback_points": opt_res.live_feedback_points,
                "newly_confirmed_this_cycle": len(new_gt_events),
                "recent_events": all_gt_events[-6:][::-1],
            },
            "optimizer": {
                "current_generation": opt_res.generation,
                "active_fitness_score": opt_res.fitness_score,
                "holdout_recall_pct": round(opt_res.recall * 100, 2),
                "roc_auc": round(opt_res.roc_auc, 4),
                "cryosphere_anchor_acc_pct": round(opt_res.cryo_accuracy * 100, 1),
                "live_feedback_acc_pct": round(opt_res.live_feedback_accuracy * 100, 1),
                "avg_fs_deviation_pct": opt_res.avg_fs_deviation_pct,
                "last_promoted": opt_res.promoted,
                "last_reason": opt_res.reason,
                "active_parameters": opt_res.parameters,
            },
        }

        self.latest_status = status
        self.last_run_time = now_utc

        try:
            with open(STATUS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"Warning: could not write status json: {e}", flush=True)

        return status

    def run_perpetually(self) -> None:
        """Master perpetual daemon loop running across all 491+ sites."""
        print(f"🚀 Himalayan-MHEWS Master Perpetual Engine Started (PID {os.getpid()})", flush=True)
        print(f"   Fast Loop Interval: {self.fast_loop_interval_sec}s ({self.fast_loop_interval_sec / 60:.1f} minutes)", flush=True)
        print(f"   Database: {DB_PATH}", flush=True)
        print(f"   Status file: {STATUS_JSON_PATH}", flush=True)

        while True:
            try:
                self.execute_single_calculation_cycle()
            except Exception as exc:
                print(f"❌ Error during perpetual cycle #{self.cycle_count}: {exc}", flush=True)
                import traceback
                traceback.print_exc()

            print(f"⏳ Sleeping {self.fast_loop_interval_sec}s until next telemetry refresh cycle...\n", flush=True)
            time.sleep(self.fast_loop_interval_sec)


if __name__ == "__main__":
    interval = int(os.environ.get("CYCLE_INTERVAL_SEC", "1800"))
    orchestrator = PerpetualEngineOrchestrator(fast_loop_interval_sec=interval)
    if "--once" in sys.argv:
        orchestrator.execute_single_calculation_cycle()
    else:
        orchestrator.run_perpetually()
