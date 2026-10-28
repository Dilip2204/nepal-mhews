#!/usr/bin/env python3
"""Does replacing fabricated physics inputs with real measurements help or hurt?

Why this script exists
----------------------
The Rule 8.4 gate in ``app/ml/continuous_equation_optimizer.py`` **cannot answer that
question**. It evaluates its own closed-form FS (``compute_candidate_fs``) over CSV
columns (sti / ommi / gii / s_perp / tcwi / fault_dmg) and never imports
``environmental_features.py`` or ``feature_synthesizer.py``. Changing the site inputs is
completely invisible to it: run the gate before and after the swap and it returns the
same numbers. Its metrics are still reported below, clearly fenced off, so the two are
never confused.

This harness instead drives the **live orchestrator path** -
``synthesize_physics_variables`` -> ``compute_infinite_slope_fs`` - over every labelled
ground-truth point, under two input regimes:

  A  fabricated  - what ``environmental_features.py`` feeds production today
  B  real        - measurements from the ``real_site_features.py`` cache

and scores both against the labels, with particular attention to the one direction that
kills people: a known-landslide point moving toward GREEN.

Four rules this harness holds itself to
---------------------------------------
1. **No silent mixing.** A point whose real cache entry is missing even one field is
   *excluded* from the A-vs-B comparison and counted in a separate bucket. A comparison
   that quietly substitutes frozen defaults for missing measurements measures nothing.
2. **No invented values.** Nothing here interpolates between sites or picks a plausible
   default. Where regime B has no measurement, that is reported as a fact about the
   cache, not patched over.
3. **The weather is real too.** The dynamic half of the inputs is read from the
   orchestrator's own recorded telemetry (``perpetual_telemetry.db``, opened read-only),
   per site, from the most recent completed cycle - not from an invented uniform
   scenario. It is byte-identical between regimes A and B, so every FS difference
   reported is attributable to the static soil/road/aspect inputs. Points with no
   recorded telemetry are scored separately and labelled.
4. **Regime B is not fully real, and says so.** ``real_site_features.py`` has no source
   for ``sentinel2_ndvi``, ``gedi_canopy_height_m`` or ``diurnal_lst_range_c``. Those
   three keep the frozen nationwide dataclass defaults in regime B, and the report names
   them every run. Regime B measures the soil/road/aspect swap only.

A FIDELITY CHECK section replays the 491 catalog sites through this harness and compares
the result to the live orchestrator's own last-cycle alert summary in
``perpetual_status.json``. If those two disagree, nothing else in the report is evidence
about production, so it is printed first.

Read-only: nothing under ``backend/app`` is touched, the telemetry DB is opened with
``mode=ro``, no file is written unless ``--json`` is passed, and no network call is made.

Usage
-----
    cd /data-mqs/users/djena/avalanche_detector/nepal-mhews
    backend/.venv/bin/python3 backend/scripts/evaluate_real_feature_impact.py

Determinism: the script re-executes itself with ``PYTHONHASHSEED=0`` and the four thread
caps, because regime A's generator derives its "site-specific" values from
``hash(f"{lat}_{lon}_{elev}")``, which Python randomises per process. (Catalog sites read
their fabricated values from the production cache instead, so those are exact.)
"""

from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------------------
# Deterministic, thread-capped re-exec. Must happen before numpy/pandas are imported.
# --------------------------------------------------------------------------------------
_REQUIRED_ENV = {
    "PYTHONHASHSEED": "0",      # regime A's generator uses hash(); Python randomises it
    "OMP_NUM_THREADS": "4",     # shared 256-core server
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}
if os.environ.get("_MHEWS_EVAL_REEXEC") != "1" and any(
    os.environ.get(k) != v for k, v in _REQUIRED_ENV.items()
):
    os.environ.update(_REQUIRED_ENV)
    os.environ["_MHEWS_EVAL_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd

from app.engine.environmental_features import (
    ENRICHED_CATALOG_PATH,
    compute_site_environmental_features,
)
from app.engine.sites_catalog import determine_basin, load_sites_of_interest
from app.ingestion.real_site_features import CACHE_PATH as REAL_CACHE_PATH
from app.physics.feature_synthesizer import (
    RawEarthObservationInputs,
    synthesize_physics_variables,
)
from app.physics.infinite_slope import (
    compute_infinite_slope_fs,
    compute_viscoplastic_creep_velocity,
)

DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TELEMETRY_DB_PATH = DATA_DIR / "perpetual_telemetry.db"
STATUS_JSON_PATH = DATA_DIR / "perpetual_status.json"

# Fields regime B must supply from measurement for a point to be comparable. Every one of
# them has a real fetcher in real_site_features.py.
REQUIRED_REAL_FIELDS = (
    "clay_fraction",
    "sand_fraction",
    "coarse_rock_fragment_fraction",
    "dry_bulk_density_g_cm3",
    "road_distance_m",
    "aspect_deg",
)

# Fields with NO real source anywhere in the codebase. In regime B these fall back to the
# frozen RawEarthObservationInputs dataclass defaults - a declared nationwide constant,
# which is honest, unlike a hash dressed up as Sentinel-2. Reported on every run.
UNMEASURABLE_FIELDS = {
    "sentinel2_ndvi": 0.32,
    "gedi_canopy_height_m": 6.5,
    "diurnal_lst_range_c": 16.0,
}

# The RawEarthObservationInputs dataclass defaults: one frozen set of numbers for the
# whole of Nepal. This is the `if env else` fallback branch in perpetual_orchestrator.py,
# and - see the VERSION SKEW check - it is what the currently running process is actually
# feeding the physics, because that process was started before environmental_features.py
# existed. So this, not regime A, is the live status quo.
FROZEN_DEFAULT_STATIC = {
    "clay_fraction": 0.16,
    "sand_fraction": 0.48,
    "coarse_rock_fragment_fraction": 0.35,
    "dry_bulk_density_g_cm3": 1.55,
    "sentinel2_ndvi": 0.32,
    "gedi_canopy_height_m": 6.5,
    "aspect_deg": 195.0,
    "diurnal_lst_range_c": 16.0,
    "distance_to_road_m": 1200.0,
    "road_cut_penalty": 0.0,
}

# Modules whose contents decide the FS a running orchestrator produces.
PHYSICS_SOURCE_FILES = (
    BACKEND_DIR / "app" / "physics" / "feature_synthesizer.py",
    BACKEND_DIR / "app" / "physics" / "infinite_slope.py",
    BACKEND_DIR / "app" / "engine" / "perpetual_orchestrator.py",
    BACKEND_DIR / "app" / "engine" / "environmental_features.py",
)

SEVERITY = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
ALERT_LEVELS = ("GREEN", "YELLOW", "ORANGE", "RED")


# ======================================================================================
# the dynamic (weather / seismic) half of the inputs
# ======================================================================================
class Met:
    """One point's dynamic inputs. Identical between regimes A and B by construction."""

    __slots__ = ("rain_72h_mm", "pga_g", "pdd_days", "wind_surcharge_kpa",
                 "cloudburst", "cyclonic", "source")

    def __init__(self, rain_72h_mm: float, pga_g: float, pdd_days: float,
                 wind_surcharge_kpa: float, cloudburst: bool, cyclonic: bool, source: str):
        self.rain_72h_mm = float(rain_72h_mm)
        self.pga_g = float(pga_g)
        self.pdd_days = float(pdd_days)
        self.wind_surcharge_kpa = float(wind_surcharge_kpa)
        self.cloudburst = bool(cloudburst)
        self.cyclonic = bool(cyclonic)
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


class MetProvider:
    """Per-site dynamic inputs from the orchestrator's own recorded telemetry.

    The DB is opened read-only (``mode=ro``): the live orchestrator writes to it every
    30 minutes and this harness must never contend with it.

    Points with no recorded telemetry - the labelled training points that are not catalog
    sites - get the declared fallback below, which is the orchestrator's own
    no-telemetry branch (rain 4.0 mm, PGA 0.04 g, PDD 0, wind 0). Those points are
    counted and scored separately; the fallback is a stated scenario, not a measurement,
    and is never mixed into a telemetry-backed number without saying so.
    """

    FALLBACK = dict(rain_72h_mm=4.0, pga_g=0.04, pdd_days=0.0, wind_surcharge_kpa=0.0)

    def __init__(self, db_path: Path, site_id_by_coord3: Dict[Tuple[float, float], str],
                 use_telemetry: bool = True, override: Optional[Dict[str, float]] = None):
        self.db_path = db_path
        self.site_id_by_coord3 = site_id_by_coord3
        self.override = override
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.cycle_timestamp: Optional[float] = None
        self.available = False
        if override is not None or not use_telemetry or not db_path.exists():
            return
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            newest = con.execute("SELECT MAX(timestamp_utc) FROM telemetry_snapshots").fetchone()[0]
            if newest is None:
                con.close()
                return
            self.cycle_timestamp = float(newest)
            for row in con.execute(
                "SELECT site_id, rain_72h_mm, seismic_pga_g, pdd_days_7d, wind_surcharge_kpa, "
                "cloudburst_flag, cyclonic_depression_flag FROM telemetry_snapshots "
                "WHERE timestamp_utc = ?", (newest,)
            ):
                self.rows[row["site_id"]] = dict(row)
            con.close()
            self.available = bool(self.rows)
        except sqlite3.Error as exc:
            print(f"  [warn] telemetry DB unreadable ({exc}) - falling back to the "
                  f"declared scenario for every point", flush=True)

    def describe(self) -> str:
        if self.override is not None:
            o = self.override
            return (f"uniform declared scenario: rain_72h={o['rain_72h_mm']:g} mm, "
                    f"PGA={o['pga_g']:g} g, PDD_7d={o['pdd_days']:g} C-days")
        if not self.available:
            return ("telemetry unavailable - orchestrator no-telemetry fallback at every "
                    "point (rain 4.0 mm, PGA 0.04 g, PDD 0)")
        when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(self.cycle_timestamp or 0))
        return (f"recorded live telemetry, cycle of {when} "
                f"({len(self.rows)} sites in perpetual_telemetry.db, read-only)")

    def _fallback(self) -> Met:
        src = self.FALLBACK if self.override is None else self.override
        return Met(src["rain_72h_mm"], src["pga_g"], src["pdd_days"],
                   src.get("wind_surcharge_kpa", 0.0), False, False,
                   "declared_scenario" if self.override is not None else "orchestrator_fallback")

    def for_site_id(self, site_id: Optional[str]) -> Met:
        if self.override is not None or site_id is None:
            return self._fallback()
        row = self.rows.get(site_id)
        if row is None:
            return self._fallback()
        return Met(
            rain_72h_mm=row["rain_72h_mm"] or 0.0,
            pga_g=row["seismic_pga_g"] if row["seismic_pga_g"] is not None else 0.04,
            pdd_days=row["pdd_days_7d"] or 0.0,
            wind_surcharge_kpa=row["wind_surcharge_kpa"] or 0.0,
            cloudburst=bool(row["cloudburst_flag"]),
            cyclonic=bool(row["cyclonic_depression_flag"]),
            source="telemetry",
        )

    def for_point(self, point: Dict[str, Any], site_id: Optional[str]) -> Met:
        if site_id is None:
            site_id = self.site_id_by_coord3.get(
                (round(point["latitude"], 3), round(point["longitude"], 3))
            )
        return self.for_site_id(site_id)


# ======================================================================================
# labelled ground truth
# ======================================================================================
def load_labelled_points() -> List[Dict[str, Any]]:
    """The three labelled catalogs the Rule 8.4 gate uses: 949 + 24 + 7 points."""
    points: List[Dict[str, Any]] = []
    sources = [
        ("training", "nepal_training_points.csv"),
        ("cryosphere", "himalayan_cryosphere_avalanches_glofs.csv"),
        ("live_gt", "live_ground_truth_feedback.csv"),
    ]
    for dataset, filename in sources:
        path = PROCESSED_DIR / filename
        if not path.exists():
            print(f"  [warn] {filename} absent - that dataset contributes 0 points", flush=True)
            continue
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            points.append({
                "dataset": dataset,
                "row_index": int(idx),
                "point_id": str(row.get("event_id") or row.get("event_name") or f"{dataset}_{idx}"),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "elevation_m": float(row["elevation_m"]),
                "slope_deg": float(row["slope_deg"]),
                "profile_curvature": float(row["profile_curvature"]),
                "local_relief_m": float(row.get("local_relief_m", 120.0)),
                "fault_distance_km": float(row.get("fault_distance_km", 4.0)),
                "hazard_type": str(row.get("hazard_type", "rainfall_landslide")),
                "label": int(row["label"]),
            })
    return points


# ======================================================================================
# regime A - the fabricated generator, exactly as production reads it
# ======================================================================================
class FabricatedProvider:
    """Regime A inputs.

    Two tiers, counted separately so the report never implies more fidelity than it has:

      * ``cache``      - the point is one of the 491 catalog sites, so the values are read
                         verbatim from ``sites_of_interest_enriched.json``: literally the
                         bytes the running orchestrator feeds the physics right now.
      * ``recomputed`` - the point is not in the catalog, so the generator is called live.
                         Its output is ``hash()``-derived and therefore only reproducible
                         because this process pins PYTHONHASHSEED=0.
    """

    def __init__(self) -> None:
        self.by_coord: Dict[Tuple[float, float], Dict[str, Any]] = {}
        self.site_id_by_coord3: Dict[Tuple[float, float], str] = {}
        self.enriched_loaded = False
        if ENRICHED_CATALOG_PATH.exists():
            try:
                with open(ENRICHED_CATALOG_PATH, "r", encoding="utf-8") as fh:
                    enriched = json.load(fh)
                for site_id, feats in enriched.items():
                    key = (round(float(feats["latitude"]), 3), round(float(feats["longitude"]), 3))
                    self.by_coord[key] = dict(feats, site_id=site_id)
                    self.site_id_by_coord3.setdefault(key, site_id)
                self.enriched_loaded = True
            except (OSError, ValueError, KeyError) as exc:
                print(f"  [warn] could not read {ENRICHED_CATALOG_PATH.name}: {exc}", flush=True)

        self.hazard_by_coord: Dict[Tuple[float, float], str] = {}
        try:
            for site in load_sites_of_interest():
                key = (round(site.latitude, 3), round(site.longitude, 3))
                self.hazard_by_coord[key] = site.hazard_type
                self.site_id_by_coord3.setdefault(key, site.site_id)
        except Exception as exc:  # catalog build is optional for this harness
            print(f"  [warn] site catalog unavailable ({type(exc).__name__}: {exc})", flush=True)

    @staticmethod
    def _key(point: Dict[str, Any]) -> Tuple[float, float]:
        # 0.001 deg (~110 m) is the same dedup radius the catalog itself was built with,
        # so a hit here is site identity, not a nearest-neighbour guess.
        return (round(point["latitude"], 3), round(point["longitude"], 3))

    def get(self, point: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Optional[str]]:
        """Return (static input dict, provenance tier, matched site_id)."""
        feats = self.by_coord.get(self._key(point))
        if feats is not None:
            return (
                {
                    "clay_fraction": float(feats["clay_fraction"]),
                    "sand_fraction": float(feats["sand_fraction"]),
                    "coarse_rock_fragment_fraction": float(feats["coarse_rock_fragment_fraction"]),
                    "dry_bulk_density_g_cm3": float(feats["dry_bulk_density_g_cm3"]),
                    "sentinel2_ndvi": float(feats["sentinel2_ndvi"]),
                    "gedi_canopy_height_m": float(feats["gedi_canopy_height_m"]),
                    "aspect_deg": float(feats["aspect_deg"]),
                    "diurnal_lst_range_c": float(feats["diurnal_lst_range_c"]),
                    "distance_to_road_m": float(feats["distance_to_road_m"]),
                    "road_cut_penalty": float(feats["road_cut_penalty"]),
                },
                "cache",
                feats.get("site_id"),
            )

        ef = compute_site_environmental_features(
            site_id=point["point_id"],
            latitude=point["latitude"],
            longitude=point["longitude"],
            elevation_m=point["elevation_m"],
            slope_deg=point["slope_deg"],
            profile_curvature=point["profile_curvature"],
            local_relief_m=point["local_relief_m"],
            basin_id=determine_basin(point["longitude"], point["latitude"]),
        )
        return (
            {
                "clay_fraction": ef.clay_fraction,
                "sand_fraction": ef.sand_fraction,
                "coarse_rock_fragment_fraction": ef.coarse_rock_fragment_fraction,
                "dry_bulk_density_g_cm3": ef.dry_bulk_density_g_cm3,
                "sentinel2_ndvi": ef.sentinel2_ndvi,
                "gedi_canopy_height_m": ef.gedi_canopy_height_m,
                "aspect_deg": ef.aspect_deg,
                "diurnal_lst_range_c": ef.diurnal_lst_range_c,
                "distance_to_road_m": ef.distance_to_road_m,
                "road_cut_penalty": ef.road_cut_penalty,
            },
            "recomputed",
            None,
        )

    def hazard_type(self, point: Dict[str, Any]) -> str:
        return self.hazard_by_coord.get(
            self._key(point), point.get("hazard_type", "rainfall_landslide")
        )


# ======================================================================================
# regime B - real measurements, or nothing
# ======================================================================================
def road_cut_penalty_from_measured_distance(distance_m: float, slope_deg: float) -> float:
    """R_cut from a MEASURED road distance, using the model's own published formula.

    The formula (AGENTS.md / HANDOFF.md Milestone 14) is a stated modelling judgement:
        R_cut = clip(exp(-d/65) * sin(beta), 0, 0.95), active within 150 m on slopes >= 20 deg.
    The judgement is the formula; the DISTANCE it is applied to is measured. Nothing is
    invented here.

    The caller has to supply this because ``feature_synthesizer.py`` cannot: its own
    fallback branch for exactly this case calls ``math.exp`` while the module never
    imports ``math``, so it raises NameError. See the bug note at the end of the report.
    """
    if distance_m <= 150.0 and slope_deg >= 20.0:
        return float(np.clip(
            math.exp(-distance_m / 65.0) * math.sin(math.radians(slope_deg)), 0.0, 0.95))
    return 0.0


class RealProvider:
    """Regime B inputs, read from the ``real_site_features.py`` cache. Never fetches."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.populated = False
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.by_coord3: Dict[Tuple[float, float], str] = {}

        if not cache_path.exists():
            return
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"  [warn] real-features cache unreadable: {exc}", flush=True)
            return
        if not isinstance(data, dict) or not data:
            return
        self.entries = data
        self.populated = True
        for key, entry in data.items():
            try:
                lat = float(entry["latitude"])
                lon = float(entry["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            self.by_coord3.setdefault((round(lat, 3), round(lon, 3)), key)

    def lookup(self, point: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """Return (features dict of Measured payloads, match mode) or (None, 'absent')."""
        exact = f"{point['latitude']:.4f},{point['longitude']:.4f}"
        if exact in self.entries:
            return (self.entries[exact].get("features") or {}), "exact_key"
        key = self.by_coord3.get((round(point["latitude"], 3), round(point["longitude"], 3)))
        if key is not None:
            # Same 0.001 deg identity rule the catalog was deduplicated with: this is the
            # same site, not a neighbouring one borrowed to fill a gap.
            return (self.entries[key].get("features") or {}), "catalog_identity"
        return None, "absent"

    def get(self, point: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str, List[str]]:
        """Return (static input dict, status, missing field names)."""
        if not self.populated:
            return None, "cache_not_populated", list(REQUIRED_REAL_FIELDS)
        features, mode = self.lookup(point)
        if features is None:
            return None, "no_cache_entry", list(REQUIRED_REAL_FIELDS)

        if not features:
            # The backfill created the row but has not fetched it yet. Distinct from a
            # fetch that ran and came back short, which is a real measurement failure.
            return None, "entry_present_nothing_fetched", list(REQUIRED_REAL_FIELDS)

        missing = [f for f in REQUIRED_REAL_FIELDS
                   if not isinstance(features.get(f), dict) or features[f].get("value") is None]
        if missing:
            return None, f"incomplete:{mode}", missing

        distance_m = float(features["road_distance_m"]["value"])
        inputs = {
            "clay_fraction": float(features["clay_fraction"]["value"]),
            "sand_fraction": float(features["sand_fraction"]["value"]),
            "coarse_rock_fragment_fraction": float(features["coarse_rock_fragment_fraction"]["value"]),
            "dry_bulk_density_g_cm3": float(features["dry_bulk_density_g_cm3"]["value"]),
            "aspect_deg": float(features["aspect_deg"]["value"]),
            "distance_to_road_m": distance_m,
            "road_cut_penalty": road_cut_penalty_from_measured_distance(
                distance_m, point["slope_deg"]),
        }
        # No real source exists for these three. Frozen nationwide default, counted and
        # named in the report rather than quietly filled.
        inputs.update(UNMEASURABLE_FIELDS)
        return inputs, mode, []


# ======================================================================================
# the live physics path
# ======================================================================================
def evaluate_point(point: Dict[str, Any], static: Dict[str, Any], hazard_type: str,
                   met: Met) -> Dict[str, Any]:
    """Reproduce perpetual_orchestrator.py's per-site physics exactly, offline."""
    slope = point["slope_deg"]
    rain = met.rain_72h_mm

    # Orchestrator's pre-convergence creep proxy -> InSAR LOS velocity input.
    est_fs = max(0.5, 1.25 - (rain / 250.0) - (0.3 if slope > 38.0 else 0.0))
    creep_proxy = compute_viscoplastic_creep_velocity(
        slope_deg=slope, dynamic_fs=est_fs, elevation_m=point["elevation_m"],
        hazard_type=hazard_type, rain_72h_mm=rain,
    )

    raw = RawEarthObservationInputs(
        elevation_m=point["elevation_m"],
        slope_deg=slope,
        aspect_deg=static["aspect_deg"],
        profile_curvature=point["profile_curvature"],
        plan_curvature=point["profile_curvature"] * 0.5,
        upslope_contributing_area_m2=4500.0,
        sentinel2_ndvi=static["sentinel2_ndvi"],
        gedi_canopy_height_m=static["gedi_canopy_height_m"],
        clay_fraction=static["clay_fraction"],
        sand_fraction=static["sand_fraction"],
        coarse_rock_fragment_fraction=static["coarse_rock_fragment_fraction"],
        dry_bulk_density_g_cm3=static["dry_bulk_density_g_cm3"],
        distance_to_fault_km=point["fault_distance_km"],
        diurnal_lst_range_c=static["diurnal_lst_range_c"],
        imerg_rain_72h_mm=rain,
        antecedent_rain_30d_api_mm=max(12.0, rain * 1.5),
        pdd_c_days_7d=met.pdd_days,
        bedrock_pga_g=met.pga_g,
        insar_los_velocity_mm_month=creep_proxy,
        snow_water_equivalent_mm=max(20.0, met.wind_surcharge_kpa * 100.0),
        distance_to_road_m=static["distance_to_road_m"],
        road_cut_penalty=static["road_cut_penalty"],
    )
    syn = synthesize_physics_variables(raw)

    fs = float(compute_infinite_slope_fs(
        c_prime=syn["c_prime_kpa"],
        gamma=syn["gamma_kn_m3"],
        z=syn["z_m"],
        z_w=syn["z_w_m"],
        beta_deg=syn.get("effective_slope_deg", slope),
        phi_prime_deg=syn["phi_prime_deg"],
        seismic_kh=syn["seismic_kh"],
        surface_surcharge_q_kpa=syn["surface_surcharge_q_kpa"] + met.wind_surcharge_kpa,
        matric_suction_kpa=syn["matric_suction_kpa"],
    ))

    converged_creep = compute_viscoplastic_creep_velocity(
        slope_deg=slope, dynamic_fs=fs, elevation_m=point["elevation_m"],
        hazard_type=hazard_type, rain_72h_mm=rain,
    )

    # Orchestrator trigger set, verbatim. Offline there is no measured-InSAR record, so
    # the creep trigger takes its MODELLED_CREEP_ACCELERATION name, as it does in
    # production for every site but glc_105.
    triggers: List[str] = []
    if rain >= 120.0 or met.cloudburst:
        triggers.append("CLOUD_BURST_SATURATION")
    if met.cyclonic:
        triggers.append("CYCLONIC_BAROMETRIC_DEPRESSION")
    if met.pga_g >= 0.12:
        triggers.append("SEISMIC_SHAKEMAP_PGA")
    if point["elevation_m"] >= 4500.0 and met.pdd_days >= 8.0:
        triggers.append("CRITICAL_THERMAL_ICE_MELT")
    if met.wind_surcharge_kpa >= 0.40:
        triggers.append("WIND_BLOWN_CORNICE_SURCHARGE")
    if converged_creep >= 10.0 and (rain >= 80.0 or met.pga_g >= 0.08 or met.pdd_days >= 6.0):
        triggers.append("MODELLED_CREEP_ACCELERATION")

    # Cascade Alert Matrix (Rule 6), verbatim.
    if fs < 1.0 and len(triggers) >= 1:
        level = "RED"
    elif fs < 1.05 or len(triggers) >= 2:
        level = "ORANGE"
    elif fs < 1.25 or len(triggers) == 1:
        level = "YELLOW"
    else:
        level = "GREEN"

    return {
        "fs": fs,
        "alert": level,
        "triggers": triggers,
        "met_source": met.source,
        "c_prime_kpa": syn["c_prime_kpa"],
        "phi_prime_deg": syn["phi_prime_deg"],
        "z_m": syn["z_m"],
        "saturation_ratio": syn["saturation_ratio"],
        "porosity": syn["porosity"],
        "road_cut_penalty": syn["road_cut_penalty"],
        "failure_prob_pct": float(100.0 / (1.0 + math.exp(6.8 * (fs - 1.0)))),
    }


# ======================================================================================
# fidelity check - does this harness actually reproduce production?
# ======================================================================================
def _orchestrator_start_utc() -> Optional[Tuple[int, float]]:
    """(pid, start time) of the running perpetual orchestrator, from /proc. Read-only."""
    try:
        clk = os.sysconf("SC_CLK_TCK")
        btime = None
        with open("/proc/stat", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("btime "):
                    btime = float(line.split()[1])
                    break
        if btime is None:
            return None
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as fh:
                    cmdline = fh.read().decode("utf-8", "replace")
                if "perpetual_orchestrator.py" not in cmdline or "python" not in cmdline:
                    continue
                with open(f"/proc/{entry}/stat", "r", encoding="utf-8") as fh:
                    stat_line = fh.read()
                fields = stat_line[stat_line.rfind(")") + 2:].split()
                return int(entry), btime + float(fields[19]) / clk
            except (OSError, ValueError, IndexError):
                continue
    except (OSError, ValueError):
        pass
    return None


def fidelity_check(fabricated: FabricatedProvider, mets: MetProvider) -> Dict[str, Any]:
    """Is the live process even running the code this harness evaluates?

    Two parts, in order of importance:

    1. **Version skew.** The orchestrator loads its Python once, at process start. If any
       physics module has been edited since, the running process is still executing the
       old code and its recorded output describes a version that no longer exists on
       disk. Then neither ``perpetual_status.json`` nor ``physics_calculations`` can
       validate this harness, and - more to the point - the pending edits ship silently
       at the next restart, unmeasured.
    2. **Per-site replay.** The 491 catalog sites are replayed through this harness with
       the same telemetry cycle and compared site-by-site against the FS values the
       orchestrator actually recorded. Only meaningful when there is no version skew.
    """
    out: Dict[str, Any] = {"available": False}

    newest_src = max((p.stat().st_mtime, p.name) for p in PHYSICS_SOURCE_FILES if p.exists())
    proc = _orchestrator_start_utc()
    out["newest_physics_source_mtime"] = newest_src[0]
    out["newest_physics_source_name"] = newest_src[1]
    if proc is not None:
        out["orchestrator_pid"], out["orchestrator_start_utc"] = proc
        out["version_skew"] = bool(proc[1] < newest_src[0])
        out["stale_modules"] = sorted(
            p.name for p in PHYSICS_SOURCE_FILES if p.exists() and p.stat().st_mtime > proc[1]
        )
    else:
        out["orchestrator_pid"] = None
        out["version_skew"] = None
        out["stale_modules"] = []

    try:
        with open(STATUS_JSON_PATH, "r", encoding="utf-8") as fh:
            status = json.load(fh)
        out["production_summary"] = {k: int(v) for k, v in (status.get("alert_summary") or {}).items()}
        out["production_cycle_utc"] = status.get("last_cycle_timestamp_utc")
    except (OSError, ValueError, TypeError) as exc:
        out["production_summary"] = {}
        out["status_error"] = str(exc)

    try:
        sites = load_sites_of_interest()
    except Exception as exc:
        out["error"] = f"catalog unavailable: {exc}"
        return out

    recorded: Dict[str, Dict[str, Any]] = {}
    try:
        con = sqlite3.connect(f"file:{mets.db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        newest = con.execute("SELECT MAX(timestamp_utc) FROM physics_calculations").fetchone()[0]
        if newest is not None:
            for row in con.execute(
                "SELECT site_id, dynamic_fs, alert_level FROM physics_calculations "
                "WHERE timestamp_utc = ?", (newest,)
            ):
                recorded[row["site_id"]] = dict(row)
            out["recorded_cycle_utc"] = float(newest)
        con.close()
    except sqlite3.Error as exc:
        out["recorded_error"] = str(exc)

    replay = {lvl: 0 for lvl in ALERT_LEVELS}
    frozen_replay = {lvl: 0 for lvl in ALERT_LEVELS}
    alert_agree = frozen_alert_agree = compared = 0
    fs_abs_err: List[float] = []
    for site in sites:
        point = {
            "latitude": site.latitude, "longitude": site.longitude,
            "elevation_m": site.elevation_m, "slope_deg": site.slope_deg,
            "profile_curvature": site.profile_curvature, "local_relief_m": site.local_relief_m,
            "fault_distance_km": site.fault_distance_km, "point_id": site.site_id,
            "hazard_type": site.hazard_type,
        }
        static, _, _ = fabricated.get(point)
        met = mets.for_site_id(site.site_id)
        res_a = evaluate_point(point, static, site.hazard_type, met)
        res_f = evaluate_point(point, FROZEN_DEFAULT_STATIC, site.hazard_type, met)
        replay[res_a["alert"]] += 1
        frozen_replay[res_f["alert"]] += 1
        rec = recorded.get(site.site_id)
        if rec is not None:
            compared += 1
            alert_agree += int(res_a["alert"] == rec["alert_level"])
            frozen_alert_agree += int(res_f["alert"] == rec["alert_level"])
            fs_abs_err.append(abs(res_a["fs"] - float(rec["dynamic_fs"])))

    out.update({
        "available": True,
        "n_sites": len(sites),
        "replayed_regime_A": replay,
        "replayed_regime_frozen": frozen_replay,
        "n_recorded_compared": compared,
        "alert_agreement_A": (alert_agree / compared) if compared else None,
        "alert_agreement_frozen": (frozen_alert_agree / compared) if compared else None,
        "median_abs_fs_error_A": float(np.median(fs_abs_err)) if fs_abs_err else None,
        "telemetry_cycle_utc": mets.cycle_timestamp,
    })
    return out


# ======================================================================================
# scoring
# ======================================================================================
def score(records: List[Dict[str, Any]], regime: str) -> Dict[str, Any]:
    """Operational scoring: a warning is issued iff the alert level is not GREEN."""
    if not records:
        return {"n": 0}
    labels = np.array([r["label"] for r in records])
    fs = np.array([r[regime]["fs"] for r in records])
    alerts = [r[regime]["alert"] for r in records]
    warned = np.array([a != "GREEN" for a in alerts])

    tp = int(np.sum((labels == 1) & warned))
    fn = int(np.sum((labels == 1) & ~warned))
    fp = int(np.sum((labels == 0) & warned))
    tn = int(np.sum((labels == 0) & ~warned))
    n_pos = tp + fn

    out = {
        "n": len(records),
        "n_positive": n_pos,
        "n_negative": fp + tn,
        "recall_at_landslide_points": (tp / n_pos) if n_pos else None,
        "accuracy": (tp + tn) / len(records),
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "mean_fs": float(np.mean(fs)),
        "median_fs": float(np.median(fs)),
        "fs_below_1_count": int(np.sum(fs < 1.0)),
        "alert_distribution": {lvl: alerts.count(lvl) for lvl in ALERT_LEVELS},
    }
    pos_mask = labels == 1
    if n_pos:
        out["mean_fs_at_landslide_points"] = float(np.mean(fs[pos_mask]))
        out["alert_distribution_at_landslide_points"] = {
            lvl: int(np.sum(pos_mask & np.array([a == lvl for a in alerts])))
            for lvl in ALERT_LEVELS
        }
    by_ds: Dict[str, Dict[str, Any]] = {}
    for ds in ("training", "cryosphere", "live_gt"):
        sub = [r for r in records if r["dataset"] == ds]
        if not sub:
            continue
        sl = np.array([r["label"] for r in sub])
        sw = np.array([r[regime]["alert"] != "GREEN" for r in sub])
        s_pos = int(np.sum(sl == 1))
        by_ds[ds] = {
            "n": len(sub),
            "accuracy": float(np.mean((sl == 1) == sw)),
            "recall": (float(np.sum((sl == 1) & sw) / s_pos) if s_pos else None),
        }
    out["by_dataset"] = by_ds
    out["met_sources"] = {
        src: sum(1 for r in records if r[regime]["met_source"] == src)
        for src in sorted({r[regime]["met_source"] for r in records})
    }
    return out


def _telemetry_coverage_by_label(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Real-telemetry coverage split by label.

    If one class is mostly scored under recorded weather and the other under the flat
    fallback, overall accuracy compares two different scenarios and is not a like-for-like
    number. Recall, computed within the positive class alone, is unaffected.
    """
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    pos_t = sum(1 for r in pos if r["met"]["source"] == "telemetry")
    neg_t = sum(1 for r in neg if r["met"]["source"] == "telemetry")
    pos_rate = (pos_t / len(pos)) if pos else 0.0
    neg_rate = (neg_t / len(neg)) if neg else 0.0
    return {
        "positive_total": len(pos), "positive_with_telemetry": pos_t,
        "negative_total": len(neg), "negative_with_telemetry": neg_t,
        "skewed": bool(pos and neg and abs(pos_rate - neg_rate) > 0.2),
    }


def compare(records: List[Dict[str, Any]], from_regime: str = "A",
            to_regime: str = "B") -> Dict[str, Any]:
    """Regime-to-regime movement, with the dangerous direction isolated."""
    records = [r for r in records if r.get(from_regime) and r.get(to_regime)]
    if not records:
        return {"n": 0}
    fs_a = np.array([r[from_regime]["fs"] for r in records])
    fs_b = np.array([r[to_regime]["fs"] for r in records])
    rel = (fs_b - fs_a) / np.maximum(fs_a, 1e-9) * 100.0

    toward_green_pos: List[Dict[str, Any]] = []
    softened_pos = 0
    hardened_pos = 0
    for r in records:
        if r["label"] != 1:
            continue
        a, b = r[from_regime]["alert"], r[to_regime]["alert"]
        if SEVERITY[b] < SEVERITY[a]:
            softened_pos += 1
        elif SEVERITY[b] > SEVERITY[a]:
            hardened_pos += 1
        if a != "GREEN" and b == "GREEN":
            toward_green_pos.append({
                "point_id": r["point_id"], "dataset": r["dataset"],
                "lat": r["latitude"], "lon": r["longitude"],
                "fs_a": round(r[from_regime]["fs"], 4), "fs_b": round(r[to_regime]["fs"], 4),
                "alert_a": a, "alert_b": b,
            })

    transitions: Dict[str, int] = {}
    for r in records:
        key = f"{r[from_regime]['alert']}->{r[to_regime]['alert']}"
        transitions[key] = transitions.get(key, 0) + 1

    n_pos = sum(1 for r in records if r["label"] == 1)
    return {
        "n": len(records),
        "from_regime": from_regime,
        "to_regime": to_regime,
        "mean_abs_fs_deviation_pct": float(np.mean(np.abs(rel))),
        "mean_signed_fs_deviation_pct": float(np.mean(rel)),
        "median_signed_fs_deviation_pct": float(np.median(rel)),
        "max_abs_fs_deviation_pct": float(np.max(np.abs(rel))),
        "alert_level_changed": int(sum(
            1 for r in records if r[from_regime]["alert"] != r[to_regime]["alert"]
        )),
        "transitions": dict(sorted(transitions.items(), key=lambda kv: -kv[1])),
        "landslide_points": n_pos,
        "landslide_points_softened": softened_pos,
        "landslide_points_hardened": hardened_pos,
        "landslide_points_newly_GREEN": len(toward_green_pos),
        "landslide_points_newly_GREEN_detail": toward_green_pos[:40],
        "landslide_points_fs_increased": int(sum(
            1 for r in records
            if r["label"] == 1 and r[to_regime]["fs"] > r[from_regime]["fs"]
        )),
    }


# ======================================================================================
# Rule 8.4 - reported, and fenced off
# ======================================================================================
def rule_84_metrics() -> Dict[str, Any]:
    try:
        from app.ml.continuous_equation_optimizer import ContinuousEquationOptimizer
        opt = ContinuousEquationOptimizer()
        acc, rec, auc, f1, cryo, fb, dev = opt.evaluate_parameter_set(opt.active_params)
        return {
            "available": True,
            "n_points": int(len(opt._dataset)),
            "accuracy": acc, "recall": rec, "roc_auc": auc, "f1": f1,
            "cryo_accuracy": cryo, "live_gt_accuracy": fb, "avg_fs_deviation_pct": dev,
            "gate_pass": bool(rec >= 0.9136 and cryo >= 0.999 and dev <= 5.0),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


# ======================================================================================
# report
# ======================================================================================
def pct(x: Optional[float]) -> str:
    return "   n/a" if x is None else f"{x * 100:6.2f}%"


def print_report(result: Dict[str, Any]) -> None:
    w = 86
    rule = "=" * w
    thin = "-" * w
    print()
    print(rule)
    print("  REAL vs FABRICATED PHYSICS INPUTS - live orchestrator path")
    print(rule)
    print(f"  labelled points loaded : {result['n_points_total']}  "
          f"(training {result['dataset_counts'].get('training', 0)}, "
          f"cryosphere {result['dataset_counts'].get('cryosphere', 0)}, "
          f"live-GT {result['dataset_counts'].get('live_gt', 0)})")
    print(f"  dynamic inputs (A==B)  : {result['met_description']}")
    print(f"  code path              : synthesize_physics_variables -> compute_infinite_slope_fs")
    print(f"  warning definition     : alert level != GREEN")
    print(f"  real-features cache    : {result['real_cache_path']}")
    print(f"                           " + (
        f"POPULATED - {result['real_cache_entries']} entries"
        if result["real_cache_populated"] else "NOT POPULATED"))

    # -- fidelity ------------------------------------------------------------------
    fc = result["fidelity_check"]
    print()
    print(thin)
    print("  VERSION SKEW - is the live process even running the code evaluated below?")
    print(thin)
    if fc.get("orchestrator_pid") is None:
        print("  No running perpetual_orchestrator.py found in /proc; skew unknown.")
    else:
        started = time.strftime("%Y-%m-%d %H:%M UTC",
                                time.gmtime(fc["orchestrator_start_utc"]))
        edited = time.strftime("%Y-%m-%d %H:%M UTC",
                               time.gmtime(fc["newest_physics_source_mtime"]))
        print(f"  orchestrator pid {fc['orchestrator_pid']} started      {started}")
        print(f"  newest physics source edited          {edited}  "
              f"({fc['newest_physics_source_name']})")
        if fc.get("version_skew"):
            print("  *** SKEW: the running process loaded its Python BEFORE these edits and is")
            print("      still executing the old code. Its recorded output describes a version")
            print("      that no longer exists on disk, so it cannot validate this harness - and")
            print("      the pending edits ship unmeasured at the next restart. Stale here:")
            for name in fc.get("stale_modules", []):
                print(f"        {name}")
        else:
            print("  No skew: the running process loaded the current source.")

    if fc.get("available"):
        print()
        print("  Replay of the 491 catalog sites on the SAME telemetry cycle:")
        ra, rf = fc["replayed_regime_A"], fc["replayed_regime_frozen"]
        pr = fc.get("production_summary") or {}
        print("      " + "  ".join(f"{l} {ra[l]}" for l in ALERT_LEVELS)
              + "   <- regime A (fabricated), current source")
        print("      " + "  ".join(f"{l} {rf[l]}" for l in ALERT_LEVELS)
              + "   <- frozen defaults, current source")
        print("      " + "  ".join(f"{l} {pr.get(l, 0)}" for l in ALERT_LEVELS)
              + "   <- perpetual_status.json (whatever the live process is running)")
        if fc.get("n_recorded_compared"):
            print(f"  per-site agreement with recorded physics_calculations "
                  f"({fc['n_recorded_compared']} sites):")
            print(f"      regime A {pct(fc['alert_agreement_A'])} of alert levels   "
                  f"frozen defaults {pct(fc['alert_agreement_frozen'])}")
            print(f"      regime A median |dFS| vs recorded: {fc['median_abs_fs_error_A']:.4f}")
            if fc.get("version_skew"):
                print("      (both disagree because of the skew above, not because the harness")
                print("       diverges from the code it is running.)")

    # -- buckets -------------------------------------------------------------------
    print()
    print(thin)
    print("  COVERAGE - which regime could be evaluated where")
    print(thin)
    b = result["buckets"]
    print(f"  A evaluated            : {b['a_total']}   "
          f"(fabricated values from production cache {b['a_from_cache']}, "
          f"regenerated {b['a_recomputed']})")
    print(f"  with real telemetry    : {b['met_from_telemetry']}   "
          f"(declared fallback at {b['met_from_fallback']})")
    cov = b.get("met_telemetry_by_label") or {}
    if cov:
        print(f"      of labelled landslides : {cov.get('positive_with_telemetry', 0)}"
              f"/{cov.get('positive_total', 0)}")
        print(f"      of labelled stable     : {cov.get('negative_with_telemetry', 0)}"
              f"/{cov.get('negative_total', 0)}")
        if cov.get("skewed"):
            print("      *** telemetry coverage is uneven between the two labels, so the two")
            print("          classes are being scored under different weather. Recall (one")
            print("          class) is unaffected; OVERALL ACCURACY mixes them - read the")
            print("          'REAL telemetry only' panel for the like-for-like number. ***")
    print(f"  A and B comparable     : {b['comparable']}")
    print(f"  B unavailable          : {b['b_unavailable']}")
    for reason, count in sorted(b["b_unavailable_reasons"].items(), key=lambda kv: -kv[1]):
        print(f"      {reason:<34} {count}")
    if b["b_missing_field_counts"]:
        print("  missing measurements by field (cache entry exists but has a gap):")
        for field, count in sorted(b["b_missing_field_counts"].items(), key=lambda kv: -kv[1]):
            print(f"      {field:<34} {count}")
    if b["comparable"] == 0:
        print("  -> no A-vs-B comparison is possible yet. Regime A below is the baseline.")

    # -- regimes -------------------------------------------------------------------
    panels = (
        ("F_all", "REGIME F (frozen nationwide defaults) - the live status quo"),
        ("A_all", "REGIME A (fabricated) - all evaluable points"),
        ("A_telemetry", "REGIME A (fabricated) - points with REAL telemetry only"),
        ("A_comparable", "REGIME A (fabricated) - comparable subset"),
        ("B_comparable", "REGIME B (real soil/road/aspect) - comparable subset"),
    )
    for key, title in panels:
        s = result["scores"].get(key)
        if not s or not s.get("n"):
            continue
        print()
        print(thin)
        print(f"  {title}")
        print(thin)
        print(f"  points {s['n']}   positives {s['n_positive']}   negatives {s['n_negative']}")
        print(f"  recall at landslide points : {pct(s['recall_at_landslide_points'])}   "
              f"(TP {s['tp']}, FN {s['fn']})")
        print(f"  overall accuracy           : {pct(s['accuracy'])}   "
              f"(FP {s['fp']}, TN {s['tn']})")
        print(f"  precision                  : {pct(s['precision'])}")
        print(f"  mean FS {s['mean_fs']:.4f}    median FS {s['median_fs']:.4f}    "
              f"FS<1.0 at {s['fs_below_1_count']} points")
        dist = s["alert_distribution"]
        print("  alert distribution         : " + "  ".join(
            f"{lvl} {dist[lvl]}" for lvl in ALERT_LEVELS))
        if "alert_distribution_at_landslide_points" in s:
            pdl = s["alert_distribution_at_landslide_points"]
            print("    at landslide points only : " + "  ".join(
                f"{lvl} {pdl[lvl]}" for lvl in ALERT_LEVELS))
        for ds, m in s.get("by_dataset", {}).items():
            print(f"    {ds:<12} n={m['n']:<5} acc {pct(m['accuracy'])}  recall {pct(m['recall'])}")

    # -- comparisons ---------------------------------------------------------------
    titles = {
        "comparison_F_to_A": ("F -> A MOVEMENT  (already pending: ships at the next restart)",
                              "positive = the fabricated generator makes slopes look SAFER"),
        "comparison": ("A -> B MOVEMENT  (the part that decides whether this change ships)",
                       "positive = real inputs make slopes look SAFER"),
    }
    for key, (title, sign_note) in titles.items():
        cmp_ = result.get(key) or {}
        if not cmp_.get("n"):
            continue
        print()
        print(thin)
        print(f"  {title}")
        print(thin)
        print(f"  mean |FS deviation|      : {cmp_['mean_abs_fs_deviation_pct']:.2f}%")
        print(f"  mean signed FS deviation : {cmp_['mean_signed_fs_deviation_pct']:+.2f}%  "
              f"({sign_note})")
        print(f"  max |FS deviation|       : {cmp_['max_abs_fs_deviation_pct']:.2f}%")
        print(f"  alert level changed at   : {cmp_['alert_level_changed']} / {cmp_['n']} points")
        print()
        print("  *** KNOWN-LANDSLIDE POINTS MOVING TOWARD GREEN ***")
        print(f"      landslide points compared        : {cmp_['landslide_points']}")
        print(f"      downgraded in severity           : {cmp_['landslide_points_softened']}")
        print(f"      NEWLY GREEN (new false negative) : {cmp_['landslide_points_newly_GREEN']}")
        print(f"      upgraded in severity             : {cmp_['landslide_points_hardened']}")
        print(f"      FS increased (toward stable)     : {cmp_['landslide_points_fs_increased']}")
        for d in cmp_["landslide_points_newly_GREEN_detail"][:15]:
            print(f"        {d['point_id'][:38]:<38} {d['alert_a']}->{d['alert_b']}  "
                  f"FS {d['fs_a']:.3f} -> {d['fs_b']:.3f}")
        if cmp_["transitions"]:
            print("  alert transitions (A->B), most common first:")
            for k, v in list(cmp_["transitions"].items())[:10]:
                print(f"      {k:<22} {v}")

    # -- unmeasurable --------------------------------------------------------------
    print()
    print(thin)
    print("  WHAT REGIME B STILL CANNOT MEASURE")
    print(thin)
    print("  real_site_features.py has no source for these, so regime B holds them at the")
    print("  frozen nationwide dataclass default - identical at every site, same as today:")
    for field, value in UNMEASURABLE_FIELDS.items():
        print(f"      {field:<28} = {value}")
    print("  PHYSICS_SENSITIVITY_FINDINGS.md measured diurnal_lst_range_c and sentinel2_ndvi")
    print("  as the two LARGEST single-input FS effects at glc_105. Regime B therefore")
    print("  measures the soil / road / aspect swap only - not 'real inputs' in general.")

    # -- rule 8.4 ------------------------------------------------------------------
    print()
    print(thin)
    print("  RULE 8.4 GATE METRICS - *** DOES NOT EXERCISE THE CODE PATH ABOVE ***")
    print(thin)
    r84 = result["rule_84"]
    if not r84.get("available"):
        print(f"  unavailable: {r84.get('error')}")
    else:
        print(f"  continuous_equation_optimizer.compute_candidate_fs over {r84['n_points']} CSV rows.")
        print("  It derives sti/ommi/gii/s_perp/tcwi/fault_dmg from lat/lon/elev/slope/curvature")
        print("  and NEVER imports environmental_features.py or feature_synthesizer.py, so these")
        print("  numbers are IDENTICAL before and after the real-input swap. They are reported")
        print("  here only so nobody mistakes them for evidence about it.")
        print(f"    recall {pct(r84['recall'])} (gate >= 91.36%)   "
              f"cryo {pct(r84['cryo_accuracy'])} (gate 100%)")
        print(f"    live-GT {pct(r84['live_gt_accuracy'])}   accuracy {pct(r84['accuracy'])}   "
              f"ROC-AUC {r84['roc_auc']:.4f}")
        print(f"    FS deviation {r84['avg_fs_deviation_pct']:.2f}% (gate <= 5.0%)   "
              f"gate: {'PASS' if r84['gate_pass'] else 'FAIL'}")

    # -- bug note ------------------------------------------------------------------
    print()
    print(thin)
    print("  BLOCKING BUG FOUND WHILE BUILDING THIS (not fixed here - read-only harness)")
    print(thin)
    print("  backend/app/physics/feature_synthesizer.py uses math.exp() and math.radians()")
    print("  in its road-cut fallback branch, but the module never imports math. The branch")
    print("  fires when road_cut_penalty <= 0 AND distance_to_road_m <= 150 AND slope >= 20,")
    print("  and raises NameError: name 'math' is not defined.")
    print("  Masked today only because environmental_features.py always supplies a non-zero")
    print("  penalty under exactly those conditions. Feed real OSM distances WITHOUT a")
    print("  precomputed penalty and the orchestrator crashes at every site within 150 m of")
    print("  a road on a slope >= 20 deg. This harness supplies the penalty itself to work")
    print("  around it; production would not.")
    print(rule)
    print()


# ======================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure whether real site inputs improve or damage prediction, "
                    "through the live orchestrator physics path.")
    ap.add_argument("--met", choices=("telemetry", "scenario"), default="telemetry",
                    help="dynamic inputs: recorded per-site telemetry from the last "
                         "orchestrator cycle (default), or a uniform declared scenario")
    ap.add_argument("--rain-mm", type=float, default=4.0,
                    help="--met scenario only: 72h rainfall at every point (default 4.0, "
                         "the orchestrator's own no-telemetry fallback)")
    ap.add_argument("--pga-g", type=float, default=0.04,
                    help="--met scenario only: bedrock PGA in g (default 0.04)")
    ap.add_argument("--pdd-days", type=float, default=0.0,
                    help="--met scenario only: 7-day positive degree days (default 0.0)")
    ap.add_argument("--db", type=Path, default=TELEMETRY_DB_PATH,
                    help=f"telemetry DB, opened read-only (default {TELEMETRY_DB_PATH})")
    ap.add_argument("--cache", type=Path, default=REAL_CACHE_PATH,
                    help=f"real-features cache (default {REAL_CACHE_PATH})")
    ap.add_argument("--json", type=Path, default=None, help="also write the full result as JSON")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N points (debug)")
    args = ap.parse_args()

    print("Loading labelled ground truth...", flush=True)
    points = load_labelled_points()
    if args.limit:
        points = points[: args.limit]
    if not points:
        print("No labelled points found - nothing to evaluate.", file=sys.stderr)
        return 2

    fabricated = FabricatedProvider()
    override = None
    if args.met == "scenario":
        override = {"rain_72h_mm": args.rain_mm, "pga_g": args.pga_g,
                    "pdd_days": args.pdd_days, "wind_surcharge_kpa": 0.0}
    mets = MetProvider(args.db, fabricated.site_id_by_coord3,
                       use_telemetry=(args.met == "telemetry"), override=override)
    real = RealProvider(args.cache)

    records: List[Dict[str, Any]] = []
    a_from_cache = a_recomputed = 0
    met_telemetry = met_fallback = 0
    b_reasons: Dict[str, int] = {}
    b_missing_fields: Dict[str, int] = {}

    for point in points:
        static_a, tier, site_id = fabricated.get(point)
        if tier == "cache":
            a_from_cache += 1
        else:
            a_recomputed += 1
        hazard = fabricated.hazard_type(point)
        met = mets.for_point(point, site_id)
        if met.source == "telemetry":
            met_telemetry += 1
        else:
            met_fallback += 1

        record = dict(point)
        record["matched_site_id"] = site_id
        record["A_provenance"] = tier
        record["met"] = met.to_dict()
        record["F"] = evaluate_point(point, FROZEN_DEFAULT_STATIC, hazard, met)
        record["A"] = evaluate_point(point, static_a, hazard, met)

        static_b, status, missing = real.get(point)
        if static_b is None:
            record["B"] = None
            record["B_status"] = status
            b_reasons[status] = b_reasons.get(status, 0) + 1
            if status.startswith("incomplete"):
                for f in missing:
                    b_missing_fields[f] = b_missing_fields.get(f, 0) + 1
        else:
            record["B"] = evaluate_point(point, static_b, hazard, met)
            record["B_status"] = status
        records.append(record)

    comparable = [r for r in records if r["B"] is not None]
    telemetry_backed = [r for r in records if r["A"]["met_source"] == "telemetry"]

    dataset_counts: Dict[str, int] = {}
    for r in records:
        dataset_counts[r["dataset"]] = dataset_counts.get(r["dataset"], 0) + 1

    result = {
        "generated_utc": time.time(),
        "n_points_total": len(records),
        "dataset_counts": dataset_counts,
        "met_mode": args.met,
        "met_description": mets.describe(),
        "telemetry_cycle_utc": mets.cycle_timestamp,
        "real_cache_path": str(args.cache),
        "real_cache_populated": real.populated,
        "real_cache_entries": len(real.entries),
        "enriched_catalog_loaded": fabricated.enriched_loaded,
        "fidelity_check": fidelity_check(fabricated, mets),
        "buckets": {
            "a_total": len(records),
            "a_from_cache": a_from_cache,
            "a_recomputed": a_recomputed,
            "met_from_telemetry": met_telemetry,
            "met_from_fallback": met_fallback,
            "met_telemetry_by_label": _telemetry_coverage_by_label(records),
            "comparable": len(comparable),
            "b_unavailable": len(records) - len(comparable),
            "b_unavailable_reasons": b_reasons,
            "b_missing_field_counts": b_missing_fields,
        },
        "scores": {
            "F_all": score(records, "F"),
            "A_all": score(records, "A"),
            "A_telemetry": score(telemetry_backed, "A") if telemetry_backed else {"n": 0},
            "A_comparable": score(comparable, "A") if comparable else {"n": 0},
            "B_comparable": score(comparable, "B") if comparable else {"n": 0},
        },
        "comparison_F_to_A": compare(records, "F", "A"),
        "comparison": compare(comparable, "A", "B") if comparable else {"n": 0},
        "unmeasurable_in_regime_b": UNMEASURABLE_FIELDS,
        "rule_84": rule_84_metrics(),
    }

    print_report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, default=str)
        print(f"Full result written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
