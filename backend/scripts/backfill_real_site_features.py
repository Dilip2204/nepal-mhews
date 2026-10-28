#!/usr/bin/env python3
"""Backfill REAL, measured site features for every coordinate the models touch.

Why this script exists
----------------------
``backend/app/engine/environmental_features.py`` manufactured "site-specific" soil
texture, NDVI, aspect and road distance from ``hash(f"{lat}_{lon}_{elev}")`` and
documented them as OpenStreetMap / Sentinel-2 / GEDI data. A hash is not an
observation. ``backend/app/ingestion/real_site_features.py`` replaced the generator
with real fetchers, but at ~5 s (SoilGrids) + ~30 s (Overpass) + ~1 s (DEM) per site
it cannot be called 1471 times inline, and it cannot be called from the 30-minute
physics loop at all.

This script does the one-off bulk fetch, politely and resumably, and writes a cache
in exactly the format ``real_site_features.fetch_site_features`` already reads - so
wiring the physics to real data becomes a cache lookup with no network in the loop.

What it covers
--------------
Every coordinate that feeds either the live system or its accuracy measurement:

* the 491 production monitoring sites (``sites_of_interest_catalog.json``)
* the 949 labelled points in ``processed/nepal_training_points.csv``
* the 24 cryosphere anchors in ``processed/himalayan_cryosphere_avalanches_glofs.csv``
* the 7 live ground-truth points in ``processed/live_ground_truth_feedback.csv``

1471 rows collapse to 981 distinct coordinates at 1e-4 deg (~11 m), the precision the
cache is keyed at. The Rule 8.4 gate reads the CSVs, not the site catalog, so covering
only the 491 sites would leave the accuracy impact unmeasurable.

The rules it obeys
------------------
1. **Measured or absent.** Every value carries its source, the exact URL, the raw
   pre-conversion response value, the method and a fetch timestamp. A query that fails
   produces an absence with a reason, never a plausible default, never a value
   interpolated from a neighbouring site.
2. **Absence is distinguishable from zero.** "No mapped road within 2000 m" and "the
   Overpass query failed" are different outcomes and are stored differently.
3. **Polite to free public services.** One request in flight per service, a delay
   between requests, exponential backoff on errors, and a disk cache so a re-run costs
   nothing. SoilGrids and Overpass are donated infrastructure.

Efficiency, measured rather than assumed (2026-10-27, from this host)
---------------------------------------------------------------------
* **Overpass batching** is what makes the road phase tractable. One request can carry
  many ``way(around:R,lat,lon)["highway"];`` clauses, and each returned way is then
  attributed to *every* site in the batch locally. Measured: 6 sites/45.2 s,
  20 sites/35.8 s, 40 sites/31.2 s - the cost is almost entirely per-request queue
  latency, not per-clause work. Batching 25 turns ~8 h of single-site queries into
  ~40 requests. (A bulk bbox query is NOT the alternative: one covering 6 widely
  spread sites returned 843,425 ways in 148 s.)
* **SoilGrids spatial dedupe** on the 250 m native pixel. Implemented, and worth
  little here: the catalog was already deduplicated to 0.001 deg, so sharing within
  125 m saves only 8 of 981 fetches. Reported honestly rather than claimed.
* **DEM aspect** reuses the GLO-30 COG range reads already in the codebase; GDAL keeps
  the tile header cached in-process, so after the first read in a 1 deg tile the rest
  cost ~0.2 s.

Phases write to independent append-only shards, so the three fetch phases can be run
as parallel processes - still one request in flight per service - and merged after.

Usage
-----
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
    export PYTHONPATH=backend
    backend/.venv/bin/python3 backend/scripts/backfill_real_site_features.py --phase all
    # or, in parallel:
    ... --phase soil &   ... --phase road &   ... --phase aspect &   wait
    ... --phase merge
"""

from __future__ import annotations

import os

# Shared 256-core host: cap before numpy/BLAS is imported anywhere (AGENTS.md 8.1).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
# Cheaper /vsicurl COG access for the DEM phase: do not list the bucket prefix.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")

import argparse
import csv
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.ingestion.real_site_features import (  # noqa: E402
    OVERPASS_URL,
    ROAD_CLASS_SEVERITY,
    ROAD_SEARCH_RADIUS_M,
    SOIL_DEPTH,
    SOILGRIDS_URL,
    USER_AGENT,
    Measured,
)

DATA_DIR = BACKEND / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SHARD_DIR = DATA_DIR / "real_site_features_cache"
MERGED_CACHE_PATH = DATA_DIR / "real_site_features.json"
SUMMARY_PATH = DATA_DIR / "real_site_features_summary.json"

CATALOG_PATH = DATA_DIR / "sites_of_interest_catalog.json"
CSV_SOURCES: Sequence[Tuple[str, Path]] = (
    ("train", PROCESSED_DIR / "nepal_training_points.csv"),
    ("cryo", PROCESSED_DIR / "himalayan_cryosphere_avalanches_glofs.csv"),
    ("gt", PROCESSED_DIR / "live_ground_truth_feedback.csv"),
)

# Cache key precision. 1e-4 deg is ~11 m at Nepal's latitude, and is what
# real_site_features.fetch_site_features already keys its cache at - matching it is
# what lets that function serve this backfill with no network.
KEY_DECIMALS = 4

# SoilGrids v2.0 is delivered on a 250 m grid. Two points closer than half a pixel are
# reading the same cell, so one fetch answers both - recorded as a share, with the
# distance, never as an independent measurement.
SOIL_NATIVE_RESOLUTION_M = 250.0
SOIL_SHARE_MAX_M = SOIL_NATIVE_RESOLUTION_M / 2.0

SOIL_DELAY_S = 1.0
SOIL_TIMEOUT_S = 120
SOIL_MAX_ATTEMPTS = 4

OVERPASS_BATCH_SIZE = 25        # measured reliable; see module docstring
OVERPASS_DELAY_S = 5.0
OVERPASS_TIMEOUT_S = 300
OVERPASS_QUERY_TIMEOUT_S = 180
OVERPASS_MAX_ATTEMPTS = 4
OVERPASS_STATUS_URL = "https://overpass-api.de/api/status"

ASPECT_DELAY_S = 0.1

EARTH_R_M = 6371008.8

SOIL_FIELDS = {
    # SoilGrids layer -> (our field, converter from target units to model units)
    "clay": ("clay_fraction", lambda pct: pct / 100.0),
    "sand": ("sand_fraction", lambda pct: pct / 100.0),
    "cfvo": ("coarse_rock_fragment_fraction", lambda pct: pct / 100.0),
    "bdod": ("dry_bulk_density_g_cm3", lambda kg_dm3: kg_dm3),  # kg/dm3 == g/cm3
}

# Physically admissible ranges. A violation means a parsing or unit bug, not an
# interesting site - so it is reported loudly rather than quietly clipped.
PLAUSIBLE_RANGE: Dict[str, Tuple[float, float]] = {
    "clay_fraction": (0.0, 1.0),
    "sand_fraction": (0.0, 1.0),
    "coarse_rock_fragment_fraction": (0.0, 1.0),
    "dry_bulk_density_g_cm3": (0.3, 2.4),
    "road_distance_m": (0.0, float(ROAD_SEARCH_RADIUS_M)),
    "road_class_severity": (0.0, 1.0),
    "aspect_deg": (0.0, 360.0),
}


# ---------------------------------------------------------------------------
# the coordinates to cover
# ---------------------------------------------------------------------------
@dataclass
class Coord:
    key: str
    lat: float
    lon: float
    source_ids: List[str] = field(default_factory=list)


def coord_key(lat: float, lon: float) -> str:
    return f"{lat:.{KEY_DECIMALS}f},{lon:.{KEY_DECIMALS}f}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * math.asin(math.sqrt(min(1.0, a)))


def collect_coordinates() -> List[Coord]:
    """Every distinct coordinate used by the live system or by its accuracy measurement."""
    out: Dict[str, Coord] = {}

    def add(lat: float, lon: float, source_id: str) -> None:
        key = coord_key(lat, lon)
        entry = out.get(key)
        if entry is None:
            out[key] = Coord(key=key, lat=float(lat), lon=float(lon), source_ids=[source_id])
        elif source_id not in entry.source_ids:
            entry.source_ids.append(source_id)

    with open(CATALOG_PATH, "r", encoding="utf-8") as fh:
        for site in json.load(fh):
            add(site["latitude"], site["longitude"], f"site:{site['site_id']}")

    for tag, path in CSV_SOURCES:
        if not path.exists():
            print(f"  ! {path.name} missing - skipped", flush=True)
            continue
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                try:
                    lat, lon = float(row["latitude"]), float(row["longitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                ident = row.get("event_id") or row.get("event_name") or str(i)
                add(lat, lon, f"{tag}:{ident}")

    # Spatially coherent ordering: neighbouring sites land in the same Overpass batch,
    # so the returned road network overlaps and less data crosses the wire.
    return sorted(out.values(), key=lambda c: (round(c.lat, 2), round(c.lon, 2), c.key))


# ---------------------------------------------------------------------------
# resumable append-only shards
# ---------------------------------------------------------------------------
class Shard:
    """One JSON record per line, flushed immediately. Interrupt-safe and resumable."""

    def __init__(self, name: str) -> None:
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        self.path = SHARD_DIR / f"{name}.jsonl"
        self.records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue           # a torn final line from a kill -9
                    if isinstance(rec, dict) and rec.get("key"):
                        self.records[rec["key"]] = rec   # last write wins

    def write(self, record: Dict[str, Any]) -> None:
        self.records[record["key"]] = record
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def done(self, key: str, retry_failed: bool) -> bool:
        rec = self.records.get(key)
        if rec is None:
            return False
        if retry_failed and rec.get("status") == "failed":
            return False
        return True

    def ok_records(self) -> Dict[str, Dict[str, Any]]:
        return {k: r for k, r in self.records.items() if r.get("status") == "ok"}


def measured(value: float, source: str, url: str, method: str, fetched: float,
             raw: Any = None, detail: Optional[str] = None) -> Dict[str, Any]:
    return Measured(value=value, source=source, url=url, method=method,
                    fetched_utc=fetched, raw=raw, detail=detail).to_dict()


# ---------------------------------------------------------------------------
# phase 1 - SoilGrids v2.0
# ---------------------------------------------------------------------------
def soilgrids_url(lat: float, lon: float) -> str:
    import urllib.parse
    query = urllib.parse.urlencode(
        [("lon", f"{lon:.6f}"), ("lat", f"{lat:.6f}"),
         ("property", "clay"), ("property", "sand"),
         ("property", "cfvo"), ("property", "bdod"),
         ("depth", SOIL_DEPTH), ("value", "mean")]
    )
    return f"{SOILGRIDS_URL}?{query}"


def fetch_soil_once(lat: float, lon: float) -> Tuple[Dict[str, Any], str, str]:
    """Measured soil features at one point, as ``(features, note, kind)``.

    ``kind`` is one of:

    ``ok``
        SoilGrids returned at least one value. ``note`` names any property it returned
        as null, so a partial record is never mistaken for a complete one.
    ``absent``
        SoilGrids answered, and answered null everywhere. That is a real statement about
        the world - no soil is mapped in that cell (open water, bare rock, ice, or
        outside the mapped domain) - and it must not be retried forever as if it were an
        outage, nor filled with a plausible number.
    ``failed``
        We could not get an answer. ``note`` carries the reason. Retried on a re-run.

    Unit conversion is read from the response's own ``d_factor`` rather than hardcoded:
    clay 252 g/kg -> 25.2 % -> 0.252, bdod 111 cg/cm3 -> 1.11 g/cm3. Hardcoding those
    factors is exactly the silent error this whole exercise exists to stop.
    """
    url = soilgrids_url(lat, lon)
    delay = 5.0
    for attempt in range(1, SOIL_MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=SOIL_TIMEOUT_S) as response:
                body = response.read()
            break
        except urllib.error.HTTPError as exc:
            reason = f"HTTP {exc.code}"
            wait = delay
            if exc.code == 429:
                try:
                    wait = max(delay, float(exc.headers.get("Retry-After") or delay))
                except (TypeError, ValueError):
                    pass
            if attempt == SOIL_MAX_ATTEMPTS:
                return {}, reason, "failed"
            time.sleep(wait)
            delay *= 3
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if attempt == SOIL_MAX_ATTEMPTS:
                return {}, reason, "failed"
            time.sleep(delay)
            delay *= 3
    else:  # pragma: no cover - loop always breaks or returns
        return {}, "exhausted attempts", "failed"

    try:
        layers = json.loads(body)["properties"]["layers"]
    except (ValueError, KeyError, TypeError) as exc:
        return {}, f"unparseable response: {type(exc).__name__}: {exc}", "failed"

    fetched = time.time()
    features: Dict[str, Any] = {}
    null_layers: List[str] = []
    for layer in layers:
        name = layer.get("name")
        if name not in SOIL_FIELDS:
            continue
        units = layer.get("unit_measure") or {}
        factor = float(units.get("d_factor") or 1.0)
        raw_value = None
        for depth in layer.get("depths") or []:
            if depth.get("label") == SOIL_DEPTH:
                raw_value = (depth.get("values") or {}).get("mean")
                break
        if raw_value is None:
            null_layers.append(SOIL_FIELDS[name][0])
            continue
        field_name, convert = SOIL_FIELDS[name]
        features[field_name] = measured(
            value=round(float(convert(float(raw_value) / factor)), 4),
            source="SoilGrids v2.0 (ISRIC)",
            url=url,
            method=(f"{SOIL_DEPTH} mean, raw {raw_value} {units.get('mapped_units')} "
                    f"/ d_factor {factor:g} -> {units.get('target_units')}"),
            fetched=fetched,
            raw=raw_value,
        )
    if not features:
        return ({}, (f"SoilGrids answered HTTP 200 with mean=null for every property at "
                     f"{SOIL_DEPTH} - no soil is mapped in this cell (open water, bare "
                     f"rock, permanent ice, or outside the mapped domain)"), "absent")
    note = ("" if not null_layers
            else f"SoilGrids returned null for: {', '.join(sorted(null_layers))}")
    return features, note, "ok"


def share_soil(donor: Dict[str, Any], distance_m: float) -> Dict[str, Any]:
    """Re-use a donor point's SoilGrids values, stamped with the sharing distance.

    This is not interpolation and not a default: it is the same 250 m source pixel,
    and every copied value says so in its own provenance.
    """
    shared: Dict[str, Any] = {}
    note = (f"shared SoilGrids value measured at {donor['lat']:.5f},{donor['lon']:.5f}, "
            f"{distance_m:.0f} m away - inside the {SOIL_NATIVE_RESOLUTION_M:.0f} m "
            f"native pixel of SoilGrids v2.0, so the same cell is being read")
    for name, payload in donor["features"].items():
        copy = dict(payload)
        copy["detail"] = note if not copy.get("detail") else f"{copy['detail']}; {note}"
        copy["method"] = f"{copy['method']} [shared, {distance_m:.0f} m]"
        shared[name] = copy
    return shared


def phase_soil(coords: List[Coord], retry_failed: bool, limit: Optional[int]) -> None:
    shard = Shard("soil")
    outstanding = [c for c in coords if not shard.done(c.key, retry_failed)]
    todo = outstanding[:limit] if limit else outstanding
    print(f"[soil] {len(coords)} coordinates, {len(coords) - len(outstanding)} already "
          f"cached, {len(outstanding)} outstanding, {len(todo)} to fetch now", flush=True)

    # Only a point with its own measured values can donate one. An absent cell has
    # nothing to share, and a shared value is never re-shared onward.
    donors = [r for r in shard.ok_records().values()
              if not r.get("shared_from") and r.get("features")]
    started = time.time()
    fetched_n = shared_n = failed_n = absent_n = 0

    for i, coord in enumerate(todo, 1):
        donor_hit = None
        for donor in donors:
            d = haversine_m(coord.lat, coord.lon, donor["lat"], donor["lon"])
            if d <= SOIL_SHARE_MAX_M:
                donor_hit = (donor, d)
                break
        if donor_hit is not None:
            donor, distance = donor_hit
            shard.write({
                "key": coord.key, "lat": coord.lat, "lon": coord.lon,
                "source_ids": coord.source_ids, "status": "ok",
                "features": share_soil(donor, distance),
                "shared_from": donor["key"], "shared_distance_m": round(distance, 1),
            })
            shared_n += 1
            continue

        features, note, kind = fetch_soil_once(coord.lat, coord.lon)
        if kind == "failed":
            shard.write({"key": coord.key, "lat": coord.lat, "lon": coord.lon,
                         "source_ids": coord.source_ids,
                         "status": "failed", "reason": note})
            failed_n += 1
            print(f"[soil] {i}/{len(todo)} {coord.key} FAILED: {note}", flush=True)
        elif kind == "absent":
            shard.write({"key": coord.key, "lat": coord.lat, "lon": coord.lon,
                         "source_ids": coord.source_ids, "status": "ok",
                         "features": {}, "absent_reason": note})
            absent_n += 1
            print(f"[soil] {i}/{len(todo)} {coord.key} ABSENT: {note}", flush=True)
        else:
            record = {"key": coord.key, "lat": coord.lat, "lon": coord.lon,
                      "source_ids": coord.source_ids, "status": "ok",
                      "features": features}
            if note:
                record["absent_reason"] = note
            shard.write(record)
            donors.append(record)
            fetched_n += 1
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - started) / max(1, i)
            left = rate * (len(todo) - i)
            print(f"[soil] {i}/{len(todo)}  fetched={fetched_n} shared={shared_n} "
                  f"absent={absent_n} failed={failed_n}  {rate:.1f} s/site  "
                  f"~{left / 60:.0f} min left", flush=True)
        time.sleep(SOIL_DELAY_S)

    print(f"[soil] done in {(time.time() - started) / 60:.1f} min - {fetched_n} fetched, "
          f"{shared_n} shared, {absent_n} absent (source says no soil here), "
          f"{failed_n} failed", flush=True)


# ---------------------------------------------------------------------------
# phase 2 - OpenStreetMap roads via a batched Overpass query
# ---------------------------------------------------------------------------
def overpass_batch_query(batch: Sequence[Coord], radius_m: int) -> str:
    clauses = "".join(
        f'way(around:{int(radius_m)},{c.lat:.6f},{c.lon:.6f})["highway"];' for c in batch
    )
    return f"[out:json][timeout:{OVERPASS_QUERY_TIMEOUT_S}];({clauses});out tags geom;"


def overpass_post(query: str) -> Tuple[Optional[Dict[str, Any]], str]:
    delay = 30.0
    for attempt in range(1, OVERPASS_MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(
                OVERPASS_URL, data=query.encode("utf-8"),
                headers={"User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=OVERPASS_TIMEOUT_S) as response:
                return json.loads(response.read()), ""
        except urllib.error.HTTPError as exc:
            reason = f"HTTP {exc.code}"
            if exc.code == 429:
                # The dispatcher is out of slots; wait for one rather than retrying into it.
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(OVERPASS_STATUS_URL,
                                               headers={"User-Agent": USER_AGENT}),
                        timeout=30,
                    ) as status:
                        print("[road]   overpass status: "
                              + " | ".join(status.read().decode().splitlines()[:4]),
                              flush=True)
                except Exception:
                    pass
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if attempt == OVERPASS_MAX_ATTEMPTS:
            return None, reason
        print(f"[road]   {reason} - backing off {delay:.0f} s "
              f"(attempt {attempt}/{OVERPASS_MAX_ATTEMPTS})", flush=True)
        time.sleep(delay)
        delay *= 2
    return None, "exhausted attempts"


def _segments_from_ways(elements: Iterable[Dict[str, Any]]):
    """Flatten Overpass ways into (lat, lon) node arrays plus segment index arrays."""
    lats: List[float] = []
    lons: List[float] = []
    seg_a: List[int] = []
    seg_b: List[int] = []
    seg_class: List[str] = []
    for way in elements:
        highway = (way.get("tags") or {}).get("highway")
        geometry = way.get("geometry") or []
        first = len(lats)
        count = 0
        for node in geometry:
            try:
                lats.append(float(node["lat"]))
                lons.append(float(node["lon"]))
                count += 1
            except (KeyError, TypeError, ValueError):
                continue
        for k in range(count - 1):
            seg_a.append(first + k)
            seg_b.append(first + k + 1)
            seg_class.append(str(highway))
        if count == 1:                      # degenerate way: keep it as a point
            seg_a.append(first)
            seg_b.append(first)
            seg_class.append(str(highway))
    return (np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64),
            np.asarray(seg_a, dtype=np.int64), np.asarray(seg_b, dtype=np.int64),
            seg_class)


def nearest_road(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray,
                 seg_a: np.ndarray, seg_b: np.ndarray, seg_class: Sequence[str],
                 radius_m: float) -> Optional[Tuple[float, str, float]]:
    """Distance to the nearest road *segment*, its class, and the nearest-node distance.

    Point-to-segment, not point-to-node: OSM digitises a straight Terai road with nodes
    hundreds of metres apart, so a node-only measure can put a site 200 m from a road it
    is actually 20 m from. The node distance is returned alongside because it is what
    ``real_site_features.fetch_road_proximity`` reports, and the two must stay comparable.
    """
    if lats.size == 0 or seg_a.size == 0:
        return None
    # Local equirectangular projection about the query point: <0.1 % error over 2 km.
    coslat = math.cos(math.radians(lat))
    x = np.radians(lons - lon) * EARTH_R_M * coslat
    y = np.radians(lats - lat) * EARTH_R_M
    # Discard anything far outside the search radius before the segment maths.
    keep = (np.abs(x) <= radius_m * 3.0) & (np.abs(y) <= radius_m * 3.0)
    if not keep.any():
        return None
    near = keep[seg_a] | keep[seg_b]
    if not near.any():
        return None
    idx = np.flatnonzero(near)
    ax, ay = x[seg_a[idx]], y[seg_a[idx]]
    bx, by = x[seg_b[idx]], y[seg_b[idx]]
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(denom > 0.0, -(ax * dx + ay * dy) / np.where(denom > 0.0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    px, py = ax + t * dx, ay + t * dy
    dist = np.hypot(px, py)
    best = int(np.argmin(dist))
    node_dist = float(np.min(np.hypot(x[keep], y[keep])))
    return float(dist[best]), seg_class[int(idx[best])], node_dist


def phase_road(coords: List[Coord], retry_failed: bool, limit: Optional[int],
               batch_size: int, radius_m: int) -> None:
    shard = Shard("road")
    outstanding = [c for c in coords if not shard.done(c.key, retry_failed)]
    todo = outstanding[:limit] if limit else outstanding
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"[road] {len(coords)} coordinates, {len(coords) - len(outstanding)} already "
          f"cached, {len(outstanding)} outstanding, {len(todo)} to fetch now in "
          f"{len(batches)} batched Overpass requests "
          f"(batch={batch_size}, radius={radius_m} m)", flush=True)

    started = time.time()
    measured_n = absent_n = failed_n = 0

    for bi, batch in enumerate(batches, 1):
        t0 = time.time()
        payload, reason = overpass_post(overpass_batch_query(batch, radius_m))
        if payload is None and len(batch) > 1:
            # A whole batch failing is usually the server, not the sites - but split
            # once so one pathological area cannot take 25 coordinates down with it.
            print(f"[road] batch {bi} failed ({reason}); splitting", flush=True)
            half = len(batch) // 2
            for part in (batch[:half], batch[half:]):
                time.sleep(OVERPASS_DELAY_S)
                sub, sub_reason = overpass_post(overpass_batch_query(part, radius_m))
                if sub is None:
                    for coord in part:
                        shard.write({"key": coord.key, "lat": coord.lat, "lon": coord.lon,
                                     "source_ids": coord.source_ids,
                                     "status": "failed", "reason": sub_reason})
                        failed_n += 1
                else:
                    m, a = _attribute_batch(shard, part, sub, radius_m)
                    measured_n += m
                    absent_n += a
            continue
        if payload is None:
            for coord in batch:
                shard.write({"key": coord.key, "lat": coord.lat, "lon": coord.lon,
                             "source_ids": coord.source_ids,
                             "status": "failed", "reason": reason})
                failed_n += 1
            continue

        ways = payload.get("elements") or []
        m, a = _attribute_batch(shard, batch, payload, radius_m)
        measured_n += m
        absent_n += a
        elapsed = time.time() - started
        rate = elapsed / bi
        print(f"[road] batch {bi}/{len(batches)}  {len(batch)} sites  {len(ways)} ways  "
              f"{time.time() - t0:.0f} s  (road={measured_n} none-within-radius={absent_n} "
              f"failed={failed_n})  ~{rate * (len(batches) - bi) / 60:.0f} min left",
              flush=True)
        if bi < len(batches):
            time.sleep(OVERPASS_DELAY_S)

    print(f"[road] done in {(time.time() - started) / 60:.1f} min - {measured_n} with a "
          f"road, {absent_n} with none within {radius_m} m, {failed_n} failed", flush=True)


def _attribute_batch(shard: Shard, batch: Sequence[Coord], payload: Dict[str, Any],
                     radius_m: int) -> Tuple[int, int]:
    """Attribute the ways one request returned back to every site in that request.

    Every site is measured against the *union* of ways, not only those its own clause
    produced: a road within the radius of site B is in the response by construction, so
    the union can only be more complete than a per-site query, never less.
    """
    elements = payload.get("elements") or []
    lats, lons, seg_a, seg_b, seg_class = _segments_from_ways(elements)
    fetched = time.time()
    measured_n = absent_n = 0
    for coord in batch:
        hit = nearest_road(coord.lat, coord.lon, lats, lons, seg_a, seg_b,
                           seg_class, float(radius_m))
        if hit is None or hit[0] > radius_m:
            shard.write({
                "key": coord.key, "lat": coord.lat, "lon": coord.lon,
                "source_ids": coord.source_ids, "status": "ok",
                "road_present": False,
                "features": {
                    "road_distance_m": measured(
                        value=float(radius_m),
                        source="OpenStreetMap (Overpass)", url=OVERPASS_URL,
                        method=(f"batched way(around:{radius_m})[\"highway\"] query; no "
                                f"highway way within {radius_m} m of this point"),
                        fetched=fetched, raw=0,
                        detail="genuine absence of mapped roads, not a failed query",
                    ),
                },
            })
            absent_n += 1
            continue
        distance_m, highway, node_distance_m = hit
        severity = ROAD_CLASS_SEVERITY.get(str(highway), 0.4)
        shard.write({
            "key": coord.key, "lat": coord.lat, "lon": coord.lon,
            "source_ids": coord.source_ids, "status": "ok", "road_present": True,
            "features": {
                "road_distance_m": measured(
                    value=round(distance_m, 1),
                    source="OpenStreetMap (Overpass)", url=OVERPASS_URL,
                    method=(f"minimum point-to-segment distance to any highway way "
                            f"within {radius_m} m, local equirectangular projection"),
                    fetched=fetched, raw=highway,
                    detail=(f"highway={highway}; nearest digitised node "
                            f"{node_distance_m:.1f} m (the node-only measure reported by "
                            f"real_site_features.fetch_road_proximity)"),
                ),
                "road_class_severity": measured(
                    value=severity,
                    source="OpenStreetMap (Overpass) + class ordering", url=OVERPASS_URL,
                    method=("ROAD_CLASS_SEVERITY lookup - a stated modelling judgement "
                            "applied to a MEASURED road class, not an assumed distance"),
                    fetched=fetched, raw=highway,
                ),
            },
        })
        measured_n += 1
    return measured_n, absent_n


# ---------------------------------------------------------------------------
# phase 3 - hillslope aspect from the Copernicus GLO-30 DEM
# ---------------------------------------------------------------------------
def phase_aspect(coords: List[Coord], retry_failed: bool, limit: Optional[int]) -> None:
    from app.ingestion.insar_timeseries import estimate_aspect_deg

    shard = Shard("aspect")
    outstanding = [c for c in coords if not shard.done(c.key, retry_failed)]
    todo = outstanding[:limit] if limit else outstanding
    print(f"[aspect] {len(coords)} coordinates, {len(coords) - len(outstanding)} already "
          f"cached, {len(outstanding)} outstanding, {len(todo)} to read now", flush=True)

    started = time.time()
    ok_n = absent_n = 0
    for i, coord in enumerate(todo, 1):
        try:
            aspect = estimate_aspect_deg(coord.lat, coord.lon)
            # Measured across the 981 coordinates: every absence was the last of these
            # - a zero central gradient on a flat Terai floodplain or a flat glacier /
            # lake surface, where the DEM resolves no downslope direction at all. That
            # is a fact about the terrain, so it is recorded rather than filled in.
            reason = ("Copernicus GLO-30 resolves no downslope direction here: the "
                      "5x5 window's central gradient is zero in both axes (flat "
                      "ground), or the window hit the tile edge or nodata")
        except Exception as exc:
            aspect, reason = None, f"{type(exc).__name__}: {exc}"
        if aspect is None:
            shard.write({"key": coord.key, "lat": coord.lat, "lon": coord.lon,
                         "source_ids": coord.source_ids,
                         "status": "ok", "features": {}, "absent_reason": reason})
            absent_n += 1
        else:
            shard.write({
                "key": coord.key, "lat": coord.lat, "lon": coord.lon,
                "source_ids": coord.source_ids, "status": "ok",
                "features": {
                    "aspect_deg": measured(
                        value=round(float(aspect), 1),
                        source="Copernicus DEM GLO-30",
                        url=("https://copernicus-dem-30m.s3.amazonaws.com/"
                             f"Copernicus_DSM_COG_10_N{math.floor(coord.lat):02d}_00_"
                             f"E{math.floor(coord.lon):03d}_00_DEM/ (COG range read)"),
                        method=("direction of steepest descent of a 5x5 windowed DEM "
                                "gradient, 30 m posting"),
                        fetched=time.time(), raw=round(float(aspect), 6),
                    ),
                },
            })
            ok_n += 1
        if i % 100 == 0 or i == len(todo):
            rate = (time.time() - started) / max(1, i)
            print(f"[aspect] {i}/{len(todo)}  measured={ok_n} absent={absent_n}  "
                  f"{rate:.2f} s/site  ~{rate * (len(todo) - i) / 60:.0f} min left",
                  flush=True)
        time.sleep(ASPECT_DELAY_S)

    print(f"[aspect] done in {(time.time() - started) / 60:.1f} min - "
          f"{ok_n} measured, {absent_n} absent", flush=True)


# ---------------------------------------------------------------------------
# phase 4 - merge and summarise
# ---------------------------------------------------------------------------
def phase_merge(coords: List[Coord], started_utc: float) -> Dict[str, Any]:
    shards = {name: Shard(name) for name in ("soil", "road", "aspect")}
    cache: Dict[str, Any] = {}
    failures: Dict[str, List[Dict[str, str]]] = {name: [] for name in shards}
    absences: Dict[str, List[Dict[str, str]]] = {name: [] for name in shards}
    shared_soil = 0

    for coord in coords:
        features: Dict[str, Any] = {}
        absent: Dict[str, str] = {}
        provenance: Dict[str, Any] = {}
        for name, shard in shards.items():
            rec = shard.records.get(coord.key)
            if rec is None:
                absent[name] = "not fetched"
                continue
            if rec.get("status") != "ok":
                absent[name] = f"fetch failed: {rec.get('reason')}"
                failures[name].append({"key": coord.key,
                                       "source_ids": ",".join(coord.source_ids[:3]),
                                       "reason": str(rec.get("reason"))})
                continue
            features.update(rec.get("features") or {})
            if rec.get("absent_reason"):
                absent[name] = rec["absent_reason"]
                absences[name].append({"key": coord.key,
                                       "reason": str(rec["absent_reason"])})
            if rec.get("shared_from"):
                shared_soil += 1
                provenance["soil_shared_from"] = rec["shared_from"]
                provenance["soil_shared_distance_m"] = rec["shared_distance_m"]
            if name == "road" and rec.get("road_present") is False:
                provenance["road_absent_within_radius_m"] = ROAD_SEARCH_RADIUS_M

        # Prefer the production site id when this coordinate is one of the 491; a
        # labelled CSV row keeps its prefixed id so its origin stays visible.
        site_label = next((sid.split(":", 1)[1] for sid in coord.source_ids
                           if sid.startswith("site:")),
                          coord.source_ids[0] if coord.source_ids else coord.key)
        entry: Dict[str, Any] = {
            "site_id": site_label,
            "latitude": coord.lat,
            "longitude": coord.lon,
            "source_ids": coord.source_ids,
            "features": features,
        }
        if absent:
            entry["absent"] = absent
        entry.update(provenance)
        cache[coord.key] = entry

    # --- distributions, for sanity-checking against physical plausibility ----
    values: Dict[str, List[float]] = {}
    for entry in cache.values():
        for name, payload in entry["features"].items():
            values.setdefault(name, []).append(float(payload["value"]))

    distribution: Dict[str, Any] = {}
    implausible: List[Dict[str, Any]] = []
    for name, vals in sorted(values.items()):
        vals_sorted = sorted(vals)
        quant = (lambda p: vals_sorted[min(len(vals_sorted) - 1,
                                           int(p * (len(vals_sorted) - 1)))])
        distribution[name] = {
            "count": len(vals),
            "min": round(vals_sorted[0], 4),
            "p05": round(quant(0.05), 4),
            "median": round(statistics.median(vals_sorted), 4),
            "p95": round(quant(0.95), 4),
            "max": round(vals_sorted[-1], 4),
            "mean": round(statistics.fmean(vals_sorted), 4),
        }
        lo, hi = PLAUSIBLE_RANGE.get(name, (-math.inf, math.inf))
        bad = [v for v in vals if not (lo <= v <= hi)]
        if bad:
            implausible.append({"property": name, "range": [lo, hi],
                                "violations": len(bad), "examples": bad[:5]})

    summary = {
        "generated_utc": time.time(),
        "elapsed_s_this_invocation": round(time.time() - started_utc, 1),
        "coordinates_total": len(coords),
        "cache_path": str(MERGED_CACHE_PATH),
        "cache_key_format": f"'{{lat:.{KEY_DECIMALS}f}},{{lon:.{KEY_DECIMALS}f}}'",
        "coverage": {
            name: {
                "count": distribution.get(name, {}).get("count", 0),
                "pct_of_coordinates": round(
                    100.0 * distribution.get(name, {}).get("count", 0) / max(1, len(coords)), 1),
            }
            for name in ("clay_fraction", "sand_fraction",
                         "coarse_rock_fragment_fraction", "dry_bulk_density_g_cm3",
                         "road_distance_m", "road_class_severity", "aspect_deg")
        },
        "soil_values_shared_within_native_pixel": shared_soil,
        "road_none_within_search_radius": sum(
            1 for e in cache.values() if e.get("road_absent_within_radius_m")),
        "failures": {name: rows for name, rows in failures.items() if rows},
        "failure_counts": {name: len(rows) for name, rows in failures.items()},
        "absences": {name: rows[:50] for name, rows in absences.items() if rows},
        "absence_counts": {name: len(rows) for name, rows in absences.items()},
        "distribution": distribution,
        "implausible_values": implausible,
    }

    # No metadata key goes in the cache itself: it is keyed by coordinate and nothing
    # else, so a consumer can iterate it without special-casing. The description lives
    # in the summary file beside it.
    summary["cache_format"] = (
        "written by backend/scripts/backfill_real_site_features.py. Keyed by "
        f"'{{lat:.{KEY_DECIMALS}f}},{{lon:.{KEY_DECIMALS}f}}' and by nothing else. Each "
        "entry: site_id, latitude, longitude, source_ids, features (a map of "
        "real_site_features.Measured payloads - value, source, url, method, fetched_utc, "
        "raw, detail) and, where something could not be measured, absent (a map of "
        "phase -> reason). Readable as-is by real_site_features.fetch_site_features(). "
        "Every value is a fetched observation or it is absent - never a default."
    )

    MERGED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MERGED_CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
    tmp.replace(MERGED_CACHE_PATH)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("REAL SITE FEATURE BACKFILL - SUMMARY")
    print("=" * 78)
    print(f"coordinates covered : {summary['coordinates_total']}")
    print(f"cache               : {summary['cache_path']}")
    print(f"keyed by            : {summary['cache_key_format']}")
    print("\ncoverage (property: measured / total)")
    for name, cov in summary["coverage"].items():
        print(f"  {name:34s} {cov['count']:5d}  ({cov['pct_of_coordinates']:.1f}%)")
    print("\ndistribution (sanity check against physical plausibility)")
    print(f"  {'property':34s} {'min':>9s} {'median':>9s} {'max':>9s}")
    for name, dist in summary["distribution"].items():
        print(f"  {name:34s} {dist['min']:9.3f} {dist['median']:9.3f} {dist['max']:9.3f}")
    if summary["implausible_values"]:
        print("\n  !! VALUES OUTSIDE PHYSICAL RANGE - likely a parsing bug:")
        for bad in summary["implausible_values"]:
            print(f"     {bad['property']}: {bad['violations']} outside {bad['range']} "
                  f"e.g. {bad['examples']}")
    else:
        print("\n  all measured values inside their physically admissible ranges")
    print(f"\nsoil values shared inside the 250 m native pixel: "
          f"{summary['soil_values_shared_within_native_pixel']}")
    print(f"points with no mapped road within {ROAD_SEARCH_RADIUS_M} m: "
          f"{summary['road_none_within_search_radius']}")
    print(f"failures  : {summary['failure_counts']}")
    print(f"absences  : {summary['absence_counts']}")
    print(f"elapsed   : {summary['elapsed_s_this_invocation'] / 60:.1f} min "
          f"(this invocation)")
    print("=" * 78, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", default="all",
                        choices=["all", "soil", "road", "aspect", "merge", "plan"])
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N outstanding coordinates")
    parser.add_argument("--batch-size", type=int, default=OVERPASS_BATCH_SIZE)
    parser.add_argument("--radius-m", type=int, default=ROAD_SEARCH_RADIUS_M)
    parser.add_argument("--skip-failed", action="store_true",
                        help="do not retry coordinates that failed on an earlier run")
    args = parser.parse_args(argv)

    started = time.time()
    coords = collect_coordinates()
    print(f"{len(coords)} distinct coordinates at {10 ** -KEY_DECIMALS:g} deg "
          f"from the 491-site catalog + "
          f"{', '.join(p.name for _, p in CSV_SOURCES)}", flush=True)
    retry_failed = not args.skip_failed

    if args.phase == "plan":
        for shard_name in ("soil", "road", "aspect"):
            shard = Shard(shard_name)
            todo = [c for c in coords if not shard.done(c.key, retry_failed)]
            print(f"  {shard_name:7s} cached={len(coords) - len(todo):4d} "
                  f"outstanding={len(todo):4d}")
        return 0

    if args.phase in ("all", "soil"):
        phase_soil(coords, retry_failed, args.limit)
    if args.phase in ("all", "road"):
        phase_road(coords, retry_failed, args.limit, args.batch_size, args.radius_m)
    if args.phase in ("all", "aspect"):
        phase_aspect(coords, retry_failed, args.limit)
    if args.phase in ("all", "merge"):
        print_summary(phase_merge(coords, started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
