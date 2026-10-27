#!/usr/bin/env python3
"""Backfill REAL distance-to-nearest-road for every coordinate, from a bulk OSM extract.

Why not the Overpass API
------------------------
Overpass answers a per-site query in roughly 30 s, so 981 coordinates is about eight
hours of hammering a free public service - and a backfill attempt using it wrote 3
coordinates in 25 minutes before stalling. Geofabrik publishes the whole of Nepal as a
415 MB `.osm.pbf`, which downloads in under a minute and then answers every coordinate
locally from a spatial index, with no rate limit and no load on anyone else's server.

Why this matters physically
---------------------------
Road construction cutting into a hillslope is among the best-documented landslide
drivers in Nepal: the cut removes toe support and oversteepens the slope above it. The
factor-of-safety model has an `R_cut` term for exactly this, but it was being fed
`18.0 + (hash_seed % 280)` - a hash of the coordinates dressed as OpenStreetMap data.
This replaces that with the real distance.

What is measured vs judged
--------------------------
The DISTANCE and the road CLASS are measured. The severity weighting applied to each
class (a national highway implies a bigger excavation than a footpath) is a stated
modelling judgement, recorded as such in the provenance rather than hidden in a formula.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "real_site_features.json"

# Ordered by how much excavation the class implies. A stated judgement, applied to a
# MEASURED class - not an assumed distance.
ROAD_CLASS_SEVERITY = {
    "motorway": 1.0, "motorway_link": 1.0, "trunk": 1.0, "trunk_link": 1.0,
    "primary": 0.9, "primary_link": 0.9, "secondary": 0.75, "secondary_link": 0.75,
    "tertiary": 0.6, "tertiary_link": 0.6, "unclassified": 0.45, "residential": 0.45,
    "living_street": 0.4, "service": 0.3, "track": 0.25,
    "path": 0.1, "footway": 0.1, "cycleway": 0.1, "bridleway": 0.1, "steps": 0.05,
}
# Beyond this a cut has no mechanical influence on the slope; recorded as a real
# "no road near here" rather than as a failure.
SEARCH_RADIUS_M = 2000.0


def load_coordinates():
    coords = {}
    try:
        from app.engine.sites_catalog import load_sites_of_interest

        for s in load_sites_of_interest():
            coords[(round(s.latitude, 4), round(s.longitude, 4))] = s.site_id
    except Exception as exc:
        print(f"[warn] site catalog unavailable: {exc}", flush=True)
    for name in ("nepal_training_points.csv", "himalayan_cryosphere_avalanches_glofs.csv",
                 "live_ground_truth_feedback.csv"):
        path = DATA_DIR / "processed" / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    key = (round(float(row["latitude"]), 4), round(float(row["longitude"]), 4))
                except (KeyError, TypeError, ValueError):
                    continue
                coords.setdefault(key, f"gt:{name[:12]}")
    return sorted(coords.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbf", required=True, help="path to nepal-latest.osm.pbf")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import pyogrio
    from shapely import STRtree
    from shapely.geometry import Point

    coords = load_coordinates()
    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.load(open(CACHE_PATH, encoding="utf-8"))
        except ValueError:
            cache = {}

    outstanding = []
    for (lat, lon), site_id in coords:
        key = f"{lat:.4f},{lon:.4f}"
        have = ((cache.get(key) or {}).get("features") or {})
        if not args.refetch and "road_distance_m" in have:
            continue
        outstanding.append((key, lat, lon, site_id))
    print(f"coordinates: {len(coords)}, outstanding: {len(outstanding)}", flush=True)
    if not outstanding:
        return 0

    print("reading highway lines from the extract...", flush=True)
    t0 = time.time()
    gdf = pyogrio.read_dataframe(
        args.pbf, layer="lines", columns=["highway"],
        where="highway IS NOT NULL",
    )
    gdf = gdf[gdf["highway"].notna()]
    print(f"  {len(gdf)} highway ways in {time.time()-t0:.0f}s", flush=True)

    geoms = list(gdf.geometry.values)
    classes = list(gdf["highway"].values)
    tree = STRtree(geoms)
    print(f"  spatial index built over {len(geoms)} geometries", flush=True)

    fetched = time.time()
    found = none_near = 0
    t0 = time.time()
    for i, (key, lat, lon, site_id) in enumerate(outstanding, 1):
        # Degrees of search radius at this latitude, used only to bound the candidate set.
        dlat = SEARCH_RADIUS_M / 111320.0
        dlon = SEARCH_RADIUS_M / (111320.0 * max(0.15, math.cos(math.radians(lat))))
        pt = Point(lon, lat)
        idx = tree.query(pt.buffer(max(dlat, dlon)))
        best_m, best_cls = None, None
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
        for j in idx:
            geom = geoms[int(j)]
            # Nearest point on the way, then convert the degree offset to metres with a
            # local equirectangular scaling - accurate well inside 2 km.
            try:
                p2 = geom.interpolate(geom.project(pt))
            except Exception:
                continue
            dx = (p2.x - lon) * m_per_deg_lon
            dy = (p2.y - lat) * m_per_deg_lat
            d = math.hypot(dx, dy)
            if best_m is None or d < best_m:
                best_m, best_cls = d, classes[int(j)]

        entry = cache.setdefault(key, {"site_id": site_id, "latitude": lat,
                                       "longitude": lon, "features": {}})
        feats = entry.setdefault("features", {})
        if best_m is None or best_m > SEARCH_RADIUS_M:
            none_near += 1
            feats["road_distance_m"] = {
                "value": SEARCH_RADIUS_M, "source": "OpenStreetMap (Geofabrik Nepal extract)",
                "url": "https://download.geofabrik.de/asia/nepal-latest.osm.pbf",
                "method": f"no highway way within {SEARCH_RADIUS_M:.0f} m",
                "fetched_utc": fetched, "raw": None,
                "detail": "genuine absence of mapped roads, not a failed query",
            }
        else:
            found += 1
            feats["road_distance_m"] = {
                "value": round(float(best_m), 1),
                "source": "OpenStreetMap (Geofabrik Nepal extract)",
                "url": "https://download.geofabrik.de/asia/nepal-latest.osm.pbf",
                "method": "nearest point on the nearest highway way, local metric scaling",
                "fetched_utc": fetched, "raw": str(best_cls),
                "detail": f"highway={best_cls}",
            }
            feats["road_class_severity"] = {
                "value": ROAD_CLASS_SEVERITY.get(str(best_cls), 0.4),
                "source": "OpenStreetMap class + stated severity ordering",
                "url": "https://download.geofabrik.de/asia/nepal-latest.osm.pbf",
                "method": "judgement applied to a MEASURED class, not an assumed distance",
                "fetched_utc": fetched, "raw": str(best_cls),
            }
        if i % 100 == 0 or i == len(outstanding):
            print(f"  {i}/{len(outstanding)}  near={found} none_within_2km={none_near} "
                  f"{i/max(1e-9,time.time()-t0):.0f}/s", flush=True)
            tmp = CACHE_PATH.with_suffix(".json.tmp")
            json.dump(cache, open(tmp, "w", encoding="utf-8"), indent=2, sort_keys=True)
            tmp.replace(CACHE_PATH)

    tmp = CACHE_PATH.with_suffix(".json.tmp")
    json.dump(cache, open(tmp, "w", encoding="utf-8"), indent=2, sort_keys=True)
    tmp.replace(CACHE_PATH)
    print(f"\ndone: {found} with a road within {SEARCH_RADIUS_M:.0f} m, "
          f"{none_near} genuinely none, in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
