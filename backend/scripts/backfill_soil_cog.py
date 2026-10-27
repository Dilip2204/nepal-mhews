#!/usr/bin/env python3
"""Backfill REAL soil properties for every coordinate the models touch, from rasters.

Why not the SoilGrids REST API
------------------------------
The obvious route is ISRIC's point-query endpoint, and it works - but it is rate limited
to roughly a handful of requests per minute. Measured on this host: a backfill using it
wrote **3 coordinates in 25 minutes**, which extrapolates to about 136 hours for the 982
coordinates this project needs. That is not a backfill, it is an outage.

ISRIC also publishes the same data as public COG/VRT mosaics over plain HTTPS, readable
with GDAL's /vsicurl - the identical pattern this codebase already uses for the
Copernicus DEM. Opening each layer costs ~13 s once, and every coordinate after that is a
cheap windowed read against an open handle.

The two routes were cross-checked at glc_105 (29.5023, 82.2504) and agree EXACTLY:

    property   REST API   COG/VRT
    clay            252       252
    sand            400       400
    cfvo            144       144
    bdod            111       111

so this is a speed change, not a data change.

Honest absences
---------------
SoilGrids masks built-up land. Kathmandu (27.7172, 85.3240) returns the nodata sentinel
-32768 from every layer, and that is recorded as absent rather than filled with a
plausible number. An urban pixel genuinely has no modelled soil profile.

Unit conversion follows SoilGrids' published d_factor for each property, the same factors
the REST response self-reports, so the two routes cannot silently diverge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
# A larger GDAL block cache pays for itself: nearby coordinates hit the same tiles.
os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "real_site_features.json"

ISRIC_BASE = "/vsicurl/https://files.isric.org/soilgrids/latest/data"

# The whole PROFILE is read, not one horizon, and this matters more than it looks.
#
# An earlier version sampled 0-5 cm only, reasoning that the near-surface horizon governs
# infiltration. That is wrong for the infinite-slope model, which treats the column
# 0 -> z as a single homogeneous layer, and z at glc_105 is 1.45 m. Measured down the
# profile there:
#
#     depth        clay%  sand%  cfvo%  bdod  porosity
#     0-5cm         25.2   40.0   14.4  1.11     0.581
#     100-200cm     24.6   41.6   26.9  1.33     0.498
#
# Bulk density is 20% higher at the failure plane and coarse-fragment content nearly
# doubles. Topsoil density understates the overburden, which overstates porosity, which
# lowers the baseline water table, which RAISES the factor of safety and suppresses
# alerts. Sampling the wrong horizon makes a warning system quieter - the worst
# direction for an error to point.
# Only the horizons that bracket the failure plane are read. Reading all six costs
# ~11 s per coordinate (24 range-read rasters) which is 3 hours for the catalog; these
# three cover 30-200 cm, which contains z for the overwhelming majority of sites, at
# ~5 s. The shallow horizons are deliberately excluded - they are the ones that caused
# the under-alerting bias in the first place.
SOIL_DEPTHS = ("30-60cm", "60-100cm", "100-200cm")
DEPTH_BOUNDS = {"0-5cm": (0, 5), "5-15cm": (5, 15), "15-30cm": (15, 30),
                "30-60cm": (30, 60), "60-100cm": (60, 100), "100-200cm": (100, 200)}

# property -> (our field name, d_factor, converter to model units)
PROPERTIES = {
    "clay": ("clay_fraction", 10.0, lambda pct: pct / 100.0),
    "sand": ("sand_fraction", 10.0, lambda pct: pct / 100.0),
    "cfvo": ("coarse_rock_fragment_fraction", 10.0, lambda pct: pct / 100.0),
    # bdod is cg/cm3 with d_factor 100 -> kg/dm3, which is numerically g/cm3
    "bdod": ("dry_bulk_density_g_cm3", 100.0, lambda v: v),
}

# Default integration depth when the consumer does not state one. 1.5 m is the typical
# Himalayan colluvial failure plane and close to the z the synthesizer computes.
DEFAULT_FAILURE_DEPTH_CM = 150.0


def depth_weighted(profile, to_cm=DEFAULT_FAILURE_DEPTH_CM):  # noqa: D401
    """Thickness-weighted mean over 0 -> to_cm, the column the infinite-slope model
    treats as homogeneous."""
    num = den = 0.0
    for label, value in profile.items():
        lo, hi = DEPTH_BOUNDS.get(label, (None, None))
        if lo is None or value is None or lo >= to_cm:
            continue
        thickness = min(hi, to_cm) - lo
        if thickness <= 0:
            continue
        num += float(value) * thickness
        den += thickness
    return (num / den) if den > 0 else None

PLAUSIBLE = {
    "clay_fraction": (0.0, 1.0),
    "sand_fraction": (0.0, 1.0),
    "coarse_rock_fragment_fraction": (0.0, 1.0),
    "dry_bulk_density_g_cm3": (0.3, 2.2),
}


def load_coordinates() -> list:
    """Every coordinate the models touch: the 491 monitored sites plus the labelled
    ground-truth points the accuracy evaluation needs."""
    coords = {}
    try:
        from app.engine.sites_catalog import load_sites_of_interest

        for s in load_sites_of_interest():
            coords[(round(s.latitude, 4), round(s.longitude, 4))] = s.site_id
    except Exception as exc:
        print(f"[warn] site catalog unavailable: {exc}", flush=True)

    import csv
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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refetch", action="store_true",
                    help="re-read coordinates that already have soil values")
    args = ap.parse_args()

    coords = load_coordinates()
    print(f"coordinates to cover: {len(coords)}", flush=True)

    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.load(open(CACHE_PATH, encoding="utf-8"))
        except ValueError:
            cache = {}
    print(f"existing cache entries: {len(cache)}", flush=True)

    outstanding = []
    for (lat, lon), site_id in coords:
        key = f"{lat:.4f},{lon:.4f}"
        have = ((cache.get(key) or {}).get("features") or {})
        if not args.refetch and "clay_fraction" in have:
            continue
        outstanding.append((key, lat, lon, site_id))
    if args.limit:
        outstanding = outstanding[: args.limit]
    print(f"outstanding: {len(outstanding)}", flush=True)
    if not outstanding:
        return 0

    # Sort north-to-south then west-to-east so consecutive reads land in the same tiles
    # and GDAL's block cache actually helps.
    outstanding.sort(key=lambda r: (-r[1], r[2]))

    opened = {}
    t_open = time.time()
    for prop, (field, factor, convert) in PROPERTIES.items():
        for depth in SOIL_DEPTHS:
            url = f"{ISRIC_BASE}/{prop}/{prop}_{depth}_mean.vrt"
            try:
                opened[(prop, depth)] = (rasterio.open(url), field, factor, convert, url)
            except Exception as exc:
                print(f"  [warn] {prop} {depth}: {type(exc).__name__}", flush=True)
        print(f"  opened {prop}: {sum(1 for k in opened if k[0]==prop)} horizons", flush=True)
    if not opened:
        print("[fatal] no SoilGrids layer could be opened", flush=True)
        return 1
    print(f"opened {len(opened)} layers in {time.time()-t_open:.0f}s", flush=True)

    fetched = time.time()
    done = nodata_hits = implausible = 0
    t0 = time.time()
    for i, (key, lat, lon, site_id) in enumerate(outstanding, 1):
        entry = cache.setdefault(key, {"site_id": site_id, "latitude": lat,
                                       "longitude": lon, "features": {}})
        feats = entry.setdefault("features", {})
        got_any = False
        profiles = {}
        for (prop, depth), (src, field, factor, convert, url) in opened.items():
            try:
                xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
                row, col = src.index(xs[0], ys[0])
                if not (0 <= row < src.height and 0 <= col < src.width):
                    continue
                raw = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
            except Exception:
                continue
            if src.nodata is not None and raw == src.nodata:
                nodata_hits += 1
                continue
            value = float(convert(float(raw) / factor))
            lo, hi = PLAUSIBLE[field]
            if not (lo <= value <= hi):
                implausible += 1
                continue
            profiles.setdefault(field, {"_url": url})[depth] = value

        for field, prof in profiles.items():
            url = prof.pop("_url")
            weighted = depth_weighted(prof)
            if weighted is None:
                continue
            feats[field] = {
                "value": round(weighted, 4),
                "source": "SoilGrids v2.0 (ISRIC) COG mosaic",
                "url": url,
                "method": (f"thickness-weighted mean over 0-{DEFAULT_FAILURE_DEPTH_CM:.0f} cm "
                           f"- the infinite-slope column - from {len(prof)} horizons"),
                "fetched_utc": fetched,
                "profile_by_depth": {k: round(v, 4) for k, v in sorted(prof.items())},
            }
            got_any = True
        if got_any:
            done += 1
        if i % 50 == 0 or i == len(outstanding):
            rate = i / max(1e-9, time.time() - t0)
            print(f"  {i}/{len(outstanding)}  ok={done} nodata={nodata_hits} "
                  f"implausible={implausible}  {rate:.1f}/s", flush=True)
            tmp = CACHE_PATH.with_suffix(".json.tmp")
            json.dump(cache, open(tmp, "w", encoding="utf-8"), indent=2, sort_keys=True)
            tmp.replace(CACHE_PATH)

    tmp = CACHE_PATH.with_suffix(".json.tmp")
    json.dump(cache, open(tmp, "w", encoding="utf-8"), indent=2, sort_keys=True)
    tmp.replace(CACHE_PATH)
    for src, *_ in opened.values():
        try:
            src.close()
        except Exception:
            pass
    print(f"\ndone: {done} coordinates got soil, {nodata_hits} nodata reads "
          f"(SoilGrids masks built-up land), {implausible} rejected as implausible, "
          f"in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
