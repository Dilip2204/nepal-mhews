#!/usr/bin/env python3
"""Does the morphometric DSGSD rule actually separate deep-seated from shallow failures?

Why this exists
---------------
The DSGSD factor in ``app/physics/feature_synthesizer.py`` can be driven two ways:

  MAPPED    - the site sits on a polygon an author mapped as deep-seated
  INFERRED  - the site satisfies a morphometric rule (large valley-scale relief, steep
              flank, concave toe)

The morphometric rule is an *inference*, and this project's standing rule is that an
inference must be labelled as modelled. But "labelled" is not the same as "tested". This
script tests it, against the only mapped ground truth that exists for Nepal:

  Multi-temporal Landslide Inventory for the Far-Western region of Nepal
  doi:10.5281/zenodo.4290100  (CC-BY-4.0)
  26,350 polygons over a 6,409 km2 AOI, each classified by the authors as
  'Depth' (deep-seated, n=1,413) or 'Shallow' (n=24,915).

For each polygon it measures valley-scale terrain from Copernicus DEM GLO-30 at the
polygon's representative point, then asks: of the polygons the morphometric rule flags,
what fraction are actually the deep-seated class? A rule with no skill will flag
deep-seated and shallow polygons at the same rate, and the printed lift will be 1.0.

DEM tiles
---------
The four GLO-30 tiles covering the AOI are ~43 MB each. They are downloaded once into
``--dem-cache`` (default ``backend/data/raw/dem_tiles``, which is gitignored) rather than
being range-read 26,000 times. That is the polite option: 4 requests instead of 26,000.

Usage
-----
    cd /data-mqs/users/djena/avalanche_detector/nepal-mhews
    backend/.venv/bin/python3 backend/scripts/validate_dsgsd_morphometry.py
"""

from __future__ import annotations

import os
import sys

_REQUIRED_ENV = {
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}
if os.environ.get("_MHEWS_DSGSD_VAL_REEXEC") != "1" and any(
    os.environ.get(k) != v for k, v in _REQUIRED_ENV.items()
):
    os.environ.update(_REQUIRED_ENV)
    os.environ["_MHEWS_DSGSD_VAL_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np

from app.ingestion.valley_terrain import DEFAULT_RADIUS_M, derive_valley_metrics
from app.physics.dsgsd import MORPHOMETRIC_CRITERIA, morphometric_indicator

DATA_DIR = BACKEND_DIR / "data"
POLY_PATH = DATA_DIR / "processed" / "dsgsd_mapped_deepseated_farwest_nepal.geojson"
RAW_INVENTORY = (
    DATA_DIR
    / "raw"
    / "inventories"
    / "LandslideInventory_FarWesternNepal"
    / "LandslideInventory_FarWesternNepal_Pol.shp"
)
TILES = ("N28_00_E080", "N28_00_E081", "N29_00_E080", "N29_00_E081")
TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{t}_00_DEM/Copernicus_DSM_COG_10_{t}_00_DEM.tif"
)


class TileMosaic:
    """The 4 GLO-30 tiles over the Far-Western AOI, held in memory for fast windowing."""

    def __init__(self, cache_dir: Path):
        import rasterio

        self.tiles = []
        for t in TILES:
            p = cache_dir / f"{t}.tif"
            if not p.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
                url = TILE_URL.format(t=t)
                print(f"  downloading {url}", flush=True)
                urllib.request.urlretrieve(url, p)
                time.sleep(1.0)  # polite
            with rasterio.open(p) as src:
                arr = src.read(1).astype(np.float32)
                arr[arr < -1000.0] = np.nan
                self.tiles.append(
                    {
                        "path": str(p),
                        "arr": arr,
                        "transform": src.transform,
                        "bounds": src.bounds,
                        "res_lon": abs(src.transform.a),
                        "res_lat": abs(src.transform.e),
                    }
                )
        self.res_lon = self.tiles[0]["res_lon"]
        self.res_lat = self.tiles[0]["res_lat"]

    def window(self, lat: float, lon: float, radius_m: float) -> Tuple[np.ndarray, float]:
        deg_lat = radius_m / 111_320.0
        deg_lon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.05))
        half_y = int(round(deg_lat / self.res_lat))
        half_x = int(round(deg_lon / self.res_lon))
        n = (2 * half_y + 1, 2 * half_x + 1)
        out = np.full(n, np.nan, dtype=np.float64)
        for t in self.tiles:
            b = t["bounds"]
            if not (b.left - 0.01 <= lon <= b.right + 0.01 and b.bottom - 0.01 <= lat <= b.top + 0.01):
                # still may overlap the window edge; test the window box instead
                if (
                    lon + deg_lon < b.left
                    or lon - deg_lon > b.right
                    or lat + deg_lat < b.bottom
                    or lat - deg_lat > b.top
                ):
                    continue
            inv = ~t["transform"]
            col, row = inv * (lon, lat)
            r0 = int(round(row)) - half_y
            c0 = int(round(col)) - half_x
            arr = t["arr"]
            rs, re_ = max(r0, 0), min(r0 + n[0], arr.shape[0])
            cs, ce = max(c0, 0), min(c0 + n[1], arr.shape[1])
            if rs >= re_ or cs >= ce:
                continue
            patch = arr[rs:re_, cs:ce].astype(np.float64)
            dst = out[rs - r0 : re_ - r0, cs - c0 : ce - c0]
            out[rs - r0 : re_ - r0, cs - c0 : ce - c0] = np.where(
                np.isfinite(dst), dst, patch
            )
        cell_m = 0.5 * (
            self.res_lat * 111_320.0 + self.res_lon * 111_320.0 * math.cos(math.radians(lat))
        )
        return out, cell_m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem-cache", type=Path, default=DATA_DIR / "raw" / "dem_tiles")
    ap.add_argument("--shallow-sample", type=int, default=4000)
    ap.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import geopandas as gpd

    if not RAW_INVENTORY.exists():
        print(f"missing {RAW_INVENTORY}", file=sys.stderr)
        print("fetch doi:10.5281/zenodo.4290100 into backend/data/raw/inventories/", file=sys.stderr)
        return 2

    print("loading mapped inventory ...", flush=True)
    g = gpd.read_file(RAW_INVENTORY, columns=["OBJECTID", "Depth", "area_sqm", "Depth_DEM"])
    g = g[g["Depth"].isin(["Depth", "Shallow"])].to_crs(4326)
    deep = g[g["Depth"] == "Depth"].copy()
    shallow = g[g["Depth"] == "Shallow"].copy()
    rng = np.random.default_rng(args.seed)
    if len(shallow) > args.shallow_sample:
        idx = rng.choice(len(shallow), size=args.shallow_sample, replace=False)
        shallow = shallow.iloc[np.sort(idx)].copy()
    print(f"  deep-seated {len(deep)}   shallow sample {len(shallow)}", flush=True)

    print("loading DEM tiles ...", flush=True)
    mosaic = TileMosaic(args.dem_cache)

    rows: List[dict] = []
    for label, gdf in (("deep", deep), ("shallow", shallow)):
        pts = gdf.geometry.representative_point()
        t0 = time.time()
        for i, (pt, oid, area, ddem) in enumerate(
            zip(pts, gdf["OBJECTID"], gdf["area_sqm"], gdf["Depth_DEM"]), 1
        ):
            try:
                dem, cell = mosaic.window(pt.y, pt.x, args.radius_m)
                m = derive_valley_metrics(dem, cell, radius_m=args.radius_m)
            except Exception:
                continue
            ind = morphometric_indicator(
                valley_relief_m=m["valley_relief_m"],
                slope_deg=m["valley_mean_slope_deg"],
                valley_profile_curvature=m["valley_profile_curvature"],
                scarp_relief_above_m=m["scarp_relief_above_m"],
            )
            rows.append(
                {
                    "class": label,
                    "object_id": int(oid),
                    "area_m2": float(area),
                    "mapped_depth_dem_m": float(ddem) if ddem == ddem else None,
                    "lat": float(pt.y),
                    "lon": float(pt.x),
                    **{k: m[k] for k in (
                        "valley_relief_m",
                        "valley_mean_slope_deg",
                        "valley_profile_curvature",
                        "scarp_relief_above_m",
                        "centre_elevation_m",
                    )},
                    "indicator": ind["indicator"],
                    "criteria_met": ind["criteria_met"],
                }
            )
            if i % 500 == 0:
                print(f"  {label} {i}/{len(gdf)}  {time.time()-t0:.0f}s", flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    print("\n" + "=" * 86)
    print("VALLEY-SCALE TERRAIN AT MAPPED POLYGONS "
          f"(Copernicus GLO-30, {args.radius_m:.0f} m radius window)")
    print("=" * 86)
    print(f"criteria under test: {json.dumps(MORPHOMETRIC_CRITERIA)}")
    print()
    for col in ("valley_relief_m", "valley_mean_slope_deg", "valley_profile_curvature",
                "scarp_relief_above_m"):
        d = df[df["class"] == "deep"][col]
        s = df[df["class"] == "shallow"][col]
        print(f"{col:>26}  deep median {d.median():10.2f}   shallow median {s.median():10.2f}")

    print()
    base = len(df[df["class"] == "deep"]) / max(len(df), 1)
    print(f"base rate of the deep-seated class in this sample: {100*base:.2f}%")
    print(f"{'rule':>34} {'flags deep':>12} {'flags shallow':>14} {'lift':>7}")
    for name, mask in (
        ("relief > 500 m", df["valley_relief_m"] > MORPHOMETRIC_CRITERIA["min_valley_relief_m"]),
        ("mean flank slope > 32 deg",
         df["valley_mean_slope_deg"] > MORPHOMETRIC_CRITERIA["min_slope_deg"]),
        ("concave flank (curv < 0)", df["valley_profile_curvature"] < 0.0),
        ("scarp relief above > 300 m",
         df["scarp_relief_above_m"] > MORPHOMETRIC_CRITERIA["min_scarp_relief_above_m"]),
        ("ALL THREE (indicator == 1)", df["indicator"] >= 1.0),
        ("indicator > 0", df["indicator"] > 0.0),
    ):
        pd_ = float(mask[df["class"] == "deep"].mean())
        ps_ = float(mask[df["class"] == "shallow"].mean())
        lift = (pd_ / ps_) if ps_ > 0 else float("inf")
        print(f"{name:>34} {100*pd_:11.1f}% {100*ps_:13.1f}% {lift:7.2f}")

    print()
    print("Read the lift column as: how many times more often the rule fires on a mapped")
    print("deep-seated polygon than on a mapped shallow one. 1.0 means no skill at all.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(args.json, orient="records")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
