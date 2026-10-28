#!/usr/bin/env python3
"""Valley-scale terrain metrics for every catalog site, read from the Copernicus DEM.

Why this exists
---------------
``SiteInfo.local_relief_m`` in the catalog is measured over a **330 m** window. That is
the right scale for a shallow translational slide and the wrong scale for a deep-seated
gravitational slope deformation, which involves a whole valley flank. Measured on the
330 m field, only 7 of 491 catalog sites exceed 500 m of relief - an artefact of the
window, not a statement about Nepal's topography.

This script measures relief over a **valley-scale window** (default 2 km radius, i.e. a
4 km box) directly from Copernicus DEM GLO-30 Cloud-Optimized GeoTIFFs over ``/vsicurl``,
and writes the result to ``backend/data/valley_terrain_cache.json``.

Rules this script holds itself to
---------------------------------
1. **Measured or absent.** A site whose DEM window cannot be read, or that lands on
   nodata, is written with ``"status": "UNAVAILABLE"`` and no numbers. Nothing is
   interpolated from neighbouring sites and no plausible default is substituted.
2. **One request per tile-window, serialised.** No parallel flooding of the AWS bucket.
   Results are cached to disk and re-runs skip sites already measured.
3. Every record carries its provenance: the exact COG URL(s), the window geometry, the
   read timestamp.

Usage
-----
    cd /data-mqs/users/djena/avalanche_detector/nepal-mhews
    backend/.venv/bin/python3 backend/scripts/backfill_valley_relief.py [--radius-m 2000]
                                                                        [--limit N]
                                                                        [--force]
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
if os.environ.get("_MHEWS_RELIEF_REEXEC") != "1" and any(
    os.environ.get(k) != v for k, v in _REQUIRED_ENV.items()
):
    os.environ.update(_REQUIRED_ENV)
    os.environ["_MHEWS_RELIEF_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np

from app.engine.sites_catalog import load_sites_of_interest
from app.ingestion.valley_terrain import (
    CACHE_PATH,
    COPERNICUS_COG_TEMPLATE,
    DEFAULT_RADIUS_M,
    measure_valley_terrain,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-measure sites already cached")
    ap.add_argument("--sleep", type=float, default=0.15, help="pause between reads (politeness)")
    args = ap.parse_args()

    cache: Dict[str, dict] = {}
    if CACHE_PATH.exists() and not args.force:
        cache = json.loads(CACHE_PATH.read_text())

    sites = load_sites_of_interest()
    todo = [s for s in sites if s.site_id not in cache]
    if args.limit:
        todo = todo[: args.limit]
    print(f"catalog {len(sites)} sites | cached {len(cache)} | to measure {len(todo)}")
    print(f"window radius {args.radius_m:.0f} m  ({2*args.radius_m/1000:.1f} km box)")
    print(f"source template: {COPERNICUS_COG_TEMPLATE}")

    t0 = time.time()
    ok = fail = 0
    for i, s in enumerate(todo, 1):
        rec = measure_valley_terrain(s.latitude, s.longitude, radius_m=args.radius_m)
        cache[s.site_id] = rec
        if rec.get("status") == "MEASURED":
            ok += 1
        else:
            fail += 1
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            print(
                f"  [{i}/{len(todo)}] ok={ok} unavailable={fail} "
                f"{el:.0f}s ({el/max(i,1):.2f}s/site)",
                flush=True,
            )
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True))
        if args.sleep:
            time.sleep(args.sleep)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True))

    measured = [v for v in cache.values() if v.get("status") == "MEASURED"]
    print(f"\nwrote {CACHE_PATH}  ({len(cache)} records, {len(measured)} measured)")
    if measured:
        rel = np.array([v["valley_relief_m"] for v in measured], dtype=float)
        print(
            "valley relief (m): "
            f"min {rel.min():.0f}  p25 {np.percentile(rel,25):.0f}  "
            f"median {np.median(rel):.0f}  p75 {np.percentile(rel,75):.0f}  max {rel.max():.0f}"
        )
        for thr in (300.0, 500.0, 800.0, 1000.0):
            print(f"  sites with valley relief > {thr:.0f} m: {int((rel>thr).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
