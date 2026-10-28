#!/usr/bin/env python3
"""Build the REAL valley grid: every cell's terrain, soil, roads and faults measured.

What this is for
----------------
``valley_grid_physics._generate_corridor_cells`` invented its grid. Slope, curvature,
elevation, clay, sand, coarse fragments, bulk density, road distance, river distance and
fault distance were each a draw from ``np.random`` seeded on ``hash(corridor_id)`` - a
seed that is not even stable between processes, so the "grid" reshuffled on every API
restart. 320 cells stood in for a corridor whose bounding box holds ~36,000 at 240 m.

This script replaces that with measurements, cached once:

    Copernicus DEM GLO-30  ->  elevation, slope, aspect, profile curvature, local relief
    SoilGrids v2.0 (ISRIC) ->  clay, sand, coarse fragments, bulk density (30-200 cm)
    OpenStreetMap          ->  nearest road + class, nearest river, nearest settlement
    GEM Global Active Faults -> distance to the nearest mapped active fault trace

Scope
-----
The user's instruction was explicit: not only river corridors, but "other slopes too that
include the human settlements, and roads and places where human are present and there is
risk to human or human made structures". The corridor bounding boxes are already far
wider than the river line - Karnali spans ~150 x 122 km, whole districts - so a real
slope filter over them reaches ridge settlements and hill roads that the river-centred
generator never touched. Cells below 20 deg are dropped because the translational
failures this system models do not occur there.

Honest absence
--------------
A cell SoilGrids cannot resolve - bare rock and ice above the soil mask - keeps None for
its soil fields rather than a plausible fill. Roughly 18% of Langtang's cells are in that
category, and that is the correct answer for a glacier.

Usage
-----
    python scripts/build_valley_grid.py --pbf nepal-latest.osm.pbf \\
                                        --faults gem_faults.geojson
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import valley_grid_terrain as vgt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbf", required=True, help="Geofabrik nepal-latest.osm.pbf")
    ap.add_argument("--faults", required=True, help="GEM active faults GeoJSON")
    ap.add_argument("--corridors", default=None,
                    help="comma-separated subset (default: all)")
    ap.add_argument("--min-slope", type=float, default=vgt.MIN_SLOPE_DEG)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from app.physics.valley_grid_physics import PRIMARY_CORRIDORS

    wanted = (args.corridors.split(",") if args.corridors
              else list(PRIMARY_CORRIDORS.keys()))

    # The road index costs ~30 s to build over 617,562 ways and is reused for every
    # corridor; building it per corridor would dominate the whole run.
    print(f"loading OSM road index from {args.pbf} ...", flush=True)
    t0 = time.time()
    road_index = vgt.load_road_index(args.pbf)
    print(f"  {len(road_index[1])} highway ways in {time.time()-t0:.0f}s", flush=True)

    grids = {}
    existing = vgt.load_grid(args.out)
    for cid in wanted:
        meta = PRIMARY_CORRIDORS[cid]
        print(f"\n=== {cid}  bbox={meta.bbox} ===", flush=True)
        t0 = time.time()
        cells = vgt.build_corridor_grid(cid, meta.bbox, min_slope_deg=args.min_slope)
        print(f"  DEM      {len(cells):7d} cells        {time.time()-t0:6.1f}s", flush=True)
        if not cells:
            print("  [warn] no cells - DEM unreadable, corridor skipped", flush=True)
            continue

        t0 = time.time()
        n = vgt.attach_soil(cells, meta.bbox)
        print(f"  soil     {n:7d} field fills  {time.time()-t0:6.1f}s", flush=True)

        t0 = time.time()
        vgt.attach_fault_distance(cells, args.faults)
        print(f"  faults            attached    {time.time()-t0:6.1f}s", flush=True)

        t0 = time.time()
        vgt._attach_road_exposure(cells, road_index, vgt.EXPOSURE_RADIUS_M)
        print(f"  roads             attached    {time.time()-t0:6.1f}s", flush=True)

        t0 = time.time()
        vgt.attach_rivers_and_settlements(cells, args.pbf)
        print(f"  rivers/places     attached    {time.time()-t0:6.1f}s", flush=True)

        from collections import Counter
        exp = Counter(c.exposure for c in cells)
        have_soil = sum(1 for c in cells if c.clay_fraction is not None)
        print(f"  -> exposure {dict(exp)}", flush=True)
        print(f"  -> soil resolved {have_soil}/{len(cells)} "
              f"({100.0*have_soil/len(cells):.0f}%)", flush=True)

        grids[cid] = cells
        # Persist after every corridor: a 20-minute run should not lose everything to a
        # transient /vsicurl failure on the last one. Merge with what is already on disk
        # so a --corridors subset run never deletes the others.
        merged = {k: v for k, v in existing.items() if k not in grids}
        vgt.save_grid({**{k: [vgt.TerrainCell(**c) for c in v] for k, v in merged.items()},
                       **grids}, args.out)

    total = sum(len(v) for v in grids.values())
    print(f"\nbuilt {total} real cells across {len(grids)} corridors -> "
          f"{args.out or vgt.GRID_CACHE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
