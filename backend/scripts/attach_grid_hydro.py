#!/usr/bin/env python3
"""Attach real hydro-infrastructure distances to an already-built valley grid.

Separate from ``build_valley_grid.py`` on purpose. The DEM and SoilGrids passes cost
~3 minutes per corridor and should not be repeated to add one OSM layer, so this runs
over the cached grid in place and rewrites it.

Source: the same Geofabrik Nepal extract - ``waterway=dam|weir`` ways plus
``power=plant`` / ``landuse=reservoir`` multipolygons. A cell further than 1.5 km from
any of them keeps ``hydro_distance_m`` None, which is what puts it in some other
category rather than in a quota-filled "HYDRO_INFRASTRUCTURE" bucket.

    python scripts/attach_grid_hydro.py --pbf nepal-latest.osm.pbf
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import valley_grid_terrain as vgt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--radius-m", type=float, default=1500.0)
    ap.add_argument("--grid", default=None)
    args = ap.parse_args()

    grid = vgt.load_grid(args.grid)
    if not grid:
        print("[fatal] no built grid found - run build_valley_grid.py first", flush=True)
        return 1

    rebuilt = {}
    for cid, raw in grid.items():
        cells = [vgt.TerrainCell(**c) for c in raw]
        t0 = time.time()
        n = vgt.attach_hydro_infrastructure(cells, args.pbf, radius_m=args.radius_m)
        print(f"{cid:28s} {n:6d}/{len(cells):6d} cells near hydro  "
              f"{time.time()-t0:5.1f}s", flush=True)
        rebuilt[cid] = cells

    vgt.save_grid(rebuilt, args.grid)
    total = sum(sum(1 for c in v if c.hydro_distance_m is not None)
                for v in rebuilt.values())
    print(f"\n{total} cells carry a measured hydro distance", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
