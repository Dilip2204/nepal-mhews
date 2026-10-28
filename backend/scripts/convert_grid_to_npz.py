#!/usr/bin/env python3
"""Convert a legacy single-file JSON valley grid to per-corridor columnar .npz.

The first build wrote every corridor into one JSON. That reached 52 MB with three
corridors and would have passed 300 MB with seven, so a request for Langtang had to
parse Karnali's 198,554 cells to reach it - about 4 s per cold request, with a memory
spike well past the array data it yielded.

This reads that file once and writes one compressed npz per corridor, which is what
``valley_grid_terrain.load_corridor`` now reads. Run it once; afterwards
``build_valley_grid.py`` writes npz directly.

    python scripts/convert_grid_to_npz.py [--json backend/data/valley_grid_terrain.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import valley_grid_terrain as vgt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(vgt.DATA_DIR / "valley_grid_terrain.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.json)
    if not src.exists():
        print(f"[fatal] {src} not found", flush=True)
        return 1

    print(f"reading {src} ({src.stat().st_size/1e6:.0f} MB) ...", flush=True)
    with open(src, encoding="utf-8") as fh:
        data = json.load(fh)

    # Converted one corridor at a time: holding all of them as TerrainCell objects at
    # once is exactly the memory profile this change exists to avoid.
    for cid, rows in data.items():
        cells = []
        for r in rows:
            r = {k: v for k, v in r.items() if k in vgt.TerrainCell.__dataclass_fields__}
            cells.append(vgt.TerrainCell(**r))
        vgt.save_grid({cid: cells}, args.out)
        del cells
        print(f"  {cid:28s} {len(rows):7d} cells -> npz", flush=True)

    base = Path(args.out) if args.out else vgt.GRID_DIR
    total_mb = sum(p.stat().st_size for p in base.glob("*.npz")) / 1e6
    print(f"\nwrote {len(data)} corridors, {total_mb:.0f} MB total "
          f"(from {src.stat().st_size/1e6:.0f} MB of JSON) -> {base}", flush=True)
    print("the JSON file is now redundant and can be deleted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
