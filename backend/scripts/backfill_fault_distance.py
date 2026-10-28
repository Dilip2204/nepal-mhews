#!/usr/bin/env python3
"""Recompute every site's distance-to-fault from the measured trace layer.

Why this exists
---------------
``fault_distance_km`` feeds the cohesion-reduction term in both engines. Audited on
2026-10-28 against the merged layer from ``app/ingestion/fault_traces.py``, the two
source CSVs behind the 491-site catalogue were in completely different states:

    source                        n     r(stored, measured)   median |err|
    nepal_training_points.csv   472          +1.000              0.00 km
    himalayan_cryosphere_*.csv   24          +0.134              7.30 km

The 472 NASA-GLC landslide points are genuine: they reproduce a GEM-derived distance
exactly, to under 10 m, for every single row. The 24 cryosphere sites do not. Their
values are uncorrelated with the distance to any mapped trace, they ascend roughly
monotonically down the file, and the errors are large and two-sided:

    Langtang_Lirung_2015_Detachment   stored  3.8 km   measured 17.1 km
    Thulagi_Dona_Hanging_Glacier      stored  9.1 km   measured  2.3 km
    Manaslu_East_Trench_Avalanche     stored  7.8 km   measured  1.6 km
    Dig_Tsho_Langmoche_Ice_Avalanche  stored 14.2 km   measured 33.4 km

Those are not measurements. The fabrication cuts both ways - it has been making some
sites look more dangerous than the mapped faults justify and others look safer - so
correcting it is not a one-directional safety trade.

The second reason to re-run this is that the layer itself got better. Until 2026-10-28
neither GEM nor HimaTibetMap carried the Main Boundary Thrust or the Main Central
Thrust, so every corridor sat 89-117 km from the nearest mapped thrust and the
fault-damage term was saturated into silence. The Whipple et al. (2024) compilation
supplies both, and the median site distance drops from 32.5 km to 9.3 km.

What is measured and what is judged
-----------------------------------
The DISTANCE is measured, in UTM 45N, against published trace geometry. Which published
trace a fault's identity comes from is recorded per trace in the layer, including the
caveat that the Whipple shapefiles name a fault only by filename. Nothing here invents
a value: a coordinate with no trace within the layer's window gets no number at all.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED = DATA_DIR / "processed"

#: The CSVs that feed `app/engine/sites_catalog.py`.
TARGETS = (
    PROCESSED / "nepal_training_points.csv",
    PROCESSED / "himalayan_cryosphere_avalanches_glofs.csv",
)

#: Nothing is written without a dated backup beside it.
BACKUP_SUFFIX = "fault_backfill"


def build_index():
    """Project the merged layer into UTM 45N and index it for nearest-trace queries."""
    from pyproj import Transformer
    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    from app.ingestion import fault_traces as ft

    if not ft.MERGED_PATH.exists():
        raise SystemExit(
            f"{ft.MERGED_PATH} not found - run fault_traces.write_nepal_fault_layer() first"
        )
    layer = json.loads(ft.MERGED_PATH.read_text(encoding="utf-8"))
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)

    geoms, meta = [], []
    for feat in layer["features"]:
        props = feat["properties"]
        for part in ft._line_parts(feat["geometry"]):
            if len(part) < 2:
                continue
            xs, ys = tr.transform([p[0] for p in part], [p[1] for p in part])
            geoms.append(LineString(list(zip(xs, ys))))
            meta.append(props)
    if not geoms:
        raise SystemExit("merged fault layer contains no line geometry")
    return tr, STRtree(geoms), geoms, meta, layer["properties"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write the CSVs; without it the script only reports")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from shapely.geometry import Point

    tr, tree, geoms, meta, layer_props = build_index()
    print(f"layer: {layer_props['feature_count']} features, "
          f"{len(geoms)} line parts; expected faults {layer_props.get('expected_faults')}",
          flush=True)
    if layer_props.get("known_absent"):
        print(f"  [warn] still absent from the layer: {layer_props['known_absent']}",
              flush=True)

    stamp = time.strftime("%Y%m%d")
    total_changed = 0

    for path in TARGETS:
        if not path.exists():
            print(f"[skip] {path.name} not present", flush=True)
            continue
        df = pd.read_csv(path)
        if "fault_distance_km" not in df.columns:
            print(f"[skip] {path.name} has no fault_distance_km column", flush=True)
            continue

        xs, ys = tr.transform(df["longitude"].tolist(), df["latitude"].tolist())
        pts = [Point(a, b) for a, b in zip(xs, ys)]
        idx = tree.nearest(pts)
        measured = np.array([pts[i].distance(geoms[j]) for i, j in enumerate(idx)]) / 1000.0
        nearest_name = [meta[j].get("fault_name") or "<unnamed>" for j in idx]
        nearest_class = [meta[j].get("structure_class") for j in idx]

        stored = df["fault_distance_km"].to_numpy(dtype=float)
        err = np.abs(stored - measured)
        r = float(np.corrcoef(stored, measured)[0, 1]) if len(stored) > 2 else float("nan")

        print(f"\n{path.name}: n={len(df)}  r(stored,measured)={r:+.3f}  "
              f"median|err|={np.median(err):.2f} km  "
              f"rows changing >0.01 km: {(err > 0.01).sum()}", flush=True)
        import collections
        print("  nearest structure now: "
              f"{dict(collections.Counter(nearest_class))}", flush=True)
        worst = np.argsort(-err)[:5]
        for i in worst:
            label = df.iloc[i].get("name", df.iloc[i].get("site_id", i))
            print(f"    {str(label)[:34]:34s} {stored[i]:7.2f} -> {measured[i]:7.2f} km "
                  f"({nearest_name[i]})", flush=True)

        total_changed += int((err > 0.01).sum())

        if args.apply:
            backup = path.with_suffix(f".csv.{BACKUP_SUFFIX}_{stamp}.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            df["fault_distance_km"] = np.round(measured, 2)
            df["fault_nearest_name"] = nearest_name
            df["fault_nearest_class"] = nearest_class
            tmp = path.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(path)
            print(f"  written; previous values kept at {backup.name}", flush=True)

    if not args.apply:
        print(f"\nDRY RUN - {total_changed} rows would change. Re-run with --apply to write.",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
