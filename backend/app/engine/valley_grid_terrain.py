"""Real DEM-derived valley grid cells - terrain measured, not generated.

What this replaces
------------------
``valley_grid_physics._generate_corridor_cells`` built its "240 m terrain grid" like this:

    rng = np.random.default_rng(abs(hash(corridor_id)) % (2**31))

Every elevation and slope the dashboard displayed, and every factor of safety computed
from them, came out of that generator. Three further problems compounded it:

* **The seed is not stable.** Python's string hash is ``PYTHONHASHSEED``-dependent.
  Three runs of ``abs(hash('langtang')) % 2**31`` gave 110651952, 475701804 and
  1872610634, so the "grid" reshuffled on every API restart - despite a comment reading
  "Reproducible pseudo-random seed ... for spatial consistency".
* **The category split was hardcoded** as ``n_cells * 0.32 / 0.34 / 0.16 / 0.13 / 0.05``,
  so a reported "911 settlement cells" was ``int(round(2850 * 0.32))`` - not a count of
  anything observed.
* **The extent was fictional.** 2,850 cells were presented as contiguous 240 m coverage
  of seven corridors whose bounding boxes actually hold **926,225** cells at 240 m. The
  grid covered 0.3% of what it claimed, and was not contiguous.

Meanwhile ``app.ingestion.valley_terrain`` already read the real Copernicus DEM GLO-30
over ``/vsicurl`` and returned sane values - 1712 m relief and 33.4 deg mean slope at
Dhunche, 3468 m and 40.4 deg at Langtang Lirung. The measurement existed; nothing was
plugged into it. This module does the plugging.

What a cell is here
-------------------
A real 240 m cell on a regular grid over the corridor, carrying elevation, slope, aspect
and curvature read from the DEM by decimated windowed COG reads, plus its distance to
the nearest **mapped** OSM road. Cells are kept where a landslide could plausibly
threaten people: steep enough to fail, and close enough to a road or settlement to
matter. Flat valley floor and uninhabited back-country are dropped, which is what makes
the problem tractable without inventing anything.

Nothing here is interpolated between cells and nothing is defaulted. A cell the DEM
cannot resolve is omitted, not filled.

Relationship to ``app.engine.valley_grid_engine``
-------------------------------------------------
Two grid builders exist and they are not duplicates; read this before extending either.

``valley_grid_engine`` frames corridors as 2 km buffers around OpenStreetMap river
centrelines, clipped to reaches within 5 km of a monitored site, and does its geometry
properly in UTM (EPSG:32645). Its scale analysis is the one worth keeping: the seven
basins as bounding boxes are 7.07 million 100 m cells, Karnali alone 2.25M.

This module frames them as the corridor bounding boxes filtered by slope and by human
exposure instead, because river centrelines were the wrong scope for the question being
asked: "this includes only river corridors, include other slopes too that include the
human settlements, and roads and places where human are present and there is risk to
human or human made structures". A centreline buffer structurally cannot contain a ridge
settlement. Langtang's cells reach 18.7 km from the nearest mapped river, which is the
point.

THIS module is the one the live API reads. ``valley_grid_engine`` is not wired into any
route; it is kept because its river-reach selection and its UTM handling are both better
than what is here - this module works in degrees with a cos(latitude) correction, which
is fine at 240 m in Nepal's latitude band and would not be at higher latitudes. A future
consolidation should take the geodesy from there and the scope from here, not pick one
module and delete the other.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nepal_mhews.valley_grid_terrain")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
# Legacy single-file cache. Superseded by GRID_DIR (one .npz per corridor); kept only
# so convert_grid_to_npz.py can find and migrate an older build.
GRID_CACHE = DATA_DIR / "valley_grid_terrain.json"

DEM_TILE = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/"
            "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")

CELL_SIZE_M = 240.0

# A slope shallower than this does not produce the translational failures this system
# monitors; including them would inflate the grid without adding hazard.
MIN_SLOPE_DEG = 20.0

# How close a cell has to be to a mapped road before we call it exposed. 1 km is a
# deliberate over-reach: a slope failing 800 m above a highway still reaches it.
EXPOSURE_RADIUS_M = 1000.0


@dataclass
class TerrainCell:
    """One real 240 m cell. Every field is measured or explicitly absent."""

    corridor_id: str
    row: int
    col: int
    lat: float
    lon: float
    elevation_m: float
    slope_deg: float
    aspect_deg: float
    local_relief_m: float
    profile_curvature: float = 0.0
    # Everything below is filled by an attach_* pass. None means the source did not
    # resolve this cell - never a stand-in value.
    road_distance_m: Optional[float] = None
    road_class: Optional[str] = None
    distance_to_river_m: Optional[float] = None
    fault_distance_km: Optional[float] = None
    settlement_distance_m: Optional[float] = None
    settlement_name: Optional[str] = None
    settlement_kind: Optional[str] = None
    hydro_distance_m: Optional[float] = None
    hydro_name: Optional[str] = None
    clay_fraction: Optional[float] = None
    sand_fraction: Optional[float] = None
    coarse_rock_fraction: Optional[float] = None
    dry_bulk_density_g_cm3: Optional[float] = None
    exposure: str = "TERRAIN_ONLY"      # ROAD | SETTLEMENT | TERRAIN_ONLY

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tiles_for_bbox(bbox: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return [(la, lo)
            for la in range(int(math.floor(min_lat)), int(math.floor(max_lat)) + 1)
            for lo in range(int(math.floor(min_lon)), int(math.floor(max_lon)) + 1)]


def _slope_aspect(elev, px_m: float):
    """Slope, aspect and profile curvature from the DEM.

    Aspect is the compass bearing of steepest DESCENT (0 = N, 90 = E), which is the
    convention the physics uses for the downslope direction.
    """
    import numpy as np

    dzdy, dzdx = np.gradient(elev, px_m)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdy))) % 360.0

    # Profile curvature: the second derivative of elevation along the steepest-descent
    # line, in 1/100 m. Negative is concave (convergent, water-collecting, the hollows
    # that fail); positive is convex (divergent noses). The physics reads the sign, so
    # it has to be the real one rather than rng.uniform(-0.45, 0.15).
    d2y, d2x = np.gradient(dzdx, px_m)
    _, d2yy = np.gradient(dzdy, px_m)
    p = dzdx ** 2 + dzdy ** 2
    denom = np.where(p > 1e-12, p * np.power(1.0 + p, 1.5), np.nan)
    curv = -((dzdx ** 2) * d2x + 2.0 * dzdx * dzdy * d2y + (dzdy ** 2) * d2yy) / denom
    return slope, aspect, np.nan_to_num(curv * 100.0, nan=0.0, posinf=0.0, neginf=0.0)


def build_corridor_grid(
    corridor_id: str,
    bbox: Tuple[float, float, float, float],
    *,
    min_slope_deg: float = MIN_SLOPE_DEG,
    exposure_radius_m: float = EXPOSURE_RADIUS_M,
    road_index: Optional[Any] = None,
) -> List[TerrainCell]:
    """Real 240 m cells over one corridor, from decimated Copernicus DEM reads.

    ``road_index`` is an optional (STRtree, geometries, classes) triple from
    :func:`load_road_index`; pass it once and reuse across corridors rather than
    re-reading a 415 MB extract per corridor.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
    os.environ.setdefault("VSI_CACHE", "TRUE")

    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    deg_lat = CELL_SIZE_M / 111_320.0
    deg_lon = CELL_SIZE_M / (111_320.0 * max(0.15, math.cos(math.radians(mid_lat))))

    cells: List[TerrainCell] = []
    for tile_lat, tile_lon in _tiles_for_bbox(bbox):
        url = DEM_TILE.format(lat=tile_lat, lon=tile_lon)
        # Clip the corridor to this tile.
        w = max(min_lon, float(tile_lon)); e = min(max_lon, float(tile_lon + 1))
        s = max(min_lat, float(tile_lat)); n = min(max_lat, float(tile_lat + 1))
        if w >= e or s >= n:
            continue
        n_cols = max(2, int((e - w) / deg_lon))
        n_rows = max(2, int((n - s) / deg_lat))
        try:
            with rasterio.open(url) as src:
                window = from_bounds(w, s, e, n, src.transform)
                # Decimated read: pull the tile straight down to ~240 m instead of
                # reading 30 m and averaging, which would move an order of magnitude
                # more bytes for the same answer.
                elev = src.read(1, window=window, out_shape=(n_rows, n_cols),
                                resampling=rasterio.enums.Resampling.average).astype("float64")
                nodata = src.nodata
        except Exception as exc:
            logger.warning("DEM tile N%02dE%03d unreadable for %s: %s",
                           tile_lat, tile_lon, corridor_id, exc)
            continue

        valid = np.isfinite(elev)
        if nodata is not None:
            valid &= elev != nodata
        valid &= elev > -400.0
        if not valid.any():
            continue

        masked = np.where(valid, elev, np.nan)
        slope, aspect, curv = _slope_aspect(masked, CELL_SIZE_M)
        # Local relief over a ~1.2 km window (5 cells), the scale at which a valley
        # flank reads as a flank rather than as surface roughness. Separable rank
        # filters, not a Python double loop - a 500x500 tile is 250k cells and the
        # naive version takes minutes per tile for an identical answer.
        from scipy.ndimage import maximum_filter, minimum_filter

        filled = np.where(np.isfinite(masked), masked, float(np.nanmedian(masked)))
        relief = maximum_filter(filled, size=5) - minimum_filter(filled, size=5)

        keep = valid & np.isfinite(slope) & (slope >= float(min_slope_deg))
        rows, cols = np.nonzero(keep)
        for r, c in zip(rows, cols):
            lat = n - (r + 0.5) * (n - s) / n_rows
            lon = w + (c + 0.5) * (e - w) / n_cols
            cells.append(TerrainCell(
                corridor_id=corridor_id, row=int(r), col=int(c),
                lat=round(float(lat), 5), lon=round(float(lon), 5),
                elevation_m=round(float(elev[r, c]), 1),
                slope_deg=round(float(slope[r, c]), 2),
                aspect_deg=round(float(aspect[r, c]), 1),
                local_relief_m=round(float(relief[r, c]), 1)
                if np.isfinite(relief[r, c]) else 0.0,
                profile_curvature=round(float(curv[r, c]), 4),
            ))

    if road_index is not None and cells:
        _attach_road_exposure(cells, road_index, exposure_radius_m)
    return cells


def load_road_index(pbf_path: str):
    """(STRtree, geometries, classes) over every mapped highway in the extract.

    The same Geofabrik Nepal extract the road backfill uses: 617,562 highway ways,
    answering a nearest-road query locally in microseconds instead of ~30 s per site
    against Overpass.
    """
    import pyogrio
    from shapely import STRtree

    gdf = pyogrio.read_dataframe(pbf_path, layer="lines", columns=["highway"],
                                 where="highway IS NOT NULL")
    gdf = gdf[gdf["highway"].notna()]
    geoms = list(gdf.geometry.values)
    classes = [str(x) for x in gdf["highway"].values]
    return STRtree(geoms), geoms, classes


def _attach_road_exposure(cells, road_index, radius_m: float) -> None:
    """Nearest mapped road per cell, by a single vectorised nearest-neighbour query.

    ``STRtree.query_nearest`` over all cells at once replaces a per-cell buffer query;
    on a 100k-cell corridor that is the difference between seconds and hours. The
    returned index is the nearest geometry in DEGREE space, so the metric distance is
    recomputed properly afterwards and cells beyond ``radius_m`` keep ``road_distance_m``
    None rather than a clipped value.
    """
    import numpy as np
    from shapely.geometry import Point

    tree, geoms, classes = road_index
    pts = [Point(c.lon, c.lat) for c in cells]
    try:
        nearest = np.asarray(tree.query_nearest(pts, all_matches=False))
    except Exception as exc:
        logger.warning("road nearest-neighbour query failed: %s", exc)
        return
    # shapely returns either a flat array of tree indices or a (2, n) index pair array
    pairs = zip(nearest[0], nearest[1]) if nearest.ndim == 2 else enumerate(nearest)

    for i, j in pairs:
        cell = cells[int(i)]
        geom = geoms[int(j)]
        pt = pts[int(i)]
        try:
            p2 = geom.interpolate(geom.project(pt))
        except Exception:
            continue
        m_lon = 111_320.0 * max(0.15, math.cos(math.radians(cell.lat)))
        d = math.hypot((p2.x - cell.lon) * m_lon, (p2.y - cell.lat) * 111_320.0)
        if d <= radius_m:
            cell.road_distance_m = round(float(d), 1)
            cell.road_class = classes[int(j)]
            cell.exposure = "ROAD"


GRID_DIR = DATA_DIR / "valley_grid_terrain"

_NUMERIC_FIELDS = ("lat", "lon", "elevation_m", "slope_deg", "aspect_deg",
                   "local_relief_m", "profile_curvature", "road_distance_m",
                   "distance_to_river_m", "fault_distance_km",
                   "settlement_distance_m", "hydro_distance_m",
                   "clay_fraction", "sand_fraction", "coarse_rock_fraction",
                   "dry_bulk_density_g_cm3", "row", "col")
_TEXT_FIELDS = ("road_class", "settlement_name", "settlement_kind",
                "hydro_name", "exposure")


def save_grid(grids: Dict[str, List[TerrainCell]], path: Optional[Path] = None) -> None:
    """Write each corridor as a columnar .npz.

    One JSON file for every corridor was the obvious first choice and the wrong one:
    the cache reached 52 MB with three corridors loaded and would pass 300 MB with all
    seven, so a single request that wanted Langtang parsed Karnali's 198,554 cells to
    get at it - about 4 s of JSON parsing per cold request, and a memory spike well
    past the array data it produced.

    Columnar npz loads the arrays the physics actually wants, per corridor, in
    milliseconds, and stores float columns as float32 - 240 m cells do not carry seven
    significant figures. Absent values stay absent as NaN rather than becoming a number.
    """
    import numpy as np

    base = Path(path) if path else GRID_DIR
    base.mkdir(parents=True, exist_ok=True)
    for cid, cells in grids.items():
        cols: Dict[str, Any] = {}
        for f in _NUMERIC_FIELDS:
            cols[f] = np.array(
                [(getattr(c, f, None) if getattr(c, f, None) is not None else np.nan)
                 for c in cells], dtype=np.float32)
        for f in _TEXT_FIELDS:
            cols[f] = np.array([(getattr(c, f, None) or "") for c in cells], dtype=object)
        # The temp name must itself end in .npz: savez_compressed appends the suffix
        # when the path lacks it, so "<cid>.npz.tmp" is silently written as
        # "<cid>.npz.tmp.npz" and the rename then fails on a missing file.
        tmp = base / f".{cid}.partial.npz"
        np.savez_compressed(tmp, **cols)
        tmp.replace(base / f"{cid}.npz")


def load_corridor(corridor_id: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """Columns for one corridor, or {} when it has not been built.

    Returning {} rather than a generated stand-in is deliberate: a caller that finds no
    grid must say so, not draw one.
    """
    import numpy as np

    base = Path(path) if path else GRID_DIR
    f = base / f"{corridor_id}.npz"
    if not f.exists():
        return {}
    try:
        with np.load(f, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError) as exc:
        logger.warning("grid for %s unreadable: %s", corridor_id, exc)
        return {}


def built_corridors(path: Optional[Path] = None) -> List[str]:
    base = Path(path) if path else GRID_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.npz"))


def corridor_cell_count(corridor_id: str, path: Optional[Path] = None) -> int:
    cols = load_corridor(corridor_id, path)
    return int(cols["lat"].size) if cols else 0


def load_grid(path: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Every built corridor as row dicts.

    Kept for the build script and tests, which do want whole corridors as objects.
    Request paths should call :func:`load_corridor` instead and stay columnar.
    """
    import numpy as np

    out: Dict[str, List[Dict[str, Any]]] = {}
    for cid in built_corridors(path):
        cols = load_corridor(cid, path)
        if not cols:
            continue
        n = int(cols["lat"].size)
        rows = []
        for i in range(n):
            row: Dict[str, Any] = {"corridor_id": cid}
            for f in _NUMERIC_FIELDS:
                v = float(cols[f][i])
                row[f] = None if np.isnan(v) else (int(v) if f in ("row", "col") else v)
            for f in _TEXT_FIELDS:
                v = str(cols[f][i])
                row[f] = v or None
            row["exposure"] = row.get("exposure") or "TERRAIN_ONLY"
            rows.append(row)
        out[cid] = rows
    return out


# --------------------------------------------------------------------------------------
# Real per-cell attributes beyond the DEM
#
# Each attach_* function below fills one group of fields that the FS physics consumes and
# that the previous generator invented. They all share a rule: a cell the source cannot
# resolve keeps None for that field. ``compute_vectorized_grid_fs`` is then responsible
# for refusing or falling back explicitly, instead of being handed a plausible number
# that nothing measured.
# --------------------------------------------------------------------------------------

ISRIC_BASE = "/vsicurl/https://files.isric.org/soilgrids/latest/data"

# Same horizons and conversion the 491-site soil backfill settled on: the 30-200 cm
# column that brackets the failure plane, not the topsoil. Sampling 0-5 cm understated
# bulk density by 20%, which raises FS and suppresses alerts - see backfill_soil_cog.py.
SOIL_DEPTHS = ("30-60cm", "60-100cm", "100-200cm")
DEPTH_BOUNDS = {"30-60cm": (30, 60), "60-100cm": (60, 100), "100-200cm": (100, 200)}
SOIL_PROPERTIES = {
    "clay": ("clay_fraction", 10.0, 0.01),
    "sand": ("sand_fraction", 10.0, 0.01),
    "cfvo": ("coarse_rock_fraction", 10.0, 0.01),
    "bdod": ("dry_bulk_density_g_cm3", 100.0, 1.0),
}
FAILURE_DEPTH_CM = 150.0


def attach_soil(cells: List[TerrainCell], bbox: Tuple[float, float, float, float]) -> int:
    """Real SoilGrids values per cell, by one block read per layer over the corridor.

    The 491-site backfill reads SoilGrids a coordinate at a time, which is right for 491
    scattered points and hopeless for 36,000 contiguous ones. Here the corridor window is
    read once per layer and every cell indexes into that array - 12 reads per corridor
    instead of 12 per cell.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    if not cells:
        return 0
    min_lon, min_lat, max_lon, max_lat = bbox
    lats = np.array([c.lat for c in cells])
    lons = np.array([c.lon for c in cells])

    profiles: Dict[str, Dict[str, Any]] = {}
    for prop, (field, d_factor, scale) in SOIL_PROPERTIES.items():
        for depth in SOIL_DEPTHS:
            url = f"{ISRIC_BASE}/{prop}/{prop}_{depth}_mean.vrt"
            try:
                with rasterio.open(url) as src:
                    xs, ys = warp_transform("EPSG:4326", src.crs,
                                            [min_lon, max_lon, min_lon, max_lon],
                                            [min_lat, min_lat, max_lat, max_lat])
                    win = from_bounds(min(xs), min(ys), max(xs), max(ys),
                                      src.transform).round_offsets().round_lengths()
                    block = src.read(1, window=win, boundless=True,
                                     fill_value=src.nodata if src.nodata is not None else 0)
                    wt = src.window_transform(win)
                    cx, cy = warp_transform("EPSG:4326", src.crs, list(lons), list(lats))
                    inv = ~wt
                    cols, rows = inv * (np.asarray(cx), np.asarray(cy))
                    rows = np.floor(rows).astype(int)
                    cols = np.floor(cols).astype(int)
                    ok = ((rows >= 0) & (rows < block.shape[0])
                          & (cols >= 0) & (cols < block.shape[1]))
                    vals = np.full(len(cells), np.nan)
                    raw = block[np.clip(rows, 0, block.shape[0] - 1),
                                np.clip(cols, 0, block.shape[1] - 1)].astype("float64")
                    if src.nodata is not None:
                        ok &= raw != src.nodata
                    vals[ok] = (raw[ok] / d_factor) * scale
                    profiles.setdefault(field, {})[depth] = vals
            except Exception as exc:
                logger.warning("SoilGrids %s %s unreadable: %s", prop, depth, exc)

    filled = 0
    for field, by_depth in profiles.items():
        num = np.zeros(len(cells))
        den = np.zeros(len(cells))
        for depth, vals in by_depth.items():
            lo, hi = DEPTH_BOUNDS[depth]
            thickness = min(hi, FAILURE_DEPTH_CM) - lo
            if thickness <= 0:
                continue
            good = np.isfinite(vals)
            num[good] += vals[good] * thickness
            den[good] += thickness
        weighted = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        for i, cell in enumerate(cells):
            if np.isfinite(weighted[i]):
                setattr(cell, field, round(float(weighted[i]), 4))
                filled += 1
    return filled


def attach_fault_distance(cells: List[TerrainCell], faults_geojson: str) -> int:
    """Distance to the nearest mapped active fault trace, from the GEM database.

    The field this replaces was ``4.0 + 12.0 * |lat - bbox_centre| / bbox_height`` - a
    function of where the cell sat in an arbitrary bounding box, not of any fault.

    KNOWN LIMITATION OF THE SOURCE, and it biases in the unsafe direction.
    GEM's harmonized database maps 59 traces across the 80-89E / 26-31N window, but they
    are overwhelmingly NORMAL faults at 29-31N - the N-S rifts of southern Tibet, north
    of the Himalayan crest. Only 5 cross central Nepal (84-86E, 27-28N). The Main
    Frontal Thrust, Main Boundary Thrust and MCT, which run E-W through Nepal at roughly
    27-28N and dominate its seismic hazard, are largely absent from this harmonized
    layer.

    So these are honestly "distance to the nearest GEM-mapped active trace", which is
    NOT "distance to the nearest active fault". The gap matters, because the value feeds
    ``fault_integrity = 0.68 + 0.32 * clip(d / 15, 0, 1)`` - the factor that reduces
    ``c_prime`` near a fault. That saturates at 15 km, and measured across all 590,611
    cells:

        corridor                   cells    min km   median    < 15 km   mean integrity
        langtang                  36,279      15.5     40.5          0          1.0000
        budhi_gandaki             57,326      20.5     52.5          0          1.0000
        sun_koshi                 76,295      16.0     61.3          0          1.0000
        sapta_koshi_kanchenjunga 106,923       4.3     50.4      2,557          0.9981
        bagmati_narayani          46,236       0.8     31.0      7,909          0.9856
        karnali                  198,554       0.0     30.3     40,611          0.9709
        kali_gandaki              68,998       0.0     20.7     25,708          0.9422

    87% of cells sit at or past saturation, so for them the fault-damage term contributes
    exactly nothing. It is not uniformly dead - Kali Gandaki has 37% of its cells inside
    15 km and a mean integrity of 0.9422 - but in three corridors it is entirely inert.

    The bias direction is the unsafe one: an overstated distance means less modelled
    damage, higher cohesion, higher FS and fewer alerts. It fails quiet, the same
    direction as the soil-horizon bug fixed in backfill_soil_cog.py.

    A fuller trace set for the Himalayan thrust front (HimaTibetMap / Taylor & Yin, or
    the Nepal-specific active fault compilations) is the real fix and is not done. Until
    then the consumer should treat this as a lower bound on fault proximity rather than
    a measurement of it.
    """
    import json as _json
    import math as _m

    import numpy as np
    from shapely.geometry import LineString, Point, shape
    from shapely import STRtree

    if not cells:
        return 0
    with open(faults_geojson, encoding="utf-8") as fh:
        gj = _json.load(fh)
    geoms = []
    for feat in gj.get("features", []):
        try:
            g = shape(feat["geometry"])
        except Exception:
            continue
        # GEM carries a Z ordinate; drop it so 2-D distance is well defined.
        if g.geom_type == "LineString":
            geoms.append(LineString([(x, y) for x, y, *_ in g.coords]))
        elif g.geom_type == "MultiLineString":
            for part in g.geoms:
                geoms.append(LineString([(x, y) for x, y, *_ in part.coords]))
    if not geoms:
        return 0

    tree = STRtree(geoms)
    pts = [Point(c.lon, c.lat) for c in cells]
    nearest = np.asarray(tree.query_nearest(pts, all_matches=False))
    pairs = zip(nearest[0], nearest[1]) if nearest.ndim == 2 else enumerate(nearest)
    filled = 0
    for i, j in pairs:
        cell = cells[int(i)]
        p2 = geoms[int(j)].interpolate(geoms[int(j)].project(pts[int(i)]))
        m_lon = 111_320.0 * max(0.15, _m.cos(_m.radians(cell.lat)))
        d_m = _m.hypot((p2.x - cell.lon) * m_lon, (p2.y - cell.lat) * 111_320.0)
        cell.fault_distance_km = round(d_m / 1000.0, 3)
        filled += 1
    return filled


def attach_rivers_and_settlements(cells, pbf_path: str, settlement_radius_m: float = 2000.0) -> Dict[str, int]:
    """Distance to the nearest mapped river, and the nearest named settlement.

    Both come from the same Geofabrik extract the roads use. ``distance_to_river_m``
    drives the gorge-undercutting term; the settlement name is what the dashboard shows
    as the exposed asset, replacing the 140 hand-authored place entries with whatever
    OSM actually maps inside the corridor.
    """
    import math as _m

    import numpy as np
    import pyogrio
    from shapely.geometry import Point
    from shapely import STRtree

    out = {"river": 0, "settlement": 0}
    if not cells:
        return out
    pts = [Point(c.lon, c.lat) for c in cells]

    try:
        rivers = pyogrio.read_dataframe(
            pbf_path, layer="lines", columns=["waterway"],
            where="waterway IN ('river','stream')")
        rgeoms = [g for g in rivers.geometry.values if g is not None]
        if rgeoms:
            rtree = STRtree(rgeoms)
            near = np.asarray(rtree.query_nearest(pts, all_matches=False))
            pairs = zip(near[0], near[1]) if near.ndim == 2 else enumerate(near)
            for i, j in pairs:
                cell = cells[int(i)]
                g = rgeoms[int(j)]
                p2 = g.interpolate(g.project(pts[int(i)]))
                m_lon = 111_320.0 * max(0.15, _m.cos(_m.radians(cell.lat)))
                cell.distance_to_river_m = round(_m.hypot(
                    (p2.x - cell.lon) * m_lon, (p2.y - cell.lat) * 111_320.0), 1)
                out["river"] += 1
    except Exception as exc:
        logger.warning("river layer unreadable: %s", exc)

    try:
        places = pyogrio.read_dataframe(
            pbf_path, layer="points", columns=["name", "place"],
            where="place IN ('city','town','village','hamlet') AND name IS NOT NULL")
        pgeoms = [g for g in places.geometry.values if g is not None]
        pnames = [str(n) for n in places["name"].values]
        pkinds = [str(k) for k in places["place"].values]
        if pgeoms:
            ptree = STRtree(pgeoms)
            near = np.asarray(ptree.query_nearest(pts, all_matches=False))
            pairs = zip(near[0], near[1]) if near.ndim == 2 else enumerate(near)
            for i, j in pairs:
                cell = cells[int(i)]
                g = pgeoms[int(j)]
                m_lon = 111_320.0 * max(0.15, _m.cos(_m.radians(cell.lat)))
                d = _m.hypot((g.x - cell.lon) * m_lon, (g.y - cell.lat) * 111_320.0)
                cell.settlement_distance_m = round(d, 1)
                cell.settlement_name = pnames[int(j)]
                cell.settlement_kind = pkinds[int(j)]
                if d <= settlement_radius_m and (
                        cell.road_distance_m is None or d < cell.road_distance_m):
                    cell.exposure = "SETTLEMENT"
                out["settlement"] += 1
    except Exception as exc:
        logger.warning("places layer unreadable: %s", exc)
    return out


def attach_hydro_infrastructure(cells, pbf_path: str, radius_m: float = 1500.0) -> int:
    """Distance to the nearest mapped dam, weir or hydro plant.

    Hydro assets are the highest-consequence structures in these corridors - the
    corridor metadata lists Upper Tamakoshi (456 MW), Arun-3 (900 MW), Kali Gandaki A
    (144 MW) - and a slope failing above a headrace or a powerhouse threatens both the
    structure and everyone downstream of it. The previous grid assigned 16% of its cells
    to "HYDRO_INFRASTRUCTURE" by quota; this measures where the structures actually are.

    OSM maps them across two layers, so both are read: 337 ``waterway=dam|weir`` ways
    and 26 ``power=plant`` / ``landuse=reservoir`` multipolygons in the Nepal extract,
    carrying real names such as "Kaligandaki A Hydropower Plant".
    """
    import math as _m

    import numpy as np
    import pyogrio
    from shapely.geometry import Point
    from shapely import STRtree

    if not cells:
        return 0
    geoms, names = [], []
    for layer, where, namecol in (
        ("lines", "waterway='dam' OR waterway='weir'", "name"),
        ("multipolygons",
         "other_tags LIKE '%\"power\"=>\"plant\"%' OR landuse='reservoir'", "name"),
    ):
        try:
            df = pyogrio.read_dataframe(pbf_path, layer=layer,
                                        columns=[namecol], where=where)
        except Exception as exc:
            logger.warning("hydro layer %s unreadable: %s", layer, exc)
            continue
        for g, n in zip(df.geometry.values, df[namecol].values):
            if g is None:
                continue
            geoms.append(g)
            # `n is not None` is True for a pandas missing value, and str(nan) is the
            # string "nan" - which then reached the dashboard as a structure's name.
            # Test the value, not the identity.
            missing = n is None or (isinstance(n, float) and n != n) or str(n) == "nan"
            names.append("unnamed hydro structure" if missing else str(n))
    if not geoms:
        return 0

    tree = STRtree(geoms)
    pts = [Point(c.lon, c.lat) for c in cells]
    near = np.asarray(tree.query_nearest(pts, all_matches=False))
    pairs = zip(near[0], near[1]) if near.ndim == 2 else enumerate(near)
    filled = 0
    for i, j in pairs:
        cell = cells[int(i)]
        g = geoms[int(j)]
        pt = pts[int(i)]
        try:
            p2 = g.interpolate(g.project(pt)) if g.geom_type in (
                "LineString", "MultiLineString") else g.exterior.interpolate(
                    g.exterior.project(pt))
        except Exception:
            p2 = g.centroid
        m_lon = 111_320.0 * max(0.15, _m.cos(_m.radians(cell.lat)))
        d = _m.hypot((p2.x - cell.lon) * m_lon, (p2.y - cell.lat) * 111_320.0)
        if d <= radius_m:
            cell.hydro_distance_m = round(float(d), 1)
            cell.hydro_name = names[int(j)]
            filled += 1
    return filled
