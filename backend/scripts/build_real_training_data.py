"""Build a real-data training set for Tier A landslide susceptibility (Nepal).

Data sources (all open):
  - Labels: NASA Global Landslide Catalog (COOLR), Nepal subset.
  - Terrain: Copernicus DEM GLO-30, read directly over HTTP via GDAL /vsicurl/
    (no bulk download — small windowed reads per point).
  - Faults: GEM Global Active Faults Database (GitHub, CC-BY-SA 4.0).

Produces backend/data/processed/nepal_training_points.csv with real
slope/elevation/curvature/relief/fault-distance features + landslide labels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "backend" / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

DEM_TEMPLATE = (
    "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif"
)
FAULTS_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/"
    "gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson"
)

NEPAL_BBOX = (80.0, 26.3, 88.3, 30.5)  # lon_min, lat_min, lon_max, lat_max
WINDOW_PX = 11  # ~330m x 330m at 30m resolution
PIXEL_M = 30.0
MIN_ABSENCE_DIST_DEG = 0.02  # ~2.2km buffer around known slides for pseudo-absences
RNG = np.random.default_rng(42)


def load_nepal_landslides() -> pd.DataFrame:
    df = pd.read_csv(RAW / "nasa_glc_export.csv", low_memory=False)
    df = df.dropna(subset=["latitude", "longitude"])
    nepal = df[df["country_name"] == "Nepal"].copy()
    nepal = nepal[
        (nepal["longitude"] >= NEPAL_BBOX[0])
        & (nepal["longitude"] <= NEPAL_BBOX[2])
        & (nepal["latitude"] >= NEPAL_BBOX[1])
        & (nepal["latitude"] <= NEPAL_BBOX[3])
    ]
    nepal = nepal.drop_duplicates(subset=["latitude", "longitude"])
    return nepal[["latitude", "longitude", "landslide_trigger", "event_date"]]


def generate_pseudo_absences(positives: pd.DataFrame, n: int) -> pd.DataFrame:
    """Random points inside the Nepal bbox, kept away from any known slide."""
    pos_pts = positives[["latitude", "longitude"]].to_numpy()
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        lon = RNG.uniform(NEPAL_BBOX[0], NEPAL_BBOX[2])
        lat = RNG.uniform(NEPAL_BBOX[1], NEPAL_BBOX[3])
        d = np.sqrt((pos_pts[:, 0] - lat) ** 2 + (pos_pts[:, 1] - lon) ** 2)
        if d.min() >= MIN_ABSENCE_DIST_DEG:
            out.append((lat, lon))
    return pd.DataFrame(out, columns=["latitude", "longitude"])


def open_dem_tiles(lat_lons: list[tuple[float, float]]) -> dict[tuple[int, int], rasterio.DatasetReader]:
    tiles: dict[tuple[int, int], rasterio.DatasetReader] = {}
    for lat, lon in lat_lons:
        key = (math.floor(lat), math.floor(lon))
        if key in tiles:
            continue
        url = DEM_TEMPLATE.format(lat=key[0], lon=key[1])
        try:
            tiles[key] = rasterio.open(url)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  WARN: could not open tile {key}: {exc}")
    return tiles


def sample_terrain(lat: float, lon: float, tiles: dict) -> dict | None:
    key = (math.floor(lat), math.floor(lon))
    src = tiles.get(key)
    if src is None:
        return None
    row, col = src.index(lon, lat)
    half = WINDOW_PX // 2
    if row - half < 0 or col - half < 0 or row + half >= src.height or col + half >= src.width:
        return None
    window = Window(col - half, row - half, WINDOW_PX, WINDOW_PX)
    arr = src.read(1, window=window).astype(np.float64)
    if np.any(arr <= -9000) or np.isnan(arr).any():
        return None

    gy, gx = np.gradient(arr, PIXEL_M)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = float(np.degrees(slope_rad[half, half]))

    gyy, _ = np.gradient(gy, PIXEL_M)
    _, gxx = np.gradient(gx, PIXEL_M)
    profile_curvature = float((gxx[half, half] + gyy[half, half]) * 100.0)

    elevation_m = float(arr[half, half])
    local_relief_m = float(arr.max() - arr.min())
    return {
        "elevation_m": elevation_m,
        "slope_deg": slope_deg,
        "profile_curvature": profile_curvature,
        "local_relief_m": local_relief_m,
    }


def fault_distances_km(points: gpd.GeoSeries) -> np.ndarray:
    print("Loading GEM Global Active Faults (filtered to Nepal bbox)...")
    faults = gpd.read_file(FAULTS_URL)
    minx, miny, maxx, maxy = NEPAL_BBOX
    pad = 2.0
    faults = faults.cx[minx - pad : maxx + pad, miny - pad : maxy + pad]
    if faults.empty:
        print("  WARN: no faults found in bbox, falling back to NaN")
        return np.full(len(points), np.nan)
    faults_m = faults.to_crs(32645)  # UTM 45N, covers Nepal
    points_m = points.to_crs(32645)
    dists = np.array([faults_m.distance(p).min() / 1000.0 for p in points_m.geometry])
    return dists


def main() -> None:
    print("Loading Nepal landslide points from NASA GLC...")
    positives = load_nepal_landslides()
    positives["label"] = 1
    print(f"  {len(positives)} positive points")

    print("Generating pseudo-absence points...")
    absences = generate_pseudo_absences(positives, n=len(positives))
    absences["label"] = 0
    print(f"  {len(absences)} pseudo-absence points")

    all_points = pd.concat(
        [positives[["latitude", "longitude", "label"]], absences[["latitude", "longitude", "label"]]],
        ignore_index=True,
    )

    print("Opening Copernicus DEM COG tiles over HTTP (windowed reads, no bulk download)...")
    tiles = open_dem_tiles(list(zip(all_points["latitude"], all_points["longitude"])))
    print(f"  {len(tiles)} unique 1-deg tiles opened")

    print("Sampling terrain features per point...")
    records = []
    for i, row in all_points.iterrows():
        feats = sample_terrain(row["latitude"], row["longitude"], tiles)
        if feats is None:
            continue
        feats.update(
            latitude=row["latitude"],
            longitude=row["longitude"],
            label=int(row["label"]),
        )
        records.append(feats)
        if len(records) % 100 == 0:
            print(f"    {len(records)} points sampled...")
    terrain_df = pd.DataFrame(records)
    print(f"  {len(terrain_df)} points with valid terrain features")

    gdf = gpd.GeoDataFrame(
        terrain_df,
        geometry=[Point(xy) for xy in zip(terrain_df["longitude"], terrain_df["latitude"])],
        crs="EPSG:4326",
    )
    terrain_df["fault_distance_km"] = fault_distances_km(gdf)

    out_path = PROCESSED / "nepal_training_points.csv"
    terrain_df.to_csv(out_path, index=False)
    print(f"Saved {len(terrain_df)} labeled points -> {out_path}")
    print(terrain_df["label"].value_counts())


if __name__ == "__main__":
    main()
