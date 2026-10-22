"""Unified Sites of Interest Catalog across Nepal.

Assembles and manages the 472 NASA Global Landslide Catalog (COOLR) events,
24 high-mountain cryosphere avalanches/GLOFs/anchors, and 10 live station hotspots
(506 total unique sites of interest) for perpetual monitoring and calculation.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CATALOG_CACHE_PATH = DATA_DIR / "sites_of_interest_catalog.json"


@dataclass
class SiteInfo:
    site_id: str
    name: str
    hazard_type: str  # "rainfall_landslide", "rock_ice_avalanche", "glof", "monitoring_beacon"
    latitude: float
    longitude: float
    elevation_m: float
    slope_deg: float
    profile_curvature: float
    local_relief_m: float
    fault_distance_km: float
    basin_id: str


def determine_basin(lon: float, lat: float) -> str:
    """Classify site into primary Himalayan river watershed corridor."""
    if 85.15 <= lon <= 85.80 and 28.05 <= lat <= 28.45:
        return "langtang"
    elif 84.50 <= lon <= 85.10 and 28.10 <= lat <= 28.75:
        return "budhi_gandaki"
    elif lon < 82.5:
        return "karnali"
    elif lon < 84.5:
        return "kali_gandaki"
    elif lon < 86.0:
        return "bagmati_narayani"
    elif lon < 87.5:
        return "sun_koshi"
    else:
        return "sapta_koshi_kanchenjunga"


def load_sites_of_interest(force_rebuild: bool = False) -> List[SiteInfo]:
    """Load the unified catalog of all 479+ sites of interest across Nepal."""
    if CATALOG_CACHE_PATH.exists() and not force_rebuild:
        try:
            with open(CATALOG_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [SiteInfo(**item) for item in data]
        except Exception:
            pass

    sites: List[SiteInfo] = []
    seen_coords = set()

    # 1. 10 Primary Monitored Observation Stations
    STATIONS_SEED = [
        ("langtang_lirung", "Langtang Lirung & Kyangjin Radar", "rock_ice_avalanche", 28.232, 85.518, 5680.0, 46.0, -0.62, 640.0, 3.8, "langtang"),
        ("shalbachum_lake", "Shalbachum Glacial Lake (Rasuwa)", "glof", 28.258, 85.582, 5380.0, 43.0, -0.48, 480.0, 5.2, "langtang"),
        ("melamchi_pemdan", "Melamchi / Pemdan Moraine Gorge", "rainfall_landslide", 28.034, 85.562, 4580.0, 39.0, -0.75, 420.0, 6.4, "langtang"),
        ("birendra_tal", "Birendra Tal & Manaslu Icefall", "glof", 28.596, 84.636, 4620.0, 39.0, -0.58, 510.0, 7.2, "budhi_gandaki"),
        ("thulagi_dona", "Thulagi (Dona) Glacial Lake", "glof", 28.538, 84.485, 5120.0, 45.0, -0.52, 490.0, 9.1, "budhi_gandaki"),
        ("philim_gorge", "Budhi Gandaki Philim Gorge Choke Point", "rainfall_landslide", 28.315, 84.725, 2650.0, 41.0, -0.65, 380.0, 4.5, "budhi_gandaki"),
        ("seti_annapurna", "Seti Gorge / Annapurna IV Sabche Cirque", "rock_ice_avalanche", 28.612, 84.085, 5420.0, 48.0, -0.68, 720.0, 8.5, "budhi_gandaki"),
        ("tsho_rolpa", "Tsho Rolpa Glacial Lake (Rolwaling)", "glof", 27.862, 86.476, 5190.0, 44.0, -0.55, 460.0, 12.4, "langtang"),
        ("imja_dig_tsho", "Imja Tsho & Dig Tsho (Everest)", "glof", 27.901, 86.936, 5540.0, 47.0, -0.59, 580.0, 14.2, "langtang"),
        ("lower_barun", "Lower Barun Glacial Lake (Sankhuwasabha)", "glof", 27.798, 87.092, 4890.0, 44.0, -0.60, 450.0, 10.5, "langtang"),
    ]
    for sid, name, htype, lat, lon, elev, slp, curv, rel, flt, basin in STATIONS_SEED:
        coord_key = (round(lat, 3), round(lon, 3))
        seen_coords.add(coord_key)
        sites.append(SiteInfo(
            site_id=sid,
            name=name,
            hazard_type=htype,
            latitude=lat,
            longitude=lon,
            elevation_m=elev,
            slope_deg=slp,
            profile_curvature=curv,
            local_relief_m=rel,
            fault_distance_km=flt,
            basin_id=basin,
        ))

    # 2. 24 High-Mountain Cryosphere Avalanches & GLOFs
    cryo_path = PROCESSED_DIR / "himalayan_cryosphere_avalanches_glofs.csv"
    if cryo_path.exists():
        cryo_df = pd.read_csv(cryo_path)
        for _, row in cryo_df.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            coord_key = (round(lat, 3), round(lon, 3))
            if coord_key in seen_coords:
                continue
            seen_coords.add(coord_key)
            name = str(row.get("event_name", "Cryosphere Site")).replace("_", " ")
            htype = str(row.get("hazard_type", "rock_ice_avalanche"))
            sites.append(SiteInfo(
                site_id=f"cryo_{len(sites)+1:03d}",
                name=name,
                hazard_type=htype,
                latitude=lat,
                longitude=lon,
                elevation_m=float(row["elevation_m"]),
                slope_deg=float(row["slope_deg"]),
                profile_curvature=float(row["profile_curvature"]),
                local_relief_m=float(row.get("local_relief_m", 350.0)),
                fault_distance_km=float(row.get("fault_distance_km", 5.0)),
                basin_id=determine_basin(lon, lat),
            ))

    # 3. 472 Verified NASA Global Landslide Catalog (COOLR) Nepal Points
    training_points_path = PROCESSED_DIR / "nepal_training_points.csv"
    if training_points_path.exists():
        tp_df = pd.read_csv(training_points_path)
        landslides = tp_df[tp_df["label"] == 1]
        for idx, row in landslides.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            coord_key = (round(lat, 3), round(lon, 3))
            if coord_key in seen_coords:
                continue
            seen_coords.add(coord_key)
            elev = float(row["elevation_m"])
            slp = float(row["slope_deg"])
            curv = float(row["profile_curvature"])
            rel = float(row["local_relief_m"])
            flt = float(row["fault_distance_km"])
            basin = determine_basin(lon, lat)
            sites.append(SiteInfo(
                site_id=f"glc_{len(sites)+1:03d}",
                name=f"Nepal Landslide Site #{len(sites)+1} ({basin.title()} Corridor)",
                hazard_type="rainfall_landslide",
                latitude=lat,
                longitude=lon,
                elevation_m=round(elev, 1),
                slope_deg=round(slp, 1),
                profile_curvature=round(curv, 3),
                local_relief_m=round(rel, 1),
                fault_distance_km=round(flt, 2),
                basin_id=basin,
            ))

    # Persist catalog cache
    try:
        CATALOG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CATALOG_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in sites], f, indent=2)
    except Exception as e:
        print(f"Warning: could not write catalog cache: {e}")

    return sites


if __name__ == "__main__":
    sites = load_sites_of_interest(force_rebuild=True)
    print(f"Loaded {len(sites)} unified sites of interest across Nepal:")
    df = pd.DataFrame([asdict(s) for s in sites])
    print(df["hazard_type"].value_counts())
    print("\nBy Basin Corridor:")
    print(df["basin_id"].value_counts())
