"""Vectorized Himalayan Valley-Corridor Physics & Factor of Safety (FS) Engine.

Dissolves discrete site point calculations into a contiguous 240m valley grid across
Nepal's major river corridors, hillside human settlements, road networks, and active fault zones.
Directly aligns with NISAR L-band GUNW 80m posting (exactly 3x3 radar pixels per cell).

Physical Formulations:
  1. Road-cut excavation geometry & slope steepening:
       R_cut = clip(exp(-d_road / 65.0) * sin(beta), 0.0, 0.95)
       beta_eff = beta + 4.5 * R_cut
  2. Multi-sensor soil/regolith slab depth z:
       z_base = 0.40 + 3.10 * exp(-0.042 * beta)
       z = clip(z_base * curvature_mult * aspect_insolation_mult, 0.35, 4.5)
  3. Effective cohesion with fault damage & road-cut reduction:
       c_mineral = 3.5 + 14.0 * clay + 4.0 * coarse_rock
       c'_eff = c_mineral * (1.0 - 0.30 * R_cut) * fault_integrity_factor
       (+ permafrost ice-bonding above 4500m)
  4. Perched water table zw from 72h precipitation + 30d antecedent API sponge:
       porosity = 1.0 - bulk_density / 2.65
       delta_theta = max(porosity * (1.0 - S_ante), 0.04)
       z_w = clip(z * S_ante * 0.50 + (rain_72h / 1000.0) / delta_theta, 0.0, z)
  5. Generalized Infinite Slope Factor of Safety (FS) & Continuous Gradient:
       FS = (c'_eff + (gamma*z*cos^2(b) - gamma*z*kh*sin(b)*cos(b) - gamma_w*zw*cos^2(b))*tan(phi')) /
            (gamma*z * (sin(b)*cos(b) + kh*cos^2(b)))
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GRID_CACHE_DIR = DATA_DIR / "valley_grid"


def compute_vectorized_grid_fs(
    slope_deg: np.ndarray,
    elevation_m: np.ndarray,
    aspect_deg: Optional[np.ndarray] = None,
    profile_curvature: Optional[np.ndarray] = None,
    road_distance_m: Optional[np.ndarray] = None,
    fault_distance_km: Optional[np.ndarray] = None,
    clay_fraction: Optional[np.ndarray] = None,
    sand_fraction: Optional[np.ndarray] = None,
    coarse_rock_fraction: Optional[np.ndarray] = None,
    dry_bulk_density_g_cm3: Optional[np.ndarray] = None,
    rainfall_72h_mm: float = 0.0,
    api_30d_mm: float = 25.0,
    s_ante: Optional[float] = None,
    pdd_snowmelt_mm: float = 0.0,
    seismic_kh: float = 0.0,
    dsgsd_indicated: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute vectorized Generalized Infinite Slope FS and physical state across grid cells.

    Inputs can be 1D NumPy arrays of matching length. Returns parallel NumPy arrays
    suitable for high-throughput raster or vector map styling.
    """
    slp = np.asarray(slope_deg, dtype=np.float64)
    elev = np.asarray(elevation_m, dtype=np.float64)
    n = slp.size

    # Defaults for optional spatial features if absent (imputed physically, not guessed)
    asp = np.asarray(aspect_deg, dtype=np.float64) if aspect_deg is not None else np.full(n, 180.0)
    curv = np.asarray(profile_curvature, dtype=np.float64) if profile_curvature is not None else np.zeros(n)
    d_road = np.asarray(road_distance_m, dtype=np.float64) if road_distance_m is not None else np.full(n, 2000.0)
    d_fault = np.asarray(fault_distance_km, dtype=np.float64) if fault_distance_km is not None else np.full(n, 12.0)

    # Soil characteristics (default to typical Himalayan weathered regolith: 18% clay, 48% sand, 28% rock, 1.45 g/cm3)
    clay = np.asarray(clay_fraction, dtype=np.float64) if clay_fraction is not None else np.full(n, 0.18)
    sand = np.asarray(sand_fraction, dtype=np.float64) if sand_fraction is not None else np.full(n, 0.48)
    coarse = np.asarray(coarse_rock_fraction, dtype=np.float64) if coarse_rock_fraction is not None else np.full(n, 0.28)
    bdod = np.asarray(dry_bulk_density_g_cm3, dtype=np.float64) if dry_bulk_density_g_cm3 is not None else np.full(n, 1.45)

    # Clean NaNs in inputs to stable defaults
    clay = np.where(np.isnan(clay), 0.18, clay)
    sand = np.where(np.isnan(sand), 0.48, sand)
    coarse = np.where(np.isnan(coarse), 0.28, coarse)
    bdod = np.where(np.isnan(bdod), 1.45, bdod)
    d_road = np.where(np.isnan(d_road), 2000.0, d_road)
    d_fault = np.where(np.isnan(d_fault), 12.0, d_fault)
    curv = np.where(np.isnan(curv), 0.0, curv)
    asp = np.where(np.isnan(asp), 180.0, asp)

    # 1. Road-cut undercut destabilization
    # High steepening effect within 65m of cut on steep slopes
    rad_slp = np.radians(np.clip(slp, 0.0, 85.0))
    r_cut = np.clip(np.exp(-np.maximum(d_road, 0.0) / 65.0) * np.sin(rad_slp), 0.0, 0.95)
    beta_eff_deg = slp + 4.5 * r_cut
    rad_beta_eff = np.radians(np.clip(beta_eff_deg, 1.0, 88.5))

    # 2. Multi-sensor regolith slab depth z (m)
    # Steep slopes shed colluvium; hollows accumulate colluvium; south aspects weather deeper
    z_base = 0.40 + 3.10 * np.exp(-0.042 * slp)
    curv_factor = 1.0 - 0.20 * np.clip(curv, -1.5, 1.5)
    southness = 0.5 * (1.0 - np.cos(np.radians(asp)))
    aspect_factor = 0.88 + 0.24 * southness
    z = np.clip(z_base * curv_factor * aspect_factor, 0.35, 4.5)

    # 3. Effective cohesion c' (kPa)
    c_mineral = 3.5 + 14.0 * clay + 4.0 * coarse
    # Fault integrity factor: damaged within 15 km of major thrust faults
    fault_integrity = 0.68 + 0.32 * np.clip(d_fault / 15.0, 0.0, 1.0)
    c_eff = c_mineral * (1.0 - 0.30 * r_cut) * fault_integrity

    # High-altitude permafrost ice-bonding (>4500m)
    permafrost_ice = np.maximum(0.0, 14.0 * (elev - 4500.0) / 1000.0)
    c_eff = c_eff + permafrost_ice

    # DSGSD rock-mass degradation modifier (if slope shows deep-seated creeping deformation)
    if dsgsd_indicated is not None:
        dsgsd_mask = np.asarray(dsgsd_indicated, dtype=bool)
        c_eff = np.where(dsgsd_mask, c_eff * 0.65, c_eff)

    # 4. Effective internal friction angle phi' (degrees)
    phi_prime_deg = np.clip(28.0 + 8.5 * sand + 6.0 * coarse - 3.0 * clay, 24.0, 44.0)
    if dsgsd_indicated is not None:
        phi_prime_deg = np.where(dsgsd_mask, phi_prime_deg - 3.0, phi_prime_deg)
    rad_phi = np.radians(phi_prime_deg)

    # 5. Hydrological state & perched water table zw
    # Soil porosity from dry bulk density (assuming particle density 2.65 g/cm3)
    porosity = np.clip(1.0 - bdod / 2.65, 0.25, 0.60)
    
    # Antecedent saturation degree S_ante: combines 30d API sponge memory
    if s_ante is None:
        s_ante_val = float(np.clip(0.12 + 0.80 * (1.0 - math.exp(-max(api_30d_mm, 0.0) / 110.0)), 0.12, 0.95))
    else:
        s_ante_val = float(np.clip(s_ante, 0.05, 0.98))

    delta_theta_avail = np.maximum(porosity * (1.0 - s_ante_val), 0.04)
    total_water_inflow_m = (max(rainfall_72h_mm, 0.0) + max(pdd_snowmelt_mm, 0.0)) / 1000.0
    zw = np.clip(z * s_ante_val * 0.50 + total_water_inflow_m / delta_theta_avail, 0.0, z)

    # 6. Infinite Slope Stability Factor of Safety (FS)
    gamma_soil = 17.5 + 2.5 * np.clip(bdod / 1.5, 0.7, 1.3)  # kN/m3
    gamma_water = 9.81  # kN/m3

    sin_b = np.sin(rad_beta_eff)
    cos_b = np.cos(rad_beta_eff)
    cos2_b = cos_b * cos_b
    tan_phi = np.tan(rad_phi)
    kh = max(float(seismic_kh), 0.0)

    # Generalized infinite slope equation with pseudo-static seismic kh:
    # Numerator: resisting shear strength
    numerator = c_eff + (gamma_soil * z * cos2_b - gamma_soil * z * kh * sin_b * cos_b - gamma_water * zw * cos2_b) * tan_phi
    # Denominator: driving shear stress
    denominator = gamma_soil * z * (sin_b * cos_b + kh * cos2_b)

    fs = np.maximum(numerator, 0.01) / np.maximum(denominator, 0.01)
    fs = np.round(fs, 4)

    # 7. Probabilistic failure risk %
    fail_prob = 1.0 / (1.0 + np.exp(6.8 * (fs - 1.0))) * 100.0
    fail_prob = np.round(fail_prob, 2)

    return {
        "fs": fs,
        "soil_depth_z_m": np.round(z, 2),
        "water_table_zw_m": np.round(zw, 2),
        "saturation_ratio": np.round(zw / np.maximum(z, 0.01), 3),
        "effective_slope_deg": np.round(beta_eff_deg, 1),
        "road_cut_penalty": np.round(r_cut, 3),
        "effective_cohesion_kpa": np.round(c_eff, 2),
        "friction_angle_deg": np.round(phi_prime_deg, 1),
        "failure_prob_pct": fail_prob,
    }


def fs_to_gradient_color(fs: float) -> str:
    """Map Factor of Safety (FS) smoothly to a hex color gradient.

    Scale:
      FS <= 0.80 -> #dc2626 (Deep Crimson Red - Critical Collapse)
      FS == 1.00 -> #ef4444 (Bright Red - Limit Equilibrium Failure)
      FS == 1.08 -> #f97316 (Warning Orange - Severe Shear Strain)
      FS == 1.18 -> #eab308 (Watch Yellow - Creeping Margin)
      FS == 1.25 -> #84cc16 (Lime Green - Quasi-Stable)
      FS >= 1.50 -> #10b981 (Emerald Green - High Equilibrium)
    """
    if math.isnan(fs):
        return "#6b7280"  # gray
    if fs <= 0.80:
        return "#dc2626"
    if fs < 1.00:
        t = (fs - 0.80) / 0.20
        # interpolate between #dc2626 (220, 38, 38) and #ef4444 (239, 68, 68)
        r = int(220 + (239 - 220) * t)
        g = int(38 + (68 - 38) * t)
        b = int(38 + (68 - 38) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    if fs < 1.08:
        t = (fs - 1.00) / 0.08
        # interpolate between #ef4444 (239, 68, 68) and #f97316 (249, 115, 22)
        r = int(239 + (249 - 239) * t)
        g = int(68 + (115 - 68) * t)
        b = int(68 + (22 - 68) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    if fs < 1.18:
        t = (fs - 1.08) / 0.10
        # interpolate between #f97316 (249, 115, 22) and #eab308 (234, 179, 8)
        r = int(249 + (234 - 249) * t)
        g = int(115 + (179 - 115) * t)
        b = int(22 + (8 - 22) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    if fs < 1.25:
        t = (fs - 1.18) / 0.07
        # interpolate between #eab308 (234, 179, 8) and #84cc16 (132, 204, 22)
        r = int(234 + (132 - 234) * t)
        g = int(179 + (204 - 179) * t)
        b = int(8 + (22 - 8) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    if fs < 1.50:
        t = (fs - 1.25) / 0.25
        # interpolate between #84cc16 (132, 204, 22) and #10b981 (16, 185, 129)
        r = int(132 + (16 - 132) * t)
        g = int(204 + (185 - 204) * t)
        b = int(22 + (129 - 22) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#10b981"


def classify_alert_level(fs: float, fail_prob_pct: float) -> str:
    """Classify alert level based on FS and failure probability."""
    if fs < 1.0 or fail_prob_pct >= 50.0:
        return "RED"
    if fs < 1.08 or fail_prob_pct >= 35.0:
        return "ORANGE"
    if fs < 1.25 or fail_prob_pct >= 15.0:
        return "YELLOW"
    return "GREEN"


@dataclass
class CorridorMetadata:
    id: str
    name: str
    district: str
    bbox: Tuple[float, float, float, float]  # [min_lon, min_lat, max_lon, max_lat]
    center: Tuple[float, float]
    cell_count: int
    river_length_km: float
    highways: List[str] = field(default_factory=list)
    key_settlements: List[str] = field(default_factory=list)
    infrastructure_assets: List[str] = field(default_factory=list)


PRIMARY_CORRIDORS: Dict[str, CorridorMetadata] = {
    "langtang": CorridorMetadata(
        id="langtang",
        name="Trishuli & Pasang Lhamu Highway Corridor (Dhunche, Syabrubesi, Hydro)",
        district="Rasuwa",
        bbox=(85.15, 28.05, 85.80, 28.45),
        center=(85.515, 28.215),
        cell_count=320,
        river_length_km=42.5,
        highways=["Pasang Lhamu Highway (NH09)", "Chilime Access Road", "Syabrubesi-Timure Freight Road"],
        key_settlements=["Dhunche District HQ", "Syabrubesi Bazaars", "Kyangjin Gompa", "Gatlang", "Thuman", "Timure", "Langtang Village"],
        infrastructure_assets=["Upper Trishuli-1 Hydro Dam (216MW)", "Chilime Hydro (22MW)", "Upper Trishuli-3A (60MW)", "Syabrubesi Strategic Motor Bridge"],
    ),
    "budhi_gandaki": CorridorMetadata(
        id="budhi_gandaki",
        name="Budhi Gandaki & Barpak Corridor (Manaslu, Barpak, Philim, Arughat)",
        district="Gorkha",
        bbox=(84.50, 28.10, 85.10, 28.75),
        center=(84.630, 28.540),
        cell_count=370,
        river_length_km=56.0,
        highways=["Gorkha-Barpak Road", "Benighat-Arughat-Larkya La Road", "Arughat-Soti Khola Road"],
        key_settlements=["Barpak (2015 Epicenter)", "Laprak Hillside Town", "Samagaun (Manaslu Base)", "Lho Village", "Namrung", "Philim", "Arughat Bazar"],
        infrastructure_assets=["Budhi Gandaki Mega Dam Site (1200MW)", "Philim Suspension Bridge", "Barpak Health Complex", "Manaslu Micro-Hydro"],
    ),
    "kali_gandaki": CorridorMetadata(
        id="kali_gandaki",
        name="Kali Gandaki & Siddhartha Highway Corridor (Jomsom, Marpha, Tatopani, Beni)",
        district="Mustang / Myagdi / Baglung",
        bbox=(83.45, 28.15, 84.10, 29.05),
        center=(83.750, 28.600),
        cell_count=550,
        river_length_km=85.2,
        highways=["Kali Gandaki Highway (NH10 / Korala Border)", "Beni-Jomsom Highway", "Pokhara-Baglung Highway (Mid-Hill)"],
        key_settlements=["Jomsom Airport Town", "Marpha Heritage Village", "Tatopani Hot Springs", "Beni Bazar HQ", "Dana Village", "Ghasa", "Kagbeni"],
        infrastructure_assets=["Kali Gandaki A Hydro (144MW)", "Rahughat Hydro (40MW)", "Tatopani Suspension Crossing", "Jomsom Mountain Runway"],
    ),
    "karnali": CorridorMetadata(
        id="karnali",
        name="Karnali Highway & Bheri Corridor (Kalikot, Manma, Dailekh, Surkhet)",
        district="Humla / Kalikot / Dailekh / Jumla",
        bbox=(81.45, 28.75, 82.80, 29.85),
        center=(82.150, 29.300),
        cell_count=510,
        river_length_km=94.5,
        highways=["Karnali Highway (H13 - Jumla Lifeline)", "Mid-Hill Highway (Pushpalal Lokmarg Dailekh)", "Nagma-Gamgadhi Road"],
        key_settlements=["Manma (Kalikot HQ)", "Jumla Khalanga", "Dailekh Bazar", "Raskot Hill Town", "Nagma Confluence Bazar", "Pili Village"],
        infrastructure_assets=["Upper Karnali Hydro Dam Site (900MW)", "Tila River Hydro", "Chisapani Cable Bridge", "Kalikot District Hospital"],
    ),
    "bagmati_narayani": CorridorMetadata(
        id="bagmati_narayani",
        name="Prithvi Highway & Bagmati Corridor (Mugling, Malekhu, Naubise, Kulekhani)",
        district="Kathmandu / Makwanpur / Chitwan / Dhading",
        bbox=(84.45, 27.40, 85.55, 28.05),
        center=(85.050, 27.750),
        cell_count=410,
        river_length_km=68.0,
        highways=["Prithvi Highway (H04 - National Arterial Lifeline)", "Tribhuvan Highway (H02)", "Narayanghat-Mugling Gorge Highway (H05)"],
        key_settlements=["Mugling Commercial Junction", "Malekhu Bazar", "Gajuri Hill Town", "Naubise Gateway", "Daman Plateau", "Bhimphedi"],
        infrastructure_assets=["Kulekhani I Reservoir Dam (60MW)", "Kulekhani II & III Plants", "Mugling Motor Bridge", "Manakamana Cable Car Pylons"],
    ),
    "sun_koshi": CorridorMetadata(
        id="sun_koshi",
        name="Araniko Highway & Sun Koshi Corridor (Bahrabise, Tatopani, Upper Tamakoshi)",
        district="Sindhupalchok / Dolakha",
        bbox=(85.55, 27.45, 86.45, 28.15),
        center=(86.000, 27.800),
        cell_count=360,
        river_length_km=58.4,
        highways=["Araniko Highway (H03 - China Trade Arterial)", "Pushpalal Mid-Hill Highway Charikot Section", "BP Highway (H06)"],
        key_settlements=["Bahrabise Commercial Town", "Tatopani Border Bazar", "Charikot District HQ", "Jiri Mountain Gateway", "Dolalghat", "Chautara"],
        infrastructure_assets=["Upper Tamakoshi Hydro (456MW Mega Project)", "Bhotekoshi Hydro (45MW)", "Sun Koshi Hydro (10MW)", "Friendship Bridge Kodari"],
    ),
    "sapta_koshi_kanchenjunga": CorridorMetadata(
        id="sapta_koshi_kanchenjunga",
        name="Arun-3 & Kanchenjunga Corridor (Khandbari, Num, Taplejung, Koshi Hwy)",
        district="Sankhuwasabha / Taplejung / Bhojpur",
        bbox=(87.00, 27.15, 88.05, 27.95),
        center=(87.550, 27.550),
        cell_count=330,
        river_length_km=51.8,
        highways=["Koshi Highway (NH08 / Dharan-Khandbari-Kimathanka)", "Mechi Highway (NH07 / Taplejung Arterial)"],
        key_settlements=["Khandbari District HQ", "Taplejung / Phungling", "Num (Makalu Gateway)", "Hedangna Hillside Village", "Bhojpur Bazar", "Chainpur"],
        infrastructure_assets=["Arun-3 Mega Hydro Dam Site (900MW)", "Arun-3 Powerhouse Complex", "Lower Arun Hydro (669MW)", "Kabeli-A Hydro (37MW)"],
    ),
}


# --------------------------------------------------------------------------------------
# Human exposure is MEASURED now, not listed here.
#
# A 211-line CORRIDOR_HUMAN_ASSETS table used to sit at this point: ~140 hand-authored
# places carrying unsourced population figures ("2,500 residents", "4,200 residents")
# and prose asserting slope angles - "32 deg colluvial hillslope", "34 deg weathered
# schist" - for terrain nothing had measured. The generator sampled it with
# rng.integers() and scattered cells around the picked entry, so those descriptions
# were the only thing standing in for topography.
#
# It fed nothing once the grid became real, and it is removed rather than left in
# place, because a dead table of plausible numbers is the thing most likely to get
# wired back in. The equivalent now comes from sources, per cell, in
# app/engine/valley_grid_terrain.py:
#
#     settlement name, kind and distance   OpenStreetMap place=city|town|village|hamlet
#     road distance and class              OpenStreetMap highway=*
#     dam / powerhouse distance and name   OpenStreetMap waterway=dam|weir, power=plant
#     slope, aspect, curvature, relief     Copernicus DEM GLO-30
#
# Population per settlement is still absent - OSM population tags are sparse in Nepal
# and were not used rather than guessed. Nepal CBS ward-level census figures are the
# real source if that metric is wanted; it is not invented in the meantime.
# --------------------------------------------------------------------------------------


def resolve_corridor_live_telemetry(corridor_id: str) -> Dict[str, Any]:
    """Retrieve actual raw live telemetry for a corridor from the perpetual database or status.

    Queries recent real-time snapshots from NASA IMERG, Open-Meteo, and USGS seismic feeds
    recorded by the perpetual orchestrator for all observation points in the specified corridor.
    """
    db_path = DATA_DIR / "perpetual_telemetry.db"
    status_path = DATA_DIR / "perpetual_status.json"

    rain_72h = 0.0
    api_30d = 12.0
    seismic_kh = 0.010
    temp_c = 15.0
    pdd_melt = 0.0
    live_ts = time.time()
    provenance = "REAL_TIME_FALLBACK_DEFAULT"

    if db_path.exists():
        try:
            import sqlite3
            with sqlite3.connect(str(db_path), timeout=2.0) as con:
                cur = con.cursor()
                cur.execute("""
                    SELECT t.rain_72h_mm, t.rain_peak_1h_mm, t.api_rain_30d_mm,
                           t.seismic_pga_g, t.temp_2m_c, t.pdd_days_7d, t.timestamp_utc
                    FROM telemetry_snapshots t
                    JOIN sites s ON t.site_id = s.site_id
                    WHERE s.basin_id = ?
                    ORDER BY t.timestamp_utc DESC
                    LIMIT 60
                """, (corridor_id,))
                rows = cur.fetchall()
                if not rows:
                    cur.execute("""
                        SELECT t.rain_72h_mm, t.rain_peak_1h_mm, t.api_rain_30d_mm,
                               t.seismic_pga_g, t.temp_2m_c, t.pdd_days_7d, t.timestamp_utc
                        FROM telemetry_snapshots t
                        ORDER BY t.timestamp_utc DESC
                        LIMIT 60
                    """)
                    rows = cur.fetchall()

                if rows:
                    rains = [r[0] for r in rows if r[0] is not None]
                    if rains:
                        rain_72h = float(np.mean(rains))
                    apis = [r[2] for r in rows if r[2] is not None]
                    if apis:
                        api_30d = float(np.mean(apis))
                    else:
                        api_30d = float(max(12.0, rain_72h * 1.5))
                    pgas = [r[3] for r in rows if r[3] is not None]
                    if pgas:
                        seismic_pga = float(np.max(pgas))
                        seismic_kh = float(0.5 * seismic_pga)
                    temps = [r[4] for r in rows if r[4] is not None]
                    if temps:
                        temp_c = float(np.mean(temps))
                    melts = [r[5] for r in rows if r[5] is not None]
                    if melts:
                        pdd_melt = float(np.mean(melts))
                    timestamps = [r[6] for r in rows if r[6] is not None]
                    if timestamps:
                        live_ts = float(max(timestamps))
                    provenance = "MEASURED_LIVE_RAW_TELEMETRY"
        except Exception:
            pass

    if provenance != "MEASURED_LIVE_RAW_TELEMETRY" and status_path.exists():
        try:
            import json
            with open(status_path, "r", encoding="utf-8") as f:
                st = json.load(f)
                live_ts = float(st.get("last_cycle_timestamp_utc", time.time()))
                rain_72h = float(st.get("cadence", {}).get("max_rain_72h_mm", 0.0))
                api_30d = max(12.0, rain_72h * 1.5)
                provenance = "MEASURED_STATUS_SNAPSHOT"
        except Exception:
            pass

    return {
        "corridor_id": corridor_id,
        "rainfall_72h_mm": round(rain_72h, 2),
        "api_30d_mm": round(api_30d, 2),
        "seismic_kh": round(seismic_kh, 4),
        "temp_2m_c": round(temp_c, 1),
        "pdd_snowmelt_mm": round(pdd_melt, 2),
        "timestamp_utc": live_ts,
        "provenance": provenance,
    }


class ValleyGridPhysicsService:
    """Service that manages and generates physical grid cells across river corridors, settlements & roads."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or GRID_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cells_cache: Dict[str, Dict[str, Any]] = {}

    def get_corridors_list(self) -> List[Dict[str, Any]]:
        """Return the monitored corridors, with their REAL measured cell counts.

        ``CorridorMetadata.cell_count`` is a hand-written number that the old generator
        used as a target to fabricate up to. It is reported here as ``declared_cells``
        for traceability, but ``cell_count`` is now whatever the built grid actually
        holds - and 0, plus ``grid_built: False``, when it has not been built. The
        dashboard must be able to tell "no grid" from "a grid".
        """
        from app.engine.valley_grid_terrain import load_corridor

        out = []
        for cid, meta in PRIMARY_CORRIDORS.items():
            cols = load_corridor(cid)
            n_cells = int(cols["lat"].size) if cols else 0
            with_soil = int((~np.isnan(cols["clay_fraction"])).sum()) if cols else 0
            exposed = (int(sum(1 for e in cols["exposure"]
                               if str(e) in ("ROAD", "SETTLEMENT"))) if cols else 0)
            cells = range(n_cells)
            out.append({
                "id": meta.id,
                "name": meta.name,
                "district": meta.district,
                "bbox": list(meta.bbox),
                "center": list(meta.center),
                "cell_count": with_soil,
                "measured_cells_total": len(cells),
                "cells_without_soil": len(cells) - with_soil,
                "human_exposed_cells": exposed,
                "grid_built": bool(cells),
                "declared_cells": meta.cell_count,
                "terrain_source": "Copernicus DEM GLO-30 + SoilGrids v2.0 + OSM + GEM faults",
                "river_length_km": meta.river_length_km,
                "highways": list(meta.highways),
                "key_settlements": list(meta.key_settlements),
                "infrastructure_assets": list(meta.infrastructure_assets),
            })
        return out

    def _generate_corridor_cells(self, corridor_id: str) -> Dict[str, Any]:
        """Deprecated name kept for callers; delegates to the real measured grid."""
        return self._load_corridor_cells(corridor_id)

    def _load_corridor_cells(self, corridor_id: str) -> Dict[str, Any]:
        """Load the REAL measured 240 m grid for a corridor.

        Replaces a generator that drew every terrain and soil value from
        ``np.random.default_rng(abs(hash(corridor_id)) % 2**31)``. Because Python's
        string hash is salted per process, that seed changed on every API restart, so
        the "reproducible" grid silently reshuffled - and 320 invented cells stood in
        for a bounding box that holds ~36,000 real ones at 240 m.

        Everything returned here is measured:

            elevation, slope, aspect, profile curvature   Copernicus DEM GLO-30
            clay / sand / coarse fragments / bulk density SoilGrids v2.0, 30-200 cm
            road distance and class, river, settlement    OpenStreetMap
            fault distance                                GEM Global Active Faults

        Cells whose soil SoilGrids cannot resolve - bare rock and glacier ice above the
        soil mask - are excluded rather than filled. The infinite-slope soil model does
        not describe a glacier, and inventing a clay fraction for one would be the exact
        error this rewrite exists to remove. The count is reported in
        ``self._excluded_no_soil`` so the API can state it instead of hiding it.

        Raises FileNotFoundError when the grid has not been built. A warning system that
        cannot load its terrain must fail loudly, not improvise one.
        """
        from app.engine.valley_grid_terrain import load_corridor

        if corridor_id not in PRIMARY_CORRIDORS:
            raise KeyError(f"Unknown corridor '{corridor_id}'. "
                           f"Available: {list(PRIMARY_CORRIDORS.keys())}")

        cols = load_corridor(corridor_id)
        if not cols:
            raise FileNotFoundError(
                f"No measured grid for corridor '{corridor_id}'. Build it with "
                f"scripts/build_valley_grid.py - the grid is not generated at runtime.")

        # Soil is required, not optional: the FS needs the unit weight as much as the
        # cohesion. NaN here means SoilGrids masked the pixel - glacier or bare rock -
        # and those cells are dropped rather than filled.
        has_soil = (~np.isnan(cols["clay_fraction"])
                    & ~np.isnan(cols["dry_bulk_density_g_cm3"]))
        n_raw = int(cols["lat"].size)
        keep = np.nonzero(has_soil)[0]
        if not hasattr(self, "_excluded_no_soil"):
            self._excluded_no_soil = {}
        self._excluded_no_soil[corridor_id] = n_raw - int(keep.size)
        if keep.size == 0:
            raise FileNotFoundError(
                f"Corridor '{corridor_id}' has {n_raw} cells but none with resolved "
                f"soil; nothing can be evaluated with the infinite-slope model.")

        def arr(key, default=np.nan):
            """One column, soil-masked, with absences replaced by an explicit default.

            ``default`` is a sentinel the physics can reason about (1e6 for "nothing
            mapped within range"), never a stand-in measurement.
            """
            if key not in cols:
                return np.full(keep.size, default, dtype=np.float64)
            v = cols[key][keep].astype(np.float64)
            return np.where(np.isnan(v), default, v)

        def txt(key):
            if key not in cols:
                return np.array([""] * keep.size, dtype=object)
            return cols[key][keep]

        n = int(keep.size)
        elevation = arr("elevation_m")
        slope = arr("slope_deg")
        road_dist = arr("road_distance_m", default=1.0e6)   # nothing mapped within 1 km
        river_dist = arr("distance_to_river_m", default=1.0e6)
        settle_dist = arr("settlement_distance_m", default=1.0e6)
        hydro_dist = arr("hydro_distance_m", default=1.0e6)
        relief = arr("local_relief_m", default=0.0)
        curvature = arr("profile_curvature", default=0.0)
        road_class = txt("road_class")
        settle_name = txt("settlement_name")
        settle_kind = txt("settlement_kind")
        hydro_name = txt("hydro_name")

        # Asset category, derived from measured distances rather than assigned by quota.
        # The old code set these by hardcoded share - 32% settlement, 34% highway and so
        # on - so "911 settlement cells" was int(round(2850 * 0.32)) and described
        # nothing observed. Each test below is a real threshold on a real distance, and
        # the order encodes consequence: a slope above a powerhouse is not relabelled a
        # gorge merely because the structure sits on a river.
        HIGHWAY = {"trunk", "primary", "secondary", "tertiary",
                   "trunk_link", "primary_link", "secondary_link", "motorway"}
        is_highway = np.array([str(rc) in HIGHWAY for rc in road_class], dtype=bool)

        asset_type = np.full(n, "HILLSLOPE", dtype=object)
        asset_type[road_dist <= 1000.0] = "HIGHWAY_ROADCUT"
        asset_type[elevation >= 4500.0] = "CRYOSPHERE_GLACIAL"
        asset_type[settle_dist <= 2000.0] = "SETTLEMENT"
        asset_type[is_highway & (road_dist <= 500.0)] = "HIGHWAY_ROADCUT"
        asset_type[river_dist <= 150.0] = "RIVER_GORGE"
        asset_type[hydro_dist <= 1500.0] = "HYDRO_INFRASTRUCTURE"

        # Name a cell after the nearest settlement only when it is actually near one.
        # The stored name is the nearest at ANY distance, so without this bound a
        # hillslope 20 km from Chule gets labelled "Chule" on the map.
        NAME_RADIUS_M = 3000.0
        asset_name = np.empty(n, dtype=object)
        desc = np.empty(n, dtype=object)
        metric = np.empty(n, dtype=object)
        for i in range(n):
            if hydro_dist[i] <= 1500.0 and hydro_name[i]:
                asset_name[i] = str(hydro_name[i])
            elif settle_dist[i] <= NAME_RADIUS_M and settle_name[i]:
                asset_name[i] = str(settle_name[i])
            elif road_class[i]:
                asset_name[i] = f"{road_class[i]} road"
            else:
                asset_name[i] = "Unnamed slope"

            # Exposure prose is stated from measured distances. Nothing is invented for
            # a cell: if nothing is mapped nearby, it says exactly that.
            bits = []
            if road_dist[i] < 1.0e5:
                bits.append(f"{road_dist[i]:.0f} m from a mapped "
                            f"{road_class[i] or 'road'}")
            if settle_dist[i] <= 5000.0 and settle_name[i]:
                bits.append(f"{settle_dist[i]:.0f} m from {settle_name[i]}")
            if river_dist[i] <= 2000.0:
                bits.append(f"{river_dist[i]:.0f} m from a mapped river")
            if hydro_dist[i] <= 1500.0:
                bits.append(f"{hydro_dist[i]:.0f} m from "
                            f"{hydro_name[i] or 'a hydro structure'}")
            desc[i] = ("; ".join(bits) if bits
                       else "No mapped road, settlement, river or structure within range")
            metric[i] = (f"{settle_kind[i] or 'slope'} @ {elevation[i]:.0f} m, "
                         f"{slope[i]:.0f} deg")

        # DSGSD indication from measured morphology: high, steep, concave, and with the
        # km-scale relief deep-seated deformation requires. The previous version tested
        # these same thresholds against random numbers.
        dsgsd = ((elevation >= 2800.0) & (slope >= 28.0)
                 & (curvature < -0.05) & (relief >= 500.0))

        return {
            "latitude": arr("lat"),
            "longitude": arr("lon"),
            "elevation_m": elevation,
            "slope_deg": slope,
            "aspect_deg": arr("aspect_deg", default=180.0),
            "profile_curvature": curvature,
            "local_relief_m": relief,
            "distance_to_river_m": river_dist,
            "road_distance_m": road_dist,
            "settlement_distance_m": settle_dist,
            "hydro_distance_m": hydro_dist,
            "road_class": road_class,
            "fault_distance_km": arr("fault_distance_km", default=50.0),
            "clay_fraction": arr("clay_fraction"),
            "sand_fraction": arr("sand_fraction"),
            "coarse_rock_fraction": arr("coarse_rock_fraction", default=0.0),
            "dry_bulk_density_g_cm3": arr("dry_bulk_density_g_cm3"),
            "dsgsd_indicated": dsgsd,
            "asset_type": asset_type,
            "asset_name": asset_name,
            "human_exposure_desc": desc,
            "human_asset_metric": metric,
        }

    def evaluate_corridor_grid(
        self,
        corridor_id: str,
        rainfall_72h_mm: Optional[float] = None,
        api_30d_mm: Optional[float] = None,
        s_ante: Optional[float] = None,
        seismic_kh: Optional[float] = None,
        asset_type: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        exposed_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate physics and Factor of Safety across a corridor's MEASURED cells.

        ``bbox`` (min_lon, min_lat, max_lon, max_lat), ``exposed_only`` and ``limit``
        narrow which cells become records. They are applied before record construction
        rather than after, and what they dropped is recorded in ``self._last_selection``
        so the caller can report it instead of presenting a truncated grid as a whole one.
        """
        if corridor_id not in self._cells_cache:
            self._cells_cache[corridor_id] = self._generate_corridor_cells(corridor_id)

        # Resolve live raw telemetry from perpetual database or status when parameters are omitted
        live_telem = resolve_corridor_live_telemetry(corridor_id)
        actual_rain = rainfall_72h_mm if rainfall_72h_mm is not None else float(live_telem["rainfall_72h_mm"])
        actual_api = api_30d_mm if api_30d_mm is not None else float(live_telem["api_30d_mm"])
        actual_kh = seismic_kh if seismic_kh is not None else float(live_telem["seismic_kh"])
        actual_pdd = float(live_telem.get("pdd_snowmelt_mm", 0.0))

        if not hasattr(self, "_latest_corridor_telemetry"):
            self._latest_corridor_telemetry = {}
        self._latest_corridor_telemetry[corridor_id] = {
            **live_telem,
            "effective_rainfall_72h_mm": actual_rain,
            "effective_api_30d_mm": actual_api,
            "effective_seismic_kh": actual_kh,
            "is_live_telemetry": (rainfall_72h_mm is None and api_30d_mm is None and seismic_kh is None),
        }

        terrain = self._cells_cache[corridor_id]

        phys = compute_vectorized_grid_fs(
            slope_deg=terrain["slope_deg"],
            elevation_m=terrain["elevation_m"],
            aspect_deg=terrain["aspect_deg"],
            profile_curvature=terrain["profile_curvature"],
            road_distance_m=terrain["road_distance_m"],
            fault_distance_km=terrain["fault_distance_km"],
            clay_fraction=terrain["clay_fraction"],
            sand_fraction=terrain["sand_fraction"],
            coarse_rock_fraction=terrain["coarse_rock_fraction"],
            dry_bulk_density_g_cm3=terrain["dry_bulk_density_g_cm3"],
            rainfall_72h_mm=actual_rain,
            api_30d_mm=actual_api,
            s_ante=s_ante,
            pdd_snowmelt_mm=actual_pdd,
            seismic_kh=actual_kh,
            dsgsd_indicated=terrain["dsgsd_indicated"],
        )

        # Decide WHICH cells become records before building any.
        #
        # A corridor holds tens of thousands of measured cells and the physics above is
        # vectorised over all of them cheaply, but turning each into a ~25-key dict is
        # not cheap. Building 29,849 records and discarding 27,349 downstream cost 4.3 s
        # a request; selecting first costs a boolean mask.
        #
        # Selection order matters: the viewport and exposure filters are applied first,
        # then the cap keeps the LOWEST-FS cells. A cap applied before the sort would
        # drop exactly the cells a warning system exists to show.
        n_all = terrain["latitude"].size
        keep_idx = np.arange(n_all)
        if bbox is not None:
            w, s_lat, e, n_lat = bbox
            in_view = ((terrain["longitude"] >= w) & (terrain["longitude"] <= e)
                       & (terrain["latitude"] >= s_lat) & (terrain["latitude"] <= n_lat))
            keep_idx = keep_idx[in_view[keep_idx]]
        if exposed_only:
            is_exposed = terrain["asset_type"] != "HILLSLOPE"
            keep_idx = keep_idx[is_exposed[keep_idx]]

        self._last_selection = {
            "evaluated": int(n_all),
            "after_filters": int(keep_idx.size),
            "truncated": bool(limit is not None and keep_idx.size > limit),
        }
        if limit is not None and keep_idx.size > limit:
            keep_idx = keep_idx[np.argsort(phys["fs"][keep_idx])[:limit]]

        records: List[Dict[str, Any]] = []

        half_w = 0.00108  # ~120m half-width in degrees for 240m grid rectangle (NISAR-aligned 3x3 patch)
        for i in keep_idx:
            i = int(i)
            lat = float(terrain["latitude"][i])
            lon = float(terrain["longitude"][i])
            fs_val = float(phys["fs"][i])
            fail_p = float(phys["failure_prob_pct"][i])
            lvl = classify_alert_level(fs_val, fail_p)
            col = fs_to_gradient_color(fs_val)

            records.append({
                "cell_id": f"{corridor_id}_{i+1:05d}",
                "corridor_id": corridor_id,
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "bounds": [
                    round(lat - half_w, 5),
                    round(lon - half_w, 5),
                    round(lat + half_w, 5),
                    round(lon + half_w, 5),
                ],
                "asset_type": str(terrain["asset_type"][i]),
                "asset_name": str(terrain["asset_name"][i]),
                "human_exposure_desc": str(terrain["human_exposure_desc"][i]),
                "human_asset_metric": str(terrain["human_asset_metric"][i]),
                "elevation_m": round(float(terrain["elevation_m"][i]), 1),
                "slope_deg": round(float(terrain["slope_deg"][i]), 1),
                "effective_slope_deg": round(float(phys["effective_slope_deg"][i]), 1),
                # 1e6 is the "nothing mapped within range" sentinel the physics needs
                # so the road-cut penalty decays to zero. It is not a distance, and must
                # not be reported as one.
                "road_distance_m": (round(float(terrain["road_distance_m"][i]), 1)
                                    if terrain["road_distance_m"][i] < 1.0e5 else None),
                "distance_to_river_m": (round(float(terrain["distance_to_river_m"][i]), 1)
                                        if terrain["distance_to_river_m"][i] < 1.0e5 else None),
                "fault_distance_km": round(float(terrain["fault_distance_km"][i]), 2),
                "soil_depth_z_m": round(float(phys["soil_depth_z_m"][i]), 2),
                "water_table_zw_m": round(float(phys["water_table_zw_m"][i]), 2),
                "saturation_ratio": round(float(phys["saturation_ratio"][i]), 2),
                "effective_cohesion_kpa": round(float(phys["effective_cohesion_kpa"][i]), 2),
                "friction_angle_deg": round(float(phys["friction_angle_deg"][i]), 1),
                "road_cut_penalty": round(float(phys["road_cut_penalty"][i]), 3),
                "dsgsd_indicated": bool(terrain["dsgsd_indicated"][i]),
                "fs": fs_val,
                "failure_prob_pct": fail_p,
                "alert_level": lvl,
                "color_hex": col,
            })

        # Filter by asset category if requested
        if asset_type and asset_type.lower() not in ("all", "any", ""):
            t = asset_type.strip().lower()
            # Normalize shorthand variants
            if t in ("settlement", "settlements", "village", "villages"):
                norm = "settlement"
            elif t in ("highway", "road", "roadcut", "highway_cut", "highway_roadcut"):
                norm = "highway_roadcut"
            elif t in ("hydro", "hydropower", "infrastructure", "hydro_infrastructure"):
                norm = "hydro_infrastructure"
            elif t in ("gorge", "river", "river_gorge", "canyon"):
                norm = "river_gorge"
            elif t in ("cryo", "cryosphere", "glacial", "glacier", "cryosphere_glacial"):
                norm = "cryosphere_glacial"
            elif t in ("hillslope", "slope", "slopes", "terrain"):
                norm = "hillslope"
            else:
                norm = t
            records = [r for r in records if r["asset_type"].lower() == norm]

        return records


# Global singleton instance
valley_grid_service = ValleyGridPhysicsService()
