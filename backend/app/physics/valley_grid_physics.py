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
# Ground-truth human settlements, highway road-cuts, and critical infrastructure assets
# --------------------------------------------------------------------------------------
CORRIDOR_HUMAN_ASSETS: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "langtang": {
        "SETTLEMENT": [
            {"name": "Dhunche District HQ & Hospital", "lat": 28.112, "lon": 85.297, "elev": 1960.0, "pop": "4,200 residents", "desc": "Dense district administrative center on 32° colluvial hillslope above highway"},
            {"name": "Syabrubesi Choke Point & Bazaars", "lat": 28.161, "lon": 85.337, "elev": 1460.0, "pop": "2,500 residents", "desc": "Confluence settlement and commercial transport hub vulnerable to debris flows"},
            {"name": "Kyangjin Gompa Alpine Village", "lat": 28.212, "lon": 85.568, "elev": 3870.0, "pop": "350 seasonal residents / trekkers", "desc": "High-altitude glacial valley settlement directly in runout path of cirque avalanches"},
            {"name": "Gatlang Heritage Village (Tamang Heritage)", "lat": 28.140, "lon": 85.265, "elev": 2238.0, "pop": "1,800 residents", "desc": "Traditional tiered hillside stone settlements situated on steep 34° weathered schist slope"},
            {"name": "Thuman & Briddhim Hillside Wards", "lat": 28.195, "lon": 85.365, "elev": 2280.0, "pop": "1,200 residents", "desc": "Mid-slope terraced agricultural settlements on active colluvial slope"},
            {"name": "Timure Border Customs Town", "lat": 28.275, "lon": 85.378, "elev": 1762.0, "pop": "1,500 residents", "desc": "Border dry-port settlement squeezed in narrow gorge beneath vertical cliffs"},
            {"name": "Langtang Village Memorial Bench", "lat": 28.215, "lon": 85.502, "elev": 3430.0, "pop": "250 residents / lodges", "desc": "Rebuilt settlement on 2015 co-seismic rock-ice avalanche debris field"},
        ],
        "HIGHWAY_ROADCUT": [
            {"name": "Pasang Lhamu Highway (Betrawati-Ramche Cliff Cut)", "lat": 28.085, "lon": 85.250, "elev": 1350.0, "traffic": "2,800 veh/day", "desc": "Deep engineered road-cut in fractured phyllite with chronic monsoon rockfalls"},
            {"name": "Pasang Lhamu Highway (Ramche-Dhunche Switchbacks)", "lat": 28.105, "lon": 85.285, "elev": 1820.0, "traffic": "2,500 veh/day", "desc": "Steep climbing hairpins oversteepened by excavation across fault-damaged bedrock"},
            {"name": "Pasang Lhamu Highway (Syabrubesi-Timure Gorge Route)", "lat": 28.220, "lon": 85.355, "elev": 1620.0, "traffic": "3,200 veh/day (Cross-border freight)", "desc": "China-Nepal arterial transit corridor vulnerable to toe undercutting and rock bursts"},
            {"name": "Chilime Hydropower Access Road", "lat": 28.180, "lon": 85.310, "elev": 1750.0, "traffic": "450 veh/day", "desc": "Mountain industrial road cut into steep 46° colluvium"},
        ],
        "HYDRO_INFRASTRUCTURE": [
            {"name": "Upper Trishuli-1 Hydro Dam & Intake Portal", "lat": 28.175, "lon": 85.342, "elev": 1420.0, "metric": "216 MW Mega Hydro", "desc": "Major national infrastructure; dam headworks located in high-relief V-shaped gorge"},
            {"name": "Chilime Hydropower Powerhouse & Switchyard", "lat": 28.170, "lon": 85.305, "elev": 1580.0, "metric": "22 MW Hydro Facility", "desc": "Substation and penstock exposed to hillside boulders and debris cascades"},
            {"name": "Upper Trishuli-3A Headworks & Penstocks", "lat": 28.055, "lon": 85.215, "elev": 950.0, "metric": "60 MW Hydropower", "desc": "Headworks downstream vulnerable to upstream landslide dam surge breaches"},
            {"name": "Syabrubesi Motor Bridge & Suspension Crossing", "lat": 28.163, "lon": 85.339, "elev": 1450.0, "metric": "Strategic Highway Bridge", "desc": "Critical river crossing connecting central Nepal to northern border"},
        ],
        "RIVER_GORGE": [
            {"name": "Trishuli River Deep Canyon Flank", "lat": 28.130, "lon": 85.315, "elev": 1380.0, "metric": "Active Fluvial Toe", "desc": "Monsoon river torrent undercutting toe of unstable valley wall colluvium"},
            {"name": "Langtang Khola Boulder Cascade Gorge", "lat": 28.210, "lon": 85.450, "elev": 2950.0, "metric": "Debris Routing Axis", "desc": "High-velocity debris flow channel from alpine catchments"},
        ],
        "CRYOSPHERE_GLACIAL": [
            {"name": "Langtang Lirung Hanging Glacier & Cirque", "lat": 28.232, "lon": 85.518, "elev": 5680.0, "metric": "7227m Peak Avalanche Wall", "desc": "Source of catastrophic 2015 rock-ice avalanche; high thermal melt detachment risk"},
            {"name": "Shalbachum Proglacial Moraine Wall", "lat": 28.258, "lon": 85.582, "elev": 5380.0, "metric": "GLOF Hazard Lake", "desc": "Moraine-dammed glacial lake subject to ice avalanche impact wave breach"},
        ],
    },
    "budhi_gandaki": {
        "SETTLEMENT": [
            {"name": "Barpak Epicenter Town (Gorkha)", "lat": 28.200, "lon": 84.750, "elev": 1900.0, "pop": "5,000 residents", "desc": "Densely inhabited ridge settlement situated on 2015 earthquake epicenter rupture zone"},
            {"name": "Laprak Hillside Settlement", "lat": 28.225, "lon": 84.780, "elev": 2100.0, "pop": "2,800 residents", "desc": "Steep slope village vulnerable to deep colluvial sliding during monsoon"},
            {"name": "Samagaun (Manaslu Base Hub)", "lat": 28.583, "lon": 84.633, "elev": 3530.0, "pop": "800 residents / climbers", "desc": "High alpine glacial valley community beneath Manaslu icefall hazard cone"},
            {"name": "Lho Village & Monastery", "lat": 28.562, "lon": 84.693, "elev": 3180.0, "pop": "650 residents", "desc": "Perched terrace community on steep metamorphic morainic veneer"},
            {"name": "Namrung Gorge Settlement", "lat": 28.544, "lon": 84.752, "elev": 2660.0, "pop": "450 residents", "desc": "Narrow gorge neck subject to rockfall chutes and river damming"},
            {"name": "Philim Gorge Choke Point Town", "lat": 28.261, "lon": 84.898, "elev": 1570.0, "pop": "1,800 residents", "desc": "Strategic valley choke point with schools and homes beneath near-vertical schist cliffs"},
            {"name": "Arughat Bazar Commercial Center", "lat": 28.120, "lon": 84.819, "elev": 600.0, "pop": "6,500 residents", "desc": "Primary regional commercial hub on river terrace vulnerable to GLOF surge waves"},
        ],
        "HIGHWAY_ROADCUT": [
            {"name": "Gorkha-Barpak Ridge Access Road", "lat": 28.180, "lon": 84.710, "elev": 1650.0, "traffic": "1,400 veh/day", "desc": "Excavated earthen road-cuts traversing sheared Gorkha thrust fault rocks"},
            {"name": "Benighat-Arughat Corridor Cut-Slopes", "lat": 28.115, "lon": 84.820, "elev": 720.0, "traffic": "2,100 veh/day", "desc": "High traffic transport link with toe excavation destabilizing slope batters"},
            {"name": "Arughat-Soti Khola Cliff Alignment", "lat": 28.150, "lon": 84.822, "elev": 680.0, "traffic": "850 veh/day", "desc": "Single-lane cliffside cut oversteepened above churning river"},
            {"name": "Philim Army Road-Cut Trail", "lat": 28.270, "lon": 84.890, "elev": 1620.0, "traffic": "400 mules/veh daily", "desc": "Blasted rock-shelf alignment vulnerable to rock wedges and joint release"},
        ],
        "HYDRO_INFRASTRUCTURE": [
            {"name": "Budhi Gandaki Dam Site & Diversion Axis", "lat": 28.140, "lon": 84.830, "elev": 620.0, "metric": "1200 MW National Project", "desc": "Proposed mega reservoir dam axis in steep rock gorge"},
            {"name": "Philim Strategic Suspension Bridge", "lat": 28.263, "lon": 84.895, "elev": 1550.0, "metric": "Lifeline River Crossing", "desc": "Key suspension span linking northern Manaslu communities"},
            {"name": "Barpak Reconstruction Medical Center", "lat": 28.202, "lon": 84.752, "elev": 1890.0, "metric": "Essential Public Facility", "desc": "Critical community hospital facility situated on colluvial bench"},
        ],
        "RIVER_GORGE": [
            {"name": "Budhi Gandaki Canyon Choke Point", "lat": 28.315, "lon": 84.725, "elev": 1450.0, "metric": "Landslide Dam Choke Point", "desc": "Narrow gorge neck where slope failure easily dams the main river"},
            {"name": "Soti Khola Canyon Toe Undercut", "lat": 28.190, "lon": 84.830, "elev": 730.0, "metric": "Fluvial Undercut Bank", "desc": "Rapid river incision causing periodic toe collapse"},
        ],
        "CRYOSPHERE_GLACIAL": [
            {"name": "Manaslu Cirque Icefall (8163m)", "lat": 28.596, "lon": 84.636, "elev": 5800.0, "metric": "8163m Peak Glacial Shelf", "desc": "Massive hanging serac field capable of triggering catastrophic ice avalanches"},
            {"name": "Birendra Tal Moraine Dam Outlet", "lat": 28.590, "lon": 84.630, "elev": 4620.0, "metric": "Proglacial Lake Basin", "desc": "Glacial lake at base of Manaslu; site of April 2024 surge overflow"},
            {"name": "Thulagi (Dona) Glacial Lake Moraine", "lat": 28.538, "lon": 84.505, "elev": 5120.0, "metric": "High-Risk GLOF Waterbody", "desc": "Large moraine-dammed lake with documented piping and hydrostatic pressure risk"},
        ],
    },
    "kali_gandaki": {
        "SETTLEMENT": [
            {"name": "Jomsom Airport Town & Administrative HQ", "lat": 28.780, "lon": 83.725, "elev": 2720.0, "pop": "3,200 residents / tourists", "desc": "Mustang district center, airstrip, and transit corridor on arid alluvial fan"},
            {"name": "Marpha Heritage Apple Village", "lat": 28.750, "lon": 83.685, "elev": 2650.0, "pop": "1,600 residents", "desc": "Historic stone town on toe of massive ancient colluvial paleolandslide"},
            {"name": "Tatopani Hot Springs Community", "lat": 28.495, "lon": 83.655, "elev": 1190.0, "pop": "1,200 residents", "desc": "Gorge bottom tourist town between Annapurna and Dhaulagiri cliff faces"},
            {"name": "Beni Bazar Commercial Hub (Myagdi HQ)", "lat": 28.345, "lon": 83.565, "elev": 830.0, "pop": "14,000 residents", "desc": "Major district urban center at confluence of Kali Gandaki and Myagdi Khola"},
            {"name": "Dana Hillside Village", "lat": 28.540, "lon": 83.650, "elev": 1440.0, "pop": "1,100 residents", "desc": "Agricultural community on steep terraces opposite Rupse waterfall gorge"},
            {"name": "Ghasa Choke Settlement", "lat": 28.605, "lon": 83.645, "elev": 2010.0, "pop": "950 residents", "desc": "Mustang-Myagdi border settlement perched above narrowest gorge incision"},
            {"name": "Kagbeni Sacred Confluence Village", "lat": 28.835, "lon": 83.780, "elev": 2800.0, "pop": "1,300 residents", "desc": "Gateway to Upper Mustang on gravel terrace subject to debris outwash"},
        ],
        "HIGHWAY_ROADCUT": [
            {"name": "Kali Gandaki Highway (Rupse Chhahara Cliff Cut)", "lat": 28.520, "lon": 83.652, "elev": 1380.0, "traffic": "2,400 veh/day", "desc": "Carved into sheer gneiss cliffs with recurrent monsoon debris sweeps"},
            {"name": "Beni-Tatopani Highway Cut-Slopes", "lat": 28.420, "lon": 83.600, "elev": 980.0, "traffic": "3,100 veh/day", "desc": "Arterial road oversteepened by excavation across fragile phyllitic scree"},
            {"name": "Tatopani-Ghasa Vertical Cut", "lat": 28.570, "lon": 83.648, "elev": 1750.0, "traffic": "2,200 veh/day", "desc": "Deep rock cut beneath world deepest gorge walls with severe toe unloading"},
            {"name": "Pokhara-Baglung Highway (Mid-Hill Arterial)", "lat": 28.270, "lon": 83.590, "elev": 1050.0, "traffic": "5,800 veh/day", "desc": "Heavy traffic national arterial traversing weathered saprolite cut-slopes"},
        ],
        "HYDRO_INFRASTRUCTURE": [
            {"name": "Kali Gandaki A Hydropower Dam & Intake", "lat": 28.180, "lon": 83.590, "elev": 520.0, "metric": "144 MW Hydropower", "desc": "Nepal largest operating hydro plant; vulnerable to sediment and slope runout"},
            {"name": "Rahughat Hydropower Headworks", "lat": 28.380, "lon": 83.570, "elev": 920.0, "metric": "40 MW Hydro Project", "desc": "Dam intake in deeply incised tributary valley subject to flash landslides"},
            {"name": "Jomsom High-Altitude Mountain Runway", "lat": 28.782, "lon": 83.723, "elev": 2720.0, "metric": "Critical Aviation Hub", "desc": "Aviation lifeline bounded by steep dry colluvial mountain walls"},
        ],
        "RIVER_GORGE": [
            {"name": "Kali Gandaki World Deepest Gorge Axis", "lat": 28.650, "lon": 83.650, "elev": 2200.0, "metric": "Deepest Gorge on Earth", "desc": "5500m relief between Annapurna and Dhaulagiri peaks; massive rockfall energy"},
            {"name": "Rupse Chhahara Waterfall Gorge Flank", "lat": 28.525, "lon": 83.655, "elev": 1350.0, "metric": "Debris Avalanche Chute", "desc": "Steep torrent corridor that repeatedly sweeps highway spans"},
        ],
        "CRYOSPHERE_GLACIAL": [
            {"name": "Dhaulagiri I Icefall Avalanche Face", "lat": 28.698, "lon": 83.490, "elev": 5900.0, "metric": "8167m Peak Glacial Face", "desc": "North-east ice face generating high-magnitude powder and rock-ice slides"},
            {"name": "Annapurna IV Sabche Cirque Wall", "lat": 28.612, "lon": 84.085, "elev": 5420.0, "metric": "Seti Gorge Source Cirque", "desc": "Site of catastrophic 2012 Seti rock-ice avalanche and flood surge"},
        ],
    },
    "karnali": {
        "SETTLEMENT": [
            {"name": "Manma Town (Kalikot District HQ)", "lat": 29.145, "lon": 81.615, "elev": 1400.0, "pop": "9,000 residents", "desc": "High ridge district center built on 35° colluvial slopes with severe gully erosion"},
            {"name": "Jumla Khalanga Urban Valley", "lat": 29.275, "lon": 82.185, "elev": 2514.0, "pop": "19,000 residents", "desc": "High mountain valley capital connected solely by vulnerable Karnali Highway"},
            {"name": "Dailekh Bazar Commercial Center", "lat": 28.840, "lon": 81.710, "elev": 1400.0, "pop": "22,000 residents", "desc": "Historic hill district town surrounded by heavily farmed weathered slopes"},
            {"name": "Raskot Mountain Municipality", "lat": 29.230, "lon": 81.560, "elev": 1800.0, "pop": "16,000 residents", "desc": "Terraced agricultural community on steep fragile metamorphic bedrock"},
            {"name": "Nagma Confluence Bazar", "lat": 29.185, "lon": 81.890, "elev": 1200.0, "pop": "2,200 residents", "desc": "Strategic junction town at confluence of Karnali and Tila rivers"},
            {"name": "Pili Village & Memorial Ridge", "lat": 29.170, "lon": 81.650, "elev": 1650.0, "pop": "1,400 residents", "desc": "Hillside village above active Karnali Highway cut-slope failure zones"},
        ],
        "HIGHWAY_ROADCUT": [
            {"name": "Karnali Highway (Gaganepani-Sunarkhola Choke)", "lat": 28.980, "lon": 81.730, "elev": 1150.0, "traffic": "1,600 veh/day", "desc": "Infamous landslide choke point with continuous toe failure during monsoon"},
            {"name": "Karnali Highway (Kalikot-Pili Vertical Rock Cut)", "lat": 29.155, "lon": 81.635, "elev": 1380.0, "traffic": "1,200 veh/day", "desc": "Blasted into fractured phyllite cliffs; vehicle loss risk from falling rock"},
            {"name": "Nagma-Jumla Gorge Alignment", "lat": 29.230, "lon": 82.020, "elev": 1950.0, "traffic": "1,100 veh/day", "desc": "Narrow canyon road-cut along Tila River subject to undercutting"},
            {"name": "Dailekh Mid-Hill Highway Cut-Slopes", "lat": 28.870, "lon": 81.680, "elev": 1420.0, "traffic": "2,200 veh/day", "desc": "Extensive highway excavation cutting through deep residual soils"},
        ],
        "HYDRO_INFRASTRUCTURE": [
            {"name": "Upper Karnali Hydropower Dam Site", "lat": 28.820, "lon": 81.510, "elev": 480.0, "metric": "900 MW Planned Mega Project", "desc": "Major hydroelectric bend development in deep rock gorge"},
            {"name": "Tila River Small Hydro Headworks", "lat": 29.210, "lon": 81.950, "elev": 1780.0, "metric": "Regional Power Supply", "desc": "Hydro plant serving Jumla power grid; vulnerable to river sediment surges"},
            {"name": "Kalikot District Hospital Complex", "lat": 29.148, "lon": 81.618, "elev": 1420.0, "metric": "Sole Emergency Hospital", "desc": "Vital regional healthcare center on steep slope vulnerable to scarp retrogradation"},
        ],
        "RIVER_GORGE": [
            {"name": "Karnali River Deep Canyon Axis", "lat": 29.100, "lon": 81.680, "elev": 1120.0, "metric": "Wild River Fluvial Flank", "desc": "Fast-flowing antecedent river undercutting high-relief mountain flanks"},
            {"name": "Tila River Narrow Choke Point", "lat": 29.250, "lon": 82.080, "elev": 2150.0, "metric": "Tributary Gorge Neck", "desc": "Channel constriction where debris slides frequently impound water"},
        ],
        "CRYOSPHERE_GLACIAL": [
            {"name": "Saipal Glacier Slopes & Cirque Wall", "lat": 29.810, "lon": 81.490, "elev": 5850.0, "metric": "7031m Peak Glacier Head", "desc": "Far-western cryosphere source; permafrost thaw degradation trigger"},
            {"name": "Halji Limi High Glacial Pass", "lat": 29.830, "lon": 81.980, "elev": 4800.0, "metric": "GLOF Outburst Corridor", "desc": "Source of recurrent GLOF events flooding remote Himalayan villages"},
        ],
    },
    "bagmati_narayani": {
        "SETTLEMENT": [
            {"name": "Mugling Commercial Highway Junction", "lat": 27.855, "lon": 84.555, "elev": 240.0, "pop": "8,000 residents / 20k transit", "desc": "Nepal most vital transport confluence town beneath steep 44° rock slopes"},
            {"name": "Malekhu River Bazar & Fish Hub", "lat": 27.805, "lon": 84.825, "elev": 360.0, "pop": "4,500 residents", "desc": "Lifeline highway town bounded by Trishuli River and steep cut-slopes"},
            {"name": "Gajuri Hillside Town", "lat": 27.795, "lon": 84.885, "elev": 450.0, "pop": "6,000 residents", "desc": "Commercial and educational center on narrow river corridor terraces"},
            {"name": "Naubise Valley Transport Gateway", "lat": 27.720, "lon": 85.165, "elev": 890.0, "pop": "7,000 residents", "desc": "Western vehicular entrance to Kathmandu Valley beneath Nagdhunga pass slopes"},
            {"name": "Daman High Mountain Plateau", "lat": 27.605, "lon": 85.090, "elev": 2320.0, "pop": "3,200 residents", "desc": "High pass community on Mahabharat range with steep orchard slopes"},
            {"name": "Bhimphedi Historic Gateway Town", "lat": 27.550, "lon": 85.130, "elev": 1150.0, "pop": "4,000 residents", "desc": "Historical valley gateway beneath steep fractured limestone ridges"},
            {"name": "Kurintar Cable Car Commercial Base", "lat": 27.880, "lon": 84.580, "elev": 280.0, "pop": "3,500 residents / pilgrims", "desc": "Manakamana cable car terminal situated on narrow alluvial shelf"},
        ],
        "HIGHWAY_ROADCUT": [
            {"name": "Prithvi Highway (Malekhu-Mugling Cliff Cut)", "lat": 27.825, "lon": 84.710, "elev": 320.0, "traffic": "14,000 veh/day (National Lifeline)", "desc": "Continuous rock excavation in sheared phyllite; primary food & fuel corridor"},
            {"name": "Narayanghat-Mugling Gorge Highway", "lat": 27.810, "lon": 84.530, "elev": 220.0, "traffic": "12,500 veh/day", "desc": "Deep engineered highway cut with chronic high-impact monsoon landslides"},
            {"name": "Naubise-Nagdhunga Pass Climbing Cuts", "lat": 27.715, "lon": 85.195, "elev": 1250.0, "traffic": "11,000 veh/day", "desc": "Steep switchbacks prone to fill-slope collapse and mudslides"},
            {"name": "Tribhuvan Highway Hairpin Switchbacks", "lat": 27.580, "lon": 85.110, "elev": 1780.0, "traffic": "1,800 veh/day", "desc": "High-altitude mountain road cut across steep Mahabharat fault zones"},
        ],
        "HYDRO_INFRASTRUCTURE": [
            {"name": "Kulekhani I Reservoir Dam & Spillway", "lat": 27.610, "lon": 85.155, "elev": 1530.0, "metric": "60 MW Peaking Storage Dam", "desc": "Nepal only major storage dam; reservoir perimeter slopes prone to landslides"},
            {"name": "Kulekhani II & III Powerhouse Complex", "lat": 27.560, "lon": 85.145, "elev": 920.0, "metric": "46 MW Combined Capacity", "desc": "Strategic power infrastructure nestled in deep canyon floor"},
            {"name": "Mugling Arch Motor Bridge", "lat": 27.857, "lon": 84.557, "elev": 235.0, "metric": "Strategic Highway Bridge", "desc": "Vital link connecting Kathmandu, Pokhara, and the southern plains"},
            {"name": "Manakamana Cable Car Pylon Slopes", "lat": 27.890, "lon": 84.585, "elev": 850.0, "metric": "Aerial Passenger Transit", "desc": "High-voltage cable transport towers anchored on steep 48° rock ribs"},
        ],
        "RIVER_GORGE": [
            {"name": "Trishuli River Lower Canyon Flank", "lat": 27.840, "lon": 84.620, "elev": 260.0, "metric": "Arterial River Flank", "desc": "River scouring Highway embankment toe along 30 km canyon"},
            {"name": "Narayani River Alluvial Bend", "lat": 27.710, "lon": 84.470, "elev": 180.0, "metric": "Major Fluvial Confluence", "desc": "High discharge monsoon floods causing extensive bank undercutting"},
        ],
        "CRYOSPHERE_GLACIAL": [
            {"name": "Ganesh Himal Southern Runoff Slopes", "lat": 28.040, "lon": 85.250, "elev": 4950.0, "metric": "Headwater Glacial Basin", "desc": "Alpine headwaters feeding monsoon torrential surges into Trishuli canyon"},
        ],
    },
    "sun_koshi": {
        "SETTLEMENT": [
            {"name": "Bahrabise Commercial Town", "lat": 27.785, "lon": 85.900, "elev": 820.0, "pop": "9,000 residents", "desc": "Major Sindhupalchok trading hub repeatedly devastated by Bhotekoshi floods"},
            {"name": "Tatopani Border Town & Dry Port", "lat": 27.950, "lon": 85.935, "elev": 1200.0, "pop": "3,000 residents / customs", "desc": "Historic China trade port beneath vertical shattered gneiss gorges"},
            {"name": "Charikot District HQ & Hospital", "lat": 27.670, "lon": 86.030, "elev": 1970.0, "pop": "16,000 residents", "desc": "Dolakha capital on prominent ridge overlooking Tamakoshi gorge"},
            {"name": "Jiri Himalayan Gateway Town", "lat": 27.635, "lon": 86.230, "elev": 1905.0, "pop": "5,500 residents", "desc": "Historic gateway to Everest situated in high mountain valley"},
            {"name": "Dolalghat Confluence Market", "lat": 27.640, "lon": 85.710, "elev": 630.0, "pop": "3,800 residents", "desc": "Junction town at Indrawati and Sun Koshi confluence beneath steep hillslopes"},
            {"name": "Chautara Hilltop Capital", "lat": 27.775, "lon": 85.715, "elev": 1560.0, "pop": "12,000 residents", "desc": "Sindhupalchok administrative center on 2015 earthquake rupture zone"},
        ],
        "HIGHWAY_ROADCUT": [
            {"name": "Araniko Highway (Jure Landslide Scar Cut)", "lat": 27.760, "lon": 85.860, "elev": 750.0, "traffic": "4,500 veh/day", "desc": "Passes through toe of massive 2014 Jure landslide that dammed Sun Koshi"},
            {"name": "Bahrabise-Tatopani Gorge Road-Cuts", "lat": 27.880, "lon": 85.920, "elev": 980.0, "traffic": "3,200 veh/day", "desc": "Excavated into sheer valley walls with severe monsoon washouts"},
            {"name": "Charikot-Tamakoshi Highway Switchbacks", "lat": 27.710, "lon": 86.080, "elev": 1350.0, "traffic": "2,100 veh/day", "desc": "Steep hairpins descending 1000m to Upper Tamakoshi project sites"},
            {"name": "BP Highway Sun Koshi Valley Cuts", "lat": 27.665, "lon": 85.740, "elev": 680.0, "traffic": "6,500 veh/day", "desc": "Modern Japanese-engineered highway with toe protection on steep hillslopes"},
        ],
        "HYDRO_INFRASTRUCTURE": [
            {"name": "Upper Tamakoshi Dam & Desanding Basin", "lat": 27.850, "lon": 86.210, "elev": 1980.0, "metric": "456 MW National Pride", "desc": "Highest head hydropower in Nepal; dam in extreme steep V-notch gorge"},
            {"name": "Upper Tamakoshi Underground Powerhouse", "lat": 27.755, "lon": 86.130, "elev": 1150.0, "metric": "Strategic Power Grid", "desc": "Underground cavern and portal situated in high-stress metamorphic rock"},
            {"name": "Bhotekoshi Hydroelectric Powerhouse", "lat": 27.820, "lon": 85.910, "elev": 890.0, "metric": "45 MW Hydropower", "desc": "Facility repeatedly damaged by GLOFs and boulder-rolling flash floods"},
            {"name": "Friendship Bridge Border Crossing", "lat": 27.975, "lon": 85.960, "elev": 1350.0, "metric": "International Border Bridge", "desc": "Vital bridge link between Nepal and Tibet across torrential Bhotekoshi"},
        ],
        "RIVER_GORGE": [
            {"name": "Bhotekoshi Boulder Canyon Choke", "lat": 27.850, "lon": 85.915, "elev": 920.0, "metric": "High-Gradient Torrents", "desc": "Steepest river gradient in Nepal; extreme hydrodynamic scouring"},
            {"name": "Sun Koshi River Canyon Flanks", "lat": 27.600, "lon": 85.820, "elev": 580.0, "metric": "Regional Trunk River", "desc": "Deep valley walls susceptible to large-scale translational sliding"},
        ],
        "CRYOSPHERE_GLACIAL": [
            {"name": "Tsho Rolpa Glacial Lake Moraine Dam", "lat": 27.862, "lon": 86.440, "elev": 5190.0, "metric": "Nepal Largest GLOF Threat", "desc": "Contains ~85 million m3 water; threatened by terminal moraine degradation"},
        ],
    },
    "sapta_koshi_kanchenjunga": {
        "SETTLEMENT": [
            {"name": "Khandbari Urban Town (Sankhuwasabha HQ)", "lat": 27.375, "lon": 87.205, "elev": 1030.0, "pop": "26,000 residents", "desc": "Rapidly growing hill district capital along Koshi Highway corridor"},
            {"name": "Taplejung / Phungling Bazaar", "lat": 27.350, "lon": 87.670, "elev": 1820.0, "pop": "19,000 residents", "desc": "District center in far-eastern Nepal situated on steep amphitheater slopes"},
            {"name": "Num Village (Makalu Trek Gateway)", "lat": 27.550, "lon": 87.285, "elev": 1560.0, "pop": "1,800 residents", "desc": "Perched village above Arun River gorge directly overlooking Arun-3 dam axis"},
            {"name": "Hedangna Hillside Community", "lat": 27.575, "lon": 87.260, "elev": 1480.0, "pop": "1,400 residents", "desc": "Isolated indigenous settlement on 36° slopes across the Arun gorge"},
            {"name": "Chainpur Heritage Brass Town", "lat": 27.290, "lon": 87.320, "elev": 1320.0, "pop": "8,500 residents", "desc": "Historic settlement on stable terrace surrounded by deeply incised streams"},
            {"name": "Tumlingtar Airport Plateau", "lat": 27.315, "lon": 87.195, "elev": 510.0, "pop": "5,000 residents / airport", "desc": "Plateau bench between Arun and Sabha rivers; regional transit hub"},
        ],
        "HIGHWAY_ROADCUT": [
            {"name": "Koshi Highway (Hile-Khandbari Ridge Cut)", "lat": 27.250, "lon": 87.220, "elev": 1180.0, "traffic": "2,800 veh/day", "desc": "Arterial road linking eastern plains to northern Himalayan frontier"},
            {"name": "Khandbari-Num Arun-3 Access Highway", "lat": 27.460, "lon": 87.240, "elev": 1320.0, "traffic": "1,900 veh/day", "desc": "Heavy industrial road cut through steep weathered schist for dam transit"},
            {"name": "Taplejung-Kabeli Gorge Road-Cuts", "lat": 27.310, "lon": 87.720, "elev": 1450.0, "traffic": "1,100 veh/day", "desc": "Single-lane cut-slopes vulnerable to monsoon debris choking"},
        ],
        "HYDRO_INFRASTRUCTURE": [
            {"name": "Arun-3 Mega Dam Axis (Phaksinda)", "lat": 27.540, "lon": 87.255, "elev": 820.0, "metric": "900 MW Mega Hydro", "desc": "National priority export hydropower dam in deep Arun gorge"},
            {"name": "Arun-3 Powerhouse Complex (Pukhuwa)", "lat": 27.420, "lon": 87.230, "elev": 580.0, "metric": "Underground Powerhouse", "desc": "Massive cavern construction in steep valley walls"},
            {"name": "Lower Arun Hydropower Dam Site", "lat": 27.360, "lon": 87.210, "elev": 510.0, "metric": "669 MW Planned Hydro", "desc": "Cascade hydro project site in lower Arun canyon"},
            {"name": "Kabeli-A Hydroelectric Intake", "lat": 27.290, "lon": 87.740, "elev": 780.0, "metric": "37 MW Hydropower", "desc": "Headworks in narrow Tamor tributary canyon"},
        ],
        "RIVER_GORGE": [
            {"name": "Arun River Deep Trans-Himalayan Gorge", "lat": 27.480, "lon": 87.245, "elev": 750.0, "metric": "Deepest Eastern Nepal Gorge", "desc": "Cuts directly through High Himalayan crystalline rocks with enormous relief"},
            {"name": "Tamor River Boulder Cascade Narrows", "lat": 27.260, "lon": 87.650, "elev": 690.0, "metric": "Torrential Fluvial Axis", "desc": "High velocity river scouring steep toe deposits"},
        ],
        "CRYOSPHERE_GLACIAL": [
            {"name": "Makalu Southeastern Icefall (8485m)", "lat": 27.890, "lon": 87.120, "elev": 6100.0, "metric": "8485m Peak Hanging Glacier", "desc": "Major glacial detachment headwall feeding Barun river basin"},
            {"name": "Lower Barun Glacial Lake Moraine Wall", "lat": 27.798, "lon": 87.092, "elev": 4890.0, "metric": "High Expansion GLOF Lake", "desc": "Rapidly growing proglacial lake with active calving front"},
        ],
    },
}


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
        from app.engine.valley_grid_terrain import load_grid

        grid = load_grid()
        out = []
        for cid, meta in PRIMARY_CORRIDORS.items():
            cells = grid.get(cid) or []
            with_soil = sum(1 for c in cells if c.get("clay_fraction") is not None)
            exposed = sum(1 for c in cells
                          if c.get("exposure") in ("ROAD", "SETTLEMENT"))
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
        from app.engine.valley_grid_terrain import load_grid

        if corridor_id not in PRIMARY_CORRIDORS:
            raise KeyError(f"Unknown corridor '{corridor_id}'. "
                           f"Available: {list(PRIMARY_CORRIDORS.keys())}")

        grid = load_grid()
        raw = grid.get(corridor_id) or []
        if not raw:
            raise FileNotFoundError(
                f"No measured grid for corridor '{corridor_id}'. Build it with "
                f"scripts/build_valley_grid.py - the grid is not generated at runtime.")

        usable = [c for c in raw if c.get("clay_fraction") is not None
                  and c.get("dry_bulk_density_g_cm3") is not None]
        if not hasattr(self, "_excluded_no_soil"):
            self._excluded_no_soil = {}
        self._excluded_no_soil[corridor_id] = len(raw) - len(usable)
        if not usable:
            raise FileNotFoundError(
                f"Corridor '{corridor_id}' has {len(raw)} cells but none with resolved "
                f"soil; nothing can be evaluated with the infinite-slope model.")

        def arr(key, dtype=np.float64, default=np.nan):
            return np.array([c.get(key) if c.get(key) is not None else default
                             for c in usable], dtype=dtype)

        elevation = arr("elevation_m")
        slope = arr("slope_deg")
        road_dist = arr("road_distance_m", default=1.0e6)   # no road mapped within 1 km
        river_dist = arr("distance_to_river_m", default=1.0e6)
        settle_dist = arr("settlement_distance_m", default=1.0e6)
        road_class = np.array([c.get("road_class") or "" for c in usable], dtype=object)

        # Asset category, derived from the measurements rather than assigned by quota.
        # The old code set these by hardcoded share - 32% settlement, 34% highway and so
        # on - so "911 settlement cells" was int(round(2850 * 0.32)) and described
        # nothing observed. Each test below is a real threshold on a real distance.
        HIGHWAY = {"trunk", "primary", "secondary", "tertiary",
                   "trunk_link", "primary_link", "secondary_link", "motorway"}
        hydro_dist = arr("hydro_distance_m", default=1.0e6)
        asset_type = np.empty(len(usable), dtype=object)
        for i, c in enumerate(usable):
            rc = c.get("road_class") or ""
            # Hydro first: a slope above a dam or powerhouse is the highest-consequence
            # case in these corridors, and it should not be relabelled as a gorge just
            # because the structure sits on a river.
            if hydro_dist[i] <= 1500.0:
                asset_type[i] = "HYDRO_INFRASTRUCTURE"
            elif river_dist[i] <= 150.0:
                asset_type[i] = "RIVER_GORGE"
            elif rc in HIGHWAY and road_dist[i] <= 500.0:
                asset_type[i] = "HIGHWAY_ROADCUT"
            elif settle_dist[i] <= 2000.0:
                asset_type[i] = "SETTLEMENT"
            elif elevation[i] >= 4500.0:
                asset_type[i] = "CRYOSPHERE_GLACIAL"
            elif road_dist[i] <= 1000.0:
                asset_type[i] = "HIGHWAY_ROADCUT"
            else:
                asset_type[i] = "HILLSLOPE"

        # Name a cell after the nearest settlement only when it is actually near one.
        # ``settlement_name`` is the nearest at any distance, so without this bound a
        # hillslope 20 km from Chule gets labelled "Chule" on the map.
        NAME_RADIUS_M = 3000.0
        asset_name = np.array(
            [(c.get("hydro_name") if (c.get("hydro_distance_m") is not None
                                      and c["hydro_distance_m"] <= 1500.0)
              else c.get("settlement_name") if (c.get("settlement_distance_m") is not None
                                           and c["settlement_distance_m"] <= NAME_RADIUS_M)
              else (f"{c['road_class']} road" if c.get("road_class") else "Unnamed slope"))
             for c in usable], dtype=object)

        # Exposure description, stated from measured distances. No prose is invented for
        # a cell: if nothing is mapped nearby, it says so.
        desc = np.empty(len(usable), dtype=object)
        metric = np.empty(len(usable), dtype=object)
        for i, c in enumerate(usable):
            bits = []
            if c.get("road_distance_m") is not None:
                bits.append(f"{c['road_distance_m']:.0f} m from a mapped "
                            f"{c.get('road_class') or 'road'}")
            if c.get("settlement_distance_m") is not None and settle_dist[i] <= 5000.0:
                bits.append(f"{c['settlement_distance_m']:.0f} m from "
                            f"{c.get('settlement_name') or 'a settlement'}")
            if c.get("distance_to_river_m") is not None and river_dist[i] <= 2000.0:
                bits.append(f"{c['distance_to_river_m']:.0f} m from a mapped river")
            if c.get("hydro_distance_m") is not None:
                bits.append(f"{c['hydro_distance_m']:.0f} m from "
                            f"{c.get('hydro_name') or 'a hydro structure'}")
            desc[i] = "; ".join(bits) if bits else "No mapped road, settlement or river within range"
            metric[i] = (f"{c.get('settlement_kind') or 'slope'} "
                         f"@ {c['elevation_m']:.0f} m, {c['slope_deg']:.0f} deg")

        # DSGSD indication from measured morphology: high, steep, concave and with the
        # km-scale relief that deep-seated deformation requires. The previous version
        # tested the same thresholds against random numbers.
        relief = arr("local_relief_m", default=0.0)
        curvature = arr("profile_curvature", default=0.0)
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
    ) -> List[Dict[str, Any]]:
        """Evaluate physics and Factor of Safety across all cells in a corridor using live raw telemetry."""
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

        n = terrain["latitude"].size
        records: List[Dict[str, Any]] = []

        half_w = 0.00108  # ~120m half-width in degrees for 240m grid rectangle (NISAR-aligned 3x3 patch)
        for i in range(n):
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
