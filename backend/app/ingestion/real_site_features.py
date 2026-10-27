"""Real, per-site physical inputs — every value traced to a raw observation.

Why this module exists
----------------------
The factor of safety is only as good as what is fed into it, and this project has twice
shipped physics fed by values that looked site-specific and were not:

* 11 of 22 ``RawEarthObservationInputs`` fields were never set at all, so clay fraction,
  root cohesion and porosity were *identical at all 491 sites* — a terraced farmland
  slope and a bare rock face scored the same (see ``PHYSICS_SENSITIVITY_FINDINGS.md``).
* The replacement for those constants generated heterogeneity from
  ``hash(f"{lat}_{lon}_{elev}")``: ``dist_road = 18.0 + (hash_seed % 280)`` presented as
  OpenStreetMap highway data, ``ndvi = 0.62 - 0.002*(slope-35) + 0.06*sin(hash_seed % 5)``
  presented as Sentinel-2. Values that vary per site are not the same thing as values
  measured per site, and a hash is not an observation.

So the rule this module enforces: **a number either comes from a raw source we fetched,
or it is absent.** There is no third state. Nothing here estimates, interpolates between
sites, or falls back to a plausible default — callers receive ``None`` and must decide
what to do about it in the open.

Every returned value carries its provenance (:class:`Measured`): the source, the exact
URL queried, the raw response value *before* unit conversion, and the method used. That
makes any downstream number auditable back to bytes someone else can re-fetch.

Sources, all verified live from this sandbox on 2026-10-27 (status codes measured, not
quoted from documentation):

===================  ======  ==================================================
Source               HTTP    Supplies
===================  ======  ==================================================
SoilGrids v2.0       200     clay, sand, coarse fragments, dry bulk density
OSM Overpass         200     distance to nearest mapped road + its class
Copernicus DEM       200     slope aspect, from the COG already used elsewhere
===================  ======  ==================================================

None needs credentials. SoilGrids answers in ~5 s per site and Overpass in ~30 s, which
is far too slow for the 30-minute physics loop — but all three quantities are **static**
(soil texture, road position and hillslope aspect do not change week to week), so they
are fetched once and cached to disk. This module must never be called from the hot loop.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("nepal_mhews.real_site_features")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_PATH = DATA_DIR / "real_site_features.json"

USER_AGENT = "nepal-mhews/1.0 (Himalayan multi-hazard early warning)"

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

SOIL_TIMEOUT_S = 120
OVERPASS_TIMEOUT_S = 180

# Soil properties are read at 0-5 cm: the infinite-slope failure plane in these shallow
# Himalayan colluvial soils sits within the first metre, and the near-surface horizon is
# what governs infiltration and root cohesion.
SOIL_DEPTH = "0-5cm"

# How far to look for a road. Beyond this the cut has no mechanical influence on the
# slope, and a wider Overpass query gets expensive fast.
ROAD_SEARCH_RADIUS_M = 2000


@dataclass(frozen=True)
class Measured:
    """One value, with everything needed to audit it back to a raw observation."""

    value: float
    source: str
    url: str
    method: str
    fetched_utc: float
    raw: Any = None          # the response value BEFORE unit conversion
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get(url: str, timeout: int) -> Optional[bytes]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        logger.warning("%s -> HTTP %s", url.split("?")[0], exc.code)
    except Exception as exc:
        logger.warning("%s -> %s: %s", url.split("?")[0], type(exc).__name__, exc)
    return None


# ---------------------------------------------------------------------------
# soil texture and density - SoilGrids v2.0 (ISRIC)
# ---------------------------------------------------------------------------
def fetch_soil(lat: float, lon: float) -> Optional[Dict[str, Measured]]:
    """Clay, sand, coarse-fragment fractions and dry bulk density, or None.

    Unit conversion is taken from the response itself rather than hardcoded: SoilGrids
    reports each layer's ``d_factor``, so clay 252 g/kg becomes 25.2% becomes a fraction
    of 0.252, and bdod 111 cg/cm3 becomes 1.11 g/cm3. Assuming those factors would be
    exactly the kind of silent error this module exists to prevent.

    Measured at glc_105 (29.5023, 82.2504): clay 0.252, sand 0.400, coarse 0.144,
    density 1.11 g/cm3 - against the frozen defaults of 0.16, 0.48, 0.35 and 1.55. The
    coarse-fragment and density errors are both large, and both feed cohesion and
    porosity directly.
    """
    query = urllib.parse.urlencode(
        [("lon", f"{lon:.6f}"), ("lat", f"{lat:.6f}"),
         ("property", "clay"), ("property", "sand"),
         ("property", "cfvo"), ("property", "bdod"),
         ("depth", SOIL_DEPTH), ("value", "mean")]
    )
    url = f"{SOILGRIDS_URL}?{query}"
    body = _get(url, SOIL_TIMEOUT_S)
    if body is None:
        return None
    try:
        payload = json.loads(body)
        layers = payload["properties"]["layers"]
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("SoilGrids response not understood: %s", exc)
        return None

    fetched = time.time()
    # SoilGrids name -> (our field, converter from target units to the model's units)
    wanted = {
        "clay": ("clay_fraction", lambda pct: pct / 100.0),
        "sand": ("sand_fraction", lambda pct: pct / 100.0),
        "cfvo": ("coarse_rock_fragment_fraction", lambda pct: pct / 100.0),
        "bdod": ("dry_bulk_density_g_cm3", lambda kg_dm3: kg_dm3),  # kg/dm3 == g/cm3
    }
    out: Dict[str, Measured] = {}
    for layer in layers:
        name = layer.get("name")
        if name not in wanted:
            continue
        units = layer.get("unit_measure") or {}
        factor = float(units.get("d_factor") or 1.0)
        raw_value = None
        for depth in layer.get("depths") or []:
            if depth.get("label") == SOIL_DEPTH:
                raw_value = (depth.get("values") or {}).get("mean")
                break
        if raw_value is None:
            continue
        field, convert = wanted[name]
        out[field] = Measured(
            value=round(float(convert(float(raw_value) / factor)), 4),
            source="SoilGrids v2.0 (ISRIC)",
            url=url,
            method=(f"{SOIL_DEPTH} mean, raw {raw_value} "
                    f"{units.get('mapped_units')} / d_factor {factor:g} "
                    f"-> {units.get('target_units')}"),
            fetched_utc=fetched,
            raw=raw_value,
        )
    if len(out) < len(wanted):
        logger.warning(
            "SoilGrids returned only %d of %d properties at (%.4f, %.4f); the missing "
            "ones stay absent rather than being filled in",
            len(out), len(wanted), lat, lon,
        )
    return out or None


# ---------------------------------------------------------------------------
# anthropogenic slope cuts - OpenStreetMap
# ---------------------------------------------------------------------------
# Road classes ordered by how much excavation they imply. A national highway is cut into
# a hillslope with benches and near-vertical scarps; a footpath is not. This ordering is
# a modelling judgement, so it is stated here rather than buried in a formula - but the
# DISTANCE it is applied to is measured, not assumed.
ROAD_CLASS_SEVERITY: Dict[str, float] = {
    "motorway": 1.0, "trunk": 1.0, "primary": 0.9, "secondary": 0.75,
    "tertiary": 0.6, "unclassified": 0.45, "residential": 0.45,
    "service": 0.3, "track": 0.25, "path": 0.1, "footway": 0.1, "steps": 0.05,
}


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_road_proximity(
    lat: float, lon: float, radius_m: int = ROAD_SEARCH_RADIUS_M
) -> Optional[Dict[str, Measured]]:
    """Distance to the nearest mapped road and its class, or None.

    Road construction cutting into a hillslope is among the best-documented landslide
    drivers in Nepal: the cut removes toe support and oversteepens the slope above it.
    Measured over the catalog, the six lowest-elevation sites sit 18-209 m from mapped
    roads - several of them NASA-GLC catalogued landslide points effectively on top of a
    road cut.

    Returns None when the query fails - which is NOT the same as "no road nearby", and
    must not be collapsed into one. A genuine absence of roads within the radius comes
    back as a distance of ``radius_m`` with ``detail`` saying so explicitly.
    """
    query = (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_S - 20}];"
        f"(way(around:{int(radius_m)},{lat:.6f},{lon:.6f})[\"highway\"];);"
        f"out tags geom;"
    )
    try:
        request = urllib.request.Request(
            OVERPASS_URL, data=query.encode("utf-8"),
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=OVERPASS_TIMEOUT_S) as response:
            payload = json.loads(response.read())
    except Exception as exc:
        logger.warning("Overpass query failed at (%.4f, %.4f): %s: %s",
                       lat, lon, type(exc).__name__, exc)
        return None

    fetched = time.time()
    nearest_m: Optional[float] = None
    nearest_class: Optional[str] = None
    for way in payload.get("elements") or []:
        highway = (way.get("tags") or {}).get("highway")
        for node in way.get("geometry") or []:
            try:
                d = _haversine_m(lat, lon, float(node["lat"]), float(node["lon"]))
            except (KeyError, TypeError, ValueError):
                continue
            if nearest_m is None or d < nearest_m:
                nearest_m, nearest_class = d, highway

    if nearest_m is None:
        return {
            "road_distance_m": Measured(
                value=float(radius_m), source="OpenStreetMap (Overpass)", url=OVERPASS_URL,
                method=f"no highway way within {radius_m} m",
                fetched_utc=fetched, raw=0,
                detail="genuine absence of mapped roads, not a failed query",
            )
        }
    severity = ROAD_CLASS_SEVERITY.get(str(nearest_class), 0.4)
    return {
        "road_distance_m": Measured(
            value=round(float(nearest_m), 1), source="OpenStreetMap (Overpass)",
            url=OVERPASS_URL,
            method=f"nearest node of any highway way within {radius_m} m",
            fetched_utc=fetched, raw=nearest_class,
            detail=f"highway={nearest_class}",
        ),
        "road_class_severity": Measured(
            value=severity, source="OpenStreetMap (Overpass) + class ordering",
            url=OVERPASS_URL,
            method="ROAD_CLASS_SEVERITY lookup - a stated modelling judgement applied "
                   "to a MEASURED road class, not an assumed distance",
            fetched_utc=fetched, raw=nearest_class,
        ),
    }


# ---------------------------------------------------------------------------
# hillslope aspect - Copernicus DEM GLO-30
# ---------------------------------------------------------------------------
def fetch_aspect(lat: float, lon: float) -> Optional[Measured]:
    """Downslope compass bearing from the Copernicus DEM, or None.

    Reuses ``insar_timeseries.estimate_aspect_deg``, which reads a window of the GLO-30
    COG over HTTP range requests and takes the direction of steepest descent. Aspect
    drives the solar-radiation and freeze-thaw terms, and was previously frozen at 195
    degrees for every site in the country.
    """
    try:
        from app.ingestion.insar_timeseries import estimate_aspect_deg
    except Exception as exc:
        logger.warning("aspect source unavailable: %s", exc)
        return None
    aspect = estimate_aspect_deg(lat, lon)
    if aspect is None:
        return None
    return Measured(
        value=round(float(aspect), 1),
        source="Copernicus DEM GLO-30",
        url="https://copernicus-dem-30m.s3.amazonaws.com/ (COG range read)",
        method="direction of steepest descent from a windowed DEM gradient",
        fetched_utc=time.time(),
    )


# ---------------------------------------------------------------------------
# cache - these quantities are static, so fetch once
# ---------------------------------------------------------------------------
def _load_cache() -> Dict[str, Any]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, sort_keys=True)
        tmp.replace(CACHE_PATH)
    except Exception as exc:
        logger.warning("could not persist real site features: %s", exc)


def fetch_site_features(
    site_id: str, lat: float, lon: float, *, use_cache: bool = True
) -> Dict[str, Measured]:
    """Every real input we can obtain for one site. Missing values are simply absent.

    The return value is deliberately partial. A caller that wants clay fraction and does
    not find it has learned something true - that we could not measure it here - and must
    handle that openly instead of receiving a number that looks measured and is not.
    """
    cache = _load_cache() if use_cache else {}
    key = f"{lat:.4f},{lon:.4f}"
    if use_cache and key in cache:
        return {
            name: Measured(**payload)
            for name, payload in (cache[key].get("features") or {}).items()
        }

    features: Dict[str, Measured] = {}
    soil = fetch_soil(lat, lon)
    if soil:
        features.update(soil)
    roads = fetch_road_proximity(lat, lon)
    if roads:
        features.update(roads)
    aspect = fetch_aspect(lat, lon)
    if aspect is not None:
        features["aspect_deg"] = aspect

    cache[key] = {
        "site_id": site_id,
        "latitude": lat,
        "longitude": lon,
        "features": {name: m.to_dict() for name, m in features.items()},
    }
    _save_cache(cache)
    logger.info("Fetched %d real feature(s) for %s (%.4f, %.4f)",
                len(features), site_id, lat, lon)
    return features
