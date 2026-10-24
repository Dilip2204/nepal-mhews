"""Sentinel-1 ingestion from the Copernicus Data Space Ecosystem (CDSE), honestly labelled.

Why this module exists
----------------------
The dashboard and the perpetual engine surface a value labelled "Sentinel-1 InSAR LOS
Creep Velocity". That number has never been InSAR: it is a viscoplastic creep heuristic
derived from slope, factor of safety and rainfall
(``app.physics.infinite_slope.compute_viscoplastic_creep_velocity``). Presenting a modelled
quantity as a satellite measurement is a trust defect (HANDOFF.md Issue #2). This module
makes the real measurement path possible and forces every creep value to carry an explicit
provenance label.

What CDSE actually provides (verified against the live API on 2026-10-23)
-------------------------------------------------------------------------
* The legacy resto/OpenSearch endpoint (``/resto/api/collections/...``, which is still the
  value of ``settings.cdse_catalogue_url``) now answers HTTP 404. The working catalogue is
  OData at ``https://catalogue.dataspace.copernicus.eu/odata/v1/Products``; this module
  derives that endpoint from the configured host.
* The Sentinel-1 collections CDSE publishes are ``sentinel-1-slc``, ``sentinel-1-slc-burst``,
  ``sentinel-1-grd``, ``sentinel-1-ocn``, ``sentinel-1-etad`` and ``sentinel-1-global-mosaics``.
  None of them is a ground-deformation or line-of-sight velocity product. CDSE publishes no
  displacement collection covering Nepal; EGMS, the Copernicus ground-motion service, is
  restricted to the EEA39 area and is not in the CDSE catalogue at all.
* Turning the available SLC scenes into an LOS velocity requires co-registration,
  interferogram formation, phase unwrapping and time-series inversion -- an ISCE2 / LiCSAR /
  HyP3-scale pipeline that this module deliberately does not pretend to perform.

Therefore ``get_insar_creep()`` never reports ``MEASURED_INSAR`` on the strength of a CDSE
scene listing. What it does return from CDSE is genuine, checkable metadata: which
Sentinel-1 acquisitions actually cover the point, on which relative orbits, and whether an
interferometric pair is formable ("InSAR readiness"). The velocity itself is reported as
``MODELLED_PROXY`` (the physics heuristic, named as such) or ``UNAVAILABLE``.

To obtain real velocities, register an external provider through
``register_measured_velocity_provider()`` backed by one of:
  * COMET-LiCSAR / LiCSBAS (CEDA archive) -- already covers the Alpine-Himalayan belt and
    publishes automatically processed interferograms and velocity fields:
    https://comet.nerc.ac.uk/comet-lics-portal
  * ASF HyP3 on-demand Sentinel-1 InSAR (global, API-driven).
  * ESA/Copernicus EGMS -- Europe only, unusable for Nepal.

Offline behaviour: with ``settings.cdse_enabled`` false or credentials absent (the current
state), no network call is attempted at all and the proxy / unavailable path is returned.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.physics.infinite_slope import compute_viscoplastic_creep_velocity

logger = logging.getLogger(__name__)

# --- Provenance vocabulary -------------------------------------------------------------
PROVENANCE_MEASURED = "MEASURED_INSAR"
PROVENANCE_PROXY = "MODELLED_PROXY"
PROVENANCE_UNAVAILABLE = "UNAVAILABLE"

PROXY_SOURCE = (
    "physics proxy (viscoplastic creep from slope / dynamic FS / rainfall) - NOT InSAR"
)
NO_DEFORMATION_PRODUCT_NOTE = (
    "CDSE publishes Sentinel-1 SLC/GRD scenes only; it has no ground-deformation or LOS "
    "velocity collection covering Nepal. A measured velocity requires COMET-LiCSAR, "
    "ASF HyP3 on-demand InSAR, or an equivalent interferometric processing service."
)

# --- Sentinel-1 / catalogue constants ---------------------------------------------------
SENTINEL1_COLLECTION = "SENTINEL-1"
SENTINEL1_SLC_PRODUCT_TYPE = "IW_SLC__1S"
# Nominal single-satellite repeat. With two operational satellites the effective revisit on
# a given track can be 6 days; 12 days is the conservative cache window.
SENTINEL1_REPEAT_DAYS = 12
DEFAULT_SEARCH_WINDOW_DAYS = 36
DEFAULT_MAX_RECORDS = 20

# One Sentinel-1 IW swath is ~250 km wide, so a 0.25 deg (~25 km) cell collapses the
# 491-site loop onto a few dozen catalogue queries instead of 491.
DEFAULT_CACHE_GRID_DEG = 0.25
MAX_CATALOGUE_QUERIES_PER_BATCH = 24
DEFAULT_HTTP_TIMEOUT_SEC = 20.0
ERROR_CACHE_TTL_SEC = 6 * 3600
_TOKEN_EXPIRY_MARGIN_SEC = 60.0

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_CACHE_PATH = DATA_DIR / "insar_cdse_cache.json"
_CACHE_SCHEMA_VERSION = 1

_cache_path_override: Optional[Path] = None
_token_cache: Dict[str, Any] = {"access_token": None, "expires_at": 0.0}
_measured_velocity_provider: Optional[Callable[[float, float], Optional[Dict[str, Any]]]] = None


# =======================================================================================
# Shared creep proxy -- the single definition other modules should import
# =======================================================================================
def compute_creep_proxy(
    slope_deg: float,
    dynamic_fs: float,
    elevation_m: float = 2000.0,
    hazard_type: str = "rainfall_landslide",
    rain_72h_mm: float = 0.0,
    saturation_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Modelled ground-creep estimate with explicit MODELLED_PROXY provenance.

    This is the one creep formula the API, the orchestrator and the dashboard should all
    call, so the two previously divergent expressions (routes.py vs perpetual_orchestrator)
    collapse to a single number. It is a physics heuristic, not a measurement: the returned
    dict says so, and callers must surface ``provenance``/``source`` rather than the bare
    float.
    """
    velocity = compute_viscoplastic_creep_velocity(
        slope_deg=slope_deg,
        dynamic_fs=dynamic_fs,
        elevation_m=elevation_m,
        hazard_type=hazard_type,
        rain_72h_mm=rain_72h_mm,
        saturation_ratio=saturation_ratio,
    )
    return {
        "los_velocity_mm_month": float(velocity),
        "provenance": PROVENANCE_PROXY,
        "source": PROXY_SOURCE,
        "acquired_utc": None,
        "computed_utc": time.time(),
        "confidence": 0.30,
        "display_label": "Modelled creep (physics proxy)",
        "is_measurement": False,
    }


# =======================================================================================
# External measured-velocity provider hook
# =======================================================================================
def register_measured_velocity_provider(
    provider: Optional[Callable[[float, float], Optional[Dict[str, Any]]]],
) -> None:
    """Register a callable returning a genuinely measured LOS velocity for (lat, lon).

    The provider must return ``None`` when it has no measurement, or a dict carrying a
    numeric ``los_velocity_mm_month`` together with a product/dataset reference
    (``product_id`` or ``source``). Anything else is rejected and the proxy path is used, so
    ``MEASURED_INSAR`` can never be produced by this module without a real product behind it.

    Intended backends: COMET-LiCSAR frame velocities, ASF HyP3 on-demand interferograms.
    """
    global _measured_velocity_provider
    _measured_velocity_provider = provider


def _invoke_measured_provider(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Call the registered provider and validate that it really supplied a measurement."""
    if _measured_velocity_provider is None:
        return None
    try:
        result = _measured_velocity_provider(float(lat), float(lon))
    except Exception as exc:  # a broken provider must never take the engine down
        logger.warning("Measured InSAR provider raised %s; falling back to proxy", exc)
        return None
    if not isinstance(result, dict):
        return None

    velocity = result.get("los_velocity_mm_month")
    if not isinstance(velocity, (int, float)) or isinstance(velocity, bool):
        logger.warning("Measured InSAR provider returned no numeric velocity; ignoring")
        return None
    if not math.isfinite(float(velocity)):
        return None

    product_id = result.get("product_id") or ""
    source = result.get("source") or ""
    if not product_id and not source:
        logger.warning(
            "Measured InSAR provider returned a velocity with no product reference; "
            "refusing to label it %s",
            PROVENANCE_MEASURED,
        )
        return None

    observation = dict(result)
    observation["los_velocity_mm_month"] = float(velocity)
    observation["provenance"] = PROVENANCE_MEASURED
    observation["source"] = source or f"InSAR deformation product {product_id}"
    observation.setdefault("product_id", product_id or None)
    observation.setdefault("acquired_utc", None)
    observation.setdefault("confidence", 0.80)
    observation["display_label"] = "Measured InSAR LOS velocity"
    observation["is_measurement"] = True
    observation["computed_utc"] = time.time()
    return observation


# =======================================================================================
# CDSE OAuth + OData catalogue client
# =======================================================================================
def _odata_products_url() -> str:
    """Derive the working OData Products endpoint from the configured catalogue URL.

    ``settings.cdse_catalogue_url`` still points at the retired resto/OpenSearch path, which
    returns 404. Only the host is taken from it.
    """
    configured = (settings.cdse_catalogue_url or "").strip()
    parsed = urllib.parse.urlsplit(configured)
    if "/odata/" in parsed.path:
        return configured.rstrip("/")
    host = parsed.netloc or "catalogue.dataspace.copernicus.eu"
    scheme = parsed.scheme or "https"
    return f"{scheme}://{host}/odata/v1/Products"


def _http_json(
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """POST/GET a URL and decode a JSON object. Raises on transport or decode failure."""
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("CDSE response was not a JSON object")
    return decoded


def fetch_cdse_access_token(
    *, force_refresh: bool = False, timeout: float = DEFAULT_HTTP_TIMEOUT_SEC
) -> Optional[str]:
    """OAuth2 client-credentials token for CDSE, or None if unobtainable.

    Verified request shape (CDSE docs, APIs/Token.html):
        POST https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
        content-type: application/x-www-form-urlencoded
        grant_type=client_credentials&client_id=...&client_secret=...
    Response: {"access_token": "...", "expires_in": 600, "token_type": "Bearer", ...}
    """
    client_id = (settings.cdse_client_id or "").strip()
    client_secret = (settings.cdse_client_secret or "").strip()
    if not client_id or not client_secret:
        return None

    now = time.time()
    if not force_refresh:
        cached = _token_cache.get("access_token")
        if cached and float(_token_cache.get("expires_at", 0.0)) > now + _TOKEN_EXPIRY_MARGIN_SEC:
            return str(cached)

    body = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode("utf-8")
    try:
        payload = _http_json(
            settings.cdse_token_url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
    except Exception as exc:
        # Never log the secret, and never raise: the engine must keep running offline.
        logger.warning("CDSE token request failed (%s)", type(exc).__name__)
        return None

    token = payload.get("access_token")
    if not token or not isinstance(token, str):
        logger.warning("CDSE token response contained no access_token")
        return None

    try:
        expires_in = float(payload.get("expires_in", 600.0))
    except (TypeError, ValueError):
        expires_in = 600.0
    _token_cache["access_token"] = token
    _token_cache["expires_at"] = now + max(expires_in, 0.0)
    return token


def _odata_datetime(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _build_sentinel1_filter(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    product_type: str,
    collection: str,
) -> str:
    """OData $filter for Sentinel-1 scenes intersecting a point within a date window."""
    clauses = [
        f"Collection/Name eq '{collection}'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({float(lon):.6f} {float(lat):.6f})')",
        f"ContentDate/Start gt {_odata_datetime(start)}",
        f"ContentDate/Start lt {_odata_datetime(end)}",
    ]
    if product_type:
        clauses.append(
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
            f"att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        )
    return " and ".join(clauses)


def _parse_product(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one OData product record into the fields this module reports."""
    attributes: Dict[str, Any] = {}
    for attribute in raw.get("Attributes") or []:
        name = attribute.get("Name")
        if name:
            attributes[name] = attribute.get("Value")

    content_date = raw.get("ContentDate") or {}
    relative_orbit = attributes.get("relativeOrbitNumber")
    try:
        relative_orbit = int(relative_orbit) if relative_orbit is not None else None
    except (TypeError, ValueError):
        relative_orbit = None

    return {
        "product_id": raw.get("Id"),
        "name": raw.get("Name"),
        "sensing_start_utc": content_date.get("Start"),
        "product_type": attributes.get("productType"),
        "relative_orbit": relative_orbit,
        "orbit_direction": attributes.get("orbitDirection"),
        "platform": attributes.get("platformSerialIdentifier"),
        "polarisation": attributes.get("polarisationChannels"),
        "online": bool(raw.get("Online", False)),
        "s3_path": raw.get("S3Path"),
    }


def search_sentinel1_scenes(
    lat: float,
    lon: float,
    *,
    days: int = DEFAULT_SEARCH_WINDOW_DAYS,
    product_type: str = SENTINEL1_SLC_PRODUCT_TYPE,
    collection: str = SENTINEL1_COLLECTION,
    max_records: int = DEFAULT_MAX_RECORDS,
    token: Optional[str] = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Query the CDSE OData catalogue for Sentinel-1 scenes covering a point.

    Returns ``{"ok": bool, "products": [...], "error": str | None, "query_url": str}``.
    This lists acquisitions; it does not and cannot produce a deformation measurement.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(int(days), 1))
    params = {
        "$filter": _build_sentinel1_filter(lat, lon, start, end, product_type, collection),
        "$orderby": "ContentDate/Start desc",
        "$top": str(max(int(max_records), 1)),
        "$expand": "Attributes",
    }
    # quote_via=quote keeps spaces as %20, matching the request shape verified against the
    # live OData endpoint; some OData filter parsers reject '+' as a space.
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    url = f"{_odata_products_url()}?{query}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        payload = _http_json(url, headers=headers, timeout=timeout)
    except Exception as exc:
        logger.warning("CDSE catalogue query failed (%s)", type(exc).__name__)
        return {"ok": False, "products": [], "error": type(exc).__name__, "query_url": url}

    raw_products = payload.get("value")
    if not isinstance(raw_products, list):
        return {"ok": False, "products": [], "error": "malformed_response", "query_url": url}

    products = [_parse_product(item) for item in raw_products if isinstance(item, dict)]
    return {"ok": True, "products": products, "error": None, "query_url": url}


def assess_insar_readiness(products: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe, from scene metadata alone, whether interferometry is even possible here.

    An interferometric pair needs two acquisitions on the same relative orbit and pass
    direction. This reports that fact; it never implies a velocity was derived.
    """
    tracks: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for product in products:
        key = (product.get("relative_orbit"), product.get("orbit_direction"))
        tracks.setdefault(key, []).append(product)

    best_key: Optional[Tuple[Any, Any]] = None
    best_count = 0
    for key, scenes in tracks.items():
        if key[0] is not None and len(scenes) > best_count:
            best_key, best_count = key, len(scenes)

    timestamps = sorted(
        [p.get("sensing_start_utc") for p in products if p.get("sensing_start_utc")]
    )
    latest = timestamps[-1] if timestamps else None

    return {
        "scene_count": len(products),
        "track_count": len([k for k in tracks if k[0] is not None]),
        "best_relative_orbit": best_key[0] if best_key else None,
        "best_orbit_direction": best_key[1] if best_key else None,
        "best_track_scene_count": best_count,
        "interferometric_pair_formable": bool(best_count >= 2),
        "latest_acquisition_utc": latest,
        "nominal_repeat_days": SENTINEL1_REPEAT_DAYS,
    }


# =======================================================================================
# Per-cell CDSE context with a persistent 12-day cache
# =======================================================================================
def set_cache_path(path: Optional[Path]) -> None:
    """Override the on-disk cache location (used by tests)."""
    global _cache_path_override
    _cache_path_override = Path(path) if path is not None else None


def _cache_path() -> Path:
    if _cache_path_override is not None:
        return _cache_path_override
    env_path = os.getenv("MHEWS_INSAR_CACHE_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return _DEFAULT_CACHE_PATH


def _load_cache() -> Dict[str, Any]:
    path = _cache_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Discarding unreadable InSAR cache %s (%s)", path, type(exc).__name__)
    return {"version": _CACHE_SCHEMA_VERSION, "entries": {}}


def _save_cache(cache: Dict[str, Any]) -> None:
    path = _cache_path()
    cutoff = time.time() - 4.0 * SENTINEL1_REPEAT_DAYS * 86400.0
    cache["entries"] = {
        key: entry
        for key, entry in cache.get("entries", {}).items()
        if float(entry.get("fetched_utc", 0.0)) >= cutoff
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
        os.replace(temporary, path)
    except Exception as exc:
        logger.warning("Could not persist InSAR cache (%s)", type(exc).__name__)


def cache_cell_key(lat: float, lon: float, grid_deg: float = DEFAULT_CACHE_GRID_DEG) -> str:
    """Quantise a coordinate onto the catalogue-query grid."""
    step = max(float(grid_deg), 1e-6)
    return f"{math.floor(float(lat) / step) * step:.3f},{math.floor(float(lon) / step) * step:.3f}"


def clear_caches() -> None:
    """Reset the in-process token cache and the registered measured provider."""
    _token_cache["access_token"] = None
    _token_cache["expires_at"] = 0.0
    register_measured_velocity_provider(None)


def _entry_is_fresh(entry: Dict[str, Any], now: float) -> bool:
    status = (entry.get("context") or {}).get("status")
    ttl = ERROR_CACHE_TTL_SEC if status in ("TOKEN_ERROR", "CATALOGUE_ERROR") else (
        SENTINEL1_REPEAT_DAYS * 86400.0
    )
    return (now - float(entry.get("fetched_utc", 0.0))) < ttl


def get_cdse_context(
    lat: float,
    lon: float,
    *,
    cache: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    allow_network: bool = True,
    timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
) -> Tuple[Dict[str, Any], bool]:
    """Resolve the CDSE situation for a point. Returns (context, performed_network_call).

    No network call is made when CDSE is disabled, when credentials are missing, when a
    fresh cache entry exists, or when ``allow_network`` is False.
    """
    if not settings.cdse_enabled:
        return (
            {
                "status": "DISABLED",
                "measured_velocity_available": False,
                "detail": "CDSE_ENABLED=0; no Sentinel-1 catalogue query attempted.",
            },
            False,
        )
    if not (settings.cdse_client_id or "").strip() or not (
        settings.cdse_client_secret or ""
    ).strip():
        return (
            {
                "status": "NO_CREDENTIALS",
                "measured_velocity_available": False,
                "detail": "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not configured.",
            },
            False,
        )

    cell = cache_cell_key(lat, lon)
    now = time.time()
    if use_cache and cache is not None:
        entry = cache.get("entries", {}).get(cell)
        if isinstance(entry, dict) and _entry_is_fresh(entry, now):
            context = dict(entry.get("context") or {})
            context["cache_hit"] = True
            context["cached_at_utc"] = entry.get("fetched_utc")
            return context, False

    if not allow_network:
        return (
            {
                "status": "QUERY_BUDGET_EXHAUSTED",
                "measured_velocity_available": False,
                "detail": "Catalogue query budget for this cycle was spent; retry next cycle.",
            },
            False,
        )

    token = fetch_cdse_access_token(timeout=timeout_sec)
    if token is None:
        context = {
            "status": "TOKEN_ERROR",
            "measured_velocity_available": False,
            "detail": "CDSE OAuth client-credentials token could not be obtained.",
        }
    else:
        search = search_sentinel1_scenes(lat, lon, token=token, timeout=timeout_sec)
        if not search.get("ok"):
            context = {
                "status": "CATALOGUE_ERROR",
                "measured_velocity_available": False,
                "detail": f"OData query failed: {search.get('error')}",
            }
        else:
            products = search.get("products", [])
            context = {
                "status": "SCENES_ONLY",
                # CDSE has no deformation collection; this stays False by construction.
                "measured_velocity_available": False,
                "collection_searched": SENTINEL1_COLLECTION,
                "product_type_searched": SENTINEL1_SLC_PRODUCT_TYPE,
                "insar_readiness": assess_insar_readiness(products),
                "latest_product_id": products[0].get("product_id") if products else None,
                "latest_product_name": products[0].get("name") if products else None,
                "detail": NO_DEFORMATION_PRODUCT_NOTE,
            }

    if cache is not None:
        cache.setdefault("entries", {})[cell] = {"fetched_utc": now, "context": context}
    context = dict(context)
    context["cache_hit"] = False
    return context, True


# =======================================================================================
# Public API
# =======================================================================================
def get_insar_creep(
    lat: float,
    lon: float,
    *,
    slope_deg: Optional[float] = None,
    dynamic_fs: Optional[float] = None,
    elevation_m: float = 2000.0,
    hazard_type: str = "rainfall_landslide",
    rain_72h_mm: float = 0.0,
    saturation_ratio: float = 0.2,
    allow_proxy: bool = True,
    use_cache: bool = True,
    timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Ground-creep estimate for one site with honest provenance.

    Resolution order:
      1. A registered measured-velocity provider (LiCSAR / HyP3) -> ``MEASURED_INSAR``.
      2. The physics proxy, when ``slope_deg`` and ``dynamic_fs`` are supplied and
         ``allow_proxy`` is true -> ``MODELLED_PROXY``.
      3. Otherwise -> ``UNAVAILABLE`` with ``los_velocity_mm_month = None``.

    CDSE itself only ever contributes scene metadata under ``cdse``; it cannot upgrade the
    provenance, because it publishes no deformation product for Nepal.
    """
    cache = _load_cache() if use_cache else None
    context, performed_query = get_cdse_context(
        lat, lon, cache=cache, use_cache=use_cache, timeout_sec=timeout_sec
    )
    if cache is not None and performed_query:
        _save_cache(cache)

    measured = _invoke_measured_provider(lat, lon)
    if measured is not None:
        observation = measured
    elif allow_proxy and slope_deg is not None and dynamic_fs is not None:
        observation = compute_creep_proxy(
            slope_deg=float(slope_deg),
            dynamic_fs=float(dynamic_fs),
            elevation_m=float(elevation_m),
            hazard_type=hazard_type,
            rain_72h_mm=float(rain_72h_mm),
            saturation_ratio=float(saturation_ratio),
        )
    else:
        observation = {
            "los_velocity_mm_month": None,
            "provenance": PROVENANCE_UNAVAILABLE,
            "source": "no measured InSAR product and insufficient inputs for the proxy",
            "acquired_utc": None,
            "computed_utc": time.time(),
            "confidence": 0.0,
            "display_label": "Creep unavailable",
            "is_measurement": False,
        }

    observation["latitude"] = float(lat)
    observation["longitude"] = float(lon)
    observation["cdse"] = context
    observation["measured_velocity_path"] = (
        "registered provider"
        if measured is not None
        else "none registered (see register_measured_velocity_provider)"
    )
    return observation


def _site_field(site: Any, name: str, default: Any = None) -> Any:
    if isinstance(site, dict):
        return site.get(name, default)
    return getattr(site, name, default)


def get_insar_creep_batch(
    sites: Sequence[Any],
    *,
    context: Optional[Dict[str, Dict[str, float]]] = None,
    use_cache: bool = True,
    max_catalogue_queries: int = MAX_CATALOGUE_QUERIES_PER_BATCH,
    timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
    allow_proxy: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Creep estimates for the whole 491-site loop, keyed by ``site_id``.

    Sites are grouped onto a ``DEFAULT_CACHE_GRID_DEG`` cell so one catalogue query serves
    every site in that cell, the result is cached for the Sentinel-1 repeat window, and at
    most ``max_catalogue_queries`` HTTP queries are issued per call. ``sites`` accepts
    ``SiteInfo``-like objects or dicts; ``context`` optionally supplies per-site dynamic
    inputs (``dynamic_fs``, ``rain_72h_mm``, ``saturation_ratio``).
    """
    cache = _load_cache() if use_cache else None
    per_site_context = context or {}
    cell_contexts: Dict[str, Dict[str, Any]] = {}
    queries_used = 0
    cache_dirty = False
    results: Dict[str, Dict[str, Any]] = {}

    for site in sites:
        site_id = str(_site_field(site, "site_id", "") or "")
        lat = _site_field(site, "latitude")
        lon = _site_field(site, "longitude")
        if not site_id or lat is None or lon is None:
            continue

        cell = cache_cell_key(float(lat), float(lon))
        if cell not in cell_contexts:
            cell_context, performed_query = get_cdse_context(
                float(lat),
                float(lon),
                cache=cache,
                use_cache=use_cache,
                allow_network=queries_used < max(int(max_catalogue_queries), 0),
                timeout_sec=timeout_sec,
            )
            if performed_query:
                queries_used += 1
                cache_dirty = True
            cell_contexts[cell] = cell_context

        dynamic = per_site_context.get(site_id, {})
        slope_deg = _site_field(site, "slope_deg")
        dynamic_fs = dynamic.get("dynamic_fs")

        measured = _invoke_measured_provider(float(lat), float(lon))
        if measured is not None:
            observation = measured
        elif allow_proxy and slope_deg is not None and dynamic_fs is not None:
            observation = compute_creep_proxy(
                slope_deg=float(slope_deg),
                dynamic_fs=float(dynamic_fs),
                elevation_m=float(_site_field(site, "elevation_m", 2000.0) or 2000.0),
                hazard_type=str(_site_field(site, "hazard_type", "rainfall_landslide")),
                rain_72h_mm=float(dynamic.get("rain_72h_mm", 0.0)),
                saturation_ratio=float(dynamic.get("saturation_ratio", 0.2)),
            )
        else:
            observation = {
                "los_velocity_mm_month": None,
                "provenance": PROVENANCE_UNAVAILABLE,
                "source": "no measured InSAR product and insufficient inputs for the proxy",
                "acquired_utc": None,
                "computed_utc": time.time(),
                "confidence": 0.0,
                "display_label": "Creep unavailable",
                "is_measurement": False,
            }

        observation["latitude"] = float(lat)
        observation["longitude"] = float(lon)
        observation["cdse"] = cell_contexts[cell]
        observation["cache_cell"] = cell
        results[site_id] = observation

    if cache is not None and cache_dirty:
        _save_cache(cache)
    return results


def describe_insar_capability() -> Dict[str, Any]:
    """Machine-readable statement of what this module can and cannot measure.

    Intended for the dashboard provenance panel so the UI stops claiming "Sentinel-1 InSAR"
    for a modelled number.
    """
    return {
        "cdse_enabled": bool(settings.cdse_enabled),
        "cdse_credentials_present": bool(
            (settings.cdse_client_id or "").strip() and (settings.cdse_client_secret or "").strip()
        ),
        "catalogue_endpoint": _odata_products_url(),
        "configured_catalogue_url": settings.cdse_catalogue_url,
        "measured_velocity_provider_registered": _measured_velocity_provider is not None,
        "cdse_offers_deformation_product": False,
        "available_sentinel1_collections": [
            "sentinel-1-slc",
            "sentinel-1-slc-burst",
            "sentinel-1-grd",
            "sentinel-1-ocn",
            "sentinel-1-etad",
            "sentinel-1-global-mosaics",
        ],
        "limitation": NO_DEFORMATION_PRODUCT_NOTE,
        "recommended_measured_sources": [
            "COMET-LiCSAR / LiCSBAS frame velocities (CEDA) - covers the Alpine-Himalayan belt",
            "ASF HyP3 on-demand Sentinel-1 InSAR (global)",
            "ESA/Copernicus EGMS - EEA39 only, does not cover Nepal",
        ],
        "default_display_label": "Modelled creep (physics proxy)",
    }
