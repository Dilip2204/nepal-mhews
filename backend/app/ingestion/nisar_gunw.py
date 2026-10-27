"""NISAR L-band deformation from pre-computed GUNW products, read by HTTP range.

What this adds that Sentinel-1 cannot
-------------------------------------
The C-band pipeline's binding constraint is decorrelation, not cloud and not budget.
Measured over the real network at glc_105: 27 of 90 jobs failed on a catalogue gap, 9
more were genuinely unmeasurable, and usable coherence by elevation ran 3.5% below
1500 m, 43.6% at 3000-4500 m and 19.4% above 4500 m. At `langtang_lirung` (5680 m) the
C-band coherence was **0.000** with `conncomp 0` - correctly reported as no measurement
at all.

The same site, read from a NISAR L-band GUNW:

    coherenceMagnitude   0.5498      121/121 pixels valid
    connectedComponents  1           successfully unwrapped
    unwrappedPhase       0.1859 rad

24 cm penetrates the snow and vegetation that 5.5 cm scatters off. This is the
difference between a site we cannot measure and one we can.

Three further things fall out of using NISAR:

* **No processing and no credits.** NASA distributes Level-2 Geocoded Unwrapped
  Interferograms directly, so there is no HyP3 job, no 8000-credit monthly ceiling and
  no hour-long wait. ASF holds 378 GUNW granules over Nepal.
* **The monsoon.** glc_105's C-band network lost June-October to a burst-catalogue hole,
  leaving a winter-only solution. Its NISAR coverage spans 2026-06-27 to 2026-10-05 on
  an 8-day repeat.
* **An ionospheric correction.** L-band suffers far more ionospheric delay than C-band,
  and the GUNW ships `ionospherePhaseScreen` to remove it - a correction the C-band path
  has no equivalent of.

The cost is sensitivity: one phase cycle is 119.2 mm of line-of-sight displacement at
L-band against 27.7 mm at C-band, so NISAR is roughly 4.3x coarser per fringe. The trade
is coarser-but-real against finer-but-absent.

Reading without downloading
---------------------------
A GUNW granule is ~2.4 GB and a site needs a handful of pixels from it. GDAL reads the
HDF5 subdatasets over ``/vsicurl`` with HTTP range requests, so a measurement costs a few
hundred kilobytes rather than gigabytes. Two things make that work and are easy to get
wrong:

* Earthdata 302-redirects to a pre-signed S3 URL. Forwarding the ``Authorization`` header
  to S3 gets a 403, because the signature is already in the query string.
* GDAL does **not** pick up a CRS or geotransform from these subdatasets. Geolocation
  lives in sibling ``xCoordinates`` / ``yCoordinates`` arrays, and the projection is
  determined here by testing which UTM zone places the target inside that grid - derived
  from the file, never assumed from longitude.
"""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger("nepal_mhews.nisar")

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
GUNW_COLLECTIONS = (
    "NISAR_L2_GUNW_PROVISIONAL_V1",
    "NISAR_L2_GUNW_BETA_V1",
)

# NISAR L-band centre frequency 1257.5 MHz -> 0.2385 m. One 2*pi cycle is 119.2 mm of
# line-of-sight displacement, against 27.7 mm for Sentinel-1 C-band.
NISAR_WAVELENGTH_M = 0.2385

GRID = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
LAYER_PHASE = f"{GRID}/HH/unwrappedPhase"
LAYER_COHERENCE = f"{GRID}/HH/coherenceMagnitude"
LAYER_CONNCOMP = f"{GRID}/HH/connectedComponents"
LAYER_IONOSPHERE = f"{GRID}/HH/ionospherePhaseScreen"
LAYER_X = f"{GRID}/xCoordinates"
LAYER_Y = f"{GRID}/yCoordinates"

# UTM zones that can contain Nepal (80-88.3 E). The right one is chosen by testing which
# places the target inside the granule's own coordinate arrays.
CANDIDATE_EPSG = (32644, 32645, 32646)

PATCH_PX = 11                 # ~880 m at the 80 m unwrapped posting
MIN_COHERENCE = 0.3
REFERENCE_INNER_M = 600.0
REFERENCE_OUTER_M = 2500.0
MIN_TARGET_PIXELS = 8
MIN_REFERENCE_PIXELS = 40


def _auth_token() -> str:
    return (os.environ.get("EARTHDATA_TOKEN")
            or getattr(settings, "earthdata_token", "") or "").strip()


def _gdal_env() -> Dict[str, str]:
    """GDAL options for authenticated, range-read access to the NISAR archive."""
    return {
        "GDAL_HTTP_HEADERS": f"Authorization: Bearer {_auth_token()}",
        # A HEAD to the redirect target is answered without the signature and fails, so
        # skip it and let the ranged GET do the work.
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".h5",
    }


def search_gunw(
    lat: float, lon: float, *, since_days: int = 400, max_granules: int = 200
) -> List[Dict[str, Any]]:
    """GUNW granules covering a point, newest first. Open CMR, no credentials needed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(since_days))
    half = 0.02
    out: List[Dict[str, Any]] = []
    seen = set()
    for collection in GUNW_COLLECTIONS:
        query = urllib.parse.urlencode({
            "short_name": collection,
            "bounding_box": (f"{lon-half:.4f},{lat-half:.4f},{lon+half:.4f},{lat+half:.4f}"),
            "page_size": str(max_granules),
            "sort_key": "-start_date",
        })
        try:
            request = urllib.request.Request(
                f"{CMR_GRANULES}?{query}", headers={"User-Agent": "nepal-mhews/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
        except Exception as exc:
            logger.warning("NISAR CMR search failed for %s: %s", collection, exc)
            continue

        for item in payload.get("items", []):
            umm = item.get("umm") or {}
            name = umm.get("GranuleUR")
            if not name or name in seen:
                continue
            times = (umm.get("TemporalExtent") or {}).get("RangeDateTime") or {}
            start, end = times.get("BeginningDateTime"), times.get("EndingDateTime")
            if not (start and end):
                continue
            try:
                t_a = datetime.fromisoformat(start.replace("Z", "+00:00"))
                t_b = datetime.fromisoformat(end.replace("Z", "+00:00"))
            except ValueError:
                continue
            if t_b < cutoff:
                continue
            urls = [r.get("URL") for r in (umm.get("RelatedUrls") or [])
                    if r.get("Type") == "GET DATA" and str(r.get("URL", "")).endswith(".h5")]
            if not urls:
                continue
            seen.add(name)
            out.append({
                "granule": name, "url": urls[0], "collection": collection,
                # GUNW is an interferogram: these are the two acquisition times it spans.
                "t_early": min(t_a, t_b), "t_late": max(t_a, t_b),
            })
    out.sort(key=lambda g: g["t_late"], reverse=True)
    return out


def _read_axis(url: str, layer: str):
    import numpy as np
    import rasterio

    with rasterio.Env(**_gdal_env()):
        with rasterio.open(f'HDF5:"/vsicurl/{url}"://{layer}') as src:
            return np.atleast_1d(src.read(1)).ravel()


def locate(url: str, lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Pixel position of a point in this granule, plus the CRS it was found in.

    The projection is *derived*: each candidate UTM zone is tested against the granule's
    own coordinate arrays and the one that places the target inside the grid wins. Taking
    the zone from longitude would be an assumption, and a granule straddling a zone
    boundary would silently land in the wrong one.
    """
    from rasterio.warp import transform as warp_transform

    try:
        xs = _read_axis(url, LAYER_X)
        ys = _read_axis(url, LAYER_Y)
    except Exception as exc:
        logger.warning("could not read NISAR coordinate arrays: %s", exc)
        return None
    if xs.size < 2 or ys.size < 2:
        return None

    for epsg in CANDIDATE_EPSG:
        try:
            X, Y = warp_transform("EPSG:4326", f"EPSG:{epsg}", [float(lon)], [float(lat)])
        except Exception:
            continue
        x, y = X[0], Y[0]
        if not (min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)):
            continue
        dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
        if dx == 0 or dy == 0:
            continue
        col = int(round((x - float(xs[0])) / dx))
        row = int(round((y - float(ys[0])) / dy))
        if not (0 <= row < ys.size and 0 <= col < xs.size):
            continue
        return {"epsg": epsg, "row": row, "col": col,
                "res_m": abs(dx), "height": int(ys.size), "width": int(xs.size)}
    return None


def phase_to_displacement_mm(phase_rad: float) -> float:
    """LOS displacement in mm from unwrapped phase, at NISAR's L-band wavelength.

    SIGN NOT YET VERIFIED against a NISAR product README, unlike the C-band path where
    HyP3's own documentation was quoted. The magnitude is right; the direction is the
    textbook convention and has to be confirmed before any NISAR velocity is published as
    a measurement. Getting this wrong reports downslope creep as uplift at a plausible
    magnitude, which is exactly the failure this codebase has already had once.
    """
    return float(NISAR_WAVELENGTH_M / (4.0 * math.pi) * float(phase_rad) * 1000.0)


def measure_double_difference(
    url: str, lat: float, lon: float, *,
    patch_px: int = PATCH_PX, min_coherence: float = MIN_COHERENCE,
    inner_m: float = REFERENCE_INNER_M, outer_m: float = REFERENCE_OUTER_M,
    apply_ionosphere: bool = True,
) -> Optional[Dict[str, Any]]:
    """Target-minus-reference-annulus LOS displacement from one GUNW, or None.

    Double differencing is as necessary here as for C-band: an unwrapped interferogram
    carries an arbitrary constant, so only differences within one product are comparable
    across a stack.
    """
    import numpy as np
    import rasterio

    where = locate(url, lat, lon)
    if where is None:
        return None
    res = where["res_m"]
    rad_px = int(math.ceil(outer_m / res))
    r0, r1 = where["row"] - rad_px, where["row"] + rad_px + 1
    c0, c1 = where["col"] - rad_px, where["col"] + rad_px + 1
    if r0 < 0 or c0 < 0 or r1 > where["height"] or c1 > where["width"]:
        logger.debug("NISAR window falls outside the granule at (%.4f, %.4f)", lat, lon)
        return None
    window = ((r0, r1), (c0, c1))

    def read(layer):
        with rasterio.Env(**_gdal_env()):
            with rasterio.open(f'HDF5:"/vsicurl/{url}"://{layer}') as src:
                return src.read(1, window=window).astype("float64")

    try:
        phase = read(LAYER_PHASE)
        coherence = read(LAYER_COHERENCE)
        conncomp = read(LAYER_CONNCOMP)
    except Exception as exc:
        logger.warning("NISAR range read failed at (%.4f, %.4f): %s", lat, lon, exc)
        return None

    valid = np.isfinite(phase) & (phase != 0.0)
    valid &= np.isfinite(coherence) & (coherence >= float(min_coherence))
    # conncomp 0 is ISCE's authoritative "unwrapping failed here" flag.
    valid &= np.isfinite(conncomp) & (conncomp > 0)

    ionosphere_removed = False
    if apply_ionosphere:
        try:
            iono = read(LAYER_IONOSPHERE)
            if iono.shape == phase.shape and np.isfinite(iono).any():
                # L-band ionospheric delay is large; the GUNW ships the screen to remove
                # it, and the C-band path has no equivalent.
                phase = phase - np.nan_to_num(iono, nan=0.0)
                ionosphere_removed = True
        except Exception as exc:
            logger.debug("ionosphere screen unavailable: %s", exc)

    ny, nx = phase.shape
    cy, cx = where["row"] - r0, where["col"] - c0
    yy, xx = np.ogrid[:ny, :nx]
    dist_m = np.sqrt(((yy - cy) * res) ** 2 + ((xx - cx) * res) ** 2)
    half = max(1, int(patch_px) // 2)

    target_mask = valid & (np.abs(yy - cy) <= half) & (np.abs(xx - cx) <= half)
    ref_mask = valid & (dist_m >= inner_m) & (dist_m <= outer_m)
    n_t = int(np.count_nonzero(target_mask))
    n_r = int(np.count_nonzero(ref_mask))
    if n_t < MIN_TARGET_PIXELS or n_r < MIN_REFERENCE_PIXELS:
        return None

    dd_phase = float(np.median(phase[target_mask]) - np.median(phase[ref_mask]))
    return {
        "displacement_mm": round(phase_to_displacement_mm(dd_phase), 4),
        "dd_phase_rad": round(dd_phase, 6),
        "mean_coherence": round(float(np.mean(coherence[target_mask])), 4),
        "target_pixels": n_t,
        "reference_pixels": n_r,
        "ionosphere_corrected": ionosphere_removed,
        "epsg": where["epsg"],
        "resolution_m": res,
        "band": "L",
        "wavelength_m": NISAR_WAVELENGTH_M,
        "sign_convention_verified": False,
        "source": "NISAR L2 GUNW (NASA/ISRO), range-read over /vsicurl",
    }


def describe_capability() -> Dict[str, Any]:
    """Whether NISAR can be used right now, and what is missing if not."""
    token = _auth_token()
    return {
        "credentials_present": bool(token),
        "collections": list(GUNW_COLLECTIONS),
        "wavelength_m": NISAR_WAVELENGTH_M,
        "mm_per_cycle": round(NISAR_WAVELENGTH_M / 2.0 * 1000.0, 1),
        "costs_credits": False,
        "blocker": None if token else "no EARTHDATA_TOKEN configured",
        "sign_convention_verified": False,
    }
