"""ASF HyP3 on-demand Sentinel-1 InSAR — the only route to a real MEASURED_INSAR velocity.

WHY THIS EXISTS
    `insar_cdse.py` deliberately refuses to call anything MEASURED_INSAR on the strength of
    a scene listing: CDSE publishes SLC/GRD scenes only and has no deformation product, and
    COMET-LiCSAR's public tree 404s on its interferogram directories (its Nepal frames are
    also 9-27 months stale). EGMS is Europe-only. ASF HyP3 processes Sentinel-1 bursts into
    real unwrapped interferograms on demand, which is what an actual LOS velocity requires.

    This module plugs into `insar_cdse.register_measured_velocity_provider()`. That hook
    validates that a provider returned both a numeric velocity AND a product reference
    before provenance is upgraded, so a measurement can only ever be backed by a real
    downloaded product.

VERIFIED AGAINST THE LIVE API (2026-10-26, account nepal_mhews)
    * GET  /user            -> application_status APPROVED, 8000 credits/month (free tier).
    * GET  /costs           -> INSAR_ISCE_BURST costs **1 credit**; INSAR_GAMMA costs 10-15.
                               Burst jobs are ~10x cheaper, so this module uses them.
    * POST /jobs            -> job_parameters.granules must be EXACTLY 2 burst granule names
                               matching ^S1_\\d{6}_IW.*-BURST (43 chars). Optional `looks`
                               (20x4 | 10x2 | 5x1) and `apply_water_mask`.
                               `validate_only: true` checks a payload for free.
    * Polarization          -> the API rejects cross-pol: "Only VV and HH polarizations are
                               currently supported, got: VH". Co-pol filtering is mandatory.
    * CMR burst discovery   -> short_name=SENTINEL-1_BURSTS is OPEN (no credentials) and
                               current; a 0.05 deg box over Langtang returned a 20-deep
                               VV stack on a clean 12-day repeat.

COHERENCE REALITY — MEASURED, NOT ASSUMED (job dcf84b0a, Langtang, 12-day VV pair)
    A real INSAR_ISCE_BURST product over the Langtang area was processed and inspected.
    Only **27.3%** of the scene was usable (coherence >= 0.3 AND successfully unwrapped).
    Usable fraction by elevation band:

        0-1500 m :  3.5%   dense vegetation/agriculture decorrelates over 12 days
        1500-3000 m: 26.9%
        3000-4500 m: 43.6%   best band - sparse cover, bare rock
        4500+ m    : 19.4%   snow and ice decorrelate

    So a measured velocity will frequently be UNAVAILABLE, and most often exactly at the
    >4500 m cryosphere sites this project cares about for rock-ice avalanches. Both
    `langtang_lirung` and the 28.22/85.52 test point returned coherence 0.000 with
    conncomp 0 - genuinely unmeasurable, correctly reported as None rather than as a
    number. Plan for InSAR to corroborate *some* sites, never all of them.

WHAT THIS MODULE WILL NOT DO
    It never invents a velocity. Every path that cannot produce a real, downloaded,
    pixel-sampled measurement returns None so the caller falls back to the clearly-labelled
    physics proxy. Jobs take ~1 hour, so submission and collection are separate phases with
    state persisted to disk; nothing blocks a caller for an hour.

CADENCE
    Designed for the WEEKLY cron next to the AI audit, never the 30-minute physics loop.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings

logger = logging.getLogger("nepal_mhews.insar_hyp3")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
STATE_PATH = DATA_DIR / "hyp3_insar_state.json"
PRODUCT_DIR = DATA_DIR / "hyp3_products"
STATE_VERSION = 1

JOB_TYPE = "INSAR_ISCE_BURST"
JOB_CREDIT_COST = 1  # from GET /costs, verified live
CO_POLARIZATIONS = ("VV", "HH")  # the API rejects VH/HV

# S1_039571_IW1_20261023T001859_VV_33FF-BURST
BURST_RE = re.compile(
    r"^S1_(?P<burst>\d{6})_(?P<swath>IW\d)_(?P<acq>\d{8}T\d{6})_(?P<pol>VV|VH|HH|HV)_[0-9A-F]{4}-BURST$"
)

# Sentinel-1 C-band carrier wavelength (m). Used only if a product exposes unwrapped phase
# in radians rather than a ready-made displacement band; see _phase_to_displacement_m.
SENTINEL1_WAVELENGTH_M = 0.055465763

# Sentinel-1 repeat. Pairs shorter than this are not independent acquisitions; much longer
# baselines decorrelate badly in vegetated Himalayan terrain.
REPEAT_DAYS = 12
MIN_PAIR_DAYS = 6
MAX_PAIR_DAYS = 60

# A velocity measured over a decorrelated patch is noise wearing a number's clothes.
MIN_COHERENCE = 0.3
MIN_VALID_FRACTION = 0.25
PATCH_PX = 11

HTTP_TIMEOUT_S = 90


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _auth_headers() -> Optional[Dict[str, str]]:
    token = (getattr(settings, "earthdata_token", "") or "").strip()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _http_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = HTTP_TIMEOUT_S,
) -> Optional[Any]:
    """GET/POST JSON. Returns None on any failure - callers degrade, never crash."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = dict(headers or {})
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        logger.warning("HyP3/CMR HTTP %s for %s: %s", exc.code, url.split("?")[0], detail)
    except Exception as exc:
        logger.warning("HyP3/CMR request failed (%s): %s", type(exc).__name__, exc)
    return None


def _parse_acq(acq: str) -> Optional[datetime]:
    try:
        return datetime.strptime(acq, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return None if dt is None else dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# state (submission and collection are hours apart, across process restarts)
# ---------------------------------------------------------------------------
def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        if isinstance(state, dict) and state.get("version") == STATE_VERSION:
            return state
        if isinstance(state, dict) and state.get("version") != STATE_VERSION:
            # Starting fresh also discards this month's credit accounting and every
            # in-flight job, so it must never happen quietly.
            logger.warning(
                "HyP3 state at %s has version %r, expected %r - starting from empty. "
                "In-flight jobs and credit accounting for this period are lost.",
                STATE_PATH, state.get("version"), STATE_VERSION,
            )
    except (OSError, ValueError):
        pass
    return {"version": STATE_VERSION, "jobs": {}, "credits": {}, "baselines": {}}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        tmp.replace(STATE_PATH)
    except Exception as exc:
        logger.warning("Could not persist HyP3 state: %s", exc)


def _credit_period() -> str:
    """HyP3 Basic grants credits per calendar month."""
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def credits_spent_this_period(state: Optional[Dict[str, Any]] = None) -> int:
    state = state if state is not None else _load_state()
    return int((state.get("credits") or {}).get(_credit_period(), 0))


def _record_credit_spend(state: Dict[str, Any], amount: int) -> None:
    credits = state.setdefault("credits", {})
    period = _credit_period()
    credits[period] = int(credits.get(period, 0)) + int(amount)
    # keep only the last few periods so the file cannot grow without bound
    for stale in sorted(credits)[:-6]:
        credits.pop(stale, None)


def site_key(lat: float, lon: float) -> str:
    return f"{float(lat):.4f},{float(lon):.4f}"


# ---------------------------------------------------------------------------
# burst discovery (CMR is open - no credentials needed)
# ---------------------------------------------------------------------------
@dataclass
class BurstStack:
    burst_id: str
    swath: str
    polarization: str
    acquisitions: List[Tuple[datetime, str]]  # (time, granule), newest first

    @property
    def depth(self) -> int:
        return len(self.acquisitions)


def find_burst_stacks(
    lat: float,
    lon: float,
    *,
    half_box_deg: float = 0.025,
    page_size: int = 200,
    timeout: int = HTTP_TIMEOUT_S,
) -> List[BurstStack]:
    """Find co-polarized Sentinel-1 burst stacks covering a point, newest first.

    An interferometric pair must share the same relative burst id, swath and polarization;
    only the acquisition date differs. Cross-pol (VH/HV) is filtered out because the HyP3
    API rejects it outright.
    """
    bbox = (
        f"{float(lon) - half_box_deg:.4f},{float(lat) - half_box_deg:.4f},"
        f"{float(lon) + half_box_deg:.4f},{float(lat) + half_box_deg:.4f}"
    )
    query = urllib.parse.urlencode(
        {
            "short_name": "SENTINEL-1_BURSTS",
            "bounding_box": bbox,
            "page_size": str(page_size),
            "sort_key": "-start_date",
        }
    )
    url = f"{settings.cmr_granule_search_url}?{query}"
    payload = _http_json(url, timeout=timeout)
    if not payload:
        return []

    entries = (payload.get("feed") or {}).get("entry") or []
    grouped: Dict[Tuple[str, str, str], List[Tuple[datetime, str]]] = collections.defaultdict(list)
    for entry in entries:
        name = entry.get("producer_granule_id") or entry.get("title") or ""
        match = BURST_RE.match(name)
        if not match or match.group("pol") not in CO_POLARIZATIONS:
            continue
        acq = _parse_acq(match.group("acq"))
        if acq is None:
            continue
        grouped[(match.group("burst"), match.group("swath"), match.group("pol"))].append(
            (acq, name)
        )

    stacks: List[BurstStack] = []
    for (burst_id, swath, pol), items in grouped.items():
        items.sort(key=lambda it: it[0], reverse=True)
        stacks.append(BurstStack(burst_id, swath, pol, items))
    stacks.sort(key=lambda s: s.depth, reverse=True)
    return stacks


def select_pair(
    stack: BurstStack,
    *,
    min_days: int = MIN_PAIR_DAYS,
    max_days: int = MAX_PAIR_DAYS,
    before: Optional[datetime] = None,
) -> Optional[Tuple[str, str, float]]:
    """Choose (reference, secondary, days_apart) with the shortest usable baseline.

    Shortest-first because coherence decays with temporal baseline, badly so over vegetated
    Himalayan slopes. `before` restricts the reference to acquisitions at or before a time,
    which is what a retrospective measurement around a known event needs.
    """
    acquisitions = stack.acquisitions
    if before is not None:
        acquisitions = [a for a in acquisitions if a[0] <= before]
    if len(acquisitions) < 2:
        return None

    reference_time, reference = acquisitions[0]
    best: Optional[Tuple[str, str, float]] = None
    for acq_time, granule in acquisitions[1:]:
        days = (reference_time - acq_time).total_seconds() / 86400.0
        if days < min_days or days > max_days:
            continue
        if best is None or days < best[2]:
            best = (reference, granule, round(days, 2))
    return best


# ---------------------------------------------------------------------------
# job submission / collection
# ---------------------------------------------------------------------------
def submit_insar_job(
    reference: str,
    secondary: str,
    *,
    name: str = "mhews-insar",
    looks: str = "20x4",
    apply_water_mask: bool = False,
    validate_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Submit one INSAR_ISCE_BURST job (1 credit). Returns the job record or None."""
    headers = _auth_headers()
    if headers is None:
        logger.info("No EARTHDATA_TOKEN configured; cannot submit HyP3 jobs")
        return None
    for granule in (reference, secondary):
        if not BURST_RE.match(granule):
            logger.warning("Refusing to submit malformed burst granule: %r", granule)
            return None

    body = {
        "validate_only": bool(validate_only),
        "jobs": [
            {
                "job_type": JOB_TYPE,
                "name": name[:100],
                "job_parameters": {
                    "granules": [reference, secondary],
                    "looks": looks,
                    "apply_water_mask": bool(apply_water_mask),
                },
            }
        ],
    }
    result = _http_json(f"{settings.hyp3_api_url}/jobs", headers=headers, payload=body)
    if not result:
        return None
    jobs = result.get("jobs") or []
    return jobs[0] if jobs else None


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    headers = _auth_headers()
    if headers is None:
        return None
    return _http_json(f"{settings.hyp3_api_url}/jobs/{job_id}", headers=headers)


def remaining_credits() -> Optional[float]:
    """Credits left this month according to HyP3 itself (authoritative over local state)."""
    headers = _auth_headers()
    if headers is None:
        return None
    user = _http_json(f"{settings.hyp3_api_url}/user", headers=headers)
    if not user:
        return None
    try:
        return float(user.get("remaining_credits"))
    except (TypeError, ValueError):
        return None


def _budget_allows(state: Dict[str, Any], jobs_wanted: int) -> int:
    """How many jobs may still be submitted, by the stricter of local and remote budget."""
    budget = int(getattr(settings, "hyp3_monthly_credit_budget", 8000))
    per_run = int(getattr(settings, "hyp3_max_jobs_per_run", 20))
    local_left = max(0, budget - credits_spent_this_period(state)) // JOB_CREDIT_COST
    allowed = min(jobs_wanted, per_run, local_left)

    remote = remaining_credits()
    if remote is not None:
        allowed = min(allowed, int(remote // JOB_CREDIT_COST))
    return max(0, allowed)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def _phase_to_displacement_m(unwrapped_phase: "Any") -> "Any":
    """Convert unwrapped interferometric phase (radians) to LOS displacement (metres).

    d_los = +(lambda / 4*pi) * dphi

    The sign follows HyP3's OWN documented convention for `*_unw_phase.tif`, quoted from
    the product README shipped with job dcf84b0a:

        "Positive values indicate movement away from the sensor (caused by subsidence or
         lateral movement away from the sensor) and negative values indicate movement
         towards the sensor (uplift or lateral movement towards the sensor)."

    So HyP3 has ALREADY applied the sign flip of the raw interferometric convention
    (dphi = -(4pi/lambda)*dr). Negating again here - the textbook formula applied to an
    already-corrected raster - silently inverts every velocity: downslope creep would be
    reported as uplift. The resulting number looks entirely plausible, which is what makes
    this class of error dangerous.

        positive d_los  => movement AWAY from the sensor  (subsidence / downslope creep)
        negative d_los  => movement TOWARD the sensor     (uplift)

    CAUTION - a single pair is not a deformation rate. The phase also contains atmospheric
    delay, which in high-relief terrain is routinely centimetres and does not average out
    within one interferogram. The value returned here is the apparent LOS change over this
    specific pair, extrapolated to mm/month for comparability - NOT a validated velocity.
    A trustworthy rate needs a stack of pairs with time-series inversion (LiCSBAS-style).
    Treat a single-pair figure as a screening indicator, not as ground truth.
    """
    return (SENTINEL1_WAVELENGTH_M / (4.0 * math.pi)) * unwrapped_phase


def measure_los_velocity_from_product(
    product_dir: Path,
    lat: float,
    lon: float,
    days_apart: float,
    *,
    patch_px: int = PATCH_PX,
    min_coherence: float = MIN_COHERENCE,
) -> Optional[Dict[str, Any]]:
    """Sample a downloaded HyP3 interferogram at a point and convert to mm/month.

    Returns None - never a guess - when the product is missing, the point falls outside it,
    or too few pixels survive the coherence mask.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform
        from rasterio.windows import Window
    except ImportError as exc:
        logger.warning("rasterio/numpy unavailable for InSAR measurement: %s", exc)
        return None

    product_dir = Path(product_dir)
    unw = next(iter(sorted(product_dir.glob("*_unw_phase.tif"))), None)
    if unw is None:
        logger.warning("No *_unw_phase.tif in %s", product_dir)
        return None
    corr = next(iter(sorted(product_dir.glob("*_corr.tif"))), None)
    # ISCE's connected-component label. 0 means the unwrapper could not solve that pixel,
    # which is authoritative - more reliable than inferring failure from a zero phase.
    conncomp = next(iter(sorted(product_dir.glob("*_conncomp.tif"))), None)

    try:
        with rasterio.open(unw) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [float(lon)], [float(lat)])
            row, col = src.index(xs[0], ys[0])
            half = max(1, int(patch_px) // 2)
            if not (half <= row < src.height - half and half <= col < src.width - half):
                logger.info("Point (%.4f, %.4f) outside interferogram %s", lat, lon, unw.name)
                return None
            window = Window(col - half, row - half, 2 * half + 1, 2 * half + 1)
            phase = src.read(1, window=window).astype("float64")
            nodata = src.nodata

        valid = np.isfinite(phase)
        if nodata is not None:
            valid &= phase != nodata
        # HyP3 writes exact 0.0 where unwrapping failed; treating that as "no motion"
        # would silently report a stable slope for an unmeasurable one.
        valid &= phase != 0.0

        if conncomp is not None:
            with rasterio.open(conncomp) as ccsrc:
                cc = ccsrc.read(1, window=window)
            if cc.shape == phase.shape:
                valid &= cc > 0

        coherence_mean = None
        if corr is not None:
            with rasterio.open(corr) as csrc:
                coh = csrc.read(1, window=window).astype("float64")
            if coh.shape == phase.shape:
                valid &= coh >= float(min_coherence)
                if np.any(np.isfinite(coh)):
                    coherence_mean = float(np.nanmean(coh))

        total = int(phase.size)
        n_valid = int(np.count_nonzero(valid))
        if total == 0 or (n_valid / total) < MIN_VALID_FRACTION or n_valid < 4:
            logger.info(
                "Insufficient coherent pixels at (%.4f, %.4f): %d/%d", lat, lon, n_valid, total
            )
            return None

        median_phase = float(np.median(phase[valid]))
        displacement_m = float(_phase_to_displacement_m(median_phase))
        if days_apart <= 0:
            return None
        velocity_mm_month = displacement_m * 1000.0 * (30.0 / float(days_apart))

        return {
            "los_velocity_mm_month": round(velocity_mm_month, 3),
            "los_displacement_mm": round(displacement_m * 1000.0, 3),
            "unwrapped_phase_median_rad": round(median_phase, 4),
            "days_apart": round(float(days_apart), 2),
            "pixels_valid": n_valid,
            "pixels_total": total,
            "mean_coherence": None if coherence_mean is None else round(coherence_mean, 3),
            "product_file": unw.name,
        }
    except Exception as exc:
        logger.warning("Failed to measure InSAR product %s: %s", product_dir, exc)
        return None


def download_product(job: Dict[str, Any], dest_dir: Optional[Path] = None) -> Optional[Path]:
    """Download and unpack a finished job's product. Returns the extracted directory."""
    import zipfile

    files = job.get("files") or []
    if not files:
        return None
    url = files[0].get("url")
    if not url:
        return None

    dest_dir = Path(dest_dir or PRODUCT_DIR) / str(job.get("job_id"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "product.zip"
    try:
        # The product URL is pre-signed; no Authorization header is needed or wanted.
        with urllib.request.urlopen(url, timeout=600) as resp, open(archive, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
        archive.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Could not download HyP3 product for %s: %s", job.get("job_id"), exc)
        return None

    inner = [p for p in dest_dir.iterdir() if p.is_dir()]
    return inner[0] if len(inner) == 1 else dest_dir


# ---------------------------------------------------------------------------
# the two public phases: request, then collect
# ---------------------------------------------------------------------------
def request_measurement(
    lat: float,
    lon: float,
    *,
    before: Optional[datetime] = None,
    name: str = "mhews-insar",
) -> Dict[str, Any]:
    """Phase 1: submit an InSAR job for a point, if budget and imagery allow.

    Returns a status dict. Never blocks waiting for the ~1 hour of processing.
    """
    if not getattr(settings, "hyp3_enabled", False):
        return {"status": "DISABLED"}
    if _auth_headers() is None:
        return {"status": "NO_CREDENTIALS"}

    state = _load_state()
    key = site_key(lat, lon)
    existing = (state.get("jobs") or {}).get(key)
    if existing and existing.get("status_code") in ("PENDING", "RUNNING"):
        return {"status": "ALREADY_PENDING", "job_id": existing.get("job_id")}

    if _budget_allows(state, 1) < 1:
        return {"status": "BUDGET_EXHAUSTED", "spent": credits_spent_this_period(state)}

    stacks = find_burst_stacks(lat, lon)
    if not stacks:
        return {"status": "NO_BURSTS"}
    for stack in stacks:
        pair = select_pair(stack, before=before)
        if not pair:
            continue
        reference, secondary, days = pair
        job = submit_insar_job(reference, secondary, name=name)
        if not job:
            continue
        _record_credit_spend(state, JOB_CREDIT_COST)
        state.setdefault("jobs", {})[key] = {
            "job_id": job.get("job_id"),
            "status_code": job.get("status_code", "PENDING"),
            "submitted_utc": time.time(),
            "reference": reference,
            "secondary": secondary,
            "days_apart": days,
            "lat": float(lat),
            "lon": float(lon),
        }
        _save_state(state)
        return {
            "status": "SUBMITTED",
            "job_id": job.get("job_id"),
            "days_apart": days,
            "credits_spent_this_period": credits_spent_this_period(state),
        }
    return {"status": "NO_USABLE_PAIR"}


def collect_measurements(limit: int = 50) -> Dict[str, Any]:
    """Phase 2: poll submitted jobs, download finished ones, store real velocities."""
    if not getattr(settings, "hyp3_enabled", False):
        return {"status": "DISABLED", "collected": 0}
    if _auth_headers() is None:
        return {"status": "NO_CREDENTIALS", "collected": 0}

    state = _load_state()
    jobs = state.get("jobs") or {}
    summary = {"checked": 0, "collected": 0, "pending": 0, "failed": 0}

    for key, record in list(jobs.items())[:limit]:
        if record.get("status_code") == "SUCCEEDED" and record.get("measured"):
            continue
        job_id = record.get("job_id")
        if not job_id:
            continue
        summary["checked"] += 1
        job = get_job(job_id)
        if not job:
            continue
        status = job.get("status_code")
        record["status_code"] = status
        if status in ("PENDING", "RUNNING"):
            summary["pending"] += 1
            continue
        if status != "SUCCEEDED":
            summary["failed"] += 1
            continue

        product_dir = download_product(job)
        if product_dir is None:
            summary["failed"] += 1
            continue
        measurement = measure_los_velocity_from_product(
            product_dir, record["lat"], record["lon"], record.get("days_apart") or REPEAT_DAYS
        )
        if measurement is None:
            # Downloaded but unmeasurable (decorrelated / outside footprint). Recording the
            # failure stops it being retried forever, and still yields no fake number.
            record["measured"] = None
            record["measured_utc"] = time.time()
            summary["failed"] += 1
            continue

        measurement["job_id"] = job_id
        measurement["reference_granule"] = record.get("reference")
        measurement["secondary_granule"] = record.get("secondary")
        measurement["measured_utc"] = time.time()
        record["measured"] = measurement
        record["measured_utc"] = measurement["measured_utc"]
        state.setdefault("baselines", {})[key] = measurement
        summary["collected"] += 1

    _save_state(state)
    summary["status"] = "OK"
    return summary


# ---------------------------------------------------------------------------
# the provider that insar_cdse consumes
# ---------------------------------------------------------------------------
def hyp3_measured_velocity_provider(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Return a REAL measured LOS velocity for a point, or None.

    Shape matches what `insar_cdse._invoke_measured_provider` validates: a numeric velocity
    plus a product reference. Reads only already-collected measurements - it never blocks on
    processing, and never fabricates a value when none has been measured.
    """
    if not getattr(settings, "hyp3_enabled", False):
        return None
    state = _load_state()
    measurement = (state.get("baselines") or {}).get(site_key(lat, lon))
    if not measurement:
        return None

    max_age_days = float(getattr(settings, "hyp3_baseline_max_age_days", 30))
    measured_utc = float(measurement.get("measured_utc") or 0.0)
    age_days = (time.time() - measured_utc) / 86400.0 if measured_utc else None
    if age_days is not None and age_days > max_age_days:
        # A stale baseline is still a real measurement, but it is not current; surfacing it
        # as today's creep would misrepresent when it was observed.
        return None

    return {
        "los_velocity_mm_month": measurement["los_velocity_mm_month"],
        "product_id": measurement.get("job_id"),
        "product_reference": (
            f"ASF HyP3 {JOB_TYPE} {measurement.get('job_id')} "
            f"[{measurement.get('reference_granule')} / {measurement.get('secondary_granule')}]"
        ),
        "source": "ASF HyP3 on-demand Sentinel-1 burst InSAR",
        "acquired_utc": measured_utc,
        "observation_days": measurement.get("days_apart"),
        "mean_coherence": measurement.get("mean_coherence"),
        "pixels_valid": measurement.get("pixels_valid"),
        # "measured" means a real product was sampled - NOT that this is a validated
        # deformation rate. A single interferogram carries atmospheric delay that only a
        # multi-pair time series removes, so the caller must not present this as a
        # settled velocity. See _phase_to_displacement_m for the full caveat.
        "confidence": "measured_single_pair",
        "caveat": (
            "single 12-day interferogram; includes atmospheric delay. Screening "
            "indicator, not a validated deformation rate (needs time-series inversion)."
        ),
        "age_days": None if age_days is None else round(age_days, 2),
    }


def register_with_insar_cdse() -> bool:
    """Install this provider so insar_cdse can report MEASURED_INSAR. Safe to call twice."""
    try:
        from app.ingestion.insar_cdse import register_measured_velocity_provider

        register_measured_velocity_provider(hyp3_measured_velocity_provider)
        return True
    except Exception as exc:
        logger.warning("Could not register HyP3 velocity provider: %s", exc)
        return False


def describe_capability() -> Dict[str, Any]:
    """What this module can actually do right now - for dashboards and audit reports."""
    state = _load_state()
    return {
        "enabled": bool(getattr(settings, "hyp3_enabled", False)),
        "credentials_present": _auth_headers() is not None,
        "job_type": JOB_TYPE,
        "credit_cost_per_job": JOB_CREDIT_COST,
        "monthly_budget": int(getattr(settings, "hyp3_monthly_credit_budget", 8000)),
        "credits_spent_this_period": credits_spent_this_period(state),
        "tracked_sites": len(state.get("jobs") or {}),
        "measured_baselines": len(state.get("baselines") or {}),
    }
