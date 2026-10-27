"""Small-baseline (SBAS-style) InSAR time-series inversion — a real velocity, not a screen.

WHY A SINGLE PAIR IS NOT ENOUGH
    One interferogram's phase is deformation + atmospheric delay + orbital/DEM residual +
    noise. In Himalayan relief the atmospheric term is routinely centimetres, which is the
    same order as a year of real creep. A single-pair "velocity" therefore mostly measures
    weather. Averaging a redundant network of pairs suppresses the atmospheric term, because
    it is uncorrelated between acquisition dates while deformation accumulates monotonically.

TWO THINGS THAT MAKE THIS CORRECT RATHER THAN MERELY PLAUSIBLE

    1. DOUBLE DIFFERENCING IS MANDATORY.
       HyP3's product README states the unwrapping "reference point was set internally by
       SNAPHU in a manner that minimizes the statistical cost of the resulting unwrapped
       solution" - i.e. every interferogram carries its own arbitrary, undocumented phase
       offset. Absolute phase is NOT comparable across pairs. Stacking raw values would
       sum a pile of unrelated constants and produce a confident, meaningless number.
       Every observation here is (target - reference_area) within the SAME interferogram,
       which cancels that unknown offset and also removes any atmospheric signal common to
       both locations.

    2. THE NETWORK IS INVERTED, NOT AVERAGED.
       With acquisitions t_0..t_N and M interferograms, each pair (a -> b) observes
       d_b - d_a. That is a linear system G*d = obs, solved by least squares with d_0 = 0
       as the datum. The residual of that fit is the key honesty metric: a redundant network
       of internally consistent observations yields small residuals, while unwrapping errors
       or atmospheric outliers show up as large ones. A velocity is only reported when the
       network is genuinely redundant and self-consistent.

WHAT IS STILL NOT CLAIMED
    The RAW velocity from invert_timeseries() is a LOS (line-of-sight) rate, relative to the
    chosen reference area rather than absolute. project_los_to_slope_parallel() converts this
    to a downslope rate GIVEN an assumption (pure slope-parallel motion - reasonable for
    creep/landslides) and real look-vector geometry from the product; it is withheld rather
    than reported when the track's geometry barely senses the slope's direction.
    Ascending+descending combination (two tracks, two independent slope-parallel estimates,
    cross-checked for agreement) is implemented but not yet exercised on a second live
    network for any site - submitting one is the natural next step once useful.

OTHER COPERNICUS/NASA PROCESSING OPTIONS - CHECKED LIVE, 2026-10-26
    * NISAR (L-band) is NOT yet processable through HyP3. Its live job-type list is
      Sentinel-1-only: AUTORIFT, INSAR_GAMMA, RTC_GAMMA, INSAR_ISCE_BURST,
      INSAR_ISCE_MULTI_BURST, ARIA_S1_GUNW, OPERA_DISP_TMS - every granule pattern requires
      an "S1" prefix. NISAR L-band data itself has been publicly available via ASF/Earthdata
      since 2026-06-17 (the same account this module already authenticates against), but
      turning it into an interferogram needs a different, not-yet-available pipeline. This
      remains the single most promising lever for >4500m coverage once it exists - L-band's
      24cm wavelength penetrates snow/vegetation far better than Sentinel-1's 5.5cm C-band.
    * ARIA_S1_GUNW (60 credits/pair) and OPERA_DISP_TMS (10,001 credits - more than the
      ENTIRE monthly 8000-credit free tier) both exist as HyP3 job types but are not useful
      here: ARIA_S1_GUNW is 60x the cost of our burst approach for a standardized full-frame
      product we do not need, and OPERA_DISP_TMS's single-request cost alone exceeds the
      monthly budget.
    * openEO on CDSE has a REAL, separate InSAR capability: the `sentinel1_sar_interferogram`
      User-Defined Process, burst-level (compatible with our approach), publicly documented
      as of 2026-05-22 with a worked earthquake-deformation example published 2026-10-15.
      CDSE states 10,000 free compute credits/month for openEO - a SEPARATE pool from HyP3's
      8000, which could roughly double monthly network-building capacity. NOT yet integrated
      here: its exact API endpoint, auth flow, output band names/units and (critically) its
      phase sign convention have not been verified live the way HyP3's were, and every
      "confidently wrong" bug in this module so far came from an unverified convention
      (phase-to-displacement sign, resolution units, conncomp semantics). Building on it
      blind would repeat that mistake. Reference: algorithm-catalogue.apex.esa.int and
      github.com/cloudinsar/s1-workflows.
    * Atmospheric correction: GACOS is NOT a Copernicus service (it is run by COMET,
      Newcastle University) and did not resolve from this host. ERA5 via the Copernicus
      Climate Data Store IS reachable and has a documented API, but needs its own free
      registration and a non-trivial physical model (integrating temperature/humidity/
      pressure profiles into zenith delay, then projecting to LOS via the incidence angle)
      to turn into a phase correction - a bigger, separately-verifiable task. The
      ELEVATION-CORRELATED correction implemented below needs no new credentials or external
      data (it only uses the DEM band already shipped in every HyP3 product) and is a
      standard, well-established technique for exactly this failure mode in high-relief
      terrain, so it was built first; ERA5-based correction remains a legitimate follow-up.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.ingestion import insar_hyp3 as hyp3

logger = logging.getLogger("nepal_mhews.insar_timeseries")

# Network construction. Connecting each acquisition to its next K neighbours gives the
# redundancy the inversion needs: with K=1 the network is a chain and every observation is
# load-bearing, so a single unwrapping error corrupts everything after it.
DEFAULT_CONNECTIONS = 3
DEFAULT_MAX_BASELINE_DAYS = 48
# Sized so a typical ~34-epoch annual stack keeps redundancy ~2.5-3 rather than being
# truncated below MIN_REDUNDANCY. At 1 credit per pair this is ~90 of 8000/month.
DEFAULT_MAX_PAIRS = 90

# Quality gates for reporting a velocity at all.
MIN_PAIRS = 8
MIN_EPOCHS = 5
# Redundancy = pairs per unknown epoch. 1.0 is a bare chain (no redundancy at all).
MIN_REDUNDANCY = 1.3
# Residual RMS above this means the network disagrees with itself: unwrapping errors or
# severe atmosphere. Reporting a velocity from such a fit would be false precision.
MAX_RESIDUAL_RMS_MM = 12.0

# The reference annulus: far enough to be outside a local failure, near enough to share the
# same atmospheric column.
REFERENCE_INNER_M = 600.0
REFERENCE_OUTER_M = 2500.0

# A second, further-out ring used ONLY to check that the reference annulus itself is
# stable. If "reference minus outer-control" shows its own significant drift, the
# reference was never a safe zero-point and the main result must be caveated.
OUTER_CONTROL_INNER_M = 3000.0
OUTER_CONTROL_OUTER_M = 5000.0
REFERENCE_DRIFT_FLOOR_MM_YEAR = 3.0  # below this, flagging would just be noise-chasing

# Per-interferogram elevation-detrending thresholds (see measure_double_difference): below
# this many pixels, or this little elevation spread, a fitted phase/elevation slope is noise
# rather than signal, so the correction is skipped for that one pair rather than forced.
ELEVATION_DETREND_MIN_PIXELS = 60
ELEVATION_DETREND_MIN_RANGE_M = 150.0


# ---------------------------------------------------------------------------
# multi-network keying: one network per FLIGHT DIRECTION per site
# ---------------------------------------------------------------------------
# A single track measures one line-of-sight component of a three-dimensional motion, so
# one network per site can never separate downslope movement from the viewing geometry
# that happened to be available. Ascending and descending passes see the same slope from
# opposite sides, which is what makes the combination in `insar_combine` possible.
#
# The original state file held exactly one network per site, keyed by a bare
# `hyp3.site_key(lat, lon)`. Keys are now "<site_key>@<DIRECTION>", and
# `migrate_legacy_network_keys` rewrites the old form in place so an existing production
# state file keeps its solved networks instead of silently losing them.

FLIGHT_DIRECTIONS = ("ASCENDING", "DESCENDING")
DIRECTION_UNKNOWN = "UNKNOWN"
KEY_SEPARATOR = "@"


def network_key(lat: float, lon: float, direction: str) -> str:
    """State-file key for one site's network on one flight direction."""
    direction = (direction or DIRECTION_UNKNOWN).upper()
    return f"{hyp3.site_key(lat, lon)}{KEY_SEPARATOR}{direction}"


def split_network_key(key: str) -> Tuple[str, str]:
    """('29.5023,82.2504@ASCENDING') -> ('29.5023,82.2504', 'ASCENDING')."""
    site, sep, direction = str(key).rpartition(KEY_SEPARATOR)
    if not sep:
        return str(key), DIRECTION_UNKNOWN
    return site, (direction or DIRECTION_UNKNOWN).upper()


def migrate_legacy_network_keys(state: Dict[str, Any]) -> int:
    """Rewrite bare site keys to "<site_key>@<DIRECTION>". Idempotent; returns the count.

    The direction of a legacy network is genuinely unknown here - the stored record
    predates direction tracking and the observations predate look-vector extraction, so
    there is nothing locally to infer it from. It is marked UNKNOWN rather than guessed,
    and `backfill_network_direction` resolves it from CMR the next time the network is
    touched. A guessed direction would silently corrupt the asc/desc combination, which
    is exactly the kind of confident-but-wrong value this module exists to avoid.
    """
    networks = state.get("networks")
    if not isinstance(networks, dict):
        return 0
    migrated = 0
    for legacy in [k for k in list(networks) if KEY_SEPARATOR not in k]:
        net = networks.pop(legacy)
        direction = (net.get("flight_direction") or DIRECTION_UNKNOWN).upper()
        net["flight_direction"] = direction
        new_key = f"{legacy}{KEY_SEPARATOR}{direction}"
        # If a network for this direction somehow already exists, keep whichever has
        # actually been solved rather than clobbering a real result.
        if new_key in networks:
            incumbent = networks[new_key]
            if incumbent is net:
                migrated += 1
                continue
            # Both key forms exist for the same direction. Keep whichever is actually
            # solved; park the loser under a suffixed key rather than dropping it, because
            # a discarded network is spent credits and unrecoverable pair history.
            if net.get("solution") and not incumbent.get("solution"):
                networks[new_key] = net
                networks[f"{new_key}+superseded"] = incumbent
            else:
                networks[f"{new_key}+legacy"] = net
            logger.warning(
                "Both a legacy and a direction-scoped network exist for %s; kept the "
                "solved one and preserved the other under a suffixed key", new_key,
            )
            migrated += 1
            continue
        networks[new_key] = net
        migrated += 1
    if migrated:
        logger.info("Migrated %d legacy InSAR network key(s) to direction-scoped keys",
                    migrated)
    return migrated


def networks_for_site(
    state: Dict[str, Any], lat: float, lon: float
) -> Dict[str, Dict[str, Any]]:
    """Every network recorded for one site, keyed by flight direction."""
    prefix = f"{hyp3.site_key(lat, lon)}{KEY_SEPARATOR}"
    out: Dict[str, Dict[str, Any]] = {}
    for key, net in (state.get("networks") or {}).items():
        if key.startswith(prefix):
            out[split_network_key(key)[1]] = net
    return out


@dataclass
class NetworkPlan:
    """A small-baseline pair network for one site."""

    burst_id: str
    swath: str
    polarization: str
    epochs: List[datetime]
    pairs: List[Tuple[str, str, datetime, datetime]] = field(default_factory=list)
    # Which side the slope was viewed from. Carried through to the state record so the
    # asc/desc combination knows what geometry each velocity came from; "UNKNOWN" when
    # the burst metadata did not say, never inferred.
    flight_direction: str = "UNKNOWN"
    path_number: Optional[str] = None

    @property
    def redundancy(self) -> float:
        unknowns = max(1, len(self.epochs) - 1)
        return len(self.pairs) / unknowns


def plan_network(
    lat: float,
    lon: float,
    *,
    connections: int = DEFAULT_CONNECTIONS,
    max_baseline_days: int = DEFAULT_MAX_BASELINE_DAYS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    since_days: Optional[int] = 400,
    direction: Optional[str] = None,
) -> Optional[NetworkPlan]:
    """Build a redundant small-baseline network over the deepest co-pol stack at a point.

    ``direction`` restricts the choice to ASCENDING or DESCENDING stacks. Left as None it
    keeps the original behaviour - the deepest stack regardless of geometry - which is what
    the single-network callers still want.
    """
    stacks = hyp3.find_burst_stacks(lat, lon)
    if not stacks:
        return None
    if direction:
        wanted = direction.upper()
        stacks = [
            st for st in stacks
            if str(getattr(st, "flight_direction", DIRECTION_UNKNOWN)).upper() == wanted
        ]
        if not stacks:
            return None
    stack = stacks[0]

    acquisitions = sorted(stack.acquisitions, key=lambda a: a[0])  # oldest first
    if since_days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(since_days))
        acquisitions = [a for a in acquisitions if a[0] >= cutoff]
    if len(acquisitions) < MIN_EPOCHS:
        return None

    epochs = [a[0] for a in acquisitions]
    pairs: List[Tuple[str, str, datetime, datetime]] = []
    for i, (t_a, g_a) in enumerate(acquisitions):
        for j in range(i + 1, min(i + 1 + connections, len(acquisitions))):
            t_b, g_b = acquisitions[j]
            if (t_b - t_a).days > max_baseline_days:
                break
            pairs.append((g_a, g_b, t_a, t_b))  # always earlier -> later
    # Prefer the shortest baselines when capping: they are the most coherent.
    pairs.sort(key=lambda p: (p[3] - p[2]).days)
    pairs = pairs[:max_pairs]
    if not pairs:
        return None

    plan = NetworkPlan(
        stack.burst_id, stack.swath, stack.polarization, epochs, pairs,
        flight_direction=str(
            getattr(stack, "flight_direction", DIRECTION_UNKNOWN) or DIRECTION_UNKNOWN
        ).upper(),
        path_number=getattr(stack, "path_number", None),
    )
    if plan.redundancy < MIN_REDUNDANCY:
        # Submitting this would spend credits on a network the inversion will then refuse.
        logger.info(
            "Planned network redundancy %.2f is below the %.2f the inversion requires "
            "(%d pairs / %d epochs) - raise max_pairs or shorten the time span",
            plan.redundancy, MIN_REDUNDANCY, len(pairs), len(epochs),
        )
    return plan


def plan_networks(
    lat: float,
    lon: float,
    *,
    connections: int = DEFAULT_CONNECTIONS,
    max_baseline_days: int = DEFAULT_MAX_BASELINE_DAYS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    since_days: Optional[int] = 400,
) -> Dict[str, NetworkPlan]:
    """One network plan per available flight direction, keyed by direction.

    Why bother with the second direction - measured, not assumed (INSAR_NOTES.md s13):

      * On MODERATE slopes it buys coverage. At 35 degrees, six of twelve aspects are
        visible to only one geometry; at 30 degrees, eight. There a second track is the
        difference between a measurement and none.
      * On STEEP slopes it buys no coverage at all. Sensitivity carries a
        -sin(slope)*look_Up term that is aspect-independent and dominates as the slope
        steepens, so at glc_105's 45.5 degrees BOTH tracks clear the gate at all twelve
        aspects. Quoting the coverage benefit without the slope attached is wrong.
      * Precision: 1.01x to 1.41x, sqrt(2) only where both tracks are equally sensitive.
      * A testable assumption, at any slope - see combined_velocity_for_site.
    """
    stacks = hyp3.find_burst_stacks(lat, lon)
    plans: Dict[str, NetworkPlan] = {}
    directions = {
        str(getattr(st, "flight_direction", DIRECTION_UNKNOWN) or DIRECTION_UNKNOWN).upper()
        for st in stacks
    }
    for direction in sorted(directions):
        if direction == DIRECTION_UNKNOWN:
            # A stack whose geometry is unknown cannot be combined with anything, and
            # pairing it blind would corrupt the decomposition. Skip rather than guess.
            logger.info(
                "Skipping a burst stack at (%.4f, %.4f) with no flight direction in its "
                "metadata - it cannot be combined safely", lat, lon,
            )
            continue
        plan = plan_network(
            lat, lon, connections=connections, max_baseline_days=max_baseline_days,
            max_pairs=max_pairs, since_days=since_days, direction=direction,
        )
        if plan is not None:
            plans[direction] = plan
    return plans


# ---------------------------------------------------------------------------
# double-differenced observation from one interferogram
# ---------------------------------------------------------------------------
def measure_double_difference(
    product_dir: Path,
    lat: float,
    lon: float,
    *,
    patch_px: int = hyp3.PATCH_PX,
    min_coherence: float = hyp3.MIN_COHERENCE,
    inner_m: float = REFERENCE_INNER_M,
    outer_m: float = REFERENCE_OUTER_M,
    outer_control_inner_m: float = OUTER_CONTROL_INNER_M,
    outer_control_outer_m: float = OUTER_CONTROL_OUTER_M,
) -> Optional[Dict[str, Any]]:
    """LOS displacement of the target RELATIVE to a surrounding reference annulus.

    Returns None rather than a number whenever either the target patch or the reference
    annulus lacks enough coherent, successfully-unwrapped pixels. The arbitrary per-
    interferogram phase datum makes an un-differenced value meaningless, so "no reference"
    must mean "no observation".

    Also opportunistically reads, when the bands are present in the product:
      - mean DEM elevation of the target and reference areas (``*_dem.tif``), used for the
        elevation-correlated atmospheric correction in ``invert_timeseries``.
      - mean look-vector elevation/azimuth angles over the target patch (``*_lv_theta.tif``,
        ``*_lv_phi.tif``, both in RADIANS per the HyP3 product README), used to project an
        ambiguous LOS rate onto the slope-parallel direction.
      - a SECOND, further-out control ring's double-differenced phase against the reference
        annulus, used to check the reference annulus is itself stable.
    None of these being unavailable fails the core observation; they are additive.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform
    except ImportError as exc:
        logger.warning("rasterio/numpy unavailable: %s", exc)
        return None

    product_dir = Path(product_dir)
    unw = next(iter(sorted(product_dir.glob("*_unw_phase.tif"))), None)
    if unw is None:
        return None
    corr = next(iter(sorted(product_dir.glob("*_corr.tif"))), None)
    conncomp = next(iter(sorted(product_dir.glob("*_conncomp.tif"))), None)
    dem_file = next(iter(sorted(product_dir.glob("*_dem.tif"))), None)
    lv_theta_file = next(iter(sorted(product_dir.glob("*_lv_theta.tif"))), None)
    lv_phi_file = next(iter(sorted(product_dir.glob("*_lv_phi.tif"))), None)

    try:
        with rasterio.open(unw) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [float(lon)], [float(lat)])
            x0, y0 = xs[0], ys[0]
            row, col = src.index(x0, y0)
            # Pixel size in METRES. HyP3 ships UTM (EPSG:326xx) so src.res is already
            # metres, but a geographic product would report degrees - and treating 0.0008
            # deg as 0.0008 m makes the reference radius ~3.1 million pixels, which falls
            # out of bounds and silently returns None forever.
            res = abs(src.res[0]) or 80.0
            if src.crs is not None and not src.crs.is_projected:
                res = res * 111_320.0 * math.cos(math.radians(float(lat)))
            res = abs(res) or 80.0
            outer_radius_m = max(outer_m, outer_control_outer_m)
            rad_px = int(math.ceil(outer_radius_m / res))
            half = max(1, int(patch_px) // 2)

            r_lo, r_hi = row - rad_px, row + rad_px + 1
            c_lo, c_hi = col - rad_px, col + rad_px + 1
            if r_lo < 0 or c_lo < 0 or r_hi > src.height or c_hi > src.width:
                return None
            from rasterio.windows import Window

            win = Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)
            phase = src.read(1, window=win).astype("float64")
            nodata = src.nodata

        valid = np.isfinite(phase) & (phase != 0.0)
        if nodata is not None:
            valid &= phase != nodata
        if conncomp is not None:
            with rasterio.open(conncomp) as ccs:
                cc = ccs.read(1, window=win)
            if cc.shape == phase.shape:
                valid &= cc > 0
        coh = None
        if corr is not None:
            with rasterio.open(corr) as cs:
                coh = cs.read(1, window=win).astype("float64")
            if coh.shape == phase.shape:
                valid &= coh >= float(min_coherence)
            else:
                coh = None

        # distance grid about the centre of the window
        ny, nx = phase.shape
        cy, cx = (row - r_lo), (col - c_lo)
        yy, xx = np.ogrid[:ny, :nx]
        dist_m = np.sqrt(((yy - cy) * res) ** 2 + ((xx - cx) * res) ** 2)

        target_mask = valid & (np.abs(yy - cy) <= half) & (np.abs(xx - cx) <= half)
        ref_mask = valid & (dist_m >= inner_m) & (dist_m <= outer_m)

        n_t = int(np.count_nonzero(target_mask))
        n_r = int(np.count_nonzero(ref_mask))
        if n_t < 4 or n_r < 30:
            return None

        # --- Per-interferogram elevation-correlated atmospheric detrending ---
        # Dry-troposphere delay is layered by altitude, so it is the dominant SPATIALLY
        # correlated phase term in high-relief terrain - and it is specific to THIS pair's
        # two acquisition dates, not to the (fixed) target/reference geography. It must
        # therefore be estimated and removed WITHIN one interferogram using pixels that
        # span a real elevation range (the whole read window, already in memory), never as
        # a single coefficient shared across the network: target_elevation minus
        # reference_elevation is the SAME two points every time, so it cannot represent a
        # per-date atmospheric state (confirmed: solving it as one joint network-wide
        # unknown was tried and found mathematically ill-posed - see invert_timeseries).
        atmo_applied = False
        atmo_coefficient: Optional[float] = None
        phase_for_median = phase
        dem_elev: Optional["np.ndarray"] = None
        if dem_file is not None:
            try:
                with rasterio.open(dem_file) as dsrc:
                    dem_elev = dsrc.read(1, window=win).astype("float64")
                if dem_elev.shape != phase.shape:
                    dem_elev = None
            except Exception:
                dem_elev = None
        if dem_elev is not None:
            fit_mask = valid & (dem_elev > -1000.0)
            n_fit = int(np.count_nonzero(fit_mask))
            elev_fit = dem_elev[fit_mask]
            elev_range = float(elev_fit.max() - elev_fit.min()) if n_fit else 0.0
            # Need enough pixels AND enough elevation spread for the regression slope to be
            # identifiable; a flat local window gives a meaningless, noise-dominated slope.
            if n_fit >= ELEVATION_DETREND_MIN_PIXELS and elev_range >= ELEVATION_DETREND_MIN_RANGE_M:
                design = np.column_stack([elev_fit, np.ones(n_fit)])
                coeffs, *_ = np.linalg.lstsq(design, phase[fit_mask], rcond=None)
                atmo_coefficient = float(coeffs[0])
                # Subtract the fitted elevation-dependent component everywhere (target AND
                # reference), referenced to the mean elevation so the overall phase level
                # used downstream is unaffected - only the elevation-correlated TREND is
                # removed, never a change to the absolute reference point.
                phase_for_median = phase - atmo_coefficient * (dem_elev - float(elev_fit.mean()))
                atmo_applied = True

        target_phase = float(np.median(phase_for_median[target_mask]))
        ref_phase = float(np.median(phase_for_median[ref_mask]))
        dd_phase = target_phase - ref_phase
        displacement_mm = float(hyp3._phase_to_displacement_m(dd_phase)) * 1000.0

        result: Dict[str, Any] = {
            "displacement_mm": round(displacement_mm, 4),
            "dd_phase_rad": round(dd_phase, 5),
            "target_pixels": n_t,
            "reference_pixels": n_r,
            "mean_coherence": (
                None if coh is None else round(float(np.nanmean(coh[target_mask])), 3)
            ),
            "atmospheric_correction_applied": atmo_applied,
            "atmospheric_coefficient_mm_per_m": (
                None if atmo_coefficient is None
                else round(float(hyp3._phase_to_displacement_m(atmo_coefficient)) * 1000.0, 5)
            ),
        }

        # --- Reference-stability check: reference annulus vs. a further-out control ring ---
        outer_mask = valid & (dist_m >= outer_control_inner_m) & (dist_m <= outer_control_outer_m)
        n_o = int(np.count_nonzero(outer_mask))
        if n_o >= 30:
            outer_phase = float(np.median(phase[outer_mask]))
            result["ref_vs_outer_displacement_mm"] = round(
                float(hyp3._phase_to_displacement_m(ref_phase - outer_phase)) * 1000.0, 4
            )
            result["outer_control_pixels"] = n_o

        # --- Elevation metadata (audit/debugging only - reuses the array already read
        #     above for detrending; no second file open). Not used by invert_timeseries. ---
        if dem_elev is not None:
            dem_valid = dem_elev > -1000.0  # no legitimate sub-sea-level Nepal terrain
            t_elev = dem_elev[target_mask & dem_valid]
            r_elev = dem_elev[ref_mask & dem_valid]
            if t_elev.size and r_elev.size:
                result["target_elevation_m"] = round(float(np.median(t_elev)), 1)
                result["reference_elevation_m"] = round(float(np.median(r_elev)), 1)

        # --- Look vector over the target patch, for slope-parallel projection ---
        if lv_theta_file is not None and lv_phi_file is not None:
            try:
                with rasterio.open(lv_theta_file) as ts_:
                    theta = ts_.read(1, window=win).astype("float64")
                with rasterio.open(lv_phi_file) as ps_:
                    phi = ps_.read(1, window=win).astype("float64")
                if theta.shape == phase.shape and phi.shape == phase.shape:
                    tmask = target_mask & np.isfinite(theta) & np.isfinite(phi)
                    if np.any(tmask):
                        # Circular mean for phi (an angle), plain mean for theta.
                        result["lv_theta_rad"] = round(float(np.median(theta[tmask])), 4)
                        result["lv_phi_rad"] = round(
                            float(np.arctan2(
                                np.mean(np.sin(phi[tmask])), np.mean(np.cos(phi[tmask]))
                            )),
                            4,
                        )
            except Exception as exc:
                logger.debug("Look-vector read failed for %s: %s", product_dir.name, exc)

        return result
    except Exception as exc:
        logger.warning("Double-difference failed for %s: %s", product_dir.name, exc)
        return None


# ---------------------------------------------------------------------------
# the inversion
# ---------------------------------------------------------------------------
def _largest_connected_component(
    observations: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Keep only the observations in the biggest connected block of epochs.

    Each interferogram is an edge between two acquisition dates. If decorrelation removes
    enough edges the graph splits, and epochs in a different component are not constrained
    relative to the datum at all. Returns (kept_observations, n_epochs_dropped).
    """
    parent: Dict[Any, Any] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for o in observations:
        union(o["t_early"], o["t_late"])

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for o in observations:
        groups.setdefault(find(o["t_early"]), []).append(o)
    if not groups:
        return [], 0

    best = max(groups.values(), key=len)
    all_epochs = {o["t_early"] for o in observations} | {o["t_late"] for o in observations}
    kept_epochs = {o["t_early"] for o in best} | {o["t_late"] for o in best}
    dropped = len(all_epochs) - len(kept_epochs)
    if dropped:
        logger.info(
            "Network split by decorrelation: solving the largest connected component "
            "(%d of %d epochs, %d of %d pairs)",
            len(kept_epochs), len(all_epochs), len(best), len(observations),
        )
    return best, dropped


def invert_timeseries(
    observations: Sequence[Dict[str, Any]],
    *,
    min_pairs: int = MIN_PAIRS,
    min_epochs: int = MIN_EPOCHS,
    min_redundancy: float = MIN_REDUNDANCY,
    max_residual_rms_mm: float = MAX_RESIDUAL_RMS_MM,
    apply_atmospheric_correction: bool = True,
) -> Optional[Dict[str, Any]]:
    """Least-squares SBAS inversion of a double-differenced pair network.

    Each observation is {'t_early', 't_late', 'displacement_mm'} and constrains
    d(t_late) - d(t_early). Solving the whole network at once lets redundant pairs average
    down the atmospheric term, which a single pair cannot do.

    ``apply_atmospheric_correction`` is accepted for API symmetry with the per-observation
    detrending done in ``measure_double_difference`` (see that function). It is NOT solved
    for here: target and reference are the SAME two geographic locations on every pair, so
    (target_elevation - reference_elevation) is structurally constant across the whole
    network and cannot represent a per-acquisition-date atmospheric state - fitting it as one
    joint network-wide unknown is mathematically ill-posed (confirmed: it drove the design
    matrix rank-deficient in testing). The correction that actually works is per-interferogram:
    regressing phase against elevation across the many pixels already read in one scene,
    where elevation genuinely varies, and removing that trend before the target/reference
    difference is even taken. ``apply_atmospheric_correction`` here only controls whether the
    per-observation ``atmospheric_correction_applied`` flags are surfaced/aggregated.

    Returns None - never a velocity - when the network is too small, insufficiently
    redundant, or internally inconsistent.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    obs = [
        o for o in observations
        if o.get("displacement_mm") is not None and o.get("t_early") and o.get("t_late")
    ]
    if len(obs) < min_pairs:
        return None

    # Decorrelated pairs routinely fragment a network: on the real glc_105 run, 36 of 90
    # pairs were unusable and the survivors left 2 epochs unconstrained (rank 25 of 27).
    # Inverting that whole set would let lstsq invent minimum-norm values for the
    # unconstrained epochs. Instead, solve the largest FULLY CONNECTED subnetwork and
    # report how much was dropped - the standard SBAS treatment.
    obs, dropped_epochs = _largest_connected_component(obs)
    if not obs:
        return None

    epochs = sorted({o["t_early"] for o in obs} | {o["t_late"] for o in obs})
    if len(epochs) < min_epochs:
        return None
    index = {t: i for i, t in enumerate(epochs)}

    n_unknown = len(epochs) - 1  # epoch 0 is the datum, d_0 == 0

    redundancy = len(obs) / max(1, n_unknown)
    if redundancy < min_redundancy:
        logger.info("Network redundancy %.2f below %.2f", redundancy, min_redundancy)
        return None

    G = np.zeros((len(obs), n_unknown), dtype="float64")
    y = np.zeros(len(obs), dtype="float64")
    for k, o in enumerate(obs):
        a, b = index[o["t_early"]], index[o["t_late"]]
        if a > 0:
            G[k, a - 1] = -1.0
        if b > 0:
            G[k, b - 1] = +1.0
        y[k] = float(o["displacement_mm"])

    # Rank check: a disconnected network leaves some epochs unconstrained, and lstsq would
    # happily return a minimum-norm answer for them that means nothing.
    if np.linalg.matrix_rank(G) < n_unknown:
        logger.info("Interferogram network is disconnected; refusing to invert")
        return None

    solution, *_ = np.linalg.lstsq(G, y, rcond=None)
    displacements = np.concatenate([[0.0], solution])

    atmo_flags = [bool(o.get("atmospheric_correction_applied")) for o in obs]
    atmo_coeffs = [
        o["atmospheric_coefficient_mm_per_m"] for o in obs
        if o.get("atmospheric_coefficient_mm_per_m") is not None
    ]
    n_atmo_corrected = sum(1 for f in atmo_flags if f) if apply_atmospheric_correction else 0
    mean_atmo_coefficient = (
        float(np.mean(atmo_coeffs)) if (apply_atmospheric_correction and atmo_coeffs) else None
    )

    residuals = y - G @ solution
    residual_rms = float(np.sqrt(np.mean(residuals ** 2)))
    if residual_rms > max_residual_rms_mm:
        logger.info(
            "Network residual RMS %.1f mm exceeds %.1f mm; not reporting a velocity",
            residual_rms, max_residual_rms_mm,
        )
        return None

    # Linear velocity through the recovered time series.
    t0 = epochs[0]
    days = np.array([(t - t0).total_seconds() / 86400.0 for t in epochs], dtype="float64")
    A = np.vstack([days, np.ones_like(days)]).T
    (slope_mm_per_day, _intercept), *_ = np.linalg.lstsq(A, displacements, rcond=None)

    fit = A @ np.array([slope_mm_per_day, _intercept])
    dof = max(1, len(days) - 2)
    scatter = float(np.sqrt(np.sum((displacements - fit) ** 2) / dof))
    denom = float(np.sum((days - days.mean()) ** 2)) or 1.0
    slope_stderr = scatter / math.sqrt(denom)

    return {
        "velocity_mm_year": round(float(slope_mm_per_day) * 365.25, 3),
        "velocity_mm_month": round(float(slope_mm_per_day) * 30.0, 3),
        "velocity_stderr_mm_year": round(slope_stderr * 365.25, 3),
        "residual_rms_mm": round(residual_rms, 3),
        "timeseries_scatter_mm": round(scatter, 3),
        "n_pairs": len(obs),
        "n_epochs": len(epochs),
        "redundancy": round(redundancy, 2),
        "span_days": round(float(days[-1]), 1),
        "first_epoch": epochs[0].isoformat(),
        "last_epoch": epochs[-1].isoformat(),
        "displacements_mm": [round(float(d), 3) for d in displacements],
        "epochs": [t.isoformat() for t in epochs],
        "epochs_dropped_disconnected": dropped_epochs,
        "n_pairs_atmo_corrected": n_atmo_corrected,
        "mean_atmo_coefficient_mm_per_m": (
            None if mean_atmo_coefficient is None else round(mean_atmo_coefficient, 5)
        ),
        "method": (
            "SBAS least-squares on double-differenced, per-interferogram "
            "elevation-detrended pairs"
            if n_atmo_corrected
            else "SBAS least-squares on double-differenced pairs"
        ),
    }


# ---------------------------------------------------------------------------
# network lifecycle: submit -> collect -> invert, persisted across runs
# ---------------------------------------------------------------------------
def submit_network(
    lat: float,
    lon: float,
    *,
    site_id: str = "site",
    connections: int = DEFAULT_CONNECTIONS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    since_days: Optional[int] = 400,
    direction: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan and submit a whole small-baseline network for one site and one geometry.

    Costs one credit per pair. Submission and collection are separate because each job
    takes roughly an hour; nothing here blocks on processing.

    ``direction`` picks the flight direction. Left as None the deepest stack wins,
    whichever way it was looking, and the record is keyed by whatever direction that
    stack turned out to be - so a caller that does not care still gets a correctly
    labelled network rather than an unlabelled one.
    """
    if not getattr(settings, "hyp3_enabled", False):
        return {"status": "DISABLED"}
    if hyp3._auth_headers() is None:
        return {"status": "NO_CREDENTIALS"}

    state = hyp3._load_state()
    migrate_legacy_network_keys(state)

    plan = plan_network(
        lat, lon, connections=connections, max_pairs=max_pairs,
        since_days=since_days, direction=direction,
    )
    if plan is None:
        return {"status": "NO_NETWORK"}

    # The key is direction-scoped, and the direction comes from the plan rather than the
    # caller, so an unlabelled network can never be written.
    key = network_key(lat, lon, plan.flight_direction)
    existing = (state.get("networks") or {}).get(key)

    # A site can have TWO stacks on the same direction (glc_105 has 337397/IW1 and
    # 118306/IW3, both ascending), and the deepest one can change between runs. They share
    # a key, so appending blindly would merge two different look geometries into one SBAS
    # inversion - which produced +13.2 +/- 4.9 mm/yr from data whose honest answer is
    # -0.3 +/- 6.7, with a residual small enough to pass the quality gate. Refuse instead.
    if existing and existing.get("burst_id") and existing["burst_id"] != plan.burst_id:
        return {
            "status": "BURST_MISMATCH",
            "existing_burst": existing.get("burst_id"),
            "planned_burst": plan.burst_id,
            "flight_direction": plan.flight_direction,
            "note": (
                "a different burst stack is already recorded for this site and direction; "
                "appending would mix two look geometries into one inversion"
            ),
        }

    # Top up rather than refuse: a part-built network is useless until it clears
    # MIN_REDUNDANCY, so an interrupted or budget-truncated submission must be completable.
    already = set()
    if existing:
        already = {(p.get("reference"), p.get("secondary")) for p in existing.get("pairs", [])}
    todo = [p for p in plan.pairs if (p[0], p[1]) not in already]
    if not todo:
        if existing and existing.get("solution"):
            return {"status": "ALREADY_SOLVED"}
        return {"status": "ALREADY_SUBMITTED", "pairs": len(already)}

    # A network is submitted against the MONTHLY budget only. hyp3_max_jobs_per_run caps
    # how many *sites* the single-pair screening path touches per run; applying it here
    # silently truncated a 90-pair network to 20 (redundancy 0.61), which the inversion
    # then refuses - spending credits for nothing.
    budget = int(getattr(settings, "hyp3_monthly_credit_budget", 8000))
    local_left = max(0, budget - hyp3.credits_spent_this_period(state)) // hyp3.JOB_CREDIT_COST
    allowed = min(len(todo), local_left)
    remote = hyp3.remaining_credits()
    if remote is not None:
        allowed = min(allowed, int(remote // hyp3.JOB_CREDIT_COST))

    projected = (len(already) + allowed) / max(1, len(plan.epochs) - 1)
    if projected < MIN_REDUNDANCY:
        return {
            "status": "BUDGET_EXHAUSTED",
            "needed": len(todo),
            "allowed": allowed,
            "projected_redundancy": round(projected, 2),
            "required_redundancy": MIN_REDUNDANCY,
            "note": "refusing to part-build a network the inversion would reject",
        }
    pairs = todo[:allowed]

    submitted: List[Dict[str, Any]] = []
    for ref, sec, t_a, t_b in pairs:
        # HyP3 orders the product earlier->later itself; submit oldest first for clarity.
        job = hyp3.submit_insar_job(ref, sec, name=f"mhews-ts-{site_id}"[:100])
        if not job:
            continue
        hyp3._record_credit_spend(state, hyp3.JOB_CREDIT_COST)
        submitted.append(
            {
                "job_id": job.get("job_id"),
                "status": job.get("status_code", "PENDING"),
                "reference": ref,
                "secondary": sec,
                "t_early": t_a.isoformat(),
                "t_late": t_b.isoformat(),
            }
        )

    if not submitted:
        return {"status": "SUBMISSION_FAILED"}

    record = (state.get("networks") or {}).get(key) or {
        "site_id": site_id,
        "lat": float(lat),
        "lon": float(lon),
        "burst_id": plan.burst_id,
        "swath": plan.swath,
        "polarization": plan.polarization,
        "flight_direction": plan.flight_direction,
        "path_number": plan.path_number,
        "submitted_utc": time.time(),
        "pairs": [],
        "solution": None,
    }
    record["pairs"] = list(record.get("pairs", [])) + submitted
    # Backfill geometry onto a record created before direction tracking existed.
    record.setdefault("flight_direction", plan.flight_direction)
    record.setdefault("path_number", plan.path_number)
    state.setdefault("networks", {})[key] = record
    hyp3._save_state(state)

    total = len(record["pairs"])
    return {
        "status": "SUBMITTED",
        "new_pairs": len(submitted),
        "pairs_total": total,
        "epochs": len(plan.epochs),
        "redundancy": round(total / max(1, len(plan.epochs) - 1), 2),
        "flight_direction": plan.flight_direction,
        "credits_spent_this_period": hyp3.credits_spent_this_period(state),
    }


def backfill_network_direction(
    state: Dict[str, Any], lat: float, lon: float
) -> Optional[str]:
    """Resolve an UNKNOWN network's flight direction from CMR and re-key it.

    Networks recorded before direction tracking carry UNKNOWN, and an UNKNOWN network can
    never take part in a combination - its look geometry is exactly what the combination
    needs to know. The direction is recoverable because the burst id was stored, and the
    burst id maps to one authoritative ASCENDING_DESCENDING value in CMR.

    Makes a network call, so this belongs on the weekly collection path and never in the
    30-minute physics loop. Returns the resolved direction, or None if nothing changed.
    """
    networks = state.get("networks") or {}
    unknown_key = network_key(lat, lon, DIRECTION_UNKNOWN)
    net = networks.get(unknown_key)
    if not net or not net.get("burst_id"):
        return None
    resolved = hyp3.burst_direction(
        net["burst_id"], net.get("swath", ""), net.get("polarization", "VV"), lat, lon
    )
    if resolved == DIRECTION_UNKNOWN:
        return None

    target = network_key(lat, lon, resolved)
    if target in networks and networks[target] is not net:
        # Something already occupies the resolved slot; leave both alone rather than
        # silently merging two networks that were built from different stacks.
        logger.warning(
            "Cannot backfill direction for %s: %s already exists", unknown_key, target
        )
        return None
    net["flight_direction"] = resolved
    networks.pop(unknown_key, None)
    networks[target] = net
    logger.info(
        "Backfilled flight direction %s onto the network at (%.4f, %.4f) from burst %s",
        resolved, lat, lon, net["burst_id"],
    )
    return resolved


def collect_network(
    lat: float, lon: float, direction: Optional[str] = None
) -> Dict[str, Any]:
    """Poll one network, double-difference every finished pair, then invert.

    ``direction`` selects which geometry's network to collect. With None and exactly one
    network recorded for the site, that one is used - which keeps every existing caller
    working. With None and several, the caller must say which, because collecting "the"
    network is no longer a well-defined request.
    """
    if not getattr(settings, "hyp3_enabled", False):
        return {"status": "DISABLED"}
    if hyp3._auth_headers() is None:
        return {"status": "NO_CREDENTIALS"}

    state = hyp3._load_state()
    migrated = migrate_legacy_network_keys(state)
    # An UNKNOWN network cannot be combined with anything, so take the one chance the
    # weekly path has to resolve it properly. Network call; never on the hot loop.
    backfilled = backfill_network_direction(state, lat, lon)
    if migrated or backfilled:
        hyp3._save_state(state)
    available = networks_for_site(state, lat, lon)
    if not available:
        return {"status": "NO_NETWORK"}
    if direction and backfilled and direction.upper() == DIRECTION_UNKNOWN:
        # The backfill just re-keyed this network from UNKNOWN to its real direction, so
        # the caller's argument - read from the key BEFORE the backfill - now points at a
        # key that no longer exists. Without this the weekly cron silently reports
        # NO_NETWORK for the only solved network in the system.
        logger.info("Network re-keyed UNKNOWN -> %s during collection; following it",
                    backfilled)
        direction = backfilled
    if direction:
        resolved = direction.upper()
        if resolved not in available:
            return {"status": "NO_NETWORK", "available_directions": sorted(available)}
    elif len(available) == 1:
        resolved = next(iter(available))
    else:
        return {
            "status": "AMBIGUOUS_DIRECTION",
            "available_directions": sorted(available),
            "note": "several geometries recorded for this site; name one",
        }
    key = network_key(lat, lon, resolved)
    net = available[resolved]

    pending = done = failed = newly_measured = 0
    for pair in net.get("pairs", []):
        if pair.get("observation") is not None:
            done += 1
            continue
        if pair.get("status") in ("FAILED", "UNMEASURABLE"):
            failed += 1
            continue
        job = hyp3.get_job(pair.get("job_id") or "")
        if not job:
            pending += 1
            continue
        status = job.get("status_code")
        pair["status"] = status
        if status in ("PENDING", "RUNNING"):
            pending += 1
            continue
        if status != "SUCCEEDED":
            failed += 1
            continue

        product_dir = hyp3.download_product(job)
        if product_dir is None:
            pair["status"] = "FAILED"
            failed += 1
            continue
        obs = measure_double_difference(Path(product_dir), net["lat"], net["lon"])
        if obs is None:
            # Decorrelated or no usable reference annulus. Record it so the pair is not
            # re-downloaded forever, and leave it out of the inversion entirely.
            pair["status"] = "UNMEASURABLE"
            failed += 1
            continue
        pair["observation"] = obs
        done += 1
        newly_measured += 1

    observations = [
        {
            "t_early": datetime.fromisoformat(p["t_early"]),
            "t_late": datetime.fromisoformat(p["t_late"]),
            "displacement_mm": p["observation"]["displacement_mm"],
            # Per-interferogram detrending already happened in measure_double_difference
            # (target/reference are fixed geography, so correction has to happen THERE,
            # per pair - see invert_timeseries docstring). These two are carried through
            # only so invert_timeseries can report how many pairs were corrected.
            "atmospheric_correction_applied": p["observation"].get(
                "atmospheric_correction_applied", False
            ),
            "atmospheric_coefficient_mm_per_m": p["observation"].get(
                "atmospheric_coefficient_mm_per_m"
            ),
        }
        for p in net.get("pairs", [])
        if p.get("observation")
    ]
    solution = invert_timeseries(observations)

    # Reference-stability check: was the "stable" reference annulus actually stable, or
    # does it drift relative to a wider area? Built from the SAME pairs, same inversion
    # machinery, just substituting ref-vs-outer-control as the "displacement".
    ref_check: Dict[str, Any] = {"checked": False}
    ref_obs = [
        {
            "t_early": datetime.fromisoformat(p["t_early"]),
            "t_late": datetime.fromisoformat(p["t_late"]),
            "displacement_mm": p["observation"]["ref_vs_outer_displacement_mm"],
        }
        for p in net.get("pairs", [])
        if p.get("observation") and p["observation"].get("ref_vs_outer_displacement_mm") is not None
    ]
    if len(ref_obs) >= MIN_PAIRS:
        ref_solution = invert_timeseries(ref_obs, apply_atmospheric_correction=False)
        if ref_solution:
            drift = abs(ref_solution["velocity_mm_year"])
            stderr = ref_solution["velocity_stderr_mm_year"]
            unstable = drift > REFERENCE_DRIFT_FLOOR_MM_YEAR and drift > 2.0 * max(stderr, 0.1)
            ref_check = {
                "checked": True,
                "stable": not unstable,
                "drift_mm_year": ref_solution["velocity_mm_year"],
                "drift_stderr_mm_year": stderr,
                "n_pairs": ref_solution["n_pairs"],
            }
            if unstable:
                logger.warning(
                    "Reference annulus for network at (%.4f, %.4f) appears unstable: "
                    "%.1f +/- %.1f mm/yr relative to the wider surroundings",
                    net["lat"], net["lon"], ref_solution["velocity_mm_year"], stderr,
                )
    if solution is not None:
        solution["reference_stability"] = ref_check

    if solution is not None:
        net["solution"] = solution
        net["solved_utc"] = time.time()
    elif net.get("solution"):
        # Re-inversion failed - one decorrelated pair with an unwrapping error is enough,
        # and so is numpy failing to import. Overwriting a good solution with None would
        # throw away a result that cost 90 credits and silently revert the site to the
        # modelled proxy. Keep the previous answer and say that it is now stale.
        logger.warning(
            "Re-inversion of the network at (%.4f, %.4f) produced no solution; keeping "
            "the previous one (solved %.1f days ago) rather than discarding it",
            net["lat"], net["lon"],
            (time.time() - float(net.get("solved_utc") or time.time())) / 86400.0,
        )
        net["reinversion_failed_utc"] = time.time()
    else:
        net["solution"] = None
        net["solved_utc"] = None
    hyp3._save_state(state)

    return {
        "status": "OK",
        "pairs_total": len(net.get("pairs", [])),
        "measured": done,
        "pending": pending,
        "unusable": failed,
        "newly_measured": newly_measured,
        "velocity_mm_year": (solution or {}).get("velocity_mm_year"),
        "reference_stable": ref_check.get("stable"),
        "summary": summarize(solution),
    }


def timeseries_velocity_provider(
    lat: float, lon: float, allow_network: bool = True, direction: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """A MEASURED_INSAR velocity backed by a full inverted network, or None.

    ``direction`` picks one geometry's network. Left as None the best-solved network for
    the site wins, which keeps single-geometry sites behaving exactly as before.

    ``allow_network=False`` forbids the one remote read in here (the Copernicus DEM
    lookup used to estimate slope aspect). The 30-minute physics loop passes False:
    491 sites x a /vsicurl DEM fetch would make the loop depend on an external host
    being up, and the loop must never do that. Aspect is cached into the state file on
    the first weekly collection run, so the hot path reads it for free afterwards.

    Deliberately stricter than the single-pair path: a lone interferogram carries
    atmospheric delay of the same order as a year of creep, so it is a screening indicator
    and is NOT offered here as a measurement. Only an inverted, redundant, self-consistent
    network earns the MEASURED label.
    """
    if not getattr(settings, "hyp3_enabled", False):
        return None
    state = hyp3._load_state()
    # Keys are direction-scoped now. Looking up the bare site key would silently return
    # nothing for every migrated network - i.e. quietly drop a real measured velocity
    # back to the modelled proxy - so the lookup goes through networks_for_site.
    available = networks_for_site(state, lat, lon)
    if not available:
        # A state file that has not been migrated yet still carries the bare key.
        legacy = (state.get("networks") or {}).get(hyp3.site_key(lat, lon))
        if legacy:
            available = {DIRECTION_UNKNOWN: legacy}
    solved = {d: n for d, n in available.items() if n.get("solution")}
    if not solved:
        return None
    if direction:
        net = solved.get(direction.upper())
        if net is None:
            return None
    else:
        # Prefer the tightest solved network when the caller does not care.
        net = min(
            solved.values(),
            key=lambda n: float(
                (n.get("solution") or {}).get("velocity_stderr_mm_year") or 1e9
            ),
        )

    solution = net["solution"]
    solved_utc = float(net.get("solved_utc") or 0.0)
    max_age = float(getattr(settings, "hyp3_baseline_max_age_days", 30))
    age_days = (time.time() - solved_utc) / 86400.0 if solved_utc else None
    if age_days is not None and age_days > max_age * 3:
        # A time series ages far more gracefully than a single pair - it describes a rate
        # over a long span - but it should not be served indefinitely without refresh.
        return None

    measured_pairs = [p for p in net.get("pairs", []) if p.get("observation")]

    caveats: List[str] = [
        "LOS rate relative to the local reference annulus, not absolute. "
        "DEM error and long-wavelength orbital ramps are unmodelled."
    ]
    ref_stability = solution.get("reference_stability") or {}
    if ref_stability.get("checked") and not ref_stability.get("stable", True):
        caveats.append(
            f"REFERENCE ANNULUS MAY BE UNSTABLE: drifting "
            f"{ref_stability['drift_mm_year']:+.1f} +/- {ref_stability['drift_stderr_mm_year']:.1f} "
            f"mm/yr relative to the wider surroundings - treat this velocity with caution"
        )
    if solution.get("n_pairs_atmo_corrected"):
        mean_coef = solution.get("mean_atmo_coefficient_mm_per_m")
        caveats.append(
            f"per-interferogram elevation-detrending applied to "
            f"{solution['n_pairs_atmo_corrected']}/{solution['n_pairs']} pairs"
            + (f" (mean {mean_coef:+.3f} mm per m)" if mean_coef is not None else "")
        )

    result: Dict[str, Any] = {
        "los_velocity_mm_month": solution["velocity_mm_month"],
        "los_velocity_mm_year": solution["velocity_mm_year"],
        "uncertainty_mm_year": solution["velocity_stderr_mm_year"],
        "product_id": f"sbas-{net.get('burst_id')}-{net.get('swath')}",
        "product_reference": (
            f"ASF HyP3 SBAS network: {solution['n_pairs']} interferograms over "
            f"{solution['n_epochs']} epochs ({solution['span_days']:.0f} d), "
            f"bursts {net.get('burst_id')} {net.get('swath')} {net.get('polarization')}, "
            f"jobs {', '.join(p['job_id'][:8] for p in measured_pairs[:4])}..."
        ),
        "source": "ASF HyP3 Sentinel-1 burst InSAR, SBAS time-series inversion",
        "acquired_utc": solved_utc,
        "observation_days": solution["span_days"],
        "residual_rms_mm": solution["residual_rms_mm"],
        "redundancy": solution["redundancy"],
        "confidence": "measured_timeseries",
        "flight_direction": (net.get("flight_direction") or DIRECTION_UNKNOWN).upper(),
        "path_number": net.get("path_number"),
        "reference_stability": ref_stability,
        "caveat": "; ".join(caveats),
        "age_days": None if age_days is None else round(age_days, 2),
    }

    # Slope-parallel projection, best-effort: needs a known slope + aspect. Aspect is not
    # in SiteInfo, so it is estimated from the open Copernicus DEM on first use and cached
    # in the network record (same free, no-credential source as the Tier A terrain pipeline).
    dirty = False
    slope_deg = net.get("slope_deg")
    if slope_deg is None:
        slope_deg = _lookup_slope_deg(lat, lon)  # local catalog read, no network
        if slope_deg is not None:
            net["slope_deg"] = slope_deg
            dirty = True
    aspect_deg = net.get("aspect_deg")
    if aspect_deg is None and slope_deg is not None and allow_network:
        aspect_deg = estimate_aspect_deg(lat, lon)
        if aspect_deg is not None:
            net["aspect_deg"] = aspect_deg
            dirty = True
    if dirty and allow_network:
        # Persist, or the DEM would be re-fetched on every single call. Gated on
        # allow_network because the 30-minute loop passes False and must stay a READER:
        # it holds a stale snapshot for the length of a 491-site sweep, so writing it back
        # would clobber whatever the weekly script committed in the meantime - including
        # the credit ledger, which would then read as zero spent.
        hyp3._save_state(state)

    thetas = [p["observation"].get("lv_theta_rad") for p in measured_pairs if p.get("observation", {}).get("lv_theta_rad") is not None]
    phis = [p["observation"].get("lv_phi_rad") for p in measured_pairs if p.get("observation", {}).get("lv_phi_rad") is not None]
    if thetas and phis and slope_deg is not None and aspect_deg is not None:
        import statistics

        mean_theta = statistics.fmean(thetas)
        mean_phi = math.atan2(statistics.fmean(math.sin(p) for p in phis),
                               statistics.fmean(math.cos(p) for p in phis))
        projection = project_los_to_slope_parallel(
            solution["velocity_mm_year"], mean_theta, mean_phi, slope_deg, aspect_deg
        )
        result["slope_parallel_velocity_mm_year"] = projection["slope_parallel_value"]
        result["slope_parallel_sensitivity"] = projection["sensitivity"]
        result["slope_parallel_reliable"] = projection["reliable"]
        # The projection divides the velocity by the sensitivity, so the uncertainty must
        # be divided by exactly the same number. Pairing a downslope rate with the raw
        # line-of-sight sigma understates the error by 1/|s| - up to 6x at the 0.15 gate -
        # and that ratio is precisely what decides MEASURED vs PROXY downstream.
        sens = projection.get("sensitivity")
        if projection["reliable"] and sens:
            result["slope_parallel_uncertainty_mm_year"] = round(
                abs(float(solution["velocity_stderr_mm_year"]) / float(sens)), 3
            )
        result["slope_parallel_note"] = projection["note"]
        result["aspect_deg_used"] = round(float(aspect_deg), 1)

    # The stack-mean look vector is what makes this network combinable with the other
    # geometry's. Exposed even when the slope/aspect lookup failed above, because the
    # combination needs the geometry regardless of whether a single-track projection
    # was possible.
    if thetas and phis:
        import statistics

        result["lv_theta_rad"] = round(statistics.fmean(thetas), 6)
        result["lv_phi_rad"] = round(
            math.atan2(statistics.fmean(math.sin(x) for x in phis),
                       statistics.fmean(math.cos(x) for x in phis)), 6)
    span = float(solution.get("span_days") or 0.0)
    result["t_end_utc"] = solved_utc or None
    result["t_start_utc"] = (solved_utc - span * 86400.0) if solved_utc and span else None

    return result


# ---------------------------------------------------------------------------
# ascending + descending: one velocity per site from every geometry it has
# ---------------------------------------------------------------------------
def combined_velocity_for_site(
    lat: float, lon: float, allow_network: bool = False
) -> Optional[Dict[str, Any]]:
    """One measured record per site, combining every solved geometry it has.

    With a single geometry this returns that network's record unchanged - the common
    case today, and the reason the combination is additive rather than a rewrite.

    With two it runs `insar_combine.combine_tracks`, which is worth doing for three
    separate reasons (measured, see INSAR_NOTES.md section 13):
      * A TESTABLE ASSUMPTION, the only benefit that holds at every slope: two
        observations constraining one downslope unknown leave one degree of freedom, so
        chi-square can test "motion is purely downslope". One track has zero degrees of
        freedom and can never be shown wrong, which is not the same as being right. The
        test has a blind spot of its own on due north/south slopes, which is measured and
        published per combination rather than assumed away.
      * COVERAGE, but only on MODERATE slopes: six of twelve aspects are single-geometry
        at 35 degrees, eight at 30 - and ZERO at glc_105's 45.5 degrees, where the
        aspect-independent -sin(slope)*look_Up term puts both tracks over the sensitivity
        gate at every aspect. Not a headline benefit without the slope attached.
      * PRECISION, 1.01x-1.41x, and only where both tracks see the slope.
      * EAST/UP: vertical separated from horizontal, which no single track can do, and
        which needs no slope model at all.
    """
    state = hyp3._load_state()
    available = networks_for_site(state, lat, lon)
    if not available:
        legacy = (state.get("networks") or {}).get(hyp3.site_key(lat, lon))
        if legacy:
            available = {DIRECTION_UNKNOWN: legacy}
    directions = [d for d, n in available.items() if n.get("solution")]
    if not directions:
        return None

    records: Dict[str, Dict[str, Any]] = {}
    for direction in directions:
        rec = timeseries_velocity_provider(
            lat, lon, allow_network=allow_network, direction=direction
        )
        if rec:
            records[direction] = rec
    if not records:
        return None

    # Prefer the tightest as the record we hand back, then enrich it if a genuine
    # multi-geometry combination is possible.
    best_direction = min(
        records, key=lambda d: float(records[d].get("uncertainty_mm_year") or 1e9)
    )
    result = dict(records[best_direction])
    result["geometries_available"] = sorted(records)
    if len(records) < 2:
        return result

    try:
        from app.ingestion import insar_combine
    except Exception as exc:
        logger.warning("insar_combine unavailable; serving the single best geometry: %s", exc)
        return result

    slope_deg = next(
        (n.get("slope_deg") for n in available.values() if n.get("slope_deg") is not None),
        None,
    )
    aspect_deg = next(
        (n.get("aspect_deg") for n in available.values() if n.get("aspect_deg") is not None),
        None,
    )
    tracks = [
        {
            "key": direction,
            "flight_direction": direction,
            "velocity_mm_year": rec.get("los_velocity_mm_year"),
            "uncertainty_mm_year": rec.get("uncertainty_mm_year"),
            "lv_theta_rad": rec.get("lv_theta_rad"),
            "lv_phi_rad": rec.get("lv_phi_rad"),
            "t_start_utc": rec.get("t_start_utc"),
            "t_end_utc": rec.get("t_end_utc"),
        }
        for direction, rec in sorted(records.items())
    ]
    combined = insar_combine.combine_tracks(tracks, slope_deg, aspect_deg)

    # The East/Up decomposition needs only the two look vectors and the two LOS rates -
    # no slope, no aspect. combine_tracks skips it when the downslope direction is
    # unknown, but that throws away a capability that is still available: separating
    # vertical from horizontal motion, which no single track can do at any slope. This is
    # the normal case for a site whose aspect has not been resolved from the DEM yet.
    if combined.get("east_up") is None and len(tracks) == 2:
        by_direction = {t["flight_direction"]: t for t in tracks}
        asc, desc = by_direction.get("ASCENDING"), by_direction.get("DESCENDING")
        if asc and desc:
            try:
                combined["east_up"] = insar_combine.decompose_east_up(asc, desc)
                combined.setdefault("warnings", []).append(
                    "east/up decomposed without a slope model; the downslope rate was "
                    "withheld because slope/aspect were unavailable"
                )
            except Exception as exc:
                logger.debug("east/up decomposition unavailable: %s", exc)

    result["combination"] = combined
    east_up = combined.get("east_up") or {}
    if east_up and not east_up.get("refused"):
        result["east_velocity_mm_year"] = east_up.get("east_mm_year")
        result["up_velocity_mm_year"] = east_up.get("up_mm_year")
        result["east_up_assumes_no_north_motion"] = True

    if combined.get("status") == "OK" and combined.get("velocity_mm_year") is not None:
        # A combined downslope rate supersedes the single-track projection: it uses both
        # geometries and carries an uncertainty that has been honestly inflated when the
        # two tracks disagree.
        result["slope_parallel_velocity_mm_year"] = combined["velocity_mm_year"]
        result["slope_parallel_reliable"] = True
        result["slope_parallel_note"] = combined.get("summary")
        result["combined_uncertainty_mm_year"] = combined.get("uncertainty_mm_year")
        # The combination's uncertainty is the one that has been honestly inflated when
        # the two tracks disagree. If it does not travel with the velocity it describes,
        # the Birge inflation is computed and then ignored, and "TRACKS DISAGREE" changes
        # nothing about what the physics engine is told.
        result["slope_parallel_uncertainty_mm_year"] = combined.get("uncertainty_mm_year")
        result["combination_consistent"] = combined.get("consistent")
        result["source"] = (
            f"ASF HyP3 Sentinel-1 burst InSAR, SBAS time-series inversion, "
            f"{'+'.join(sorted(records))} combined"
        )
    return result


# ---------------------------------------------------------------------------
# look-vector projection: LOS -> slope-parallel (downslope) velocity
# ---------------------------------------------------------------------------
# Minimum |look-vector . downslope-unit-vector|. Below this the track barely senses motion
# in the downslope direction (it is looking nearly perpendicular to it), so dividing by a
# near-zero number would amplify noise into a large, unreliable "velocity".
MIN_PROJECTION_SENSITIVITY = 0.15

DEM_TILE_TEMPLATE = (
    "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif"
)


def _lookup_slope_deg(lat: float, lon: float, tolerance_deg: float = 0.01) -> Optional[float]:
    """Find this point's slope in the monitored-site catalog, if it is one of ours.

    Tolerance ~0.01 deg (~1 km) comfortably covers float round-tripping through the state
    JSON file. Returns None (never a guess) for a point outside the catalog.
    """
    try:
        from app.engine.sites_catalog import load_sites_of_interest

        for site in load_sites_of_interest():
            if abs(site.latitude - lat) <= tolerance_deg and abs(site.longitude - lon) <= tolerance_deg:
                return float(site.slope_deg)
    except Exception as exc:
        logger.debug("Site slope lookup failed for (%.4f, %.4f): %s", lat, lon, exc)
    return None


def estimate_aspect_deg(lat: float, lon: float, window_px: int = 5) -> Optional[float]:
    """Local downslope compass bearing (0=N, 90=E, ...) from the open Copernicus DEM.

    Needed because the project's SiteInfo catalog does not carry aspect. Verified data
    source: the same public, no-credential Copernicus DEM GLO-30 bucket already used for
    the Tier A terrain features (see reference_open_geohazard_data_sources memory).
    """
    try:
        import numpy as np
        import rasterio
    except ImportError:
        return None
    try:
        tile = DEM_TILE_TEMPLATE.format(lat=math.floor(lat), lon=math.floor(lon))
        with rasterio.open(tile) as src:
            row, col = src.index(lon, lat)
            half = max(2, int(window_px) // 2)
            if row - half < 0 or col - half < 0 or row + half >= src.height or col + half >= src.width:
                return None
            from rasterio.windows import Window

            arr = src.read(1, window=Window(col - half, row - half, 2 * half + 1, 2 * half + 1))
            arr = arr.astype("float64")
            if np.any(arr <= -9000) or not np.all(np.isfinite(arr)):
                return None
            pixel_m = 30.0  # GLO-30
            gy, gx = np.gradient(arr, pixel_m)
            # Downslope direction is the direction of steepest DESCENT: -gradient.
            dz_dx, dz_dy = -gx[half, half], -gy[half, half]
            if abs(dz_dx) < 1e-9 and abs(dz_dy) < 1e-9:
                return None
            # Compass bearing: 0=N(+y), 90=E(+x).
            aspect_rad = math.atan2(dz_dx, dz_dy)
            return math.degrees(aspect_rad) % 360.0
    except Exception as exc:
        logger.debug("Aspect estimation failed at (%.4f, %.4f): %s", lat, lon, exc)
        return None


def project_los_to_slope_parallel(
    los_value: float,
    lv_theta_rad: float,
    lv_phi_rad: float,
    slope_deg: float,
    aspect_deg: float,
) -> Dict[str, Any]:
    """Project an ambiguous LOS rate onto the downslope direction.

    Geometry, confirmed against HyP3's own product README (not assumed): the look vector
    points from the ground back toward the satellite. lv_theta is its elevation angle above
    the local horizontal (radians, positive = sensor above the surface); lv_phi is its
    azimuth, measured from East increasing toward North (radians). From these:

        look_East = cos(theta) * cos(phi)
        look_North = cos(theta) * sin(phi)
        look_Up = sin(theta)

    The downslope unit vector, from `slope_deg`/`aspect_deg` (aspect = compass bearing the
    slope faces, i.e. the downhill direction, 0=N/90=E):

        down_East = sin(aspect) * cos(slope)
        down_North = cos(aspect) * cos(slope)
        down_Up = -sin(slope)

    If ground motion is assumed to be PURELY along the downslope direction (the standard,
    physically reasonable assumption for landslide/creep - material does not move sideways
    across a slope, it moves down it), the measured LOS value is
    ``los = v_slope * dot(look, down)``, so ``v_slope = los / dot(look, down)``. When the
    dot product is small the track is nearly blind to downslope motion and the division is
    refused rather than amplifying noise into a large fake number.
    """
    theta, phi = float(lv_theta_rad), float(lv_phi_rad)
    slope_rad = math.radians(float(slope_deg))
    aspect_rad = math.radians(float(aspect_deg))

    look = (
        math.cos(theta) * math.cos(phi),
        math.cos(theta) * math.sin(phi),
        math.sin(theta),
    )
    down = (
        math.sin(aspect_rad) * math.cos(slope_rad),
        math.cos(aspect_rad) * math.cos(slope_rad),
        -math.sin(slope_rad),
    )
    sensitivity = look[0] * down[0] + look[1] * down[1] + look[2] * down[2]

    reliable = abs(sensitivity) >= MIN_PROJECTION_SENSITIVITY
    slope_parallel = (float(los_value) / sensitivity) if reliable else None
    return {
        "slope_parallel_value": None if slope_parallel is None else round(slope_parallel, 3),
        "sensitivity": round(float(sensitivity), 3),
        "reliable": reliable,
        "note": (
            "LOS projected onto the downslope direction assuming motion is purely "
            "slope-parallel (the standard creep/landslide assumption)"
            if reliable
            else f"track geometry nearly perpendicular to the slope direction "
                 f"(|sensitivity|={abs(sensitivity):.2f} < {MIN_PROJECTION_SENSITIVITY}); "
                 "projection withheld rather than amplifying noise"
        ),
    }


def register_with_insar_cdse() -> bool:
    """Install the time-series provider as the MEASURED_INSAR source."""
    try:
        from app.ingestion.insar_cdse import register_measured_velocity_provider

        register_measured_velocity_provider(timeseries_velocity_provider)
        return True
    except Exception as exc:
        logger.warning("Could not register time-series velocity provider: %s", exc)
        return False


def summarize(result: Optional[Dict[str, Any]]) -> str:
    """One auditable line for an email or report."""
    if not result:
        return "no time-series velocity (network too small, disconnected, or inconsistent)"
    return (
        f"{result['velocity_mm_year']:+.1f} +/- {result['velocity_stderr_mm_year']:.1f} mm/yr LOS "
        f"({result['n_pairs']} pairs over {result['n_epochs']} epochs, "
        f"{result['span_days']:.0f} d, residual RMS {result['residual_rms_mm']:.1f} mm)"
    )


# ---------------------------------------------------------------------------
# hot-path access: measured velocities for the 30-minute physics loop
# ---------------------------------------------------------------------------
# The loop runs over ~491 sites every 30 minutes. It must not make a network call, must
# not re-read a 500 kB state file per site, and must degrade to the modelled proxy without
# complaint when no network has been inverted for a site (which is the normal case today).

_VELOCITY_CACHE: Dict[str, Any] = {"mtime": None, "values": {}}

# How far a null result has to be from zero before we call it motion: 2 sigma.
SIGNIFICANCE_SIGMA = 2.0


def load_measured_velocities(force: bool = False) -> Dict[str, Dict[str, Any]]:
    """All inverted-network velocities, keyed by ``hyp3.site_key``. Local reads only.

    Cached against the state file's mtime, so a 491-site sweep parses the JSON once and
    only re-parses after the weekly collection run has actually written something.
    """
    if not getattr(settings, "hyp3_enabled", False):
        return {}
    try:
        mtime = hyp3.STATE_PATH.stat().st_mtime
    except OSError:
        return {}
    if not force and _VELOCITY_CACHE["mtime"] == mtime:
        return _VELOCITY_CACHE["values"]

    values: Dict[str, Dict[str, Any]] = {}
    state = hyp3._load_state()
    # Keyed by the BARE site key, because that is what the physics loop looks up. The
    # state file is keyed per geometry, so collapse to one record per site first -
    # otherwise a site with both an ascending and a descending network would be visited
    # twice and stored under keys no caller ever asks for.
    sites: Dict[str, Tuple[float, float]] = {}
    for key, net in (state.get("networks") or {}).items():
        if not net.get("solution"):
            continue
        try:
            lat, lon = float(net["lat"]), float(net["lon"])
        except (KeyError, TypeError, ValueError):
            logger.warning("InSAR network %s has no usable coordinate; skipped", key)
            continue
        sites[hyp3.site_key(lat, lon)] = (lat, lon)

    for site, (lat, lon) in sites.items():
        try:
            record = combined_velocity_for_site(lat, lon, allow_network=False)
        except Exception as exc:  # one bad network must not blind the whole loop
            logger.warning("measured velocity unavailable for %s: %s", site, exc)
            continue
        if record:
            values[site] = record
    _VELOCITY_CACHE["mtime"] = mtime
    _VELOCITY_CACHE["values"] = values
    return values


def resolve_creep_input(
    heuristic_mm_month: float, measured: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Decide what creep rate the physics should actually consume, and say why.

    The modelled proxy is a function of slope, FS and rainfall - the very things the factor
    of safety is already computed from - so feeding it in as "InSAR LOS velocity" told the
    physics nothing it did not already know. A real measurement is independent information.

    Three outcomes, in order:

    1. **Motion detected** (|v| >= 2 sigma): use the measured rate.
    2. **Precise null**: no motion detected, and the 2-sigma detection limit is *below* what
       the proxy claims. The data genuinely rules the proxy's rate out, so use the detection
       limit - the fastest motion still consistent with what we saw. This is the case that
       corrects an over-pessimistic heuristic.
    3. **Imprecise null**: no motion detected, but the detection limit is above the proxy's
       rate, so the measurement cannot distinguish the two. It is uninformative here; keep
       the proxy. A wide null must never be read as "the slope is still".

    An unstable reference annulus disqualifies the measurement entirely - the velocity is
    relative to that annulus, so if it is drifting, the number is measuring the wrong thing.
    """
    heuristic_mm_month = max(0.0, float(heuristic_mm_month))
    out: Dict[str, Any] = {
        "value_mm_month": heuristic_mm_month,
        "heuristic_mm_month": round(heuristic_mm_month, 3),
        "provenance": "MODELLED_PROXY",
        "measured_mm_year": None,
        "uncertainty_mm_year": None,
        "detection_limit_mm_year": None,
        "reason": "no inverted InSAR network for this site",
    }
    if not measured:
        return out

    ref = measured.get("reference_stability") or {}
    if ref.get("checked") and not ref.get("stable", True):
        out["reason"] = (
            f"InSAR network present but its reference annulus is drifting "
            f"{ref.get('drift_mm_year', float('nan')):+.1f} mm/yr - measurement disqualified"
        )
        return out

    # Prefer the slope-parallel (downslope) rate when the viewing geometry supports it;
    # that is the quantity the creep term actually means. Otherwise fall back to raw LOS.
    if measured.get("slope_parallel_reliable") and measured.get("slope_parallel_velocity_mm_year") is not None:
        v_year = float(measured["slope_parallel_velocity_mm_year"])
        geometry = "slope-parallel"
        # Must be the SLOPE-PARALLEL sigma, not the line-of-sight one. They are different
        # physical quantities and the whole MEASURED-vs-PROXY decision below is a
        # comparison between a velocity and its own uncertainty.
        sigma = measured.get("slope_parallel_uncertainty_mm_year")
        if sigma is None:
            out["reason"] = (
                "slope-parallel velocity offered without a matching slope-parallel "
                "uncertainty; refusing to pair it with the line-of-sight sigma"
            )
            return out
    elif measured.get("los_velocity_mm_year") is not None:
        v_year = float(measured["los_velocity_mm_year"])
        geometry = "line-of-sight"
        sigma = measured.get("uncertainty_mm_year")
    else:
        return out
    if sigma is None:
        return out
    sigma = float(sigma)
    detection_limit = SIGNIFICANCE_SIGMA * sigma

    out["measured_mm_year"] = round(v_year, 2)
    out["uncertainty_mm_year"] = round(sigma, 2)
    out["detection_limit_mm_year"] = round(detection_limit, 2)
    out["geometry"] = geometry
    out["age_days"] = measured.get("age_days")

    # Magnitude, not signed value: the downstream softening term is a strain-rate proxy, and
    # the sign of a LOS rate encodes which way the satellite was looking, not whether the
    # slope is in trouble.
    if abs(v_year) >= detection_limit and detection_limit > 0.0:
        out["value_mm_month"] = abs(v_year) / 12.0
        out["provenance"] = "MEASURED_INSAR"
        out["reason"] = (
            f"measured {v_year:+.1f} +/- {sigma:.1f} mm/yr {geometry} "
            f"(significant at {SIGNIFICANCE_SIGMA:.0f} sigma)"
        )
        return out

    limit_mm_month = detection_limit / 12.0
    if limit_mm_month < heuristic_mm_month:
        out["value_mm_month"] = limit_mm_month
        out["provenance"] = "MEASURED_INSAR_NULL"
        out["reason"] = (
            f"no motion detected ({v_year:+.1f} +/- {sigma:.1f} mm/yr {geometry}); "
            f"rules out the modelled {heuristic_mm_month:.1f} mm/month, so the "
            f"{SIGNIFICANCE_SIGMA:.0f}-sigma limit {limit_mm_month:.2f} mm/month is used"
        )
        return out

    out["reason"] = (
        f"InSAR inconclusive here: detection limit {limit_mm_month:.1f} mm/month exceeds "
        f"the modelled {heuristic_mm_month:.1f} mm/month, so it cannot confirm or deny it"
    )
    return out
