"""Combine several InSAR tracks (ascending + descending) into one downslope rate.

WHY THIS EXISTS
    A single Sentinel-1 track measures one number per pixel: the projection of the true 3-D
    surface motion onto that track's line of sight. Two things follow, and both are the
    reason this module exists.

    1. ONE TRACK CANNOT BE CHECKED. Converting LOS to a downslope rate requires ASSUMING the
       motion is purely slope-parallel (see insar_timeseries.project_los_to_slope_parallel).
       With one observation and one unknown there are zero degrees of freedom, so the
       assumption fits perfectly by construction and the fit tells you nothing. Any residual
       is identically zero whether the assumption is true or wildly false. A second track
       with a DIFFERENT look geometry gives one degree of freedom, and the chi-square of that
       one spare observation is the first real test the assumption has ever faced here.

    2. TWO TRACKS ARE TIGHTER THAN ONE. The weighted least-squares combination of N LOS rates
       has formal uncertainty 1/sqrt(sum s_i^2/sigma_i^2), which is strictly smaller than any
       single track's sigma_i/|s_i|. That is the headline of the feature and
       test_insar_combine.py pins it analytically rather than taking it on faith.

WHAT IS MEASURED AND WHAT IS ASSUMED
    VERIFIED (geometry taken from the HyP3 products themselves, not modelled): each track's
    look vector (lv_theta, lv_phi) and its LOS velocity + formal uncertainty from the SBAS
    inversion.
    ASSUMED, and stated in every return value that depends on it:
      * combine_slope_parallel() assumes motion is purely downslope. That assumption is now
        TESTABLE (chi-square) rather than merely declared, and when the data contradict it
        the reported uncertainty is inflated instead of the tight number being published.
      * decompose_east_up() assumes the north component is zero. Two LOS observations cannot
        resolve three unknowns; there is no way around this with two tracks, so the
        assumption is returned explicitly alongside every number it produced.
      * classify_flight_direction() assumes Sentinel-1's fixed right-looking attitude. It is
        only used when a track does not declare its own direction, and the output always
        records which tracks were declared and which were inferred.

PURITY
    Deliberately no network, no state files, no settings, no I/O, no numpy - stdlib `math`
    only. Everything here is a function of its arguments, which is what makes the tests below
    actual proofs of the mathematics rather than snapshots of a pipeline. Integration with
    insar_timeseries.py (which tracks exist, how they are keyed) lives on the caller's side.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# Mirror of insar_timeseries.MIN_PROJECTION_SENSITIVITY. Deliberately duplicated rather than
# imported: importing that module drags in app.config settings, insar_hyp3 state handling and
# rasterio, which would destroy this module's purity (and its import-time cost) for the sake
# of one float. test_insar_combine.py asserts the two values are identical, so they cannot
# silently drift apart.
MIN_PROJECTION_SENSITIVITY = 0.15

# Refusal threshold for the east-up decomposition, chosen from a numerical experiment
# (documented in decompose_east_up) rather than picked by eye. Sweeping the real Sentinel-1
# IW incidence range (29-46 deg) with +/-15 deg of look-azimuth variation gives two cleanly
# separated populations: genuine ascending x descending pairs never exceed cond(G) = 2.04,
# while the most benign SAME-direction pair in the same sweep is 5.70 (the real same-direction
# pair at glc_105 is 11.5). 3.5 sits at the geometric midpoint of that gap (3.41) and caps
# the 1-sigma noise amplification into dE/dU at 2.45x the input 1-sigma.
MAX_EAST_UP_CONDITION_NUMBER = 3.5

# Two velocities are rates averaged over their own observation windows. If those windows
# barely overlap, the two numbers describe different periods of the slope's life and
# combining them is not a measurement of anything - it is an average of two different things.
# 0.5 = the shorter track's window must be at least half-covered by the other's.
MIN_TEMPORAL_OVERLAP_FRACTION = 0.5

# A real SBAS velocity uncertainty lives between roughly 0.1 and 100 mm/yr. These bounds
# are deliberately far wider than that: they exist to reject values that break the
# arithmetic (1/sigma**2 raises ZeroDivisionError below ~2e-162 and OverflowError above
# ~1.4e154) and values that silently collapse the combined uncertainty to exactly zero.
# Above this, the consistency test cannot see contamination on the scale the system acts
# on (SAR_SLOPE_CREEP_ACCELERATION triggers at 10 mm/month = 120 mm/yr), so a "pass" is
# not evidence of anything. See vertical_blind_spot.
VERTICAL_BLINDNESS_LIMIT_MM_YEAR = 100.0

MIN_PLAUSIBLE_SIGMA_MM_YEAR = 1e-6
MAX_PLAUSIBLE_SIGMA_MM_YEAR = 1e6

# Chi-square 95th-percentile critical values, dof 1..10. Hardcoded even though scipy IS in
# backend/requirements.txt: this module is pure math with a stdlib-only import list, and a
# 10-entry table is not worth importing scipy.stats for. test_insar_combine.py checks every
# entry (and the >10 approximation) against scipy.stats.chi2.ppf, so the table cannot be
# quietly wrong.
_CHI2_CRITICAL_95: Dict[int, float] = {
    1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
    6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
}
_Z_95 = 1.6448536269514722  # standard-normal 95th percentile, for Wilson-Hilferty


def chi2_critical_95(dof: int) -> Optional[float]:
    """95% chi-square critical value for `dof` degrees of freedom, or None for dof < 1.

    Table lookup to dof=10, then the Wilson-Hilferty cube-root approximation
    k*(1 - 2/(9k) + z*sqrt(2/(9k)))^3, which is accurate to well under 1% in this tail and is
    verified against scipy in the tests. dof < 1 returns None, not a number: with no spare
    observation there is nothing to test and a critical value would imply otherwise.
    """
    try:
        dof = int(dof)
    except (TypeError, ValueError):
        return None
    if dof < 1:
        return None
    if dof in _CHI2_CRITICAL_95:
        return _CHI2_CRITICAL_95[dof]
    k = float(dof)
    return k * (1.0 - 2.0 / (9.0 * k) + _Z_95 * math.sqrt(2.0 / (9.0 * k))) ** 3


# ---------------------------------------------------------------------------
# geometry - must match insar_timeseries.project_los_to_slope_parallel exactly
# ---------------------------------------------------------------------------


def look_vector(lv_theta_rad: float, lv_phi_rad: float) -> Tuple[float, float, float]:
    """Unit vector from the ground TOWARD the satellite, as (East, North, Up).

    lv_theta = elevation above the local horizontal (rad); lv_phi = azimuth measured from
    East increasing toward North (rad). This is HyP3's own documented convention, already
    relied on by insar_timeseries.project_los_to_slope_parallel - the two implementations are
    cross-checked against each other in the tests.
    """
    theta, phi = float(lv_theta_rad), float(lv_phi_rad)
    return (
        math.cos(theta) * math.cos(phi),
        math.cos(theta) * math.sin(phi),
        math.sin(theta),
    )


def downslope_vector(slope_deg: float, aspect_deg: float) -> Tuple[float, float, float]:
    """Unit vector pointing down the slope, as (East, North, Up).

    aspect_deg is the compass bearing the slope faces downhill (0=N, 90=E).
    """
    slope = math.radians(float(slope_deg))
    aspect = math.radians(float(aspect_deg))
    return (
        math.sin(aspect) * math.cos(slope),
        math.cos(aspect) * math.cos(slope),
        -math.sin(slope),
    )


def projection_sensitivity(
    lv_theta_rad: float, lv_phi_rad: float, slope_deg: float, aspect_deg: float
) -> float:
    """s = dot(look, downslope): how much of a downslope motion this track actually sees.

    A purely downslope velocity v_slope appears in this track's line of sight as
    v_los = s * v_slope. |s| near 1 means the track is looking straight along the slope;
    |s| near 0 means it is looking across it and is nearly blind to the motion.
    """
    look = look_vector(lv_theta_rad, lv_phi_rad)
    down = downslope_vector(slope_deg, aspect_deg)
    return look[0] * down[0] + look[1] * down[1] + look[2] * down[2]


def classify_flight_direction(lv_phi_rad: Optional[float]) -> str:
    """"ASCENDING" / "DESCENDING" / "UNKNOWN" from the look-vector azimuth alone.

    ASSUMPTION, stated because it is the only thing making this possible: Sentinel-1 is
    always RIGHT-looking. Given that, the sign of the East component of the ground->satellite
    look vector fixes the orbit direction:
      * satellite to the WEST of the target (cos(phi) < 0) -> the sensor was on a northbound
        (ascending) pass looking east;
      * satellite to the EAST (cos(phi) > 0) -> southbound (descending) pass looking west.
    Consistent with the real geometry at glc_105: ascending lv_phi ~ -2.9 rad (cos = -0.97),
    descending ~ -0.23 rad (cos = +0.97).

    Near |cos(phi)| ~ 0 the look direction is almost due north/south, the sign carries no
    information, and this returns "UNKNOWN" rather than a coin flip. Real Sentinel-1 IW
    geometry never lands there (|cos(phi)| > 0.9 in practice), so UNKNOWN here means the
    input is not what it claims to be. Always prefer a track's declared flight_direction;
    this is the fallback, and callers are told which tracks were inferred.
    """
    if lv_phi_rad is None:
        return "UNKNOWN"
    try:
        c = math.cos(float(lv_phi_rad))
    except (TypeError, ValueError):
        return "UNKNOWN"
    if not math.isfinite(c) or abs(c) < 0.1:  # within ~5.7 deg of due north/south
        return "UNKNOWN"
    return "ASCENDING" if c < 0 else "DESCENDING"


# ---------------------------------------------------------------------------
# input parsing helpers
# ---------------------------------------------------------------------------

_DIRECTION_ALIASES = {
    "ASCENDING": "ASCENDING", "ASC": "ASCENDING", "A": "ASCENDING",
    "DESCENDING": "DESCENDING", "DESC": "DESCENDING", "DSC": "DESCENDING", "D": "DESCENDING",
}


def _finite(value: Any) -> Optional[float]:
    """float(value) if it is a real finite number, else None. Never raises, never guesses."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _track_key(track: Dict[str, Any], index: int) -> str:
    key = track.get("key") if isinstance(track, dict) else None
    if isinstance(key, str) and key.strip():
        return key.strip()
    return f"track[{index}]"


def _normalize_direction(track: Dict[str, Any]) -> Tuple[str, str]:
    """(direction, provenance) where provenance is "declared", "inferred" or "unknown".

    Provenance is returned, not folded away, because an inferred direction rests on the
    right-looking assumption in classify_flight_direction and a declared one does not. The
    caller is entitled to know which it got.
    """
    raw = track.get("flight_direction")
    if isinstance(raw, str):
        hit = _DIRECTION_ALIASES.get(raw.strip().upper())
        if hit:
            return hit, "declared"
    inferred = classify_flight_direction(_finite(track.get("lv_phi_rad")))
    if inferred != "UNKNOWN":
        return inferred, "inferred"
    return "UNKNOWN", "unknown"


def _velocity_of(track: Dict[str, Any]) -> Optional[float]:
    """The track's LOS velocity in mm/yr.

    Accepts `velocity_mm_year` or the equivalent `los_velocity_mm_year` spelling, because
    insar_timeseries uses the first inside its SBAS solution dict and the second in the
    provider result it publishes; both are the same quantity and refusing one of them would
    just move a rename into the caller.
    """
    for name in ("velocity_mm_year", "los_velocity_mm_year"):
        value = _finite(track.get(name))
        if value is not None:
            return value
    return None


def _parse_track(track: Any, index: int) -> Dict[str, Any]:
    """Pull one track into a flat, validated record. Collects problems; never raises."""
    if not isinstance(track, dict):
        return {
            "index": index,
            "key": f"track[{index}]", "ok": False, "problem": "not a dict",
            "direction": "UNKNOWN", "direction_provenance": "unknown",
            "velocity": None, "sigma": None, "theta": None, "phi": None,
            "t_start_utc": None, "t_end_utc": None,
        }
    direction, provenance = _normalize_direction(track)
    velocity = _velocity_of(track)
    sigma = _finite(track.get("uncertainty_mm_year"))
    theta = _finite(track.get("lv_theta_rad"))
    phi = _finite(track.get("lv_phi_rad"))

    problem: Optional[str] = None
    if velocity is None:
        problem = "no finite velocity_mm_year"
    elif theta is None or phi is None:
        problem = "missing look vector (lv_theta_rad / lv_phi_rad)"
    elif sigma is None:
        problem = "no finite uncertainty_mm_year"
    elif sigma <= 0.0:
        # A non-positive sigma is a weight of 1/sigma^2 = infinity: that one track would
        # silently become the only track, and the combined uncertainty would collapse to
        # zero. Refuse it rather than let it dominate.
        problem = f"non-positive uncertainty_mm_year ({sigma:g}); weight 1/sigma^2 is undefined"
    elif not (MIN_PLAUSIBLE_SIGMA_MM_YEAR <= sigma <= MAX_PLAUSIBLE_SIGMA_MM_YEAR):
        # `sigma > 0` is not enough. Below ~2e-162 the weight 1/sigma**2 raises
        # ZeroDivisionError and above ~1.4e154 it raises OverflowError, so the two
        # "never raises" guarantees in this module were false for extreme inputs. Worse,
        # a merely very small sigma (say 1e-154) passes the positivity check and then
        # drives the combined uncertainty to exactly 0.0 with status OK and nothing
        # excluded - the precise collapse the check above exists to prevent. An SBAS
        # velocity uncertainty outside 1e-6..1e6 mm/yr is not a measurement, it is a bug
        # upstream, so it is refused by range rather than by sign.
        problem = (
            f"implausible uncertainty_mm_year ({sigma:g}); outside "
            f"{MIN_PLAUSIBLE_SIGMA_MM_YEAR:g}..{MAX_PLAUSIBLE_SIGMA_MM_YEAR:g} mm/yr"
        )

    return {
        "index": index,
        "key": _track_key(track, index),
        "ok": problem is None,
        "problem": problem,
        "direction": direction,
        "direction_provenance": provenance,
        "velocity": velocity,
        "sigma": sigma,
        "theta": theta,
        "phi": phi,
        "t_start_utc": _finite(track.get("t_start_utc")),
        "t_end_utc": _finite(track.get("t_end_utc")),
    }


def _r(value: Optional[float], nd: int = 6) -> Optional[float]:
    """Round for readable output without throwing away precision the tests rely on."""
    return None if value is None else round(float(value), nd)


# ---------------------------------------------------------------------------
# 1. the primary estimator: weighted least squares for one downslope rate
# ---------------------------------------------------------------------------


def vertical_blind_spot(
    used: Sequence[Dict[str, Any]], critical: Optional[float]
) -> Dict[str, Any]:
    """How much pure VERTICAL motion could pass the consistency test unnoticed?

    This is the honest companion to the chi-square verdict, and without it that verdict
    is routinely misread. Two line-of-sight observations impose ONE linear constraint on
    a THREE-component motion vector, so two whole directions stay invisible. For an
    ascending/descending pair the difference of the look vectors is almost purely East
    (`look_asc - look_desc = (-1.110, +0.000, -0.001)` for the measured Sentinel-1
    geometry), so the test's entire discriminating power runs along East. It therefore
    scales as |sin(aspect)| * cos(slope) and is **exactly zero on north- and south-facing
    slopes**, where a pure vertical signal is reported as a downslope creep rate with
    "tracks agree".

    The arithmetic. With true motion `v_down * down_hat + m * up_hat`, each track sees
    `v_i = s_i * v_down + m * a_i` where `a_i = look_i . up_hat = sin(theta_i)`. Fitting
    only `v_down` by weighted least squares leaves residuals `r_i = m * (a_i - s_i * beta)`
    with `beta = sum(w s a) / sum(w s^2)`, so `chi2 = m^2 * K` with
    `K = sum(w_i (a_i - s_i beta)^2)`. The smallest detectable vertical motion is then
    `sqrt(critical / K)`, and `K -> 0` is the blind case.
    """
    out: Dict[str, Any] = {
        "min_detectable_vertical_mm_year": None,
        "blind_to_vertical": None,
        "note": "",
    }
    if critical is None or len(used) < 2:
        out["note"] = "no spare observation, so the assumption is untested entirely"
        return out
    try:
        weights = [1.0 / r["sigma"] ** 2 for r in used]
        sens = [r["sensitivity"] for r in used]
        a = [math.sin(r["theta"]) for r in used]
        denom = sum(w * x * x for w, x in zip(weights, sens))
        if denom <= 0.0 or not math.isfinite(denom):
            return out
        beta = sum(w * x * y for w, x, y in zip(weights, sens, a)) / denom
        k = sum(w * (ai - si * beta) ** 2 for w, ai, si in zip(weights, a, sens))
    except (ZeroDivisionError, OverflowError, ValueError, KeyError, TypeError):
        return out

    if k <= 0.0 or not math.isfinite(k):
        out["blind_to_vertical"] = True
        out["note"] = (
            "the two geometries respond IDENTICALLY to vertical motion here, so the "
            "consistency test cannot see it at all: any amount of uplift or subsidence "
            "would be reported as downslope creep with the tracks appearing to agree"
        )
        return out

    m_min = math.sqrt(critical / k)
    out["min_detectable_vertical_mm_year"] = round(m_min, 3)
    # The flag is a judgement about relevance, so it is tied to the number this system
    # actually acts on: SAR_SLOPE_CREEP_ACCELERATION fires at 10 mm/month = 120 mm/yr.
    # A consistency test that cannot see vertical contamination below roughly that scale
    # cannot protect the decision it is supposed to protect, so it is reported as no test
    # at all. Note the degradation is CONTINUOUS - `min_detectable_vertical_mm_year` is
    # always published, and is the number to read; the boolean only drives the
    # consistent=True -> None downgrade.
    out["blind_to_vertical"] = m_min > VERTICAL_BLINDNESS_LIMIT_MM_YEAR
    out["note"] = (
        f"the consistency test would only notice non-downslope (vertical) motion larger "
        f"than {m_min:.1f} mm/yr; anything smaller is absorbed into the reported "
        f"downslope rate and still shows the tracks as agreeing"
    )
    return out


def combine_slope_parallel(
    tracks: Sequence[Dict[str, Any]],
    slope_deg: Optional[float],
    aspect_deg: Optional[float],
) -> Dict[str, Any]:
    """Weighted least squares for ONE unknown (the downslope rate) from N LOS observations.

    Model: track i with sensitivity s_i = dot(look_i, downslope) observes v_i = s_i * v_slope
    plus noise of standard deviation sigma_i. The WLS solution and its formal uncertainty are

        v_hat  = sum(s_i*v_i/sigma_i^2) / sum(s_i^2/sigma_i^2)
        sigma_hat = 1 / sqrt(sum(s_i^2/sigma_i^2))

    sigma_hat is strictly smaller than every single-track sigma_i/|s_i| - that inequality is
    the entire justification for this feature and is proved, not asserted, in the tests.

    THE CONSISTENCY TEST (what a single track can never give you). N observations and one
    unknown leave dof = N-1 degrees of freedom, so

        chi2 = sum (v_i - s_i*v_hat)^2 / sigma_i^2

    is a real test of the "motion is purely downslope" assumption. If chi2 exceeds the 95%
    critical value the data CONTRADICT that assumption - most likely the real motion has a
    large component the slope model does not allow (e.g. subsidence, or an aspect estimate
    that is wrong). Publishing the tight sigma_hat in that situation would be a lie, so:
    `consistent` is set False and the reported uncertainty is inflated by the Birge ratio
    sqrt(chi2/dof) (the standard scaled-covariance correction), which degrades the number
    honestly instead of hiding the disagreement. `formal_uncertainty_mm_year` keeps the
    uninflated value so the caller can see both.

    Tracks with |s_i| < MIN_PROJECTION_SENSITIVITY are excluded with a stated reason: they
    are nearly blind to downslope motion, and giving them weight means dividing by a
    near-zero number, which amplifies their noise instead of adding information.

    Never raises. Returns a dict; `velocity_mm_year` is None whenever nothing could be
    estimated, and `excluded` always says why for every track that did not contribute.
    """
    out: Dict[str, Any] = {
        "status": "OK",
        "velocity_mm_year": None,
        "uncertainty_mm_year": None,
        "formal_uncertainty_mm_year": None,
        "n_tracks_input": len(tracks) if tracks is not None else 0,
        "n_tracks_used": 0,
        "tracks_used": [],
        "excluded": [],
        "per_track": [],
        "chi_square": None,
        "dof": 0,
        "chi_square_critical_95": None,
        "consistent": None,
        "inflation_factor": 1.0,
        "best_single_uncertainty_mm_year": None,
        "improvement_factor": None,
        "slope_deg": None,
        "aspect_deg": None,
        "note": "",
    }

    slope = _finite(slope_deg)
    aspect = _finite(aspect_deg)
    if not tracks:
        out["status"] = "NO_TRACKS"
        out["note"] = "no tracks supplied; nothing to combine"
        return out
    if slope is None or aspect is None:
        # Without slope+aspect there is no downslope direction, hence no sensitivity, hence
        # no projection. Returning None is correct; a default slope would be fabrication.
        out["status"] = "NO_SLOPE_GEOMETRY"
        out["note"] = (
            "slope_deg/aspect_deg unavailable, so the downslope direction is undefined; "
            "slope-parallel combination withheld rather than assuming a slope"
        )
        return out
    out["slope_deg"] = _r(slope, 2)
    out["aspect_deg"] = _r(aspect, 2)

    used: List[Dict[str, Any]] = []
    for index, raw in enumerate(tracks):
        rec = _parse_track(raw, index)
        if not rec["ok"]:
            out["excluded"].append(
                {"key": rec["key"], "reason": rec["problem"], "sensitivity": None}
            )
            continue
        s = projection_sensitivity(rec["theta"], rec["phi"], slope, aspect)
        rec["sensitivity"] = s
        if abs(s) < MIN_PROJECTION_SENSITIVITY:
            out["excluded"].append({
                "key": rec["key"],
                "reason": (
                    f"look geometry nearly perpendicular to the slope direction "
                    f"(|sensitivity|={abs(s):.3f} < {MIN_PROJECTION_SENSITIVITY}); including it "
                    f"would amplify its noise rather than add information"
                ),
                "sensitivity": _r(s, 4),
            })
            continue
        used.append(rec)

    if not used:
        out["status"] = "NO_USABLE_TRACKS"
        out["note"] = (
            f"all {out['n_tracks_input']} supplied track(s) were excluded; see `excluded` for "
            f"the reason on each"
        )
        return out

    # --- the weighted least squares itself -------------------------------------------
    numerator = sum(r["sensitivity"] * r["velocity"] / r["sigma"] ** 2 for r in used)
    denominator = sum(r["sensitivity"] ** 2 / r["sigma"] ** 2 for r in used)
    if denominator <= 0.0 or not math.isfinite(denominator) or not math.isfinite(numerator):
        # Unreachable given the validation above (|s| >= 0.15, sigma > 0, all finite), but a
        # division that could produce an inf must be guarded, not assumed away.
        out["status"] = "DEGENERATE_NORMAL_EQUATION"
        out["note"] = "normal equation is degenerate; no velocity reported"
        return out

    v_hat = numerator / denominator
    formal_sigma = 1.0 / math.sqrt(denominator)

    # --- consistency ------------------------------------------------------------------
    dof = len(used) - 1
    chi2 = sum(
        ((r["velocity"] - r["sensitivity"] * v_hat) / r["sigma"]) ** 2 for r in used
    )
    critical = chi2_critical_95(dof)
    inflation = 1.0
    if dof >= 1 and critical is not None:
        consistent = chi2 <= critical
        if chi2 / dof > 1.0:
            # Birge ratio / scaled covariance: the spread of the observations is larger
            # than their quoted sigmas can explain, so the quoted sigmas (or the model)
            # are too optimistic by about this factor.
            #
            # Applied continuously from chi2/dof > 1, NOT only once the 95% test fails.
            # Inflating at the critical value made the published uncertainty a step
            # function: at dof=1 a 7e-5 relative change in the data doubled it (1.00 ->
            # 1.96) and no factor between those two was ever reachable. Two tracks whose
            # own estimates differ by 1.9 sigma would publish the full tight number with
            # consistent=True. Scaling from chi2/dof > 1 is the standard continuous form.
            inflation = math.sqrt(chi2 / dof)
    else:
        # dof == 0: one observation, one unknown, a perfect fit by construction. chi2 is
        # identically zero and proves nothing, so the verdict is None - not True.
        consistent = None

    # A passing chi-square means "no evidence against", never "assumption verified". How
    # much it is actually worth depends entirely on geometry, so the power is published
    # next to the verdict rather than left for the reader to assume.
    blind_spot = vertical_blind_spot(used, critical)
    out["vertical_blind_spot"] = blind_spot

    # If the test cannot detect realistic contamination, it has not tested anything, and
    # reporting consistent=True would be read as evidence that the downslope assumption
    # holds. It is not evidence - it is the absence of a measurement. Downgrade the
    # verdict to "untested", exactly as dof==0 already is.
    #
    # Deliberately NOT withholding the velocity itself: a single track at the same aspect
    # is equally blind and has no test at all, so refusing here would be stricter than
    # the path it replaces. The number stays; the false reassurance goes.
    if consistent is True and blind_spot.get("blind_to_vertical"):
        consistent = None
        out["consistency_untestable"] = True

    reported_sigma = formal_sigma * inflation

    # The tightest single-track estimate, kept so the improvement is measurable rather than
    # claimed: a lone track's downslope uncertainty is sigma_i/|s_i|.
    best_single = min(r["sigma"] / abs(r["sensitivity"]) for r in used)

    out.update({
        "velocity_mm_year": _r(v_hat),
        "uncertainty_mm_year": _r(reported_sigma),
        "formal_uncertainty_mm_year": _r(formal_sigma),
        "n_tracks_used": len(used),
        "tracks_used": [r["key"] for r in used],
        "chi_square": _r(chi2),
        "dof": dof,
        "chi_square_critical_95": _r(critical, 4) if critical is not None else None,
        "consistent": consistent,
        "inflation_factor": _r(inflation),
        "best_single_uncertainty_mm_year": _r(best_single),
        "improvement_factor": _r(best_single / formal_sigma),
        "per_track": [
            {
                "key": r["key"],
                "flight_direction": r["direction"],
                "direction_provenance": r["direction_provenance"],
                "sensitivity": _r(r["sensitivity"], 4),
                "velocity_mm_year": _r(r["velocity"]),
                "uncertainty_mm_year": _r(r["sigma"]),
                # What this track alone would have claimed, and how far its observation sits
                # from the joint fit in units of its own sigma. The second number is where a
                # disagreement becomes attributable to a specific track.
                "single_track_slope_parallel_mm_year": _r(r["velocity"] / r["sensitivity"]),
                "residual_mm_year": _r(r["velocity"] - r["sensitivity"] * v_hat),
                "normalized_residual": _r((r["velocity"] - r["sensitivity"] * v_hat) / r["sigma"]),
            }
            for r in used
        ],
    })

    if consistent is None:
        out["note"] = (
            f"single usable track: downslope rate is the LOS rate divided by its sensitivity "
            f"({used[0]['sensitivity']:.3f}). With one observation and one unknown there are no "
            f"degrees of freedom, so the slope-parallel assumption is UNTESTED here, not confirmed"
        )
    elif consistent:
        out["note"] = (
            f"{len(used)} tracks combined by weighted least squares; chi2={chi2:.2f} on dof={dof} "
            f"is within the 95% critical value {critical:.3f}, so there is NO EVIDENCE AGAINST "
            f"the purely-downslope assumption. That is not the same as confirming it: "
            + (blind_spot["note"] or "the power of this test is unknown here")
        )
    else:
        out["note"] = (
            f"TRACKS DISAGREE: chi2={chi2:.2f} on dof={dof} exceeds the 95% critical value "
            f"{critical:.3f}, so the purely-downslope assumption is CONTRADICTED by the data "
            f"(the real motion likely has a component the slope model does not allow, or the "
            f"aspect used here is wrong). The formal uncertainty "
            f"{formal_sigma:.3f} mm/yr is therefore NOT trustworthy and has been inflated by the "
            f"Birge ratio sqrt(chi2/dof)={inflation:.2f} to {reported_sigma:.3f} mm/yr"
        )
    return out


# ---------------------------------------------------------------------------
# 2. the diagnostic: east/up decomposition from one ascending + one descending track
# ---------------------------------------------------------------------------


def _condition_number_2x2(m: Sequence[Sequence[float]]) -> float:
    """2-norm condition number of a 2x2 matrix, in stdlib math (no numpy import).

    The singular values of G are the square roots of the eigenvalues of G^T G, and a 2x2
    symmetric matrix [[a,b],[b,d]] has eigenvalues ((a+d) +/- sqrt((a-d)^2 + 4b^2))/2.
    Returns inf for a singular (or numerically singular) matrix, which is the correct answer
    and is what the caller's threshold check needs to see.
    """
    (g11, g12), (g21, g22) = m[0], m[1]
    a = g11 * g11 + g21 * g21
    b = g11 * g12 + g21 * g22
    d = g12 * g12 + g22 * g22
    trace, diff = a + d, a - d
    root = math.sqrt(max(diff * diff + 4.0 * b * b, 0.0))
    lambda_max = (trace + root) / 2.0
    lambda_min = (trace - root) / 2.0
    if lambda_min <= 0.0 or not math.isfinite(lambda_max):
        return math.inf
    return math.sqrt(lambda_max / lambda_min)


def decompose_east_up(
    asc: Optional[Dict[str, Any]], desc: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve two LOS rates into East and Up components, ASSUMING the North component is 0.

    Two observations cannot resolve three unknowns. Something must be fixed, and north is the
    conventional choice for Sentinel-1 because its near-polar orbit makes every track almost
    blind to north-south motion anyway (the north coefficient cos(theta)*sin(phi) is small
    AND nearly equal on ascending and descending, so even a third track barely helps). This
    is a genuine limitation, not a detail: `north_assumed_zero` is True in every result and
    any real northward motion leaks into dE and dU. Treat the output as a diagnostic that
    says which WAY the surface is moving, not as three-component GNSS.

        G = [[cos(theta_a)*cos(phi_a), sin(theta_a)],
             [cos(theta_d)*cos(phi_d), sin(theta_d)]]        G*[dE, dU]^T = [v_a, v_d]^T

    with covariance propagated properly as C_x = (G^T C_d^-1 G)^-1, C_d = diag(sigma^2).

    THE GUARD. If the two geometries are not genuinely complementary - the classic case being
    two tracks from the SAME orbit direction, which happens the moment a site has three
    stacks and only the burst/swath key separates them - then G is ill-conditioned. The solve
    still succeeds and still returns a formal sigma, but both numbers are nearly pure
    amplified noise wearing a confident error bar. So the condition number is computed and
    the decomposition is REFUSED above MAX_EAST_UP_CONDITION_NUMBER.

    That threshold (3.5) comes from a numerical experiment, not a guess. Sweeping the real
    Sentinel-1 IW incidence range (29-46 deg) over every pairing, with look-azimuth variation
    of +/-8 deg (realistic) and +/-15 deg (deliberately generous) around the measured
    glc_105 azimuths:
      * GENUINE ascending x descending pairs: cond(G) in [1.01, 1.93] at +/-8 deg and never
        above 2.04 at +/-15 deg. So 3.5 rejects no real asc/desc pair, with 1.7x of headroom.
      * SAME-direction pairs (the mistake this guard exists for - a site with three stacks
        where only burst/swath separates them): the most benign one anywhere in the sweep is
        5.70, the real glc_105 pair (both ASCENDING, 337397/IW1 near range vs 118306/IW3 far
        range) is 11.5, and two tracks in the same subswath reach 57.7. All refused, with
        1.6x of margin to the most benign.
      * 3.5 is the geometric midpoint of that gap (sqrt(2.04*5.70) = 3.41) and corresponds to
        capping noise amplification at 2.45x the input 1-sigma: a 1 mm/yr LOS uncertainty can
        become at most a 2.45 mm/yr east uncertainty, which is the most a "measurement" of
        this kind can inflate and still be worth reporting.
    test_insar_combine.py re-runs both halves of that sweep, so a future edit that moved the
    threshold into either population would fail the suite.

    Never raises and never returns None - a refusal is a dict with east_mm_year/up_mm_year
    set to None, `refused` True and a `reason`, because a bare None cannot tell the caller
    WHY and this codebase treats a stated refusal as a result.
    """
    out: Dict[str, Any] = {
        "refused": True,
        "reason": None,
        "east_mm_year": None,
        "east_sigma_mm_year": None,
        "up_mm_year": None,
        "up_sigma_mm_year": None,
        "east_up_correlation": None,
        "condition_number": None,
        "condition_number_limit": MAX_EAST_UP_CONDITION_NUMBER,
        "north_assumed_zero": True,
        "assumption": (
            "north component assumed EXACTLY ZERO - two line-of-sight observations cannot "
            "resolve three components; any real north-south motion leaks into dE and dU"
        ),
        "tracks": [],
        "flight_directions": [],
        "note": "",
    }

    records = [_parse_track(asc, 0), _parse_track(desc, 1)]
    out["tracks"] = [r["key"] for r in records]
    out["flight_directions"] = [r["direction"] for r in records]

    bad = [f"{r['key']}: {r['problem']}" for r in records if not r["ok"]]
    if bad:
        out["reason"] = "unusable input track(s) -> " + "; ".join(bad)
        out["note"] = out["reason"]
        return out

    a, d = records
    g = (
        (math.cos(a["theta"]) * math.cos(a["phi"]), math.sin(a["theta"])),
        (math.cos(d["theta"]) * math.cos(d["phi"]), math.sin(d["theta"])),
    )
    cond = _condition_number_2x2(g)
    out["condition_number"] = None if math.isinf(cond) else _r(cond, 4)

    if not (cond <= MAX_EAST_UP_CONDITION_NUMBER):  # also catches inf / nan
        shown = "infinite (singular)" if math.isinf(cond) else f"{cond:.2f}"
        out["reason"] = (
            f"look geometries are not complementary: condition number {shown} exceeds the "
            f"{MAX_EAST_UP_CONDITION_NUMBER} limit"
            + (
                f" (both tracks are {a['direction']})"
                if a["direction"] == d["direction"] and a["direction"] != "UNKNOWN"
                else ""
            )
            + ". An east/up split from these two observations would be amplified noise with a "
              "confident-looking error bar, so it is withheld"
        )
        out["note"] = out["reason"]
        return out

    # C_x = (G^T C_d^-1 G)^-1, with C_d = diag(sigma_a^2, sigma_d^2).
    wa, wd = 1.0 / a["sigma"] ** 2, 1.0 / d["sigma"] ** 2
    n11 = g[0][0] * g[0][0] * wa + g[1][0] * g[1][0] * wd
    n12 = g[0][0] * g[0][1] * wa + g[1][0] * g[1][1] * wd
    n22 = g[0][1] * g[0][1] * wa + g[1][1] * g[1][1] * wd
    det = n11 * n22 - n12 * n12
    if det <= 0.0 or not math.isfinite(det):
        # Unreachable once cond <= 5, but an inverse is never taken here on trust.
        out["reason"] = "normal matrix is singular; decomposition withheld"
        out["note"] = out["reason"]
        return out
    c11, c12, c22 = n22 / det, -n12 / det, n11 / det

    # x = C_x * G^T * C_d^-1 * v
    r1 = g[0][0] * wa * a["velocity"] + g[1][0] * wd * d["velocity"]
    r2 = g[0][1] * wa * a["velocity"] + g[1][1] * wd * d["velocity"]
    d_east = c11 * r1 + c12 * r2
    d_up = c12 * r1 + c22 * r2
    sigma_e, sigma_u = math.sqrt(c11), math.sqrt(c22)

    out.update({
        "refused": False,
        "reason": None,
        "east_mm_year": _r(d_east),
        "east_sigma_mm_year": _r(sigma_e),
        "up_mm_year": _r(d_up),
        "up_sigma_mm_year": _r(sigma_u),
        # dE and dU from two LOS observations are always correlated; reporting the two sigmas
        # without it would understate how much the pair can trade off against each other.
        "east_up_correlation": _r(c12 / (sigma_e * sigma_u), 4),
        "note": (
            f"east/up decomposition from {a['direction']} + {d['direction']} "
            f"(condition number {cond:.2f} <= {MAX_EAST_UP_CONDITION_NUMBER}); "
            f"dE={d_east:+.2f}+/-{sigma_e:.2f}, dU={d_up:+.2f}+/-{sigma_u:.2f} mm/yr, "
            f"WITH THE NORTH COMPONENT ASSUMED ZERO"
        ),
    })
    return out


# ---------------------------------------------------------------------------
# 3. temporal overlap
# ---------------------------------------------------------------------------


def assess_temporal_overlap(tracks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Do these tracks actually describe the same period of time?

    An InSAR velocity is an average rate over its own observation window. Two networks
    covering different years describe different things: combining a 2024 rate with a 2026
    rate produces a number that is not a measurement of either, and the chi-square test in
    combine_slope_parallel would blame "inconsistent geometry" for what is really a real
    change in the slope's behaviour between the two periods.

    Overlap is the intersection of ALL tracks' windows (latest start to earliest end),
    expressed as a fraction of the SHORTEST window - the shortest is the right denominator
    because a 90-day track fully inside a 900-day one is perfectly comparable, even though
    the overlap is only 10% of the longer window.

    Verdict threshold MIN_TEMPORAL_OVERLAP_FRACTION = 0.5, stated in the output so it can be
    argued with. `sufficient` is None, not False, when fewer than two tracks carry usable
    timestamps: that is an absence of evidence, not a failure.

    Never raises.
    """
    out: Dict[str, Any] = {
        "status": "OK",
        "overlap_days": None,
        "overlap_fraction_of_shorter": None,
        "shorter_span_days": None,
        "longer_span_days": None,
        "overlap_start_utc": None,
        "overlap_end_utc": None,
        "sufficient": None,
        "threshold_fraction": MIN_TEMPORAL_OVERLAP_FRACTION,
        "n_tracks_with_times": 0,
        "tracks_without_times": [],
        "note": "",
    }
    if not tracks:
        out["status"] = "NO_TRACKS"
        out["note"] = "no tracks supplied"
        return out

    spans: List[Tuple[str, float, float]] = []
    for index, raw in enumerate(tracks):
        rec = _parse_track(raw, index)
        start, end = rec["t_start_utc"], rec["t_end_utc"]
        if start is None or end is None or end <= start:
            out["tracks_without_times"].append(rec["key"])
            continue
        spans.append((rec["key"], start, end))

    out["n_tracks_with_times"] = len(spans)
    if len(spans) < 2:
        out["status"] = "INSUFFICIENT_TIMESTAMPS"
        out["note"] = (
            f"only {len(spans)} track(s) carry a usable t_start_utc/t_end_utc pair, so temporal "
            f"comparability cannot be assessed - this is unknown, not acceptable"
        )
        return out

    latest_start = max(s for _, s, _ in spans)
    earliest_end = min(e for _, _, e in spans)
    overlap_days = max(0.0, (earliest_end - latest_start) / 86400.0)
    span_days = [(e - s) / 86400.0 for _, s, e in spans]
    shorter, longer = min(span_days), max(span_days)
    fraction = overlap_days / shorter if shorter > 0 else 0.0
    sufficient = fraction >= MIN_TEMPORAL_OVERLAP_FRACTION

    out.update({
        "overlap_days": _r(overlap_days, 3),
        "overlap_fraction_of_shorter": _r(fraction, 4),
        "shorter_span_days": _r(shorter, 3),
        "longer_span_days": _r(longer, 3),
        "overlap_start_utc": _r(latest_start, 3) if overlap_days > 0 else None,
        "overlap_end_utc": _r(earliest_end, 3) if overlap_days > 0 else None,
        "sufficient": sufficient,
        "note": (
            f"windows overlap by {overlap_days:.0f} d = {fraction * 100:.0f}% of the shorter "
            f"{shorter:.0f} d span, at or above the {MIN_TEMPORAL_OVERLAP_FRACTION:.0%} threshold, "
            f"so the rates describe comparable periods"
            if sufficient else
            f"windows overlap by only {overlap_days:.0f} d = {fraction * 100:.0f}% of the shorter "
            f"{shorter:.0f} d span, below the {MIN_TEMPORAL_OVERLAP_FRACTION:.0%} threshold: these "
            f"rates largely describe DIFFERENT periods, so any disagreement between them may be a "
            f"real change over time rather than a geometry or model problem"
        ),
    })
    return out


# ---------------------------------------------------------------------------
# 4. the single entry point
# ---------------------------------------------------------------------------


def combine_tracks(
    tracks: Sequence[Dict[str, Any]],
    slope_deg: Optional[float] = None,
    aspect_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the whole combination: temporal check, slope-parallel WLS, east/up decomposition.

    The east/up decomposition is attempted only when exactly one ASCENDING and exactly one
    DESCENDING track survive the slope-parallel step. A track dropped there for low downslope
    sensitivity would still have been usable for an east/up solve, so when that is what
    blocked the decomposition it is recorded in `warnings` rather than passed over in silence.

    GUARANTEED NOT TO RAISE. Any unexpected failure comes back as status "ERROR" with the
    exception text in `error`, because this runs inside an unattended production loop and a
    traceback there would take down a 491-site collection pass over one bad track dict.
    """
    result: Dict[str, Any] = {
        "status": "OK",
        "error": None,
        "velocity_mm_year": None,
        "uncertainty_mm_year": None,
        "consistent": None,
        "n_tracks_input": 0,
        "n_tracks_used": 0,
        "tracks_used": [],
        "excluded": [],
        "flight_directions": {},
        "slope_parallel": None,
        "temporal_overlap": None,
        "east_up": None,
        "warnings": [],
        "summary": "",
    }
    try:
        track_list: List[Dict[str, Any]] = list(tracks) if tracks else []
        result["n_tracks_input"] = len(track_list)
        if not track_list:
            result["status"] = "NO_TRACKS"
            result["summary"] = "No InSAR tracks supplied; nothing to combine."
            return result

        parsed = [_parse_track(t, i) for i, t in enumerate(track_list)]
        result["flight_directions"] = {
            r["key"]: {"direction": r["direction"], "provenance": r["direction_provenance"]}
            for r in parsed
        }
        if any(r["direction_provenance"] == "inferred" for r in parsed):
            inferred = [r["key"] for r in parsed if r["direction_provenance"] == "inferred"]
            result["warnings"].append(
                f"flight direction INFERRED from the look-vector azimuth (assumes Sentinel-1's "
                f"right-looking attitude) rather than declared, for: {', '.join(inferred)}"
            )

        keys = [r["key"] for r in parsed]
        if len(set(keys)) != len(keys):
            # Every grouping below reports by key. Duplicates make `tracks_used` and
            # `excluded` ambiguous for the caller, so say so instead of quietly shipping a
            # result that cannot be traced back to a specific network.
            result["warnings"].append(
                "KEYS: two or more tracks share the same `key`; per-track output cannot be "
                "attributed unambiguously"
            )

        overlap = assess_temporal_overlap(track_list)
        result["temporal_overlap"] = overlap
        if overlap.get("sufficient") is False:
            result["warnings"].append("TEMPORAL: " + overlap["note"])
        elif overlap.get("sufficient") is None and len(track_list) > 1:
            result["warnings"].append("TEMPORAL: " + overlap["note"])

        slope_parallel = combine_slope_parallel(track_list, slope_deg, aspect_deg)
        result["slope_parallel"] = slope_parallel
        result["velocity_mm_year"] = slope_parallel["velocity_mm_year"]
        result["uncertainty_mm_year"] = slope_parallel["uncertainty_mm_year"]
        result["consistent"] = slope_parallel["consistent"]
        result["n_tracks_used"] = slope_parallel["n_tracks_used"]
        result["tracks_used"] = list(slope_parallel["tracks_used"])
        result["excluded"] = list(slope_parallel["excluded"])
        if slope_parallel["status"] != "OK":
            result["status"] = slope_parallel["status"]
        if slope_parallel.get("consistent") is False:
            result["warnings"].append("CONSISTENCY: " + slope_parallel["note"])

        # --- east/up, only for a genuine one-ascending + one-descending pair -------------
        used_keys = set(result["tracks_used"])
        survivors = [r for r in parsed if r["key"] in used_keys]
        asc = [r for r in survivors if r["direction"] == "ASCENDING"]
        desc = [r for r in survivors if r["direction"] == "DESCENDING"]

        directions_used = {r["direction"] for r in survivors}
        if len(survivors) > 1 and len(directions_used) == 1 and "UNKNOWN" not in directions_used:
            # The WLS is still valid, but its uncertainty reduction assumes the tracks' errors
            # are INDEPENDENT. Two stacks from the same orbit direction see nearly the same
            # atmosphere from nearly the same angle, so part of their error is common and the
            # real improvement is smaller than the formal factor. Say so rather than let the
            # headline number be quoted as if two same-direction tracks were two independent
            # measurements.
            result["warnings"].append(
                f"INDEPENDENCE: all {len(survivors)} combined tracks are "
                f"{next(iter(directions_used))}; the formal uncertainty reduction assumes "
                f"independent errors, but same-direction tracks share atmospheric and geometric "
                f"error, so the true improvement is smaller than the quoted factor"
            )
        if len(asc) == 1 and len(desc) == 1:
            # Index, not key: two tracks could share a `key` and a key lookup would then
            # silently decompose the wrong pair.
            result["east_up"] = decompose_east_up(
                track_list[asc[0]["index"]], track_list[desc[0]["index"]]
            )
            if result["east_up"]["refused"]:
                result["warnings"].append("EAST_UP: " + str(result["east_up"]["reason"]))
        else:
            valid = [r for r in parsed if r["ok"]]
            v_asc = [r for r in valid if r["direction"] == "ASCENDING"]
            v_desc = [r for r in valid if r["direction"] == "DESCENDING"]
            if len(v_asc) == 1 and len(v_desc) == 1:
                # The pair existed and was individually valid; the slope-sensitivity gate is
                # what removed it. Say so - an east/up solve does not need that gate, so the
                # caller may legitimately want to call decompose_east_up() directly.
                result["warnings"].append(
                    "EAST_UP: one ascending + one descending track were available, but at least "
                    "one was excluded from the combination (see `excluded`); east/up "
                    "decomposition not attempted automatically"
                )

        result["summary"] = _summarize(result)
        return result
    except Exception as exc:  # pragma: no cover - the guarantee, not a code path
        result["status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["summary"] = f"InSAR track combination failed: {result['error']}"
        return result


def _summarize(result: Dict[str, Any]) -> str:
    """One human-readable sentence, honest about what the combination did and did not prove."""
    sp = result.get("slope_parallel") or {}
    if sp.get("velocity_mm_year") is None:
        return (
            f"No combined downslope rate from {result['n_tracks_input']} track(s): "
            f"{sp.get('note') or 'nothing usable'}."
        )

    n = sp["n_tracks_used"]
    directions = sorted({
        result["flight_directions"].get(k, {}).get("direction", "UNKNOWN")
        for k in sp["tracks_used"]
    })
    parts = [
        f"{sp['velocity_mm_year']:+.1f} +/- {sp['uncertainty_mm_year']:.1f} mm/yr downslope from "
        f"{n} track{'s' if n != 1 else ''} ({'+'.join(directions)})"
    ]
    if n > 1 and sp.get("improvement_factor"):
        parts.append(
            f"{sp['improvement_factor']:.2f}x tighter than the best single track "
            f"(+/-{sp['best_single_uncertainty_mm_year']:.1f} mm/yr)"
        )
    if sp.get("consistent") is True:
        blind = sp.get("vertical_blind_spot") or {}
        limit = blind.get("min_detectable_vertical_mm_year")
        if blind.get("blind_to_vertical"):
            parts.append(
                f"chi2={sp['chi_square']:.2f} shows no disagreement, but this geometry is "
                f"BLIND to vertical motion - a pure uplift/subsidence signal would look "
                f"exactly like downslope creep here"
            )
        elif limit is not None:
            parts.append(
                f"no disagreement (chi2={sp['chi_square']:.2f} < "
                f"{sp['chi_square_critical_95']:.2f}; would only catch vertical motion "
                f"above {limit:.0f} mm/yr)"
            )
        else:
            parts.append(
                f"no disagreement (chi2={sp['chi_square']:.2f} < "
                f"{sp['chi_square_critical_95']:.2f})"
            )
    elif sp.get("consistent") is False:
        parts.append(
            f"TRACKS DISAGREE (chi2={sp['chi_square']:.2f} > {sp['chi_square_critical_95']:.2f}), "
            f"uncertainty inflated {sp['inflation_factor']:.2f}x - the purely-downslope assumption "
            f"is contradicted"
        )
    elif (sp.get("vertical_blind_spot") or {}).get("blind_to_vertical"):
        parts.append(
            "consistency UNTESTABLE here: the two geometries respond identically to "
            "vertical motion, so uplift or subsidence is indistinguishable from "
            "downslope creep"
        )
    else:
        parts.append("slope-parallel assumption UNTESTED (a single track cannot check itself)")

    east_up = result.get("east_up")
    if east_up and not east_up.get("refused"):
        parts.append(
            f"east/up split dE={east_up['east_mm_year']:+.1f}+/-{east_up['east_sigma_mm_year']:.1f}, "
            f"dU={east_up['up_mm_year']:+.1f}+/-{east_up['up_sigma_mm_year']:.1f} mm/yr "
            f"(north assumed zero)"
        )
    elif east_up and east_up.get("refused"):
        parts.append("east/up decomposition refused")

    overlap = result.get("temporal_overlap") or {}
    if overlap.get("sufficient") is False:
        parts.append(f"WARNING: windows overlap only {overlap['overlap_fraction_of_shorter']:.0%}")
    return "; ".join(parts) + "."
