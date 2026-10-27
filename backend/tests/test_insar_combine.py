"""Tests for ascending+descending InSAR combination.

These are meant to be PROOFS, not restatements of the implementation:

  * the geometry is cross-checked against the already-deployed
    insar_timeseries.project_los_to_slope_parallel, so the two cannot drift apart;
  * the chi-square table is cross-checked against scipy (which the module deliberately does
    not import);
  * the condition number computed in stdlib `math` is cross-checked against numpy;
  * every synthetic LOS value is forward-modelled LONGHAND in the test from an explicitly
    written-out look vector, never by calling the module's own helpers, so a sign error in
    the convention would fail here rather than cancel out;
  * the headline claim (combined sigma beats every single track, and equals the analytic
    1/sqrt(sum s^2/sigma^2)) is pinned numerically;
  * the refusal paths are tested as hard as the success paths.

Offline, deterministic, no network, no LLM.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from app.ingestion import insar_combine as ic
from app.ingestion import insar_timeseries as ts


# Real-ish Sentinel-1 IW geometry (the numbers verified at site glc_105):
# look-vector elevation 0.70 rad above the horizontal, ground->satellite azimuth measured
# from East toward North - ascending looks west-northwest, descending east-southeast.
THETA = 0.70
PHI_ASC = -2.9
PHI_DESC = -0.23

# A north-facing 35-degree Himalayan slope: both tracks sense it comparably, which is
# where combination buys the most PRECISION - sqrt(2) is reached only when the two
# sensitivities are equal, i.e. at exactly 0 or 180.
SLOPE_N, ASPECT_N = 35.0, 0.0

# ...and an off-axis aspect for the CONSISTENCY tests, because of an awkward physical
# coincidence worth stating plainly: the two look vectors differ almost purely in East, so
# at exactly 0/180 they respond IDENTICALLY to vertical motion and the chi-square test has
# no power whatsoever. The geometry that gives the best precision is the one where the
# consistency check is blind. Asserting consistent=True at ASPECT_N would therefore be
# asserting that a test which never ran had passed.
SLOPE_OFF, ASPECT_OFF = 35.0, 30.0
# An east-facing slope: the two tracks see downslope motion with OPPOSITE signs. Sign errors
# that cancel on the north-facing case do not cancel here.
SLOPE_E, ASPECT_E = 35.0, 90.0


def _epoch(year: int, month: int, day: int) -> float:
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def _look_longhand(theta: float, phi: float):
    """(E, N, U) written out here on purpose - an independent copy of the convention."""
    return (math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), math.sin(theta))


def _down_longhand(slope_deg: float, aspect_deg: float):
    s, a = math.radians(slope_deg), math.radians(aspect_deg)
    return (math.sin(a) * math.cos(s), math.cos(a) * math.cos(s), -math.sin(s))


def _sensitivity_longhand(theta, phi, slope_deg, aspect_deg) -> float:
    look, down = _look_longhand(theta, phi), _down_longhand(slope_deg, aspect_deg)
    return sum(l * d for l, d in zip(look, down))


def _track(key, direction, velocity, sigma, theta=THETA, phi=PHI_ASC,
           t_start=None, t_end=None):
    return {
        "key": key,
        "flight_direction": direction,
        "velocity_mm_year": velocity,
        "uncertainty_mm_year": sigma,
        "lv_theta_rad": theta,
        "lv_phi_rad": phi,
        "t_start_utc": _epoch(2024, 1, 1) if t_start is None else t_start,
        "t_end_utc": _epoch(2026, 1, 1) if t_end is None else t_end,
    }


def _synthetic_pair(true_downslope_mm_year, slope_deg, aspect_deg, sigma_a=6.7, sigma_d=6.7):
    """Two LOS observations forward-modelled from ONE known downslope velocity, no noise."""
    s_a = _sensitivity_longhand(THETA, PHI_ASC, slope_deg, aspect_deg)
    s_d = _sensitivity_longhand(THETA, PHI_DESC, slope_deg, aspect_deg)
    return [
        _track("asc_158", "ASCENDING", s_a * true_downslope_mm_year, sigma_a, phi=PHI_ASC),
        _track("desc_92", "DESCENDING", s_d * true_downslope_mm_year, sigma_d, phi=PHI_DESC),
    ], s_a, s_d


# ---------------------------------------------------------------------------
# the module's maths vs. independent implementations
# ---------------------------------------------------------------------------


def test_sensitivity_matches_the_deployed_production_projection():
    """The convention here must be byte-for-byte the one already serving production.

    insar_timeseries.project_los_to_slope_parallel converts LOS -> downslope by dividing by
    its own `sensitivity`. If this module computed that dot product differently the combined
    velocity would silently disagree with the single-track one already being published.
    """
    for theta in (0.60, 0.70, 0.95):
        for phi in (PHI_ASC, PHI_DESC, 1.1):
            for slope, aspect in ((35.0, 20.0), (20.0, 90.0), (48.0, 215.0)):
                mine = ic.projection_sensitivity(theta, phi, slope, aspect)
                theirs = ts.project_los_to_slope_parallel(1.0, theta, phi, slope, aspect)
                assert round(mine, 3) == theirs["sensitivity"]
                assert mine == pytest.approx(
                    _sensitivity_longhand(theta, phi, slope, aspect), rel=1e-12
                )


def test_min_sensitivity_gate_is_the_same_number_as_production():
    """The mirrored constant must not drift from insar_timeseries'."""
    assert ic.MIN_PROJECTION_SENSITIVITY == ts.MIN_PROJECTION_SENSITIVITY


def test_chi_square_table_matches_scipy():
    """The hardcoded table (and the >10 approximation) must be the real chi-square values.

    scipy is a project dependency and is used HERE, in the test, precisely so the module
    itself can stay stdlib-only without the table being taken on trust.
    """
    from scipy.stats import chi2

    for dof in range(1, 11):
        assert ic.chi2_critical_95(dof) == pytest.approx(chi2.ppf(0.95, dof), abs=5e-4)
    for dof in (11, 15, 25, 40, 100):
        # Wilson-Hilferty: accurate to well under 1% in this tail.
        assert ic.chi2_critical_95(dof) == pytest.approx(chi2.ppf(0.95, dof), rel=2e-3)
    assert ic.chi2_critical_95(0) is None  # no spare observation -> nothing to test
    assert ic.chi2_critical_95(-3) is None
    assert ic.chi2_critical_95("nonsense") is None


def test_condition_number_matches_numpy():
    """The stdlib 2x2 SVD must agree with numpy, including the singular case."""
    def g(ta, pa, td, pd):
        return [
            [math.cos(ta) * math.cos(pa), math.sin(ta)],
            [math.cos(td) * math.cos(pd), math.sin(td)],
        ]

    for matrix in (
        g(THETA, PHI_ASC, THETA, PHI_DESC),           # real asc/desc
        g(0.99, PHI_ASC, 0.82, PHI_ASC),              # two ascending (glc_105's real case)
        g(0.70, 1.0, 1.10, -2.0),
        [[1.0, 0.0], [0.0, 1.0]],
    ):
        assert ic._condition_number_2x2(matrix) == pytest.approx(
            float(np.linalg.cond(np.array(matrix))), rel=1e-9
        )
    assert math.isinf(ic._condition_number_2x2([[1.0, 0.5], [2.0, 1.0]]))  # exactly singular


# ---------------------------------------------------------------------------
# 1. ground truth recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slope_deg, aspect_deg, truth",
    [
        (SLOPE_OFF, ASPECT_OFF, 42.0),  # off-axis: both tracks see it, test has power
        (SLOPE_E, ASPECT_E, 42.0),    # east-facing: the two tracks see OPPOSITE signs
        (SLOPE_OFF, ASPECT_OFF, -17.5),  # upslope/negative rate
        (SLOPE_OFF, ASPECT_OFF, 0.0),    # a still slope must come back still
    ],
)
def test_combination_recovers_a_known_downslope_velocity(slope_deg, aspect_deg, truth):
    """Forward-model two LOS values from one known downslope rate; get the rate back."""
    tracks, s_a, s_d = _synthetic_pair(truth, slope_deg, aspect_deg)
    out = ic.combine_slope_parallel(tracks, slope_deg, aspect_deg)

    assert out["status"] == "OK"
    assert out["n_tracks_used"] == 2
    assert out["velocity_mm_year"] == pytest.approx(truth, abs=1e-6)
    # Noise-free data generated by exactly this model must be perfectly consistent.
    assert out["chi_square"] == pytest.approx(0.0, abs=1e-9)
    assert out["consistent"] is True
    assert out["inflation_factor"] == 1.0
    assert out["uncertainty_mm_year"] == out["formal_uncertainty_mm_year"]

    if aspect_deg == ASPECT_E:
        # Guard the guard: this case is only a sign test if the signs really do differ.
        assert s_a * s_d < 0


# ---------------------------------------------------------------------------
# 2. the headline claim: the combination is genuinely tighter
# ---------------------------------------------------------------------------


def test_combined_uncertainty_beats_every_single_track_and_matches_the_analytic_form():
    """THE justification for this feature. Pinned analytically, not approximately.

    sigma_hat = 1/sqrt(sum s_i^2/sigma_i^2) must (a) equal the closed form computed
    independently here, and (b) be strictly smaller than sigma_i/|s_i| for EVERY track.
    """
    sigma_a, sigma_d = 6.7, 8.1
    tracks, s_a, s_d = _synthetic_pair(42.0, SLOPE_N, ASPECT_N, sigma_a, sigma_d)
    out = ic.combine_slope_parallel(tracks, SLOPE_N, ASPECT_N)

    # abs=1e-6 is the tightest meaningful tolerance: the module rounds its outputs to 6
    # decimal places (a millionth of a mm/yr), so anything smaller would be testing round().
    analytic = 1.0 / math.sqrt(s_a ** 2 / sigma_a ** 2 + s_d ** 2 / sigma_d ** 2)
    assert out["formal_uncertainty_mm_year"] == pytest.approx(analytic, abs=1e-6)

    single_a, single_d = sigma_a / abs(s_a), sigma_d / abs(s_d)
    assert analytic < single_a
    assert analytic < single_d
    assert out["best_single_uncertainty_mm_year"] == pytest.approx(min(single_a, single_d), abs=1e-6)
    assert out["improvement_factor"] == pytest.approx(min(single_a, single_d) / analytic, abs=1e-6)
    assert out["improvement_factor"] > 1.0

    # And the combined weighted mean must also be the analytic WLS value.
    expected_v = (
        (s_a * tracks[0]["velocity_mm_year"] / sigma_a ** 2
         + s_d * tracks[1]["velocity_mm_year"] / sigma_d ** 2)
        / (s_a ** 2 / sigma_a ** 2 + s_d ** 2 / sigma_d ** 2)
    )
    assert out["velocity_mm_year"] == pytest.approx(expected_v, abs=1e-6)


def test_equally_good_tracks_approach_the_root_two_ceiling_and_never_exceed_it():
    """Two equally weighted, equally sensitive observations can tighten sigma by at most
    sqrt(2). A result claiming more than that would mean the maths is wrong, so bound it
    from ABOVE as well as below - an over-tight error bar is the failure mode this project
    cares about most."""
    # True north deliberately: sqrt(2) is reached only when both sensitivities are equal,
    # which happens at exactly 0/180. (That is also the blind spot for the consistency
    # test - see SLOPE_OFF/ASPECT_OFF - so precision and testability peak in different
    # places. This test is about precision only.)
    tracks, s_a, s_d = _synthetic_pair(20.0, SLOPE_N, ASPECT_N, 6.7, 6.7)
    out = ic.combine_slope_parallel(tracks, SLOPE_N, ASPECT_N)
    assert abs(s_a) == pytest.approx(abs(s_d), rel=0.05)  # premise of the ceiling
    assert 1.3 < out["improvement_factor"] <= math.sqrt(2) + 1e-9


# ---------------------------------------------------------------------------
# 3. inconsistency must be caught AND must degrade the number
# ---------------------------------------------------------------------------


def test_contradictory_tracks_are_flagged_and_the_uncertainty_is_inflated():
    """Two LOS values that no single downslope rate can explain.

    On this slope both sensitivities share a sign, so a pure downslope
    motion MUST produce two LOS values of the same sign and similar size. Here they have
    opposite signs: whatever is moving, it is not moving purely downslope. The tight formal
    sigma must not be published as if it were.
    """
    s_a = _sensitivity_longhand(THETA, PHI_ASC, SLOPE_OFF, ASPECT_OFF)
    s_d = _sensitivity_longhand(THETA, PHI_DESC, SLOPE_OFF, ASPECT_OFF)
    assert s_a * s_d > 0  # premise: same sign expected under the model

    tracks = [
        _track("asc_158", "ASCENDING", s_a * 10.0, 1.0, phi=PHI_ASC),
        _track("desc_92", "DESCENDING", s_d * -30.0, 1.0, phi=PHI_DESC),
    ]
    out = ic.combine_slope_parallel(tracks, SLOPE_OFF, ASPECT_OFF)

    assert out["dof"] == 1
    assert out["chi_square_critical_95"] == pytest.approx(3.841, abs=1e-6)
    assert out["chi_square"] > out["chi_square_critical_95"]
    assert out["consistent"] is False

    # The whole point: the reported number degrades instead of lying.
    inflation = math.sqrt(out["chi_square"] / out["dof"])
    assert out["inflation_factor"] == pytest.approx(inflation, abs=1e-6)
    assert out["uncertainty_mm_year"] == pytest.approx(
        out["formal_uncertainty_mm_year"] * inflation, abs=1e-5
    )
    assert out["uncertainty_mm_year"] > out["formal_uncertainty_mm_year"]
    # ... and by a lot, for data this contradictory - not a token 1.01x.
    assert out["uncertainty_mm_year"] > 5.0 * out["formal_uncertainty_mm_year"]
    assert "CONTRADICTED" in out["note"]
    # The caller can still see the untouched formal value.
    assert out["formal_uncertainty_mm_year"] < out["best_single_uncertainty_mm_year"]


def test_mild_scatter_within_the_quoted_errors_stays_consistent():
    """The flag must not fire on ordinary noise, or it would be useless.

    Offsetting one track by 1 sigma gives chi2 well under the 3.841 critical value.
    """
    s_a = _sensitivity_longhand(THETA, PHI_ASC, SLOPE_OFF, ASPECT_OFF)
    s_d = _sensitivity_longhand(THETA, PHI_DESC, SLOPE_OFF, ASPECT_OFF)
    tracks = [
        _track("asc_158", "ASCENDING", s_a * 20.0, 2.0, phi=PHI_ASC),
        _track("desc_92", "DESCENDING", s_d * 20.0 + 2.0, 2.0, phi=PHI_DESC),
    ]
    out = ic.combine_slope_parallel(tracks, SLOPE_OFF, ASPECT_OFF)
    assert out["consistent"] is True
    assert out["chi_square"] < out["chi_square_critical_95"]
    assert out["inflation_factor"] == 1.0


def test_a_single_track_is_not_called_consistent():
    """dof=0 fits perfectly by construction. The verdict must be None, never True."""
    tracks, s_a, _ = _synthetic_pair(42.0, SLOPE_N, ASPECT_N)
    out = ic.combine_slope_parallel(tracks[:1], SLOPE_N, ASPECT_N)
    assert out["n_tracks_used"] == 1
    assert out["dof"] == 0
    assert out["consistent"] is None
    assert out["chi_square_critical_95"] is None
    assert out["velocity_mm_year"] == pytest.approx(42.0, abs=1e-6)
    assert out["improvement_factor"] == pytest.approx(1.0, rel=1e-9)
    assert "UNTESTED" in out["note"]


# ---------------------------------------------------------------------------
# 4. east/up decomposition
# ---------------------------------------------------------------------------


def test_east_up_recovers_a_known_east_and_up_motion():
    """Forward-model two LOS values from a known (dE, dN=0, dU) and get (dE, dU) back."""
    true_e, true_u = 12.0, -5.0
    look_a, look_d = _look_longhand(THETA, PHI_ASC), _look_longhand(THETA, PHI_DESC)
    v_a = look_a[0] * true_e + look_a[2] * true_u   # dN deliberately zero
    v_d = look_d[0] * true_e + look_d[2] * true_u

    out = ic.decompose_east_up(
        _track("asc_158", "ASCENDING", v_a, 1.5, phi=PHI_ASC),
        _track("desc_92", "DESCENDING", v_d, 1.5, phi=PHI_DESC),
    )
    assert out["refused"] is False
    assert out["east_mm_year"] == pytest.approx(true_e, abs=1e-6)
    assert out["up_mm_year"] == pytest.approx(true_u, abs=1e-6)
    assert out["north_assumed_zero"] is True
    assert "north" in out["assumption"].lower()

    # Covariance must be the propagated (G^T C_d^-1 G)^-1, not a hand-wave.
    g = np.array([[look_a[0], look_a[2]], [look_d[0], look_d[2]]])
    cx = np.linalg.inv(g.T @ np.diag([1 / 1.5 ** 2, 1 / 1.5 ** 2]) @ g)
    assert out["east_sigma_mm_year"] == pytest.approx(math.sqrt(cx[0, 0]), abs=1e-6)
    assert out["up_sigma_mm_year"] == pytest.approx(math.sqrt(cx[1, 1]), abs=1e-6)
    assert out["east_up_correlation"] == pytest.approx(
        cx[0, 1] / math.sqrt(cx[0, 0] * cx[1, 1]), abs=1e-4
    )
    # Good geometry: the error bars are of the same order as the inputs, not blown up.
    assert out["east_sigma_mm_year"] < 3.0 * 1.5


def test_east_up_is_biased_by_real_north_motion_which_is_why_the_assumption_is_reported():
    """The dN=0 assumption is not free. Prove it costs something, so nobody treats the
    output as a full 3-D solution."""
    true_e, true_n, true_u = 12.0, 20.0, -5.0
    look_a, look_d = _look_longhand(THETA, PHI_ASC), _look_longhand(THETA, PHI_DESC)
    v_a = sum(l * x for l, x in zip(look_a, (true_e, true_n, true_u)))
    v_d = sum(l * x for l, x in zip(look_d, (true_e, true_n, true_u)))

    out = ic.decompose_east_up(
        _track("a", "ASCENDING", v_a, 1.0, phi=PHI_ASC),
        _track("d", "DESCENDING", v_d, 1.0, phi=PHI_DESC),
    )
    assert out["refused"] is False
    # 20 mm/yr of northward motion leaks in; the solution is NOT the truth.
    assert abs(out["up_mm_year"] - true_u) > 1.0
    assert out["north_assumed_zero"] is True


def test_two_ascending_tracks_are_refused_for_east_up_but_still_combine_slope_parallel():
    """glc_105's real situation: three stacks, two of them ASCENDING (337397/IW1 near range
    and 118306/IW3 far range). They differ only in incidence, so G is ill-conditioned and an
    east/up split from them would be amplified noise with a confident error bar. The
    slope-parallel WLS, which solves for ONE unknown, remains perfectly valid.
    """
    theta_iw1, theta_iw3 = math.radians(90 - 33), math.radians(90 - 43)
    a1 = _track("asc_158_iw1", "ASCENDING", -8.0, 6.7, theta=theta_iw1, phi=PHI_ASC)
    a2 = _track("asc_56_iw3", "ASCENDING", -9.0, 6.7, theta=theta_iw3, phi=PHI_ASC)

    refused = ic.decompose_east_up(a1, a2)
    assert refused["refused"] is True
    assert refused["east_mm_year"] is None and refused["up_mm_year"] is None
    assert refused["condition_number"] > ic.MAX_EAST_UP_CONDITION_NUMBER
    assert "ASCENDING" in refused["reason"]

    combined = ic.combine_tracks([a1, a2], SLOPE_E, ASPECT_E)
    assert combined["status"] == "OK"
    assert combined["n_tracks_used"] == 2
    assert combined["velocity_mm_year"] is not None
    assert combined["east_up"] is None  # never even attempted: no descending track
    assert combined["slope_parallel"]["improvement_factor"] > 1.0
    # ... but the caller is warned that two same-direction tracks are not two independent
    # measurements, so the formal improvement is optimistic.
    assert any(w.startswith("INDEPENDENCE:") for w in combined["warnings"])


def test_the_condition_number_threshold_separates_real_geometries():
    """The numerical experiment behind MAX_EAST_UP_CONDITION_NUMBER, as a test.

    Every genuinely complementary asc/desc pair across the full Sentinel-1 IW incidence
    range must be ACCEPTED, and same-direction pairs must be REFUSED. If a future edit moves
    the threshold into either population, this fails.
    """
    thetas = [math.radians(90 - inc) for inc in (29, 33, 37, 41, 46)]
    worst_genuine = 0.0
    for t_a in thetas:
        for t_d in thetas:
            for p_a in (-2.9 - 0.14, -2.9, -2.9 + 0.14):
                for p_d in (-0.23 - 0.14, -0.23, -0.23 + 0.14):
                    g = [
                        [math.cos(t_a) * math.cos(p_a), math.sin(t_a)],
                        [math.cos(t_d) * math.cos(p_d), math.sin(t_d)],
                    ]
                    worst_genuine = max(worst_genuine, ic._condition_number_2x2(g))
    assert worst_genuine < ic.MAX_EAST_UP_CONDITION_NUMBER  # no real asc/desc pair rejected
    assert worst_genuine == pytest.approx(1.93, abs=0.05)
    assert ic.MAX_EAST_UP_CONDITION_NUMBER / worst_genuine > 1.5  # with real headroom

    best_same_direction = math.inf
    for t_a in thetas:
        for t_b in thetas:
            if t_a == t_b:
                continue
            for phi in (PHI_ASC, PHI_DESC):
                g = [
                    [math.cos(t_a) * math.cos(phi), math.sin(t_a)],
                    [math.cos(t_b) * math.cos(phi), math.sin(t_b)],
                ]
                best_same_direction = min(best_same_direction, ic._condition_number_2x2(g))
    assert best_same_direction > ic.MAX_EAST_UP_CONDITION_NUMBER  # all same-direction refused
    assert best_same_direction == pytest.approx(6.74, abs=0.1)
    assert best_same_direction / ic.MAX_EAST_UP_CONDITION_NUMBER > 1.5  # with real margin
    # The two populations must not touch; the threshold lives in the gap between them.
    assert worst_genuine < ic.MAX_EAST_UP_CONDITION_NUMBER < best_same_direction


def test_identical_geometry_is_refused_rather_than_producing_an_enormous_number():
    """The exactly-singular case must not come back as a confident huge velocity."""
    a = _track("a", "ASCENDING", -8.0, 1.0, phi=PHI_ASC)
    b = _track("b", "ASCENDING", -8.4, 1.0, phi=PHI_ASC)
    out = ic.decompose_east_up(a, b)
    assert out["refused"] is True
    assert out["east_mm_year"] is None
    assert out["condition_number"] is None  # infinite, reported as unknown rather than faked
    assert "singular" in out["reason"]


# ---------------------------------------------------------------------------
# 5. the sensitivity gate
# ---------------------------------------------------------------------------


def test_a_track_nearly_blind_to_the_slope_is_excluded_with_a_stated_reason():
    """On a 20-degree slope facing 328 degrees the ascending look direction is almost exactly
    perpendicular to downslope. Dividing by that sensitivity would turn its noise into a
    large fake velocity, so it must be dropped - and the drop must be explained.
    """
    slope, aspect = 20.0, 328.0
    s_a = _sensitivity_longhand(THETA, PHI_ASC, slope, aspect)
    s_d = _sensitivity_longhand(THETA, PHI_DESC, slope, aspect)
    assert abs(s_a) < ic.MIN_PROJECTION_SENSITIVITY  # premise of the test
    assert abs(s_d) > 0.5

    tracks = [
        _track("asc_blind", "ASCENDING", 3.0, 6.7, phi=PHI_ASC),
        _track("desc_92", "DESCENDING", s_d * 25.0, 6.7, phi=PHI_DESC),
    ]
    out = ic.combine_slope_parallel(tracks, slope, aspect)

    assert out["tracks_used"] == ["desc_92"]
    assert len(out["excluded"]) == 1
    excluded = out["excluded"][0]
    assert excluded["key"] == "asc_blind"
    assert "perpendicular" in excluded["reason"]
    assert str(ic.MIN_PROJECTION_SENSITIVITY) in excluded["reason"]
    assert excluded["sensitivity"] == pytest.approx(s_a, abs=1e-4)
    # The surviving track still yields its own honest answer.
    assert out["velocity_mm_year"] == pytest.approx(25.0, abs=1e-6)
    assert out["consistent"] is None

    # combine_tracks must also say WHY no east/up split was attempted rather than stay mute.
    whole = ic.combine_tracks(tracks, slope, aspect)
    assert whole["east_up"] is None
    assert any(w.startswith("EAST_UP:") for w in whole["warnings"])


# ---------------------------------------------------------------------------
# 6. temporal overlap
# ---------------------------------------------------------------------------


def test_non_overlapping_windows_are_flagged():
    tracks = [
        _track("asc_2023", "ASCENDING", -8.0, 6.7, phi=PHI_ASC,
               t_start=_epoch(2023, 1, 1), t_end=_epoch(2024, 1, 1)),
        _track("desc_2025", "DESCENDING", 4.0, 6.7, phi=PHI_DESC,
               t_start=_epoch(2025, 1, 1), t_end=_epoch(2026, 1, 1)),
    ]
    out = ic.assess_temporal_overlap(tracks)
    assert out["overlap_days"] == 0.0
    assert out["overlap_fraction_of_shorter"] == 0.0
    assert out["sufficient"] is False
    assert out["threshold_fraction"] == ic.MIN_TEMPORAL_OVERLAP_FRACTION
    assert "DIFFERENT periods" in out["note"]

    whole = ic.combine_tracks(tracks, SLOPE_N, ASPECT_N)
    assert any(w.startswith("TEMPORAL:") for w in whole["warnings"])
    assert "WARNING" in whole["summary"]


def test_a_short_window_fully_inside_a_long_one_is_sufficient():
    """The denominator must be the SHORTER span: a 6-month track nested inside a 2-year one
    is perfectly comparable even though it covers only a quarter of the longer window."""
    tracks = [
        _track("long", "ASCENDING", -8.0, 6.7, phi=PHI_ASC,
               t_start=_epoch(2024, 1, 1), t_end=_epoch(2026, 1, 1)),
        _track("short", "DESCENDING", 4.0, 6.7, phi=PHI_DESC,
               t_start=_epoch(2025, 7, 1), t_end=_epoch(2026, 1, 1)),
    ]
    out = ic.assess_temporal_overlap(tracks)
    assert out["overlap_fraction_of_shorter"] == pytest.approx(1.0, abs=1e-9)
    assert out["shorter_span_days"] == pytest.approx(184.0, abs=0.05)
    assert out["longer_span_days"] == pytest.approx(731.0, abs=0.05)
    assert out["sufficient"] is True


def test_partial_overlap_below_the_threshold_is_refused():
    tracks = [
        _track("a", "ASCENDING", -8.0, 6.7, phi=PHI_ASC,
               t_start=_epoch(2024, 1, 1), t_end=_epoch(2025, 1, 1)),
        _track("b", "DESCENDING", 4.0, 6.7, phi=PHI_DESC,
               t_start=_epoch(2024, 10, 1), t_end=_epoch(2025, 10, 1)),
    ]
    out = ic.assess_temporal_overlap(tracks)
    assert 0.2 < out["overlap_fraction_of_shorter"] < 0.3
    assert out["sufficient"] is False


def test_missing_timestamps_are_unknown_not_acceptable():
    tracks = [
        {"key": "a", "flight_direction": "ASCENDING", "velocity_mm_year": -8.0,
         "uncertainty_mm_year": 6.7, "lv_theta_rad": THETA, "lv_phi_rad": PHI_ASC},
        _track("b", "DESCENDING", 4.0, 6.7, phi=PHI_DESC),
    ]
    out = ic.assess_temporal_overlap(tracks)
    assert out["sufficient"] is None          # not False: this is an absence of evidence
    assert out["status"] == "INSUFFICIENT_TIMESTAMPS"
    assert out["tracks_without_times"] == ["a"]


def test_reversed_timestamps_are_treated_as_missing():
    tracks = [
        _track("backwards", "ASCENDING", -8.0, 6.7, phi=PHI_ASC,
               t_start=_epoch(2026, 1, 1), t_end=_epoch(2024, 1, 1)),
        _track("b", "DESCENDING", 4.0, 6.7, phi=PHI_DESC),
    ]
    out = ic.assess_temporal_overlap(tracks)
    assert out["tracks_without_times"] == ["backwards"]
    assert out["sufficient"] is None


# ---------------------------------------------------------------------------
# 7. flight-direction handling
# ---------------------------------------------------------------------------


def test_flight_direction_is_classified_from_the_look_azimuth_when_not_declared():
    assert ic.classify_flight_direction(PHI_ASC) == "ASCENDING"
    assert ic.classify_flight_direction(PHI_DESC) == "DESCENDING"
    assert ic.classify_flight_direction(math.pi / 2) == "UNKNOWN"   # due north: ambiguous
    assert ic.classify_flight_direction(-math.pi / 2) == "UNKNOWN"
    assert ic.classify_flight_direction(None) == "UNKNOWN"
    assert ic.classify_flight_direction(float("nan")) == "UNKNOWN"


def test_inferred_directions_are_labelled_as_inferred():
    """Provenance must survive: an inferred direction rests on the right-looking assumption."""
    a = _track("a", None, -8.0, 6.7, phi=PHI_ASC)
    d = _track("d", "DESCENDING", 4.0, 6.7, phi=PHI_DESC)
    out = ic.combine_tracks([a, d], SLOPE_N, ASPECT_N)
    assert out["flight_directions"]["a"] == {"direction": "ASCENDING", "provenance": "inferred"}
    assert out["flight_directions"]["d"] == {"direction": "DESCENDING", "provenance": "declared"}
    assert any("INFERRED" in w for w in out["warnings"])
    # The inference is good enough to drive the pairing, so the decomposition still happens.
    assert out["east_up"] is not None and out["east_up"]["refused"] is False


def test_direction_aliases_are_accepted():
    for alias in ("asc", "ASC", "Ascending", " ascending "):
        out = ic.combine_tracks([_track("x", alias, -8.0, 6.7, phi=PHI_ASC)], SLOPE_N, ASPECT_N)
        assert out["flight_directions"]["x"]["direction"] == "ASCENDING"
        assert out["flight_directions"]["x"]["provenance"] == "declared"


# ---------------------------------------------------------------------------
# 8. the entry point, end to end
# ---------------------------------------------------------------------------


def test_combine_tracks_end_to_end_on_a_realistic_asc_plus_desc_pair():
    tracks, s_a, s_d = _synthetic_pair(31.0, SLOPE_OFF, ASPECT_OFF, 6.7, 6.7)
    out = ic.combine_tracks(tracks, SLOPE_OFF, ASPECT_OFF)

    assert out["status"] == "OK" and out["error"] is None
    assert out["velocity_mm_year"] == pytest.approx(31.0, abs=1e-6)
    assert out["consistent"] is True
    assert out["tracks_used"] == ["asc_158", "desc_92"]
    assert out["excluded"] == []
    assert out["temporal_overlap"]["sufficient"] is True
    assert out["east_up"]["refused"] is False
    assert out["uncertainty_mm_year"] < out["slope_parallel"]["best_single_uncertainty_mm_year"]
    # The summary must carry the numbers a human would need, not just "ok".
    # "no disagreement", not "tracks agree": a passing chi-square is the absence of
    # evidence against, and the summary now says so and states the detection limit.
    for fragment in ("mm/yr downslope", "2 tracks", "tighter than the best single track",
                     "no disagreement", "would only catch vertical motion above",
                     "north assumed zero"):
        assert fragment in out["summary"]
    assert out["warnings"] == []


def test_combine_tracks_keeps_the_three_stack_glc105_case_sane():
    """All three real glc_105 stacks at once: 2 ascending + 1 descending.

    All three can enter the slope-parallel WLS, but the east/up decomposition needs exactly
    one of each and must therefore stay unattempted rather than picking an arbitrary pair.
    """
    tracks = [
        _track("337397_IW1_asc", "ASCENDING", -8.2, 6.7,
               theta=math.radians(90 - 33), phi=PHI_ASC),
        _track("118306_IW3_asc", "ASCENDING", -8.9, 7.4,
               theta=math.radians(90 - 43), phi=PHI_ASC),
        _track("196362_IW3_desc", "DESCENDING", -4.1, 7.1,
               theta=math.radians(90 - 41), phi=PHI_DESC),
    ]
    out = ic.combine_tracks(tracks, SLOPE_N, ASPECT_N)
    assert out["n_tracks_used"] == 3
    assert out["slope_parallel"]["dof"] == 2
    assert out["slope_parallel"]["chi_square_critical_95"] == pytest.approx(5.991, abs=1e-6)
    assert out["east_up"] is None          # 2 ascending: no unambiguous pair
    assert out["velocity_mm_year"] is not None


# ---------------------------------------------------------------------------
# 9. degenerate inputs - none of these may raise
# ---------------------------------------------------------------------------


def test_empty_and_none_inputs():
    for empty in ([], None, ()):
        out = ic.combine_tracks(empty, SLOPE_N, ASPECT_N)
        assert out["status"] == "NO_TRACKS"
        assert out["velocity_mm_year"] is None
        assert "nothing to combine" in out["summary"]
    assert ic.combine_slope_parallel([], SLOPE_N, ASPECT_N)["status"] == "NO_TRACKS"
    assert ic.assess_temporal_overlap([])["status"] == "NO_TRACKS"


def test_missing_slope_or_aspect_withholds_rather_than_assuming_one():
    tracks, _, _ = _synthetic_pair(31.0, SLOPE_N, ASPECT_N)
    for slope, aspect in ((None, ASPECT_N), (SLOPE_N, None), (None, None)):
        out = ic.combine_tracks(tracks, slope, aspect)
        assert out["status"] == "NO_SLOPE_GEOMETRY"
        assert out["velocity_mm_year"] is None
        assert "withheld" in out["slope_parallel"]["note"]


@pytest.mark.parametrize(
    "broken, expected_fragment",
    [
        ({"key": "k", "uncertainty_mm_year": 6.7, "lv_theta_rad": THETA,
          "lv_phi_rad": PHI_ASC}, "no finite velocity_mm_year"),
        ({"key": "k", "velocity_mm_year": -8.0, "uncertainty_mm_year": 6.7},
         "missing look vector"),
        ({"key": "k", "velocity_mm_year": -8.0, "lv_theta_rad": THETA,
          "lv_phi_rad": PHI_ASC}, "no finite uncertainty_mm_year"),
        ({"key": "k", "velocity_mm_year": -8.0, "uncertainty_mm_year": 0.0,
          "lv_theta_rad": THETA, "lv_phi_rad": PHI_ASC}, "non-positive"),
        ({"key": "k", "velocity_mm_year": -8.0, "uncertainty_mm_year": -3.0,
          "lv_theta_rad": THETA, "lv_phi_rad": PHI_ASC}, "non-positive"),
        ({"key": "k", "velocity_mm_year": float("nan"), "uncertainty_mm_year": 6.7,
          "lv_theta_rad": THETA, "lv_phi_rad": PHI_ASC}, "no finite velocity_mm_year"),
        ({"key": "k", "velocity_mm_year": -8.0, "uncertainty_mm_year": float("inf"),
          "lv_theta_rad": THETA, "lv_phi_rad": PHI_ASC}, "no finite uncertainty_mm_year"),
    ],
)
def test_unusable_tracks_are_excluded_with_an_explanation_never_defaulted(broken, expected_fragment):
    out = ic.combine_tracks([broken], SLOPE_N, ASPECT_N)
    assert out["status"] == "NO_USABLE_TRACKS"
    assert out["velocity_mm_year"] is None
    assert len(out["excluded"]) == 1
    assert expected_fragment in out["excluded"][0]["reason"]


def test_a_zero_uncertainty_track_cannot_hijack_the_combination():
    """sigma = 0 is a weight of infinity: it would make the combined sigma collapse to zero
    and the combined velocity equal to that one track. It is refused, not clipped."""
    good, s_a, s_d = _synthetic_pair(31.0, SLOPE_N, ASPECT_N)
    good[0]["uncertainty_mm_year"] = 0.0
    out = ic.combine_slope_parallel(good, SLOPE_N, ASPECT_N)
    assert out["tracks_used"] == ["desc_92"]
    assert out["uncertainty_mm_year"] > 0.0


def test_garbage_entries_do_not_raise():
    tracks, _, _ = _synthetic_pair(31.0, SLOPE_N, ASPECT_N)
    out = ic.combine_tracks([None, "not a track", 17, {}] + tracks, SLOPE_N, ASPECT_N)
    assert out["status"] == "OK"
    assert out["n_tracks_used"] == 2
    assert out["velocity_mm_year"] == pytest.approx(31.0, abs=1e-6)
    assert len(out["excluded"]) == 4


def test_duplicate_keys_are_called_out_rather_than_silently_mis_attributed():
    """Per-track output is reported by key; duplicates make it untraceable."""
    a = _track("same", "ASCENDING", -8.0, 6.7, phi=PHI_ASC)
    d = _track("same", "DESCENDING", 4.0, 6.7, phi=PHI_DESC)
    out = ic.combine_tracks([a, d], SLOPE_N, ASPECT_N)
    assert any(w.startswith("KEYS:") for w in out["warnings"])
    # The east/up pair is still resolved by position, so the right two tracks are used.
    assert out["east_up"] is not None and out["east_up"]["refused"] is False


def test_decompose_east_up_never_returns_none_and_never_raises():
    for a, b in ((None, None), ({}, {}), (None, _track("d", "DESCENDING", 4.0, 6.7, phi=PHI_DESC))):
        out = ic.decompose_east_up(a, b)
        assert isinstance(out, dict)
        assert out["refused"] is True
        assert out["reason"]
        assert out["east_mm_year"] is None and out["up_mm_year"] is None


def test_every_public_function_tolerates_a_list_of_junk():
    junk = [None, 3, "x", {"key": "k"}]
    assert ic.combine_slope_parallel(junk, SLOPE_N, ASPECT_N)["status"] == "NO_USABLE_TRACKS"
    assert ic.assess_temporal_overlap(junk)["sufficient"] is None
    assert ic.combine_tracks(junk, SLOPE_N, ASPECT_N)["velocity_mm_year"] is None


# ------------------------------------------------------------------ chi-square power
def _vertical_only_pair(d_up, sigma=1.0):
    """Two tracks seeing PURE vertical motion and NO downslope motion whatsoever.
    LOS forward-modelled longhand so a sign error here cannot cancel out."""
    return [
        _track("asc_158", "ASCENDING",
               _look_longhand(THETA, PHI_ASC)[2] * d_up, sigma, phi=PHI_ASC),
        _track("desc_92", "DESCENDING",
               _look_longhand(THETA, PHI_DESC)[2] * d_up, sigma, phi=PHI_DESC),
    ]


@pytest.mark.parametrize("aspect", [0.0, 180.0])
def test_due_north_or_south_is_blind_to_vertical_motion(aspect):
    """The failure this guards against: pure subsidence published as a 12-sigma downslope
    creep rate with "tracks agree". The two look vectors differ almost purely in East, so
    on a north- or south-facing slope they respond IDENTICALLY to vertical motion and the
    consistency test has zero power. A test with no power has not passed - it never ran -
    so the verdict must be None, never True."""
    # slope 35, not 30: at slope 30 / aspect 180 the limit is 96 mm/yr, which sits right
    # on the 100 mm/yr flag boundary. Power degrades continuously, so a test pinned to a
    # boundary case would be brittle without testing anything extra.
    out = ic.combine_tracks(_vertical_only_pair(-10.0), 35.0, aspect)
    assert out["consistent"] is None
    sp = out["slope_parallel"]
    assert sp["consistency_untestable"] is True
    assert sp["vertical_blind_spot"]["blind_to_vertical"] is True
    assert "UNTESTABLE" in out["summary"]
    # the east/up half, which CAN see it, reports the truth - and used to contradict
    # the downslope claim in the very same payload with nothing reconciling them
    assert out["east_up"]["up_mm_year"] == pytest.approx(-10.0, abs=0.2)


@pytest.mark.parametrize("aspect", [15.0, 30.0, 330.0, 345.0])
def test_off_axis_aspects_can_actually_detect_vertical_contamination(aspect):
    """Away from due north/south the test has real power and must use it."""
    out = ic.combine_tracks(_vertical_only_pair(-30.0), 35.0, aspect)
    sp = out["slope_parallel"]
    assert sp["vertical_blind_spot"]["blind_to_vertical"] is False
    assert out["consistent"] is False
    assert sp["uncertainty_mm_year"] > sp["formal_uncertainty_mm_year"]


def test_the_detection_limit_is_published_even_when_the_test_passes():
    """A passing chi-square means 'no evidence against', not 'assumption confirmed'."""
    tracks, _, _ = _synthetic_pair(20.0, SLOPE_N, ASPECT_N)
    out = ic.combine_tracks(tracks, SLOPE_N, ASPECT_N)
    bs = out["slope_parallel"]["vertical_blind_spot"]
    assert bs["min_detectable_vertical_mm_year"] > 0
    assert "would only notice" in bs["note"]


# ------------------------------------------------------------------ sigma robustness
@pytest.mark.parametrize("sigma", [1e-300, 1e-200, 1e200, 1e300])
def test_extreme_sigmas_are_refused_rather_than_raising(sigma):
    """1/sigma**2 raises ZeroDivisionError below ~2e-162 and OverflowError above ~1.4e154,
    so the documented 'never raises' guarantee was false for these inputs."""
    tracks, _, _ = _synthetic_pair(20.0, SLOPE_N, ASPECT_N)
    tracks[0]["uncertainty_mm_year"] = sigma
    out = ic.combine_slope_parallel(tracks, SLOPE_N, ASPECT_N)      # must not raise
    assert any("implausible uncertainty" in e["reason"] for e in out["excluded"])


def test_a_tiny_positive_sigma_cannot_collapse_the_uncertainty_to_zero():
    """sigma=1.5e-154 is positive, so it passed the sign check, then drove the published
    uncertainty to exactly 0.0 with status OK and nothing flagged as excluded."""
    tracks, _, _ = _synthetic_pair(20.0, SLOPE_N, ASPECT_N)
    tracks[1]["uncertainty_mm_year"] = 1.5e-154
    out = ic.combine_slope_parallel(tracks, SLOPE_N, ASPECT_N)
    assert out["uncertainty_mm_year"] != 0.0
    assert len(out["excluded"]) == 1


def test_inflation_is_continuous_across_the_critical_value():
    """Inflating only once chi2 exceeds the 95% value made the published uncertainty a
    step function: a 7e-5 relative change in the data doubled it, and no factor in
    (1.0, 1.96) was reachable at dof=1."""
    seen = []
    for offset in (2.0, 2.5, 2.7716, 2.7717, 3.0):
        tracks, s_a, s_d = _synthetic_pair(20.0, SLOPE_N, ASPECT_N, 1.0, 1.0)
        tracks[0]["velocity_mm_year"] = s_a * 20.0 + offset / 2
        tracks[1]["velocity_mm_year"] = s_d * 20.0 - offset / 2
        out = ic.combine_slope_parallel(tracks, SLOPE_N, ASPECT_N)
        seen.append(out["uncertainty_mm_year"])
    assert seen == sorted(seen)            # monotonic in the disagreement
    assert seen[3] / seen[2] < 1.05        # no doubling across the critical value


def test_the_vertical_detection_limit_degrades_continuously_toward_north():
    """The boolean is a relevance judgement; the limit itself is the real output, and it
    must vary smoothly rather than jumping, so a reader can see how close to blind a
    given geometry is."""
    limits = []
    for aspect in (45.0, 30.0, 15.0, 5.0, 0.0):
        used = []
        for phi in (PHI_ASC, PHI_DESC):
            sens = _sensitivity_longhand(THETA, phi, 35.0, aspect)
            if abs(sens) >= ic.MIN_PROJECTION_SENSITIVITY:
                used.append({"sensitivity": sens, "sigma": 6.7, "theta": THETA, "phi": phi})
        if len(used) < 2:
            continue
        limits.append(
            ic.vertical_blind_spot(used, ic.chi2_critical_95(1))
            ["min_detectable_vertical_mm_year"]
        )
    assert limits == sorted(limits)          # strictly worse as the slope faces north
    assert limits[-1] > 10 * limits[0]       # and it blows up, it does not merely drift
