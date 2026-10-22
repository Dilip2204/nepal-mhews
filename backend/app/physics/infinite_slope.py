"""Infinite Slope Stability Factor of Safety (FS) Calculator & Accuracy Extensions.

Implements the domain equation specified in AGENTS.md:
    FS = (c' + (gamma * z - gamma_w * z_w) * cos^2(beta) * tan(phi'))
         / (gamma * z * sin(beta) * cos(beta))
Critical failure threshold: FS < 1.0.

Also includes physical accuracy enhancements:
1. Slope-dependent regolith depth estimation (`estimate_regolith_depth`).
2. High-altitude (>4500m) permafrost ice-cohesion degradation (`compute_permafrost_effective_cohesion`).
3. Vectorized Monte Carlo probabilistic failure probability `P(FS < 1.0)` (`compute_probabilistic_fs`).
"""

from typing import Dict, Union
import numpy as np
from app.core.config import settings

ArrayLike = Union[float, np.ndarray]


def estimate_regolith_depth(
    beta_deg: ArrayLike,
    z_min_m: float = 0.4,
    z_max_m: float = 4.0,
    decay_rate: float = 0.045,
) -> np.ndarray:
    """Estimate vertical soil/regolith thickness z (m) as an inverse function of slope angle.

    Layman intuition: Gentle valley floors collect thick soil (up to 4m), whereas very
    steep 55-degree cliffs shed loose dirt and only retain a thin veneer (0.4m - 1.0m).
    """
    slope = np.clip(np.asarray(beta_deg, dtype=np.float64), 0.0, 85.0)
    z_est = z_min_m + (z_max_m - z_min_m) * np.exp(-decay_rate * slope)
    return np.round(z_est, 3)


def compute_permafrost_effective_cohesion(
    base_cohesion_kpa: ArrayLike,
    elevation_m: ArrayLike,
    pdd_c_days: float = 0.0,
    permafrost_bonus_kpa: float = 18.0,
    thaw_sensitivity_kpa_per_pdd: float = 1.1,
    cryosphere_elevation_m: float = 4500.0,
) -> np.ndarray:
    """Adjust effective cohesion c' (kPa) for high-altitude permafrost ice-glue and thaw.

    Layman intuition: Above 4500m, frozen ice inside rock fractures acts like cement,
    adding extra grip (`permafrost_bonus_kpa`). During a warm spell (`pdd_c_days > 0`),
    that interstitial ice melts and reduces the glue holding the rock face together.
    """
    c_base = np.asarray(base_cohesion_kpa, dtype=np.float64)
    elev = np.asarray(elevation_m, dtype=np.float64)

    # Active permafrost zone above 4500m
    in_cryosphere = elev >= cryosphere_elevation_m
    thaw_loss = min(permafrost_bonus_kpa, max(0.0, pdd_c_days) * thaw_sensitivity_kpa_per_pdd)
    net_ice_cohesion = max(0.0, permafrost_bonus_kpa - thaw_loss)

    c_effective = np.where(in_cryosphere, c_base + net_ice_cohesion, c_base)
    return np.maximum(c_effective, 0.5)


def compute_infinite_slope_fs(
    c_prime: ArrayLike,
    gamma: ArrayLike,
    z: ArrayLike,
    z_w: ArrayLike,
    beta_deg: ArrayLike,
    phi_prime_deg: ArrayLike,
    gamma_w: float = settings.water_unit_weight_kn_m3,
    seismic_kh: ArrayLike = 0.0,
    surface_surcharge_q_kpa: ArrayLike = 0.0,
    matric_suction_kpa: ArrayLike = 0.0,
) -> np.ndarray:
    """Compute the Generalized Infinite Slope Factor of Safety (FS).

    Args:
        c_prime: Effective cohesion c' (kPa = kN/m^2), combining mineral, root, and permafrost ice glue.
        gamma: Unit weight of slope material (kN/m^3).
        z: Vertical soil/regolith/ice slab thickness (m), must be > 0.
        z_w: Saturated water table height above slip surface (m), clamped to [0, z].
        beta_deg: Slope angle beta in degrees.
        phi_prime_deg: Effective internal friction angle phi' in degrees.
        gamma_w: Unit weight of water (kN/m^3), default 9.81.
        seismic_kh: Horizontal seismic coefficient k_h = PGA / g (dimensionless, default 0.0).
        surface_surcharge_q_kpa: Surface load q_s (kPa) from snowpack SWE or forest biomass (default 0.0).
        matric_suction_kpa: Unsaturated capillary suction strength (kPa) above water table (default 0.0).

    Returns:
        NumPy array (or 0-d scalar array) of Factor of Safety values.
    """
    c_arr = np.asarray(c_prime, dtype=np.float64) + np.asarray(
        matric_suction_kpa, dtype=np.float64
    )
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    q_arr = np.maximum(np.asarray(surface_surcharge_q_kpa, dtype=np.float64), 0.0)
    if np.any(z_arr <= 0):
        raise ValueError("Slab thickness 'z' must be strictly positive.")
    if np.any(gamma_arr <= 0):
        raise ValueError("Material unit weight 'gamma' must be strictly positive.")

    zw_arr = np.clip(np.asarray(z_w, dtype=np.float64), 0.0, z_arr)
    beta_rad = np.deg2rad(np.clip(np.asarray(beta_deg, dtype=np.float64), 0.1, 89.0))
    phi_rad = np.deg2rad(np.asarray(phi_prime_deg, dtype=np.float64))
    kh_arr = np.asarray(seismic_kh, dtype=np.float64)

    cos_b = np.cos(beta_rad)
    sin_b = np.sin(beta_rad)
    cos2_b = cos_b * cos_b
    tan_phi = np.tan(phi_rad)

    total_vertical_stress = gamma_arr * z_arr + q_arr
    effective_normal_stress = (total_vertical_stress - gamma_w * zw_arr) * cos2_b
    if np.any(kh_arr > 0):
        effective_normal_stress = (
            effective_normal_stress - total_vertical_stress * kh_arr * sin_b * cos_b
        )
    effective_normal_stress = np.maximum(effective_normal_stress, 0.0)

    resisting_stress = c_arr + effective_normal_stress * tan_phi
    driving_stress = total_vertical_stress * (sin_b * cos_b + kh_arr * cos2_b)
    driving_stress = np.maximum(driving_stress, 1e-9)

    fs = resisting_stress / driving_stress
    return fs


def compute_probabilistic_fs(
    c_prime: float,
    gamma: float,
    z: float,
    z_w: float,
    beta_deg: float,
    phi_prime_deg: float,
    seismic_kh: float = 0.0,
    n_simulations: int = 250,
    random_state: int = 42,
) -> Dict[str, float]:
    """Run Monte Carlo uncertainty analysis to estimate Failure Probability P(FS < 1.0).

    Layman intuition: Because underground soil glue (`c_prime`) and friction (`phi'`)
    vary naturally across a mountain, we test 250 slight variations around our best
    estimate and count what percentage of them collapse (FS < 1.0).
    """
    rng = np.random.default_rng(random_state)
    c_samples = np.maximum(rng.normal(c_prime, 0.20 * max(c_prime, 1.0), n_simulations), 0.5)
    phi_samples = np.clip(rng.normal(phi_prime_deg, 2.5, n_simulations), 15.0, 48.0)
    z_samples = np.maximum(rng.normal(z, 0.12 * z, n_simulations), 0.3)

    fs_samples = compute_infinite_slope_fs(
        c_prime=c_samples,
        gamma=gamma,
        z=z_samples,
        z_w=z_w,
        beta_deg=beta_deg,
        phi_prime_deg=phi_samples,
        seismic_kh=seismic_kh,
    )
    prob_failure = float(np.mean(fs_samples < settings.critical_fs_threshold))
    return {
        "deterministic_fs": round(
            float(
                compute_infinite_slope_fs(
                    c_prime, gamma, z, z_w, beta_deg, phi_prime_deg, seismic_kh=seismic_kh
                )
            ),
            4,
        ),
        "mean_fs": round(float(np.mean(fs_samples)), 4),
        "p10_fs": round(float(np.percentile(fs_samples, 10)), 4),
        "p90_fs": round(float(np.percentile(fs_samples, 90)), 4),
        "failure_probability": round(prob_failure, 4),
    }


def classify_stability(fs: ArrayLike) -> Dict[str, object]:
    """Evaluate Factor of Safety array against domain thresholds (FS < 1.0 critical)."""
    fs_arr = np.asarray(fs, dtype=np.float64)
    critical_mask = fs_arr < settings.critical_fs_threshold
    watch_mask = (fs_arr >= settings.critical_fs_threshold) & (
        fs_arr < settings.watch_fs_threshold
    )

    min_fs = float(np.min(fs_arr))
    mean_fs = float(np.mean(fs_arr))
    critical_fraction = float(np.mean(critical_mask))

    if min_fs < settings.critical_fs_threshold:
        regime = "CRITICAL_FAILURE"
    elif min_fs < settings.watch_fs_threshold:
        regime = "MARGINAL_WATCH"
    else:
        regime = "STABLE"

    return {
        "min_fs": round(min_fs, 4),
        "mean_fs": round(mean_fs, 4),
        "critical_cell_count": int(np.sum(critical_mask)),
        "watch_cell_count": int(np.sum(watch_mask)),
        "critical_fraction": round(critical_fraction, 4),
        "stability_regime": regime,
        "critical_mask": critical_mask,
    }
