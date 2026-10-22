"""Debris-flow and rock-ice avalanche runout routing on 30m DEM grids.

Combines steepest-descent / Holmgren multi-direction routing with a Voellmy-fluid
frictional stopping criterion (Coulomb basal friction mu and turbulent drag xi)
to map downstream inundation corridors and landslide dam choke points.
"""

from typing import Dict, List, Tuple
import numpy as np

# 8-neighbor offsets: (d_row, d_col, distance_multiplier)
D8_NEIGHBORS: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1.41421356),
    (-1, 1, 1.41421356),
    (1, -1, 1.41421356),
    (1, 1, 1.41421356),
)


def simulate_debris_runout(
    dem_m: np.ndarray,
    source_mask: np.ndarray,
    cell_size_m: float = 30.0,
    basal_friction_mu: float = 0.12,
    turbulent_drag_xi_m_s2: float = 1500.0,
    initial_flow_depth_m: float = 4.0,
    max_steps: int = 250,
) -> Dict[str, object]:
    """Route rock-ice avalanche or debris flow from unstable source cells (FS < 1.0).

    Uses a Voellmy energy-head dissipation law along steepest descent paths:
        dv^2 / (2g * ds) = sin(beta) - mu * cos(beta) - v^2 / (xi * h)

    Args:
        dem_m: 2D elevation grid in meters.
        source_mask: 2D boolean mask of unstable release cells (e.g., FS < 1.0).
        cell_size_m: Grid resolution in meters (default 30m).
        basal_friction_mu: Dry Coulomb friction coefficient mu (0.08-0.20 for rock-ice/debris).
        turbulent_drag_xi_m_s2: Voellmy turbulent friction parameter xi (m/s^2).
        initial_flow_depth_m: Characteristic slab/flow depth h (m).
        max_steps: Maximum propagation steps down-valley.

    Returns:
        Dictionary with inundation_mask, max_velocity_m_s grid, runout_length_m,
        and ordered centerline path coordinates (row, col).
    """
    dem = np.asarray(dem_m, dtype=np.float64)
    src = np.asarray(source_mask, dtype=bool)
    if dem.ndim != 2 or src.shape != dem.shape:
        raise ValueError("dem_m and source_mask must be 2D arrays of identical shape.")

    rows, cols = dem.shape
    inundation_mask = src.copy()
    max_velocity = np.zeros_like(dem, dtype=np.float64)
    g = 9.81

    source_indices = np.argwhere(src)
    if source_indices.size == 0:
        return {
            "inundation_mask": inundation_mask,
            "max_velocity_m_s": max_velocity,
            "runout_length_m": 0.0,
            "peak_velocity_m_s": 0.0,
            "affected_cells": 0,
            "paths": [],
        }

    all_paths: List[List[Tuple[int, int]]] = []
    max_runout_length = 0.0

    for start_r, start_c in source_indices:
        r, c = int(start_r), int(start_c)
        v_sq = 0.0
        dist_traveled = 0.0
        path: List[Tuple[int, int]] = [(r, c)]

        for _ in range(max_steps):
            best_slope = -np.inf
            best_next = None
            best_step_dist = cell_size_m

            for dr, dc, mult in D8_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    step_dist = cell_size_m * mult
                    dz = dem[r, c] - dem[nr, nc]
                    slope_tan = dz / step_dist
                    if slope_tan > best_slope:
                        best_slope = slope_tan
                        best_next = (nr, nc)
                        best_step_dist = step_dist

            if best_next is None:
                break

            beta = np.arctan(max(best_slope, -0.5))
            sin_b = np.sin(beta)
            cos_b = np.cos(beta)

            # Voellmy acceleration a = g * (sin(beta) - mu * cos(beta) - v^2 / (xi * h))
            drag_term = v_sq / max(turbulent_drag_xi_m_s2 * initial_flow_depth_m, 1.0)
            accel = g * (sin_b - basal_friction_mu * cos_b) - g * drag_term
            v_sq_next = v_sq + 2.0 * accel * best_step_dist

            if v_sq_next <= 0.0:
                break

            r, c = best_next
            v_sq = v_sq_next
            vel = float(np.sqrt(v_sq))
            dist_traveled += best_step_dist

            inundation_mask[r, c] = True
            if vel > max_velocity[r, c]:
                max_velocity[r, c] = vel
            path.append((r, c))

        max_runout_length = max(max_runout_length, dist_traveled)
        all_paths.append(path)

    return {
        "inundation_mask": inundation_mask,
        "max_velocity_m_s": max_velocity,
        "runout_length_m": round(float(max_runout_length), 2),
        "peak_velocity_m_s": round(float(np.max(max_velocity)), 2),
        "affected_cells": int(np.sum(inundation_mask)),
        "paths": all_paths,
    }
