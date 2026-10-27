#!/usr/bin/env python3
"""Weekly refresh of measured Sentinel-1 InSAR baselines via ASF HyP3.

Belongs on the WEEKLY cron next to the AI audit, never in the 30-minute physics loop:
HyP3 jobs take roughly an hour, so submission and collection are separate phases and
neither blocks the engine.

Typical cron use (collect first, so last week's jobs land before new ones go out):

    refresh_insar_baselines.py --collect-networks --extend-networks --collect --submit

``--extend-networks`` is what makes the measurement improve over time: it adds each week's
new Sentinel-1 acquisitions to the existing networks, lengthening the observed span and
tightening the velocity uncertainty. Without it a network is frozen at the precision it
had on the day it was built.

Site prioritisation, highest first:
  1. sites currently at RED or ORANGE
  2. cryosphere sites (>= 4500 m, or rock-ice avalanche / GLOF hazard types)
  3. sites whose measured baseline is older than HYP3_BASELINE_MAX_AGE_DAYS
Within a run the per-run job cap and the monthly credit budget both apply, and HyP3's own
reported balance overrides optimistic local accounting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.ingestion import insar_hyp3 as hyp3  # noqa: E402
from app.ingestion import insar_timeseries as ts  # noqa: E402


def _load_status() -> dict:
    path = Path(__file__).resolve().parent.parent / "data" / "perpetual_status.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def prioritised_sites(limit: int) -> list:
    """Order sites by how much a real deformation measurement would tell us."""
    try:
        from app.engine.sites_catalog import load_sites_of_interest

        sites = load_sites_of_interest()
    except Exception as exc:
        print(f"[error] cannot load site catalog: {exc}", flush=True)
        return []

    alerting = {
        s.get("site_id")
        for s in _load_status().get("latest_top_risk_sites", [])
        if s.get("alert") in ("RED", "ORANGE")
    }
    state = hyp3._load_state()
    baselines = state.get("baselines") or {}
    max_age = float(getattr(settings, "hyp3_baseline_max_age_days", 30))
    now = time.time()

    def rank(site):
        key = hyp3.site_key(site.latitude, site.longitude)
        measured = baselines.get(key) or {}
        age_days = (
            (now - float(measured.get("measured_utc", 0.0))) / 86400.0 if measured else 1e9
        )
        is_cryo = site.elevation_m >= 4500.0 or site.hazard_type in (
            "rock_ice_avalanche",
            "glof",
        )
        return (
            0 if site.site_id in alerting else 1,
            0 if is_cryo else 1,
            -age_days,  # oldest baseline first within a tier
        )

    ordered = sorted(sites, key=rank)
    # Skip sites whose baseline is still fresh - re-measuring them wastes credits.
    fresh = {
        k for k, m in baselines.items()
        if (now - float(m.get("measured_utc", 0.0))) / 86400.0 <= max_age
    }
    return [s for s in ordered if hyp3.site_key(s.latitude, s.longitude) not in fresh][:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="submit new single-pair InSAR jobs")
    parser.add_argument("--collect", action="store_true", help="download and measure finished jobs")
    parser.add_argument("--status", action="store_true", help="print capability and exit")
    parser.add_argument("--limit", type=int, default=None, help="max jobs to submit this run")
    parser.add_argument(
        "--network", metavar="SITE_ID",
        help="submit a full SBAS pair network for one site (~40 credits) - the only path "
             "to a MEASURED velocity, since a single pair is atmosphere-dominated",
    )
    parser.add_argument(
        "--collect-networks", action="store_true",
        help="poll all submitted networks, double-difference finished pairs, and invert",
    )
    parser.add_argument(
        "--direction", choices=["ASCENDING", "DESCENDING"], default=None,
        help="restrict --network to one flight direction",
    )
    parser.add_argument(
        "--both-directions", action="store_true",
        help="with --network, submit an ascending AND a descending network for the "
             "site (~2x the credits, 160 at glc_105). The second geometry always buys a "
             "testable assumption (chi-square on one spare observation) and an East/Up "
             "split no single track can give. It buys COVERAGE only on moderate slopes: "
             "six of twelve aspects are single-geometry at 35 degrees, ZERO at 45.5 "
             "where both tracks clear the gate everywhere. Precision 1.01x-1.41x. "
             "See INSAR_NOTES.md section 13.",
    )
    parser.add_argument(
        "--extend-networks", action="store_true",
        help="add newly available acquisitions to every existing network. Each new epoch "
             "lengthens the observed span, which is what actually tightens the velocity "
             "uncertainty over time - a stale network stays at its original precision "
             "forever. Costs 1 credit per genuinely new pair; already-submitted pairs are "
             "deduplicated, so this is safe to run weekly.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show what would be submitted without spending credits (default)",
    )
    args = parser.parse_args()

    cap = hyp3.describe_capability()
    print("=== HyP3 measured-InSAR capability ===", flush=True)
    for k, v in cap.items():
        print(f"  {k}: {v}", flush=True)
    remote = hyp3.remaining_credits()
    print(f"  remaining_credits (live): {remote}", flush=True)

    if args.status:
        return 0
    if not cap["enabled"]:
        print("\n[hold] HYP3_ENABLED=0 - nothing to do.", flush=True)
        return 0
    if not cap["credentials_present"]:
        print("\n[hold] no EARTHDATA_TOKEN - nothing to do.", flush=True)
        return 0

    exit_code = 0

    if args.collect_networks:
        print("\n--- Collect networks (SBAS) ---", flush=True)
        state = hyp3._load_state()
        networks = state.get("networks") or {}
        if not networks:
            print("  no networks submitted yet", flush=True)
        for key, net in list(networks.items()):
            direction = ts.split_network_key(key)[1]
            res = ts.collect_network(net["lat"], net["lon"], direction)
            print(f"  {net.get('site_id', key)} [{direction}]: "
                  f"measured={res.get('measured')} pending={res.get('pending')} "
                  f"unusable={res.get('unusable')}", flush=True)
            print(f"    {res.get('summary')}", flush=True)

    if args.extend_networks:
        print("\n--- Extend networks (new acquisitions) ---", flush=True)
        state = hyp3._load_state()
        networks = state.get("networks") or {}
        if not networks:
            print("  no networks to extend", flush=True)
        for key, net in list(networks.items()):
            direction = ts.split_network_key(key)[1]
            label = f"{net.get('site_id', key)} [{direction}]"
            # Extend on the SAME geometry the network was built from. Planning without a
            # direction picks the deepest stack regardless, which may be the OTHER track:
            # submit_network then keys the result by that stack's real direction, finds no
            # existing record there, and so re-submits every pair as if it were new. On
            # the production network that is 90 already-paid-for credits spent again.
            # Resolve the direction first; if it still cannot be established, skip.
            if direction == ts.DIRECTION_UNKNOWN:
                state_now = hyp3._load_state()
                resolved = ts.backfill_network_direction(
                    state_now, net["lat"], net["lon"]
                )
                if resolved:
                    hyp3._save_state(state_now)
                    direction = resolved
                    print(f"  {label}: direction resolved to {resolved}", flush=True)
                else:
                    print(f"  {label}: flight direction unknown and not resolvable - "
                          f"skipped rather than planning blind and re-paying for "
                          f"existing pairs", flush=True)
                    continue
            plan_direction = direction
            if args.dry_run:
                plan = ts.plan_network(net["lat"], net["lon"], direction=plan_direction)
                if plan is None:
                    print(f"  {label}: no usable burst network", flush=True)
                    continue
                already = {(p.get("reference"), p.get("secondary"))
                           for p in net.get("pairs", [])}
                todo = [p for p in plan.pairs if (p[0], p[1]) not in already]
                print(f"  [dry-run] {label}: {len(todo)} new pair(s) would be submitted "
                      f"({len(already)} already in the network)", flush=True)
                continue
            res = ts.submit_network(
                net["lat"], net["lon"], site_id=net.get("site_id", "site"),
                direction=plan_direction,
            )
            print(f"  {label}: {res.get('status')} "
                  f"submitted={res.get('submitted', 0)} "
                  f"total_pairs={res.get('total_pairs', len(net.get('pairs', [])))}", flush=True)

    if args.network:
        try:
            from app.engine.sites_catalog import load_sites_of_interest

            site = next(s for s in load_sites_of_interest() if s.site_id == args.network)
        except StopIteration:
            print(f"\n[error] unknown site_id {args.network!r}", flush=True)
            return 2
        print(f"\n--- SBAS network for {site.site_id} "
              f"({site.latitude:.4f}, {site.longitude:.4f}, {site.elevation_m:.0f} m) ---", flush=True)

        if args.both_directions:
            plans = ts.plan_networks(site.latitude, site.longitude)
            if not plans:
                print("  no usable burst network at this location", flush=True)
                return 1
            if len(plans) < 2:
                print(f"  [warn] only {sorted(plans)} available here - the asc/desc "
                      f"combination needs both, so this will stay a single-geometry "
                      f"measurement", flush=True)
        else:
            one = ts.plan_network(site.latitude, site.longitude, direction=args.direction)
            if one is None:
                print(f"  no usable burst network at this location"
                      + (f" on a {args.direction} pass" if args.direction else ""), flush=True)
                return 1
            plans = {one.flight_direction: one}

        total_credits = sum(len(p.pairs) for p in plans.values())
        for direction, plan in sorted(plans.items()):
            print(f"  [{direction}] planned: {len(plan.pairs)} pairs over "
                  f"{len(plan.epochs)} epochs (redundancy {plan.redundancy:.2f}), "
                  f"burst {plan.burst_id} {plan.swath} {plan.polarization}"
                  + (f", path {plan.path_number}" if plan.path_number else ""), flush=True)
        if args.dry_run:
            print(f"  [dry-run] would spend {total_credits} credits across "
                  f"{len(plans)} network(s) - nothing submitted", flush=True)
        else:
            for direction, plan in sorted(plans.items()):
                res = ts.submit_network(
                    site.latitude, site.longitude, site_id=site.site_id,
                    direction=direction,
                )
                print(f"  [{direction}] {res}", flush=True)

    if args.collect:
        print("\n--- Collect ---", flush=True)
        summary = hyp3.collect_measurements()
        print(f"  {summary}", flush=True)

    if args.submit:
        limit = args.limit if args.limit is not None else int(
            getattr(settings, "hyp3_max_jobs_per_run", 20)
        )
        targets = prioritised_sites(limit)
        print(f"\n--- Submit ({len(targets)} candidate sites) ---", flush=True)
        if args.dry_run:
            print("  [dry-run] no jobs submitted, no credits spent", flush=True)
            for s in targets:
                print(f"    would submit: {s.site_id} ({s.latitude:.4f}, {s.longitude:.4f})",
                      flush=True)
            return 0
        submitted = 0
        for site in targets:
            result = hyp3.request_measurement(
                site.latitude, site.longitude, name=f"mhews-{site.site_id}"[:100]
            )
            status = result.get("status")
            print(f"    {site.site_id}: {status}", flush=True)
            if status == "SUBMITTED":
                submitted += 1
            elif status == "BUDGET_EXHAUSTED":
                print("  [stop] monthly credit budget reached", flush=True)
                break
        print(f"  submitted {submitted} job(s); "
              f"credits spent this period: {hyp3.credits_spent_this_period()}", flush=True)

    if not (args.submit or args.collect or args.network or args.collect_networks
            or args.extend_networks):
        print("\n[info] nothing requested. Use --network SITE_ID for a measured "
              "velocity, --collect-networks to invert finished ones, or "
              "--collect/--submit for single-pair screening.", flush=True)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
