"""Guard: a fabricated number must not be able to reach the physics engine.

Why this file exists
--------------------
Twice now the factor of safety has been fed values that looked site-specific and were
not. The second time was worse than the first, because the values *varied* per site and
were documented as observations:

.. code-block:: python

    hash_seed = abs(hash(f"{lat:.3f}_{lon:.3f}_{elev:.1f}"))
    dist_road = 18.0 + (hash_seed % 280)                               # "OpenStreetMap"
    ndvi = 0.62 - 0.002*(slp - 35.0) + 0.06*math.sin(hash_seed % 5)    # "Sentinel-2 NDVI"

Varying per site is not the same thing as measured per site. Worse, ``hash()`` on a str
is salted per interpreter, so those numbers are not even stable across a restart: the
same slope gets a different "observation" every time the orchestrator is relaunched
(see :func:`test_builtin_str_hash_is_not_stable_across_processes`, which demonstrates
the salt rather than taking it on trust).

Review caught both incidents. Review is not a control. This file makes the failure mode
structurally detectable instead.

The contract being enforced
---------------------------
1. **No fabricated value reaches a declared physical quantity.** Not "do not call
   ``hash()``" - ``hash()`` for a cache key or a bulletin id is perfectly fine, and this
   guard deliberately does not flag it. What is forbidden is a value *derived* from a
   hash or a random draw flowing into a field of :class:`RawEarthObservationInputs` or
   :class:`EnvironmentalFeatures`. The scanner is sink-driven for exactly that reason:
   it only reports when the tainted value lands on a named physical quantity.
2. **Absent, never assumed.** A source that cannot be fetched yields ``None``. It does
   not yield a plausible default, an interpolation from a neighbouring site, or a
   nationwide frozen constant dressed up as a measurement.
3. **Every measured value carries its provenance** - the source, the URL, the method -
   all the way through any cache it passes through.

Parked assertions (how to turn them on)
---------------------------------------
``backend/app/engine/environmental_features.py`` violates (1), (2) and (3) today. That
is the whole reason this file was written, and its replacement by
``app/ingestion/real_site_features.py`` is tracked separately. So the assertions aimed
at it are marked ``xfail(strict=True)`` behind two module-level flags:

* ``ENFORCE_REAL_ENVIRONMENTAL_FEATURES`` - hash-derived physics inputs, cross-process
  reproducibility, rounding-grid quantisation, provenance on the consumed record.
* ``ENFORCE_NO_FROZEN_FALLBACKS`` - ``env.clay_fraction if env else 0.16`` style silent
  substitution of a nationwide constant for an absent measurement.

Flipping either to ``True`` is a one-line change and enables every assertion in its
group. Because the marks are ``strict``, the suite goes red the moment the fabrication
is actually gone: an XPASS here means "the TODO is done, turn the guard on permanently".

Everything else in this file runs for real today, including the scanner's own
self-tests, so the guard cannot rot into a no-op while it waits.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pytest

from app.engine.environmental_features import (
    EnvironmentalFeatures,
    compute_site_environmental_features,
)
from app.ingestion import real_site_features as rsf
from app.physics.feature_synthesizer import RawEarthObservationInputs


# ======================================================================================
# TODO(real-inputs): flip to True once app/engine/environmental_features.py no longer
# derives physics inputs from hash() - i.e. once it is backed by
# app/ingestion/real_site_features.py (SoilGrids / Overpass / Copernicus DEM) and
# absent measurements come back absent. One line, enables four assertions.
ENFORCE_REAL_ENVIRONMENTAL_FEATURES = True

# TODO(absent-not-assumed): flip to True once perpetual_orchestrator.py and
# api/routes.py stop substituting frozen nationwide constants for absent per-site
# measurements (`env.clay_fraction if env else 0.16`). One line, enables one assertion.
ENFORCE_NO_FROZEN_FALLBACKS = True
# ======================================================================================


def _parked(enabled: bool, todo: str) -> pytest.MarkDecorator:
    """Mark an assertion as a known, tracked failure until `enabled` is flipped on."""
    return pytest.mark.xfail(
        not enabled,
        strict=True,
        raises=AssertionError,
        reason=(
            f"PARKED, EXPECTED TO FAIL TODAY: {todo} "
            "An XPASS here means the fabrication is gone - set the matching "
            "ENFORCE_* flag at the top of test_no_fabricated_inputs.py to True."
        ),
    )


parked_hash = _parked(
    ENFORCE_REAL_ENVIRONMENTAL_FEATURES,
    "environmental_features.py still derives physics inputs from hash().",
)
parked_defaults = _parked(
    ENFORCE_NO_FROZEN_FALLBACKS,
    "the orchestrator still substitutes frozen constants for absent measurements.",
)


BACKEND_DIR = Path(rsf.__file__).resolve().parents[2]
APP_DIR = BACKEND_DIR / "app"
ENVIRONMENTAL_FEATURES_PY = APP_DIR / "engine" / "environmental_features.py"


def _rel(path: Path) -> str:
    return str(path.relative_to(BACKEND_DIR.parent))


# --------------------------------------------------------------------------------------
# what counts as a declared physical quantity
# --------------------------------------------------------------------------------------
# Taken from the dataclasses themselves, so a new field is covered the day it is added.
# `site_id` is an identifier, not a measurement, and is the only exclusion.
_NON_MEASUREMENT_FIELDS = {"site_id"}

PHYSICS_FIELDS: Set[str] = (
    {f.name for f in dataclass_fields(RawEarthObservationInputs)}
    | {f.name for f in dataclass_fields(EnvironmentalFeatures)}
) - _NON_MEASUREMENT_FIELDS

# Fields echoed straight back from the caller's arguments rather than derived, so they
# are not evidence of anything in the behavioural comparisons below.
_PASSTHROUGH_FIELDS = {"site_id", "latitude", "longitude", "elevation_m", "slope_deg"}

PHYSICS_CONSTRUCTORS = {"RawEarthObservationInputs", "EnvironmentalFeatures"}


# --------------------------------------------------------------------------------------
# AST taint scan: hash()/random -> a named physical quantity
# --------------------------------------------------------------------------------------
_RNG_FACTORIES = {
    "np.random.default_rng", "numpy.random.default_rng",
    "np.random.RandomState", "numpy.random.RandomState",
    "random.Random", "np.random.Generator",
}

# Narrow, deliberate exemptions, keyed by (file, function). An entry here only silences
# the scanner inside that one function, and `test_allowlisted_functions_do_not_build_
# physics_records` proves the exemption cannot be used to feed the physics.
_SYNTHETIC_ALLOWLIST: Dict[Tuple[str, str], str] = {
    ("backend/app/ml/susceptibility.py", "generate_synthetic_basin_training_data"): (
        "explicitly-named synthetic fallback *training* data for standalone unit tests. "
        "It draws slope_deg/elevation_m from a seeded rng, but it never claims to be an "
        "observation of a real site and never constructs a physics input record."
    ),
}


@dataclass(frozen=True)
class _Origin:
    kind: str          # "hash()" | "random draw"
    lineno: int
    src: str


@dataclass(frozen=True)
class _Taint:
    origin: _Origin
    chain: Tuple[str, ...]   # variable names the value flowed through, origin first


@dataclass(frozen=True)
class Fabrication:
    """One fabricated value landing on a named physical quantity."""

    rel_path: str
    lineno: int
    quantity: str
    sink: str
    sink_kind: str
    scope: str
    origin_kind: str
    origin_lineno: int
    origin_src: str
    chain: Tuple[str, ...]

    def describe(self) -> str:
        # The last hop is often the local variable the quantity is named after; saying
        # "aspect_deg <- aspect_deg <- ..." helps nobody.
        chain = self.chain[:-1] if self.chain[-1:] == (self.quantity,) else self.chain
        flow = " <- ".join(reversed(chain)) if chain else "(directly)"
        return (
            f"{self.rel_path}:{self.lineno}: physical quantity '{self.quantity}' "
            f"is fabricated ({self.sink}, in {self.scope})\n"
            f"        value flow: {self.quantity} <- {flow} "
            f"<- {self.origin_kind} at line {self.origin_lineno}\n"
            f"        origin:     {self.origin_src}"
        )


def _dotted(node: ast.AST) -> Optional[str]:
    """`np.random.default_rng` -> 'np.random.default_rng'; anything else -> None."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _src(node: ast.AST, limit: int = 110) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is total on parsed trees
        return "<unparseable>"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _target_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _target_names(elt)]
    return []


def _scopes(tree: ast.Module) -> List[Tuple[str, List[ast.AST]]]:
    """Split a module into independent scopes: module body and each def/class body.

    Taint is not propagated across scopes. A hash drawn in one function cannot be blamed
    for a quantity assigned in another, which keeps the finding's value-flow trace
    honest and the false-positive rate at zero in practice.
    """
    out: List[Tuple[str, List[ast.AST]]] = []

    def walk(node: ast.AST, name: str) -> None:
        own: List[ast.AST] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, f"{name}.{child.name}" if name else child.name)
            else:
                own.append(child)
        out.append((name or "<module>", own))

    walk(tree, "")
    return out


def _scan_scope(
    scope: str, stmts: Sequence[ast.AST], rel_path: str
) -> List[Fabrication]:
    taint: Dict[str, _Taint] = {}
    rng_handles: Set[str] = set()

    def expr_taint(expr: Optional[ast.AST]) -> Optional[_Taint]:
        """Does this expression carry a hash- or random-derived value?"""
        if expr is None:
            return None
        inherited: Optional[_Taint] = None
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Call):
                dotted = _dotted(sub.func)
                if dotted == "hash":
                    return _Taint(_Origin("hash()", sub.lineno, _src(sub)), ())
                if dotted is not None:
                    parts = dotted.split(".")
                    if "random" in parts[:-1] or dotted in _RNG_FACTORIES:
                        return _Taint(
                            _Origin("random draw", sub.lineno, _src(sub)), ()
                        )
                if (
                    isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id in rng_handles
                ):
                    return _Taint(_Origin("random draw", sub.lineno, _src(sub)), ())
            elif isinstance(sub, ast.Name) and sub.id in taint and inherited is None:
                inherited = taint[sub.id]
        return inherited

    def assignments() -> Iterable[ast.AST]:
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    yield node

    # Pass 1: propagate taint to local names. Repeated to a fixpoint so that source
    # order (and branches) cannot hide a flow; three rounds is well past saturation for
    # the chain depths that occur here.
    for _ in range(3):
        for node in assignments():
            value = getattr(node, "value", None)
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n for t in targets for n in _target_names(t)]
            if isinstance(value, ast.Call) and _dotted(value.func) in _RNG_FACTORIES:
                rng_handles.update(names)
                continue
            found = expr_taint(value)
            if found is None:
                continue
            for name in names:
                if name not in taint:
                    taint[name] = _Taint(found.origin, found.chain + (name,))

    if (rel_path, scope.split(".")[-1]) in _SYNTHETIC_ALLOWLIST:
        return []

    # Pass 2: report only where a tainted value lands on a declared physical quantity.
    findings: Dict[Tuple[int, str], Fabrication] = {}

    def record(
        node: ast.AST, quantity: str, sink: str, sink_kind: str, found: _Taint
    ) -> None:
        key = (node.lineno, quantity)
        findings.setdefault(
            key,
            Fabrication(
                rel_path=rel_path,
                lineno=node.lineno,
                quantity=quantity,
                sink=sink,
                sink_kind=sink_kind,
                scope=scope,
                origin_kind=found.origin.kind,
                origin_lineno=found.origin.lineno,
                origin_src=found.origin.src,
                chain=found.chain,
            ),
        )

    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                found = expr_taint(getattr(node, "value", None))
                if found is None:
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in PHYSICS_FIELDS:
                        record(
                            node,
                            target.id,
                            "assigned to a physics input",
                            "assignment",
                            found,
                        )
                    elif (
                        isinstance(target, ast.Attribute)
                        and target.attr in PHYSICS_FIELDS
                    ):
                        record(
                            node,
                            target.attr,
                            f"assigned to {_src(target)}",
                            "attribute",
                            found,
                        )
            elif isinstance(node, ast.Call):
                callee = _dotted(node.func) or "<call>"
                for kw in node.keywords:
                    found = expr_taint(kw.value)
                    if found is None:
                        continue
                    is_constructor = callee.split(".")[-1] in PHYSICS_CONSTRUCTORS
                    if kw.arg in PHYSICS_FIELDS:
                        record(
                            kw.value,
                            kw.arg,
                            f"keyword argument to {callee}()",
                            "constructor" if is_constructor else "call",
                            found,
                        )
                    elif kw.arg is None and is_constructor:
                        record(
                            kw.value,
                            f"**{_src(kw.value, 40)}",
                            f"splatted into {callee}()",
                            "constructor",
                            found,
                        )
            elif isinstance(node, ast.Dict):
                for key_node, value_node in zip(node.keys, node.values):
                    if (
                        isinstance(key_node, ast.Constant)
                        and key_node.value in PHYSICS_FIELDS
                    ):
                        found = expr_taint(value_node)
                        if found is not None:
                            record(
                                value_node,
                                str(key_node.value),
                                "dict entry keyed by a physics input",
                                "dict",
                                found,
                            )

    # One quantity fabricated once is one finding. Where the same value is both built
    # into a local and then handed to the physics record, report the point closest to
    # the physics and keep the full value-flow trace.
    deduped: Dict[Tuple[str, str], List[Fabrication]] = {}
    for finding in findings.values():
        deduped.setdefault((finding.scope, finding.quantity), []).append(finding)
    out: List[Fabrication] = []
    for group in deduped.values():
        at_physics = [f for f in group if f.sink_kind == "constructor"]
        out.extend(at_physics or group)
    return sorted(out, key=lambda f: (f.lineno, f.quantity))


def scan_source(source: str, rel_path: str) -> List[Fabrication]:
    """Every fabricated value that reaches a declared physical quantity."""
    tree = ast.parse(source)
    out: List[Fabrication] = []
    for scope, stmts in _scopes(tree):
        out.extend(_scan_scope(scope, stmts, rel_path))
    return sorted(out, key=lambda f: (f.lineno, f.quantity))


def scan_path(path: Path) -> List[Fabrication]:
    return scan_source(path.read_text(encoding="utf-8"), _rel(path))


def _report(findings: Sequence[Fabrication], header: str) -> str:
    lines = [f"{header}: {len(findings)} fabricated physics input(s)", ""]
    lines += ["  - " + f.describe() for f in findings]
    lines += [
        "",
        "  A value either comes from a source that was actually fetched, or it is "
        "absent (None).",
        "  Site-to-site variation produced by hash() or a random draw is not a "
        "measurement, and",
        "  str hashing is salted per process, so it is not even stable across a "
        "restart.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# which modules feed the physics
# --------------------------------------------------------------------------------------
def _modules_constructing(symbol: str) -> List[Path]:
    """Every module under backend/app that calls `symbol(...)`."""
    hits: List[Path] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if symbol not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (_dotted(node.func) or "").split(".")[
                -1
            ] == symbol:
                hits.append(path)
                break
    return hits


# The explicit three plus anything that builds the physics input record, so a new
# feeder module joins the scan the day it is written rather than when someone
# remembers to add it here.
_EXPLICIT_FEEDERS = [
    ENVIRONMENTAL_FEATURES_PY,
    APP_DIR / "physics" / "feature_synthesizer.py",
    APP_DIR / "engine" / "sites_catalog.py",
    APP_DIR / "ingestion" / "real_site_features.py",
    # The valley grid reached production fabricating every terrain and soil value from
    # np.random seeded on hash(corridor_id), and this scan never saw it: discovery keyed
    # on RawEarthObservationInputs, and the grid calls the vectorised FS kernel directly
    # instead. Both the module and the second entry point are named now.
    APP_DIR / "physics" / "valley_grid_physics.py",
    APP_DIR / "engine" / "valley_grid_terrain.py",
]
PHYSICS_FEED_MODULES: List[Path] = sorted(
    set(_EXPLICIT_FEEDERS)
    | set(_modules_constructing("RawEarthObservationInputs"))
    | set(_modules_constructing("compute_vectorized_grid_fs")),
    key=lambda p: str(p),
)


def _feeder_params() -> List[Any]:
    params = []
    for path in PHYSICS_FEED_MODULES:
        marks = [parked_hash] if path == ENVIRONMENTAL_FEATURES_PY else []
        params.append(pytest.param(path, marks=marks, id=_rel(path)))
    return params


# ======================================================================================
# 1. the scanner itself
# ======================================================================================
INCIDENT_SOURCE = textwrap.dedent(
    """
    import math

    def compute_site_environmental_features(site_id, lat, lon, elev, slp):
        hash_seed = abs(hash(f"{lat:.3f}_{lon:.3f}_{elev:.1f}"))
        dist_road = 18.0 + (hash_seed % 280)
        ndvi = 0.62 - 0.002 * (slp - 35.0) + 0.06 * math.sin(hash_seed % 5)
        return EnvironmentalFeatures(
            site_id=site_id,
            distance_to_road_m=round(dist_road, 1),
            sentinel2_ndvi=round(ndvi, 3),
        )
    """
)

LEGITIMATE_SOURCE = textwrap.dedent(
    """
    import hashlib
    import random
    import numpy as np

    CACHE = {}

    def cache_key(lat, lon):
        return hash((round(lat, 4), round(lon, 4)))

    def bulletin_id(site_id, issued_utc):
        return f"NDRRMA-{site_id}-{abs(hash(issued_utc)) % 10**8:08d}"

    def seen(record, registry):
        return hash(record.text) in registry

    class Site:
        def __hash__(self):
            return hash(self.site_id)

    def jitter_for_display(n):
        return [random.random() for _ in range(n)]

    def monte_carlo_fs(mean_c, sd_c, n=2000):
        rng = np.random.default_rng(42)
        samples = rng.normal(mean_c, sd_c, n)
        return float(np.mean(samples < 1.0))

    def build(raw_clay, raw_sand):
        return RawEarthObservationInputs(
            clay_fraction=raw_clay,
            sand_fraction=raw_sand,
            distance_to_road_m=CACHE[cache_key(1.0, 2.0)],
        )
    """
)


def test_scanner_catches_the_incident_pattern():
    """The guard must catch the exact code that shipped, or it guards nothing."""
    findings = scan_source(INCIDENT_SOURCE, "incident.py")
    quantities = {f.quantity for f in findings}

    assert quantities == {"distance_to_road_m", "sentinel2_ndvi"}, (
        "scanner missed the known-fabricated quantities; found "
        f"{sorted(quantities)}\n{_report(findings, 'incident.py')}"
    )
    for finding in findings:
        assert finding.origin_kind == "hash()"
        assert "hash_seed" in finding.chain
        assert finding.origin_lineno == 5


def test_scanner_does_not_ban_the_hash_builtin():
    """hash() for a cache key, an id, dedup or __hash__ is fine. Only sinks matter."""
    findings = scan_source(LEGITIMATE_SOURCE, "legitimate.py")
    assert findings == [], _report(
        findings, "false positives on legitimate hash/random use"
    )


def test_scanner_follows_a_value_through_an_alias_chain():
    """One-hop detection would have missed the real incident, which was three hops."""
    source = textwrap.dedent(
        """
        def f(lat, lon):
            seed = abs(hash(f"{lat}_{lon}"))
            a = seed % 97
            b = a * 1.5
            c = b + 12.0
            return RawEarthObservationInputs(dry_bulk_density_g_cm3=c)
        """
    )
    findings = scan_source(source, "chain.py")
    assert [f.quantity for f in findings] == ["dry_bulk_density_g_cm3"]
    assert findings[0].chain == ("seed", "a", "b", "c")


def test_scanner_catches_random_draws_and_dict_sinks():
    source = textwrap.dedent(
        """
        import numpy as np

        def f():
            rng = np.random.default_rng(0)
            return {"clay_fraction": float(rng.uniform(0.1, 0.3))}
        """
    )
    findings = scan_source(source, "rng.py")
    assert [f.quantity for f in findings] == ["clay_fraction"]
    assert findings[0].origin_kind == "random draw"


def test_physics_field_set_is_populated():
    """A silently empty sink set would make every scan vacuously pass."""
    assert len(PHYSICS_FIELDS) >= 25
    for expected in (
        "clay_fraction",
        "sentinel2_ndvi",
        "distance_to_road_m",
        "aspect_deg",
        "dry_bulk_density_g_cm3",
    ):
        assert expected in PHYSICS_FIELDS


def test_module_discovery_finds_the_known_physics_feeders():
    """Discovery is dynamic, so assert it still sees the modules we know about."""
    found = {_rel(p) for p in PHYSICS_FEED_MODULES}
    for expected in (
        "backend/app/engine/environmental_features.py",
        "backend/app/engine/perpetual_orchestrator.py",
        "backend/app/physics/feature_synthesizer.py",
        "backend/app/api/routes.py",
        "backend/app/ml/susceptibility.py",
        "backend/app/physics/valley_grid_physics.py",
    ):
        assert expected in found, f"{expected} dropped out of the scan set: {sorted(found)}"


def test_allowlisted_functions_do_not_build_physics_records():
    """An exemption may silence the scanner, but never for code feeding the physics."""
    for (rel_path, func_name), reason in _SYNTHETIC_ALLOWLIST.items():
        path = BACKEND_DIR.parent / rel_path
        assert path.exists(), f"stale allowlist entry: {rel_path}"
        assert reason.strip(), "every allowlist entry must carry a reason"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != func_name:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    callee = (_dotted(sub.func) or "").split(".")[-1]
                    assert callee not in PHYSICS_CONSTRUCTORS, (
                        f"{rel_path}::{func_name} is allowlisted from the fabrication "
                        f"scan but constructs {callee} at line {sub.lineno}. The "
                        "exemption must be removed or narrowed."
                    )


# ======================================================================================
# 2. the real modules
# ======================================================================================
@pytest.mark.parametrize("module_path", _feeder_params())
def test_no_fabricated_values_reach_the_physics(module_path: Path):
    """No hash- or random-derived value may land on a declared physical quantity."""
    findings = scan_path(module_path)
    assert not findings, _report(findings, _rel(module_path))


def test_environmental_features_fabrication_is_still_exactly_where_we_think_it_is():
    """Pin the known violation so the parked xfails above cannot drift silently.

    This is the inverse of the guard: while the fabrication is still in production, we
    assert it is still *there*. When the replacement lands this test is deleted along
    with the ENFORCE_* flags, and `test_no_fabricated_values_reach_the_physics` takes
    over permanently.
    """
    if ENFORCE_REAL_ENVIRONMENTAL_FEATURES:
        pytest.skip("the replacement has landed; this scaffolding is obsolete")

    findings = scan_path(ENVIRONMENTAL_FEATURES_PY)
    quantities = {f.quantity for f in findings}
    assert {
        "aspect_deg",
        "distance_to_road_m",
        "sentinel2_ndvi",
    } <= quantities, (
        "the known fabricated quantities moved or changed name; re-check the parked "
        f"assertions.\n{_report(findings, _rel(ENVIRONMENTAL_FEATURES_PY))}"
    )
    assert all(f.origin_kind == "hash()" for f in findings)


# ======================================================================================
# 3. behaviour: a hash is not an observation
# ======================================================================================
_CHILD_PROGRAM = textwrap.dedent(
    """
    import json, sys
    from dataclasses import asdict
    from app.engine.environmental_features import compute_site_environmental_features

    sites = json.loads(sys.argv[1])
    print(json.dumps([asdict(compute_site_environmental_features(**s)) for s in sites]))
    """
)

# Three probes spanning the elevation bands the generator branches on. None of them is
# a catalogued site id, so none picks up the hand-calibrated overrides that would mask
# the instability at, say, glc_105.
_PROBE_SITES = [
    {
        "site_id": "probe_lowland",
        "latitude": 27.6821,
        "longitude": 84.4312,
        "elevation_m": 420.0,
        "slope_deg": 28.0,
        "basin_id": "bagmati_narayani",
    },
    {
        "site_id": "probe_midhill",
        "latitude": 29.5023,
        "longitude": 82.2504,
        "elevation_m": 1860.0,
        "slope_deg": 34.0,
        "basin_id": "karnali",
    },
    {
        "site_id": "probe_alpine",
        "latitude": 28.2320,
        "longitude": 85.5180,
        "elevation_m": 4800.0,
        "slope_deg": 41.0,
        "basin_id": "langtang",
    },
]


def _child_env(hashseed: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": hashseed,
            "PYTHONPATH": str(BACKEND_DIR),
            # shared 256-core box: never let a child grab the machine
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    return env


def _features_in_subprocess(hashseed: str, sites: Sequence[Dict[str, Any]]) -> List[Dict]:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM, json.dumps(list(sites))],
        capture_output=True,
        text=True,
        env=_child_env(hashseed),
        cwd=str(BACKEND_DIR),
        timeout=180,
    )
    if proc.returncode != 0:  # pragma: no cover - surfaced as an error, not an xfail
        raise RuntimeError(
            f"feature child (PYTHONHASHSEED={hashseed}) failed:\n{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout)


def test_builtin_str_hash_is_not_stable_across_processes():
    """The premise, demonstrated rather than asserted from memory.

    `hash()` on a str is salted per interpreter unless PYTHONHASHSEED is pinned. Any
    "observation" derived from it changes every time the process restarts - so a site's
    soil texture would differ between two orchestrator runs on the same day. This test
    does not touch the project; it establishes why the ones that follow matter.
    """
    program = "import sys; print([hash(s) for s in ('glc_105', '29.502_82.250_1860.0')])"
    out = []
    for seed in ("1", "2"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, env=env, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        out.append(proc.stdout.strip())

    assert out[0] != out[1], (
        "str hashing appears unsalted in this environment, which would hide the "
        "cross-process instability the guard below is testing for"
    )


@parked_hash
def test_site_features_are_reproducible_across_processes():
    """The same site must yield the same values in a fresh interpreter.

    Physical properties of a hillslope do not depend on the interpreter's hash salt.
    """
    first = _features_in_subprocess("1", _PROBE_SITES)
    second = _features_in_subprocess("2", _PROBE_SITES)

    differing = {
        (a["site_id"], key): (a[key], b[key])
        for a, b in zip(first, second)
        for key in a
        if key not in _PASSTHROUGH_FIELDS and a[key] != b[key]
    }
    assert not differing, (
        f"{_rel(ENVIRONMENTAL_FEATURES_PY)} produced different 'measurements' in two "
        "processes that differ only in PYTHONHASHSEED:\n"
        + "\n".join(
            f"        {site}.{k}: {a!r} (seed 1) vs {b!r} (seed 2)"
            for (site, k), (a, b) in sorted(differing.items())
        )
        + "\n        A restart of the orchestrator therefore changes the physics inputs "
        "under every site."
    )


@parked_hash
def test_site_values_do_not_jump_at_the_hash_key_rounding_boundary():
    """Variability must not be concentrated at a decimal-rounding boundary.

    The fabricated generator keys on ``f"{lat:.3f}_{lon:.3f}_{elev:.1f}"``, so every
    coordinate inside one 0.001-degree cell (~110 m) gets byte-identical "measurements",
    and two points 2 m apart on opposite sides of the boundary get unrelated ones. Real
    data is the other way round, whatever its resolution: two points 2 m apart are at
    least as similar as two points 40 m apart.

    This is a signature no physical field can produce, and it does not depend on knowing
    how the values were made.
    """
    lon, elev, slope = 82.2504, 1860.0, 34.0

    def at(lat: float) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(
            compute_site_environmental_features(
                site_id="probe",  # not a catalogued site: no hand-calibrated override
                latitude=lat,
                longitude=lon,
                elevation_m=elev,
                slope_deg=slope,
                basin_id="karnali",
            )
        )

    # 39 m apart, both inside the cell that rounds to 28.210
    inside_a, inside_b = at(28.21005), at(28.21040)
    # 2.2 m apart, straddling the 28.210 / 28.211 boundary
    across_a, across_b = at(28.21049), at(28.21051)

    compared = sorted(
        set(inside_a)
        - _PASSTHROUGH_FIELDS
        - {"anchor_lat", "anchor_lon", "anchor_name", "road_type"}
    )

    def moved(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        return {k: (x[k], y[k]) for k in compared if x[k] != y[k]}

    within_cell = moved(inside_a, inside_b)
    across_cell = moved(across_a, across_b)

    assert not (across_cell and not within_cell), (
        f"{_rel(ENVIRONMENTAL_FEATURES_PY)} is quantised to the 0.001-degree grid of "
        "its hash key, not to anything physical:\n"
        f"        39 m apart, same grid cell : {len(within_cell)} of "
        f"{len(compared)} values differ (identical slopes)\n"
        f"        2.2 m apart, across the cell boundary: "
        f"{len(across_cell)} of {len(compared)} values differ\n"
        + "\n".join(
            f"        {k}: {a!r} -> {b!r} over 2.2 m"
            for k, (a, b) in sorted(across_cell.items())
        )
    )


# ======================================================================================
# 4. provenance
# ======================================================================================
PROVENANCE_FIELDS = ("value", "source", "url", "method", "fetched_utc")


def test_measured_record_carries_full_provenance():
    """The contract a measured value must satisfy before the physics may use it."""
    names = {f.name for f in dataclass_fields(rsf.Measured)}
    for field in PROVENANCE_FIELDS:
        assert field in names, f"Measured lost its '{field}' field"
    assert "raw" in names, "Measured must keep the pre-conversion response value"

    m = rsf.Measured(
        value=0.252,
        source="SoilGrids v2.0 (ISRIC)",
        url="https://rest.isric.org/soilgrids/v2.0/properties/query?lon=82.25",
        method="0-5cm mean, raw 252 g/kg / d_factor 10 -> %",
        fetched_utc=1.0,
        raw=252,
    )
    with pytest.raises(FrozenInstanceError):
        m.value = 0.9  # type: ignore[misc]

    round_tripped = rsf.Measured(**json.loads(json.dumps(m.to_dict())))
    assert round_tripped == m, "provenance must survive the cache round-trip"


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


_SOILGRIDS_PAYLOAD = {
    "properties": {
        "layers": [
            {
                "name": "clay",
                "unit_measure": {
                    "d_factor": 10,
                    "mapped_units": "g/kg",
                    "target_units": "%",
                },
                "depths": [{"label": "0-5cm", "values": {"mean": 252}}],
            },
            {
                "name": "sand",
                "unit_measure": {
                    "d_factor": 10,
                    "mapped_units": "g/kg",
                    "target_units": "%",
                },
                "depths": [{"label": "0-5cm", "values": {"mean": 400}}],
            },
            {
                "name": "cfvo",
                "unit_measure": {
                    "d_factor": 10,
                    "mapped_units": "cm3/dm3",
                    "target_units": "cm3/100cm3",
                },
                "depths": [{"label": "0-5cm", "values": {"mean": 144}}],
            },
            {
                "name": "bdod",
                "unit_measure": {
                    "d_factor": 100,
                    "mapped_units": "cg/cm3",
                    "target_units": "kg/dm3",
                },
                "depths": [{"label": "0-5cm", "values": {"mean": 111}}],
            },
        ]
    }
}

_OVERPASS_PAYLOAD = {
    "elements": [
        {
            "type": "way",
            "tags": {"highway": "trunk"},
            "geometry": [
                {"lat": 29.5026, "lon": 82.2504},
                {"lat": 29.5030, "lon": 82.2510},
            ],
        }
    ]
}


@pytest.fixture
def offline_sources(monkeypatch, tmp_path):
    """Canned responses for the real fetchers. Nothing here touches the network."""
    monkeypatch.setattr(rsf, "CACHE_PATH", tmp_path / "real_site_features.json")
    monkeypatch.setattr(
        rsf, "_get", lambda url, timeout: json.dumps(_SOILGRIDS_PAYLOAD).encode("utf-8")
    )
    monkeypatch.setattr(
        rsf.urllib.request,
        "urlopen",
        lambda request, timeout=None: _FakeResponse(_OVERPASS_PAYLOAD),
    )
    from app.ingestion import insar_timeseries

    monkeypatch.setattr(insar_timeseries, "estimate_aspect_deg", lambda lat, lon: 212.4)
    return tmp_path


def test_fetched_values_carry_provenance_through_the_cache(offline_sources):
    """Every value handed to a caller must name the source it came from."""
    features = rsf.fetch_site_features("glc_105", 29.5023, 82.2504)

    assert set(features) >= {
        "clay_fraction",
        "sand_fraction",
        "coarse_rock_fragment_fraction",
        "dry_bulk_density_g_cm3",
        "road_distance_m",
        "aspect_deg",
    }
    for name, measured in features.items():
        assert isinstance(measured, rsf.Measured), f"{name} is not a Measured record"
        assert measured.source.strip(), f"{name} has no source attribution"
        assert measured.url.startswith("http"), f"{name} has no auditable URL"
        assert measured.method.strip(), f"{name} does not say how it was derived"
        assert measured.fetched_utc > 0, f"{name} has no fetch timestamp"

    # Unit conversion comes from the response, not from an assumption.
    assert features["clay_fraction"].value == pytest.approx(0.252)
    assert features["dry_bulk_density_g_cm3"].value == pytest.approx(1.11)
    assert features["clay_fraction"].raw == 252

    cached = json.loads((offline_sources / "real_site_features.json").read_text())
    entry = cached["29.5023,82.2504"]["features"]
    for name, payload in entry.items():
        assert payload["source"] and payload["url"], (
            f"the cache dropped provenance for {name}; a cached number with no source "
            "is indistinguishable from a fabricated one"
        )
    reloaded = rsf.fetch_site_features("glc_105", 29.5023, 82.2504)
    assert reloaded == features, "provenance must survive a cache reload unchanged"


def test_a_failed_fetch_is_absent_not_defaulted(monkeypatch, tmp_path):
    """Absent is a correct outcome. A plausible default is not."""
    monkeypatch.setattr(rsf, "CACHE_PATH", tmp_path / "real_site_features.json")
    monkeypatch.setattr(rsf, "_get", lambda url, timeout: None)

    def _refuse(*args: Any, **kwargs: Any):
        raise OSError("network unavailable")

    monkeypatch.setattr(rsf.urllib.request, "urlopen", _refuse)
    from app.ingestion import insar_timeseries

    monkeypatch.setattr(insar_timeseries, "estimate_aspect_deg", lambda lat, lon: None)

    assert rsf.fetch_soil(29.5023, 82.2504) is None
    assert rsf.fetch_road_proximity(29.5023, 82.2504) is None
    assert rsf.fetch_aspect(29.5023, 82.2504) is None

    features = rsf.fetch_site_features("glc_105", 29.5023, 82.2504)
    assert features == {}, (
        "a failed fetch invented values instead of reporting absence: "
        f"{sorted(features)}"
    )


def test_an_absent_road_is_distinguishable_from_a_failed_query(monkeypatch, tmp_path):
    """'No road within 2 km' and 'Overpass timed out' must never collapse into one."""
    monkeypatch.setattr(rsf, "CACHE_PATH", tmp_path / "real_site_features.json")
    monkeypatch.setattr(
        rsf.urllib.request,
        "urlopen",
        lambda request, timeout=None: _FakeResponse({"elements": []}),
    )
    empty = rsf.fetch_road_proximity(29.5023, 82.2504)
    assert empty is not None and "road_distance_m" in empty
    assert empty["road_distance_m"].detail and "genuine absence" in (
        empty["road_distance_m"].detail
    )


@parked_hash
def test_values_consumed_by_the_physics_carry_a_source():
    """Whatever record the orchestrator reads must attribute each measured quantity.

    The contract: for every measured field on the record consumed at
    ``perpetual_orchestrator.py`` (``self.enriched_env``), there must be a companion
    provenance entry - either the field is a :class:`Measured`, or the record exposes
    ``provenance[field]`` / ``<field>_source``. Unattributed numbers are exactly what
    both incidents looked like from the consumer's side.
    """
    names = {f.name for f in dataclass_fields(EnvironmentalFeatures)}
    measured_fields = sorted(names - _PASSTHROUGH_FIELDS)
    assert measured_fields, "no measured fields found on the consumed record"

    annotations = {f.name: str(f.type) for f in dataclass_fields(EnvironmentalFeatures)}
    unattributed = [
        field
        for field in measured_fields
        if "Measured" not in annotations.get(field, "")
        and f"{field}_source" not in names
        and "provenance" not in names
    ]
    assert not unattributed, (
        f"{len(unattributed)} of {len(measured_fields)} quantities on "
        "EnvironmentalFeatures reach the physics with no source attribution:\n"
        f"        {', '.join(unattributed)}\n"
        "        Each must carry its source (a Measured record, a provenance mapping, "
        "or a <field>_source field)."
    )


# ======================================================================================
# 5. absent, never assumed
# ======================================================================================
@dataclass(frozen=True)
class _FrozenFallback:
    rel_path: str
    lineno: int
    quantity: str
    literal: Any
    src: str


def _frozen_fallbacks(path: Path) -> List[_FrozenFallback]:
    """`clay_frac = env.clay_fraction if env else 0.16` and friends.

    The pattern substitutes a nationwide constant for an absent per-site measurement,
    which is how 11 of 22 physics inputs came to be identical at all 491 sites
    (PHYSICS_SENSITIVITY_FINDINGS.md section 3). Only this one precise shape is matched:
    a conditional reading an attribute named like a physics input, whose else-branch is
    a bare numeric literal.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: List[_FrozenFallback] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        if not isinstance(node.body, ast.Attribute):
            continue
        if node.body.attr not in PHYSICS_FIELDS:
            continue
        if not isinstance(node.orelse, ast.Constant):
            continue
        if not isinstance(node.orelse.value, (int, float)) or isinstance(
            node.orelse.value, bool
        ):
            continue
        out.append(
            _FrozenFallback(
                rel_path=_rel(path),
                lineno=node.lineno,
                quantity=node.body.attr,
                literal=node.orelse.value,
                src=_src(node),
            )
        )
    return sorted(out, key=lambda f: (f.lineno, f.quantity))


@parked_defaults
@pytest.mark.parametrize(
    "rel_path",
    ["backend/app/engine/perpetual_orchestrator.py", "backend/app/api/routes.py"],
)
def test_absent_measurements_are_not_replaced_by_frozen_constants(rel_path: str):
    """An unmeasured site must be handled openly, not handed a nationwide constant."""
    findings = _frozen_fallbacks(BACKEND_DIR.parent / rel_path)
    assert not findings, (
        f"{rel_path}: {len(findings)} physics input(s) silently fall back to a frozen "
        "nationwide constant when the per-site measurement is absent:\n"
        + "\n".join(
            f"  - {f.rel_path}:{f.lineno}: {f.quantity} -> {f.literal!r}  ({f.src})"
            for f in findings
        )
        + "\n  The same value at all 491 sites is not a measurement. Prefer propagating "
        "the absence\n  so the caller can see which sites are unmeasured."
    )


# ======================================================================================
# 5. the valley grid regression
#
# The valley grid shipped to production with every terrain and soil value drawn from
# np.random.default_rng(abs(hash(corridor_id)) % 2**31), and this scanner passed the
# whole time: discovery keyed on modules constructing RawEarthObservationInputs, and the
# grid calls compute_vectorized_grid_fs directly instead, so it was never scanned.
#
# Two tests below. The first proves the scanner CAN see the pattern - a guard that
# passes because it is blind is worse than no guard. The second proves the real module
# is clean.
# ======================================================================================
VALLEY_GRID_PY = APP_DIR / "physics" / "valley_grid_physics.py"

_THE_INCIDENT = textwrap.dedent(
    """
    import numpy as np
    def _generate_corridor_cells(self, corridor_id):
        rng = np.random.default_rng(abs(hash(corridor_id)) % (2**31))
        c_slope = rng.normal(44.0, 4.2)
        c_clay = np.clip(0.16 - 0.04 * (c_slope / 45.0), 0.08, 0.22)
        c_density = 1.55 + 0.25 * c_clay
        return {
            "slope_deg": np.array([c_slope]),
            "clay_fraction": np.array([c_clay]),
            "dry_bulk_density_g_cm3": np.array([c_density]),
        }
    """
)


def test_scanner_sees_the_valley_grid_pattern():
    """The exact shape that shipped must be detectable, chain and all."""
    findings = scan_source(_THE_INCIDENT, _rel(VALLEY_GRID_PY))
    quantities = {f.quantity for f in findings}
    assert {"slope_deg", "clay_fraction", "dry_bulk_density_g_cm3"} <= quantities, (
        f"scanner missed the valley-grid fabrication pattern; saw {quantities}"
    )
    # The derived quantities must be traced back through the chain to the draw itself,
    # not merely flagged where they are assigned.
    density = next(f for f in findings if f.quantity == "dry_bulk_density_g_cm3")
    assert density.origin_kind == "random draw"
    assert density.chain == ("c_slope", "c_clay", "c_density")


def test_valley_grid_terrain_is_measured_not_generated():
    """No physics input in the live valley grid may originate in a draw or a hash."""
    assert VALLEY_GRID_PY.exists()
    findings = scan_path(VALLEY_GRID_PY)
    assert not findings, _report(findings, _rel(VALLEY_GRID_PY))


def test_valley_grid_does_not_generate_cells_at_runtime():
    """The grid is loaded from a measured cache, or it fails - it is never synthesised.

    The old loader silently produced a full grid whenever it was called. The replacement
    must raise when the cache is absent, so a missing grid surfaces as an outage rather
    than as a plausible hazard surface.
    """
    import ast

    tree = ast.parse(VALLEY_GRID_PY.read_text(encoding="utf-8"))
    loader = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_load_corridor_cells"),
        None,
    )
    assert loader is not None, "_load_corridor_cells disappeared from the valley grid"
    raises = [n for n in ast.walk(loader) if isinstance(n, ast.Raise)]
    assert raises, "_load_corridor_cells must raise when the measured grid is missing"

    banned = {"default_rng", "RandomState", "uniform", "normal", "random", "seed"}
    for node in ast.walk(loader):
        if isinstance(node, ast.Call):
            callee = (_dotted(node.func) or "").split(".")[-1]
            assert callee not in banned, (
                f"_load_corridor_cells calls {callee}() at line {node.lineno}; the "
                "valley grid must be read from measurements, never drawn"
            )
