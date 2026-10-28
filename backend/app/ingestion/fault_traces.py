"""Active fault and structural traces for Nepal, merged from sources with provenance.

Why this module exists
----------------------
The engine reduces effective cohesion near active faults::

    valley grid : fault_integrity = 0.68 + 0.32 * clip(d_km / 15, 0, 1)
    site level  : fault_integrity = 1.00 - 0.40 * exp(-d_km / 2.5)

Both consume a single number, ``distance to the nearest mapped trace``. Measured
against the real layer in production, that number is not describing what the code
says it describes.

MEASURED 2026-10-28, against GEM harmonized (the file production actually uses, per
AGENTS.md) and the 590,611-cell valley grid:

    corridor                   cells   median d_any   median d_MFT   nearest structure
    langtang                  36,279        40.4 km       113.8 km   Normal 100%
    kali_gandaki              68,998        20.5 km       105.8 km   Normal 95%
    bagmati_narayani          46,236        30.6 km        49.9 km   Anticline 55%, Syncline 32%
    budhi_gandaki             57,326        52.5 km       117.3 km   Normal 76%
    karnali                  198,554        30.2 km       114.4 km   Dextral 91%
    sapta_koshi_kanchenjunga 106,923        50.1 km        88.9 km   Normal 65%, MFT 26%
    sun_koshi                 76,295        61.0 km        89.6 km   Normal 66%, MFT 28%

Two separate problems are visible there, and only the first was previously recorded:

1. **The term is saturated.** 87% of cells sit at or past the 15 km cutoff, so the
   fault-damage factor contributes exactly nothing for them. At site level the same
   holds: 78% of the 491 sites have an integrity factor above 0.999.

2. **It is keyed to structures the comment does not describe.** ``valley_grid_physics``
   comments the term as "damaged within 15 km of major thrust faults", but for Langtang
   the nearest mapped structure is a Tibetan normal-fault rift 100% of the time, and for
   Bagmati-Narayani it is a FOLD AXIS 87% of the time. Across the 491 production sites,
   161 - a third of them - have a fold hinge as their nearest "active fault". The number
   being fed in is "distance to the nearest line in the file", not "distance to the
   nearest seismogenic thrust".

   This is NOT automatically a bug to be fixed by deleting the fold axes. In the
   Himalayan foreland an active anticline overlies a blind thrust, so a Siwalik fold
   hinge can be a legitimate proxy for a buried seismogenic structure. The measured
   consequence of excluding them is recorded at ``SEISMOGENIC_CLASSES`` below, and it
   points the dangerous way: seven sites move toward GREEN and none toward alarm.

This module exists to fix (2) and to make (1) honest, by emitting a layer in which
every trace carries a ``structure_class`` and its own provenance, so a consumer can ask
for the distance to the thrust system specifically rather than to whatever line happens
to be closest.

What the sources actually contain - verified, not assumed
----------------------------------------------------------
``GEM Global Active Faults (harmonized)`` and ``HimaTibetMap`` are **not independent**.
Every one of the 57 GEM features inside 80-88.5E / 26-31N carries
``catalog_name = "HimaTibetMap"``: Styron harmonized Taylor & Yin's compilation into GEM.
Swapping one for the other is a no-op, and this module does not pretend otherwise.

What HimaTibetMap adds over the harmonized copy is **attributes GEM dropped**: fault
names, the originating reference, and net slip rates. Nine Main Frontal Thrust segments
carry ``net_slip_r = (15,10,21)`` mm/yr and the Karakoram Fault ``(6,1,11)``. A trace
slipping 15 mm/yr and a dormant one damage the surrounding rock mass differently, and
that distinction was simply unavailable before.

What NEITHER source contains, checked by exhaustive string search over all 644 KB of
HimaTibetMap and all 57 GEM Nepal features: the **Main Boundary Thrust** and the
**Main Central Thrust**. Zero matches for "Main Boundary", "Main Central", "MBT", "MCT".
Those two run through the Lesser and Higher Himalaya - which is exactly where the
monitored corridors sit - so their absence is the reason the thrust-distance column is
~90-120 km for every corridor. The Main Frontal Thrust we do have hugs the southern
Siwalik front and is nowhere near the high-mountain cells.

Adding the MFT therefore does NOT revive the term for the corridors, and this was
measured rather than assumed: median distance to the MFT is 88.9-117.3 km across all
seven, every one of which is past the 15 km cutoff by a factor of six or more.

Safety direction - and the one place it reverses
------------------------------------------------
ADDING traces can only ever *reduce* the distance to the nearest trace, which reduces
``fault_integrity``, which reduces ``c_prime``, which reduces FS. So enriching this layer
makes the system more cautious, never less - the same direction the DSGSD factor moves
in, and the only direction an early-warning system may move unprompted.
``backend/tests/test_fault_traces.py`` pins that.

RECLASSIFYING reverses it. Narrowing the layer to a subset of structure classes
*lengthens* the nearest-trace distance, which raises FS and lowers alarm. That is why
this module changes no production input on its own: it emits a richer, classified layer
and leaves the consumption decision explicit. Nothing here silently quiets the system.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "fault_sources"

#: Where the merged layer is written.
MERGED_PATH = PROCESSED_DIR / "nepal_fault_traces.geojson"

#: Nepal plus a working margin, so a trace just outside the border still contributes a
#: correct distance to a cell just inside it.
NEPAL_WINDOW = (79.0, 89.5, 25.0, 32.0)  # W, E, S, N

# ---------------------------------------------------------------------------
# Structure classification.
#
# The point of this table is that the physics asks for a *thrust* distance. Grouping a
# syncline hinge and the Main Frontal Thrust under one "fault distance" number is what
# produced the Bagmati-Narayani result above.
# ---------------------------------------------------------------------------

THRUST_SYSTEM = "THRUST_SYSTEM"
EXTENSIONAL = "EXTENSIONAL"
STRIKE_SLIP = "STRIKE_SLIP"
FOLD_AXIS = "FOLD_AXIS"
SUTURE = "SUTURE"
UNCLASSIFIED = "UNCLASSIFIED"

#: Classes treated as seismogenic rupture sources.
#:
#: READ THE CAVEAT BEFORE USING THIS. Excluding ``FOLD_AXIS`` is defensible in general -
#: a syncline hinge is not an earthquake source - but it is NOT safely true in the
#: Himalayan foreland, where an active anticline is typically the surface expression of
#: a BLIND THRUST at depth. Dropping Siwalik fold axes therefore risks discarding real
#: hazard, not just noise.
#:
#: MEASURED consequence of switching the 491-site engine to this subset, 2026-10-28:
#: 7 sites move at 120 mm rain and ALL SEVEN move toward GREEN, including glc_163
#: crossing RED -> ORANGE (FS 0.988 -> 1.017) and langtang_lirung YELLOW -> GREEN.
#: Zero move toward more alarm.
#:
#: Because of that, nothing in production consumes this tuple yet. It exists so the
#: choice can be made explicitly, with those numbers on the table, rather than inherited
#: from whichever line happened to be nearest.
SEISMOGENIC_CLASSES = (THRUST_SYSTEM, EXTENSIONAL, STRIKE_SLIP)

_CLASS_BY_TYPE: Dict[str, str] = {
    "thrust": THRUST_SYSTEM,
    "reverse": THRUST_SYSTEM,
    "thrust_inferred": THRUST_SYSTEM,
    "normal": EXTENSIONAL,
    "detachment": EXTENSIONAL,
    "normal_approximate": EXTENSIONAL,
    "right-slip": STRIKE_SLIP,
    "rightslip": STRIKE_SLIP,
    "left-slip": STRIKE_SLIP,
    "leftslip": STRIKE_SLIP,
    "strike-slip": STRIKE_SLIP,
    "dextral": STRIKE_SLIP,
    "sinistral": STRIKE_SLIP,
    "anticline": FOLD_AXIS,
    "syncline": FOLD_AXIS,
    "snycline": FOLD_AXIS,  # the upstream file's own typo, preserved deliberately
    "suture": SUTURE,
}


def classify_structure(raw_type: Optional[str], name: Optional[str] = None) -> str:
    """Map a source's own type string onto a structure class.

    An unrecognised type becomes ``UNCLASSIFIED`` rather than being guessed into a
    class. A name is consulted only to catch the named Himalayan thrusts, whose type
    field is occasionally blank in the upstream data.
    """
    t = (raw_type or "").strip().lower()
    if t in _CLASS_BY_TYPE:
        return _CLASS_BY_TYPE[t]

    n = (name or "").strip().lower()
    if n:
        for marker in ("main frontal thrust", "main boundary thrust",
                       "main central thrust", "main himalayan thrust", "thrust"):
            if marker in n:
                return THRUST_SYSTEM
    return UNCLASSIFIED


# ---------------------------------------------------------------------------
# Slip rates.
# ---------------------------------------------------------------------------

_SLIP_RE = re.compile(r"\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?")


def parse_slip_rate(raw: Any) -> Optional[Dict[str, float]]:
    """Parse HimaTibetMap's ``(best, min, max)`` slip-rate triple, in mm/yr.

    Returns ``None`` for anything that is not a well-formed triple, including the
    single values and free text that appear in some rows. A rate that cannot be parsed
    is absent, never defaulted - a fault with no published rate is not a fault that
    creeps at zero.
    """
    if raw in (None, "", "None"):
        return None
    m = _SLIP_RE.search(str(raw))
    if not m:
        return None
    try:
        best, lo, hi = (float(m.group(i)) for i in (1, 2, 3))
    except ValueError:
        return None
    if lo > hi:
        lo, hi = hi, lo
    return {"best_mm_yr": best, "min_mm_yr": lo, "max_mm_yr": hi}


# ---------------------------------------------------------------------------
# Sources.
# ---------------------------------------------------------------------------


class Source:
    """One upstream fault dataset, with the fields needed to attribute every trace.

    ``kind`` is ``"geojson"`` for a plain FeatureCollection, or ``"shapefile_zip"`` for a
    zipped set of ESRI shapefiles read through GDAL's ``/vsizip/``.

    ``layer_identity`` matters for the shapefile case and is deliberately separate from
    the attribute fields. Some published compilations name a fault only by the FILE it
    lives in - there is no name column to read. Mapping a filename to a fault is an act
    of interpretation, not a measurement, so it is declared here and recorded on every
    feature as ``identity_from`` rather than being passed off as source data.
    """

    def __init__(self, key: str, url: str, licence: str, note: str,
                 name_field: Optional[str] = None, type_field: Optional[str] = None,
                 slip_field: Optional[str] = None,
                 reference_field: Optional[str] = None,
                 kind: str = "geojson",
                 layer_identity: Optional[Dict[str, Tuple[str, str]]] = None) -> None:
        self.key = key
        self.url = url
        self.licence = licence
        self.note = note
        self.name_field = name_field
        self.type_field = type_field
        self.slip_field = slip_field
        self.reference_field = reference_field
        self.kind = kind
        self.layer_identity = layer_identity or {}


SOURCES: Dict[str, Source] = {
    "himatibetmap": Source(
        key="himatibetmap",
        url="https://raw.githubusercontent.com/HimaTibetMap/HimaTibetMap/master/geojson/HimaTibetMap.geojson",
        licence="licence not stated in repository; cite Taylor & Yin 2009, doi:10.1130/GES00217.1",
        note=("940 traces across the Himalaya and Tibet. Carries fault NAMES, the "
              "originating REFERENCE and NET SLIP RATES, all three of which the GEM "
              "harmonisation drops. Contains the Main Frontal Thrust; contains neither "
              "the Main Boundary Thrust nor the Main Central Thrust."),
        name_field="Name", type_field="Type",
        slip_field="net_slip_r", reference_field="Source",
    ),
    "himatibetmap_sutures": Source(
        key="himatibetmap_sutures",
        url="https://raw.githubusercontent.com/HimaTibetMap/HimaTibetMap/master/geojson/sutures.geojson",
        licence="licence not stated in repository; cite Taylor & Yin 2009, doi:10.1130/GES00217.1",
        note="Suture zones. Structural boundaries, not active seismogenic sources.",
        name_field="Name", type_field="Type", reference_field="Source",
    ),
    "whipple_himalaya": Source(
        key="whipple_himalaya",
        kind="shapefile_zip",
        url="https://download.pangaea.de/dataset/969013/files/Geology_shapefiles.zip",
        licence="CC-BY-4.0. Cite Whipple, Adams, Forte & Hodges (2024), "
                "doi:10.1594/PANGAEA.969013; companion paper doi:10.1086/731260",
        note=("The only located source carrying the Main Boundary Thrust and Main "
              "Central Thrust as line geometry. IMPORTANT: these shapefiles have NO "
              "name attribute - the only columns are Id, ltype (a cartographic "
              "line-type code) and Shape_Leng. A fault's identity comes from the "
              "FILENAME, so every feature is marked identity_from='source filename'."),
        layer_identity={
            # stem -> (fault name, structure class). Declared, not read from the file.
            "MBT": ("Main Boundary Thrust", THRUST_SYSTEM),
            "MCT": ("Main Central Thrust", THRUST_SYSTEM),
            "MFT": ("Main Frontal Thrust", THRUST_SYSTEM),
            "MT": ("Munsiari Thrust", THRUST_SYSTEM),
            "STD": ("South Tibetan Detachment", EXTENSIONAL),
        },
    ),
    "gem": Source(
        key="gem",
        url="https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson",
        licence="CC-BY-SA 4.0",
        note=("The layer production already uses. VERIFIED 2026-10-28: all 57 features "
              "inside the Nepal window carry catalog_name='HimaTibetMap', so within "
              "Nepal this source is a strict subset of himatibetmap with fewer "
              "attributes. Kept in the registry because it is global and may contribute "
              "outside the Himalayan compilation's footprint."),
        name_field="fs_name", type_field="slip_type", reference_field="reference",
    ),
}


def cache_path(key: str) -> Path:
    suffix = "zip" if SOURCES[key].kind == "shapefile_zip" else "geojson"
    return CACHE_DIR / f"{key}.{suffix}"


def fetch_source(key: str, refresh: bool = False, timeout: int = 180) -> Path:
    """Download a source to the cache. Returns the local path.

    Raises rather than returning a partial file: a truncated GeoJSON that still parses
    would silently shorten a fault trace, which moves distances the wrong way.
    """
    src = SOURCES[key]
    path = cache_path(key)
    if path.exists() and not refresh:
        return path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    req = urllib.request.Request(src.url, headers={"User-Agent": "Himalayan-MHEWS/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        payload = resp.read()
    if src.kind == "geojson":
        json.loads(payload.decode("utf-8"))  # parse before committing to the cache
    elif not payload.startswith(b"PK"):
        raise ValueError(f"{key}: expected a zip archive, got {payload[:16]!r}")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Geometry helpers. Deliberately stdlib-only so the merge runs anywhere.
# ---------------------------------------------------------------------------


def _line_parts(geom: Optional[Dict[str, Any]]) -> List[List[Sequence]]:
    if not geom:
        return []
    t = geom.get("type")
    if t == "LineString":
        return [geom.get("coordinates") or []]
    if t == "MultiLineString":
        return list(geom.get("coordinates") or [])
    return []


def _in_window(geom: Optional[Dict[str, Any]], window: Tuple[float, float, float, float]) -> bool:
    w, e, s, n = window
    for part in _line_parts(geom):
        for p in part:
            if len(p) >= 2 and w <= p[0] <= e and s <= p[1] <= n:
                return True
    return False


def _signature(geom: Optional[Dict[str, Any]]) -> Optional[str]:
    """A coordinate fingerprint used to detect the same trace arriving twice.

    Rounded to ~1 m. Two sources that digitised the same line independently will NOT
    collide here, which is correct: this catches genuine republication (GEM carrying
    HimaTibetMap's own geometry), not independent mappings of the same fault.
    """
    parts = _line_parts(geom)
    if not parts:
        return None
    flat = [f"{p[0]:.5f},{p[1]:.5f}" for part in parts for p in part if len(p) >= 2]
    if not flat:
        return None
    return f"{len(flat)}|{flat[0]}|{flat[len(flat) // 2]}|{flat[-1]}"



def _load_shapefile_zip(src: Source, window: Tuple[float, float, float, float],
                        refresh: bool) -> List[Dict[str, Any]]:
    """Read the line layers of a zipped shapefile set through GDAL's /vsizip/.

    Only layers named in ``src.layer_identity`` are taken. A layer this module cannot
    name is skipped rather than imported as an anonymous "fault": an unnamed trace that
    silently shortens somebody's distance-to-fault is exactly the kind of unattributable
    input this project refuses.
    """
    import zipfile

    import pyogrio

    path = fetch_source(src.key, refresh=refresh)
    # Enumerate the archive directly. list_layers() on the zip root fails when the
    # shapefiles sit in a subdirectory, which is how this archive is laid out.
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".shp")]

    out: List[Dict[str, Any]] = []
    for member in sorted(members):
        stem = member.rsplit("/", 1)[-1][:-4]
        if stem not in src.layer_identity:
            continue
        fault_name, structure_class = src.layer_identity[stem]
        gdf = pyogrio.read_dataframe(f"/vsizip/{path}/{member}")
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            gj = geom.__geo_interface__
            if not _line_parts(gj) or not _in_window(gj, window):
                continue
            out.append({
                "type": "Feature",
                "geometry": json.loads(json.dumps(gj)),
                "properties": {
                    "fault_name": fault_name,
                    "structure_class": structure_class,
                    "source_type": stem,
                    "slip_rate": None,
                    "reference": "Whipple, Adams, Forte & Hodges (2024)",
                    "source": src.key,
                    "source_url": src.url,
                    "licence": src.licence,
                    "identity_from": (
                        "source filename - these shapefiles carry no name attribute, "
                        "so the fault's identity is an interpretation of which file it "
                        "was published in, not a value read from the data"
                    ),
                },
            })
    return out


def load_source_features(key: str, window: Tuple[float, float, float, float] = NEPAL_WINDOW,
                         refresh: bool = False) -> List[Dict[str, Any]]:
    """Fetch a source and return its line features inside the window, normalised."""
    src = SOURCES[key]
    if src.kind == "shapefile_zip":
        return _load_shapefile_zip(src, window, refresh)
    path = fetch_source(key, refresh=refresh)
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for feat in raw.get("features") or []:
        geom = feat.get("geometry")
        if not _line_parts(geom) or not _in_window(geom, window):
            continue
        props = feat.get("properties") or {}
        name = props.get(src.name_field)
        name = str(name).strip() if name not in (None, "") else None
        slip = parse_slip_rate(props.get(src.slip_field)) if src.slip_field else None
        reference = props.get(src.reference_field) if src.reference_field else None
        out.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "fault_name": name,
                "structure_class": classify_structure(props.get(src.type_field), name),
                "source_type": props.get(src.type_field),
                "slip_rate": slip,
                "reference": str(reference).strip() if reference else None,
                "source": src.key,
                "source_url": src.url,
                "licence": src.licence,
            },
        })
    return out


def build_nepal_fault_layer(
    keys: Iterable[str] = ("himatibetmap", "gem", "himatibetmap_sutures"),
    window: Tuple[float, float, float, float] = NEPAL_WINDOW,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Merge the named sources into one layer, de-duplicating republished geometry.

    Order matters: the first source to contribute a given geometry wins, so richer
    sources should be listed first. ``himatibetmap`` precedes ``gem`` because GEM's
    Nepal content is that same compilation with the names, references and slip rates
    stripped out.
    """
    features: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    duplicates: Dict[str, int] = {}

    for key in keys:
        for feat in load_source_features(key, window=window, refresh=refresh):
            sig = _signature(feat["geometry"])
            if sig is not None and sig in seen:
                duplicates[key] = duplicates.get(key, 0) + 1
                continue
            if sig is not None:
                seen[sig] = key
            features.append(feat)

    #: Faults whose presence is load-bearing for the Himalayan thrust-distance term.
    #: Checked against what actually landed in the layer rather than asserted.
    EXPECTED = {
        "Main Frontal Thrust": "MFT",
        "Main Boundary Thrust": "MBT",
        "Main Central Thrust": "MCT",
    }

    classes: Dict[str, int] = {}
    named: Dict[str, int] = {}
    with_slip = 0
    for f in features:
        p = f["properties"]
        classes[p["structure_class"]] = classes.get(p["structure_class"], 0) + 1
        if p.get("fault_name"):
            named[p["fault_name"]] = named.get(p["fault_name"], 0) + 1
        if p.get("slip_rate"):
            with_slip += 1

    expected = {label: named.get(full, 0) for full, label in EXPECTED.items()}
    absent = [f"{full} ({label})" for full, label in EXPECTED.items()
              if named.get(full, 0) == 0]

    return {
        "type": "FeatureCollection",
        "properties": {
            "built_by": "backend/app/ingestion/fault_traces.py",
            "window": {"west": window[0], "east": window[1],
                       "south": window[2], "north": window[3]},
            "sources": [
                {"key": k, "url": SOURCES[k].url, "licence": SOURCES[k].licence,
                 "note": SOURCES[k].note}
                for k in keys
            ],
            "feature_count": len(features),
            "by_structure_class": classes,
            "named_faults": named,
            "with_slip_rate": with_slip,
            "duplicates_dropped": duplicates,
            "expected_faults": {k: v for k, v in expected.items()},
            "known_absent": absent,
            "absence_note": (
                "Faults listed in known_absent are not present in any source merged "
                "here. This block is DERIVED from the built layer on every run, never "
                "hardcoded, so it cannot go stale when a source is added or removed."
            ),
        },
        "features": features,
    }


def write_nepal_fault_layer(path: Path = MERGED_PATH, **kwargs: Any) -> Dict[str, Any]:
    """Build the merged layer and write it. Returns its ``properties`` block."""
    layer = build_nepal_fault_layer(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(layer), encoding="utf-8")
    tmp.replace(path)
    return layer["properties"]


def filter_by_class(layer: Dict[str, Any], classes: Iterable[str]) -> Dict[str, Any]:
    """Subset a built layer to the given structure classes.

    This is what lets a consumer ask for the distance to the thrust system rather than
    to the nearest line of any kind.
    """
    wanted = set(classes)
    feats = [f for f in layer.get("features", [])
             if f["properties"].get("structure_class") in wanted]
    return {"type": "FeatureCollection",
            "properties": {**layer.get("properties", {}),
                           "filtered_to": sorted(wanted),
                           "feature_count": len(feats)},
            "features": feats}
