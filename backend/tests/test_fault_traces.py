"""Tests for the merged Nepal fault-trace layer.

These never touch the network: each test writes fixture GeoJSON into a temporary cache
directory, which makes ``fetch_source`` return early.
"""

from __future__ import annotations

import json

import pytest
from app.ingestion import fault_traces as ft


# ---------------------------------------------------------------------------
# Structure classification. This is the part that decides whether a syncline hinge
# gets to masquerade as an earthquake source.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_type,expected",
    [
        ("thrust", ft.THRUST_SYSTEM),
        ("Reverse", ft.THRUST_SYSTEM),
        ("normal", ft.EXTENSIONAL),
        ("Detachment", ft.EXTENSIONAL),
        ("dextral", ft.STRIKE_SLIP),
        ("leftslip", ft.STRIKE_SLIP),
        ("right-slip", ft.STRIKE_SLIP),
        ("anticline", ft.FOLD_AXIS),
        ("syncline", ft.FOLD_AXIS),
        ("snycline", ft.FOLD_AXIS),  # upstream's own typo must still classify
        ("suture", ft.SUTURE),
    ],
)
def test_classify_structure_maps_known_types(raw_type, expected):
    assert ft.classify_structure(raw_type) == expected


def test_classify_structure_refuses_to_guess():
    """An unrecognised type becomes UNCLASSIFIED rather than being guessed."""
    assert ft.classify_structure("c") == ft.UNCLASSIFIED
    assert ft.classify_structure("f") == ft.UNCLASSIFIED
    assert ft.classify_structure(None) == ft.UNCLASSIFIED
    assert ft.classify_structure("") == ft.UNCLASSIFIED


def test_classify_structure_falls_back_to_the_name_for_named_thrusts():
    """Upstream sometimes leaves Type blank on a named thrust."""
    assert ft.classify_structure(None, "Main Boundary Thrust") == ft.THRUST_SYSTEM
    assert ft.classify_structure("", "Main Central Thrust") == ft.THRUST_SYSTEM
    assert ft.classify_structure(None, "Main Frontal Thrust") == ft.THRUST_SYSTEM
    # but a name that is not a thrust must not be promoted into one
    assert ft.classify_structure(None, "Tangra Yum Co Rift") == ft.UNCLASSIFIED


def test_seismogenic_tuple_membership_is_explicit():
    """Pin which classes the SEISMOGENIC_CLASSES subset contains.

    This pins the tuple, not a geological verdict. Excluding FOLD_AXIS is defensible in
    general but NOT safely true in the Himalayan foreland, where an active anticline
    overlies a blind thrust. Measured consequence of adopting this subset: seven of the
    491 sites move toward GREEN and none toward alarm. Nothing in production consumes
    it; see the caveat on SEISMOGENIC_CLASSES.
    """
    assert ft.FOLD_AXIS not in ft.SEISMOGENIC_CLASSES
    assert ft.SUTURE not in ft.SEISMOGENIC_CLASSES
    assert ft.UNCLASSIFIED not in ft.SEISMOGENIC_CLASSES
    assert ft.THRUST_SYSTEM in ft.SEISMOGENIC_CLASSES


# ---------------------------------------------------------------------------
# Slip rates.
# ---------------------------------------------------------------------------


def test_parse_slip_rate_reads_the_upstream_triple():
    """HimaTibetMap stores net slip as a (best, min, max) triple in mm/yr."""
    assert ft.parse_slip_rate("(15,10,21)") == {
        "best_mm_yr": 15.0, "min_mm_yr": 10.0, "max_mm_yr": 21.0,
    }
    assert ft.parse_slip_rate("(6,1,11)") == {
        "best_mm_yr": 6.0, "min_mm_yr": 1.0, "max_mm_yr": 11.0,
    }


def test_parse_slip_rate_is_absent_rather_than_zero():
    """A fault with no published rate is not a fault that creeps at zero."""
    for raw in (None, "", "None", "unknown", "fast", "15"):
        assert ft.parse_slip_rate(raw) is None


def test_parse_slip_rate_orders_the_bounds():
    out = ft.parse_slip_rate("(15,21,10)")
    assert out["min_mm_yr"] == 10.0
    assert out["max_mm_yr"] == 21.0


# ---------------------------------------------------------------------------
# Merge behaviour.
# ---------------------------------------------------------------------------


def _write_cache(tmp_path, monkeypatch, payloads):
    """Point the module's cache at tmp_path and plant fixture files in it."""
    monkeypatch.setattr(ft, "CACHE_DIR", tmp_path)
    for key, features in payloads.items():
        (tmp_path / f"{key}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features}),
            encoding="utf-8",
        )


def _line(coords, **props):
    return {"type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props}


# A trace inside the Nepal window, and the same geometry republished by a second source.
MFT_COORDS = [[84.0, 27.5], [84.5, 27.6], [85.0, 27.7]]
RIFT_COORDS = [[85.5, 29.5], [85.6, 29.9]]
OUTSIDE_COORDS = [[70.0, 20.0], [70.5, 20.5]]


def test_merge_drops_republished_geometry_and_keeps_the_richer_copy(tmp_path, monkeypatch):
    """GEM republishes HimaTibetMap's own geometry with the attributes stripped.

    The merge must keep exactly one copy, and it must be the one that still has the
    name and the slip rate.
    """
    _write_cache(tmp_path, monkeypatch, {
        "himatibetmap": [
            _line(MFT_COORDS, Name="Main Frontal Thrust", Type="thrust",
                  net_slip_r="(15,10,21)", Source="Lave and Avouac, 2000"),
        ],
        "gem": [
            # identical coordinates, no name, no slip rate - exactly what GEM carries
            _line(MFT_COORDS, fs_name=None, slip_type="Reverse", reference="x"),
        ],
    })
    layer = ft.build_nepal_fault_layer(keys=("himatibetmap", "gem"))

    assert layer["properties"]["feature_count"] == 1
    assert layer["properties"]["duplicates_dropped"] == {"gem": 1}

    kept = layer["features"][0]["properties"]
    assert kept["source"] == "himatibetmap"
    assert kept["fault_name"] == "Main Frontal Thrust"
    assert kept["slip_rate"]["best_mm_yr"] == 15.0
    assert kept["structure_class"] == ft.THRUST_SYSTEM


def test_merge_keeps_a_genuinely_unique_trace_from_the_second_source(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch, {
        "himatibetmap": [_line(MFT_COORDS, Name="Main Frontal Thrust", Type="thrust")],
        "gem": [_line(RIFT_COORDS, fs_name="Some Rift", slip_type="Normal")],
    })
    layer = ft.build_nepal_fault_layer(keys=("himatibetmap", "gem"))
    assert layer["properties"]["feature_count"] == 2
    assert layer["properties"]["duplicates_dropped"] == {}
    assert {f["properties"]["source"] for f in layer["features"]} == {"himatibetmap", "gem"}


def test_merge_excludes_geometry_outside_the_window(tmp_path, monkeypatch):
    _write_cache(tmp_path, monkeypatch, {
        "himatibetmap": [
            _line(MFT_COORDS, Name="Main Frontal Thrust", Type="thrust"),
            _line(OUTSIDE_COORDS, Name="Somewhere Else", Type="thrust"),
        ],
    })
    layer = ft.build_nepal_fault_layer(keys=("himatibetmap",))
    assert layer["properties"]["feature_count"] == 1
    assert layer["features"][0]["properties"]["fault_name"] == "Main Frontal Thrust"


def test_every_merged_trace_carries_its_provenance(tmp_path, monkeypatch):
    """No trace may enter the layer without saying where it came from."""
    _write_cache(tmp_path, monkeypatch, {
        "himatibetmap": [
            _line(MFT_COORDS, Name="Main Frontal Thrust", Type="thrust"),
            _line(RIFT_COORDS, Name=None, Type="normal"),
        ],
    })
    layer = ft.build_nepal_fault_layer(keys=("himatibetmap",))
    for feat in layer["features"]:
        p = feat["properties"]
        assert p["source"] == "himatibetmap"
        assert p["source_url"].startswith("https://")
        assert p["licence"]
        assert p["structure_class"] in (
            ft.THRUST_SYSTEM, ft.EXTENSIONAL, ft.STRIKE_SLIP,
            ft.FOLD_AXIS, ft.SUTURE, ft.UNCLASSIFIED,
        )


def test_layer_records_the_mbt_mct_gap_rather_than_hiding_it(tmp_path, monkeypatch):
    """The absence of the MBT and MCT is a published property of the layer.

    If a future source supplies them, this test should be updated deliberately - not
    quietly satisfied by a trace that merely looks like a thrust.
    """
    _write_cache(tmp_path, monkeypatch, {
        "himatibetmap": [_line(MFT_COORDS, Name="Main Frontal Thrust", Type="thrust")],
    })
    props = ft.build_nepal_fault_layer(keys=("himatibetmap",))["properties"]
    assert "Main Boundary Thrust (MBT)" in props["known_absent"]
    assert "Main Central Thrust (MCT)" in props["known_absent"]
    assert props["absence_note"]


def test_filter_by_class_selects_the_thrust_system(tmp_path, monkeypatch):
    """The whole point: let a consumer ask for thrusts, not for any nearby line."""
    _write_cache(tmp_path, monkeypatch, {
        "himatibetmap": [
            _line(MFT_COORDS, Name="Main Frontal Thrust", Type="thrust"),
            _line(RIFT_COORDS, Name="A Rift", Type="normal"),
            _line([[86.0, 27.9], [86.2, 28.0]], Name=None, Type="anticline"),
        ],
    })
    layer = ft.build_nepal_fault_layer(keys=("himatibetmap",))
    thrusts = ft.filter_by_class(layer, [ft.THRUST_SYSTEM])
    assert thrusts["properties"]["feature_count"] == 1
    assert thrusts["features"][0]["properties"]["fault_name"] == "Main Frontal Thrust"

    seismogenic = ft.filter_by_class(layer, ft.SEISMOGENIC_CLASSES)
    assert seismogenic["properties"]["feature_count"] == 2  # the fold axis is excluded


def test_enriching_the_layer_can_only_shorten_the_distance_to_a_trace(tmp_path, monkeypatch):
    """The safety invariant for this layer.

    fault_integrity rises with distance and multiplies c', so a SHORTER distance means
    lower cohesion and a lower factor of safety. Adding traces can only shorten the
    nearest-trace distance, so enriching this layer can only make the system more
    cautious. It must never be possible for an added trace to move a slope toward safe.
    """
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import LineString, Point

    _write_cache(tmp_path, monkeypatch, {
        "himatibetmap": [_line(MFT_COORDS, Name="Main Frontal Thrust", Type="thrust")],
        "gem": [_line(RIFT_COORDS, fs_name="A Rift", slip_type="Normal")],
    })

    def nearest_km(layer, lat, lon):
        pt = Point(lon, lat)
        return min(
            pt.distance(LineString(f["geometry"]["coordinates"]))
            for f in layer["features"]
        )

    base = ft.build_nepal_fault_layer(keys=("himatibetmap",))
    enriched = ft.build_nepal_fault_layer(keys=("himatibetmap", "gem"))
    assert enriched["properties"]["feature_count"] > base["properties"]["feature_count"]

    for lat, lon in [(27.5, 84.2), (28.2, 85.5), (29.5, 85.55), (27.0, 86.0)]:
        assert nearest_km(enriched, lat, lon) <= nearest_km(base, lat, lon) + 1e-12
