"""Offline tests for CDSE Sentinel-1 ingestion and creep provenance labelling.

Every test runs with no network and no credentials: ``urllib.request.urlopen`` is replaced
by a recorder that either serves canned CDSE payloads or fails the test if it is called at
all. The central guarantee under test is that ``MEASURED_INSAR`` can never be produced
unless a real deformation product backed the number.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

import pytest

from app.core.config import settings
from app.ingestion import insar_cdse
from app.ingestion.insar_cdse import (
    PROVENANCE_MEASURED,
    PROVENANCE_PROXY,
    PROVENANCE_UNAVAILABLE,
    assess_insar_readiness,
    compute_creep_proxy,
    describe_insar_capability,
    fetch_cdse_access_token,
    get_insar_creep,
    get_insar_creep_batch,
    register_measured_velocity_provider,
    search_sentinel1_scenes,
)

LANGTANG_LAT, LANGTANG_LON = 28.215, 85.515

TOKEN_PAYLOAD = {"access_token": "fake-token-not-a-secret", "expires_in": 600}

# Trimmed from a real CDSE OData response captured on 2026-10-23.
ODATA_PAYLOAD = {
    "@odata.context": "$metadata#Products(Attributes())",
    "value": [
        {
            "Id": "e8071a8f-39b6-4124-b6c6-146e8ceba15c",
            "Name": "S1D_IW_SLC__1SDV_20261023T001844_20261023T001911_005135_009BBB_33FF.SAFE",
            "Online": True,
            "S3Path": "/eodata/Sentinel-1/SAR/IW_SLC__1S/2026/10/23/S1D_IW_SLC.SAFE",
            "ContentDate": {"Start": "2026-10-23T00:18:44.731153Z", "End": "2026-10-23T00:19:11Z"},
            "Attributes": [
                {"Name": "productType", "Value": "IW_SLC__1S"},
                {"Name": "orbitDirection", "Value": "DESCENDING"},
                {"Name": "relativeOrbitNumber", "Value": 19},
                {"Name": "platformSerialIdentifier", "Value": "D"},
                {"Name": "polarisationChannels", "Value": "VV&VH"},
            ],
        },
        {
            "Id": "11111111-2222-3333-4444-555555555555",
            "Name": "S1D_IW_SLC__1SDV_20261011T001844_20261011T001911_004987_0098AA_1122.SAFE",
            "Online": True,
            "S3Path": "/eodata/Sentinel-1/SAR/IW_SLC__1S/2026/10/11/S1D_IW_SLC.SAFE",
            "ContentDate": {"Start": "2026-10-11T00:18:44.000000Z", "End": "2026-10-11T00:19:11Z"},
            "Attributes": [
                {"Name": "productType", "Value": "IW_SLC__1S"},
                {"Name": "orbitDirection", "Value": "DESCENDING"},
                {"Name": "relativeOrbitNumber", "Value": 19},
                {"Name": "platformSerialIdentifier", "Value": "D"},
                {"Name": "polarisationChannels", "Value": "VV&VH"},
            ],
        },
    ],
}


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _Transport:
    """Records every HTTP attempt and serves canned CDSE payloads."""

    def __init__(self, token_payload=TOKEN_PAYLOAD, odata_payload=ODATA_PAYLOAD, fail=None):
        self.calls = []
        self.token_payload = token_payload
        self.odata_payload = odata_payload
        self.fail = fail

    def __call__(self, request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        self.calls.append(url)
        if self.fail is not None:
            raise self.fail
        if "openid-connect/token" in url:
            return _FakeResponse(json.dumps(self.token_payload).encode("utf-8"))
        if "/odata/" in url:
            return _FakeResponse(json.dumps(self.odata_payload).encode("utf-8"))
        raise AssertionError(f"unexpected URL requested: {url}")

    @property
    def token_calls(self):
        return [u for u in self.calls if "openid-connect/token" in u]

    @property
    def catalogue_calls(self):
        return [u for u in self.calls if "/odata/" in u]


def _forbid_network(request, timeout=None):
    raise AssertionError("network call attempted when none was allowed")


class _Site:
    def __init__(self, site_id, lat, lon, slope_deg=38.0, elevation_m=3100.0,
                 hazard_type="rainfall_landslide"):
        self.site_id = site_id
        self.latitude = lat
        self.longitude = lon
        self.slope_deg = slope_deg
        self.elevation_m = elevation_m
        self.hazard_type = hazard_type


@pytest.fixture(autouse=True)
def isolated_module_state(tmp_path, monkeypatch):
    """Give every test a private cache file and a clean token/provider state."""
    insar_cdse.set_cache_path(tmp_path / "insar_cdse_cache.json")
    insar_cdse.clear_caches()
    monkeypatch.setattr(settings, "cdse_enabled", False, raising=False)
    monkeypatch.setattr(settings, "cdse_client_id", "", raising=False)
    monkeypatch.setattr(settings, "cdse_client_secret", "", raising=False)
    yield
    insar_cdse.clear_caches()
    insar_cdse.set_cache_path(None)


def _enable_cdse(monkeypatch):
    monkeypatch.setattr(settings, "cdse_enabled", True, raising=False)
    monkeypatch.setattr(settings, "cdse_client_id", "fake-client", raising=False)
    monkeypatch.setattr(settings, "cdse_client_secret", "fake-secret", raising=False)


# --- disabled / credential-less paths: no network at all --------------------------------
def test_disabled_returns_proxy_without_any_network_call(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON, slope_deg=42.0, dynamic_fs=1.05)

    assert result["provenance"] == PROVENANCE_PROXY
    assert result["is_measurement"] is False
    assert result["los_velocity_mm_month"] > 0.0
    assert "NOT InSAR" in result["source"]
    assert result["cdse"]["status"] == "DISABLED"
    assert result["cdse"]["measured_velocity_available"] is False


def test_disabled_without_proxy_inputs_returns_unavailable(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON)

    assert result["provenance"] == PROVENANCE_UNAVAILABLE
    assert result["los_velocity_mm_month"] is None
    assert result["confidence"] == 0.0


def test_enabled_but_no_credentials_makes_no_network_call(monkeypatch):
    monkeypatch.setattr(settings, "cdse_enabled", True, raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON, slope_deg=35.0, dynamic_fs=1.2)

    assert result["cdse"]["status"] == "NO_CREDENTIALS"
    assert result["provenance"] == PROVENANCE_PROXY


def test_token_fetch_returns_none_without_credentials(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
    assert fetch_cdse_access_token() is None


# --- degraded network paths --------------------------------------------------------------
def test_token_failure_degrades_cleanly(monkeypatch):
    _enable_cdse(monkeypatch)
    transport = _Transport(fail=urllib.error.URLError("no route to host"))
    monkeypatch.setattr(urllib.request, "urlopen", transport)

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON, slope_deg=40.0, dynamic_fs=0.98)

    assert result["cdse"]["status"] == "TOKEN_ERROR"
    assert result["provenance"] == PROVENANCE_PROXY
    assert result["los_velocity_mm_month"] > 0.0
    assert transport.token_calls, "a token request should have been attempted"
    assert not transport.catalogue_calls


def test_catalogue_failure_degrades_cleanly(monkeypatch):
    _enable_cdse(monkeypatch)

    class _TokenOkCatalogueBroken(_Transport):
        def __call__(self, request, timeout=None):
            url = request.full_url
            self.calls.append(url)
            if "openid-connect/token" in url:
                return _FakeResponse(json.dumps(TOKEN_PAYLOAD).encode("utf-8"))
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

    transport = _TokenOkCatalogueBroken()
    monkeypatch.setattr(urllib.request, "urlopen", transport)

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON, slope_deg=40.0, dynamic_fs=0.98)

    assert result["cdse"]["status"] == "CATALOGUE_ERROR"
    assert result["provenance"] == PROVENANCE_PROXY


# --- mocked happy path --------------------------------------------------------------------
def test_token_and_catalogue_response_parse(monkeypatch):
    _enable_cdse(monkeypatch)
    transport = _Transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)

    token = fetch_cdse_access_token()
    assert token == "fake-token-not-a-secret"

    search = search_sentinel1_scenes(LANGTANG_LAT, LANGTANG_LON, token=token)
    assert search["ok"] is True
    assert len(search["products"]) == 2

    first = search["products"][0]
    assert first["product_id"] == "e8071a8f-39b6-4124-b6c6-146e8ceba15c"
    assert first["product_type"] == "IW_SLC__1S"
    assert first["relative_orbit"] == 19
    assert first["orbit_direction"] == "DESCENDING"
    assert first["online"] is True

    query = transport.catalogue_calls[0]
    assert "%24filter=" in query and "SENTINEL-1" in urllib.parse.unquote(query)
    assert "POINT(85.515000 28.215000)" in urllib.parse.unquote(query)


def test_readiness_reports_pair_formable_without_claiming_a_velocity(monkeypatch):
    _enable_cdse(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", _Transport())

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON, slope_deg=44.0, dynamic_fs=0.95)
    context = result["cdse"]

    assert context["status"] == "SCENES_ONLY"
    assert context["measured_velocity_available"] is False
    assert context["insar_readiness"]["scene_count"] == 2
    assert context["insar_readiness"]["interferometric_pair_formable"] is True
    assert context["insar_readiness"]["best_relative_orbit"] == 19
    # Scenes were found, but the reported number is still the proxy.
    assert result["provenance"] == PROVENANCE_PROXY


def test_assess_readiness_single_scene_is_not_pair_formable():
    readiness = assess_insar_readiness(
        [{"relative_orbit": 19, "orbit_direction": "DESCENDING", "sensing_start_utc": "2026-10-23T00:00:00Z"}]
    )
    assert readiness["interferometric_pair_formable"] is False
    assert readiness["scene_count"] == 1


# --- batching and caching ------------------------------------------------------------------
def test_batch_groups_nearby_sites_into_one_catalogue_query(monkeypatch):
    _enable_cdse(monkeypatch)
    transport = _Transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)

    sites = [
        _Site("a", 28.21, 85.51),
        _Site("b", 28.22, 85.52),
        _Site("c", 28.24, 85.53),
    ]
    results = get_insar_creep_batch(
        sites, context={s.site_id: {"dynamic_fs": 1.1, "rain_72h_mm": 40.0} for s in sites}
    )

    assert set(results) == {"a", "b", "c"}
    assert len(transport.catalogue_calls) == 1
    assert all(r["provenance"] == PROVENANCE_PROXY for r in results.values())


def test_batch_cache_prevents_requery_within_repeat_window(monkeypatch):
    _enable_cdse(monkeypatch)
    transport = _Transport()
    monkeypatch.setattr(urllib.request, "urlopen", transport)

    sites = [_Site("a", 28.21, 85.51), _Site("b", 28.60, 84.63)]
    ctx = {"a": {"dynamic_fs": 1.1}, "b": {"dynamic_fs": 1.0}}

    get_insar_creep_batch(sites, context=ctx)
    first_round = len(transport.catalogue_calls)
    assert first_round == 2

    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
    second = get_insar_creep_batch(sites, context=ctx)

    assert second["a"]["cdse"]["cache_hit"] is True
    assert second["a"]["cdse"]["status"] == "SCENES_ONLY"


def test_batch_respects_query_budget(monkeypatch):
    _enable_cdse(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)

    sites = [_Site("a", 28.21, 85.51), _Site("b", 27.70, 85.21)]
    results = get_insar_creep_batch(
        sites, context={"a": {"dynamic_fs": 1.1}, "b": {"dynamic_fs": 1.1}},
        max_catalogue_queries=0,
    )

    assert results["a"]["cdse"]["status"] == "QUERY_BUDGET_EXHAUSTED"
    assert results["a"]["provenance"] == PROVENANCE_PROXY


def test_batch_accepts_dict_sites_and_marks_missing_inputs_unavailable(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)

    results = get_insar_creep_batch(
        [{"site_id": "d1", "latitude": 28.2, "longitude": 85.5, "slope_deg": 30.0}]
    )

    assert results["d1"]["provenance"] == PROVENANCE_UNAVAILABLE
    assert results["d1"]["los_velocity_mm_month"] is None


# --- the central honesty invariant -----------------------------------------------------------
def test_provenance_is_never_measured_from_cdse_scenes(monkeypatch):
    _enable_cdse(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", _Transport())

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON, slope_deg=44.0, dynamic_fs=0.9)
    assert result["provenance"] != PROVENANCE_MEASURED
    assert describe_insar_capability()["cdse_offers_deformation_product"] is False


@pytest.mark.parametrize(
    "bad_provider",
    [
        lambda lat, lon: None,
        lambda lat, lon: {"source": "LiCSAR"},  # no velocity
        lambda lat, lon: {"los_velocity_mm_month": 12.0},  # no product reference
        lambda lat, lon: {"los_velocity_mm_month": "12.0", "product_id": "x"},  # not numeric
        lambda lat, lon: {"los_velocity_mm_month": float("nan"), "product_id": "x"},
        lambda lat, lon: "not a dict",
        lambda lat, lon: (_ for _ in ()).throw(RuntimeError("provider exploded")),
    ],
)
def test_unbacked_provider_results_never_become_measured(monkeypatch, bad_provider):
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
    register_measured_velocity_provider(bad_provider)

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON, slope_deg=40.0, dynamic_fs=1.0)

    assert result["provenance"] == PROVENANCE_PROXY
    assert result["is_measurement"] is False


def test_real_product_backed_provider_yields_measured_insar(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
    register_measured_velocity_provider(
        lambda lat, lon: {
            "los_velocity_mm_month": -7.4,
            "product_id": "LiCSBAS_vel_filt_056D_05139_131313",
            "source": "COMET-LiCSAR frame 056D_05139_131313 LiCSBAS velocity",
            "acquired_utc": 1760000000.0,
            "confidence": 0.72,
        }
    )

    result = get_insar_creep(LANGTANG_LAT, LANGTANG_LON, slope_deg=40.0, dynamic_fs=1.0)

    assert result["provenance"] == PROVENANCE_MEASURED
    assert result["is_measurement"] is True
    assert result["los_velocity_mm_month"] == pytest.approx(-7.4)
    assert result["product_id"].startswith("LiCSBAS_vel_filt")


# --- shared proxy function ----------------------------------------------------------------------
def test_compute_creep_proxy_matches_the_physics_engine():
    from app.physics.infinite_slope import compute_viscoplastic_creep_velocity

    proxy = compute_creep_proxy(slope_deg=41.0, dynamic_fs=0.97, elevation_m=4800.0,
                                hazard_type="rock_ice_avalanche", rain_72h_mm=90.0)
    expected = compute_viscoplastic_creep_velocity(
        slope_deg=41.0, dynamic_fs=0.97, elevation_m=4800.0,
        hazard_type="rock_ice_avalanche", rain_72h_mm=90.0,
    )

    assert proxy["los_velocity_mm_month"] == pytest.approx(expected)
    assert proxy["provenance"] == PROVENANCE_PROXY
    assert proxy["display_label"] == "Modelled creep (physics proxy)"


def test_odata_endpoint_is_derived_from_the_stale_resto_setting():
    # settings.cdse_catalogue_url still points at the retired resto path (now HTTP 404).
    endpoint = insar_cdse._odata_products_url()
    assert endpoint == "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
