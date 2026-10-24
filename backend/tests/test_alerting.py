"""Tests for transition-only emergency alert dispatch (webhook / Telegram / email).

Fully offline: every transport is monkeypatched, no credentials are required and
no network call is made. These tests guard the property that actually matters
during the 6-month monsoon run — a site that stays RED must not re-alert on
every 30-minute cycle, and recipients must still get the all-clear.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.alerting import notifier
from app.alerting.notifier import EmergencyAlertNotifier, dispatch_alerts
from app.core.config import settings
from app.ml import ollama_evaluator


def _site(level: str, site_id: str = "glc_361", fs: float = 0.92) -> Dict[str, Any]:
    """One orchestrator-shaped site evaluation dict."""
    return {
        "site_id": site_id,
        "name": "Jhyaple Khola Scarp",
        "lat": 28.232,
        "lon": 85.518,
        "elev": 5680.0,
        "slope": 46.0,
        "cal_fs": fs,
        "alert": level,
        "triggers": ["CLOUD_BURST_SATURATION", "SAR_SLOPE_CREEP_ACCELERATION"],
    }


def _fake_broadcast(**kwargs: Any) -> Dict[str, str]:
    """Stand in for the Milestone 8 Ollama broadcast helper (no LLM, no network)."""
    return {
        "sms_nepali": "परीक्षण सूचना",
        "sms_english": "TEST ALERT",
        "cap_headline": f"MHEWS {kwargs.get('alert_level')} test",
    }


class _Clock:
    def __init__(self, t0: float = 1_700_000_000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def transports(monkeypatch) -> Dict[str, List[Any]]:
    """Enable all three channels with fake credentials and capture every send."""
    monkeypatch.setattr(settings, "alerts_enabled", True)
    monkeypatch.setattr(settings, "alert_min_level", "ORANGE")
    monkeypatch.setattr(settings, "alert_cooldown_minutes", 180)
    monkeypatch.setattr(settings, "alert_webhook_url", "https://hook.example.invalid/mhews")
    monkeypatch.setattr(settings, "telegram_bot_token", "TEST:TOKEN")
    monkeypatch.setattr(settings, "telegram_chat_id", "-1001234567890")
    monkeypatch.setattr(settings, "alert_smtp_host", "smtp.example.invalid")
    monkeypatch.setattr(settings, "alert_smtp_port", 587)
    monkeypatch.setattr(settings, "alert_smtp_user", "")
    monkeypatch.setattr(settings, "alert_smtp_password", "")
    monkeypatch.setattr(settings, "alert_email_to", "focal.person@example.invalid")

    captured: Dict[str, List[Any]] = {"http": [], "smtp": []}

    def fake_http(url: str, payload: Dict[str, Any], timeout: int = 10) -> int:
        captured["http"].append((url, payload))
        return 200

    def fake_smtp(**kwargs: Any) -> None:
        captured["smtp"].append(kwargs)

    monkeypatch.setattr(notifier, "_http_post_json", fake_http)
    monkeypatch.setattr(notifier, "_smtp_send", fake_smtp)
    monkeypatch.setattr(
        ollama_evaluator, "generate_nepali_civil_protection_broadcast", _fake_broadcast
    )
    return captured


def test_disabled_alerting_is_a_noop(monkeypatch, tmp_path: Path):
    """With alerts_enabled false nothing is sent and no state file is created."""
    monkeypatch.setattr(settings, "alerts_enabled", False)
    state = tmp_path / "alert_state.json"

    summary = EmergencyAlertNotifier(state_path=state).dispatch([_site("RED")])
    assert summary == {
        "enabled": False,
        "evaluated": 0,
        "transitions": 0,
        "sent": 0,
        "suppressed": 0,
        "failed": 0,
        "all_clear": 0,
        "channels": {},
    }
    assert not state.exists()

    module_summary = dispatch_alerts([_site("RED")])
    assert module_summary["enabled"] is False
    assert module_summary["sent"] == 0


def test_green_to_red_transition_fires_on_all_channels(transports, tmp_path: Path):
    """A site crossing into RED dispatches one message per configured channel."""
    n = EmergencyAlertNotifier(state_path=tmp_path / "state.json", clock=_Clock())
    assert n.dispatch([_site("GREEN", fs=1.6)])["sent"] == 0  # baseline, no transition

    summary = n.dispatch([_site("RED")])
    assert summary["sent"] == 1
    assert summary["transitions"] == 1
    assert summary["failed"] == 0
    assert summary["channels"] == {"webhook": 1, "telegram": 1, "email": 1}

    urls = [u for u, _ in transports["http"]]
    assert "https://hook.example.invalid/mhews" in urls
    assert any(u.startswith("https://api.telegram.org/botTEST:TOKEN/sendMessage") for u in urls)
    assert len(transports["smtp"]) == 1

    payload = transports["http"][0][1]
    assert payload["event"] == "ESCALATION"
    assert payload["alert_level"] == "RED"
    assert payload["previous_level"] == "GREEN"
    assert payload["level_set_by"] == "physics_engine"
    assert payload["site_id"] == "glc_361"
    assert "CLOUD_BURST_SATURATION" in payload["active_triggers"]
    assert payload["failure_probability_pct"] > 50.0
    assert "openstreetmap.org" in payload["map_url"]
    assert payload["timestamp_nepal_iso"].endswith("+05:45")
    assert "TEST ALERT" in payload["text"]


def test_sustained_level_does_not_refire(transports, tmp_path: Path):
    """Staying RED across cycles must not re-alert (monsoon message-flood guard)."""
    n = EmergencyAlertNotifier(state_path=tmp_path / "state.json", clock=_Clock())
    assert n.dispatch([_site("RED")])["sent"] == 1
    before = len(transports["http"])

    for _ in range(5):
        summary = n.dispatch([_site("RED")])
        assert summary["sent"] == 0
        assert summary["transitions"] == 0
    assert len(transports["http"]) == before


def test_cooldown_suppresses_rapid_reescalation(transports, tmp_path: Path):
    """Flapping GREEN->RED->GREEN->RED is bounded by the per-site+level cooldown."""
    clock = _Clock()
    n = EmergencyAlertNotifier(state_path=tmp_path / "state.json", clock=clock)
    assert n.dispatch([_site("RED")])["sent"] == 1

    clock.advance(1800)
    assert n.dispatch([_site("GREEN", fs=1.6)])["sent"] == 1  # all-clear

    clock.advance(1800)
    suppressed = n.dispatch([_site("RED")])
    assert suppressed["sent"] == 0
    assert suppressed["transitions"] == 1
    assert suppressed["suppressed"] == 1

    clock.advance(settings.alert_cooldown_minutes * 60)  # cooldown expires, still RED
    assert n.dispatch([_site("RED")])["sent"] == 1


def test_all_clear_on_return_to_green(transports, tmp_path: Path):
    """Recipients get an explicit stand-down when the site drops back to GREEN."""
    n = EmergencyAlertNotifier(state_path=tmp_path / "state.json", clock=_Clock())
    n.dispatch([_site("ORANGE", fs=0.99)])
    transports["http"].clear()

    summary = n.dispatch([_site("GREEN", fs=1.7)])
    assert summary["sent"] == 1
    assert summary["all_clear"] == 1

    payload = transports["http"][0][1]
    assert payload["event"] == "ALL_CLEAR"
    assert payload["alert_level"] == "GREEN"
    assert payload["previous_level"] == "ORANGE"
    assert "ALL CLEAR" in payload["text"]

    # A site that simply stays GREEN afterwards never alerts again.
    assert n.dispatch([_site("GREEN", fs=1.7)])["sent"] == 0


def test_yellow_below_min_level_is_not_dispatched(transports, tmp_path: Path):
    """YELLOW is below alert_min_level=ORANGE, so nothing is sent."""
    n = EmergencyAlertNotifier(state_path=tmp_path / "state.json", clock=_Clock())
    summary = n.dispatch([_site("YELLOW", fs=1.2)])
    assert summary["sent"] == 0
    assert summary["transitions"] == 0
    assert transports["http"] == []


def test_one_failing_channel_does_not_block_the_others(transports, monkeypatch, tmp_path: Path):
    """A webhook that raises must not stop Telegram or email delivery."""

    def flaky_http(url: str, payload: Dict[str, Any], timeout: int = 10) -> int:
        if "hook.example.invalid" in url:
            raise OSError("connection reset by peer")
        transports["http"].append((url, payload))
        return 200

    monkeypatch.setattr(notifier, "_http_post_json", flaky_http)

    n = EmergencyAlertNotifier(state_path=tmp_path / "state.json", clock=_Clock())
    summary = n.dispatch([_site("RED")])

    assert summary["sent"] == 1
    assert summary["failed"] == 0
    assert summary["channels"] == {"telegram": 1, "email": 1}
    assert len(transports["smtp"]) == 1


def test_all_channels_failing_is_reported_and_retried(transports, monkeypatch, tmp_path: Path):
    """When nothing is delivered the alert is not marked sent, so the next cycle retries."""

    def dead_http(url: str, payload: Dict[str, Any], timeout: int = 10) -> int:
        raise OSError("network unreachable")

    def dead_smtp(**kwargs: Any) -> None:
        raise OSError("smtp unreachable")

    monkeypatch.setattr(notifier, "_http_post_json", dead_http)
    monkeypatch.setattr(notifier, "_smtp_send", dead_smtp)

    n = EmergencyAlertNotifier(state_path=tmp_path / "state.json", clock=_Clock())
    first = n.dispatch([_site("RED")])
    assert first["sent"] == 0
    assert first["failed"] == 1

    # Transport recovers; the still-RED site is retried rather than silently lost.
    monkeypatch.setattr(notifier, "_http_post_json", lambda url, payload, timeout=10: 200)
    assert n.dispatch([_site("RED")])["sent"] == 1


def test_state_survives_a_restart(transports, tmp_path: Path):
    """A fresh notifier (simulating an orchestrator restart) must not replay alerts."""
    state = tmp_path / "state.json"
    first = EmergencyAlertNotifier(state_path=state, clock=_Clock())
    assert first.dispatch([_site("RED")])["sent"] == 1
    assert state.exists()

    restarted = EmergencyAlertNotifier(state_path=state, clock=_Clock())
    summary = restarted.dispatch([_site("RED")])
    assert summary["sent"] == 0
    assert summary["transitions"] == 0


def test_enabled_with_no_channels_configured_is_safe(monkeypatch, tmp_path: Path):
    """Alerts enabled but unconfigured: no crash, transitions reported as undeliverable."""
    monkeypatch.setattr(settings, "alerts_enabled", True)
    monkeypatch.setattr(settings, "alert_webhook_url", "")
    monkeypatch.setattr(settings, "telegram_bot_token", "")
    monkeypatch.setattr(settings, "telegram_chat_id", "")
    monkeypatch.setattr(settings, "alert_smtp_host", "")
    monkeypatch.setattr(settings, "alert_email_to", "")

    summary = EmergencyAlertNotifier(state_path=tmp_path / "state.json").dispatch([_site("RED")])
    assert summary["sent"] == 0
    assert summary["suppressed"] == 1
    assert "note" in summary


def test_district_is_never_guessed_from_a_coarse_basin_id(transports):
    """A Dolakha/Solukhumbu site tagged basin_id='langtang' must not print 'Rasuwa'."""
    tsho_rolpa = {
        "site_id": "tsho_rolpa",
        "name": "Tsho Rolpa Glacial Lake (Rolwaling)",
        "lat": 27.862,
        "lon": 86.476,
        "elev": 5190.0,
        "slope": 44.0,
        "cal_fs": 0.95,
        "alert": "RED",
        "triggers": ["CRITICAL_THERMAL_ICE_MELT"],
        "basin_id": "langtang",
    }
    payload = notifier.build_alert_payload(tsho_rolpa, "ESCALATION", "GREEN")
    assert payload["district"] is None
    assert payload["basin_id"] != "langtang"

    in_basin = notifier.build_alert_payload(_site("RED"), "ESCALATION", "GREEN")
    assert in_basin["district"] == "Rasuwa"


def test_malformed_evaluations_never_raise(transports, tmp_path: Path):
    """Garbage in a site evaluation must not take down the physics engine."""
    n = EmergencyAlertNotifier(state_path=tmp_path / "state.json", clock=_Clock())
    summary = n.dispatch(
        [
            {},  # no site_id: skipped
            {"site_id": "broken_site", "alert": "RED", "lat": None, "lon": None, "cal_fs": None},
            _site("RED", site_id="good_site"),
        ]
    )
    assert summary["sent"] >= 1
    assert isinstance(summary["failed"], int)
