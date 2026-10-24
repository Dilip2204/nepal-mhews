"""Multi-channel emergency alert dispatch for ORANGE/RED transitions.

Real-world purpose: during the unattended 6-month run nobody is watching the
dashboard. When the physics engine moves a site into an alerting level, the
people who would act on it (district focal persons, the ward office, the
project operator) have to be told on a channel they already read. This module
is the transport layer for that, and nothing more:

  * The physics engine sets ``alert``. This module never computes, upgrades or
    downgrades a level, and no LLM is allowed to either (an LLM is used only to
    phrase the bilingual civil-protection wording, via the Milestone 8 helper,
    with a deterministic template as its own fallback).
  * It fires on **transitions only** — a site crossing up into
    ``settings.alert_min_level`` (default ORANGE), and the de-escalation back to
    GREEN that lets recipients stand down. A site that simply stays RED for a
    week does not produce 336 messages.
  * Per site+level cooldown (``settings.alert_cooldown_minutes``, default 180)
    bounds a flapping site to one escalation plus one all-clear per window.
  * Transition state is persisted to JSON under ``backend/data/`` so a restart
    of the orchestrator does not replay alerts that were already sent.

Channels (standard library only — ``urllib.request`` and ``smtplib``):
  1. Generic JSON webhook (Slack / Teams / any custom relay)
  2. Telegram Bot API ``sendMessage``
  3. SMTP email

Every channel is independent and failure-isolated: one raising or timing out
never stops the others and never propagates to the caller. Everything is
disabled by default (``settings.alerts_enabled`` is false and no credentials
are configured), so importing or calling this module in a default checkout is a
no-op.

WhatsApp is deliberately NOT implemented. There is no credential-less path to
WhatsApp; it requires a Meta Business account and an approved WhatsApp Cloud
API message template, which this project does not have. Rather than pretend,
the webhook payload is kept relay-friendly — ``payload["text"]`` is a complete
ready-to-send message body and ``payload["broadcast"]`` holds the Nepali and
English SMS strings — so an external WhatsApp/SMS relay can consume the webhook
and forward it.

Public API::

    from app.alerting.notifier import dispatch_alerts
    summary = dispatch_alerts(site_evaluations)

where ``site_evaluations`` is the list of dicts the perpetual orchestrator
already builds per cycle.
"""

from __future__ import annotations

import json
import logging
import math
import os
import smtplib
import ssl
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings

logger = logging.getLogger("nepal_mhews.alerting")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_STATE_PATH = DATA_DIR / "alert_dispatch_state.json"
STATE_SCHEMA_VERSION = 1

LEVEL_ORDER: Dict[str, int] = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
ALL_CLEAR_LEVEL = "GREEN"
NEPAL_TZ = timezone(timedelta(hours=5, minutes=45))  # Nepal Standard Time, UTC+05:45

HTTP_TIMEOUT_S = 10
SMTP_TIMEOUT_S = 20
# The bilingual broadcast helper talks to Ollama and defaults to a 180 s timeout.
# An unreachable tunnel must not stall a 30-minute physics cycle, so cap it hard;
# the helper falls back to its own deterministic Nepali/English template.
BROADCAST_TIMEOUT_S = 8


# --------------------------------------------------------------------------
# Low-level transports (monkeypatched in tests; never called with creds absent)
# --------------------------------------------------------------------------


def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = HTTP_TIMEOUT_S) -> int:
    """POST a JSON document and return the HTTP status code."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Himalayan-MHEWS/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(getattr(resp, "status", 200) or 200)


def _smtp_send(host: str, port: int, user: str, password: str, message: EmailMessage) -> None:
    """Deliver one message over SMTP, using implicit TLS on 465 and STARTTLS elsewhere."""
    if int(port) == 465:
        with smtplib.SMTP_SSL(host, int(port), timeout=SMTP_TIMEOUT_S, context=ssl.create_default_context()) as srv:
            if user:
                srv.login(user, password)
            srv.send_message(message)
        return

    with smtplib.SMTP(host, int(port), timeout=SMTP_TIMEOUT_S) as srv:
        srv.ehlo()
        try:
            srv.starttls(context=ssl.create_default_context())
            srv.ehlo()
        except smtplib.SMTPException:
            logger.warning("SMTP server %s:%s refused STARTTLS; continuing unencrypted", host, port)
        if user:
            srv.login(user, password)
        srv.send_message(message)


# --------------------------------------------------------------------------
# Payload construction
# --------------------------------------------------------------------------


def _level_rank(level: Optional[str]) -> int:
    return LEVEL_ORDER.get(str(level or "").strip().upper(), 0)


def _min_level_rank() -> int:
    rank = LEVEL_ORDER.get(str(getattr(settings, "alert_min_level", "ORANGE")).strip().upper())
    if rank is None:
        logger.warning(
            "Unrecognised alert_min_level %r; falling back to ORANGE",
            getattr(settings, "alert_min_level", None),
        )
        return LEVEL_ORDER["ORANGE"]
    return rank


def _failure_probability_pct(fs: float) -> float:
    """Engine's failure-probability sigmoid (same constant as the orchestrator)."""
    try:
        return float(1.0 / (1.0 + math.exp(6.8 * (float(fs) - 1.0)))) * 100.0
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _map_url(lat: float, lon: float) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=13/{lat:.5f}/{lon:.5f}"


def _resolve_basin_and_district(ev: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (basin_id, district). District stays None when it cannot be resolved honestly.

    The site catalog carries no district field, and this project has no district
    polygons, so a district is reported only when the site's own coordinates fall
    inside a pilot basin whose district is known from config. The catalog's
    ``basin_id`` is deliberately not trusted for this: it is coarse, and several
    far-eastern stations (Tsho Rolpa, Imja) are tagged ``langtang`` there, which
    would print "Rasuwa" on an alert for Dolakha or Solukhumbu. Never guess one.
    """
    basin_from_coords: Optional[str] = None
    try:
        from app.engine.sites_catalog import determine_basin

        basin_from_coords = determine_basin(float(ev["lon"]), float(ev["lat"]))
    except Exception as exc:  # catalog import is optional for the transport layer
        logger.debug("Basin resolution unavailable: %s", exc)

    basin_id = basin_from_coords or ev.get("basin_id")

    district = ev.get("district")
    if not district and basin_from_coords:
        try:
            from app.core.config import PILOT_BASINS

            basin_cfg = PILOT_BASINS.get(basin_from_coords)
            district = basin_cfg.district if basin_cfg else None
        except Exception:
            district = None
    return basin_id, district


def _settlements_for(ev: Dict[str, Any], basin_id: Optional[str], level: str) -> List[Dict[str, Any]]:
    """Downstream arrival times via the Milestone 8 routing helper (empty if unavailable)."""
    if not basin_id:
        return []
    try:
        from app.physics.runout_routing import compute_settlement_impact_and_arrival_times

        return compute_settlement_impact_and_arrival_times(
            basin_id=basin_id,
            peak_velocity_m_s=35.0 if level in ("RED", "ORANGE") else 14.0,
            runout_length_m=12500.0 if level == "RED" else 6500.0 if level == "ORANGE" else 1500.0,
            source_station_name=str(ev.get("name", ev.get("site_id", "site"))),
        )
    except Exception as exc:
        logger.warning("Settlement ETA helper unavailable: %s", exc)
        return []


def _broadcast_for(
    basin_name: str, level: str, fs: float, triggers: Sequence[str], settlements: Sequence[Dict[str, Any]]
) -> Dict[str, str]:
    """Bilingual civil-protection wording from the Milestone 8 helper, English-only on failure."""
    try:
        from app.ml.ollama_evaluator import generate_nepali_civil_protection_broadcast

        out = generate_nepali_civil_protection_broadcast(
            basin_name=basin_name,
            alert_level=level,
            dynamic_fs=float(fs),
            active_triggers=list(triggers),
            impacted_settlements=list(settlements),
            timeout_s=BROADCAST_TIMEOUT_S,
        )
        if isinstance(out, dict) and out.get("sms_english"):
            return {
                "sms_nepali": str(out.get("sms_nepali", "")),
                "sms_english": str(out.get("sms_english", "")),
                "cap_headline": str(out.get("cap_headline", f"MHEWS {level} - {basin_name}")),
                "language": "ne+en",
            }
        logger.warning("Bilingual broadcast returned no usable text; using English-only fallback")
    except Exception as exc:
        logger.warning("Bilingual broadcast helper unavailable (%s); using English-only fallback", exc)

    return {
        "sms_nepali": "",
        "sms_english": (
            f"MHEWS {level} alert for {basin_name}: factor of safety {float(fs):.2f}, "
            f"active triggers: {', '.join(triggers) if triggers else 'none'}."
        ),
        "cap_headline": f"MHEWS {level} Hazard Warning - {basin_name}",
        "language": "en",
    }


def build_alert_payload(
    ev: Dict[str, Any],
    event: str,
    previous_level: str,
    now_utc: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the channel-agnostic alert document for one site transition.

    ``event`` is ``ESCALATION`` or ``ALL_CLEAR``. The returned dict is what the
    webhook POSTs verbatim; ``payload["text"]`` is the rendered message body used
    for Telegram and email (and consumable by a WhatsApp/SMS relay).
    """
    ts = float(now_utc if now_utc is not None else time.time())
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    level = str(ev.get("alert", "GREEN")).upper()
    fs = float(ev.get("cal_fs", ev.get("dynamic_fs", 0.0)) or 0.0)
    lat, lon = float(ev.get("lat", 0.0)), float(ev.get("lon", 0.0))
    triggers = [str(t) for t in (ev.get("triggers") or [])]

    basin_id, district = _resolve_basin_and_district(ev)
    basin_name = f"{basin_id.replace('_', ' ').title()} Corridor" if basin_id else str(ev.get("name", "Nepal"))
    settlements = _settlements_for(ev, basin_id, level)
    broadcast = _broadcast_for(basin_name, level, fs, triggers, settlements)

    fail_prob = ev.get("failure_prob_pct")
    if fail_prob is None:
        fail_prob = _failure_probability_pct(fs)

    payload: Dict[str, Any] = {
        "schema": "mhews.alert.v1",
        "event": event,
        "source": "Himalayan-MHEWS perpetual physics engine",
        "level_set_by": "physics_engine",
        "site_id": ev.get("site_id"),
        "site_name": ev.get("name"),
        "district": district,
        "basin_id": basin_id,
        "latitude": lat,
        "longitude": lon,
        "elevation_m": ev.get("elev"),
        "slope_deg": ev.get("slope"),
        "alert_level": level,
        "previous_level": str(previous_level or "").upper() or None,
        "dynamic_fs": round(fs, 3),
        "failure_probability_pct": round(float(fail_prob), 1),
        "active_triggers": triggers,
        "map_url": _map_url(lat, lon),
        "timestamp_utc": round(ts, 1),
        "timestamp_utc_iso": dt_utc.isoformat(timespec="seconds"),
        "timestamp_nepal_iso": dt_utc.astimezone(NEPAL_TZ).isoformat(timespec="seconds"),
        "downstream_settlements": settlements[:3],
        "broadcast": broadcast,
    }
    payload["text"] = render_alert_text(payload)
    return payload


def render_alert_text(payload: Dict[str, Any]) -> str:
    """Render the plain-text message body sent over Telegram, email and relays."""
    level = payload.get("alert_level", "GREEN")
    is_clear = payload.get("event") == "ALL_CLEAR"
    header = (
        f"[MHEWS ALL CLEAR] {payload.get('site_name')} back to GREEN"
        if is_clear
        else f"[MHEWS {level} ALERT] {payload.get('site_name')}"
    )
    where = payload.get("district") or payload.get("basin_id") or "location not resolved"
    lines = [
        header,
        f"Site: {payload.get('site_name')} ({payload.get('site_id')})",
        f"Area: {where}",
        f"Coords: {payload.get('latitude'):.4f}, {payload.get('longitude'):.4f}",
        f"Level: {payload.get('previous_level')} -> {level}",
        f"Dynamic factor of safety: {payload.get('dynamic_fs')}",
        f"Failure probability: {payload.get('failure_probability_pct')}%",
        "Active triggers: " + (", ".join(payload.get("active_triggers") or []) or "none"),
    ]

    settlements = payload.get("downstream_settlements") or []
    if settlements and not is_clear:
        lines.append("Downstream arrival estimates:")
        for s in settlements:
            lines.append(
                f"  - {s.get('settlement_name')} ({s.get('population_at_risk')} people): "
                f"~{s.get('eta_minutes')} min, {s.get('evacuation_status')}"
            )

    lines.append(f"Map: {payload.get('map_url')}")
    lines.append(f"Time: {payload.get('timestamp_utc_iso')} UTC / {payload.get('timestamp_nepal_iso')} Nepal")

    broadcast = payload.get("broadcast") or {}
    if not is_clear:
        if broadcast.get("sms_english"):
            lines.append("EN: " + broadcast["sms_english"])
        if broadcast.get("sms_nepali"):
            lines.append("NE: " + broadcast["sms_nepali"])
    else:
        lines.append("Conditions have returned to GREEN. Stand down; continue routine monitoring.")

    lines.append("Alert level computed by the MHEWS physics engine (infinite-slope FS + precursor triggers).")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Channels — each returns True on delivery, False on failure, and never raises
# --------------------------------------------------------------------------


def _send_webhook(payload: Dict[str, Any]) -> bool:
    url = getattr(settings, "alert_webhook_url", "") or ""
    try:
        status = _http_post_json(url, payload)
        if 200 <= status < 300:
            return True
        logger.warning("Webhook alert for %s returned HTTP %s", payload.get("site_id"), status)
        return False
    except Exception as exc:
        logger.warning("Webhook alert for %s failed: %s", payload.get("site_id"), exc)
        return False


def _send_telegram(payload: Dict[str, Any]) -> bool:
    token = getattr(settings, "telegram_bot_token", "") or ""
    chat_id = getattr(settings, "telegram_chat_id", "") or ""
    try:
        status = _http_post_json(
            f"https://api.telegram.org/bot{token}/sendMessage",
            {"chat_id": chat_id, "text": payload.get("text", ""), "disable_web_page_preview": False},
        )
        if 200 <= status < 300:
            return True
        logger.warning("Telegram alert for %s returned HTTP %s", payload.get("site_id"), status)
        return False
    except Exception as exc:
        logger.warning("Telegram alert for %s failed: %s", payload.get("site_id"), exc)
        return False


def _send_email(payload: Dict[str, Any]) -> bool:
    recipients = [a.strip() for a in str(getattr(settings, "alert_email_to", "") or "").split(",") if a.strip()]
    if not recipients:
        return False
    try:
        msg = EmailMessage()
        subject_level = "ALL CLEAR" if payload.get("event") == "ALL_CLEAR" else f"{payload.get('alert_level')} ALERT"
        msg["Subject"] = f"[MHEWS {subject_level}] {payload.get('site_name')} ({payload.get('site_id')})"
        msg["From"] = getattr(settings, "alert_email_from", "mhews@localhost") or "mhews@localhost"
        msg["To"] = ", ".join(recipients)
        msg.set_content(payload.get("text", ""))
        _smtp_send(
            host=getattr(settings, "alert_smtp_host", ""),
            port=getattr(settings, "alert_smtp_port", 587),
            user=getattr(settings, "alert_smtp_user", "") or "",
            password=getattr(settings, "alert_smtp_password", "") or "",
            message=msg,
        )
        return True
    except Exception as exc:
        logger.warning("Email alert for %s failed: %s", payload.get("site_id"), exc)
        return False


def _configured_channels() -> List[Tuple[str, Callable[[Dict[str, Any]], bool]]]:
    """Channels with complete credentials, in delivery order."""
    channels: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = []
    if getattr(settings, "alert_webhook_url", ""):
        channels.append(("webhook", _send_webhook))
    if getattr(settings, "telegram_bot_token", "") and getattr(settings, "telegram_chat_id", ""):
        channels.append(("telegram", _send_telegram))
    if getattr(settings, "alert_smtp_host", "") and getattr(settings, "alert_email_to", ""):
        channels.append(("email", _send_email))
    return channels


# --------------------------------------------------------------------------
# Notifier
# --------------------------------------------------------------------------


class EmergencyAlertNotifier:
    """Transition-aware, restart-safe dispatcher for site alert levels.

    State lives in a small JSON file so that restarting the orchestrator does
    not replay alerts already delivered. For a site seen for the first time the
    previous level is assumed GREEN, so a first-ever run does alert on sites
    that are already ORANGE/RED — which is the safe direction.
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.state_path = Path(state_path) if state_path else DEFAULT_STATE_PATH
        self.clock = clock
        self._state: Dict[str, Any] = self._load_state()

    # -- state ------------------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("sites"), dict):
                return data
            logger.warning("Alert state file %s has unexpected shape; starting fresh", self.state_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Could not read alert state %s (%s); starting fresh", self.state_path, exc)
        return {"version": STATE_SCHEMA_VERSION, "sites": {}}

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2)
            os.replace(tmp, self.state_path)
        except Exception as exc:
            logger.warning("Could not persist alert state to %s: %s", self.state_path, exc)

    # -- decision ---------------------------------------------------------

    def _classify(self, site_id: str, level: str, now: float, min_rank: int) -> Tuple[Optional[str], str]:
        """Update stored state for the site and decide whether it owes a message."""
        sites: Dict[str, Any] = self._state.setdefault("sites", {})
        entry: Dict[str, Any] = sites.setdefault(site_id, {})

        stored_level = entry.get("level")
        if stored_level != level:
            entry["prev_level"] = stored_level if stored_level in LEVEL_ORDER else ALL_CLEAR_LEVEL
            entry["level"] = level
            entry["level_since_utc"] = now
        entry.setdefault("prev_level", ALL_CLEAR_LEVEL)
        entry.setdefault("level_since_utc", now)
        notified: Dict[str, Any] = entry.setdefault("notified", {})

        cur_rank = _level_rank(level)
        prev_rank = _level_rank(entry["prev_level"])

        if cur_rank >= min_rank and cur_rank > prev_rank:
            event: Optional[str] = "ESCALATION"
        elif level == ALL_CLEAR_LEVEL and prev_rank >= min_rank:
            event = "ALL_CLEAR"
        else:
            return None, "no_transition"

        # Already delivered for this occupancy of the level: do not re-fire while it persists.
        last_sent = float(notified.get(level, 0.0) or 0.0)
        if last_sent >= float(entry["level_since_utc"]):
            return None, "already_notified"

        cooldown_s = max(0, int(getattr(settings, "alert_cooldown_minutes", 180))) * 60
        if cooldown_s and (now - last_sent) < cooldown_s:
            return None, "cooldown"

        return event, "fire"

    def _mark_notified(self, site_id: str, level: str, now: float) -> None:
        self._state["sites"][site_id].setdefault("notified", {})[level] = now
        self._state["sites"][site_id]["last_notified_utc"] = now

    # -- dispatch ---------------------------------------------------------

    def dispatch(self, site_evaluations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Deliver messages for level transitions. Never raises."""
        summary: Dict[str, Any] = {
            "enabled": bool(getattr(settings, "alerts_enabled", False)),
            "evaluated": 0,
            "transitions": 0,
            "sent": 0,
            "suppressed": 0,
            "failed": 0,
            "all_clear": 0,
            "channels": {},
        }
        if not summary["enabled"]:
            return summary

        try:
            channels = _configured_channels()
            min_rank = _min_level_rank()
            evaluations = list(site_evaluations or [])
            summary["evaluated"] = len(evaluations)

            if not channels:
                # Keep tracking levels so that configuring a channel later does not
                # replay a backlog, but say plainly that nothing can be delivered.
                for ev in evaluations:
                    site_id = str(ev.get("site_id") or "")
                    if site_id:
                        event, _ = self._classify(site_id, str(ev.get("alert", "GREEN")).upper(), self.clock(), min_rank)
                        if event:
                            summary["transitions"] += 1
                            summary["suppressed"] += 1
                self._save_state()
                summary["note"] = "alerts enabled but no channel is configured"
                logger.warning("Alerting is enabled but no channel is configured; %d transition(s) undeliverable", summary["suppressed"])
                return summary

            for ev in evaluations:
                try:
                    site_id = str(ev.get("site_id") or "")
                    if not site_id:
                        continue
                    level = str(ev.get("alert", "GREEN")).upper()
                    now = self.clock()
                    event, reason = self._classify(site_id, level, now, min_rank)
                    if not event:
                        if reason == "cooldown":
                            summary["transitions"] += 1
                            summary["suppressed"] += 1
                        continue

                    summary["transitions"] += 1
                    prev_level = self._state["sites"][site_id].get("prev_level", ALL_CLEAR_LEVEL)
                    payload = build_alert_payload(ev, event, prev_level, now_utc=now)

                    delivered = 0
                    for name, send in channels:
                        try:
                            ok = bool(send(payload))
                        except Exception as exc:  # a channel must never break the others
                            logger.warning("Alert channel %s raised for %s: %s", name, site_id, exc)
                            ok = False
                        if ok:
                            delivered += 1
                            summary["channels"][name] = summary["channels"].get(name, 0) + 1

                    if delivered:
                        summary["sent"] += 1
                        if event == "ALL_CLEAR":
                            summary["all_clear"] += 1
                        self._mark_notified(site_id, level, now)
                        logger.info(
                            "Dispatched %s for %s (%s -> %s) on %d/%d channel(s)",
                            event, site_id, prev_level, level, delivered, len(channels),
                        )
                    else:
                        # Not marked as notified, so the next cycle retries while the level holds.
                        summary["failed"] += 1
                        logger.error("All %d channel(s) failed for %s %s", len(channels), site_id, event)
                except Exception as exc:
                    summary["failed"] += 1
                    logger.exception("Alert dispatch failed for site %s: %s", ev.get("site_id"), exc)

            self._save_state()
        except Exception as exc:
            summary["error"] = str(exc)
            logger.exception("Alert dispatch aborted: %s", exc)
        return summary


_default_notifier: Optional[EmergencyAlertNotifier] = None


def dispatch_alerts(site_evaluations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Dispatch emergency alerts for the current cycle's site evaluations.

    Safe to call unconditionally: a no-op returning zero counts when alerting is
    disabled, and it never raises — an alerting bug must not stop the physics
    engine. Returns a summary dict with ``sent`` / ``suppressed`` / ``failed``
    counts for the orchestrator to log.
    """
    global _default_notifier
    try:
        if not getattr(settings, "alerts_enabled", False):
            return {
                "enabled": False,
                "evaluated": 0,
                "transitions": 0,
                "sent": 0,
                "suppressed": 0,
                "failed": 0,
                "all_clear": 0,
                "channels": {},
            }
        if _default_notifier is None:
            _default_notifier = EmergencyAlertNotifier()
        return _default_notifier.dispatch(site_evaluations)
    except Exception as exc:
        logger.exception("dispatch_alerts failed: %s", exc)
        return {
            "enabled": True,
            "evaluated": 0,
            "transitions": 0,
            "sent": 0,
            "suppressed": 0,
            "failed": 0,
            "all_clear": 0,
            "channels": {},
            "error": str(exc),
        }
