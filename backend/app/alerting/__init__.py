"""Emergency alert transport for Himalayan-MHEWS.

The physics engine decides the alert level; this package only delivers it to
humans (webhook, Telegram, email) when a site crosses into or out of an
alerting level.
"""

from app.alerting.notifier import (  # noqa: F401
    EmergencyAlertNotifier,
    build_alert_payload,
    dispatch_alerts,
    render_alert_text,
)

__all__ = [
    "EmergencyAlertNotifier",
    "build_alert_payload",
    "dispatch_alerts",
    "render_alert_text",
]
