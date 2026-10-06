"""
Notification stubs for the alert system.

For the prototype, "notifications" are just log messages.
Replace the stub implementations with real email/SMS/webhook calls
in production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.database.models import Severity

logger = logging.getLogger(__name__)


@dataclass
class NotificationPayload:
    pothole_id: int
    severity: str
    department: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    note: str = ""


def send_notification(payload: NotificationPayload) -> bool:
    """
    Send an alert notification.

    Prototype: logs to stdout.
    Production: swap with email/SMS/webhook/push-notification.

    Returns True if the notification was sent successfully.
    """
    logger.warning(
        "[ALERT] Pothole #%d | Severity: %s | Dept: %s | Note: %s",
        payload.pothole_id,
        payload.severity,
        payload.department,
        payload.note,
    )
    # TODO: implement real notification (email, webhook, etc.)
    return True


def maybe_notify(
    pothole_id: int,
    severity: Severity,
    department: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> None:
    """
    Send a notification only for critical and high severity defects.

    Silently ignores medium/low — they are handled via scheduled reports.
    """
    if severity.value in ("critical", "high"):
        payload = NotificationPayload(
            pothole_id=pothole_id,
            severity=severity.value,
            department=department,
            lat=lat,
            lon=lon,
            note=f"Road defect requires attention: severity={severity.value}",
        )
        send_notification(payload)
