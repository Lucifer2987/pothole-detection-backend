"""
Alert routing logic.

Determines which department/team should be notified for a given pothole
severity level, and optionally triggers downstream notification hooks.

For the prototype, this is a simple rules-based router.  In production,
swap the routing table for a database-driven department config or an
external dispatch API call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.database.models import Severity


# ---------------------------------------------------------------------------
# Routing table  (severity → department)
# ---------------------------------------------------------------------------

DEFAULT_ROUTING: Dict[str, str] = {
    Severity.critical.value: "Emergency Road Repair Unit",
    Severity.high.value:     "Priority Maintenance Team",
    Severity.medium.value:   "Roads & Bridges Department",
    Severity.low.value:      "Scheduled Maintenance Queue",
}


@dataclass
class RouteDecision:
    severity: str
    department: str
    should_notify: bool
    note: str


def route_alert(
    severity: Severity,
    routing_table: Optional[Dict[str, str]] = None,
) -> RouteDecision:
    """
    Decide which department should handle a pothole of a given severity.

    Args:
        severity:      The Severity enum value for the pothole / incident.
        routing_table: Optional override mapping.  Keys are severity strings,
                       values are department names.

    Returns:
        RouteDecision with department and whether an immediate notification
        should be sent.
    """
    table = routing_table or DEFAULT_ROUTING
    sev_str = severity.value if isinstance(severity, Severity) else str(severity)
    department = table.get(sev_str, DEFAULT_ROUTING[Severity.low.value])
    should_notify = sev_str in {Severity.critical.value, Severity.high.value}

    note = (
        "IMMEDIATE ACTION REQUIRED — critical road defect detected."
        if sev_str == Severity.critical.value
        else f"Pothole severity: {sev_str}. Assigned to {department}."
    )

    return RouteDecision(
        severity=sev_str,
        department=department,
        should_notify=should_notify,
        note=note,
    )


def assign_department_for_incident(
    severity: Severity,
    routing_table: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convenience wrapper: return just the department name string.
    Used by the scan pipeline when auto-creating incidents.
    """
    decision = route_alert(severity, routing_table)
    return decision.department
