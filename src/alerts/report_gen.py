"""
Incident report generation.

Generates structured text / JSON reports summarising:
    - All potholes detected in a scan
    - Growth trends over multiple scans
    - Open incidents by severity

For the prototype, reports are returned as Python dicts (serialisable to JSON).
A PDF/CSV export layer can be added later.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.database import crud
from src.database.models import Incident, Pothole, PotholeHistory, Scan


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def build_scan_report(db: Session, scan_id: int) -> Dict[str, Any]:
    """
    Generate a summary report for a completed scan.

    Returns a dict with:
        - scan metadata
        - pothole counts by class and severity
        - list of all potholes detected in this scan
    """
    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise ValueError(f"Scan {scan_id} not found")

    # Fetch PotholeHistory rows for this scan
    histories: List[PotholeHistory] = (
        db.query(PotholeHistory)
        .filter(PotholeHistory.scan_id == scan_id)
        .all()
    )

    pothole_ids = {h.pothole_id for h in histories}
    potholes: List[Pothole] = [crud.get_pothole(db, pid) for pid in pothole_ids if pid]

    summary = {
        "report_generated_at": datetime.utcnow().isoformat(),
        "scan_id": scan.id,
        "camera_id": scan.camera_id,
        "source_path": scan.source_path,
        "scan_timestamp": scan.timestamp.isoformat(),
        "scan_status": scan.status.value,
        "total_detections": len(histories),
        "potholes_detected": len(potholes),
        "new_potholes": sum(1 for p in potholes if p and p.status.value == "new"),
        "existing_potholes": sum(1 for p in potholes if p and p.status.value == "existing"),
        "by_class": {
            "pothole": sum(1 for p in potholes if p and p.class_type.value == "pothole"),
            "crack": sum(1 for p in potholes if p and p.class_type.value == "crack"),
        },
        "detections": [
            {
                "pothole_id": h.pothole_id,
                "area_px2": h.area,
                "confidence": h.confidence,
                "timestamp": h.timestamp.isoformat(),
            }
            for h in histories
        ],
    }
    return summary


def build_pothole_growth_report(db: Session, pothole_id: int) -> Dict[str, Any]:
    """
    Generate a growth timeline report for a single pothole.

    Returns area measurements over time, enabling trend analysis.
    """
    p = crud.get_pothole(db, pothole_id)
    if not p:
        raise ValueError(f"Pothole {pothole_id} not found")

    history = crud.get_pothole_history(db, pothole_id)

    areas = [h.area for h in history if h.area is not None]
    growth_pct: Optional[float] = None
    if len(areas) >= 2:
        growth_pct = round((areas[-1] - areas[0]) / max(areas[0], 1.0) * 100, 2)

    return {
        "report_generated_at": datetime.utcnow().isoformat(),
        "pothole_id": pothole_id,
        "class_type": p.class_type.value,
        "status": p.status.value,
        "first_detected": p.first_detected.isoformat(),
        "last_seen": p.last_seen.isoformat(),
        "initial_area_px2": areas[0] if areas else None,
        "latest_area_px2": areas[-1] if areas else None,
        "growth_pct": growth_pct,
        "scan_count": len(history),
        "timeline": [
            {
                "scan_id": h.scan_id,
                "area_px2": h.area,
                "confidence": h.confidence,
                "timestamp": h.timestamp.isoformat(),
            }
            for h in history
        ],
    }


def build_open_incidents_report(db: Session) -> Dict[str, Any]:
    """
    Generate a report of all open (pending/reviewed/assigned) incidents.
    """
    open_incidents: List[Incident] = crud.list_incidents(
        db, status=None, limit=1000
    )
    open_incidents = [
        i for i in open_incidents if i.status.value in ("pending", "reviewed", "assigned")
    ]

    by_severity: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for inc in open_incidents:
        sev = inc.severity.value
        by_severity[sev] = by_severity.get(sev, 0) + 1

    return {
        "report_generated_at": datetime.utcnow().isoformat(),
        "total_open_incidents": len(open_incidents),
        "by_severity": by_severity,
        "incidents": [
            {
                "incident_id": i.id,
                "pothole_id": i.pothole_id,
                "severity": i.severity.value,
                "department": i.department,
                "status": i.status.value,
                "created_at": i.created_at.isoformat(),
            }
            for i in open_incidents
        ],
    }
