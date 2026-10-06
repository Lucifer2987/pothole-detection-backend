"""
Report endpoints.

GET /reports/scan/{scan_id}          – summary report for one scan
GET /reports/pothole/{pothole_id}    – growth timeline for one pothole
GET /reports/incidents/open          – all open incidents dashboard
"""

from fastapi import APIRouter, HTTPException

from src.alerts.report_gen import (
    build_open_incidents_report,
    build_pothole_growth_report,
    build_scan_report,
)
from src.api.deps import DbDep

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/scan/{scan_id}")
def scan_report(scan_id: int, db: DbDep):
    """Return a JSON summary report for a completed scan."""
    try:
        return build_scan_report(db, scan_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/pothole/{pothole_id}")
def pothole_growth_report(pothole_id: int, db: DbDep):
    """Return a growth timeline report for a single pothole."""
    try:
        return build_pothole_growth_report(db, pothole_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/incidents/open")
def open_incidents_report(db: DbDep):
    """Return an overview of all open maintenance incidents."""
    return build_open_incidents_report(db)
