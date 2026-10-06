"""
Scan history routes.

Endpoints:
    GET /scans/          – list scan records
    GET /scans/{id}      – get a single scan record
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from src.api.deps import DbDep
from src.api.schemas import ScanOut
from src.database import crud

router = APIRouter(prefix="/scans", tags=["scans"])


@router.get("/", response_model=list[ScanOut])
def list_scans(
    db: DbDep,
    camera_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
):
    """Return scan history records, optionally filtered by camera."""
    return crud.list_scans(db, camera_id=camera_id, skip=skip, limit=limit)


@router.get("/{scan_id}", response_model=ScanOut)
def get_scan(scan_id: int, db: DbDep):
    """Return a single scan record."""
    scan = crud.get_scan(db, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan
