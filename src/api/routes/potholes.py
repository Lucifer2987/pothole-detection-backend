"""
Pothole read and status-update routes.

Endpoints:
    GET   /potholes/              – list potholes (filterable)
    GET   /potholes/{id}          – get a pothole + its history timeline
    PATCH /potholes/{id}/status   – mark a pothole as resolved / existing / new
    GET   /potholes/{id}/history  – growth timeline for a single pothole
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from src.api.deps import DbDep
from src.api.schemas import PotholeDetailOut, PotholeHistoryOut, PotholeOut, PotholeStatusUpdate
from src.database import crud
from src.database.models import ClassType, PotholeStatus

router = APIRouter(prefix="/potholes", tags=["potholes"])


@router.get("/", response_model=list[PotholeOut])
def list_potholes(
    db: DbDep,
    camera_id: Optional[int] = None,
    status: Optional[PotholeStatus] = None,
    class_type: Optional[ClassType] = None,
    skip: int = 0,
    limit: int = 100,
):
    """
    List all known potholes.

    Query parameters:
    - **camera_id**: filter by camera
    - **status**: new | existing | resolved
    - **class_type**: pothole | crack
    """
    return crud.list_potholes(
        db,
        camera_id=camera_id,
        status=status,
        class_type=class_type,
        skip=skip,
        limit=limit,
    )


@router.get("/{pothole_id}", response_model=PotholeDetailOut)
def get_pothole(pothole_id: int, db: DbDep):
    """Return a pothole record including its full growth history."""
    p = crud.get_pothole(db, pothole_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pothole not found")
    return p


@router.patch("/{pothole_id}/status", response_model=PotholeOut)
def update_pothole_status(pothole_id: int, body: PotholeStatusUpdate, db: DbDep):
    """Update the lifecycle status of a pothole (e.g. mark as resolved)."""
    p = crud.update_pothole_status(db, pothole_id, body.status)
    if not p:
        raise HTTPException(status_code=404, detail="Pothole not found")
    return p


@router.get("/{pothole_id}/history", response_model=list[PotholeHistoryOut])
def get_pothole_history(pothole_id: int, db: DbDep):
    """Return the chronological growth timeline for a single pothole."""
    p = crud.get_pothole(db, pothole_id)
    if not p:
        raise HTTPException(status_code=404, detail="Pothole not found")
    return crud.get_pothole_history(db, pothole_id)
