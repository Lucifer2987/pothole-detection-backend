"""
Incident (maintenance dispatch) routes.

Endpoints:
    POST   /incidents/              – create a new incident
    GET    /incidents/              – list incidents (filterable)
    GET    /incidents/{id}          – get a single incident
    PATCH  /incidents/{id}/status   – update incident status
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, status

from src.api.deps import DbDep
from src.api.schemas import IncidentCreate, IncidentOut, IncidentStatusUpdate
from src.database import crud
from src.database.models import IncidentStatus

router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.post("/", response_model=IncidentOut, status_code=status.HTTP_201_CREATED)
def create_incident(body: IncidentCreate, db: DbDep):
    """Create a maintenance incident for a pothole."""
    # Verify the pothole exists
    p = crud.get_pothole(db, body.pothole_id)
    if not p:
        raise HTTPException(status_code=404, detail=f"Pothole {body.pothole_id} not found")
    return crud.create_incident(
        db,
        pothole_id=body.pothole_id,
        severity=body.severity,
        department=body.department,
    )


@router.get("/", response_model=list[IncidentOut])
def list_incidents(
    db: DbDep,
    pothole_id: Optional[int] = None,
    status: Optional[IncidentStatus] = None,
    skip: int = 0,
    limit: int = 100,
):
    """List all incidents, optionally filtered by pothole or status."""
    return crud.list_incidents(
        db,
        pothole_id=pothole_id,
        status=status,
        skip=skip,
        limit=limit,
    )


@router.get("/{incident_id}", response_model=IncidentOut)
def get_incident(incident_id: int, db: DbDep):
    """Return a single incident record."""
    inc = crud.get_incident(db, incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc


@router.patch("/{incident_id}/status", response_model=IncidentOut)
def update_incident_status(incident_id: int, body: IncidentStatusUpdate, db: DbDep):
    """Update the status of an incident (e.g. reviewed, assigned, resolved)."""
    inc = crud.update_incident_status(db, incident_id, body.status)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc
