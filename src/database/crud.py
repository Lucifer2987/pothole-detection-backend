"""
CRUD (Create, Read, Update, Delete) operations for all database models.

All functions accept a SQLAlchemy Session and return ORM objects or lists.
No business logic lives here — only database I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from src.database.models import (
    Camera,
    ClassType,
    Incident,
    IncidentStatus,
    Pothole,
    PotholeHistory,
    PotholeStatus,
    Scan,
    ScanStatus,
    Severity,
)


# ===========================================================================
# Camera
# ===========================================================================

def create_camera(
    db: Session,
    name: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    roi_coords: Optional[str] = None,
    is_active: bool = True,
) -> Camera:
    cam = Camera(name=name, lat=lat, lon=lon, roi_coords=roi_coords, is_active=is_active)
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


def get_camera(db: Session, camera_id: int) -> Optional[Camera]:
    return db.query(Camera).filter(Camera.id == camera_id).first()


def list_cameras(db: Session, skip: int = 0, limit: int = 100) -> List[Camera]:
    return db.query(Camera).offset(skip).limit(limit).all()


def update_camera(
    db: Session,
    camera_id: int,
    **kwargs,
) -> Optional[Camera]:
    cam = get_camera(db, camera_id)
    if not cam:
        return None
    for key, value in kwargs.items():
        if hasattr(cam, key) and value is not None:
            setattr(cam, key, value)
    db.commit()
    db.refresh(cam)
    return cam


def delete_camera(db: Session, camera_id: int) -> bool:
    cam = get_camera(db, camera_id)
    if not cam:
        return False
    db.delete(cam)
    db.commit()
    return True


# ===========================================================================
# Scan
# ===========================================================================

def create_scan(
    db: Session,
    source_path: str,
    camera_id: Optional[int] = None,
    status: ScanStatus = ScanStatus.pending,
) -> Scan:
    scan = Scan(
        camera_id=camera_id,
        source_path=source_path,
        status=status,
        timestamp=datetime.utcnow(),
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


def get_scan(db: Session, scan_id: int) -> Optional[Scan]:
    return db.query(Scan).filter(Scan.id == scan_id).first()


def list_scans(
    db: Session,
    camera_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Scan]:
    q = db.query(Scan)
    if camera_id is not None:
        q = q.filter(Scan.camera_id == camera_id)
    return q.order_by(Scan.timestamp.desc()).offset(skip).limit(limit).all()


def update_scan_status(
    db: Session,
    scan_id: int,
    status: ScanStatus,
) -> Optional[Scan]:
    scan = get_scan(db, scan_id)
    if not scan:
        return None
    scan.status = status
    db.commit()
    db.refresh(scan)
    return scan


# ===========================================================================
# Pothole
# ===========================================================================

def get_pothole(db: Session, pothole_id: int) -> Optional[Pothole]:
    return db.query(Pothole).filter(Pothole.id == pothole_id).first()


def list_potholes(
    db: Session,
    camera_id: Optional[int] = None,
    status: Optional[PotholeStatus] = None,
    class_type: Optional[ClassType] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Pothole]:
    q = db.query(Pothole)
    if camera_id is not None:
        q = q.filter(Pothole.camera_id == camera_id)
    if status is not None:
        q = q.filter(Pothole.status == status)
    if class_type is not None:
        q = q.filter(Pothole.class_type == class_type)
    return q.order_by(Pothole.first_detected.desc()).offset(skip).limit(limit).all()


def update_pothole_status(
    db: Session,
    pothole_id: int,
    status: PotholeStatus,
) -> Optional[Pothole]:
    p = get_pothole(db, pothole_id)
    if not p:
        return None
    p.status = status
    db.commit()
    db.refresh(p)
    return p


def get_pothole_history(
    db: Session,
    pothole_id: int,
) -> List[PotholeHistory]:
    return (
        db.query(PotholeHistory)
        .filter(PotholeHistory.pothole_id == pothole_id)
        .order_by(PotholeHistory.timestamp.asc())
        .all()
    )


# ===========================================================================
# Incident
# ===========================================================================

def create_incident(
    db: Session,
    pothole_id: int,
    severity: Severity,
    department: Optional[str] = None,
) -> Incident:
    incident = Incident(
        pothole_id=pothole_id,
        severity=severity,
        department=department,
        status=IncidentStatus.pending,
        created_at=datetime.utcnow(),
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


def get_incident(db: Session, incident_id: int) -> Optional[Incident]:
    return db.query(Incident).filter(Incident.id == incident_id).first()


def list_incidents(
    db: Session,
    pothole_id: Optional[int] = None,
    status: Optional[IncidentStatus] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[Incident]:
    q = db.query(Incident)
    if pothole_id is not None:
        q = q.filter(Incident.pothole_id == pothole_id)
    if status is not None:
        q = q.filter(Incident.status == status)
    return q.order_by(Incident.created_at.desc()).offset(skip).limit(limit).all()


def update_incident_status(
    db: Session,
    incident_id: int,
    status: IncidentStatus,
) -> Optional[Incident]:
    inc = get_incident(db, incident_id)
    if not inc:
        return None
    inc.status = status
    db.commit()
    db.refresh(inc)
    return inc
