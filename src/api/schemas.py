"""
Pydantic schemas for request validation and response serialization.

One schema group per model:
    Camera, Scan, Pothole, PotholeHistory, Incident

Naming convention:
    <Model>Create  – request body for POST endpoints
    <Model>Update  – request body for PATCH endpoints (all fields optional)
    <Model>Out     – response body (includes id and computed fields)
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from src.database.models import (
    ClassType,
    IncidentStatus,
    PotholeStatus,
    ScanStatus,
    Severity,
)


# ===========================================================================
# Camera
# ===========================================================================

class CameraCreate(BaseModel):
    name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    roi_coords: Optional[str] = None
    is_active: bool = True


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    roi_coords: Optional[str] = None
    is_active: Optional[bool] = None


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    lat: Optional[float]
    lon: Optional[float]
    roi_coords: Optional[str]
    is_active: bool


# ===========================================================================
# Scan
# ===========================================================================

class ScanCreate(BaseModel):
    camera_id: Optional[int] = None
    source_path: str


class ScanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: Optional[int]
    source_path: str
    status: ScanStatus
    timestamp: datetime


# ===========================================================================
# PotholeHistory
# ===========================================================================

class PotholeHistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    pothole_id: int
    scan_id: Optional[int]
    area: Optional[float]
    confidence: Optional[float]
    timestamp: datetime


# ===========================================================================
# Pothole
# ===========================================================================

class PotholeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: Optional[int]
    first_detected: datetime
    last_seen: datetime
    lat: Optional[float]
    lon: Optional[float]
    class_type: ClassType
    current_area: Optional[float]
    status: PotholeStatus


class PotholeDetailOut(PotholeOut):
    """Pothole with full history timeline."""
    history: List[PotholeHistoryOut] = []


class PotholeStatusUpdate(BaseModel):
    status: PotholeStatus


# ===========================================================================
# Incident
# ===========================================================================

class IncidentCreate(BaseModel):
    pothole_id: int
    severity: Severity
    department: Optional[str] = None


class IncidentStatusUpdate(BaseModel):
    status: IncidentStatus


class IncidentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    pothole_id: int
    severity: Severity
    department: Optional[str]
    status: IncidentStatus
    created_at: datetime


# ===========================================================================
# Upload / Scan trigger response
# ===========================================================================

class ScanResult(BaseModel):
    """Response returned when a scan completes."""
    scan_id: int
    camera_id: Optional[int]
    source_path: str
    status: ScanStatus
    total_detections: int
    new_potholes: int
    existing_potholes: int
    growing_potholes: int
    detections: List[dict] = []


# ===========================================================================
# Health check
# ===========================================================================

class HealthOut(BaseModel):
    status: str
    db_reachable: bool
    model_loaded: bool
