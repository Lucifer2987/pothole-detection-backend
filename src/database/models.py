"""
SQLAlchemy ORM models for the Pothole Detection system.

Tables:
    Camera         – registered cameras / scan locations
    Scan           – a single processing run against a video or image file
    Pothole        – a unique road defect instance tracked over time
    PotholeHistory – per-scan snapshot of a pothole (enables growth timeline)
    Incident       – maintenance/dispatch record linked to a pothole
"""

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base — all models inherit from this."""
    pass


# ---------------------------------------------------------------------------
# Enum definitions  (stored as VARCHAR in Postgres)
# ---------------------------------------------------------------------------

class ScanStatus(str, enum.Enum):
    pending    = "pending"
    processing = "processing"
    completed  = "completed"
    failed     = "failed"


class ClassType(str, enum.Enum):
    pothole = "pothole"
    crack   = "crack"


class PotholeStatus(str, enum.Enum):
    new      = "new"
    existing = "existing"
    resolved = "resolved"


class Severity(str, enum.Enum):
    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"


class IncidentStatus(str, enum.Enum):
    pending  = "pending"
    reviewed = "reviewed"
    assigned = "assigned"
    resolved = "resolved"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera(Base):
    """
    A physical camera or scan location.

    roi_coords stores an optional region-of-interest as a JSON string, e.g.:
        '[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]'
    """
    __tablename__ = "cameras"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(255), nullable=False)
    lat        = Column(Float, nullable=True,  doc="Latitude of camera location")
    lon        = Column(Float, nullable=True,  doc="Longitude of camera location")
    roi_coords = Column(Text,  nullable=True,  doc="JSON string of ROI polygon vertices")
    is_active  = Column(Boolean, default=True, nullable=False)

    # Relationships
    scans    = relationship("Scan",    back_populates="camera", cascade="all, delete-orphan")
    potholes = relationship("Pothole", back_populates="camera")

    def __repr__(self) -> str:
        return f"<Camera id={self.id} name={self.name!r} active={self.is_active}>"


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

class Scan(Base):
    """
    One processing run — ties a source file (video or image) to a camera.

    source_path: absolute or repo-relative path to the input file used,
                 e.g. 'data/raw/own_collected/road_clip_01.mp4'
    """
    __tablename__ = "scans"

    id          = Column(Integer, primary_key=True, index=True)
    camera_id   = Column(Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    timestamp   = Column(DateTime, default=datetime.utcnow, nullable=False)
    source_path = Column(Text, nullable=False,  doc="Path to the video/image file processed")
    status      = Column(
        Enum(ScanStatus, name="scan_status"),
        default=ScanStatus.pending,
        nullable=False,
    )

    # Relationships
    camera           = relationship("Camera", back_populates="scans")
    pothole_histories = relationship("PotholeHistory", back_populates="scan")

    def __repr__(self) -> str:
        return f"<Scan id={self.id} status={self.status} ts={self.timestamp}>"


# ---------------------------------------------------------------------------
# Pothole
# ---------------------------------------------------------------------------

class Pothole(Base):
    """
    A unique road defect instance, tracked across multiple scans.

    lat / lon are nullable — they're only populated when the camera has known
    GPS coordinates and a geo-projection is done.

    current_area is updated on every scan where this defect is seen, enabling
    growth tracking via PotholeHistory.
    """
    __tablename__ = "potholes"

    id             = Column(Integer, primary_key=True, index=True)
    camera_id      = Column(Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    first_detected = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen      = Column(DateTime, default=datetime.utcnow, nullable=False)
    lat            = Column(Float, nullable=True,  doc="Estimated geo-latitude of defect")
    lon            = Column(Float, nullable=True,  doc="Estimated geo-longitude of defect")
    class_type     = Column(
        Enum(ClassType, name="class_type"),
        nullable=False,
        doc="Defect class: 'pothole' or 'crack'",
    )
    current_area   = Column(Float, nullable=True,  doc="Latest bounding-box pixel area (px²)")
    status         = Column(
        Enum(PotholeStatus, name="pothole_status"),
        default=PotholeStatus.new,
        nullable=False,
    )

    # Relationships
    camera    = relationship("Camera",         back_populates="potholes")
    history   = relationship("PotholeHistory", back_populates="pothole", cascade="all, delete-orphan")
    incidents = relationship("Incident",       back_populates="pothole", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<Pothole id={self.id} type={self.class_type} "
            f"status={self.status} area={self.current_area}>"
        )


# ---------------------------------------------------------------------------
# PotholeHistory
# ---------------------------------------------------------------------------

class PotholeHistory(Base):
    """
    Per-scan snapshot of a pothole.

    This is the append-only table that drives growth timeline queries:
        SELECT * FROM pothole_histories WHERE pothole_id = X ORDER BY timestamp;

    area:       bounding-box pixel area at the time of this scan
    confidence: YOLO detection confidence score (0.0 – 1.0)
    """
    __tablename__ = "pothole_histories"

    id         = Column(Integer, primary_key=True, index=True)
    pothole_id = Column(Integer, ForeignKey("potholes.id", ondelete="CASCADE"), nullable=False, index=True)
    scan_id    = Column(Integer, ForeignKey("scans.id",   ondelete="SET NULL"), nullable=True,  index=True)
    area       = Column(Float,   nullable=True,  doc="Bounding-box pixel area (px²)")
    confidence = Column(Float,   nullable=True,  doc="YOLO detection confidence 0–1")
    timestamp  = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    pothole = relationship("Pothole", back_populates="history")
    scan    = relationship("Scan",    back_populates="pothole_histories")

    def __repr__(self) -> str:
        return (
            f"<PotholeHistory id={self.id} pothole_id={self.pothole_id} "
            f"area={self.area} conf={self.confidence}>"
        )


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

class Incident(Base):
    """
    A maintenance / dispatch record generated from a high-severity pothole.

    department: free-text name of the responsible municipal department,
                e.g. 'PWD Zone 3' or 'Roads & Bridges Dept'.
    """
    __tablename__ = "incidents"

    id          = Column(Integer, primary_key=True, index=True)
    pothole_id  = Column(Integer, ForeignKey("potholes.id", ondelete="CASCADE"), nullable=False, index=True)
    severity    = Column(
        Enum(Severity, name="severity"),
        nullable=False,
    )
    department  = Column(String(255), nullable=True,  doc="Responsible municipal department")
    status      = Column(
        Enum(IncidentStatus, name="incident_status"),
        default=IncidentStatus.pending,
        nullable=False,
    )
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    pothole = relationship("Pothole", back_populates="incidents")

    def __repr__(self) -> str:
        return (
            f"<Incident id={self.id} pothole_id={self.pothole_id} "
            f"severity={self.severity} status={self.status}>"
        )
