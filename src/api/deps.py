"""
FastAPI dependencies (shared across routes).

Provides:
    get_db      – yields a SQLAlchemy session per request
    get_detector – returns the cached YOLOv8 detector singleton
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from src.database.db import SessionLocal
from src.detection.infer import Detector, get_detector as _get_detector


# ---------------------------------------------------------------------------
# Database session dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy DB session.

    Automatically closed after each request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Detector dependency
# ---------------------------------------------------------------------------

def get_detector() -> Detector:
    """
    FastAPI dependency that returns the cached YOLOv8 Detector.

    The model is loaded once at startup (via the lifespan hook in main.py)
    and reused for all requests.
    """
    return _get_detector()


# ---------------------------------------------------------------------------
# Type aliases for cleaner route signatures
# ---------------------------------------------------------------------------

DbDep = Annotated[Session, Depends(get_db)]
DetectorDep = Annotated[Detector, Depends(get_detector)]
