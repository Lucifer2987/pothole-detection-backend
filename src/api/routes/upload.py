"""
File upload and scan-trigger endpoint.

POST /upload
    – Accept a local file path OR a multipart file upload
    – Run the full detection pipeline (infer → change detect → DB write)
    – Return a ScanResult summary

POST /scan
    – Accept a local file path (JSON body) for direct server-side scanning
      (useful when the file is already on the server, e.g. from CCTV storage)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.deps import DbDep, DetectorDep
from src.api.schemas import ScanResult
from src.database import crud
from src.database.models import ScanStatus, Severity
from src.tracking.change_detector import ChangeDetector

router = APIRouter(tags=["scanning"])

# Directory where uploaded files are stored temporarily
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared scan pipeline
# ---------------------------------------------------------------------------

def _run_scan_pipeline(
    db,
    detector,
    source_path: str,
    camera_id: Optional[int],
    is_video: bool,
) -> ScanResult:
    """
    Core scan pipeline:
        1. Create Scan record (pending)
        2. Run detection (video or image)
        3. Run change detection → update/create Pothole rows
        4. Auto-create incidents for critical/high potholes
        5. Mark scan as completed
        6. Return ScanResult
    """
    # 1. Create scan record
    scan = crud.create_scan(db, source_path=source_path, camera_id=camera_id)
    crud.update_scan_status(db, scan.id, ScanStatus.processing)

    # Fetch camera metadata for geo-matching
    cam = crud.get_camera(db, camera_id) if camera_id else None
    cam_lat = cam.lat if cam else None
    cam_lon = cam.lon if cam else None
    roi_coords = cam.roi_coords if cam else None

    try:
        # 2. Run inference
        all_detections = []

        if is_video:
            timeline = detector.infer_video(source_path, target_fps=2.0)
            for _ts, dets in timeline:
                all_detections.extend(dets)
        else:
            # For image: optionally apply ROI mask, then infer
            if roi_coords:
                import cv2
                from src.preprocessing.roi_mask import apply_roi_mask
                frame = cv2.imread(source_path)
                if frame is not None:
                    masked = apply_roi_mask(frame, roi_coords)
                    dets = detector.infer_frame(masked)
                else:
                    dets = detector.infer_image_path(source_path)
            else:
                dets = detector.infer_image_path(source_path)
            all_detections.extend(dets)

        # 3. Change detection
        change_detector = ChangeDetector(
            camera_id=camera_id or 0,
            scan_id=scan.id,
        )
        changes = change_detector.process_detections(
            db=db,
            detections=all_detections,
            camera_lat=cam_lat,
            camera_lon=cam_lon,
        )

        # 4. Auto-create incidents for critical/high severity new detections
        for change in changes:
            if change.severity in ("critical", "high"):
                crud.create_incident(
                    db,
                    pothole_id=change.pothole_id,
                    severity=Severity(change.severity),
                    department=None,  # unassigned; dispatcher updates later
                )

        # 5. Mark scan complete
        crud.update_scan_status(db, scan.id, ScanStatus.completed)

        # 6. Build response
        new_count = sum(1 for c in changes if c.is_new)
        existing_count = sum(1 for c in changes if not c.is_new)
        growing_count = sum(1 for c in changes if c.is_growing)

        return ScanResult(
            scan_id=scan.id,
            camera_id=camera_id,
            source_path=source_path,
            status=ScanStatus.completed,
            total_detections=len(all_detections),
            new_potholes=new_count,
            existing_potholes=existing_count,
            growing_potholes=growing_count,
            detections=[d.to_dict() for d in all_detections],
        )

    except Exception as exc:
        crud.update_scan_status(db, scan.id, ScanStatus.failed)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan pipeline failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=ScanResult, status_code=status.HTTP_201_CREATED)
async def upload_and_scan(
    db: DbDep,
    detector: DetectorDep,
    file: UploadFile = File(..., description="Video (.mp4, .avi) or image (.jpg, .png) file"),
    camera_id: Optional[int] = Form(None, description="Camera ID to associate with this scan"),
):
    """
    Upload a video or image file and run the full pothole detection pipeline.

    Returns a summary of all detections, new potholes, and growing potholes.
    """
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # Save upload to temp dir
    tmp_path = UPLOAD_DIR / file.filename
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        return _run_scan_pipeline(
            db=db,
            detector=detector,
            source_path=str(tmp_path),
            camera_id=camera_id,
            is_video=is_video,
        )
    finally:
        # Clean up uploaded file after scan
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@router.post("/scan", response_model=ScanResult, status_code=status.HTTP_201_CREATED)
def scan_local_file(
    db: DbDep,
    detector: DetectorDep,
    source_path: str = Form(..., description="Absolute or relative path to a local video/image"),
    camera_id: Optional[int] = Form(None),
):
    """
    Run the pothole detection pipeline on a file already on the server.

    Useful when footage is stored locally (e.g. CCTV archive on same machine).
    """
    p = Path(source_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {source_path}")

    ext = p.suffix.lower()
    is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    return _run_scan_pipeline(
        db=db,
        detector=detector,
        source_path=str(p),
        camera_id=camera_id,
        is_video=is_video,
    )
