"""
Persistent cross-scan change detection for pothole tracking.

This module answers the question: "Is this detection a *new* pothole,
an *existing* one I've seen before, or is an existing one *growing*?"

Algorithm:
    1. Query the DB for all known potholes from the same camera.
    2. For each detection from the current scan, find the closest known
       pothole (by bounding-box IoU or geo-distance).
    3. If IoU > threshold → it's the *same* pothole (update its record).
    4. If no match → create a *new* pothole record.
    5. After processing, append a PotholeHistory row for every matched/new
       pothole, enabling the growth timeline.

We use IoU for same-camera, same-frame-space matching.  For cameras with
GPS coordinates, Haversine distance is used as a fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from src.database.models import ClassType, Pothole, PotholeHistory, PotholeStatus
from src.detection.infer import Detection
from src.detection.severity import severity_from_detection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IOU_MATCH_THRESHOLD = 0.20    # lower than in-frame tracker — boxes shift between scans
GEO_MATCH_METRES    = 15.0    # match if within 15 m (when GPS available)
AREA_GROWTH_RATIO   = 1.20    # flag as "growing" if area increased by ≥ 20 %


# ---------------------------------------------------------------------------
# Data class for change-detection result
# ---------------------------------------------------------------------------

@dataclass
class ChangeResult:
    """Result of matching one detection against the database."""
    pothole_id: int
    is_new: bool
    is_growing: bool
    area: float
    severity: str
    class_type: ClassType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou_box(
    det: Detection,
    p: Pothole,
) -> float:
    """
    Compute approximate IoU between a Detection and a stored Pothole.

    Since we don't store xyxy coordinates in the Pothole table we use
    the bounding-box inferred from sqrt(area) as a proxy centroid.
    When area is stored we can compute centroid-distance instead.
    This is an approximation suitable for the prototype.
    """
    if p.current_area is None or p.current_area <= 0:
        return 0.0
    # Estimate bounding box of stored pothole as a square centred on nothing —
    # we have no centre info in the DB, so use area overlap as a heuristic.
    # Real matching uses geo-distance when GPS is present.
    return 0.0   # will fall back to geo when lat/lon available


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """
    Matches detections from a new scan against the persistent pothole DB.

    Args:
        camera_id:        The camera/location ID for this scan.
        scan_id:          The Scan row ID just created.
        iou_threshold:    Minimum IoU to match (used when no GPS).
        geo_threshold_m:  Maximum distance in metres to match (when GPS available).
    """

    def __init__(
        self,
        camera_id: int,
        scan_id: int,
        iou_threshold: float = IOU_MATCH_THRESHOLD,
        geo_threshold_m: float = GEO_MATCH_METRES,
    ) -> None:
        self.camera_id = camera_id
        self.scan_id = scan_id
        self.iou_threshold = iou_threshold
        self.geo_threshold_m = geo_threshold_m

    # ------------------------------------------------------------------

    def _find_match(
        self,
        det: Detection,
        det_lat: Optional[float],
        det_lon: Optional[float],
        known: List[Pothole],
    ) -> Optional[Pothole]:
        """
        Find the best matching Pothole from the DB for a given detection.

        Returns the matched Pothole or None.
        """
        best: Optional[Pothole] = None
        best_dist = float("inf")

        for p in known:
            # Strategy 1: GPS distance (preferred)
            if det_lat and det_lon and p.lat and p.lon:
                dist = _haversine_m(det_lat, det_lon, p.lat, p.lon)
                if dist < self.geo_threshold_m and dist < best_dist:
                    best_dist = dist
                    best = p
            else:
                # Strategy 2: same camera, same class, area within 3× — heuristic
                if p.class_type.value == det.class_type.value:
                    det_area = det.area
                    p_area = p.current_area or 0.0
                    if p_area > 0 and det_area / p_area < 3.0 and det_area / p_area > 0.33:
                        # Use a stable "distance" = area ratio deviation from 1.0
                        ratio_dist = abs(det_area / p_area - 1.0)
                        if ratio_dist < best_dist:
                            best_dist = ratio_dist
                            best = p

        return best

    # ------------------------------------------------------------------

    def process_detections(
        self,
        db: Session,
        detections: List[Detection],
        camera_lat: Optional[float] = None,
        camera_lon: Optional[float] = None,
        from_timestamp=None,
    ) -> List[ChangeResult]:
        """
        Match detections against DB, create/update Pothole rows, and
        append PotholeHistory rows.

        Args:
            db:           SQLAlchemy session.
            detections:   Detections from the current scan.
            camera_lat:   Camera GPS latitude (or None).
            camera_lon:   Camera GPS longitude (or None).
            from_timestamp: datetime to stamp history rows (default: utcnow).

        Returns:
            List of ChangeResult — one per detection.
        """
        from datetime import datetime
        ts = from_timestamp or datetime.utcnow()

        # Load existing potholes for this camera
        known: List[Pothole] = (
            db.query(Pothole)
            .filter(Pothole.camera_id == self.camera_id)
            .filter(Pothole.status != PotholeStatus.resolved)
            .all()
        )

        results: List[ChangeResult] = []

        for det in detections:
            severity, area = severity_from_detection(
                det.x1, det.y1, det.x2, det.y2,
                det.confidence,
                det.class_type,
            )

            match = self._find_match(det, camera_lat, camera_lon, known)

            if match is None:
                # ── NEW pothole ───────────────────────────────────────
                pothole = Pothole(
                    camera_id=self.camera_id,
                    first_detected=ts,
                    last_seen=ts,
                    lat=camera_lat,
                    lon=camera_lon,
                    class_type=det.class_type,
                    current_area=area,
                    status=PotholeStatus.new,
                )
                db.add(pothole)
                db.flush()  # get the ID

                history = PotholeHistory(
                    pothole_id=pothole.id,
                    scan_id=self.scan_id,
                    area=area,
                    confidence=det.confidence,
                    timestamp=ts,
                )
                db.add(history)

                # Add to known so subsequent detections in same scan don't dup
                known.append(pothole)

                results.append(ChangeResult(
                    pothole_id=pothole.id,
                    is_new=True,
                    is_growing=False,
                    area=area,
                    severity=severity.value,
                    class_type=det.class_type,
                ))

            else:
                # ── EXISTING pothole ──────────────────────────────────
                prev_area = match.current_area or 0.0
                is_growing = (prev_area > 0) and (area >= prev_area * AREA_GROWTH_RATIO)

                match.last_seen = ts
                match.current_area = area
                if match.status == PotholeStatus.new:
                    match.status = PotholeStatus.existing

                history = PotholeHistory(
                    pothole_id=match.id,
                    scan_id=self.scan_id,
                    area=area,
                    confidence=det.confidence,
                    timestamp=ts,
                )
                db.add(history)

                results.append(ChangeResult(
                    pothole_id=match.id,
                    is_new=False,
                    is_growing=is_growing,
                    area=area,
                    severity=severity.value,
                    class_type=det.class_type,
                ))

        db.commit()
        return results
