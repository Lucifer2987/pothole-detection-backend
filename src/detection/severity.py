"""
Severity scoring for detected potholes and cracks.

Severity is determined from a combination of:
    1. Bounding-box area (pixel²) — proxy for defect size
    2. Detection confidence — model certainty
    3. Defect class (crack vs pothole) — cracks score slightly lower by default

Thresholds can be tuned; the defaults are calibrated for 640×640 images.
"""

from __future__ import annotations

from src.database.models import ClassType, Severity


# ---------------------------------------------------------------------------
# Area thresholds (px²) at inference resolution 640×640
# ---------------------------------------------------------------------------
# Adjust these if you resize images differently before inference.

AREA_THRESHOLDS = {
    "critical": 40_000,   # > 40 000 px²  (~10 % of a 640×640 frame)
    "high":     15_000,   # > 15 000 px²
    "medium":    4_000,   # >  4 000 px²
    # below medium → low
}


def compute_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return bounding-box pixel area from XYXY coordinates."""
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def score_severity(
    area: float,
    confidence: float,
    class_type: ClassType,
) -> Severity:
    """
    Assign a Severity level to a single detection.

    The area-based bucket is used as the primary signal; a low confidence
    (<0.35) demotes the severity by one level.  Cracks start one level
    lower than potholes of equivalent area (they are typically thinner and
    harder to measure accurately by bounding-box alone).

    Args:
        area:       Bounding-box pixel area (px²).
        confidence: YOLO confidence score in [0, 1].
        class_type: ClassType.pothole or ClassType.crack.

    Returns:
        Severity enum value.
    """
    # --- Step 1: area bucket ---
    if area >= AREA_THRESHOLDS["critical"]:
        level = 3          # critical
    elif area >= AREA_THRESHOLDS["high"]:
        level = 2          # high
    elif area >= AREA_THRESHOLDS["medium"]:
        level = 1          # medium
    else:
        level = 0          # low

    # --- Step 2: crack penalty (cracks are thinner, box area underestimates) ---
    if class_type == ClassType.crack and level > 0:
        level -= 1

    # --- Step 3: confidence penalty ---
    if confidence < 0.35 and level > 0:
        level -= 1

    severity_map = {0: Severity.low, 1: Severity.medium, 2: Severity.high, 3: Severity.critical}
    return severity_map[level]


def severity_from_detection(
    x1: float, y1: float, x2: float, y2: float,
    confidence: float,
    class_type: ClassType,
) -> tuple[Severity, float]:
    """
    Convenience wrapper: compute area and severity in one call.

    Returns:
        (Severity, area_px²)
    """
    area = compute_area(x1, y1, x2, y2)
    sev = score_severity(area, confidence, class_type)
    return sev, area
