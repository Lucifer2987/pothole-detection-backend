"""
ROI (Region of Interest) masking for road frames.

Applies a polygon mask to an image so only the road area is visible.
The camera's roi_coords JSON string (stored in the DB) is parsed here.
"""

import json
from typing import Optional

import cv2
import numpy as np


def parse_roi(roi_coords: Optional[str]) -> Optional[np.ndarray]:
    """
    Parse an ROI coordinate string into a NumPy polygon array.

    Args:
        roi_coords: JSON string like '[[x1,y1],[x2,y2],...]' or None.

    Returns:
        np.ndarray of shape (N, 1, 2) with dtype int32, or None if input is None.
    """
    if not roi_coords:
        return None
    pts = json.loads(roi_coords)
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def apply_roi_mask(frame: np.ndarray, roi_coords: Optional[str]) -> np.ndarray:
    """
    Apply a polygon ROI mask to a BGR frame.

    Pixels outside the polygon are set to black; pixels inside are unchanged.

    Args:
        frame:      BGR image as np.ndarray (H, W, 3).
        roi_coords: JSON ROI string from the Camera record, or None.

    Returns:
        Masked BGR frame.  If roi_coords is None, the original frame is
        returned unchanged.
    """
    polygon = parse_roi(roi_coords)
    if polygon is None:
        return frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)
