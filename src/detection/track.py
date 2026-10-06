"""
Simple IoU-based spatial tracker for matching detections across frames.

For the prototype we avoid heavy multi-object tracking libraries (ByteTrack,
DeepSORT) and instead use a greedy IoU-matching approach:

    1. Each new detection is compared against known "active tracks".
    2. If IoU > threshold with an existing track, they are merged (track
       position updated to the new detection's box).
    3. Detections with no matching track start a new track.
    4. Tracks not seen for `max_missed` frames are marked as lost.

This gives consistent integer track IDs within a single video run.
Across runs (persistent matching) is handled by change_detector.py using
the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.infer import Detection


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """Active track maintained by the Tracker."""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    class_id: int
    frames_seen: int = 1
    frames_missed: int = 0
    last_timestamp: float = 0.0

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def update(self, det: Detection, timestamp: float) -> None:
        """Update track position from a new matching detection."""
        self.x1, self.y1 = det.x1, det.y1
        self.x2, self.y2 = det.x2, det.y2
        self.confidence = det.confidence
        self.frames_seen += 1
        self.frames_missed = 0
        self.last_timestamp = timestamp


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union for two XYXY boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Tracker:
    """
    Greedy IoU-based multi-object tracker.

    Args:
        iou_threshold: Minimum IoU to associate a detection with a track.
        max_missed:    Frames without a detection before a track is dropped.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_missed: int = 5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        timestamp: float = 0.0,
    ) -> List[Track]:
        """
        Process detections from one frame.

        Args:
            detections: List of Detection objects for this frame.
            timestamp:  Frame timestamp in seconds.

        Returns:
            List of currently active Track objects (matched + new).
        """
        # Step 1: Increment missed count for all existing tracks
        for track in self._tracks.values():
            track.frames_missed += 1

        # Step 2: Greedy match detections → tracks
        matched_track_ids = set()
        matched_det_indices = set()

        track_ids = list(self._tracks.keys())
        for det_idx, det in enumerate(detections):
            best_iou = self.iou_threshold
            best_tid = None

            for tid in track_ids:
                if tid in matched_track_ids:
                    continue
                track = self._tracks[tid]
                # Only match same class
                if track.class_id != det.class_id:
                    continue
                iou = _iou(track.box, (det.x1, det.y1, det.x2, det.y2))
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self._tracks[best_tid].update(det, timestamp)
                # Undo the missed-count increment applied above
                self._tracks[best_tid].frames_missed -= 1
                matched_track_ids.add(best_tid)
                matched_det_indices.add(det_idx)

        # Step 3: Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            new_track = Track(
                track_id=self._next_id,
                x1=det.x1, y1=det.y1,
                x2=det.x2, y2=det.y2,
                confidence=det.confidence,
                class_name=det.class_name,
                class_id=det.class_id,
                last_timestamp=timestamp,
            )
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # Step 4: Remove stale tracks
        stale = [tid for tid, t in self._tracks.items() if t.frames_missed > self.max_missed]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())

    def active_tracks(self) -> List[Track]:
        """Return all currently active tracks."""
        return list(self._tracks.values())

    def reset(self) -> None:
        """Clear all tracks (call between unrelated video runs)."""
        self._tracks.clear()
        self._next_id = 1
