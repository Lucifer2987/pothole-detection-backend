"""
YOLOv8 inference wrapper for pothole and crack detection.

Loads a fine-tuned YOLOv8 model and runs it on individual frames or image
paths, returning structured Detection objects.

Usage::

    detector = Detector("models/finetuned/pothole_yolov8s_sanity_best.pt")
    detections = detector.infer_frame(bgr_frame)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.database.models import ClassType


# ---------------------------------------------------------------------------
# Data class for a single detection
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    One bounding-box detection returned by the model.

    Coordinates are in pixel space (XYXY) at the inference resolution.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> float:
        """Bounding-box pixel area (px²)."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def class_type(self) -> ClassType:
        """Map class name to the ClassType enum."""
        name = self.class_name.lower()
        if "crack" in name:
            return ClassType.crack
        return ClassType.pothole

    def to_dict(self) -> dict:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "area": self.area,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """
    Wraps a YOLOv8 model for pothole / crack detection.

    Args:
        model_path:  Path to the fine-tuned .pt weights file.
        conf_thresh: Minimum confidence to accept a detection (default 0.25).
        iou_thresh:  IoU threshold for NMS (default 0.45).
        imgsz:       Inference image size (default 640).
        device:      Torch device string, e.g. 'cpu', 'cuda:0'.
                     If None, YOLO picks automatically.
    """

    CLASS_NAMES = {0: "pothole", 1: "crack"}

    def __init__(
        self,
        model_path: str = "models/finetuned/pothole_yolov8s_sanity_best.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        self.model_path = str(path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.device = device or ""  # empty string → YOLO auto-selects

        self._model: Optional[YOLO] = None

    def _load(self) -> YOLO:
        """Lazy-load the model on first inference call."""
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def _parse_results(self, results) -> List[Detection]:
        """Convert ultralytics Results object to a list of Detection objects."""
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()       # (N, 4)
            confs = r.boxes.conf.cpu().numpy()        # (N,)
            clsids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, clsids):
                name = self.CLASS_NAMES.get(cid, f"class_{cid}")
                detections.append(
                    Detection(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=float(conf),
                        class_id=int(cid),
                        class_name=name,
                    )
                )
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR numpy frame.

        Args:
            frame: BGR image as np.ndarray (H, W, 3).

        Returns:
            List of Detection objects (may be empty if nothing detected).
        """
        model = self._load()
        results = model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def infer_image_path(self, image_path: str) -> List[Detection]:
        """
        Run inference on an image file path.

        Args:
            image_path: Path to a .jpg / .png image.

        Returns:
            List of Detection objects.
        """
        model = self._load()
        results = model.predict(
            source=str(image_path),
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    def infer_video(
        self,
        video_path: str,
        target_fps: float = 2.0,
        max_frames: Optional[int] = None,
    ) -> List[tuple[float, List[Detection]]]:
        """
        Run inference on a video file, sampling at target_fps.

        Returns:
            List of (timestamp_seconds, detections) tuples.
        """
        from src.ingestion.frame_capture import FrameCapture

        results_timeline: List[tuple[float, List[Detection]]] = []
        with FrameCapture(video_path, target_fps=target_fps, max_frames=max_frames) as fc:
            for frame, ts in fc.frames():
                dets = self.infer_frame(frame)
                results_timeline.append((ts, dets))

        return results_timeline


# ---------------------------------------------------------------------------
# Module-level singleton helper
# ---------------------------------------------------------------------------

_default_detector: Optional[Detector] = None


def get_detector(model_path: str = "models/finetuned/pothole_yolov8s_sanity_best.pt") -> Detector:
    """
    Return (and cache) a module-level Detector singleton.

    Useful for FastAPI lifespan startup — avoids reloading the model on
    every request.
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = Detector(model_path=model_path)
    return _default_detector
