"""
Frame extraction from local video files and static images.

For the prototype we use local files as input (not live camera feeds).
This module provides:
    - FrameCapture  – extract frames from video files at a given FPS
    - load_image    – load a single image file as a BGR numpy array
"""

from pathlib import Path
from typing import Generator, List, Optional

import cv2
import numpy as np


class FrameCapture:
    """
    Extract frames from a local video file.

    Args:
        video_path:  Path to the video file.
        target_fps:  Desired frames-per-second to sample.  If the video's
                     native FPS is lower than target_fps, every frame is
                     returned.  Use None to return every frame.
        max_frames:  Stop after this many frames (None = no limit).

    Usage::

        for frame, ts in FrameCapture("road.mp4", target_fps=2):
            process(frame)
    """

    def __init__(
        self,
        video_path: str,
        target_fps: Optional[float] = 2.0,
        max_frames: Optional[int] = None,
    ) -> None:
        self.video_path = str(video_path)
        self.target_fps = target_fps
        self.max_frames = max_frames

    def __enter__(self) -> "FrameCapture":
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        self._native_fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        # Compute frame step: how many native frames to skip between samples
        if self.target_fps is None:
            self._step = 1
        else:
            self._step = max(1, int(round(self._native_fps / self.target_fps)))
        return self

    def __exit__(self, *_) -> None:
        if hasattr(self, "_cap"):
            self._cap.release()

    def frames(self) -> Generator[tuple[np.ndarray, float], None, None]:
        """
        Yield (frame_bgr, timestamp_seconds) tuples.

        Automatically applies the frame-step so the output rate matches
        target_fps as closely as possible.
        """
        frame_idx = 0
        emitted = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if frame_idx % self._step == 0:
                ts = frame_idx / self._native_fps
                yield frame, ts
                emitted += 1
                if self.max_frames is not None and emitted >= self.max_frames:
                    break

            frame_idx += 1

    # Allow use as a simple iterator (without context manager) for convenience
    def __iter__(self):
        with self as fc:
            yield from fc.frames()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @property
    def native_fps(self) -> float:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return fps

    @property
    def total_frames(self) -> int:
        cap = cv2.VideoCapture(self.video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / max(self.native_fps, 1e-6)


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """
    Load a single image file and return it as a BGR numpy array.

    Args:
        image_path: Path to the image (.jpg, .png, etc.).

    Returns:
        BGR np.ndarray (H, W, 3).

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError:           If OpenCV cannot decode the file.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"OpenCV could not decode image: {image_path}")
    return img


def extract_frames_from_video(
    video_path: str,
    target_fps: float = 2.0,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Convenience wrapper: extract all sampled frames into a list.

    Prefer the FrameCapture context manager for large videos to avoid
    loading all frames into memory at once.
    """
    frames: List[np.ndarray] = []
    with FrameCapture(video_path, target_fps=target_fps, max_frames=max_frames) as fc:
        for frame, _ in fc.frames():
            frames.append(frame)
    return frames
