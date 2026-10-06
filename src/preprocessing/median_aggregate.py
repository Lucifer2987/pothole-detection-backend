"""
Median-frame aggregation for background-subtraction / noise reduction.

When multiple frames are available, the pixel-wise median produces a stable
background image that suppresses moving objects (vehicles, pedestrians) and
reduces sensor noise.  This background can then be differenced from a target
frame to highlight static defects like potholes and cracks.
"""

from typing import List

import cv2
import numpy as np


def compute_median_frame(frames: List[np.ndarray]) -> np.ndarray:
    """
    Compute a pixel-wise median across a list of BGR frames.

    All frames must be the same shape (H, W, 3).  Uses float32 internally
    to avoid uint8 overflow, then clips and converts back.

    Args:
        frames: List of BGR images (np.ndarray) of identical shape.

    Returns:
        Median frame as uint8 BGR image.

    Raises:
        ValueError: If the frame list is empty or frames have differing shapes.
    """
    if not frames:
        raise ValueError("frames list must not be empty")

    ref_shape = frames[0].shape
    for i, f in enumerate(frames[1:], start=1):
        if f.shape != ref_shape:
            raise ValueError(
                f"Frame {i} shape {f.shape} differs from frame 0 shape {ref_shape}"
            )

    stack = np.stack(frames, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.uint8)
    return median


def subtract_background(
    frame: np.ndarray,
    background: np.ndarray,
    threshold: int = 30,
) -> np.ndarray:
    """
    Produce a binary foreground mask by differencing a frame against a background.

    Args:
        frame:      Current BGR frame.
        background: Median background BGR frame (same shape as frame).
        threshold:  Absolute difference threshold (0-255).  Pixels with mean
                    channel difference above this value are considered foreground.

    Returns:
        Binary uint8 mask (255 = foreground / defect candidate, 0 = background).
    """
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def sample_frames_evenly(
    video_path: str,
    n: int = 10,
) -> List[np.ndarray]:
    """
    Read n frames evenly spaced throughout a video file.

    Args:
        video_path: Path to the video file.
        n:          Number of frames to sample.

    Returns:
        List of BGR frames.

    Raises:
        IOError: If the video cannot be opened.
        ValueError: If the video has fewer frames than n.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n:
        n = max(1, total)

    indices = np.linspace(0, total - 1, n, dtype=int)
    frames: List[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames
