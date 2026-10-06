"""
Minimal camera client stub.

For the prototype, input comes from local video/image files rather than live
camera feeds.  This module provides a thin interface that returns local file
paths when "connecting" to a camera, making it easy to swap in a real RTSP
client later without changing downstream code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CameraConfig:
    """
    Configuration for a single camera (or scan location).

    For the prototype, source_path points to a local video or image file.
    """
    camera_id: int
    name: str
    source_path: str                        # local file path
    lat: Optional[float] = None
    lon: Optional[float] = None
    roi_coords: Optional[str] = None       # JSON ROI string
    is_active: bool = True


class LocalCameraClient:
    """
    Minimal camera client that resolves local file paths.

    This makes the ingestion pipeline camera-agnostic: the same
    FrameCapture code works whether the source is a local file or
    a future live RTSP stream.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config

    def validate(self) -> bool:
        """Return True if the source file exists and is readable."""
        p = Path(self.config.source_path)
        return p.exists() and p.is_file()

    def get_source_path(self) -> str:
        """Return the local file path for this camera's input."""
        if not self.validate():
            raise FileNotFoundError(
                f"Source file not found for camera '{self.config.name}': "
                f"{self.config.source_path}"
            )
        return self.config.source_path

    def __repr__(self) -> str:
        return (
            f"<LocalCameraClient id={self.config.camera_id} "
            f"name={self.config.name!r} source={self.config.source_path!r}>"
        )
