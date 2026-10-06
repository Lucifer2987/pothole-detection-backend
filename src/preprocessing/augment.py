"""
Data augmentation module for bbox-aware image transforms.

Uses albumentations with bounding-box-safe augmentations:
  - Horizontal flip
  - Brightness/contrast jitter
  - Slight rotation (+/-10 deg)
  - Motion blur (simulates vehicle camera movement)
  - Random shadow (simulates shadows on road)

Usage:
    from src.preprocessing.augment import augment_image

    aug_img, aug_boxes, aug_labels = augment_image(image, bboxes, class_labels)

    # bboxes: list of [x_center, y_center, width, height] (YOLO normalised 0-1)
    # class_labels: list of int class indices
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.core.composition import Compose
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False


# ---------------------------------------------------------------------------
# Build the augmentation pipeline
# ---------------------------------------------------------------------------

def _build_pipeline(p: float = 0.5) -> "Compose":
    """
    Build an albumentations pipeline with bbox-aware transforms.

    Args:
        p: overall probability for the stochastic transforms.
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations is required. Install it with: pip install albumentations"
        )

    return A.Compose(
        [
            # Geometry
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4),

            # Photometry
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3
            ),

            # Motion / blur (simulates camera shake / vehicle motion)
            A.MotionBlur(blur_limit=(3, 9), p=0.3),

            # Road conditions
            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 1),
                num_shadows_limit=(1, 2),    # albumentations 2.0.x exact param name
                shadow_dimension=5,
                p=0.25,
            ),

            # Noise / quality
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",             # (cx, cy, w, h) normalised
            label_fields=["class_labels"],
            min_visibility=0.25,       # drop boxes that become <25% visible
            clip=True,
        ),
    )


# Cache the pipeline so it isn't rebuilt on every call
_PIPELINE: Optional["Compose"] = None


def _get_pipeline() -> "Compose":
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _build_pipeline()
    return _PIPELINE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_image(
    image: np.ndarray,
    bboxes: List[List[float]],
    class_labels: List[int],
    pipeline: Optional["Compose"] = None,
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Apply bbox-aware augmentations to a single image.

    Args:
        image:        BGR numpy array (H, W, 3).
        bboxes:       List of [cx, cy, w, h] bounding boxes (YOLO normalised 0–1).
        class_labels: List of integer class indices, same length as bboxes.
        pipeline:     Optional custom albumentations Compose pipeline.
                      If None, uses the module-level cached default pipeline.

    Returns:
        Tuple of (augmented_image, augmented_bboxes, augmented_class_labels).
        augmented_bboxes and augmented_class_labels may be shorter than the
        inputs if any box was lost (e.g. flipped/rotated out of frame).

    Raises:
        ImportError: If albumentations is not installed.
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required: pip install albumentations")

    # albumentations works in RGB internally
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pipe = pipeline or _get_pipeline()

    result = pipe(
        image=image_rgb,
        bboxes=bboxes,
        class_labels=class_labels,
    )

    aug_image = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
    aug_bboxes = [list(b) for b in result["bboxes"]]
    aug_labels = [int(c) for c in result["class_labels"]]  # albumentations may return floats

    return aug_image, aug_bboxes, aug_labels


def augment_yolo_label_file(
    image: np.ndarray,
    label_text: str,
    pipeline: Optional["Compose"] = None,
) -> Tuple[np.ndarray, str]:
    """
    Convenience wrapper that accepts/returns raw YOLO label text.

    Args:
        image:       BGR frame.
        label_text:  Multi-line string in YOLO format.
        pipeline:    Optional pipeline override.

    Returns:
        (augmented_image, augmented_label_text)
    """
    bboxes: List[List[float]] = []
    class_labels: List[int] = []

    for line in label_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        box = [float(x) for x in parts[1:5]]
        class_labels.append(cid)
        bboxes.append(box)

    aug_img, aug_boxes, aug_cls = augment_image(image, bboxes, class_labels, pipeline)

    lines = [
        f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
        for c, b in zip(aug_cls, aug_boxes)
    ]
    return aug_img, "\n".join(lines)
