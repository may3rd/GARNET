from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return skeleton


def _draw_overlay(image_bgr: np.ndarray, skeleton_mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay[skeleton_mask > 0] = np.array([0, 255, 0], dtype=np.uint8)
    return overlay


def run_pipe_skeleton_stage(
    *,
    image_bgr: np.ndarray,
    sealed_mask: np.ndarray,
    image_id: str,
) -> dict[str, Any]:
    if sealed_mask.ndim != 2:
        raise ValueError("sealed_mask must be a 2D array")

    skeleton_mask = _skeletonize_mask(sealed_mask)
    input_pixels = int(np.count_nonzero(sealed_mask))
    skeleton_pixels = int(np.count_nonzero(skeleton_mask))

    return {
        "skeleton_image": skeleton_mask,
        "overlay_image": _draw_overlay(image_bgr, skeleton_mask),
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "input_mask_pixel_count": input_pixels,
            "skeleton_pixel_count": skeleton_pixels,
            "pixel_reduction": input_pixels - skeleton_pixels,
            "source_artifacts": [
                "stage6_pipe_mask_sealed.png",
            ],
        },
    }
