from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _draw_overlay(image_bgr: np.ndarray, original_mask: np.ndarray, sealed_mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    added = (sealed_mask > 0) & (original_mask == 0)
    kept = sealed_mask > 0
    overlay[kept] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[added] = np.array([0, 255, 255], dtype=np.uint8)
    return overlay


def run_pipe_seal_stage(
    *,
    image_bgr: np.ndarray,
    pipe_mask: np.ndarray,
    image_id: str,
    horizontal_close_kernel: int = 5,
    vertical_close_kernel: int = 5,
    min_component_area: int = 16,
) -> dict[str, Any]:
    if pipe_mask.ndim != 2:
        raise ValueError("pipe_mask must be a 2D array")

    original = np.where(pipe_mask > 0, 255, 0).astype(np.uint8)
    work = original.copy()

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, horizontal_close_kernel), 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, vertical_close_kernel)))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel_h)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel_v)

    before_components = int(cv2.connectedComponents((original > 0).astype(np.uint8), connectivity=8)[0] - 1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(work)
    removed_small_components = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_component_area:
            removed_small_components += 1
            continue
        filtered[labels == label_idx] = 255

    after_components = int(cv2.connectedComponents((filtered > 0).astype(np.uint8), connectivity=8)[0] - 1)
    added_pixels = int(np.count_nonzero((filtered > 0) & (original == 0)))
    removed_pixels = int(np.count_nonzero((original > 0) & (filtered == 0)))
    changed_pixels = int(np.count_nonzero(filtered != original))

    return {
        "sealed_mask_image": filtered,
        "overlay_image": _draw_overlay(image_bgr, original, filtered),
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "mask_pixel_count": int(np.count_nonzero(filtered)),
            "connected_component_count_before": before_components,
            "connected_component_count_after": after_components,
            "removed_small_components": removed_small_components,
            "added_pixels": added_pixels,
            "removed_pixels": removed_pixels,
            "changed_pixel_count": changed_pixels,
            "horizontal_close_kernel": horizontal_close_kernel,
            "vertical_close_kernel": vertical_close_kernel,
            "min_component_area": min_component_area,
            "source_artifacts": [
                "stage5_pipe_mask.png",
            ],
        },
    }
