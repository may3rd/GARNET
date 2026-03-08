from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _suppress_boxes(mask: np.ndarray, boxes: list[dict[str, Any]], padding: int = 0) -> tuple[np.ndarray, int]:
    suppressed = mask.copy()
    removed = 0
    height, width = suppressed.shape[:2]
    for item in boxes:
        bbox = item["bbox"]
        x_min = max(0, int(bbox["x_min"]) - padding)
        y_min = max(0, int(bbox["y_min"]) - padding)
        x_max = min(width - 1, int(bbox["x_max"]) + padding)
        y_max = min(height - 1, int(bbox["y_max"]) + padding)
        region = suppressed[y_min : y_max + 1, x_min : x_max + 1]
        removed += int(np.count_nonzero(region))
        suppressed[y_min : y_max + 1, x_min : x_max + 1] = 0
    return suppressed, removed


def _suppress_object_interiors(mask: np.ndarray, boxes: list[dict[str, Any]], inset: int = 1) -> tuple[np.ndarray, int]:
    suppressed = mask.copy()
    removed = 0
    height, width = suppressed.shape[:2]
    for item in boxes:
        bbox = item["bbox"]
        x_min = max(0, int(bbox["x_min"]) + inset)
        y_min = max(0, int(bbox["y_min"]) + inset)
        x_max = min(width - 1, int(bbox["x_max"]) - inset)
        y_max = min(height - 1, int(bbox["y_max"]) - inset)
        if x_min > x_max or y_min > y_max:
            continue
        region = suppressed[y_min : y_max + 1, x_min : x_max + 1]
        removed += int(np.count_nonzero(region))
        suppressed[y_min : y_max + 1, x_min : x_max + 1] = 0
    return suppressed, removed


def _filter_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    if min_area <= 0:
        return mask.copy(), 0
    work = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    filtered = np.zeros_like(mask)
    removed = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            removed += 1
            continue
        filtered[labels == label] = 255
    return filtered, removed


def _draw_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = np.array([255, 0, 0], dtype=np.uint8)
    return overlay


def run_pipe_mask_stage(
    *,
    image_bgr: np.ndarray,
    gray_image: np.ndarray,
    adaptive_mask: np.ndarray,
    otsu_mask: np.ndarray,
    ocr_regions: list[dict[str, Any]],
    object_regions: list[dict[str, Any]],
    image_id: str,
    ocr_padding: int = 1,
    object_inset: int = 1,
    min_component_area: int = 16,
) -> dict[str, Any]:
    del gray_image
    candidate_mask = cv2.bitwise_or(adaptive_mask, otsu_mask)
    candidate_mask = np.where(candidate_mask > 0, 255, 0).astype(np.uint8)

    ocr_suppressed, ocr_removed = _suppress_boxes(candidate_mask, ocr_regions, padding=ocr_padding)
    object_suppressed, object_removed = _suppress_object_interiors(ocr_suppressed, object_regions, inset=object_inset)
    filtered_mask, small_component_removals = _filter_small_components(object_suppressed, min_component_area)

    return {
        "mask_image": filtered_mask,
        "overlay_image": _draw_overlay(image_bgr, filtered_mask),
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "mask_pixel_count": int(np.count_nonzero(filtered_mask)),
            "connected_component_count": int(cv2.connectedComponents((filtered_mask > 0).astype(np.uint8), connectivity=8)[0] - 1),
            "small_component_removals": small_component_removals,
            "ocr_suppression_pixel_count": ocr_removed,
            "object_suppression_pixel_count": object_removed,
            "source_artifacts": [
                "stage1_gray.png",
                "stage1_binary_adaptive.png",
                "stage1_binary_otsu.png",
                "stage2_ocr_regions.json",
                "stage4_objects.json",
            ],
            "ocr_padding": ocr_padding,
            "object_inset": object_inset,
            "min_component_area": min_component_area,
        },
    }
