from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage as nd


def _degree_map(skeleton_mask: np.ndarray) -> np.ndarray:
    binary = (skeleton_mask > 0).astype(np.int32)
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    return nd.convolve(binary, kernel, mode="constant")


def _point_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) > 0:
        mask[points[:, 0], points[:, 1]] = 255
    return mask


def _draw_overlay(image_bgr: np.ndarray, endpoints: np.ndarray, junctions: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for row, col in endpoints:
        cv2.circle(overlay, (int(col), int(row)), 2, (0, 255, 0), -1)
    for row, col in junctions:
        cv2.circle(overlay, (int(col), int(row)), 2, (0, 0, 255), -1)
    return overlay


def run_pipe_node_stage(
    *,
    image_bgr: np.ndarray,
    skeleton_mask: np.ndarray,
    image_id: str,
) -> dict[str, Any]:
    if skeleton_mask.ndim != 2:
        raise ValueError("skeleton_mask must be a 2D array")

    degree_map = _degree_map(skeleton_mask)
    endpoints = np.argwhere(degree_map == 11)
    junctions = np.argwhere(degree_map >= 13)

    endpoint_mask = _point_mask(endpoints, skeleton_mask.shape)
    junction_mask = _point_mask(junctions, skeleton_mask.shape)

    return {
        "endpoint_image": endpoint_mask,
        "junction_image": junction_mask,
        "overlay_image": _draw_overlay(image_bgr, endpoints, junctions),
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "endpoint_count": int(len(endpoints)),
            "junction_count": int(len(junctions)),
            "skeleton_pixel_count": int(np.count_nonzero(skeleton_mask)),
            "source_artifacts": [
                "stage7_pipe_skeleton.png",
            ],
        },
    }
