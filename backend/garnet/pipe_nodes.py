from __future__ import annotations

import math
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


def _neighbors(row: int, col: int, skeleton_mask: np.ndarray) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    row_max, col_max = skeleton_mask.shape[:2]
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            next_row = row + row_offset
            next_col = col + col_offset
            if next_row < 0 or next_col < 0:
                continue
            if next_row >= row_max or next_col >= col_max:
                continue
            if skeleton_mask[next_row, next_col] > 0:
                neighbors.append((next_row, next_col))
    return neighbors


def _angle_distance_deg(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _direction_group_count(row: int, col: int, skeleton_mask: np.ndarray, tolerance_deg: float = 20.0) -> int:
    angles: list[float] = []
    for next_row, next_col in _neighbors(row, col, skeleton_mask):
        dy = float(next_row - row)
        dx = float(next_col - col)
        angles.append(math.degrees(math.atan2(dy, dx)) % 360.0)
    angles.sort()
    groups: list[float] = []
    for angle in angles:
        if not groups or _angle_distance_deg(angle, groups[-1]) > tolerance_deg:
            groups.append(angle)
            continue
        groups[-1] = (groups[-1] + angle) / 2.0
    if len(groups) > 1 and _angle_distance_deg(groups[0], groups[-1]) <= tolerance_deg:
        merged = (groups[0] + groups[-1]) / 2.0
        groups = [merged] + groups[1:-1]
    return len(groups)


def _filter_points_by_direction_groups(
    points: np.ndarray,
    skeleton_mask: np.ndarray,
    *,
    min_groups: int | None = None,
    exact_groups: int | None = None,
) -> np.ndarray:
    kept: list[np.ndarray] = []
    for point in points:
        row, col = int(point[0]), int(point[1])
        group_count = _direction_group_count(row, col, skeleton_mask)
        if exact_groups is not None and group_count != exact_groups:
            continue
        if min_groups is not None and group_count < min_groups:
            continue
        kept.append(point)
    if not kept:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(kept, dtype=np.int64)


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
    raw_endpoints = np.argwhere(degree_map == 11)
    raw_junctions = np.argwhere(degree_map >= 13)

    endpoints = _filter_points_by_direction_groups(raw_endpoints, skeleton_mask, exact_groups=1)
    junctions = _filter_points_by_direction_groups(raw_junctions, skeleton_mask, min_groups=3)

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
            "raw_endpoint_count": int(len(raw_endpoints)),
            "raw_junction_count": int(len(raw_junctions)),
            "skeleton_pixel_count": int(np.count_nonzero(skeleton_mask)),
            "source_artifacts": [
                "stage7_pipe_skeleton.png",
            ],
        },
    }
