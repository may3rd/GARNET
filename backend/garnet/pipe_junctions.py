from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


Point = tuple[int, int]


def _neighbors(pixel: Point, skeleton: np.ndarray) -> list[Point]:
    row, col = pixel
    neighbors: list[Point] = []
    row_max, col_max = skeleton.shape[:2]
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
            if skeleton[next_row, next_col] > 0:
                neighbors.append((next_row, next_col))
    return neighbors


def _cluster_member_set(cluster: dict[str, Any]) -> set[Point]:
    return {(int(m["row"]), int(m["col"])) for m in cluster.get("members", [])}


def _angle_groups(cluster: dict[str, Any], skeleton: np.ndarray) -> list[float]:
    centroid_x = float(cluster["centroid"]["x"])
    centroid_y = float(cluster["centroid"]["y"])
    members = _cluster_member_set(cluster)
    angles: list[float] = []
    for pixel in members:
        for neighbor in _neighbors(pixel, skeleton):
            if neighbor in members:
                continue
            dy = float(neighbor[0]) - centroid_y
            dx = float(neighbor[1]) - centroid_x
            angle = math.degrees(math.atan2(dy, dx)) % 360.0
            angles.append(angle)
    if not angles:
        return []
    angles.sort()
    groups: list[float] = []
    for angle in angles:
        if not groups or min(abs(angle - groups[-1]), 360.0 - abs(angle - groups[-1])) > 20.0:
            groups.append(angle)
    return groups


def _draw_points(shape: tuple[int, int], clusters: list[dict[str, Any]]) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    for cluster in clusters:
        x = int(round(cluster["centroid"]["x"]))
        y = int(round(cluster["centroid"]["y"]))
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            cv2.circle(image, (x, y), 3, 255, -1)
    return image


def _draw_overlay(
    image_bgr: np.ndarray,
    confirmed: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> np.ndarray:
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for cluster in confirmed:
        x = int(round(cluster["centroid"]["x"]))
        y = int(round(cluster["centroid"]["y"]))
        cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
    for cluster in unresolved:
        x = int(round(cluster["centroid"]["x"]))
        y = int(round(cluster["centroid"]["y"]))
        cv2.circle(overlay, (x, y), 4, (0, 165, 255), -1)
    return overlay


def run_pipe_junction_stage(
    *,
    image_bgr: np.ndarray,
    skeleton_mask: np.ndarray,
    node_clusters: list[dict[str, Any]],
    image_id: str,
) -> dict[str, Any]:
    confirmed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for cluster in node_clusters:
        if cluster.get("kind") != "junction":
            continue
        branch_angles = _angle_groups(cluster, skeleton_mask)
        reviewed = dict(cluster)
        reviewed["branch_count"] = len(branch_angles)
        reviewed["branch_angles_deg"] = [round(angle, 2) for angle in branch_angles]
        if len(branch_angles) >= 3:
            confirmed.append(reviewed)
        else:
            unresolved.append(reviewed)

    return {
        "confirmed_junction_image": _draw_points(skeleton_mask.shape, confirmed),
        "unresolved_junction_image": _draw_points(skeleton_mask.shape, unresolved),
        "overlay_image": _draw_overlay(image_bgr, confirmed, unresolved),
        "junctions_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "confirmed_junctions": confirmed,
            "unresolved_junctions": unresolved,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "confirmed_junction_count": len(confirmed),
            "unresolved_junction_count": len(unresolved),
            "source_artifacts": [
                "stage7_pipe_skeleton.png",
                "stage9_node_clusters.json",
            ],
        },
    }
