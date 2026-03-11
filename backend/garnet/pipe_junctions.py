from __future__ import annotations

from typing import Any

import cv2
import numpy as np


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
    crossing_candidates: list[dict[str, Any]],
    image_id: str,
) -> dict[str, Any]:
    confirmed = [dict(item) for item in crossing_candidates if str(item.get("classification")) == "confirmed_junction"]
    unresolved = [dict(item) for item in crossing_candidates if str(item.get("classification")) == "unresolved"]
    non_connecting_crossings = [
        dict(item) for item in crossing_candidates if str(item.get("classification")) == "non_connecting_crossing"
    ]

    return {
        "confirmed_junction_image": _draw_points(image_bgr.shape[:2], confirmed),
        "unresolved_junction_image": _draw_points(image_bgr.shape[:2], unresolved),
        "overlay_image": _draw_overlay(image_bgr, confirmed, unresolved),
        "junctions_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "confirmed_junctions": confirmed,
            "unresolved_junctions": unresolved,
            "non_connecting_crossings": non_connecting_crossings,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "confirmed_junction_count": len(confirmed),
            "unresolved_junction_count": len(unresolved),
            "non_connecting_crossing_count": len(non_connecting_crossings),
            "source_artifacts": [
                "stage10_crossing_resolution.json",
            ],
        },
    }
