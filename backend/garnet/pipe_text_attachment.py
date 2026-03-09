from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _center_from_bbox(bbox: dict[str, int]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
    )


def _project_point_to_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[tuple[float, float], float]:
    px, py = point
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return a, math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    proj = (ax + t * abx, ay + t * aby)
    return proj, math.hypot(px - proj[0], py - proj[1])


def _nearest_edge(text_center: tuple[float, float], edges: list[dict[str, Any]]) -> tuple[str | None, float]:
    best_edge_id = None
    best_dist = float("inf")
    for edge in edges:
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue
        for start, end in zip(polyline, polyline[1:]):
            a = (float(start["col"]), float(start["row"]))
            b = (float(end["col"]), float(end["row"]))
            _, dist = _project_point_to_segment(text_center, a, b)
            if dist < best_dist:
                best_dist = dist
                best_edge_id = str(edge["id"])
    return best_edge_id, best_dist


def run_pipe_text_attachment_stage(
    *,
    image_id: str,
    image_bgr: np.ndarray,
    text_regions: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    max_distance_px: float = 80.0,
) -> dict[str, Any]:
    line_number_regions = [
        item
        for item in text_regions
        if item.get("class") == "line_number" or "source_object_id" in item
    ]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for region in line_number_regions:
        center = _center_from_bbox(region["bbox"])
        edge_id, distance_px = _nearest_edge(center, edges)
        payload = {
            "region_id": region.get("id", region.get("source_object_id")),
            "text": region["text"],
            "normalized_text": region.get("normalized_text", ""),
            "bbox": region["bbox"],
            "edge_id": edge_id,
            "distance_px": None if math.isinf(distance_px) else round(float(distance_px), 3),
        }
        if edge_id is not None and distance_px <= max_distance_px:
            accepted.append(payload)
        else:
            rejected.append(payload)

    overlay = image_bgr.copy()
    for edge in edges:
        polyline = edge.get("polyline", [])
        for start, end in zip(polyline, polyline[1:]):
            cv2.line(
                overlay,
                (int(start["col"]), int(start["row"])),
                (int(end["col"]), int(end["row"])),
                (120, 120, 120),
                1,
            )
    for item in accepted:
        bbox = item["bbox"]
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            (255, 0, 0),
            2,
        )
        if item["edge_id"] is not None:
            center_x = int(round((bbox["x_min"] + bbox["x_max"]) / 2))
            center_y = int(round((bbox["y_min"] + bbox["y_max"]) / 2))
            cv2.putText(
                overlay,
                str(item["text"])[:32],
                (center_x + 4, center_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return {
        "attachments_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "accepted": accepted,
            "rejected": rejected,
            "text_class": "line_number",
        },
        "overlay_image": overlay,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "candidate_count": len(line_number_regions),
            "accepted_attachment_count": len(accepted),
            "rejected_attachment_count": len(rejected),
            "max_distance_px": max_distance_px,
            "text_class": "line_number",
        },
    }
