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


def _sample_bbox_points(bbox: dict[str, int]) -> list[tuple[float, float]]:
    x_min = float(bbox["x_min"])
    y_min = float(bbox["y_min"])
    x_max = float(bbox["x_max"])
    y_max = float(bbox["y_max"])
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return [
        (x_min, y_min),
        (x_max, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (cx, y_min),
        (cx, y_max),
        (x_min, cy),
        (x_max, cy),
        (cx, cy),
    ]


def _adaptive_attachment_threshold(region: dict[str, Any], base_threshold_px: float, text_class: str) -> float:
    if text_class != "line_number":
        return float(base_threshold_px)
    bbox = region["bbox"]
    width = max(0.0, float(bbox["x_max"]) - float(bbox["x_min"]))
    height = max(1.0, float(bbox["y_max"]) - float(bbox["y_min"]))
    normalized_text = str(region.get("normalized_text") or region.get("text") or "")
    major_digit_groups = len([m for m in normalized_text.split("-") if any(ch.isdigit() for ch in m) and len("".join(ch for ch in m if ch.isdigit())) >= 3])
    width_bonus = min(70.0, width * 0.45)
    length_bonus = min(35.0, max(0, len(normalized_text) - 18) * 1.8)
    digit_group_bonus = min(20.0, max(0, major_digit_groups - 1) * 10.0)
    slender_bonus = 12.0 if width > height * 4.0 else 0.0
    return min(180.0, float(base_threshold_px) + width_bonus + length_bonus + digit_group_bonus + slender_bonus)


def _nearest_edge(bbox: dict[str, int], edges: list[dict[str, Any]]) -> tuple[str | None, float]:
    best_edge_id = None
    best_dist = float("inf")
    sample_points = _sample_bbox_points(bbox)
    for edge in edges:
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue
        for start, end in zip(polyline, polyline[1:]):
            a = (float(start["col"]), float(start["row"]))
            b = (float(end["col"]), float(end["row"]))
            for point in sample_points:
                _, dist = _project_point_to_segment(point, a, b)
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
    text_class: str = "line_number",
) -> dict[str, Any]:
    line_number_regions = [
        item
        for item in text_regions
        if item.get("class") == text_class
        or item.get("semantic_class") == text_class
        or ("source_object_id" in item and text_class == "line_number" and item.get("semantic_class") is None)
    ]
    line_number_regions = [item for item in line_number_regions if str(item.get("text", "")).strip()]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for region in line_number_regions:
        edge_id, distance_px = _nearest_edge(region["bbox"], edges)
        threshold_px = _adaptive_attachment_threshold(region, max_distance_px, text_class)
        payload = {
            "region_id": region.get("id", region.get("source_object_id")),
            "text": region["text"],
            "normalized_text": region.get("normalized_text", ""),
            "bbox": region["bbox"],
            "edge_id": edge_id,
            "distance_px": None if math.isinf(distance_px) else round(float(distance_px), 3),
            "threshold_px": round(float(threshold_px), 3),
        }
        if edge_id is not None and distance_px <= threshold_px:
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
            "text_class": text_class,
        },
        "overlay_image": overlay,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "candidate_count": len(line_number_regions),
            "accepted_attachment_count": len(accepted),
            "rejected_attachment_count": len(rejected),
            "max_distance_px": max_distance_px,
            "text_class": text_class,
        },
    }


def render_text_attachment_overlay(
    *,
    image_bgr: np.ndarray,
    edges: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
) -> np.ndarray:
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
    for item in attachments:
        bbox = item["bbox"]
        label = str(item.get("text", ""))[:32]
        color = (255, 0, 0)
        if item.get("semantic_class") == "instrument_semantic":
            color = (0, 165, 255)
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            color,
            2,
        )
        center_x = int(round((bbox["x_min"] + bbox["x_max"]) / 2))
        center_y = int(round((bbox["y_min"] + bbox["y_max"]) / 2))
        cv2.putText(
            overlay,
            label,
            (center_x + 4, center_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay
