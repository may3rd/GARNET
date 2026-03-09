from __future__ import annotations

import math
from typing import Any


def _bbox_ios(a: dict[str, int], b: dict[str, int]) -> float:
    inter_x1 = max(a["x_min"], b["x_min"])
    inter_y1 = max(a["y_min"], b["y_min"])
    inter_x2 = min(a["x_max"], b["x_max"])
    inter_y2 = min(a["y_max"], b["y_max"])
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, a["x_max"] - a["x_min"]) * max(0, a["y_max"] - a["y_min"])
    area_b = max(0, b["x_max"] - b["x_min"]) * max(0, b["y_max"] - b["y_min"])
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def _center(bbox: dict[str, int]) -> tuple[float, float]:
    return ((bbox["x_min"] + bbox["x_max"]) / 2.0, (bbox["y_min"] + bbox["y_max"]) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def run_line_number_fusion_stage(
    *,
    image_id: str,
    object_regions: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    max_distance_px: float = 80.0,
) -> dict[str, Any]:
    line_number_objects = [obj for obj in object_regions if str(obj.get("class_name", "")).lower() == "line number"]
    line_number_texts = [region for region in text_regions if str(region.get("class", "")).lower() == "line_number"]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx, obj in enumerate(line_number_objects, start=1):
        bbox = obj["bbox"]
        obj_center = _center(bbox)
        best = None
        best_score = float("-inf")
        for region in line_number_texts:
            region_bbox = region["bbox"]
            ios = _bbox_ios(bbox, region_bbox)
            dist = _distance(obj_center, _center(region_bbox))
            distance_score = max(0.0, 1.0 - (dist / max_distance_px))
            score = (ios * 2.0) + distance_score + float(region.get("confidence", 0.0))
            if score > best_score:
                best_score = score
                best = (region, ios, dist)

        entry = {
            "id": f"line_number_{idx:06d}",
            "source_object_id": obj["id"],
            "bbox": bbox,
            "text": "",
            "normalized_text": "",
            "ocr_region_id": None,
            "score": None,
            "distance_px": None,
        }
        if best is not None:
            region, ios, dist = best
            if ios > 0 or dist <= max_distance_px:
                entry.update(
                    {
                        "bbox": region["bbox"],
                        "text": region["text"],
                        "normalized_text": region.get("normalized_text", ""),
                        "ocr_region_id": region["id"],
                        "score": round(best_score, 4),
                        "distance_px": round(dist, 3),
                    }
                )
                accepted.append(entry)
                continue
        rejected.append(entry)

    return {
        "line_numbers_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "line_numbers": accepted,
            "rejected": rejected,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "line_number_object_count": len(line_number_objects),
            "matched_line_number_count": len(accepted),
            "rejected_line_number_count": len(rejected),
            "max_distance_px": max_distance_px,
        },
    }
