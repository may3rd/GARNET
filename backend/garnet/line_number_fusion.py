from __future__ import annotations

import math
import re
from typing import Any

import cv2
import numpy as np


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


def _looks_like_line_number(text: str) -> bool:
    token = re.sub(r"\s+", "", text).upper()
    if len(token) < 6:
        return False
    has_digit = any(ch.isdigit() for ch in token)
    has_dash_or_underscore = "-" in token or "_" in token
    has_alpha = any(ch.isalpha() for ch in token)
    return has_digit and has_dash_or_underscore and has_alpha


def _candidate_text_regions(
    bbox: dict[str, int],
    text_regions: list[dict[str, Any]],
    max_distance_px: float,
) -> list[dict[str, Any]]:
    center = _center(bbox)
    candidates: list[dict[str, Any]] = []
    for region in text_regions:
        region_bbox = region["bbox"]
        ios = _bbox_ios(bbox, region_bbox)
        dist = _distance(center, _center(region_bbox))
        if ios > 0 or dist <= max_distance_px:
            candidates.append(region)
    return sorted(candidates, key=lambda item: (item["bbox"]["y_min"], item["bbox"]["x_min"]))


def _fuse_candidate_texts(
    bbox: dict[str, int],
    regions: list[dict[str, Any]],
) -> tuple[str, str | None]:
    if not regions:
        return "", None
    bbox_center_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
    bbox_height = max(1, bbox["y_max"] - bbox["y_min"])
    same_line = [
        region
        for region in regions
        if abs(((region["bbox"]["y_min"] + region["bbox"]["y_max"]) / 2.0) - bbox_center_y) <= max(12, bbox_height * 0.8)
    ]
    chosen_regions = same_line or regions[:1]
    chosen_regions = sorted(chosen_regions, key=lambda item: item["bbox"]["x_min"])
    fused_text = " ".join(str(region.get("text", "")).strip() for region in chosen_regions if str(region.get("text", "")).strip()).strip()
    fused_ids = [str(region["id"]) for region in chosen_regions]
    return fused_text, ",".join(fused_ids) if fused_ids else None


def run_line_number_fusion_stage(
    *,
    image_id: str,
    image_bgr: np.ndarray,
    object_regions: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    max_distance_px: float = 80.0,
) -> dict[str, Any]:
    line_number_objects = [obj for obj in object_regions if str(obj.get("class_name", "")).lower() == "line number"]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    confirmed_by_ocr = 0
    for idx, obj in enumerate(line_number_objects, start=1):
        bbox = obj["bbox"]
        candidates = _candidate_text_regions(bbox, text_regions, max_distance_px)
        fused_text, fused_region_ids = _fuse_candidate_texts(bbox, candidates)
        best_score = float("-inf")
        best_dist = None
        if candidates:
            best_score = max(
                (_bbox_ios(bbox, region["bbox"]) * 2.0)
                + max(0.0, 1.0 - (_distance(_center(bbox), _center(region["bbox"])) / max_distance_px))
                + float(region.get("confidence", 0.0))
                for region in candidates
            )
            best_dist = min(_distance(_center(bbox), _center(region["bbox"])) for region in candidates)

        entry = {
            "id": f"line_number_{idx:06d}",
            "source_object_id": obj["id"],
            "bbox": bbox,
            "text": "",
            "normalized_text": "",
            "ocr_region_id": None,
            "score": None,
            "distance_px": None,
            "ocr_confirmed": False,
            "detection_confidence": float(obj.get("confidence", 0.0)),
            "fused_confidence": float(obj.get("confidence", 0.0)),
            "semantic_class": "line_number",
        }
        if fused_text and _looks_like_line_number(fused_text):
            entry.update(
                {
                    "text": fused_text,
                    "normalized_text": fused_text.strip(),
                    "ocr_region_id": fused_region_ids,
                    "score": round(best_score, 4) if best_score != float("-inf") else None,
                    "distance_px": round(float(best_dist), 3) if best_dist is not None else None,
                    "ocr_confirmed": True,
                    "fused_confidence": max(float(obj.get("confidence", 0.0)), 0.95),
                }
            )
            confirmed_by_ocr += 1
        if float(obj.get("confidence", 0.0)) >= 0.5 or entry["ocr_confirmed"]:
            accepted.append(entry)
        else:
            rejected.append(entry)

    overlay = image_bgr.copy()
    for entry in accepted:
        bbox = entry["bbox"]
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            (255, 0, 0),
            2,
        )
    for entry in rejected:
        bbox = entry["bbox"]
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            (0, 0, 255),
            2,
        )

    return {
        "line_numbers_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "line_numbers": accepted,
            "rejected": rejected,
        },
        "overlay_image": overlay,
            "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "line_number_object_count": len(line_number_objects),
            "matched_line_number_count": len(accepted),
            "ocr_confirmed_line_number_count": confirmed_by_ocr,
            "od_only_line_number_count": len([item for item in accepted if not item["ocr_confirmed"]]),
            "rejected_line_number_count": len(rejected),
            "max_distance_px": max_distance_px,
        },
    }
