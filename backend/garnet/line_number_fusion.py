from __future__ import annotations

import math
import platform
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image


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


def _bbox_gap(a: dict[str, int], b: dict[str, int]) -> float:
    dx = max(0, max(a["x_min"] - b["x_max"], b["x_min"] - a["x_max"]))
    dy = max(0, max(a["y_min"] - b["y_max"], b["y_min"] - a["y_max"]))
    return math.hypot(dx, dy)


def _looks_like_line_number(text: str) -> bool:
    token = _normalize_line_number(text)
    if len(token) < 6:
        return False
    has_digit = any(ch.isdigit() for ch in token)
    has_dash = "-" in token
    has_alpha = any(ch.isalpha() for ch in token)
    has_major_digit_group = re.search(r"\d{3,}", token) is not None
    return has_digit and has_dash and has_alpha and has_major_digit_group


def _looks_like_line_number_fragment(text: str) -> bool:
    token = _normalize_line_number(text)
    if len(token) < 4:
        return False
    has_digit = any(ch.isdigit() for ch in token)
    has_dash = "-" in token
    has_alpha = any(ch.isalpha() for ch in token)
    return has_digit and has_dash and has_alpha


def _normalize_line_number(text: str) -> str:
    normalized = text.upper().strip()
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized


def _get_ocrmac_module() -> Any:
    if platform.system() != "Darwin":
        return None
    try:
        from ocrmac import ocrmac as module  # type: ignore
    except Exception:  # pragma: no cover
        return None
    return module


def _crop_image(image_bgr: np.ndarray, bbox: dict[str, int], padding: int = 4) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    x_min = max(0, int(bbox["x_min"]) - padding)
    y_min = max(0, int(bbox["y_min"]) - padding)
    x_max = min(width, int(bbox["x_max"]) + padding)
    y_max = min(height, int(bbox["y_max"]) + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    crop = image_bgr[y_min:y_max, x_min:x_max]
    return crop if crop.size else None


def _enhance_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.GaussianBlur(upscaled, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _parse_ocrmac_annotations(annotations: list[tuple[Any, Any, Any]]) -> list[str]:
    texts: list[str] = []
    for item in annotations:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        text = str(item[0]).strip()
        if text:
            texts.append(text)
    return texts


def _assemble_line_number_from_parts(parts: list[str]) -> str:
    normalized_parts = [_normalize_line_number(part) for part in parts if str(part).strip()]
    normalized_parts = [part for part in normalized_parts if part]
    if not normalized_parts:
        return ""

    candidate = "".join(normalized_parts)
    if _looks_like_line_number(candidate):
        return _normalize_line_number(candidate)

    candidate = "-".join(normalized_parts)
    if _looks_like_line_number(candidate):
        return _normalize_line_number(candidate)

    for part in normalized_parts:
        if _looks_like_line_number(part):
            return _normalize_line_number(part)
    return ""


def _confirm_with_crop_ocr(image_bgr: np.ndarray, bbox: dict[str, int]) -> str:
    crop = _crop_image(image_bgr, bbox)
    if crop is None:
        return ""
    ocrmac = _get_ocrmac_module()
    if ocrmac is None:
        return ""
    crops = [("crop_ocr", crop)]
    if crop.shape[0] > max(32, crop.shape[1] * 1.2):
        rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        crops.append(("crop_ocr_rotated", rotated))
        crops.append(("crop_ocr_rotated_preprocessed", _enhance_crop_for_ocr(rotated)))
    crops.append(("crop_ocr_preprocessed", _enhance_crop_for_ocr(crop)))

    for source, crop_view in crops:
        annotations = ocrmac.OCR(
            Image.fromarray(cv2.cvtColor(crop_view, cv2.COLOR_BGR2RGB)),
            recognition_level="accurate",
            framework="vision",
            language_preference=["en-US"],
        ).recognize()
        parts = _parse_ocrmac_annotations(annotations)
        assembled = _assemble_line_number_from_parts(parts)
        if assembled:
            return source, assembled
    return "", ""


def _candidate_text_regions(
    bbox: dict[str, int],
    text_regions: list[dict[str, Any]],
    max_distance_px: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for region in text_regions:
        region_text = str(region.get("text", "")).strip()
        region_class = str(region.get("class", "")).lower()
        if not region_text:
            continue
        if not (_looks_like_line_number_fragment(region_text) or region_class == "line_number"):
            continue
        region_bbox = region["bbox"]
        ios = _bbox_ios(bbox, region_bbox)
        gap = _bbox_gap(bbox, region_bbox)
        if ios > 0 or gap <= max_distance_px:
            candidates.append(region)
    return sorted(candidates, key=lambda item: (round(_bbox_gap(bbox, item["bbox"]), 3), item["bbox"]["y_min"], item["bbox"]["x_min"]))


def _fuse_candidate_texts(
    bbox: dict[str, int],
    regions: list[dict[str, Any]],
) -> tuple[str, str | None]:
    if not regions:
        return "", None
    bbox_center_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
    bbox_center_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
    bbox_width = max(1, bbox["x_max"] - bbox["x_min"])
    bbox_height = max(1, bbox["y_max"] - bbox["y_min"])
    is_vertical = bbox_height > bbox_width * 1.5
    if is_vertical:
        same_track = [
            region
            for region in regions
            if abs(((region["bbox"]["x_min"] + region["bbox"]["x_max"]) / 2.0) - bbox_center_x) <= max(14, bbox_width * 1.2)
        ]
        chosen_regions = same_track or regions[:1]
        chosen_regions = sorted(chosen_regions, key=lambda item: item["bbox"]["y_min"])
    else:
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
            "ocr_source": None,
            "score": None,
            "distance_px": None,
            "ocr_confirmed": False,
            "detection_confidence": float(obj.get("confidence", 0.0)),
            "fused_confidence": float(obj.get("confidence", 0.0)),
            "semantic_class": "line_number",
            "review_state": "detection_only",
        }
        if fused_text and _looks_like_line_number(fused_text):
            entry.update(
                {
                    "text": fused_text,
                    "normalized_text": _normalize_line_number(fused_text),
                    "ocr_region_id": fused_region_ids,
                    "ocr_source": "sheet_ocr",
                    "score": round(best_score, 4) if best_score != float("-inf") else None,
                    "distance_px": round(float(best_dist), 3) if best_dist is not None else None,
                    "ocr_confirmed": True,
                    "fused_confidence": max(float(obj.get("confidence", 0.0)), 0.95),
                    "review_state": "ocr_confirmed",
                }
            )
            confirmed_by_ocr += 1
        else:
            crop_source, crop_text = _confirm_with_crop_ocr(image_bgr, bbox)
            if crop_text and _looks_like_line_number(crop_text):
                entry.update(
                    {
                        "text": crop_text,
                        "normalized_text": _normalize_line_number(crop_text),
                        "ocr_region_id": crop_source,
                        "ocr_source": crop_source,
                        "ocr_confirmed": True,
                        "fused_confidence": max(float(obj.get("confidence", 0.0)), 0.95),
                        "review_state": "ocr_confirmed",
                    }
                )
                confirmed_by_ocr += 1
        if float(obj.get("confidence", 0.0)) >= 0.5 or entry["ocr_confirmed"]:
            accepted.append(entry)
        else:
            entry["review_state"] = "rejected"
            rejected.append(entry)

    overlay = image_bgr.copy()
    for entry in accepted:
        bbox = entry["bbox"]
        color = (0, 165, 255)
        if entry.get("ocr_source") == "sheet_ocr":
            color = (255, 0, 0)
        elif entry.get("ocr_source") in {"crop_ocr", "crop_ocr_preprocessed"}:
            color = (0, 200, 0)
        elif str(entry.get("ocr_source", "")).startswith("crop_ocr_rotated"):
            color = (0, 220, 220)
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            color,
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
            "sheet_ocr_line_number_count": len([item for item in accepted if item.get("ocr_source") == "sheet_ocr"]),
            "crop_ocr_line_number_count": len(
                [item for item in accepted if item.get("ocr_source") in {"crop_ocr", "crop_ocr_preprocessed"}]
            ),
            "rotated_crop_ocr_line_number_count": len(
                [item for item in accepted if str(item.get("ocr_source", "")).startswith("crop_ocr_rotated")]
            ),
            "rejected_line_number_count": len(rejected),
            "max_distance_px": max_distance_px,
        },
    }
