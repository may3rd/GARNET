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


def _looks_like_instrument_tag(text: str) -> bool:
    normalized = text.upper().strip()
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^A-Z0-9-]", "", normalized)
    return re.fullmatch(r"[A-Z]{2,4}-?\d{3,4}[A-Z]?", normalized) is not None


def _normalize_instrument_tag(text: str) -> str:
    normalized = text.upper().strip()
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^A-Z0-9-]", "", normalized)
    match = re.fullmatch(r"([A-Z]{2,4})-?(\d{3,4})([A-Z]?)", normalized)
    if not match:
        return normalized
    prefix, digits, suffix = match.groups()
    return f"{prefix}-{digits}{suffix}"


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


def _assemble_instrument_tag_from_parts(parts: list[str]) -> str:
    normalized_parts = [
        re.sub(r"[^A-Z0-9-]", "", part.upper().replace("_", "-"))
        for part in parts
        if part.strip()
    ]
    normalized_parts = [part for part in normalized_parts if part]
    if not normalized_parts:
        return ""

    direct = "".join(normalized_parts)
    if _looks_like_instrument_tag(direct):
        return _normalize_instrument_tag(direct)

    alpha_parts = [part for part in normalized_parts if re.fullmatch(r"[A-Z]{2,4}", part)]
    digit_parts = [part for part in normalized_parts if re.fullmatch(r"\d{3,4}[A-Z]?", part)]
    if alpha_parts and digit_parts:
        candidate = f"{alpha_parts[0]}-{digit_parts[0]}"
        if _looks_like_instrument_tag(candidate):
            return _normalize_instrument_tag(candidate)

    for part in normalized_parts:
        if _looks_like_instrument_tag(part):
            return _normalize_instrument_tag(part)
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
        assembled = _assemble_instrument_tag_from_parts(parts)
        if assembled:
            return source, assembled
    return "", ""


def _candidate_text_regions(bbox: dict[str, int], text_regions: list[dict[str, Any]], max_distance_px: float) -> list[dict[str, Any]]:
    center = _center(bbox)
    candidates: list[dict[str, Any]] = []
    for region in text_regions:
        region_bbox = region["bbox"]
        ios = _bbox_ios(bbox, region_bbox)
        dist = _distance(center, _center(region_bbox))
        if ios > 0 or dist <= max_distance_px:
            candidates.append(region)
    return sorted(candidates, key=lambda item: (item["bbox"]["y_min"], item["bbox"]["x_min"]))


def _fuse_candidate_texts(bbox: dict[str, int], regions: list[dict[str, Any]]) -> tuple[str, str | None]:
    if not regions:
        return "", None
    bbox_center_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
    bbox_height = max(1, bbox["y_max"] - bbox["y_min"])
    same_line = [
        region
        for region in regions
        if abs(((region["bbox"]["y_min"] + region["bbox"]["y_max"]) / 2.0) - bbox_center_y) <= max(12, bbox_height)
    ]
    chosen_regions = same_line or regions[:1]
    chosen_regions = sorted(chosen_regions, key=lambda item: item["bbox"]["x_min"])
    fused_text = " ".join(str(region.get("text", "")).strip() for region in chosen_regions if str(region.get("text", "")).strip()).strip()
    fused_ids = [str(region["id"]) for region in chosen_regions]
    return fused_text, ",".join(fused_ids) if fused_ids else None


def run_instrument_tag_fusion_stage(
    *,
    image_id: str,
    image_bgr: np.ndarray,
    object_regions: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    max_distance_px: float = 60.0,
) -> dict[str, Any]:
    objects = [
        obj
        for obj in object_regions
        if str(obj.get("class_name", "")).lower() in {"instrument tag", "instrument dcs", "instrument logic"}
    ]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    confirmed_by_ocr = 0
    detection_only_count = 0

    for idx, obj in enumerate(objects, start=1):
        bbox = obj["bbox"]
        candidates = _candidate_text_regions(bbox, text_regions, max_distance_px)
        fused_text, fused_region_ids = _fuse_candidate_texts(bbox, candidates)
        entry = {
            "id": f"instrument_tag_{idx:06d}",
            "source_object_id": obj["id"],
            "bbox": bbox,
            "text": "",
            "normalized_text": "",
            "ocr_region_id": None,
            "ocr_source": None,
            "ocr_confirmed": False,
            "detection_confidence": float(obj.get("confidence", 0.0)),
            "fused_confidence": float(obj.get("confidence", 0.0)),
            "semantic_class": "instrument_semantic",
            "source_object_class": str(obj.get("class_name", "")).lower(),
            "review_state": "detection_only",
        }
        if fused_text and _looks_like_instrument_tag(fused_text):
            entry.update(
                {
                    "text": fused_text,
                    "normalized_text": _normalize_instrument_tag(fused_text),
                    "ocr_region_id": fused_region_ids,
                    "ocr_source": "sheet_ocr",
                    "ocr_confirmed": True,
                    "fused_confidence": max(float(obj.get("confidence", 0.0)), 0.95),
                    "review_state": "ocr_confirmed",
                }
            )
            confirmed_by_ocr += 1
        else:
            crop_source, crop_text = _confirm_with_crop_ocr(image_bgr, bbox)
            if crop_text and _looks_like_instrument_tag(crop_text):
                entry.update(
                    {
                        "text": crop_text,
                        "normalized_text": _normalize_instrument_tag(crop_text),
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
            if not entry["ocr_confirmed"]:
                detection_only_count += 1
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
        cv2.rectangle(overlay, (bbox["x_min"], bbox["y_min"]), (bbox["x_max"], bbox["y_max"]), color, 2)
    for entry in rejected:
        bbox = entry["bbox"]
        cv2.rectangle(overlay, (bbox["x_min"], bbox["y_min"]), (bbox["x_max"], bbox["y_max"]), (0, 0, 255), 2)

    return {
        "instrument_tags_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "instrument_tags": accepted,
            "rejected": rejected,
        },
        "overlay_image": overlay,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "instrument_semantic_object_count": len(objects),
            "matched_instrument_semantic_count": len(accepted),
            "ocr_confirmed_instrument_semantic_count": confirmed_by_ocr,
            "detection_only_instrument_semantic_count": detection_only_count,
            "sheet_ocr_instrument_semantic_count": len([item for item in accepted if item.get("ocr_source") == "sheet_ocr"]),
            "crop_ocr_instrument_semantic_count": len(
                [item for item in accepted if item.get("ocr_source") in {"crop_ocr", "crop_ocr_preprocessed"}]
            ),
            "rotated_crop_ocr_instrument_semantic_count": len(
                [item for item in accepted if str(item.get("ocr_source", "")).startswith("crop_ocr_rotated")]
            ),
            "rejected_instrument_semantic_count": len(rejected),
            "max_distance_px": max_distance_px,
        },
    }
