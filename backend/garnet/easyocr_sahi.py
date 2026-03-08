from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
from sahi.slicing import slice_image

warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

_READER_CACHE: dict[tuple[tuple[str, ...], bool], easyocr.Reader] = {}


@dataclass(frozen=True)
class EasyOcrSahiConfig:
    languages: tuple[str, ...] = ("en",)
    use_gpu: bool = False
    slice_height: int = 1600
    slice_width: int = 1600
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    min_score: float = 0.2
    min_text_len: int = 2
    text_threshold: float = 0.7
    low_text: float = 0.3
    link_threshold: float = 0.7
    line_merge_gap_px: int = 24
    line_merge_y_tolerance_px: int = 10
    enable_rotated_ocr: bool = True
    paragraph: bool = False


def _get_reader(cfg: EasyOcrSahiConfig) -> easyocr.Reader:
    key = (cfg.languages, cfg.use_gpu)
    reader = _READER_CACHE.get(key)
    if reader is None:
        reader = easyocr.Reader(list(cfg.languages), gpu=cfg.use_gpu)
        _READER_CACHE[key] = reader
    return reader


def _bbox_from_quad(quad: list[list[float]]) -> dict[str, int]:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return {
        "x_min": int(round(min(xs))),
        "y_min": int(round(min(ys))),
        "x_max": int(round(max(xs))),
        "y_max": int(round(max(ys))),
    }


def _bbox_iou(a: dict[str, int], b: dict[str, int]) -> float:
    inter_x1 = max(a["x_min"], b["x_min"])
    inter_y1 = max(a["y_min"], b["y_min"])
    inter_x2 = min(a["x_max"], b["x_max"])
    inter_y2 = min(a["y_max"], b["y_max"])
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, a["x_max"] - a["x_min"]) * max(0, a["y_max"] - a["y_min"])
    area_b = max(0, b["x_max"] - b["x_min"]) * max(0, b["y_max"] - b["y_min"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _rotate_image(tile: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "none":
        return tile
    if orientation == "cw":
        return cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "ccw":
        return cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported orientation: {orientation}")


def _rotate_quad_back(quad: list[list[float]], orientation: str, original_width: int, original_height: int) -> list[list[float]]:
    restored: list[list[float]] = []
    for x_rot, y_rot in quad:
        if orientation == "none":
            x_orig, y_orig = x_rot, y_rot
        elif orientation == "cw":
            x_orig = y_rot
            y_orig = original_height - 1 - x_rot
        elif orientation == "ccw":
            x_orig = original_width - 1 - y_rot
            y_orig = x_rot
        else:
            raise ValueError(f"Unsupported orientation: {orientation}")
        restored.append([float(x_orig), float(y_orig)])
    return restored


def _region_direction(bbox: dict[str, int], rotation: int) -> str:
    width = max(1, bbox["x_max"] - bbox["x_min"])
    height = max(1, bbox["y_max"] - bbox["y_min"])
    if rotation not in {0, 180}:
        return "rotated"
    if height > width * 1.4:
        return "vertical"
    if width > height * 1.4:
        return "horizontal"
    return "unknown"


def _region_legibility(score: float, text: str, min_text_len: int) -> str:
    if not text.strip():
        return "illegible"
    if len(text.strip()) < min_text_len:
        return "partial"
    if score >= 0.8:
        return "clear"
    if score >= 0.45:
        return "degraded"
    return "partial"


def _safe_normalized_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized if normalized == text.strip() else ""


def _classify_text(text: str) -> str:
    token = text.strip()
    upper = token.upper()
    if not token:
        return "unknown"
    if upper in {"STEAM", "CW", "AIR", "N2", "O2", "COND", "FG"}:
        return "utility_label"
    if re.fullmatch(r"[A-Z0-9\"'./()-]+", upper) and "-" in upper and any(ch.isdigit() for ch in upper):
        return "line_number"
    if re.fullmatch(r"[A-Z]{1,4}-?\d{1,4}[A-Z]?", upper):
        return "instrument_tag"
    if len(token.split()) >= 3:
        return "note"
    return "unknown"


def _exception_reasons(region: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if region["confidence"] < 0.65:
        reasons.append("low_confidence")
    if region["legibility"] in {"partial", "illegible"}:
        reasons.append("low_legibility")
    if region["reading_direction"] in {"vertical", "rotated", "unknown"}:
        reasons.append("non_horizontal_text")
    if region["class"] == "unknown":
        reasons.append("unknown_class")
    if region["normalized_text"] == "":
        reasons.append("unsafe_normalization")
    return reasons


def _line_merge_gap_allowed(left: dict[str, Any], right: dict[str, Any], line_merge_gap_px: int) -> int:
    left_height = max(1, left["bbox"]["y_max"] - left["bbox"]["y_min"])
    right_height = max(1, right["bbox"]["y_max"] - right["bbox"]["y_min"])
    return max(line_merge_gap_px, int(max(left_height, right_height) * 0.75))


def _can_merge_same_line(left: dict[str, Any], right: dict[str, Any], line_merge_gap_px: int, line_merge_y_tolerance_px: int) -> bool:
    if left["rotation"] != right["rotation"]:
        return False
    if left["reading_direction"] not in {"horizontal", "unknown"}:
        return False
    if right["reading_direction"] not in {"horizontal", "unknown"}:
        return False
    left_center_y = (left["bbox"]["y_min"] + left["bbox"]["y_max"]) / 2
    right_center_y = (right["bbox"]["y_min"] + right["bbox"]["y_max"]) / 2
    if abs(left_center_y - right_center_y) > line_merge_y_tolerance_px:
        return False
    gap = right["bbox"]["x_min"] - left["bbox"]["x_max"]
    if gap < 0:
        return False
    return gap <= _line_merge_gap_allowed(left, right, line_merge_gap_px)


def _merge_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged_text = f"{left['text'].rstrip()} {right['text'].lstrip()}".strip()
    merged_bbox = {
        "x_min": min(left["bbox"]["x_min"], right["bbox"]["x_min"]),
        "y_min": min(left["bbox"]["y_min"], right["bbox"]["y_min"]),
        "x_max": max(left["bbox"]["x_max"], right["bbox"]["x_max"]),
        "y_max": max(left["bbox"]["y_max"], right["bbox"]["y_max"]),
    }
    merged_confidence = round(max(float(left["confidence"]), float(right["confidence"])), 4)
    merged_region = {
        "id": "",
        "text": merged_text,
        "normalized_text": _safe_normalized_text(merged_text),
        "class": _classify_text(merged_text),
        "confidence": merged_confidence,
        "bbox": merged_bbox,
        "rotation": left["rotation"],
        "reading_direction": _region_direction(merged_bbox, left["rotation"]),
        "legibility": _region_legibility(merged_confidence, merged_text, 2),
    }
    return merged_region


def _merge_same_line_regions(
    regions: list[dict[str, Any]],
    line_merge_gap_px: int,
    line_merge_y_tolerance_px: int,
) -> list[dict[str, Any]]:
    if not regions:
        return []
    sorted_by_y = sorted(
        regions,
        key=lambda item: (
            (item["bbox"]["y_min"] + item["bbox"]["y_max"]) / 2,
            item["bbox"]["x_min"],
            item["text"],
        ),
    )
    lines: list[dict[str, Any]] = []
    for region in sorted_by_y:
        center_y = (region["bbox"]["y_min"] + region["bbox"]["y_max"]) / 2
        assigned = False
        for line in lines:
            if abs(center_y - line["center_y"]) <= line_merge_y_tolerance_px:
                line["regions"].append(region)
                line["center_y"] = sum(
                    (item["bbox"]["y_min"] + item["bbox"]["y_max"]) / 2 for item in line["regions"]
                ) / len(line["regions"])
                assigned = True
                break
        if not assigned:
            lines.append({"center_y": center_y, "regions": [region]})

    ordered: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: item["center_y"]):
        ordered.extend(sorted(line["regions"], key=lambda item: (item["bbox"]["x_min"], item["text"])))

    merged: list[dict[str, Any]] = []
    current = dict(ordered[0])
    for next_region in ordered[1:]:
        if _can_merge_same_line(current, next_region, line_merge_gap_px, line_merge_y_tolerance_px):
            current = _merge_pair(current, next_region)
        else:
            merged.append(current)
            current = dict(next_region)
    merged.append(current)
    return merged


def _draw_overlay(image_bgr: np.ndarray, regions: list[dict[str, Any]]) -> np.ndarray:
    overlay = image_bgr.copy()
    for region in regions:
        bbox = region["bbox"]
        cv2.rectangle(
            overlay,
            (bbox["x_min"], bbox["y_min"]),
            (bbox["x_max"], bbox["y_max"]),
            (0, 255, 255),
            2,
        )
    return overlay


def _read_tile_with_orientations(reader: easyocr.Reader, tile: np.ndarray, cfg: EasyOcrSahiConfig) -> list[tuple[str, list[Any]]]:
    orientations = ["none"]
    if cfg.enable_rotated_ocr:
        orientations.extend(["cw", "ccw"])
    oriented_results: list[tuple[str, list[Any]]] = []
    for orientation in orientations:
        oriented_tile = _rotate_image(tile, orientation)
        results = reader.readtext(
            oriented_tile,
            detail=1,
            paragraph=cfg.paragraph,
            text_threshold=cfg.text_threshold,
            low_text=cfg.low_text,
            link_threshold=cfg.link_threshold,
        )
        oriented_results.append((orientation, results))
    return oriented_results


def run_easyocr_sahi(image_path: str | Path, image_id: str = "", cfg: EasyOcrSahiConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EasyOcrSahiConfig()
    reader = _get_reader(cfg)
    image_path = str(image_path)
    base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if base_image is None:
        raise FileNotFoundError(f"Cannot read image for OCR: {image_path}")

    slice_result = slice_image(
        image=image_path,
        slice_height=cfg.slice_height,
        slice_width=cfg.slice_width,
        overlap_height_ratio=cfg.overlap_height_ratio,
        overlap_width_ratio=cfg.overlap_width_ratio,
        auto_slice_resolution=False,
        verbose=False,
    )

    merged_regions: list[dict[str, Any]] = []
    tile_count = 0
    raw_detection_count = 0

    for tile, starting_pixel in zip(slice_result.images, slice_result.starting_pixels):
        tile_count += 1
        shift_x, shift_y = int(starting_pixel[0]), int(starting_pixel[1])
        tile_height, tile_width = tile.shape[:2]
        for orientation, results in _read_tile_with_orientations(reader, tile, cfg):
            rotation = 0 if orientation == "none" else 90
            for quad, text, score in results:
                if float(score) < cfg.min_score:
                    continue
                if len(text.strip()) < cfg.min_text_len:
                    continue
                raw_detection_count += 1
                restored_quad = _rotate_quad_back(
                    [[float(point[0]), float(point[1])] for point in quad],
                    orientation,
                    tile_width,
                    tile_height,
                )
                shifted_quad = [[float(point[0] + shift_x), float(point[1] + shift_y)] for point in restored_quad]
                bbox = _bbox_from_quad(shifted_quad)
                region = {
                    "id": "",
                    "text": text,
                    "normalized_text": _safe_normalized_text(text),
                    "class": _classify_text(text),
                    "confidence": round(float(score), 4),
                    "bbox": bbox,
                    "rotation": rotation,
                    "reading_direction": _region_direction(bbox, rotation),
                    "legibility": _region_legibility(float(score), text, cfg.min_text_len),
                }
                duplicate_idx = None
                for idx, existing in enumerate(merged_regions):
                    if existing["text"].strip() == region["text"].strip() and _bbox_iou(existing["bbox"], region["bbox"]) >= 0.3:
                        duplicate_idx = idx
                        break
                if duplicate_idx is None:
                    merged_regions.append(region)
                elif region["confidence"] > merged_regions[duplicate_idx]["confidence"]:
                    merged_regions[duplicate_idx] = region

    merged_regions = _merge_same_line_regions(
        merged_regions,
        line_merge_gap_px=cfg.line_merge_gap_px,
        line_merge_y_tolerance_px=cfg.line_merge_y_tolerance_px,
    )
    merged_regions.sort(key=lambda item: (item["bbox"]["y_min"], item["bbox"]["x_min"], item["text"]))
    for idx, region in enumerate(merged_regions, start=1):
        region["id"] = f"ocr_{idx:06d}"

    exception_candidates = []
    for region in merged_regions:
        reasons = _exception_reasons(region)
        if not reasons:
            continue
        exception_candidates.append(
            {
                "source_region_id": region["id"],
                "reasons": reasons,
                "sheet_bbox": region["bbox"],
                "crop_hint": region["bbox"],
                "text": region["text"],
                "confidence": region["confidence"],
            }
        )

    return {
        "regions_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "text_regions": merged_regions,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "tile_count": tile_count,
            "raw_detection_count": raw_detection_count,
            "merged_region_count": len(merged_regions),
            "exception_candidate_count": len(exception_candidates),
            "slice_height": cfg.slice_height,
            "slice_width": cfg.slice_width,
            "overlap_height_ratio": cfg.overlap_height_ratio,
            "overlap_width_ratio": cfg.overlap_width_ratio,
            "rotated_ocr_enabled": cfg.enable_rotated_ocr,
            "line_merge_gap_px": cfg.line_merge_gap_px,
            "line_merge_y_tolerance_px": cfg.line_merge_y_tolerance_px,
        },
        "exception_candidates": exception_candidates,
        "overlay_image": _draw_overlay(base_image, merged_regions),
    }
