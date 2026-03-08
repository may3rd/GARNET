from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

_READER_CACHE: dict[tuple[tuple[str, ...], bool], easyocr.Reader] = {}
OCR_CLASSES = [
    "equipment_tag",
    "line_number",
    "instrument_tag",
    "valve_tag",
    "utility_label",
    "process_label",
    "note",
    "dimension",
    "title_block",
    "table_text",
    "legend_text",
    "unknown",
]
CATEGORY_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(OCR_CLASSES)}
CATEGORY_ID_TO_NAME = {idx + 1: name for idx, name in enumerate(OCR_CLASSES)}


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
    postprocess_type: str = "GREEDYNMM"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.1


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
            (255, 0, 0),
            2,
        )
    return overlay


def _choose_best_candidate(
    merged_bbox: dict[str, int],
    candidates: list[dict[str, Any]],
    match_threshold: float,
) -> dict[str, Any] | None:
    matches = [candidate for candidate in candidates if _bbox_ios(merged_bbox, candidate["bbox"]) >= match_threshold]
    if not matches:
        return None
    return max(matches, key=lambda item: (float(item.get("confidence", 0.0)), _bbox_ios(merged_bbox, item["bbox"])))


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


class EasyOcrSahiDetectionModel(DetectionModel):
    required_packages = ["easyocr", "numpy", "opencv-python", "sahi"]

    def __init__(self, *, cfg: EasyOcrSahiConfig, reader: easyocr.Reader) -> None:
        self.cfg = cfg
        self.reader = reader
        self.all_candidates: list[dict[str, Any]] = []
        self.slice_count = 0
        self._current_slice_shape: tuple[int, int] = (0, 0)
        super().__init__(
            load_at_init=False,
            confidence_threshold=cfg.min_score,
            category_mapping={str(idx): name for idx, name in CATEGORY_ID_TO_NAME.items()},
        )

    def load_model(self) -> None:
        self.model = self.reader

    def perform_inference(self, image: np.ndarray) -> None:
        self._current_slice_shape = image.shape[:2]
        local_regions: list[dict[str, Any]] = []
        tile_height, tile_width = image.shape[:2]
        for orientation, results in _read_tile_with_orientations(self.reader, image, self.cfg):
            rotation = 0 if orientation == "none" else 90
            for quad, text, score in results:
                if float(score) < self.cfg.min_score:
                    continue
                if len(text.strip()) < self.cfg.min_text_len:
                    continue
                restored_quad = _rotate_quad_back(
                    [[float(point[0]), float(point[1])] for point in quad],
                    orientation,
                    tile_width,
                    tile_height,
                )
                bbox = _bbox_from_quad(restored_quad)
                local_regions.append(
                    {
                        "text": text,
                        "normalized_text": _safe_normalized_text(text),
                        "class": _classify_text(text),
                        "confidence": round(float(score), 4),
                        "bbox": bbox,
                        "rotation": rotation,
                        "reading_direction": _region_direction(bbox, rotation),
                        "legibility": _region_legibility(float(score), text, self.cfg.min_text_len),
                    }
                )
        self._original_predictions = local_regions

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ) -> None:
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        shift_amount = shift_amount_list[0]
        full_shape = full_shape_list[0] if full_shape_list else None
        self.slice_count += 1

        object_predictions: list[ObjectPrediction] = []
        for region in self._original_predictions:
            bbox = region["bbox"]
            bbox_xyxy = [
                int(bbox["x_min"]),
                int(bbox["y_min"]),
                int(bbox["x_max"]),
                int(bbox["y_max"]),
            ]
            class_name = region["class"] if region["class"] in CATEGORY_NAME_TO_ID else "unknown"
            category_id = CATEGORY_NAME_TO_ID[class_name]
            object_predictions.append(
                ObjectPrediction(
                    bbox=bbox_xyxy,
                    category_id=category_id,
                    category_name=CATEGORY_ID_TO_NAME[category_id],
                    score=float(region["confidence"]),
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
            )
            self.all_candidates.append(
                {
                    **region,
                    "class": class_name,
                    "bbox": {
                        "x_min": bbox_xyxy[0] + int(shift_amount[0]),
                        "y_min": bbox_xyxy[1] + int(shift_amount[1]),
                        "x_max": bbox_xyxy[2] + int(shift_amount[0]),
                        "y_max": bbox_xyxy[3] + int(shift_amount[1]),
                    },
                }
            )

        self._object_prediction_list_per_image = [object_predictions]
        self._object_prediction_list = object_predictions


def run_easyocr_sahi(image_path: str | Path, image_id: str = "", cfg: EasyOcrSahiConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EasyOcrSahiConfig()
    reader = _get_reader(cfg)
    image_path = str(image_path)
    base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if base_image is None:
        raise FileNotFoundError(f"Cannot read image for OCR: {image_path}")
    ocr_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if ocr_image is None:
        raise FileNotFoundError(f"Cannot read image for OCR: {image_path}")

    detector = EasyOcrSahiDetectionModel(cfg=cfg, reader=reader)
    prediction_result = get_sliced_prediction(
        image=ocr_image,
        detection_model=detector,
        slice_height=cfg.slice_height,
        slice_width=cfg.slice_width,
        overlap_height_ratio=cfg.overlap_height_ratio,
        overlap_width_ratio=cfg.overlap_width_ratio,
        perform_standard_pred=False,
        postprocess_type=cfg.postprocess_type,
        postprocess_match_metric=cfg.postprocess_match_metric,
        postprocess_match_threshold=cfg.postprocess_match_threshold,
        verbose=0,
        auto_slice_resolution=False,
    )

    sahi_merged_regions: list[dict[str, Any]] = []
    for object_prediction in prediction_result.object_prediction_list:
        bbox_xyxy = object_prediction.bbox.to_xyxy()
        merged_bbox = {
            "x_min": int(bbox_xyxy[0]),
            "y_min": int(bbox_xyxy[1]),
            "x_max": int(bbox_xyxy[2]),
            "y_max": int(bbox_xyxy[3]),
        }
        chosen = _choose_best_candidate(
            merged_bbox,
            detector.all_candidates,
            match_threshold=cfg.postprocess_match_threshold,
        )
        if chosen is None:
            chosen = {
                "text": "",
                "normalized_text": "",
                "class": object_prediction.category.name,
                "confidence": float(object_prediction.score.value),
                "rotation": 0,
                "reading_direction": "unknown",
                "legibility": "illegible",
            }
        sahi_merged_regions.append(
            {
                "id": "",
                "text": chosen["text"],
                "normalized_text": chosen["normalized_text"],
                "class": chosen["class"],
                "confidence": round(float(chosen["confidence"]), 4),
                "bbox": merged_bbox,
                "rotation": chosen["rotation"],
                "reading_direction": chosen["reading_direction"],
                "legibility": chosen["legibility"],
            }
        )

    merged_regions = _merge_same_line_regions(
        sahi_merged_regions,
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
            "route": "easyocr",
            "tile_count": detector.slice_count,
            "raw_detection_count": len(detector.all_candidates),
            "merged_region_count": len(merged_regions),
            "exception_candidate_count": len(exception_candidates),
            "slice_height": cfg.slice_height,
            "slice_width": cfg.slice_width,
            "overlap_height_ratio": cfg.overlap_height_ratio,
            "overlap_width_ratio": cfg.overlap_width_ratio,
            "rotated_ocr_enabled": cfg.enable_rotated_ocr,
            "line_merge_gap_px": cfg.line_merge_gap_px,
            "line_merge_y_tolerance_px": cfg.line_merge_y_tolerance_px,
            "postprocess_type": cfg.postprocess_type,
            "postprocess_match_metric": cfg.postprocess_match_metric,
            "postprocess_match_threshold": cfg.postprocess_match_threshold,
        },
        "exception_candidates": exception_candidates,
        "overlay_image": _draw_overlay(base_image, merged_regions),
    }
