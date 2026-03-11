from __future__ import annotations

import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("pid")


@dataclass(frozen=True)
class OcrMacSahiConfig:
    recognition_level: str = "accurate"
    framework: str = "vision"
    language_preference: tuple[str, ...] = ("en-US",)
    confidence_threshold: float = 0.0
    slice_height: int = 1600
    slice_width: int = 1600
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    enable_rotated_ocr: bool = True
    tighten_bboxes: bool = True
    tighten_padding_px: int = 1
    tighten_dark_threshold: int = 200
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.1


_ocrmac_module: Any = None


def _get_ocrmac_module() -> Any:
    global _ocrmac_module
    if platform.system() != "Darwin":
        raise RuntimeError("ocrmac route requires macOS (Darwin)")
    if _ocrmac_module is None:
        try:
            from ocrmac import ocrmac as module  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "ocrmac is not installed. Install it on macOS with `pip install ocrmac`."
            ) from exc
        _ocrmac_module = module
    return _ocrmac_module


def _bbox_ios(a: dict[str, int], b: dict[str, int]) -> float:
    ix1 = max(a["x_min"], b["x_min"])
    iy1 = max(a["y_min"], b["y_min"])
    ix2 = min(a["x_max"], b["x_max"])
    iy2 = min(a["y_max"], b["y_max"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a["x_max"] - a["x_min"]) * max(0, a["y_max"] - a["y_min"])
    area_b = max(0, b["x_max"] - b["x_min"]) * max(0, b["y_max"] - b["y_min"])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _nms_ios(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for det in detections:
        suppressed = any(_bbox_ios(det["bbox"], k["bbox"]) > threshold for k in kept)
        if not suppressed:
            kept.append(det)
    return kept


def _compute_tiles(
    h: int,
    w: int,
    slice_h: int,
    slice_w: int,
    overlap_h: float,
    overlap_w: float,
) -> list[tuple[int, int, int, int]]:
    stride_h = max(1, int(slice_h * (1 - overlap_h)))
    stride_w = max(1, int(slice_w * (1 - overlap_w)))
    tiles: list[tuple[int, int, int, int]] = []
    y0 = 0
    while y0 < h:
        x0 = 0
        y1 = min(y0 + slice_h, h)
        while x0 < w:
            x1 = min(x0 + slice_w, w)
            tiles.append((y0, x0, y1, x1))
            if x1 == w:
                break
            x0 += stride_w
        if y1 == h:
            break
        y0 += stride_h
    return tiles


def _safe_normalized_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized if normalized == text.strip() else ""


def _rotate_image(tile_rgb: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "none":
        return tile_rgb
    if orientation == "cw":
        return cv2.rotate(tile_rgb, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "ccw":
        return cv2.rotate(tile_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported orientation: {orientation}")


def _rotate_points_back(
    points: list[tuple[float, float]],
    *,
    orientation: str,
    original_width: int,
    original_height: int,
) -> list[tuple[float, float]]:
    restored: list[tuple[float, float]] = []
    for x_rot, y_rot in points:
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
        restored.append((float(x_orig), float(y_orig)))
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


def _classify_text(text: str) -> str:
    token = text.strip()
    upper = token.upper()
    if not token:
        return "unknown"
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
    if region["class"] == "unknown":
        reasons.append("unknown_class")
    if region["normalized_text"] == "":
        reasons.append("unsafe_normalization")
    return reasons


def _annotation_to_detection(
    annotation: tuple[Any, Any, Any],
    *,
    tile_width: int,
    tile_height: int,
    original_width: int,
    original_height: int,
    x_offset: int,
    y_offset: int,
    orientation: str,
) -> dict[str, Any] | None:
    if len(annotation) != 3:
        return None
    text, confidence, bbox_xywh = annotation
    if not isinstance(bbox_xywh, (list, tuple)) or len(bbox_xywh) != 4:
        return None
    norm_x, norm_y, norm_w, norm_h = [float(v) for v in bbox_xywh]

    # Vision-style normalized coordinates use bottom-left origin.
    x_min = norm_x * tile_width
    box_width = norm_w * tile_width
    box_height = norm_h * tile_height
    y_top_from_bottom = norm_y + norm_h
    y_min = (1.0 - y_top_from_bottom) * tile_height
    x_max = x_min + box_width
    y_max = y_min + box_height

    rotated_points = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    restored_points = _rotate_points_back(
        rotated_points,
        orientation=orientation,
        original_width=original_width,
        original_height=original_height,
    )
    restored_x = [point[0] for point in restored_points]
    restored_y = [point[1] for point in restored_points]
    restored_bbox = {
        "x_min": int(round(min(restored_x))) + x_offset,
        "y_min": int(round(min(restored_y))) + y_offset,
        "x_max": int(round(max(restored_x))) + x_offset,
        "y_max": int(round(max(restored_y))) + y_offset,
    }
    rotation = 0 if orientation == "none" else 90
    return {
        "text": str(text),
        "normalized_text": _safe_normalized_text(str(text)),
        "class": _classify_text(str(text)),
        "confidence": round(float(confidence), 4),
        "bbox": restored_bbox,
        "rotation": rotation,
        "reading_direction": _region_direction(restored_bbox, rotation),
        "legibility": "clear" if float(confidence) >= 0.8 else "degraded",
    }


def _draw_overlay(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    overlay = image_bgr.copy()
    for det in detections:
        b = det["bbox"]
        cv2.rectangle(
            overlay,
            (b["x_min"], b["y_min"]),
            (b["x_max"], b["y_max"]),
            (255, 0, 0),
            2,
        )
    return overlay


def _tighten_bbox_to_text_ink(
    image_gray: np.ndarray,
    bbox: dict[str, int],
    *,
    dark_threshold: int,
    padding_px: int,
) -> dict[str, int]:
    height, width = image_gray.shape[:2]
    x_min = max(0, int(bbox["x_min"]))
    y_min = max(0, int(bbox["y_min"]))
    x_max = min(width - 1, int(bbox["x_max"]))
    y_max = min(height - 1, int(bbox["y_max"]))
    if x_max <= x_min or y_max <= y_min:
        return bbox

    crop = image_gray[y_min : y_max + 1, x_min : x_max + 1]
    if crop.size == 0:
        return bbox

    ink_mask = (crop <= dark_threshold).astype(np.uint8)
    if not np.any(ink_mask):
        _, ink_mask = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if not np.any(ink_mask):
        return bbox

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    kept_components: list[tuple[int, int, int, int]] = []
    fallback_components: list[tuple[int, int, int, int]] = []
    crop_h, crop_w = ink_mask.shape[:2]

    for label_idx in range(1, num_labels):
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        right = left + comp_w - 1
        bottom = top + comp_h - 1
        touches_boundary = (
            left <= 0 or top <= 0 or right >= crop_w - 1 or bottom >= crop_h - 1
        )
        component = (left, top, right, bottom)
        fallback_components.append(component)
        if not touches_boundary:
            kept_components.append(component)

    components = kept_components or fallback_components
    if not components:
        return bbox

    tight_left = min(item[0] for item in components)
    tight_top = min(item[1] for item in components)
    tight_right = max(item[2] for item in components)
    tight_bottom = max(item[3] for item in components)

    return {
        "x_min": max(0, x_min + tight_left - padding_px),
        "y_min": max(0, y_min + tight_top - padding_px),
        "x_max": min(width - 1, x_min + tight_right + padding_px),
        "y_max": min(height - 1, y_min + tight_bottom + padding_px),
    }


def _tighten_region_bboxes(
    image_gray: np.ndarray,
    regions: list[dict[str, Any]],
    cfg: OcrMacSahiConfig,
) -> list[dict[str, Any]]:
    if not cfg.tighten_bboxes:
        return regions
    tightened: list[dict[str, Any]] = []
    for region in regions:
        next_region = dict(region)
        next_region["bbox"] = _tighten_bbox_to_text_ink(
            image_gray,
            region["bbox"],
            dark_threshold=cfg.tighten_dark_threshold,
            padding_px=cfg.tighten_padding_px,
        )
        tightened.append(next_region)
    return tightened


def run_ocrmac_sahi(
    image_path: str | Path,
    *,
    image_id: str,
    cfg: OcrMacSahiConfig = OcrMacSahiConfig(),
) -> dict[str, Any]:
    ocrmac = _get_ocrmac_module()
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    if h <= cfg.slice_height and w <= cfg.slice_width:
        tiles = [(0, 0, h, w)]
    else:
        tiles = _compute_tiles(
            h, w,
            cfg.slice_height, cfg.slice_width,
            cfg.overlap_height_ratio, cfg.overlap_width_ratio,
        )

    raw_detections: list[dict[str, Any]] = []
    for y0, x0, y1, x1 in tiles:
        tile_rgb = image_rgb[y0:y1, x0:x1]
        tile_h, tile_w = tile_rgb.shape[:2]
        orientations = ["none"]
        if cfg.enable_rotated_ocr:
            orientations.extend(["cw", "ccw"])
        for orientation in orientations:
            oriented_tile_rgb = _rotate_image(tile_rgb, orientation)
            oriented_h, oriented_w = oriented_tile_rgb.shape[:2]
            tile_image = Image.fromarray(oriented_tile_rgb)
            annotations = ocrmac.OCR(
                tile_image,
                recognition_level=cfg.recognition_level,
                framework=cfg.framework,
                language_preference=list(cfg.language_preference),
            ).recognize()
            for annotation in annotations:
                det = _annotation_to_detection(
                    annotation,
                    tile_width=oriented_w,
                    tile_height=oriented_h,
                    original_width=tile_w,
                    original_height=tile_h,
                    x_offset=x0,
                    y_offset=y0,
                    orientation=orientation,
                )
                if det is None:
                    continue
                if det["confidence"] < cfg.confidence_threshold:
                    continue
                raw_detections.append(det)

    deduped = _nms_ios(raw_detections, cfg.postprocess_match_threshold)
    deduped = _tighten_region_bboxes(image_gray, deduped, cfg)
    for idx, region in enumerate(sorted(deduped, key=lambda item: (item["bbox"]["y_min"], item["bbox"]["x_min"], item["text"])), start=1):
        region["id"] = f"ocr_{idx:06d}"

    exception_candidates = []
    for region in deduped:
        reasons = _exception_reasons(region)
        if reasons:
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
            "text_regions": deduped,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "route": "ocrmac",
            "tile_count": len(tiles),
            "raw_detection_count": len(raw_detections),
            "merged_region_count": len(deduped),
            "exception_candidate_count": len(exception_candidates),
            "slice_height": cfg.slice_height,
            "slice_width": cfg.slice_width,
            "overlap_height_ratio": cfg.overlap_height_ratio,
            "overlap_width_ratio": cfg.overlap_width_ratio,
            "framework": cfg.framework,
            "recognition_level": cfg.recognition_level,
            "language_preference": list(cfg.language_preference),
            "rotated_ocr_enabled": cfg.enable_rotated_ocr,
            "tighten_bboxes": cfg.tighten_bboxes,
            "tighten_padding_px": cfg.tighten_padding_px,
            "tighten_dark_threshold": cfg.tighten_dark_threshold,
            "postprocess_match_metric": cfg.postprocess_match_metric,
            "postprocess_match_threshold": cfg.postprocess_match_threshold,
        },
        "exception_candidates": exception_candidates,
        "overlay_image": _draw_overlay(image_bgr, deduped),
    }
