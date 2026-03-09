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
    x_offset: int,
    y_offset: int,
) -> dict[str, Any] | None:
    if len(annotation) != 3:
        return None
    text, confidence, bbox_xywh = annotation
    if not isinstance(bbox_xywh, (list, tuple)) or len(bbox_xywh) != 4:
        return None
    norm_x, norm_y, norm_w, norm_h = [float(v) for v in bbox_xywh]

    # Vision-style normalized coordinates use bottom-left origin.
    x_min = int(round((norm_x * tile_width) + x_offset))
    box_width = int(round(norm_w * tile_width))
    box_height = int(round(norm_h * tile_height))
    y_top_from_bottom = norm_y + norm_h
    y_min = int(round(((1.0 - y_top_from_bottom) * tile_height) + y_offset))
    return {
        "text": str(text),
        "normalized_text": _safe_normalized_text(str(text)),
        "class": _classify_text(str(text)),
        "confidence": round(float(confidence), 4),
        "bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_min + box_width,
            "y_max": y_min + box_height,
        },
        "rotation": 0,
        "reading_direction": "horizontal",
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
        tile_image = Image.fromarray(tile_rgb)
        annotations = ocrmac.OCR(
            tile_image,
            recognition_level=cfg.recognition_level,
            framework=cfg.framework,
            language_preference=list(cfg.language_preference),
        ).recognize()
        tile_h, tile_w = tile_rgb.shape[:2]
        for annotation in annotations:
            det = _annotation_to_detection(
                annotation,
                tile_width=tile_w,
                tile_height=tile_h,
                x_offset=x0,
                y_offset=y0,
            )
            if det is None:
                continue
            if det["confidence"] < cfg.confidence_threshold:
                continue
            raw_detections.append(det)

    deduped = _nms_ios(raw_detections, cfg.postprocess_match_threshold)
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
            "postprocess_match_metric": cfg.postprocess_match_metric,
            "postprocess_match_threshold": cfg.postprocess_match_threshold,
        },
        "exception_candidates": exception_candidates,
        "overlay_image": _draw_overlay(image_bgr, deduped),
    }
