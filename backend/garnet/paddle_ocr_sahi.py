"""
PaddleOCR-compatible text-detection-only route via RapidOCR + ONNX runtime.

Uses PP-OCRv4 detection (no recognition) to produce axis-aligned bounding
boxes for text regions.  This avoids a paddlepaddle dependency by running the
detection model through rapidocr-onnxruntime which bundles ONNX weights.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("pid")

# ---------------------------------------------------------------------------
# Lazy singleton — expensive to initialise; shared across calls
# ---------------------------------------------------------------------------
_ocr_instance: Any = None


def _get_ocr() -> Any:
    global _ocr_instance
    if _ocr_instance is None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        _ocr_instance = RapidOCR()
        logger.info("RapidOCR (PP-OCRv4 det) initialised")
    return _ocr_instance


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaddleOcrSahiConfig:
    det_box_thresh: float = 0.5
    det_unclip_ratio: float = 1.6
    slice_height: int = 1600
    slice_width: int = 1600
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.1


# ---------------------------------------------------------------------------
# Bbox helpers (IOS = Intersection over Smaller area)
# ---------------------------------------------------------------------------


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


def _nms_ios(
    detections: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Greedy IOS-based NMS: keep higher-confidence box when overlap > threshold."""
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for det in detections:
        suppressed = any(
            _bbox_ios(det["bbox"], k["bbox"]) > threshold for k in kept
        )
        if not suppressed:
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


def _compute_tiles(
    h: int, w: int, slice_h: int, slice_w: int, overlap_h: float, overlap_w: float
) -> list[tuple[int, int, int, int]]:
    """Return list of (y0, x0, y1, x1) tile coordinates covering the full image."""
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


# ---------------------------------------------------------------------------
# Inference on one tile
# ---------------------------------------------------------------------------


def _detect_tile(
    tile_bgr: np.ndarray,
    cfg: PaddleOcrSahiConfig,
    x_offset: int,
    y_offset: int,
    tile_idx: int,
) -> list[dict[str, Any]]:
    ocr = _get_ocr()
    result, _ = ocr(
        tile_bgr,
        use_det=True,
        use_cls=False,
        use_rec=False,
        box_thresh=cfg.det_box_thresh,
        unclip_ratio=cfg.det_unclip_ratio,
    )
    if result is None:
        return []

    detections: list[dict[str, Any]] = []
    for item in result:
        if not item:
            continue
        # det-only mode: item IS the polygon — [[x,y], [x,y], [x,y], [x,y]]
        # full mode:     item is [polygon, text, score]
        if isinstance(item[0], (int, float)):
            # single point — skip malformed entry
            continue
        if isinstance(item[0][0], (int, float)):
            # item is the polygon directly
            polygon = item
            score: float = 1.0
        else:
            # item is [polygon, text, score]
            polygon = item[0]
            score = float(item[2]) if len(item) >= 3 else 1.0

        xs = [pt[0] + x_offset for pt in polygon]
        ys = [pt[1] + y_offset for pt in polygon]
        bbox = {
            "x_min": max(0, int(round(min(xs)))),
            "y_min": max(0, int(round(min(ys)))),
            "x_max": int(round(max(xs))),
            "y_max": int(round(max(ys))),
        }
        if bbox["x_max"] <= bbox["x_min"] or bbox["y_max"] <= bbox["y_min"]:
            continue
        detections.append({"bbox": bbox, "confidence": score, "_tile": tile_idx})
    return detections


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def _draw_overlay(
    image_gray: np.ndarray, detections: list[dict[str, Any]]
) -> np.ndarray:
    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    for det in detections:
        b = det["bbox"]
        cv2.rectangle(
            overlay,
            (b["x_min"], b["y_min"]),
            (b["x_max"], b["y_max"]),
            (220, 50, 50),  # red
            2,
        )
    return overlay


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_paddle_ocr_sahi(
    image_path: str | Path,
    *,
    image_id: str,
    cfg: PaddleOcrSahiConfig = PaddleOcrSahiConfig(),
) -> dict[str, Any]:
    """
    Run PP-OCRv4 text detection (no recognition) on *image_path*.

    Returns a dict matching the Stage 2 contract used by easyocr_sahi and
    gemini_ocr_sahi:
        regions_payload / summary / exception_candidates / overlay_image
    """
    image_path = Path(image_path)
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    h, w = image_gray.shape
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    # Decide whether to tile
    if h <= cfg.slice_height and w <= cfg.slice_width:
        tiles = [(0, 0, h, w)]
    else:
        tiles = _compute_tiles(
            h, w,
            cfg.slice_height, cfg.slice_width,
            cfg.overlap_height_ratio, cfg.overlap_width_ratio,
        )

    logger.info(
        "paddle_ocr_sahi: image %dx%d → %d tile(s)", w, h, len(tiles)
    )

    raw_detections: list[dict[str, Any]] = []
    for idx, (y0, x0, y1, x1) in enumerate(tiles):
        tile = image_bgr[y0:y1, x0:x1]
        dets = _detect_tile(tile, cfg, x_offset=x0, y_offset=y0, tile_idx=idx)
        raw_detections.extend(dets)
        logger.info("  tile %d/%d → %d raw det(s)", idx + 1, len(tiles), len(dets))

    # IOS-NMS deduplication
    if cfg.postprocess_match_metric == "IOS":
        deduped = _nms_ios(raw_detections, cfg.postprocess_match_threshold)
    else:
        deduped = raw_detections

    logger.info(
        "paddle_ocr_sahi: %d raw → %d after NMS", len(raw_detections), len(deduped)
    )

    # Format as standard text_regions
    text_regions: list[dict[str, Any]] = []
    for i, det in enumerate(deduped):
        score = det["confidence"]
        text_regions.append(
            {
                "id": f"ocr_{i + 1:06d}",
                "text": "",
                "normalized_text": "",
                "class": "unknown",
                "confidence": round(score, 4),
                "bbox": det["bbox"],
                "rotation": 0,
                "reading_direction": "unknown",
                "legibility": "clear" if score >= cfg.det_box_thresh else "degraded",
            }
        )

    overlay = _draw_overlay(image_gray, deduped)

    return {
        "regions_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "text_regions": text_regions,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "route": "paddleocr",
            "total_regions": len(text_regions),
            "tile_count": len(tiles),
            "raw_detections": len(raw_detections),
            "config": {
                "det_box_thresh": cfg.det_box_thresh,
                "det_unclip_ratio": cfg.det_unclip_ratio,
                "slice_height": cfg.slice_height,
                "slice_width": cfg.slice_width,
                "overlap_height_ratio": cfg.overlap_height_ratio,
                "overlap_width_ratio": cfg.overlap_width_ratio,
                "postprocess_match_threshold": cfg.postprocess_match_threshold,
            },
        },
        "exception_candidates": [],
        "overlay_image": overlay,
    }
