#!/usr/bin/env python
"""Stage 2 — OCR text discovery, backend-free.

Runs macOS Vision OCR (via the vendored `ocrmac_sahi`) over the sheet, saves the
Stage 2 artifacts, and hands back regions shaped for `merge_ocr_regions`.

Why this stage exists in the backend-free copy
----------------------------------------------
`build_pipe_mask` erases text so only pipework survives, and without OCR its only
source of text boxes is the fixture object list. That coverage is partial: the
detector localises the classes it was trained on, so any text it did not detect
stays in the mask as ink and the tracer can follow glyph strokes as if they were
pipe. Stage 2 supplies the missing boxes.

Route: `ocrmac` only. The backend also offers easyocr / paddleocr / gemini, but
paddleocr and pytesseract are not installed here, and ocrmac is the route the
backend defaults to on macOS. Keeping one route keeps this copy dependency-light;
add another only if a sheet actually needs it.

Usage:
    from garnet.stage5b.stage2_ocr import run_stage2_ocr, Stage2OcrConfig
    result = run_stage2_ocr(image_path, out_dir)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2

from .ocrmac_sahi import OcrMacSahiConfig, run_ocrmac_sahi

log = logging.getLogger(__name__)

ARTIFACTS = ("stage2_ocr_regions", "stage2_ocr_summary", "stage2_ocr_overlay")

# OCR emits class names in the backend's underscore form; the pipeline keys on the
# spaced form. Without this map every OCR region lands as an unrecognised class and
# `build_pipe_mask` never blanks it -- the stage would run and do nothing.
_OCR_CLASS_TO_PIPELINE = {
    "line_number": "line number",
    "instrument_tag": "instrument tag",
    "instrument_dcs": "instrument dcs",
    "instrument_logic": "instrument logic",
    "note": "note",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class Stage2OcrConfig:
    """Tunables for the OCR pass.

    Defaults follow the backend's ocrmac route. `slice_height/width` are the SAHI
    tile size: a 4962x3508 sheet at 2400 gives 6 tiles with 0.2 overlap, which
    took ~4.5s on Test sheet 0003. Larger tiles are faster but crop less context
    around a label; smaller tiles lose text that straddles a tile edge.
    """

    framework: str = "vision"
    # The backend notes macOS Vision's "accurate" level returns zero detections in
    # this environment even for large clear text; "fast" is what populates.
    recognition_level: str = "fast"
    slice_height: int = 2400
    slice_width: int = 2400
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    enable_rotated_ocr: bool = True
    tighten_bboxes: bool = True
    # Regions whose transcribed text is empty carry no information for mask
    # suppression beyond their bbox -- keep them, a box alone still blanks ink.
    drop_empty_text: bool = False


def normalize_ocr_region(region: dict[str, Any]) -> Optional[dict[str, Any]]:
    """OCR region -> the object shape the stage4/stage5b code consumes.

    Two field mismatches have to be reconciled or the region is silently ignored
    downstream: `class` (underscore, OCR) vs `class_name` (spaced, pipeline), and
    `bbox` which is already the {x_min,y_min,x_max,y_max} dict both sides use.
    """
    bbox = region.get("bbox") or {}
    if not {"x_min", "y_min", "x_max", "y_max"}.issubset(bbox):
        return None
    text = str(region.get("text") or "").strip()
    raw_class = str(region.get("class") or "unknown").strip().lower()
    class_name = _OCR_CLASS_TO_PIPELINE.get(raw_class, "unknown")
    return {
        "id": str(region.get("id") or ""),
        "class_name": class_name,
        "confidence": float(region.get("confidence") or 0.0),
        "bbox": {k: int(round(float(bbox[k]))) for k in ("x_min", "y_min", "x_max", "y_max")},
        "text": text or None,
        "text_source": "stage2_ocr",
        "ocr_class_raw": raw_class,
        "rotation": int(region.get("rotation") or 0),
        "reading_direction": region.get("reading_direction"),
    }


def run_stage2_ocr(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: Stage2OcrConfig | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run OCR, write the Stage 2 artifacts, return the normalized regions.

    Returns {"regions", "summary", "overlay", "artifacts"} where `regions` are in
    pipeline object shape (see `normalize_ocr_region`).
    """
    cfg = cfg or Stage2OcrConfig()
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = run_ocrmac_sahi(
        str(image_path),
        image_id=image_path.stem,
        cfg=OcrMacSahiConfig(
            framework=cfg.framework,
            recognition_level=cfg.recognition_level,
            slice_height=cfg.slice_height,
            slice_width=cfg.slice_width,
            overlap_height_ratio=cfg.overlap_height_ratio,
            overlap_width_ratio=cfg.overlap_width_ratio,
            enable_rotated_ocr=cfg.enable_rotated_ocr,
            tighten_bboxes=cfg.tighten_bboxes,
        ),
    )

    regions: list[dict[str, Any]] = []
    dropped = 0
    for region in raw["regions_payload"].get("text_regions", []):
        if cfg.drop_empty_text and not str(region.get("text") or "").strip():
            dropped += 1
            continue
        normalized = normalize_ocr_region(region)
        if normalized is None:
            dropped += 1
            continue
        regions.append(normalized)

    summary = dict(raw["summary"])
    summary["route"] = "ocrmac"
    summary["normalized_region_count"] = len(regions)
    summary["regions_dropped"] = dropped
    by_class: dict[str, int] = {}
    for region in regions:
        by_class[region["class_name"]] = by_class.get(region["class_name"], 0) + 1
    summary["region_class_counts"] = by_class
    summary["text_carrying_regions"] = sum(1 for r in regions if r.get("text"))

    if save:
        (out_dir / "stage2_ocr_regions.json").write_text(
            json.dumps({"image_id": image_path.stem,
                        "text_regions": regions}, indent=2),
            encoding="utf-8",
        )
        (out_dir / "stage2_ocr_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        cv2.imwrite(str(out_dir / "stage2_ocr_overlay.png"), raw["overlay_image"])

    log.info("stage2 ocr: %d regions (%d carrying text), %d dropped",
             len(regions), summary["text_carrying_regions"], dropped)
    return {
        "regions": regions,
        "summary": summary,
        "overlay": raw["overlay_image"],
        "artifacts": list(ARTIFACTS),
    }
