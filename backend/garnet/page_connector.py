from __future__ import annotations

import re
from typing import Any, Literal

PageRefClass = Literal["reference", "label", "line_number"]

_REF_RE = re.compile(
    r"(?:(?:SHEET|PAGE|PID|FIG\.?|DWG\.?|DRAWING)\s*[-:.]?\s*([A-Z]?-?\d+(?:-\d+)?[A-Z]?)|([A-Z0-9]{2,4}-\d{4}))",
    re.IGNORECASE,
)


def _bbox_center(bbox: Any) -> tuple[float, float]:
    if isinstance(bbox, dict):
        if {"x_min", "y_min", "x_max", "y_max"}.issubset(bbox):
            return (
                (float(bbox["x_min"]) + float(bbox["x_max"])) / 2,
                (float(bbox["y_min"]) + float(bbox["y_max"])) / 2,
            )
        if {"x", "y", "w", "h"}.issubset(bbox):
            return (float(bbox["x"]) + float(bbox["w"]) / 2, float(bbox["y"]) + float(bbox["h"]) / 2)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return ((float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2)
    raise ValueError(f"Unsupported bbox format: {bbox!r}")


def classify_off_page_reference(text: str) -> dict[str, Any] | None:
    text = text.strip()
    m = _REF_RE.search(text)
    if not m:
        return None
    # Group 1: SHEET/PID/FIG/DWG prefix pattern. Group 2: bare XXXX-NNNN pattern (e.g. "26-0003")
    if m.group(2):
        ref_type = "sheet"
        val = m.group(2).upper().strip()
    else:
        val = m.group(1).upper().strip()
        lower = text.lower()
        if lower.startswith("sheet"):
            ref_type = "sheet"
        elif lower.startswith("pid"):
            ref_type = "pid"
        elif lower.startswith(("fig", "figure")):
            ref_type = "figure"
        elif lower.startswith(("dwg", "drawing")):
            ref_type = "drawing"
        else:
            ref_type = "sheet"
    return {"reference_type": ref_type, "reference_value": val, "matched_text": m.group(0)}


def _is_off_page_ref_text(text: str) -> bool:
    """Return True if text matches a typical off-page reference pattern.

    Used as a fallback when OCR has classified the text as 'line_number'
    but the format (e.g. "26-0003") indicates it may be a cross-sheet reference
    in the context of a page connector.
    """
    return classify_off_page_reference(text) is not None


def find_nearby_text(
    page_connector_bbox: dict,
    text_regions: list[dict],
    max_distance_px: float = 80.0,
) -> list[dict]:
    cx, cy = _bbox_center(page_connector_bbox)
    attached = []
    for r in text_regions:
        bx, by_v = _bbox_center(r["bbox"])
        dist = ((cx - bx) ** 2 + (cy - by_v) ** 2) ** 0.5
        if dist <= max_distance_px:
            text_val = r["text"]
            # OCR sometimes classifies off-page references as "line_number" (e.g. "26-0003")
            # Check with _is_off_page_ref_text as fallback regardless of OCR class
            is_reference = _is_off_page_ref_text(text_val)
            ref = classify_off_page_reference(text_val) if is_reference or r.get("class") != "line_number" else None
            attached.append(
                {
                    "region_id": r.get("id"),
                    "text": text_val,
                    "normalized_text": r.get("normalized_text", text_val),
                    "semantic_class": "reference" if ref else ("line_number" if r.get("class") == "line_number" else "label"),
                    "distance_px": round(dist, 3),
                    "page_reference": ref,
                }
            )
    attached.sort(key=lambda x: x["distance_px"])
    return attached
