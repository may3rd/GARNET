"""Response parser for Agent 1 — Global Equipment Detector.

Extracts <|ref|> and <|box|> primitives from the VLM's chain-of-thought response,
converts relative coordinates (in the downsampled global view) to normalized
[0, 999] global space, and builds the EquipmentRegistry.
"""

from __future__ import annotations

import json
import re
import logging
from typing import Optional

from .schemas import EquipmentRegistry, EquipmentEntry, EquipmentClass, Confidence

logger = logging.getLogger(__name__)

# Regex patterns for visual primitives embedded in reasoning text.
# Matches: <|ref|>label<|/ref|><|box|>[[x1,y1,x2,y2]]<|/box|>
_BOX_PATTERN = re.compile(
    r"<\|ref\|>(.+?)<\|/ref\|>\s*<\|box\|>\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]<\|/box\|>",
    re.IGNORECASE,
)

# Extracts tag=... and confidence=... from the text following a box match.
_TAG_PATTERN = re.compile(r"tag\s*[=:]\s*(\S+)", re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r"confidence\s*[=:]\s*(high|medium|low)", re.IGNORECASE)

# Look for a JSON block in the response.
_JSON_BLOCK = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_response(
    raw_text: str,
    view_width: int,
    view_height: int,
) -> tuple[EquipmentRegistry, str]:
    """Parse a VLM response containing interleaved <box> primitives + final JSON.

    Args:
        raw_text: The full VLM response text (thinking chain + structured output).
        view_width: Width of the downsampled view the VLM was looking at (pixels).
        view_height: Height of the downsampled view.

    Returns:
        (EquipmentRegistry, thinking_chain) — the validated registry and extracted reasoning.
    """
    thinking = _extract_thinking(raw_text)
    entries_from_boxes = _extract_boxes(raw_text, view_width, view_height)
    entries_from_json_raw = _extract_json(raw_text)
    entries_from_json = _normalize_json_bboxes(entries_from_json_raw, view_width, view_height)

    # Merge: JSON entries take priority if they have better tag info, but box-derived
    # entries keep their grounded positions. Strategy: prefer JSON tags on matching bboxes.
    merged = _merge_entries(entries_from_boxes, entries_from_json)

    drawing_notes = ""
    if entries_from_json:
        notes = entries_from_json.get("drawing_notes", "")
        if isinstance(notes, str):
            drawing_notes = notes

    registry = EquipmentRegistry(equipment=merged, drawing_notes=drawing_notes)
    return registry, thinking


# ---------------------------------------------------------------------------
# Internal: extractors
# ---------------------------------------------------------------------------


def _extract_thinking(raw: str) -> str:
    """Isolate the reasoning chain — everything before the final JSON block."""
    json_match = _JSON_BLOCK.search(raw)
    if json_match:
        return raw[: json_match.start()].strip()
    # If no JSON block, check for bare JSON at the end.
    stripped = raw.strip()
    if stripped.endswith("}"):
        # Try to find where JSON starts.
        for i in range(len(raw) - 1, -1, -1):
            if raw[i] == "{":
                return raw[:i].strip()
    return raw.strip()


def _extract_boxes(
    text: str,
    view_width: int,
    view_height: int,
) -> list[EquipmentEntry]:
    """Extract equipment entries from <box> primitives in the reasoning chain.

    Coordinates are in the downsampled view space — converted to [0, 999] global.
    """
    entries: list[EquipmentEntry] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    seen_tags: dict[str, EquipmentEntry] = {}  # keep best entry per tag

    for match in _BOX_PATTERN.finditer(text):
        label = match.group(1).strip()
        try:
            rx1, ry1, rx2, ry2 = (
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
                int(match.group(5)),
            )
        except ValueError:
            continue

        # Convert view-relative -> [0, 999] normalised.
        gx1 = int(round(rx1 / view_width * 999))
        gy1 = int(round(ry1 / view_height * 999))
        gx2 = int(round(rx2 / view_width * 999))
        gy2 = int(round(ry2 / view_height * 999))

        bbox_key = (gx1, gy1, gx2, gy2)
        if bbox_key in seen_bboxes:
            continue
        seen_bboxes.add(bbox_key)

        # Look for tag and confidence in the surrounding text (±200 chars after the box).
        context_start = match.end()
        context = text[context_start : context_start + 200]

        tag_match = _TAG_PATTERN.search(context)
        tag = tag_match.group(1).strip().strip("\"'") if tag_match else "unknown"

        conf_match = _CONFIDENCE_PATTERN.search(context)
        confidence_str = conf_match.group(1).lower() if conf_match else "medium"

        eq_class = _map_class(label)

        # Extra description for "other" class.
        description = None
        if eq_class == EquipmentClass.OTHER:
            description = label

        try:
            entry = EquipmentEntry(
                tag=tag,
                equipment_class=eq_class,
                global_bbox=[gx1, gy1, gx2, gy2],
                confidence=Confidence(confidence_str),
                description=description,
            )

            # Deduplicate by tag — keep the most confident entry.
            # "unknown" tags are always distinct (no dedup).
            if tag.lower() != "unknown":
                prev = seen_tags.get(tag)
                if prev is not None:
                    conf_order = {"high": 3, "medium": 2, "low": 1}
                    if conf_order.get(entry.confidence.value, 0) <= conf_order.get(prev.confidence.value, 0):
                        continue
                seen_tags[tag] = entry

            entries.append(entry)
        except ValueError as e:
            logger.warning("Skipping invalid equipment entry: %s", e)

    return entries


def _extract_json(text: str) -> dict:
    """Extract structured JSON block from the response.

    The JSON block may contain bbox coordinates that are also view-relative.
    We accept them as-is (they'll be merged with box-derived entries later).
    """
    json_match = _JSON_BLOCK.search(text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fallback: try to find bare JSON at the end.
    stripped = text.strip()
    for i in range(len(stripped) - 1, -1, -1):
        if stripped[i] == "{":
            candidate = stripped[i:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                break
    return {}


def _normalize_json_bboxes(
    data: dict,
    view_width: int,
    view_height: int,
) -> dict:
    """Convert JSON bboxes from view-relative to [0, 999] global space.

    Returns a new dict with bboxes normalised so they are in the same
    coordinate system as the box-derived entries for correct IoU matching.
    """
    if not data or "equipment" not in data:
        return data
    normalised = {k: v for k, v in data.items() if k != "equipment"}
    normalised_equip = []
    for je in data["equipment"]:
        je_copy = dict(je)
        bbox = je_copy.get("bbox", [])
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            je_copy["bbox"] = [
                int(round(bbox[0] / view_width * 999)),
                int(round(bbox[1] / view_height * 999)),
                int(round(bbox[2] / view_width * 999)),
                int(round(bbox[3] / view_height * 999)),
            ]
        normalised_equip.append(je_copy)
    normalised["equipment"] = normalised_equip
    return normalised


def _merge_entries(
    from_boxes: list[EquipmentEntry],
    from_json: dict,
) -> list[EquipmentEntry]:
    """Merge box-derived entries with JSON-derived entries.

    JSON entries get priority for tag and class (the model's final structured
    answer). Box-derived entries provide the spatial grounding.
    """
    if not from_json.get("equipment"):
        return from_boxes

    json_entries: list[dict] = from_json["equipment"]

    # If we have box-derived entries, try to match them with JSON entries by proximity.
    if from_boxes:
        return _match_and_merge(from_boxes, json_entries)

    # No box entries — build directly from JSON.
    entries: list[EquipmentEntry] = []
    for je in json_entries:
        try:
            entry = EquipmentEntry(
                tag=str(je.get("tag", "unknown")),
                equipment_class=_map_class(je.get("equipment_class", "other")),
                global_bbox=_normalize_bbox_list(je.get("bbox", [0, 0, 0, 0])),
                confidence=Confidence(_map_confidence(je.get("confidence", "medium"))),
            )
            entries.append(entry)
        except ValueError:
            continue
    return entries


def _match_and_merge(
    box_entries: list[EquipmentEntry],
    json_entries: list[dict],
) -> list[EquipmentEntry]:
    """Match box-derived entries with JSON entries by bbox overlap (IoU)."""
    import math

    merged: list[EquipmentEntry] = []
    used_json: set[int] = set()

    for be in box_entries:
        bx1, by1, bx2, by2 = be.global_bbox
        best_iou = 0.0
        best_idx = -1

        for j, je in enumerate(json_entries):
            if j in used_json:
                continue
            jbbox = _normalize_bbox_list(je.get("bbox", [-1, -1, -1, -1]))
            jx1, jy1, jx2, jy2 = jbbox
            iou = _iou(bx1, by1, bx2, by2, jx1, jy1, jx2, jy2)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx >= 0 and best_iou > 0.1:
            je = json_entries[best_idx]
            used_json.add(best_idx)
            # Prefer JSON tag if it's not "unknown"
            json_tag = str(je.get("tag", "unknown"))
            tag = json_tag if json_tag.lower() != "unknown" else be.tag
            eq_class = _map_class(je.get("equipment_class", be.equipment_class.value))
            confidence_str = _map_confidence(je.get("confidence", be.confidence.value))
            try:
                merged.append(
                    EquipmentEntry(
                        tag=tag,
                        equipment_class=eq_class,
                        global_bbox=be.global_bbox,  # Keep box-derived coords
                        confidence=Confidence(confidence_str),
                    )
                )
            except ValueError:
                merged.append(be)
        else:
            merged.append(be)

    # Add unmatched JSON entries.
    for j, je in enumerate(json_entries):
        if j not in used_json:
            try:
                merged.append(
                    EquipmentEntry(
                        tag=str(je.get("tag", "unknown")),
                        equipment_class=_map_class(je.get("equipment_class", "other")),
                        global_bbox=_normalize_bbox_list(je.get("bbox", [0, 0, 0, 0])),
                        confidence=Confidence(_map_confidence(je.get("confidence", "medium"))),
                    )
                )
            except ValueError:
                continue

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_bbox_list(bbox) -> list[int]:
    """Ensure bbox is a 4-int list in [0, 999] range."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [0, 0, 0, 0]
    return [max(0, min(999, int(round(c)))) for c in bbox]


def _map_class(name: str) -> EquipmentClass:
    """Map a free-text class name to the 7-class taxonomy enum."""
    name = name.lower().strip().replace(" ", "_").replace("-", "_")
    mapping = {
        "distillation_column": EquipmentClass.DISTILLATION_COLUMN,
        "column": EquipmentClass.DISTILLATION_COLUMN,
        "tower": EquipmentClass.DISTILLATION_COLUMN,
        "distillation_tower": EquipmentClass.DISTILLATION_COLUMN,
        "pressure_vessel": EquipmentClass.PRESSURE_VESSEL,
        "vessel": EquipmentClass.PRESSURE_VESSEL,
        "drum": EquipmentClass.PRESSURE_VESSEL,
        "accumulator": EquipmentClass.PRESSURE_VESSEL,
        "separator": EquipmentClass.PRESSURE_VESSEL,
        "knockout_drum": EquipmentClass.PRESSURE_VESSEL,
        "heat_exchanger": EquipmentClass.HEAT_EXCHANGER,
        "exchanger": EquipmentClass.HEAT_EXCHANGER,
        "shell_and_tube": EquipmentClass.HEAT_EXCHANGER,
        "shell_tube_exchanger": EquipmentClass.HEAT_EXCHANGER,
        "reboiler": EquipmentClass.HEAT_EXCHANGER,
        "condenser": EquipmentClass.HEAT_EXCHANGER,
        "cooler": EquipmentClass.HEAT_EXCHANGER,
        "heater": EquipmentClass.HEAT_EXCHANGER,
        "storage_tank": EquipmentClass.STORAGE_TANK,
        "tank": EquipmentClass.STORAGE_TANK,
        "pump": EquipmentClass.PUMP,
        "compressor": EquipmentClass.COMPRESSOR,
        "reactor": EquipmentClass.REACTOR,
    }
    try:
        return mapping.get(name, EquipmentClass.OTHER)
    except Exception:
        return EquipmentClass.OTHER


def _map_confidence(raw: str) -> str:
    """Normalize confidence string."""
    raw = raw.lower().strip()
    if raw in ("high", "medium", "low"):
        return raw
    if raw in ("very high", "certain"):
        return "high"
    if raw in ("low", "uncertain", "poor"):
        return "low"
    return "medium"


def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    """Intersection over Union for two axis-aligned bboxes."""
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
