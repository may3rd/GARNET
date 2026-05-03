from __future__ import annotations

import re
from typing import Any, Literal


TextRegionClass = Literal["equipment_tag", "nozzle_tag", "pipe_spec", "note", "unknown"]

EQUIPMENT_TAG_RE = re.compile(r"^(?:[A-Z]{1,4})-\d{2,5}[A-Z]?$")
NOZZLE_TAG_RE = re.compile(r"^N-\d{1,4}[A-Z]?$")
PIPE_SPEC_RE = re.compile(r"^\d+(?:\.\d+)?(?:\"|IN|INCH)-[A-Z0-9][A-Z0-9./_-]*(?:-[A-Z0-9][A-Z0-9./_-]*)*$")


def normalize_text_token(text: str) -> str:
    normalized = str(text).upper().strip()
    normalized = normalized.replace("_", "-")
    normalized = normalized.replace("”", '"').replace("“", '"')
    normalized = normalized.replace("''", '"')
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized


def _region_class(region: dict[str, Any]) -> str:
    return str(region.get("class") or region.get("semantic_class") or region.get("text_class") or "").strip().lower()


def classify_text_region(region: dict[str, Any]) -> TextRegionClass:
    direct_class = _region_class(region)
    if direct_class == "note":
        return "note"

    text = normalize_text_token(str(region.get("text", "")))
    if not text:
        return "unknown"
    if NOZZLE_TAG_RE.fullmatch(text):
        return "nozzle_tag"
    if PIPE_SPEC_RE.fullmatch(text):
        return "pipe_spec"
    if EQUIPMENT_TAG_RE.fullmatch(text):
        return "equipment_tag"
    return "unknown"

