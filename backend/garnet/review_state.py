from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

VALID_BUCKETS = {
    "stage4_object",
    "stage4_line_number",
    "stage4_instrument",
    "stage6_line_association",
    "stage12_line_attachment",
    "stage12_instrument_attachment",
}
VALID_DECISIONS = {"accepted", "rejected", "deferred"}


def review_state_path(job_dir: str | Path) -> Path:
    return Path(job_dir) / "stage_review_state.json"


def empty_review_state(job_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    job_dir_path = Path(job_dir)
    return {
        "job_id": job_dir_path.name,
        "image_path": None if manifest is None else manifest.get("image_path"),
        "version": 1,
        "updated_at": time.time(),
        "items": [],
        "workspace_objects": {
            "stage4_object": [],
            "stage4_line_number": [],
            "stage4_instrument": [],
            "stage6_line_association": [],
            "stage12_line_attachment": [],
            "stage12_instrument_attachment": [],
        },
    }


def load_review_state(job_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    path = review_state_path(job_dir)
    if not path.exists():
        return empty_review_state(job_dir, manifest)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def validate_review_state_payload(payload: dict[str, Any]) -> None:
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    workspace_objects = payload.get("workspace_objects", {})
    if not isinstance(workspace_objects, dict):
        raise ValueError("workspace_objects must be an object")
    for bucket in workspace_objects.keys():
        if bucket not in VALID_BUCKETS:
            raise ValueError(f"Invalid review bucket: {bucket}")
        if not isinstance(workspace_objects[bucket], list):
            raise ValueError(f"workspace_objects.{bucket} must be a list")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("review items must be objects")
        bucket = item.get("bucket")
        decision = item.get("decision")
        if bucket not in VALID_BUCKETS:
            raise ValueError(f"Invalid review bucket: {bucket}")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid review decision: {decision}")


def save_review_state(job_dir: str | Path, payload: dict[str, Any], manifest: dict[str, Any] | None = None) -> Path:
    validate_review_state_payload(payload)
    current = empty_review_state(job_dir, manifest)
    current["updated_at"] = time.time()
    current["items"] = payload.get("items", [])
    current["workspace_objects"] = {
        **current["workspace_objects"],
        **payload.get("workspace_objects", {}),
    }

    path = review_state_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="stage_review_state_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return path


def _bbox_from_review_object(item: dict[str, Any]) -> dict[str, int]:
    raw_bbox = item.get("bbox")
    if isinstance(raw_bbox, dict):
        try:
            return {
                "x_min": int(round(float(raw_bbox.get("x_min", 0) or 0))),
                "y_min": int(round(float(raw_bbox.get("y_min", 0) or 0))),
                "x_max": int(round(float(raw_bbox.get("x_max", 0) or 0))),
                "y_max": int(round(float(raw_bbox.get("y_max", 0) or 0))),
            }
        except (TypeError, ValueError):
            pass

    left = float(item.get("Left", item.get("x_min", 0)) or 0)
    top = float(item.get("Top", item.get("y_min", 0)) or 0)
    width = float(item.get("Width", 0) or 0)
    height = float(item.get("Height", 0) or 0)
    if width <= 0 and item.get("x_max") is not None:
        width = float(item.get("x_max") or 0) - left
    if height <= 0 and item.get("y_max") is not None:
        height = float(item.get("y_max") or 0) - top
    return {
        "x_min": int(round(left)),
        "y_min": int(round(top)),
        "x_max": int(round(left + max(1.0, width))),
        "y_max": int(round(top + max(1.0, height))),
    }


def _stage4_line_number_id(item: dict[str, Any], index: int) -> str:
    for key in ("id", "SourceItemId", "source_object_id", "Text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"line_number_{index + 1:06d}"


def build_stage4_line_numbers_from_review_state(
    job_dir: str | Path,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Materialize reviewed Stage 4 line-number boxes into the Stage 4 artifact contract."""
    review_payload = load_review_state(job_dir, manifest)
    workspace_objects = review_payload.get("workspace_objects", {})
    if not isinstance(workspace_objects, dict) or "stage4_line_number" not in workspace_objects:
        return None

    reviewed_items = workspace_objects.get("stage4_line_number")
    if not isinstance(reviewed_items, list):
        return None

    image_path = "" if manifest is None else str(manifest.get("image_path") or "")
    image_id = os.path.basename(image_path) if image_path else Path(job_dir).name
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(reviewed_items):
        if not isinstance(item, dict):
            continue
        line_id = _stage4_line_number_id(item, index)
        text = str(item.get("text") or item.get("Text") or item.get("normalized_text") or "").strip()
        try:
            confidence = float(item.get("confidence", item.get("Score", 1)) or 1)
        except (TypeError, ValueError):
            confidence = 1.0
        review_state = str(item.get("review_state") or item.get("ReviewStatus") or "accepted")
        entry = {
            "id": line_id,
            "source_object_id": item.get("source_object_id") or item.get("SourceItemId") or line_id,
            "class_name": "line_number",
            "bbox": _bbox_from_review_object(item),
            "text": text,
            "normalized_text": str(item.get("normalized_text") or text).strip(),
            "ocr_region_id": item.get("ocr_region_id"),
            "ocr_source": item.get("ocr_source", "hitl"),
            "score": item.get("score"),
            "distance_px": item.get("distance_px"),
            "ocr_confirmed": bool(item.get("ocr_confirmed", bool(text))),
            "detection_confidence": confidence,
            "fused_confidence": confidence,
            "semantic_class": "line_number",
            "source": "hitl",
            "review_state": review_state,
        }
        if review_state == "rejected":
            rejected.append(entry)
        else:
            entry["review_state"] = "hitl_reviewed"
            accepted.append(entry)

    return {
        "image_id": image_id,
        "pass_type": "sheet",
        "line_numbers": accepted,
        "rejected": rejected,
        "source": "hitl",
        "source_artifact": "stage_review_state.json",
    }
