from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

WORKSPACE_ARTIFACT_NAME = "review_workspace_state.json"


def review_workspace_path(job_dir: str | Path) -> Path:
    return Path(job_dir) / WORKSPACE_ARTIFACT_NAME


def empty_review_workspace(job_id: str | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "job_id": job_id,
        "image_id": None,
        "updated_at": time.time(),
        "objects": [],
        "equipment": [],
        "manual_ports": [],
        "deleted_entities": [],
        "line_association_overrides": [],
        "trace_overrides": [],
    }


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else ({} if default is None else default)


def _list_from_payload(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, dict)]


def _normalize_workspace_payload(payload: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
    current = empty_review_workspace(job_id)
    current.update({key: value for key, value in payload.items() if key in current})
    current["version"] = int(current.get("version") or 1)
    current["updated_at"] = time.time()
    for key in (
        "objects",
        "equipment",
        "manual_ports",
        "deleted_entities",
        "line_association_overrides",
        "trace_overrides",
    ):
        current[key] = _list_from_payload(payload, key)
    return current


def load_review_workspace(job_dir: str | Path) -> dict[str, Any]:
    path = review_workspace_path(job_dir)
    if not path.exists():
        return build_workspace_from_artifacts(job_dir)
    return _normalize_workspace_payload(_read_json(path), job_id=Path(job_dir).name)


def save_review_workspace(job_dir: str | Path, state: dict[str, Any]) -> Path:
    path = review_workspace_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _normalize_workspace_payload(state, job_id=Path(job_dir).name)
    fd, tmp_path = tempfile.mkstemp(prefix="review_workspace_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return path


def build_workspace_from_artifacts(job_dir: str | Path) -> dict[str, Any]:
    job_dir_path = Path(job_dir)
    stage4 = _read_json(job_dir_path / "stage4_objects.json")
    stage3 = _read_json(job_dir_path / "stage3_equipment_bboxes.json")
    stage5_ports = _read_json(job_dir_path / "stage5_connection_ports.json")
    stage6_review = _read_json(job_dir_path / "stage6_line_number_review.json")

    workspace = empty_review_workspace(job_dir_path.name)
    workspace["image_id"] = stage4.get("image_id")
    workspace["objects"] = _list_from_payload(stage4, "objects")
    workspace["equipment"] = _list_from_payload(stage3, "equipment")
    workspace["manual_ports"] = _flatten_ports(stage5_ports)
    workspace["line_association_overrides"] = _list_from_payload(stage6_review, "accepted")
    return workspace


def _flatten_ports(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ports: list[dict[str, Any]] = []
    for key in ("equipment", "objects", "ports"):
        value = payload.get(key)
        if isinstance(value, list):
            ports.extend(dict(item) for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            for owner_id, items in value.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    port = dict(item)
                    port.setdefault("owner_id", owner_id)
                    port.setdefault("owner_type", key.rstrip("s"))
                    ports.append(port)
    return ports


def workspace_to_stage3_equipment(state: dict[str, Any]) -> dict[str, Any]:
    equipment = []
    for index, item in enumerate(_list_from_payload(state, "equipment")):
        if item.get("review_state") == "rejected" or item.get("ReviewStatus") == "rejected":
            continue
        equipment.append(
            {
                "id": item.get("id") or item.get("Text") or f"equip_{index + 1:03d}",
                "class_name": item.get("class_name") or item.get("Object") or "equipment",
                "bbox": item.get("bbox") or _bbox_from_detected_object(item),
                "source": "hitl",
                "review_state": item.get("review_state") or item.get("ReviewStatus") or "accepted",
            }
        )
    return {"equipment": equipment}


def workspace_to_stage4_objects(state: dict[str, Any], image_id: str | None = None) -> dict[str, Any]:
    objects = []
    for index, item in enumerate(_list_from_payload(state, "objects")):
        if item.get("review_state") == "rejected" or item.get("ReviewStatus") == "rejected":
            continue
        objects.append(
            {
                "id": item.get("id") or item.get("Text") or f"obj_{index + 1:06d}",
                "class_name": item.get("class_name") or item.get("Object") or "object",
                "bbox": item.get("bbox") or _bbox_from_detected_object(item),
                "confidence": item.get("confidence", item.get("Score", 1)),
                "source": "hitl",
                "review_state": item.get("review_state") or item.get("ReviewStatus") or "accepted",
            }
        )
    payload: dict[str, Any] = {"objects": objects}
    resolved_image_id = image_id or state.get("image_id")
    if resolved_image_id:
        payload["image_id"] = resolved_image_id
    return payload


def _bbox_from_detected_object(item: dict[str, Any]) -> dict[str, int]:
    left = int(round(float(item.get("Left", item.get("x_min", 0)) or 0)))
    top = int(round(float(item.get("Top", item.get("y_min", 0)) or 0)))
    width = int(round(float(item.get("Width", 0) or 0)))
    height = int(round(float(item.get("Height", 0) or 0)))
    if width <= 0 and "x_max" in item:
        width = int(round(float(item.get("x_max") or 0))) - left
    if height <= 0 and "y_max" in item:
        height = int(round(float(item.get("y_max") or 0))) - top
    return {
        "x_min": left,
        "y_min": top,
        "x_max": left + max(1, width),
        "y_max": top + max(1, height),
    }
