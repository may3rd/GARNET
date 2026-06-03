from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

WORKSPACE_ARTIFACT_NAME = "review_workspace_state.json"
EQUIPMENT_CLASSES = {"pump", "heat exchanger", "tank", "vessel", "column", "compressor", "blower", "fan", "mixer"}


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


def _normalized_class_name(item: dict[str, Any]) -> str:
    return str(item.get("class_name") or item.get("Object") or "").lower().replace("_", " ").replace("-", " ").strip()


def _is_equipment_class(item: dict[str, Any]) -> bool:
    return _normalized_class_name(item) in EQUIPMENT_CLASSES


def _is_line_number_class(item: dict[str, Any]) -> bool:
    return _normalized_class_name(item) == "line number"


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
    stage4_objects = _list_from_payload(stage4, "objects")
    workspace["objects"] = [
        item for item in stage4_objects
        if not _is_equipment_class(item) and not _is_line_number_class(item)
    ]
    workspace["equipment"] = _list_from_payload(stage3, "equipment")
    if not workspace["equipment"]:
        workspace["equipment"] = [item for item in stage4_objects if _is_equipment_class(item)]
    workspace["manual_ports"] = _flatten_ports(stage5_ports)
    workspace["line_association_overrides"] = _list_from_payload(stage6_review, "accepted")
    return workspace


def _flatten_ports(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ports: list[dict[str, Any]] = []
    handled_keys = {"equipment", "objects", "ports"}
    for key in handled_keys:
        value = payload.get(key)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    ports.append(dict(item))
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    ports.append(
                        {
                            "port_id": f"port_{index + 1:02d}",
                            "x": item[0],
                            "y": item[1],
                            "direction": item[2],
                            "source": "stage5",
                        }
                    )
        elif isinstance(value, dict):
            for owner_id, items in value.items():
                if not isinstance(items, list):
                    continue
                for index, item in enumerate(items):
                    if isinstance(item, dict):
                        port = dict(item)
                    elif isinstance(item, (list, tuple)) and len(item) >= 3:
                        port = {
                            "port_id": f"{owner_id}:port_{index + 1:02d}",
                            "x": item[0],
                            "y": item[1],
                            "direction": item[2],
                            "source": "stage5",
                        }
                    else:
                        continue
                    port.setdefault("owner_id", owner_id)
                    port.setdefault("owner_type", key.rstrip("s"))
                    ports.append(port)
    for owner_id, items in payload.items():
        if owner_id in handled_keys or not isinstance(items, list):
            continue
        owner_type = "equipment" if str(owner_id).startswith("equip_") else "object"
        for index, item in enumerate(items):
            if isinstance(item, dict):
                port = dict(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                port = {
                    "port_id": f"{owner_id}:port_{index + 1:02d}",
                    "x": item[0],
                    "y": item[1],
                    "direction": item[2],
                    "source": "stage5",
                }
            else:
                continue
            port.setdefault("owner_id", owner_id)
            port.setdefault("owner_type", owner_type)
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
        if _is_equipment_class(item) or _is_line_number_class(item):
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


def workspace_to_stage5_ports(state: dict[str, Any]) -> dict[str, list[list[Any]]]:
    ports: dict[str, list[list[Any]]] = {}
    for index, item in enumerate(_list_from_payload(state, "manual_ports")):
        if item.get("review_state") == "rejected" or item.get("ReviewStatus") == "rejected":
            continue
        owner_id = item.get("owner_id") or item.get("source_obj_id") or item.get("object_id")
        if not owner_id:
            continue
        try:
            x = int(round(float(item.get("x"))))
            y = int(round(float(item.get("y"))))
        except (TypeError, ValueError):
            continue
        direction = str(item.get("direction") or "").upper()
        if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            continue
        item.setdefault("port_id", f"{owner_id}:port_{index + 1:02d}")
        ports.setdefault(str(owner_id), []).append([x, y, direction])
    return ports


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
