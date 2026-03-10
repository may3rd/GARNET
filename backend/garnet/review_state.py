from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

VALID_BUCKETS = {
    "stage4_line_number",
    "stage4_instrument",
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
            "stage4_line_number": [],
            "stage4_instrument": [],
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
