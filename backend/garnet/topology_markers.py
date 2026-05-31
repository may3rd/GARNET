from __future__ import annotations

from collections import Counter
from typing import Any


def _normalize_class_name(value: str) -> str:
    lowered = str(value).strip().lower()
    for ch in "-_/":
        lowered = lowered.replace(ch, " ")
    return " ".join(lowered.split())


def _marker_role(class_name: str) -> str | None:
    normalized = _normalize_class_name(class_name)
    if normalized in {"arrow", "flow arrow", "direction arrow"}:
        return "flow_marker"
    if normalized in {"node", "junction node", "junction marker"}:
        return "junction_marker"
    return None


def run_topology_marker_router(
    *,
    image_id: str,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    for obj in objects:
        role = _marker_role(str(obj.get("class_name", "")))
        if role is None:
            continue
        markers.append(
            {
                "id": f"topology_marker::{obj['id']}",
                "source_object_id": obj["id"],
                "class_name": obj.get("class_name", ""),
                "role": role,
                "confidence": float(obj.get("confidence", 0.0)),
                "bbox": dict(obj.get("bbox", {})),
            }
        )

    class_counts = dict(sorted(Counter(str(item["class_name"]) for item in markers).items()))
    role_counts = dict(sorted(Counter(str(item["role"]) for item in markers).items()))
    return {
        "topology_markers_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "topology_markers": markers,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "topology_marker_count": len(markers),
            "class_counts": class_counts,
            "role_counts": role_counts,
            "source_artifacts": [
                "stage4_objects.json",
            ],
        },
    }
