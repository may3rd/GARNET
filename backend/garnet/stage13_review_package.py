from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

CATEGORY_TYPE = {
    "line_number_conflict": "line_number",
    "line_number_missing_after_propagation": "line_number",
    "unmerged_tee_terminal": "topology",
    "tee_degree_mismatch": "topology",
    "dead_end_not_expected": "topology",
    "duplicate_physical_path": "topology",
    "dead_end_trace": "trace_terminal",
    "duplicate_trace_collapsed": "info",
}

CATEGORY_PRIORITY = {
    "tee_degree_mismatch": 10,
    "line_number_conflict": 9,
    "dead_end_not_expected": 8,
    "duplicate_physical_path": 8,
    "unmerged_tee_terminal": 6,
    "line_number_missing_after_propagation": 6,
    "dead_end_trace": 5,
    "duplicate_trace_collapsed": 2,
}

SEVERITY_PRIORITY = {
    "high": 8,
    "review": 6,
    "medium": 5,
    "info": 2,
    "low": 2,
}

_EVIDENCE_KEYS = {
    "node_id",
    "edge_id",
    "component_id",
    "component_edge_ids",
    "component_trace_ids",
    "candidate_line_number_ids",
    "line_number_ids",
    "effective_line_number_ids",
    "terminal_xy",
    "source",
    "target",
    "trace_id",
    "trace_ids",
    "source_trace_id",
    "target_trace_id",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _category_for(item: dict[str, Any]) -> str:
    return str(item.get("category") or item.get("issue_type") or "review")


def _type_for(category: str) -> str:
    return CATEGORY_TYPE.get(category, "review")


def _priority_for(category: str, severity: str) -> int:
    return CATEGORY_PRIORITY.get(category, SEVERITY_PRIORITY.get(severity, 4))


def _geometry_from_item(item: dict[str, Any]) -> dict[str, float] | None:
    geometry = item.get("geometry")
    if isinstance(geometry, dict) and "x" in geometry and "y" in geometry:
        return {"x": float(geometry["x"]), "y": float(geometry["y"])}
    terminal_xy = item.get("terminal_xy")
    if isinstance(terminal_xy, dict) and "x" in terminal_xy and "y" in terminal_xy:
        return {"x": float(terminal_xy["x"]), "y": float(terminal_xy["y"])}
    if isinstance(terminal_xy, (list, tuple)) and len(terminal_xy) >= 2:
        return {"x": float(terminal_xy[0]), "y": float(terminal_xy[1])}
    return None


def _evidence_from_item(item: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    nested = item.get("evidence")
    if isinstance(nested, dict):
        evidence.update(nested)
    for key in _EVIDENCE_KEYS:
        if key in item:
            evidence[key] = item[key]
    return evidence


def _review_item_from_source(*, image_id: str, source_stage: str, item: dict[str, Any]) -> dict[str, Any]:
    category = _category_for(item)
    severity = str(item.get("severity") or "review")
    source_item_id = str(item.get("id") or f"{source_stage}::{category}")
    review_item = {
        "id": f"stage13::{source_item_id}",
        "image_id": image_id,
        "source_stage": source_stage,
        "source_item_id": source_item_id,
        "review_item_type": _type_for(category),
        "category": category,
        "severity": severity,
        "priority": _priority_for(category, severity),
        "status": "open",
        "message": str(item.get("message") or category.replace("_", " ")),
        "evidence": _evidence_from_item(item),
    }
    geometry = _geometry_from_item(item)
    if geometry is not None:
        review_item["geometry"] = geometry
    return review_item


def _merge_review_items(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    primary, secondary = (incoming, existing) if incoming.get("priority", 0) > existing.get("priority", 0) else (existing, incoming)
    merged = dict(primary)
    merged_evidence = dict(secondary.get("evidence") or {})
    merged_evidence.update(primary.get("evidence") or {})
    source_stages = sorted(set(_as_list(existing.get("source_stage")) + _as_list(incoming.get("source_stage"))))
    merged["source_stage"] = source_stages if len(source_stages) > 1 else source_stages[0]
    merged["evidence"] = merged_evidence
    if "geometry" not in merged and "geometry" in secondary:
        merged["geometry"] = secondary["geometry"]
    return merged


def build_stage13_review_package(
    *,
    image_id: str,
    graph_payload: dict[str, Any],
    stage12_qa_payload: dict[str, Any],
    stage12_review_queue_payload: dict[str, Any],
    stage11_line_number_review_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del graph_payload, stage11_line_number_review_payload
    by_source_id: dict[str, dict[str, Any]] = {}

    for issue in stage12_qa_payload.get("issues", []) or []:
        if not isinstance(issue, dict):
            continue
        item = _review_item_from_source(image_id=image_id, source_stage="stage12_graph_qa", item=issue)
        key = item["source_item_id"]
        by_source_id[key] = _merge_review_items(by_source_id[key], item) if key in by_source_id else item

    for review in stage12_review_queue_payload.get("review_queue", []) or []:
        if not isinstance(review, dict):
            continue
        item = _review_item_from_source(image_id=image_id, source_stage="stage12_review_queue", item=review)
        key = item["source_item_id"]
        by_source_id[key] = _merge_review_items(by_source_id[key], item) if key in by_source_id else item

    review_items = sorted(
        by_source_id.values(),
        key=lambda item: (-int(item.get("priority", 0)), str(item.get("category") or ""), str(item.get("id") or "")),
    )

    category_counts = Counter(str(item.get("category") or "unknown") for item in review_items)
    severity_counts = Counter(str(item.get("severity") or "unknown") for item in review_items)
    priority_counts = Counter(str(item.get("priority") or 0) for item in review_items)
    type_counts = Counter(str(item.get("review_item_type") or "review") for item in review_items)

    return {
        "review_items_payload": {
            "image_id": image_id,
            "source": "stage13_review_package",
            "review_items": review_items,
        },
        "summary": {
            "image_id": image_id,
            "review_item_count": len(review_items),
            "category_counts": dict(category_counts),
            "severity_counts": dict(severity_counts),
            "priority_counts": dict(priority_counts),
            "review_item_type_counts": dict(type_counts),
            "source_artifacts": [
                "stage12_graph.json",
                "stage12_graph_qa.json",
                "stage12_review_queue.json",
                "stage11_line_number_review.json",
            ],
        },
    }


def _color_for_priority(priority: int) -> tuple[int, int, int]:
    if priority >= 8:
        return (0, 0, 255)
    if priority >= 5:
        return (0, 165, 255)
    return (255, 255, 0)


def render_stage13_review_overlay(image_bgr: np.ndarray, review_items_payload: dict[str, Any]) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to render stage13_review_overlay") from exc

    overlay = image_bgr.copy()
    for item in review_items_payload.get("review_items", []) or []:
        if not isinstance(item, dict):
            continue
        geometry = item.get("geometry")
        if not isinstance(geometry, dict) or "x" not in geometry or "y" not in geometry:
            continue
        x = int(round(float(geometry["x"])))
        y = int(round(float(geometry["y"])))
        priority = int(item.get("priority") or 0)
        category = str(item.get("category") or "review")
        color = _color_for_priority(priority)
        cv2.circle(overlay, (x, y), 8, color, thickness=2)
        cv2.circle(overlay, (x, y), 3, color, thickness=-1)
        cv2.putText(
            overlay,
            f"{priority}:{category}",
            (x + 10, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay
