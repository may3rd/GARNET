from __future__ import annotations

from collections import Counter
from copy import deepcopy
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
    "abandoned_trace": "trace_terminal",
    "flow_direction_unknown": "flow_direction",
    "flow_direction_conflict": "flow_direction",
    "unknown_flow_direction": "flow_direction",
    "conflicting_flow_direction": "flow_direction",
}

CATEGORY_PRIORITY = {
    "tee_degree_mismatch": 10,
    "abandoned_trace": 9,
    "line_number_conflict": 9,
    "dead_end_not_expected": 8,
    "duplicate_physical_path": 8,
    "unmerged_tee_terminal": 6,
    "line_number_missing_after_propagation": 6,
    "dead_end_trace": 5,
    "duplicate_trace_collapsed": 2,
    "flow_direction_unknown": 7,
    "flow_direction_conflict": 10,
    "unknown_flow_direction": 7,
    "conflicting_flow_direction": 10,
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
    "flow_direction_state",
    "flow_direction_evidence",
    "direction_evidence",
    "edge_geometry",
}

_FLOW_STATE_ALIASES = {
    "both": "bidirectional",
    "bi_directional": "bidirectional",
    "forward_only": "forward",
    "reverse_only": "reverse",
}
_FLOW_REVIEW_STATES = {"unknown", "conflicting"}


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


def _flow_state_for_edge(edge: dict[str, Any]) -> str:
    raw_state = edge.get("flow_direction_state")
    if raw_state is None:
        raw_state = edge.get("flow_direction")
    state = str(raw_state or "unknown").strip().lower().replace("-", "_")
    return _FLOW_STATE_ALIASES.get(state, state)


def _flow_direction_evidence_for_edge(edge: dict[str, Any]) -> Any:
    evidence = edge.get("flow_direction_evidence")
    if not evidence:
        evidence = edge.get("direction_evidence")
    if not evidence:
        attachments = edge.get("attachments")
        if isinstance(attachments, dict):
            evidence = attachments.get("flow_arrows")
    return deepcopy(evidence if evidence is not None else [])


def _flow_review_item_from_edge(*, image_id: str, edge: dict[str, Any]) -> dict[str, Any] | None:
    edge_id = str(edge.get("id") or "")
    if not edge_id:
        return None
    state = _flow_state_for_edge(edge)
    if state not in _FLOW_REVIEW_STATES:
        return None

    category = "flow_direction_conflict" if state == "conflicting" else "flow_direction_unknown"
    direction_evidence = _flow_direction_evidence_for_edge(edge)
    polyline = edge.get("polyline")
    geometry: dict[str, Any] = {}
    if isinstance(polyline, list) and polyline:
        geometry["polyline"] = deepcopy(polyline)
    # Keep endpoint geometry available for consumers that cannot render a
    # complete polyline.  These fields are additive and preserve the existing
    # point geometry shape used by older review items.
    for output_key, input_key in (
        ("source", "source_point"),
        ("target", "target_point"),
        ("source", "source_xy"),
        ("target", "terminal_xy"),
    ):
        point = edge.get(input_key)
        if point is not None:
            geometry[output_key] = deepcopy(point)
    if not geometry:
        edge_geometry = edge.get("geometry")
        if isinstance(edge_geometry, dict):
            geometry = deepcopy(edge_geometry)

    item: dict[str, Any] = {
        "id": f"stage8::flow_direction::{edge_id}",
        "image_id": image_id,
        "source_stage": "stage7_graph",
        "source_item_id": f"flow_direction::{edge_id}",
        "review_item_type": "flow_direction",
        "category": category,
        "severity": "high" if state == "conflicting" else "review",
        "priority": _priority_for(category, "high" if state == "conflicting" else "review"),
        "status": "open",
        "message": (
            "Pipe edge has conflicting flow direction evidence; human direction review is required."
            if state == "conflicting"
            else "Pipe edge has no resolved flow direction; human direction review is required."
        ),
        "edge_id": edge_id,
        "direction_evidence": deepcopy(direction_evidence),
        "edge_geometry": deepcopy(geometry),
        "evidence": {
            "edge_id": edge_id,
            "flow_direction_state": state,
            "flow_direction_evidence": deepcopy(direction_evidence),
            "direction_evidence": deepcopy(direction_evidence),
            "edge_geometry": deepcopy(geometry),
        },
    }
    if geometry:
        item["geometry"] = geometry
    return item


def _review_item_from_source(*, image_id: str, source_stage: str, item: dict[str, Any]) -> dict[str, Any]:
    category = _category_for(item)
    severity = str(item.get("severity") or "review")
    source_item_id = str(item.get("id") or f"{source_stage}::{category}")
    review_item = {
        "id": f"stage8::{source_item_id}",
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


def build_stage8_review_package(
    *,
    image_id: str,
    graph_payload: dict[str, Any],
    stage7_qa_payload: dict[str, Any],
    stage7_review_queue_payload: dict[str, Any],
    stage6_line_number_review_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del stage6_line_number_review_payload
    by_source_id: dict[str, dict[str, Any]] = {}

    # Flow direction is a separate fact from physical connectivity.  Promote
    # unresolved edge states into explicit Stage 8 review items while keeping
    # the observed arrow evidence and edge polyline attached to the item.
    for edge in graph_payload.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        item = _flow_review_item_from_edge(image_id=image_id, edge=edge)
        if item is None:
            continue
        key = item["source_item_id"]
        by_source_id[key] = _merge_review_items(by_source_id[key], item) if key in by_source_id else item

    for issue in stage7_qa_payload.get("issues", []) or []:
        if not isinstance(issue, dict):
            continue
        item = _review_item_from_source(image_id=image_id, source_stage="stage7_graph_qa", item=issue)
        key = item["source_item_id"]
        by_source_id[key] = _merge_review_items(by_source_id[key], item) if key in by_source_id else item

    for review in stage7_review_queue_payload.get("review_queue", []) or []:
        if not isinstance(review, dict):
            continue
        item = _review_item_from_source(image_id=image_id, source_stage="stage7_review_queue", item=review)
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
            "source": "stage8_review_package",
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
                "stage7_graph.json",
                "stage7_graph_qa.json",
                "stage7_review_queue.json",
                "stage6_line_number_review.json",
            ],
        },
    }


def _color_for_priority(priority: int) -> tuple[int, int, int]:
    if priority >= 8:
        return (0, 0, 255)
    if priority >= 5:
        return (0, 165, 255)
    return (255, 255, 0)


def render_stage8_review_overlay(image_bgr: np.ndarray, review_items_payload: dict[str, Any]) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to render stage8_review_overlay") from exc

    overlay = image_bgr.copy()
    for item in review_items_payload.get("review_items", []) or []:
        if not isinstance(item, dict):
            continue
        geometry = item.get("geometry")
        if not isinstance(geometry, dict):
            continue
        priority = int(item.get("priority") or 0)
        category = str(item.get("category") or "review")
        color = _color_for_priority(priority)
        polyline = geometry.get("polyline")
        points: list[tuple[int, int]] = []
        if isinstance(polyline, list):
            for point in polyline:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    points.append((int(round(float(point["x"]))), int(round(float(point["y"])))))
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((int(round(float(point[0]))), int(round(float(point[1])))))
        if len(points) >= 2:
            cv2.polylines(overlay, [np.asarray(points, dtype=np.int32)], False, color, thickness=2)
            x, y = points[0]
        elif "x" in geometry and "y" in geometry:
            x = int(round(float(geometry["x"])))
            y = int(round(float(geometry["y"])))
            cv2.circle(overlay, (x, y), 8, color, thickness=2)
            cv2.circle(overlay, (x, y), 3, color, thickness=-1)
        else:
            continue
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
