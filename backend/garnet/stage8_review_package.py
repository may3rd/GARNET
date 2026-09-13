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
    "missing_line_number": "line_number",
    "unattached_line_number": "line_number",
    "ambiguous_terminal": "trace_terminal",
    "unresolved_terminal_edge": "trace_terminal",
    "malformed_trace_geometry": "trace_terminal",
    "duplicate_node_id": "topology",
    "duplicate_edge_id": "topology",
    "self_loop_or_bad_endpoint": "topology",
    "dangling_equipment_port": "topology",
    "articulation_point": "topology",
    "isolated_node": "topology",
    "isolated_component": "topology",
    "unresolved_crossing": "topology",
    "missing_line_number_component": "line_number",
    "line_number_split_components": "line_number",
}

_RELEASE_BLOCKING_TYPES = {"topology", "line_number", "flow_direction", "trace_terminal"}

# These are identifiers already emitted by the graph and QA stages.  They are
# promoted to the review item when present so a client can construct a Stage 9
# decision without having to re-read the source artifact.  Geometry is kept in
# the evidence payload and is never synthesized here.
_IDENTIFIER_KEYS = (
    "node_id",
    "node_ids",
    "edge_id",
    "edge_ids",
    "component_id",
    "component_edge_ids",
    "component_trace_ids",
    "trace_id",
    "trace_ids",
    "source_trace_id",
    "target_trace_id",
    "source",
    "target",
    "source_node_id",
    "target_node_id",
    "destination_node_id",
    "other_edge_id",
    "candidate_node_id",
    "candidate_edge_id",
    "junction_id",
    "branch_id",
    "terminal_node_id",
)

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


def _release_blocking_for(review_item_type: str, category: str, severity: str) -> bool:
    """Return the deterministic release-gate classification for an item."""
    del category
    # Severity ``info`` and the explicit info review type are advisory even if
    # a future producer gives them a category that otherwise requires review.
    if review_item_type == "info" or severity.strip().lower() == "info":
        return False
    # Unknown non-info review types are conservatively blocking.  A newly
    # introduced QA category must not silently pass the release gate until it
    # has an explicit informational classification.
    return review_item_type in _RELEASE_BLOCKING_TYPES or review_item_type != "info"


def _identifier_values(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return list(dict.fromkeys(str(item) for item in values if item is not None and str(item)))


def _target_ids_for(identifier_values: dict[str, Any]) -> dict[str, list[str]]:
    """Group existing identifiers into decision-friendly target lists."""
    node_keys = {
        "node_id",
        "node_ids",
        "source_node_id",
        "target_node_id",
        "destination_node_id",
        "candidate_node_id",
        "junction_id",
        "terminal_node_id",
        "source",
        "target",
    }
    edge_keys = {"edge_id", "edge_ids", "component_edge_ids", "candidate_edge_id", "other_edge_id"}
    trace_keys = {"trace_id", "trace_ids", "component_trace_ids", "source_trace_id", "target_trace_id"}
    targets: dict[str, list[str]] = {}
    for output_key, keys in (("node_ids", node_keys), ("edge_ids", edge_keys), ("trace_ids", trace_keys)):
        values: list[str] = []
        for key in _IDENTIFIER_KEYS:
            if key in keys:
                values.extend(_identifier_values(identifier_values.get(key)))
        if values:
            targets[output_key] = list(dict.fromkeys(values))
    return targets


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


def _promoted_identifiers(item: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Copy identifier evidence to stable top-level fields without fabrication."""
    identifiers: dict[str, Any] = {}
    for key in _IDENTIFIER_KEYS:
        if key in item:
            identifiers[key] = deepcopy(item[key])
        elif key in evidence:
            identifiers[key] = deepcopy(evidence[key])
    targets = _target_ids_for(identifiers)
    if targets:
        identifiers["target_ids"] = targets
    return identifiers


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
    item["release_blocking"] = True
    item["release_relevance"] = "blocking"
    item["target_ids"] = {"edge_ids": [edge_id]}
    if geometry:
        item["geometry"] = geometry
    return item


def _review_item_from_source(
    *,
    image_id: str,
    source_stage: str,
    item: dict[str, Any],
    graph_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    identifiers = _promoted_identifiers(item, review_item["evidence"])
    review_item.update(identifiers)

    # QA often identifies an edge but stores only its endpoint types in the
    # issue evidence.  Copy the existing endpoint IDs and route polyline from
    # the graph so topology decisions have concrete targets.  Do not derive a
    # replacement point or otherwise manufacture geometry.
    edge_id = str(review_item.get("edge_id") or "")
    if graph_payload is not None and edge_id:
        graph_edge = next(
            (
                candidate
                for candidate in graph_payload.get("edges", []) or []
                if isinstance(candidate, dict) and str(candidate.get("id") or "") == edge_id
            ),
            None,
        )
        if graph_edge is not None:
            for key in ("source", "target"):
                value = graph_edge.get(key)
                if value is None:
                    continue
                review_item.setdefault(key, deepcopy(value))
                review_item["evidence"].setdefault(key, deepcopy(value))
            polyline = graph_edge.get("polyline")
            if isinstance(polyline, list) and polyline:
                review_item.setdefault("edge_geometry", deepcopy(polyline))
                review_item["evidence"].setdefault("edge_geometry", deepcopy(polyline))
            identifiers = _promoted_identifiers(review_item, review_item["evidence"])
            review_item.update(identifiers)
    blocking = _release_blocking_for(review_item["review_item_type"], category, severity)
    review_item["release_blocking"] = blocking
    review_item["release_relevance"] = "blocking" if blocking else "informational"
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
    for key in _IDENTIFIER_KEYS + ("target_ids",):
        if key not in merged and key in secondary:
            merged[key] = deepcopy(secondary[key])
    if "target_ids" in existing or "target_ids" in incoming:
        target_ids: dict[str, list[str]] = {}
        for source in (existing.get("target_ids") or {}, incoming.get("target_ids") or {}):
            if not isinstance(source, dict):
                continue
            for key, values in source.items():
                target_ids[key] = list(dict.fromkeys(target_ids.get(key, []) + _identifier_values(values)))
        if target_ids:
            merged["target_ids"] = target_ids
    # A lower-priority duplicate source must never turn a blocking item into
    # an informational one.  Recompute the derived fields after the merge.
    merged["release_blocking"] = bool(existing.get("release_blocking")) or bool(incoming.get("release_blocking"))
    merged["release_relevance"] = "blocking" if merged["release_blocking"] else "informational"
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
        item = _review_item_from_source(
            image_id=image_id,
            source_stage="stage7_graph_qa",
            item=issue,
            graph_payload=graph_payload,
        )
        key = item["source_item_id"]
        by_source_id[key] = _merge_review_items(by_source_id[key], item) if key in by_source_id else item

    for review in stage7_review_queue_payload.get("review_queue", []) or []:
        if not isinstance(review, dict):
            continue
        item = _review_item_from_source(
            image_id=image_id,
            source_stage="stage7_review_queue",
            item=review,
            graph_payload=graph_payload,
        )
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
    blocking_count = sum(1 for item in review_items if bool(item.get("release_blocking")))

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
            "blocking_review_item_count": blocking_count,
            "release_blocking_review_item_count": blocking_count,
            "informational_review_item_count": len(review_items) - blocking_count,
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
