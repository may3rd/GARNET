from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np


DEFAULT_NODE_MERGE_TOLERANCES = {
    "tee_junction": 12.0,
    "branch": 12.0,
    "branch_start": 12.0,
    "junction": 12.0,
    "equipment": 16.0,
    "equipment_port": 16.0,
    "page_connection": 20.0,
    "connection": 20.0,
    "utility_connection": 20.0,
    "dead_end": 8.0,
    "terminal": 8.0,
    "source": 8.0,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _point_from_xy(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return {"x": _as_float(value.get("x")), "y": _as_float(value.get("y"))}
        if "col" in value and "row" in value:
            return {"x": _as_float(value.get("col")), "y": _as_float(value.get("row"))}
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return {"x": _as_float(value[0]), "y": _as_float(value[1])}
    return None


def _dict_polyline(polyline: Any) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    if not isinstance(polyline, list):
        return points
    for item in polyline:
        point = _point_from_xy(item)
        if point is not None:
            points.append(point)
    return points


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _normalize_type(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _source_node_type(edge: dict[str, Any]) -> str:
    source_type = _normalize_type(edge.get("source_obj_type"))
    trace_kind = _normalize_type(edge.get("trace_kind"))
    if trace_kind == "branch":
        return "branch_start"
    if source_type in {"page_connection", "utility_connection", "connection"}:
        return source_type
    source_id = str(edge.get("source_obj_id") or "")
    if source_id.startswith("equip_") or source_type in {"vessel", "pump", "mixer", "equipment"}:
        return "equipment_port"
    return source_type or "source"


def _terminal_node_type(edge: dict[str, Any]) -> str:
    terminal_type = _normalize_type(edge.get("terminal_type"))
    if terminal_type in {"equipment", "instrument_tag", "page_connection", "utility_connection", "connection", "tee_junction", "dead_end"}:
        return terminal_type
    return terminal_type or "terminal"


def _stable_source_node_id(edge: dict[str, Any], node_type: str) -> str | None:
    source_id = str(edge.get("source_obj_id") or "")
    if not source_id:
        return None
    if node_type == "equipment_port":
        port_index = edge.get("port_index")
        try:
            return f"equipment::{source_id}:port_{int(port_index):02d}"
        except (TypeError, ValueError):
            return f"equipment::{source_id}:port"
    if node_type in {"page_connection", "utility_connection", "connection"}:
        return f"connection::{source_id}"
    if node_type == "branch_start":
        return f"branch_start::{source_id}"
    return f"source::{source_id}"


def _stable_terminal_node_id(edge: dict[str, Any], node_type: str) -> str | None:
    terminal_id = str(edge.get("terminal_obj_id") or "")
    if not terminal_id:
        return None
    if node_type == "equipment":
        return f"equipment::{terminal_id}"
    if node_type in {"page_connection", "utility_connection", "connection"}:
        return f"connection::{terminal_id}"
    if node_type == "instrument_tag":
        return f"instrument::{terminal_id}"
    if node_type == "tee_junction":
        return f"junction::{terminal_id}"
    return f"terminal::{node_type}::{terminal_id}"


def _node_tolerance(node_type: str, tolerances: dict[str, float]) -> float:
    return float(tolerances.get(node_type, tolerances.get("terminal", 8.0)))


class _NodeRegistry:
    def __init__(self, tolerances: dict[str, float]) -> None:
        self.tolerances = tolerances
        self.nodes: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.counters: Counter[str] = Counter()

    def add(self, *, node_type: str, position: dict[str, float], stable_id: str | None, evidence: dict[str, Any]) -> str:
        if stable_id and stable_id in self.by_id:
            node = self.by_id[stable_id]
            node.setdefault("evidence", []).append(evidence)
            return stable_id

        if not stable_id:
            tolerance = _node_tolerance(node_type, self.tolerances)
            for node in self.nodes:
                if node.get("type") != node_type:
                    continue
                if _distance(node["position"], position) <= tolerance:
                    node.setdefault("evidence", []).append(evidence)
                    return str(node["id"])

        node_id = stable_id
        if not node_id:
            self.counters[node_type] += 1
            node_id = f"{node_type}::{self.counters[node_type]:05d}"

        node = {
            "id": node_id,
            "type": node_type,
            "kind": node_type,
            "position": {"x": round(float(position["x"]), 3), "y": round(float(position["y"]), 3)},
            "review_state": "accepted",
            "evidence": [evidence],
        }
        self.nodes.append(node)
        self.by_id[node_id] = node
        return node_id


def _line_number_ids(edge: dict[str, Any]) -> list[str]:
    items = ((edge.get("attachments") or {}).get("line_numbers") or [])
    result: list[str] = []
    for item in items:
        item_id = str(item.get("id") or item.get("source_object_id") or "")
        if item_id:
            result.append(item_id)
    return result


def _make_review_item(issue_type: str, trace_id: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    item = {
        "id": f"review::{issue_type}::{trace_id}",
        "issue_type": issue_type,
        "trace_id": trace_id,
        "severity": severity,
        "message": message,
    }
    item.update({key: value for key, value in extra.items() if value is not None})
    return item


def build_trace_graph_from_stage11(
    payload: dict[str, Any],
    *,
    image_id: str | None = None,
    node_merge_tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build an inspectable Stage 12 graph from Stage 11 traced paths.

    Stage 11 trace edges are already topology-vetted walking results. Stage 12
    promotes non-empty traced paths into physical graph edges and records review
    items for missing semantic associations and unresolved terminals.
    """
    tolerances = dict(DEFAULT_NODE_MERGE_TOLERANCES)
    if node_merge_tolerances:
        tolerances.update(node_merge_tolerances)

    resolved_image_id = str(image_id or payload.get("image_id") or "")
    registry = _NodeRegistry(tolerances)
    graph_edges: list[dict[str, Any]] = []
    trace_edge_nodes: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    excluded_edges: list[dict[str, Any]] = []

    for raw_edge in payload.get("trace_edges", []) or []:
        if not isinstance(raw_edge, dict):
            continue
        trace_id = str(raw_edge.get("trace_id") or f"trace_{len(graph_edges) + len(excluded_edges):05d}")
        status = str(raw_edge.get("status") or "")
        segments = raw_edge.get("segments") if isinstance(raw_edge.get("segments"), list) else []
        polyline = _dict_polyline(raw_edge.get("polyline"))

        if status == "skipped_existing_trace" or not segments:
            excluded_edges.append({"trace_id": trace_id, "status": status or "empty_segments"})
            review_queue.append(
                _make_review_item(
                    "skipped_existing_trace",
                    trace_id,
                    "info",
                    "Trace was not promoted to a physical edge because it reuses an existing path or has no segments.",
                    status=status,
                )
            )
            continue

        source_point = _point_from_xy(raw_edge.get("port"))
        terminal_point = _point_from_xy(raw_edge.get("terminal_xy"))
        if source_point is None or terminal_point is None or len(polyline) < 2:
            excluded_edges.append({"trace_id": trace_id, "status": "malformed_trace_geometry"})
            review_queue.append(
                _make_review_item(
                    "malformed_trace_geometry",
                    trace_id,
                    "blocking",
                    "Trace is missing source, terminal, or polyline geometry.",
                )
            )
            continue

        source_type = _source_node_type(raw_edge)
        terminal_type = _terminal_node_type(raw_edge)
        source_node_id = registry.add(
            node_type=source_type,
            position=source_point,
            stable_id=_stable_source_node_id(raw_edge, source_type),
            evidence={
                "role": "source",
                "trace_id": trace_id,
                "source_obj_id": raw_edge.get("source_obj_id"),
                "source_obj_type": raw_edge.get("source_obj_type"),
                "port_index": raw_edge.get("port_index"),
                "port_direction": (raw_edge.get("port") or {}).get("direction") if isinstance(raw_edge.get("port"), dict) else None,
            },
        )
        terminal_node_id = registry.add(
            node_type=terminal_type,
            position=terminal_point,
            stable_id=_stable_terminal_node_id(raw_edge, terminal_type),
            evidence={
                "role": "terminal",
                "trace_id": trace_id,
                "terminal_type": raw_edge.get("terminal_type"),
                "terminal_obj_id": raw_edge.get("terminal_obj_id"),
            },
        )

        line_number_ids = _line_number_ids(raw_edge)
        review_state = "accepted"
        if not line_number_ids or terminal_type in {"dead_end", "terminal"}:
            review_state = "unresolved"

        edge_payload = {
            "id": f"trace::{trace_id}",
            "source": source_node_id,
            "target": terminal_node_id,
            "type": "pipe_trace",
            "line_style": "solid",
            "review_state": review_state,
            "trace_id": trace_id,
            "trace_kind": raw_edge.get("trace_kind"),
            "source_obj_id": raw_edge.get("source_obj_id"),
            "source_obj_type": raw_edge.get("source_obj_type"),
            "source_port_index": raw_edge.get("port_index"),
            "terminal_type": raw_edge.get("terminal_type"),
            "terminal_obj_id": raw_edge.get("terminal_obj_id"),
            "trace_length_px": raw_edge.get("trace_length_px"),
            "polyline": polyline,
            "segments": segments,
            "turns": raw_edge.get("turns") or [],
            "hits": raw_edge.get("hits") or [],
            "attachments": raw_edge.get("attachments") or {},
            "line_number_ids": line_number_ids,
            "warnings": raw_edge.get("warnings") or [],
        }
        graph_edges.append(edge_payload)
        trace_edge_nodes.append(
            {
                "trace_id": trace_id,
                "edge_id": edge_payload["id"],
                "source_node_id": source_node_id,
                "target_node_id": terminal_node_id,
                "source_xy": source_point,
                "terminal_xy": terminal_point,
                "terminal_type": terminal_type,
            }
        )

        if not line_number_ids:
            review_queue.append(
                _make_review_item(
                    "missing_line_number",
                    trace_id,
                    "review",
                    "Trace has no associated line number.",
                    edge_id=edge_payload["id"],
                )
            )
        if terminal_type == "dead_end":
            review_queue.append(
                _make_review_item(
                    "dead_end_trace",
                    trace_id,
                    "review",
                    "Trace ended at a dead end and may need human confirmation.",
                    edge_id=edge_payload["id"],
                    terminal_xy=terminal_point,
                )
            )
        if terminal_type == "terminal" or not raw_edge.get("terminal_type"):
            review_queue.append(
                _make_review_item(
                    "ambiguous_terminal",
                    trace_id,
                    "review",
                    "Trace terminal type is missing or generic.",
                    edge_id=edge_payload["id"],
                    terminal_xy=terminal_point,
                )
            )

    unresolved = payload.get("unresolved") or {}
    for item in unresolved.get("unattached_line_numbers", []) or []:
        item_id = str(item.get("id") or item.get("source_object_id") or item.get("text") or len(review_queue))
        review_queue.append(
            _make_review_item(
                "unattached_line_number",
                item_id,
                "review",
                "Line number was not attached to a traced path.",
                source=item,
            )
        )
    for item in unresolved.get("unattached_instrument_tags", []) or []:
        item_id = str(item.get("id") or item.get("source_object_id") or item.get("text") or len(review_queue))
        review_queue.append(
            _make_review_item(
                "unattached_instrument_tag",
                item_id,
                "review",
                "Instrument tag was not attached to a traced path.",
                source=item,
            )
        )
    for item in unresolved.get("skipped_branches", []) or []:
        branch_id = str(item.get("branch_id") or item.get("id") or len(review_queue))
        review_queue.append(
            _make_review_item(
                "skipped_existing_trace",
                branch_id,
                "info",
                "Branch candidate was skipped because an existing trace already reached it.",
                source=item,
            )
        )

    node_type_counts = Counter(str(node.get("type")) for node in registry.nodes)
    edge_terminal_counts = Counter(str(edge.get("terminal_type")) for edge in graph_edges)
    review_counts = Counter(str(item.get("issue_type")) for item in review_queue)
    line_groups: dict[str, list[str]] = defaultdict(list)
    for edge in graph_edges:
        for line_id in edge.get("line_number_ids") or []:
            line_groups[str(line_id)].append(str(edge["id"]))

    graph_payload = {
        "schema_version": "stage12_trace_graph_v1",
        "image_id": resolved_image_id,
        "trace_source": payload.get("trace_source") or "stage11_trace_associations",
        "nodes": registry.nodes,
        "edges": graph_edges,
        "line_groups": [
            {"line_number_id": line_id, "edge_ids": edge_ids}
            for line_id, edge_ids in sorted(line_groups.items())
        ],
        "review_queue": review_queue,
        "metadata": {
            "source_artifacts": ["stage11_trace_associations.json"],
            "excluded_trace_edges": excluded_edges,
            "node_merge_tolerances_px": tolerances,
        },
    }
    summary = {
        "image_id": resolved_image_id,
        "source": "stage11_trace_associations",
        "node_count": len(registry.nodes),
        "edge_count": len(graph_edges),
        "excluded_trace_edge_count": len(excluded_edges),
        "line_group_count": len(line_groups),
        "review_item_count": len(review_queue),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "terminal_type_counts": dict(sorted(edge_terminal_counts.items())),
        "review_issue_counts": dict(sorted(review_counts.items())),
        "source_trace_edge_count": len(payload.get("trace_edges", []) or []),
    }
    trace_edge_nodes_payload = {
        "image_id": resolved_image_id,
        "trace_edge_nodes": trace_edge_nodes,
        "excluded_trace_edges": excluded_edges,
    }
    review_queue_payload = {
        "image_id": resolved_image_id,
        "review_queue": review_queue,
    }
    review_queue_summary = {
        "image_id": resolved_image_id,
        "review_item_count": len(review_queue),
        "issue_counts": dict(sorted(review_counts.items())),
    }
    return {
        "graph_payload": graph_payload,
        "summary": summary,
        "trace_edge_nodes_payload": trace_edge_nodes_payload,
        "review_queue_payload": review_queue_payload,
        "review_queue_summary": review_queue_summary,
    }


def _as_int_point(point: dict[str, Any]) -> tuple[int, int]:
    return int(round(_as_float(point.get("x")))), int(round(_as_float(point.get("y"))))


def _edge_color(edge: dict[str, Any]) -> tuple[int, int, int]:
    review_state = str(edge.get("review_state") or "").lower()
    terminal_type = _normalize_type(edge.get("terminal_type"))
    if review_state == "accepted":
        return (0, 170, 0)
    if terminal_type == "dead_end":
        return (0, 0, 220)
    if edge.get("line_number_ids"):
        return (0, 165, 255)
    return (0, 80, 255)


def _node_color(node_type: str) -> tuple[int, int, int]:
    normalized = _normalize_type(node_type)
    if normalized in {"tee_junction", "branch_start", "junction"}:
        return (255, 0, 255)
    if normalized in {"equipment", "equipment_port"}:
        return (255, 180, 0)
    if normalized in {"page_connection", "utility_connection", "connection"}:
        return (255, 255, 0)
    if normalized == "dead_end":
        return (0, 0, 220)
    return (180, 180, 180)


def render_stage12_graph_overlay(image_bgr: np.ndarray, graph_payload: dict[str, Any]) -> np.ndarray:
    """Render Stage 12 graph edges and nodes for visual review."""
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to render stage12_graph_overlay") from exc

    overlay = image_bgr.copy()
    node_by_id = {str(node.get("id")): node for node in graph_payload.get("nodes", [])}

    for edge in graph_payload.get("edges", []) or []:
        polyline = edge.get("polyline") or []
        points = [_point_from_xy(point) for point in polyline]
        points = [point for point in points if point is not None]
        if len(points) < 2:
            continue
        color = _edge_color(edge)
        for start, end in zip(points, points[1:]):
            cv2.line(overlay, _as_int_point(start), _as_int_point(end), color, 4, lineType=cv2.LINE_AA)
            cv2.line(overlay, _as_int_point(start), _as_int_point(end), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        mid = points[len(points) // 2]
        label = str(edge.get("trace_id") or edge.get("id") or "")
        line_numbers = edge.get("line_number_ids") or []
        if line_numbers:
            label = f"{label} ln:{len(line_numbers)}"
        x, y = _as_int_point(mid)
        cv2.putText(
            overlay,
            label,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    for node in graph_payload.get("nodes", []) or []:
        position = node.get("position") if isinstance(node.get("position"), dict) else None
        if position is None:
            continue
        node_type = str(node.get("type") or "")
        color = _node_color(node_type)
        x, y = _as_int_point(position)
        radius = 7 if node_type in {"tee_junction", "branch_start", "dead_end"} else 5
        cv2.circle(overlay, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        evidence = node.get("evidence") if isinstance(node.get("evidence"), list) else []
        label = node_type
        if evidence:
            first = evidence[0] if isinstance(evidence[0], dict) else {}
            if first.get("port_index") is not None:
                label = f"{node_type}:p{int(first['port_index']):02d}"
        cv2.putText(
            overlay,
            label,
            (x + 8, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    legend_items = [
        ("accepted", (0, 170, 0)),
        ("review", (0, 80, 255)),
        ("dead_end", (0, 0, 220)),
        ("tee/branch", (255, 0, 255)),
        ("equipment", (255, 180, 0)),
    ]
    x0, y0 = 18, 28
    for index, (label, color) in enumerate(legend_items):
        y = y0 + index * 24
        cv2.circle(overlay, (x0, y - 5), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            label,
            (x0 + 14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    return overlay
