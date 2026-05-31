from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import numpy as np


_COMPONENT_COLORS = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (0, 180, 0),
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
]

_SEVERITY_COLORS = {
    "high": (0, 0, 255),
    "medium": (0, 165, 255),
    "info": (0, 255, 255),
}

_TERMINAL_NODE_TYPES = {"equipment", "equipment_port", "page_connection", "utility_connection", "connection"}


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


def _int_point(point: dict[str, Any]) -> tuple[int, int]:
    return int(round(_as_float(point.get("x")))), int(round(_as_float(point.get("y"))))


def _node_position(node: dict[str, Any]) -> dict[str, float] | None:
    return _point_from_xy(node.get("position"))


def _edge_polyline(edge: dict[str, Any]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for item in edge.get("polyline") or []:
        point = _point_from_xy(item)
        if point is not None:
            points.append(point)
    return points


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _polyline_length(points: list[dict[str, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


def _edge_length(edge: dict[str, Any]) -> float:
    value = edge.get("trace_length_px")
    if value is not None:
        return _as_float(value)
    return _polyline_length(_edge_polyline(edge))


def _node_type(node: dict[str, Any] | None) -> str:
    return str((node or {}).get("type") or (node or {}).get("kind") or "").lower()


def _node_evidence_roles(node: dict[str, Any]) -> set[str]:
    evidence = node.get("evidence") if isinstance(node.get("evidence"), list) else []
    return {str(item.get("role") or "") for item in evidence if isinstance(item, dict)}


def _edge_effective_line_number_ids(edge: dict[str, Any]) -> list[str]:
    values = edge.get("effective_line_number_ids")
    if not values:
        values = edge.get("line_number_ids") or []
    return [str(value) for value in values if str(value)]


def _issue(
    *,
    category: str,
    severity: str,
    message: str,
    node_id: str | None = None,
    edge_id: str | None = None,
    component_id: int | None = None,
    geometry: dict[str, float] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parts = [category]
    if edge_id:
        parts.append(edge_id)
    elif node_id:
        parts.append(node_id)
    elif component_id is not None:
        parts.append(f"component_{component_id}")
    issue_id = "qa::" + "::".join(parts)
    payload: dict[str, Any] = {
        "id": issue_id,
        "category": category,
        "severity": severity,
        "message": message,
    }
    if node_id is not None:
        payload["node_id"] = node_id
    if edge_id is not None:
        payload["edge_id"] = edge_id
    if component_id is not None:
        payload["component_id"] = component_id
    if geometry is not None:
        payload["geometry"] = geometry
    if evidence is not None:
        payload["evidence"] = evidence
    return payload


def _component_geometry(component_nodes: set[str], nodes_by_id: dict[str, dict[str, Any]]) -> dict[str, float] | None:
    points = [_node_position(nodes_by_id[node_id]) for node_id in component_nodes if node_id in nodes_by_id]
    points = [point for point in points if point is not None]
    if not points:
        return None
    return {
        "x": round(sum(point["x"] for point in points) / len(points), 3),
        "y": round(sum(point["y"] for point in points) / len(points), 3),
    }


def _edge_midpoint(edge: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> dict[str, float] | None:
    points = _edge_polyline(edge)
    if points:
        return points[len(points) // 2]
    source = nodes_by_id.get(str(edge.get("source")))
    target = nodes_by_id.get(str(edge.get("target")))
    source_pos = _node_position(source or {})
    target_pos = _node_position(target or {})
    if source_pos is not None and target_pos is not None:
        return {"x": (source_pos["x"] + target_pos["x"]) / 2, "y": (source_pos["y"] + target_pos["y"]) / 2}
    return source_pos or target_pos


def _edge_endpoint_signature(edge: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    points = _edge_polyline(edge)
    if len(points) >= 2:
        return points[0], points[-1]
    source = _node_position(nodes_by_id.get(str(edge.get("source")), {}))
    target = _node_position(nodes_by_id.get(str(edge.get("target")), {}))
    return source, target


def _near_same_endpoints(
    edge_a: dict[str, Any],
    edge_b: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    *,
    tolerance_px: float,
) -> bool:
    a0, a1 = _edge_endpoint_signature(edge_a, nodes_by_id)
    b0, b1 = _edge_endpoint_signature(edge_b, nodes_by_id)
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return False
    same_direction = _distance(a0, b0) <= tolerance_px and _distance(a1, b1) <= tolerance_px
    reverse_direction = _distance(a0, b1) <= tolerance_px and _distance(a1, b0) <= tolerance_px
    return same_direction or reverse_direction


def _build_graph(graph_payload: dict[str, Any]) -> tuple[nx.MultiGraph, dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    graph = nx.MultiGraph()
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges_by_id: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []

    for node in graph_payload.get("nodes", []) or []:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        if node_id in nodes_by_id:
            issues.append(_issue(category="duplicate_node_id", severity="high", message="Duplicate node id in Stage 12 graph.", node_id=node_id))
            continue
        nodes_by_id[node_id] = node
        graph.add_node(node_id, **node)

    for edge in graph_payload.get("edges", []) or []:
        edge_id = str(edge.get("id") or "")
        if not edge_id:
            continue
        if edge_id in edges_by_id:
            issues.append(_issue(category="duplicate_edge_id", severity="high", message="Duplicate edge id in Stage 12 graph.", edge_id=edge_id))
            continue
        edges_by_id[edge_id] = edge
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source == target or source not in nodes_by_id or target not in nodes_by_id:
            issues.append(
                _issue(
                    category="self_loop_or_bad_endpoint",
                    severity="high",
                    message="Edge has a self-loop or references a missing node.",
                    edge_id=edge_id,
                    geometry=_edge_midpoint(edge, nodes_by_id),
                    evidence={"source": source, "target": target},
                )
            )
            continue
        graph.add_edge(source, target, key=edge_id, edge_id=edge_id, **edge)
    return graph, nodes_by_id, edges_by_id, issues


def run_stage12_trace_graph_qa(
    *,
    image_id: str,
    graph_payload: dict[str, Any],
    image_bgr: np.ndarray,
    duplicate_endpoint_tolerance_px: float = 8.0,
    short_trace_length_px: float = 20.0,
    min_component_length_for_line_number_px: float = 80.0,
    high_review_density_ratio: float = 0.5,
) -> dict[str, Any]:
    graph, nodes_by_id, edges_by_id, issues = _build_graph(graph_payload)
    edge_to_component: dict[str, int] = {}
    node_to_component: dict[str, int] = {}
    components: list[set[str]] = []
    if graph.number_of_nodes():
        components = [set(component) for component in nx.connected_components(graph)]
    for component_id, component_nodes in enumerate(components):
        for node_id in component_nodes:
            node_to_component[node_id] = component_id
        for source, target, _key, attrs in graph.edges(component_nodes, keys=True, data=True):
            edge_id = str(attrs.get("edge_id") or attrs.get("id") or "")
            if edge_id:
                edge_to_component[edge_id] = component_id

    degree_by_node = dict(graph.degree())
    for node_id, node in nodes_by_id.items():
        node_type = _node_type(node)
        degree = int(degree_by_node.get(node_id, 0))
        position = _node_position(node)
        if node_type == "equipment_port" and degree == 0:
            issues.append(
                _issue(
                    category="dangling_equipment_port",
                    severity="medium",
                    message="Equipment port node is not connected to any trace edge.",
                    node_id=node_id,
                    component_id=node_to_component.get(node_id),
                    geometry=position,
                    evidence={"degree": degree},
                )
            )
        if node_type == "tee_junction" and degree < 3:
            evidence_roles = _node_evidence_roles(node)
            if evidence_roles and evidence_roles <= {"terminal"}:
                issues.append(
                    _issue(
                        category="unmerged_tee_terminal",
                        severity="medium",
                        message="Trace ended at a tee-like terminal, but no corroborating branch/source path is connected there.",
                        node_id=node_id,
                        component_id=node_to_component.get(node_id),
                        geometry=position,
                        evidence={"degree": degree, "evidence_roles": sorted(evidence_roles)},
                    )
                )
                continue
            issues.append(
                _issue(
                    category="tee_degree_mismatch",
                    severity="high",
                    message="Tee junction node has degree below 3.",
                    node_id=node_id,
                    component_id=node_to_component.get(node_id),
                    geometry=position,
                    evidence={"degree": degree},
                )
            )

    edge_items = list(edges_by_id.values())
    for edge in edge_items:
        edge_id = str(edge.get("id"))
        component_id = edge_to_component.get(edge_id)
        midpoint = _edge_midpoint(edge, nodes_by_id)
        edge_length = _edge_length(edge)
        source_type = _node_type(nodes_by_id.get(str(edge.get("source"))))
        target_type = _node_type(nodes_by_id.get(str(edge.get("target"))))
        terminal_type = str(edge.get("terminal_type") or "").lower()
        if terminal_type == "dead_end" and source_type not in _TERMINAL_NODE_TYPES and target_type not in _TERMINAL_NODE_TYPES:
            issues.append(
                _issue(
                    category="dead_end_not_expected",
                    severity="medium",
                    message="Trace ended at dead end without touching equipment or page connector.",
                    edge_id=edge_id,
                    component_id=component_id,
                    geometry=midpoint,
                    evidence={"source_type": source_type, "target_type": target_type},
                )
            )
        if 0 < edge_length < short_trace_length_px:
            issues.append(
                _issue(
                    category="short_trace_edge",
                    severity="info",
                    message="Trace edge is very short and may be an artifact.",
                    edge_id=edge_id,
                    component_id=component_id,
                    geometry=midpoint,
                    evidence={"length_px": round(edge_length, 3)},
                )
            )

    for index, edge_a in enumerate(edge_items):
        for edge_b in edge_items[index + 1 :]:
            same_nodes = {str(edge_a.get("source")), str(edge_a.get("target"))} == {str(edge_b.get("source")), str(edge_b.get("target"))}
            same_path = same_nodes or _near_same_endpoints(edge_a, edge_b, nodes_by_id, tolerance_px=duplicate_endpoint_tolerance_px)
            if not same_path:
                continue
            issues.append(
                _issue(
                    category="duplicate_physical_path",
                    severity="high",
                    message="Two Stage 12 edges appear to represent the same physical path.",
                    edge_id=str(edge_b.get("id")),
                    component_id=edge_to_component.get(str(edge_b.get("id"))),
                    geometry=_edge_midpoint(edge_b, nodes_by_id),
                    evidence={"other_edge_id": str(edge_a.get("id"))},
                )
            )

    component_edges: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge_id, edge in edges_by_id.items():
        component_id = edge_to_component.get(edge_id)
        if component_id is not None:
            component_edges[component_id].append(edge)

    line_components: dict[str, set[int]] = defaultdict(set)
    for component_id, edges in component_edges.items():
        component_length = sum(_edge_length(edge) for edge in edges)
        line_ids = {line_id for edge in edges for line_id in _edge_effective_line_number_ids(edge)}
        review_edges = [edge for edge in edges if str(edge.get("review_state") or "") != "accepted"]
        component_nodes = components[component_id] if component_id < len(components) else set()
        component_node_types = {_node_type(nodes_by_id.get(node_id)) for node_id in component_nodes}
        geometry = _component_geometry(component_nodes, nodes_by_id)

        if component_length > min_component_length_for_line_number_px and not line_ids:
            issues.append(
                _issue(
                    category="missing_line_number_component",
                    severity="info",
                    message="Connected trace component has no associated line number.",
                    component_id=component_id,
                    geometry=geometry,
                    evidence={"edge_count": len(edges), "length_px": round(component_length, 3)},
                )
            )
        if len(edges) <= 2 and not (component_node_types & _TERMINAL_NODE_TYPES):
            issues.append(
                _issue(
                    category="isolated_component",
                    severity="medium",
                    message="Small trace component does not touch equipment or page connector.",
                    component_id=component_id,
                    geometry=geometry,
                    evidence={"edge_count": len(edges), "node_types": sorted(component_node_types)},
                )
            )
        if edges and len(review_edges) / len(edges) >= high_review_density_ratio:
            issues.append(
                _issue(
                    category="high_review_density_component",
                    severity="info",
                    message="Component contains a high ratio of unresolved/review edges.",
                    component_id=component_id,
                    geometry=geometry,
                    evidence={"edge_count": len(edges), "review_edge_count": len(review_edges)},
                )
            )
        for line_id in line_ids:
            line_components[line_id].add(component_id)

    for line_id, component_ids in sorted(line_components.items()):
        if len(component_ids) <= 1:
            continue
        first_component = min(component_ids)
        issues.append(
            _issue(
                category="line_number_split_components",
                severity="high",
                message="Same line number appears on disconnected graph components.",
                component_id=first_component,
                geometry=_component_geometry(components[first_component], nodes_by_id) if first_component < len(components) else None,
                evidence={"line_number_id": line_id, "component_ids": sorted(component_ids)},
            )
        )

    issue_counts = Counter(str(issue.get("category")) for issue in issues)
    severity_counts = Counter(str(issue.get("severity")) for issue in issues)
    summary = {
        "image_id": image_id,
        "source": "stage12_trace_graph",
        "node_count": len(nodes_by_id),
        "edge_count": len(edges_by_id),
        "connected_component_count": len(components),
        "issue_count": len(issues),
        "issue_counts": dict(sorted(issue_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "source_artifacts": ["stage12_graph.json"],
    }
    qa_payload = {
        "image_id": image_id,
        "source": "stage12_trace_graph",
        "components": [
            {
                "component_id": component_id,
                "node_ids": sorted(component_nodes),
                "edge_ids": sorted(edge.get("id") for edge in component_edges.get(component_id, [])),
                "edge_count": len(component_edges.get(component_id, [])),
                "node_count": len(component_nodes),
                "total_length_px": round(sum(_edge_length(edge) for edge in component_edges.get(component_id, [])), 3),
            }
            for component_id, component_nodes in enumerate(components)
        ],
        "issues": issues,
    }
    return {
        "qa_payload": qa_payload,
        "summary": summary,
        "overlay_image": render_stage12_graph_qa_overlay(
            image_bgr=image_bgr,
            graph_payload=graph_payload,
            qa_payload=qa_payload,
            component_by_edge=edge_to_component,
        ),
    }


def render_stage12_graph_qa_overlay(
    *,
    image_bgr: np.ndarray,
    graph_payload: dict[str, Any],
    qa_payload: dict[str, Any],
    component_by_edge: dict[str, int] | None = None,
) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to render stage12_graph_qa_overlay") from exc

    overlay = image_bgr.copy()
    component_by_edge = component_by_edge or {}
    nodes_by_id = {str(node.get("id")): node for node in graph_payload.get("nodes", [])}

    for edge in graph_payload.get("edges", []) or []:
        edge_id = str(edge.get("id") or "")
        component_id = component_by_edge.get(edge_id, 0)
        color = _COMPONENT_COLORS[component_id % len(_COMPONENT_COLORS)]
        points = _edge_polyline(edge)
        for start, end in zip(points, points[1:]):
            cv2.line(overlay, _int_point(start), _int_point(end), color, 3, lineType=cv2.LINE_AA)

    for node in graph_payload.get("nodes", []) or []:
        position = _node_position(node)
        if position is None:
            continue
        cv2.circle(overlay, _int_point(position), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, _int_point(position), 4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    for issue in qa_payload.get("issues", []) or []:
        geometry = _point_from_xy(issue.get("geometry"))
        if geometry is None and issue.get("node_id") in nodes_by_id:
            geometry = _node_position(nodes_by_id[str(issue.get("node_id"))])
        if geometry is None:
            continue
        severity = str(issue.get("severity") or "info")
        color = _SEVERITY_COLORS.get(severity, (0, 255, 255))
        x, y = _int_point(geometry)
        cv2.circle(overlay, (x, y), 11, color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(issue.get("category") or "qa"),
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    legend = [("high", _SEVERITY_COLORS["high"]), ("medium", _SEVERITY_COLORS["medium"]), ("info", _SEVERITY_COLORS["info"])]
    x0, y0 = 18, 28
    for idx, (label, color) in enumerate(legend):
        y = y0 + idx * 24
        cv2.circle(overlay, (x0, y - 5), 8, color, 2, lineType=cv2.LINE_AA)
        cv2.putText(overlay, label, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, lineType=cv2.LINE_AA)
    return overlay
