from __future__ import annotations

import copy
import math
from typing import Any


DEFAULT_INLINE_MATCH_DISTANCE_PX = 36.0


def _bbox_from_object(item: dict[str, Any]) -> dict[str, float] | None:
    bbox = item.get("bbox", item)
    if isinstance(bbox, dict):
        try:
            x_min = float(bbox["x_min"])
            y_min = float(bbox["y_min"])
            x_max = float(bbox["x_max"])
            y_max = float(bbox["y_max"])
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x_min, y_min, x_max, y_max = (float(value) for value in bbox[:4])
        except (TypeError, ValueError):
            return None
    else:
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def _bbox_center(bbox: dict[str, float]) -> tuple[float, float]:
    return ((bbox["x_min"] + bbox["x_max"]) / 2.0, (bbox["y_min"] + bbox["y_max"]) / 2.0)


def _polyline_points(edge: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in edge.get("polyline", []):
        try:
            points.append((float(point["col"]), float(point["row"])))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _closest_polyline_index(edge: dict[str, Any], point_xy: tuple[float, float]) -> tuple[int | None, float]:
    points = _polyline_points(edge)
    if not points:
        return None, math.inf
    px, py = point_xy
    best_index = 0
    best_distance = math.inf
    for idx, (x, y) in enumerate(points):
        distance = math.hypot(px - x, py - y)
        if distance < best_distance:
            best_index = idx
            best_distance = distance
    return best_index, best_distance


def _object_id(item: dict[str, Any]) -> str:
    return str(item.get("id", item.get("object_id", item.get("det_id", ""))))


def _connection_distance(connection: dict[str, Any], fallback: float) -> float:
    for key in ("distance_px", "distance_to_edge_px", "nearest_distance_px"):
        value = connection.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    features = connection.get("features")
    if isinstance(features, dict):
        for key in ("distance_px", "distance_to_edge_px", "nearest_distance_px"):
            value = features.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return fallback


def _match_distance_px(connection: dict[str, Any]) -> float:
    for key in ("inline_match_distance_px", "match_distance_px", "max_distance_px"):
        value = connection.get(key)
        if value is None:
            continue
        try:
            distance = float(value)
        except (TypeError, ValueError):
            continue
        if distance > 0:
            return distance
    return DEFAULT_INLINE_MATCH_DISTANCE_PX


def _edge_with_split_defaults(edge: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(edge)
    result.setdefault("is_split_edge", False)
    result.setdefault("split_parent_edge_id", None)
    result.setdefault("split_position", None)
    result.setdefault("inline_node_id", None)
    return result


def _resolve_edge_to_split(
    edge_id: str,
    active_edges: dict[str, dict[str, Any]],
    center_xy: tuple[float, float],
) -> dict[str, Any] | None:
    edge = active_edges.get(edge_id)
    if edge is not None:
        return edge
    candidates = [
        item
        for item in active_edges.values()
        if str(item.get("split_parent_edge_id", "")) == edge_id
        or str(item.get("id", "")).startswith(f"{edge_id}::split::")
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: _closest_polyline_index(item, center_xy)[1])


def _make_split_edge(
    edge: dict[str, Any],
    *,
    edge_id: str,
    source: str,
    target: str,
    polyline: list[dict[str, Any]],
    split_position: str,
    inline_node_id: str,
) -> dict[str, Any]:
    result = copy.deepcopy(edge)
    result["id"] = edge_id
    result["source"] = source
    result["target"] = target
    result["polyline"] = copy.deepcopy(polyline)
    result["is_split_edge"] = True
    result["split_parent_edge_id"] = edge.get("split_parent_edge_id") or edge.get("id")
    result["split_position"] = split_position
    result["inline_node_id"] = inline_node_id
    return result


def split_edges_at_inline_elements(
    *,
    edges: list[dict[str, Any]],
    inline_connections: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    object_by_id = {_object_id(obj): obj for obj in objects if _object_id(obj)}
    active_edges = {str(edge.get("id", "")): _edge_with_split_defaults(edge) for edge in edges}
    split_nodes: list[dict[str, Any]] = []
    split_report: list[dict[str, Any]] = []

    for connection in inline_connections:
        if str(connection.get("kind", "")) != "inline_element":
            continue

        connector_id = str(connection.get("connector_id", ""))
        connector_class = str(connection.get("connector_class", ""))
        source_edge_id = str(connection.get("source_edge_id", ""))
        target_edge_id = str(connection.get("target_edge_id", ""))
        report_base = {
            "connector_id": connector_id,
            "connector_class": connector_class,
            "edge_id": source_edge_id if source_edge_id == target_edge_id else None,
            "distance_px": 0.0,
            "split_index": None,
        }

        obj = object_by_id.get(connector_id)
        if obj is None:
            split_report.append({**report_base, "status": "skipped_missing_object"})
            continue
        bbox = _bbox_from_object(obj)
        if bbox is None:
            split_report.append({**report_base, "status": "skipped_missing_object"})
            continue
        if source_edge_id != target_edge_id:
            split_report.append({**report_base, "status": "skipped_already_connected"})
            continue

        center_xy = _bbox_center(bbox)
        edge = _resolve_edge_to_split(source_edge_id, active_edges, center_xy)
        if edge is None:
            split_report.append({**report_base, "status": "skipped_no_edge"})
            continue

        split_index, computed_distance = _closest_polyline_index(edge, center_xy)
        distance_px = _connection_distance(connection, computed_distance)
        report_base = {**report_base, "edge_id": str(edge.get("id", "")), "distance_px": float(distance_px)}
        match_distance = _match_distance_px(connection)
        distance_score = 1.0 - (distance_px / match_distance) if match_distance > 0 else 0.0
        if distance_score < confidence_threshold:
            split_report.append({**report_base, "status": "low_confidence", "split_index": split_index})
            continue

        polyline = edge.get("polyline", [])
        if split_index is None or split_index <= 0 or split_index >= len(polyline) - 1:
            split_report.append({**report_base, "status": "skipped_edge_too_short", "split_index": split_index})
            continue

        inline_node_id = f"inline::{connector_id}"
        original_edge_id = str(edge.get("id", ""))
        upstream_polyline = copy.deepcopy(polyline[:split_index])
        downstream_polyline = copy.deepcopy(polyline[split_index:])
        if not upstream_polyline or not downstream_polyline:
            split_report.append({**report_base, "status": "skipped_edge_too_short", "split_index": split_index})
            continue

        inline_node = {
            "id": inline_node_id,
            "kind": "inline",
            "type": connector_class,
            "position": {"x": center_xy[0], "y": center_xy[1]},
            "bbox": bbox,
            "inline_element_id": connector_id,
            "source_edge_id": original_edge_id,
            "review_state": "provisional",
        }
        if inline_node_id not in {node["id"] for node in split_nodes}:
            split_nodes.append(inline_node)

        upstream_id = f"{original_edge_id}::split::{connector_id}::upstream"
        downstream_id = f"{original_edge_id}::split::{connector_id}::downstream"
        upstream_edge = _make_split_edge(
            edge,
            edge_id=upstream_id,
            source=str(edge.get("source", "")),
            target=inline_node_id,
            polyline=upstream_polyline,
            split_position="upstream",
            inline_node_id=inline_node_id,
        )
        downstream_edge = _make_split_edge(
            edge,
            edge_id=downstream_id,
            source=inline_node_id,
            target=str(edge.get("target", "")),
            polyline=downstream_polyline,
            split_position="downstream",
            inline_node_id=inline_node_id,
        )
        active_edges.pop(original_edge_id, None)
        active_edges[upstream_id] = upstream_edge
        active_edges[downstream_id] = downstream_edge
        split_report.append({**report_base, "status": "split", "split_index": split_index})

    status_counts: dict[str, int] = {}
    for item in split_report:
        status_counts[str(item["status"])] = status_counts.get(str(item["status"]), 0) + 1

    return {
        "edges_payload": {
            "image_id": "",
            "pass_type": "sheet",
            "edges": list(active_edges.values()),
        },
        "split_nodes": split_nodes,
        "split_report": split_report,
        "summary": {
            "image_id": "",
            "pass_type": "sheet",
            "total_inline_connections": len([item for item in inline_connections if str(item.get("kind", "")) == "inline_element"]),
            "edges_split": status_counts.get("split", 0),
            "nodes_created": len(split_nodes),
            "skipped_already_connected": status_counts.get("skipped_already_connected", 0),
            "skipped_low_confidence": status_counts.get("low_confidence", 0),
            "skipped_edge_too_short": status_counts.get("skipped_edge_too_short", 0),
            "confidence_threshold": float(confidence_threshold),
            "source_artifacts": [
                "stage10c_edge_direction.json",
                "stage12_edge_connections.json",
                "stage4_objects.json",
            ],
        },
    }
