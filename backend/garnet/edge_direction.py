from __future__ import annotations

import copy
import math
import statistics
from typing import Any

from garnet.pipe_terminals import _normalize_class_name


def _bbox_from_object(item: dict[str, Any]) -> dict[str, Any] | None:
    bbox = item.get("bbox", item)
    if isinstance(bbox, dict):
        try:
            x_min = float(bbox["x_min"])
            y_min = float(bbox["y_min"])
            x_max = float(bbox["x_max"])
            y_max = float(bbox["y_max"])
        except (KeyError, TypeError, ValueError):
            return None
        if x_max <= x_min or y_max <= y_min:
            return None
        return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x_min, y_min, x_max, y_max = (float(value) for value in bbox[:4])
        except (TypeError, ValueError):
            return None
        if x_max <= x_min or y_max <= y_min:
            return None
        return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
    return None


def _direction_hint(payload: dict[str, Any]) -> str | None:
    for key in ("arrow_direction", "direction", "points_to", "flow_direction", "orientation"):
        value = payload.get(key)
        if value is None:
            continue
        normalized = _normalize_class_name(str(value))
        if normalized in {"right", "east", "forward"}:
            return "forward"
        if normalized in {"left", "west", "reverse"}:
            return "reverse"
        if normalized in {"down", "south"}:
            return "forward"
        if normalized in {"up", "north"}:
            return "reverse"
    return None


def _arrow_axis(arrow_bbox: dict[str, Any]) -> str:
    bbox = _bbox_from_object(arrow_bbox)
    if bbox is None:
        return "unknown"
    width = bbox["x_max"] - bbox["x_min"]
    height = bbox["y_max"] - bbox["y_min"]
    if height <= 0:
        return "unknown"
    aspect = width / height
    if aspect >= 1.5:
        return "horizontal"
    if aspect < 0.7:
        return "vertical"
    return "unknown"


def compute_arrow_direction(arrow_bbox: dict[str, Any]) -> str:
    """
    Returns 'forward', 'reverse', or 'unknown'.

    With bbox-only detections, arrow head location is not observable. If the
    detection payload carries an orientation hint, use it. Otherwise fall back
    to a conservative canonical assumption: horizontal arrows point right and
    vertical arrows point down.
    """
    bbox = _bbox_from_object(arrow_bbox)
    if bbox is None:
        return "unknown"

    hint = _direction_hint(arrow_bbox)
    if hint is not None:
        return hint

    axis = _arrow_axis(bbox)
    if axis in {"horizontal", "vertical"}:
        return "forward"
    return "unknown"


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0, (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0)


def _edge_points(edge: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in edge.get("polyline", []):
        try:
            points.append((float(point["col"]), float(point["row"])))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _nearest_point_distance(point_xy: tuple[float, float], edge_points: list[tuple[float, float]]) -> float:
    if not edge_points:
        return math.inf
    px, py = point_xy
    return min(math.hypot(px - ex, py - ey) for ex, ey in edge_points)


def _midpoint_distance(point_xy: tuple[float, float], edge_points: list[tuple[float, float]]) -> float:
    if not edge_points:
        return math.inf
    midpoint = edge_points[len(edge_points) // 2]
    return math.hypot(point_xy[0] - midpoint[0], point_xy[1] - midpoint[1])


def _edge_orientation(edge_points: list[tuple[float, float]]) -> str:
    if len(edge_points) < 2:
        return "unknown"
    dx = edge_points[-1][0] - edge_points[0][0]
    dy = edge_points[-1][1] - edge_points[0][1]
    return "horizontal" if abs(dx) >= abs(dy) else "vertical"


def _direction_vector(axis: str, direction: str) -> tuple[float, float] | None:
    if direction not in {"forward", "reverse"}:
        return None
    sign = 1.0 if direction == "forward" else -1.0
    if axis == "horizontal":
        return (sign, 0.0)
    if axis == "vertical":
        return (0.0, sign)
    return None


def _assigned_flow_direction(edge_points: list[tuple[float, float]], arrow_axis: str, arrow_direction: str) -> str | None:
    if len(edge_points) < 2:
        return None
    vector = _direction_vector(arrow_axis, arrow_direction)
    if vector is None:
        return None

    edge_dx = edge_points[-1][0] - edge_points[0][0]
    edge_dy = edge_points[-1][1] - edge_points[0][1]
    edge_length = math.hypot(edge_dx, edge_dy)
    if edge_length == 0:
        return None

    dot = (edge_dx / edge_length) * vector[0] + (edge_dy / edge_length) * vector[1]
    if abs(dot) < 0.5:
        return None
    return "forward" if dot > 0 else "reverse"


def run_edge_direction_stage(
    *,
    edges: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    image_id: str,
    arrow_proximity_px: float = 40.0,
) -> dict[str, Any]:
    directional_edges = copy.deepcopy(edges)
    arrows: list[dict[str, Any]] = []
    for obj in objects:
        if _normalize_class_name(str(obj.get("class_name", ""))) != "arrow":
            continue
        bbox = _bbox_from_object(obj)
        if bbox is None:
            continue
        arrows.append({**obj, "bbox": bbox, "_center": _bbox_center(bbox), "_axis": _arrow_axis(bbox)})

    assigned_arrow_ids: set[str] = set()
    arrow_assignments: list[dict[str, Any]] = []
    confidence_values: list[float] = []

    for edge in directional_edges:
        points = _edge_points(edge)
        edge_orientation = _edge_orientation(points)
        candidates: list[tuple[float, float, dict[str, Any]]] = []
        for arrow in arrows:
            distance = _nearest_point_distance(arrow["_center"], points)
            if distance <= arrow_proximity_px:
                midpoint_distance = _midpoint_distance(arrow["_center"], points)
                candidates.append((distance, midpoint_distance, arrow))

        best_arrow = None
        best_distance = None
        if candidates:
            best_distance, _, best_arrow = min(candidates, key=lambda item: (item[0], item[1]))

        if best_arrow is None or best_distance is None:
            edge["flow_direction"] = None
            edge["flow_direction_confidence"] = 0.0
            edge["assigned_arrow_id"] = None
            continue

        arrow_direction = compute_arrow_direction(best_arrow)
        assigned_direction = _assigned_flow_direction(points, str(best_arrow.get("_axis", "unknown")), arrow_direction)
        confidence = max(0.0, min(1.0, 1.0 - (best_distance / arrow_proximity_px))) if arrow_proximity_px > 0 else 0.0

        edge["flow_direction"] = assigned_direction
        edge["flow_direction_confidence"] = confidence if assigned_direction is not None else 0.0
        edge["assigned_arrow_id"] = best_arrow.get("id") if assigned_direction is not None else None

        if assigned_direction is not None:
            assigned_arrow_ids.add(str(best_arrow.get("id", "")))
            confidence_values.append(confidence)
            arrow_assignments.append(
                {
                    "arrow_id": str(best_arrow.get("id", "")),
                    "edge_id": str(edge.get("id", "")),
                    "distance_px": best_distance,
                    "arrow_direction": arrow_direction,
                    "edge_orientation": edge_orientation,
                    "assigned_flow_direction": assigned_direction,
                }
            )

    direction_counts = {
        "forward": sum(1 for edge in directional_edges if edge.get("flow_direction") == "forward"),
        "reverse": sum(1 for edge in directional_edges if edge.get("flow_direction") == "reverse"),
        "bidirectional": sum(1 for edge in directional_edges if edge.get("flow_direction") == "bidirectional"),
        "none": sum(1 for edge in directional_edges if edge.get("flow_direction") is None),
    }

    return {
        "edges_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "edges": directional_edges,
        },
        "arrow_assignments": arrow_assignments,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "total_edges": len(directional_edges),
            "edges_with_forward_direction": direction_counts["forward"],
            "edges_with_reverse_direction": direction_counts["reverse"],
            "edges_with_bidirectional": direction_counts["bidirectional"],
            "edges_without_direction": direction_counts["none"],
            "arrows_assigned_to_edge": len(assigned_arrow_ids),
            "arrows_unassigned": max(0, len(arrows) - len(assigned_arrow_ids)),
            "arrow_proximity_px": float(arrow_proximity_px),
            "mean_flow_direction_confidence": statistics.fmean(confidence_values) if confidence_values else 0.0,
            "source_artifacts": ["stage4_objects.json", "stage10b_pipe_edges_simplified.json"],
        },
    }
