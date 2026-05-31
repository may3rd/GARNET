from __future__ import annotations

import math
from copy import deepcopy
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

STAGE12_LINE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (30, 144, 255),
    (0, 200, 0),
    (255, 128, 0),
    (180, 0, 255),
    (0, 180, 220),
    (220, 80, 80),
    (120, 190, 40),
    (255, 0, 160),
    (90, 90, 255),
    (0, 140, 140),
    (180, 120, 0),
    (130, 0, 180),
    (80, 170, 255),
    (40, 220, 120),
    (210, 90, 170),
    (170, 170, 0),
)


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


def _point_key(point: dict[str, float], *, quantum_px: float = 1.0) -> tuple[int, int]:
    return (int(round(float(point["x"]) / quantum_px)), int(round(float(point["y"]) / quantum_px)))


def _point_near_axis_segment(
    point: dict[str, float],
    start: dict[str, float],
    end: dict[str, float],
    tolerance_px: float,
) -> dict[str, float] | None:
    """Project a point onto a horizontal/vertical segment when it is close enough."""
    px, py = float(point["x"]), float(point["y"])
    x1, y1 = float(start["x"]), float(start["y"])
    x2, y2 = float(end["x"]), float(end["y"])
    tolerance = float(tolerance_px)

    if abs(y1 - y2) <= tolerance and abs(py - y1) <= tolerance:
        min_x, max_x = sorted((x1, x2))
        if min_x - tolerance <= px <= max_x + tolerance:
            return {"x": min(max(px, min_x), max_x), "y": (y1 + y2) / 2.0}

    if abs(x1 - x2) <= tolerance and abs(px - x1) <= tolerance:
        min_y, max_y = sorted((y1, y2))
        if min_y - tolerance <= py <= max_y + tolerance:
            return {"x": (x1 + x2) / 2.0, "y": min(max(py, min_y), max_y)}

    return None


def _is_endpoint(point: dict[str, float], polyline: list[dict[str, float]], tolerance_px: float) -> bool:
    return bool(polyline) and (
        _distance(point, polyline[0]) <= tolerance_px or _distance(point, polyline[-1]) <= tolerance_px
    )


def _dedupe_split_points(points: list[dict[str, float]], tolerance_px: float) -> list[dict[str, float]]:
    deduped: list[dict[str, float]] = []
    for point in points:
        if any(_distance(point, existing) <= tolerance_px for existing in deduped):
            continue
        deduped.append({"x": float(point["x"]), "y": float(point["y"])})
    return deduped


def _polyline_split_locations(
    polyline: list[dict[str, float]],
    split_points: list[dict[str, float]],
    tolerance_px: float,
) -> list[tuple[int, float, dict[str, float]]]:
    locations: list[tuple[int, float, dict[str, float]]] = []
    for split_point in _dedupe_split_points(split_points, tolerance_px):
        if _is_endpoint(split_point, polyline, tolerance_px):
            continue
        vertex_match = None
        for vertex_index, vertex in enumerate(polyline[1:-1], start=1):
            if _distance(split_point, vertex) <= tolerance_px:
                vertex_match = (vertex_index - 1, 1.0, {"x": float(vertex["x"]), "y": float(vertex["y"])})
                break
        if vertex_match is not None:
            locations.append(vertex_match)
            continue
        for index, (start, end) in enumerate(zip(polyline, polyline[1:])):
            projected = _point_near_axis_segment(split_point, start, end, tolerance_px)
            if projected is None:
                continue
            segment_length = _distance(start, end)
            if segment_length <= 0:
                continue
            offset = _distance(start, projected) / segment_length
            if offset <= 0.0 or offset >= 1.0:
                continue
            locations.append((index, offset, projected))
            break
    locations.sort(key=lambda item: (item[0], item[1]))
    return locations


def _split_polyline_at_points(
    polyline: list[dict[str, float]],
    split_points: list[dict[str, float]],
    tolerance_px: float,
    *,
    min_split_edge_length_px: float = 8.0,
) -> list[list[dict[str, float]]]:
    clean_polyline = [{"x": float(point["x"]), "y": float(point["y"])} for point in polyline]
    if len(clean_polyline) < 2:
        return []

    split_by_segment: dict[int, list[dict[str, float]]] = defaultdict(list)
    for segment_index, _offset, projected in _polyline_split_locations(clean_polyline, split_points, tolerance_px):
        split_by_segment[segment_index].append(projected)

    if not split_by_segment:
        return [clean_polyline]

    parts: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = [clean_polyline[0]]
    for segment_index, end in enumerate(clean_polyline[1:]):
        start = clean_polyline[segment_index]
        segment_splits = split_by_segment.get(segment_index, [])
        segment_splits.sort(key=lambda point: _distance(start, point))
        for split_point in segment_splits:
            if _distance(current[-1], split_point) <= 0:
                continue
            current.append(split_point)
            if _polyline_length(current) >= min_split_edge_length_px:
                parts.append(current)
                current = [split_point]
        if _distance(current[-1], end) > 0:
            current.append(end)
    if _polyline_length(current) >= min_split_edge_length_px:
        parts.append(current)
    elif parts:
        parts[-1].extend(current[1:])
    else:
        parts.append(clean_polyline)
    return parts


def _polyline_length(polyline: list[dict[str, float]]) -> float:
    return sum(_distance(start, end) for start, end in zip(polyline, polyline[1:]))


def _direction_between(start: dict[str, float], end: dict[str, float]) -> str:
    dx = float(end["x"]) - float(start["x"])
    dy = float(end["y"]) - float(start["y"])
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx >= 0 else "LEFT"
    return "DOWN" if dy >= 0 else "UP"


def _segments_from_polyline(polyline: list[dict[str, float]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for start, end in zip(polyline, polyline[1:]):
        length = _distance(start, end)
        if length <= 0:
            continue
        segments.append(
            {
                "x1": start["x"],
                "y1": start["y"],
                "x2": end["x"],
                "y2": end["y"],
                "direction": _direction_between(start, end),
                "length_px": length,
            }
        )
    return segments


def _junction_stable_id(point: dict[str, float], terminal_obj_id: Any = None) -> str:
    if terminal_obj_id:
        return f"junction::{terminal_obj_id}"
    return f"junction::xy::{int(round(float(point['x'])))}::{int(round(float(point['y'])))}"


def _junction_override(point: dict[str, float], stable_id: str, *, reason: str) -> dict[str, Any]:
    return {
        "node_type": "tee_junction",
        "position": {"x": float(point["x"]), "y": float(point["y"])},
        "stable_id": stable_id,
        "reason": reason,
    }


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


def _node_override(edge: dict[str, Any], key: str) -> dict[str, Any] | None:
    override = edge.get(key)
    return override if isinstance(override, dict) else None


def _edge_endpoint_override(edge: dict[str, Any], key: str, fallback: dict[str, float]) -> tuple[str | None, dict[str, float] | None, str | None]:
    override = _node_override(edge, key)
    if override is None:
        return None, None, None
    node_type = str(override.get("node_type") or "")
    position = _point_from_xy(override.get("position")) or fallback
    stable_id = str(override.get("stable_id") or "") or None
    return node_type, position, stable_id


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


def _line_number_records(edge: dict[str, Any]) -> list[dict[str, Any]]:
    items = ((edge.get("attachments") or {}).get("line_numbers") or [])
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        item_id = str(item.get("id") or item.get("source_object_id") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        records.append(
            {
                "id": item_id,
                "source_object_id": item.get("source_object_id"),
                "display_text": item.get("text") or item.get("normalized_text") or "",
                "normalized_text": item.get("normalized_text") or item.get("text") or "",
                "review_state": item.get("review_state"),
                "review_source": item.get("review_source"),
            }
        )
    return records


def _reviewed_line_number_ids(edge: dict[str, Any]) -> list[str]:
    items = ((edge.get("attachments") or {}).get("line_numbers") or [])
    result: list[str] = []
    for item in items:
        item_id = str(item.get("id") or item.get("source_object_id") or "")
        if not item_id:
            continue
        if str(item.get("review_state") or "") == "accepted":
            result.append(item_id)
    return result


def _reviewed_line_number_records(edge: dict[str, Any]) -> list[dict[str, Any]]:
    reviewed_ids = set(_reviewed_line_number_ids(edge))
    return [record for record in _line_number_records(edge) if record["id"] in reviewed_ids]


def _line_record_lookup(edges: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for edge in edges:
        for record in _line_number_records(edge):
            records.setdefault(str(record["id"]), record)
    return records


def _records_for_ids(line_ids: list[str], records_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [records_by_id[line_id] for line_id in line_ids if line_id in records_by_id]


def _component_edge_groups(edges: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent: dict[str, str] = {}

    def find(node_id: str) -> str:
        parent.setdefault(node_id, node_id)
        if parent[node_id] != node_id:
            parent[node_id] = find(parent[node_id])
        return parent[node_id]

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source and target:
            union(source, target)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        source = str(edge.get("source") or "")
        root = find(source) if source else str(edge.get("id") or len(grouped))
        grouped[root].append(edge)
    return list(grouped.values())


def _edge_key(edge: dict[str, Any], fallback_index: int) -> str:
    return str(edge.get("id") or edge.get("trace_id") or f"edge_{fallback_index:05d}")


def _edge_vector_away_from_node(edge: dict[str, Any], node_id: str) -> tuple[str, int, float] | None:
    polyline = _dict_polyline(edge.get("polyline"))
    if len(polyline) < 2:
        return None

    if str(edge.get("source") or "") == node_id:
        start = polyline[0]
        toward_pipe = polyline[1]
    elif str(edge.get("target") or "") == node_id:
        start = polyline[-1]
        toward_pipe = polyline[-2]
    else:
        return None

    dx = float(toward_pipe["x"]) - float(start["x"])
    dy = float(toward_pipe["y"]) - float(start["y"])
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    if abs(dx) >= abs(dy):
        return ("horizontal", 1 if dx >= 0 else -1, length)
    return ("vertical", 1 if dy >= 0 else -1, length)


def _is_process_line_boundary_node(node_id: str) -> bool:
    return node_id.startswith(
        (
            "equipment::",
            "page_connection::",
            "utility_connection::",
            "connection::",
        )
    )


def _process_run_edge_groups(edges: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group edges where a reviewed line number should propagate.

    A plain connected component is too broad at a tee: the main run keeps the
    line number, while the branch is normally a different process line. For
    degree-3+ junctions we only join collinear opposite edges; perpendicular
    branches stay in a separate run unless they carry their own line evidence.
    """
    edge_by_key: dict[str, dict[str, Any]] = {}
    parent: dict[str, str] = {}
    incident_by_node: dict[str, list[str]] = defaultdict(list)

    def find(edge_id: str) -> str:
        parent.setdefault(edge_id, edge_id)
        if parent[edge_id] != edge_id:
            parent[edge_id] = find(parent[edge_id])
        return parent[edge_id]

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for index, edge in enumerate(edges):
        edge_id = _edge_key(edge, index)
        edge_by_key[edge_id] = edge
        parent.setdefault(edge_id, edge_id)
        for node_id in (str(edge.get("source") or ""), str(edge.get("target") or "")):
            if node_id:
                incident_by_node[node_id].append(edge_id)

    for node_id, incident_edge_ids in incident_by_node.items():
        if _is_process_line_boundary_node(node_id):
            continue
        unique_edge_ids = list(dict.fromkeys(incident_edge_ids))
        if len(unique_edge_ids) <= 1:
            continue
        if len(unique_edge_ids) <= 2:
            union(unique_edge_ids[0], unique_edge_ids[1])
            continue

        original_groups: dict[str, list[str]] = defaultdict(list)
        for edge_id in unique_edge_ids:
            original_trace_id = str(edge_by_key[edge_id].get("original_trace_id") or "")
            if original_trace_id:
                original_groups[original_trace_id].append(edge_id)
        original_group_used = False
        for original_edge_ids in original_groups.values():
            if len(original_edge_ids) < 2:
                continue
            for edge_id in original_edge_ids[1:]:
                union(original_edge_ids[0], edge_id)
            original_group_used = True
        if original_group_used:
            continue

        by_axis: dict[str, dict[int, list[tuple[str, float]]]] = {
            "horizontal": {-1: [], 1: []},
            "vertical": {-1: [], 1: []},
        }
        for edge_id in unique_edge_ids:
            vector = _edge_vector_away_from_node(edge_by_key[edge_id], node_id)
            if vector is None:
                continue
            axis, sign, length = vector
            by_axis[axis][sign].append((edge_id, length))

        for axis_groups in by_axis.values():
            negative = axis_groups[-1]
            positive = axis_groups[1]
            if not negative or not positive:
                continue
            for neg_edge_id, _neg_length in negative:
                for pos_edge_id, _pos_length in positive:
                    union(neg_edge_id, pos_edge_id)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, edge in enumerate(edges):
        edge_id = _edge_key(edge, index)
        grouped[find(edge_id)].append(edge)
    return list(grouped.values())


def _apply_line_number_component_propagation(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_items: list[dict[str, Any]] = []
    records_by_id = _line_record_lookup(edges)
    for component_index, component_edges in enumerate(_process_run_edge_groups(edges)):
        component_line_ids = sorted({line_id for edge in component_edges for line_id in _reviewed_line_number_ids(edge)})
        component_trace_ids = [str(edge.get("trace_id") or edge.get("id") or "") for edge in component_edges]
        component_edge_ids = [str(edge.get("id") or "") for edge in component_edges]
        for edge in component_edges:
            direct_ids = sorted(set(_reviewed_line_number_ids(edge)))
            edge["direct_line_number_ids"] = direct_ids
            edge["direct_line_numbers"] = _records_for_ids(direct_ids, records_by_id)
            edge["inferred_line_number_ids"] = []
            edge["inferred_line_numbers"] = []
            edge["effective_line_number_ids"] = []
            edge["effective_line_numbers"] = []
            if len(component_line_ids) > 1:
                edge["line_number_assignment_state"] = "conflict"
                edge["effective_line_number_ids"] = component_line_ids
                edge["effective_line_numbers"] = _records_for_ids(component_line_ids, records_by_id)
            elif len(component_line_ids) == 1:
                line_id = component_line_ids[0]
                edge["effective_line_number_ids"] = [line_id]
                edge["effective_line_numbers"] = _records_for_ids([line_id], records_by_id)
                if direct_ids:
                    edge["line_number_assignment_state"] = "direct"
                    if edge.get("terminal_type") not in {"dead_end", "terminal"}:
                        edge["review_state"] = "accepted"
                else:
                    edge["line_number_assignment_state"] = "inferred"
                    if edge.get("terminal_type") not in {"dead_end", "terminal"}:
                        edge["review_state"] = "accepted"
                    edge["inferred_line_number_ids"] = [line_id]
                    edge["inferred_line_numbers"] = _records_for_ids([line_id], records_by_id)
                    review_items.append(
                        _make_review_item(
                            "line_number_inferred",
                            str(edge.get("trace_id") or edge.get("id") or "trace"),
                            "info",
                            "Line number was inferred from reviewed line evidence in the same process run.",
                            edge_id=edge.get("id"),
                            inferred_line_number_ids=[line_id],
                        )
                    )
            else:
                edge["line_number_assignment_state"] = "missing"
        if len(component_line_ids) > 1:
            review_items.append(
                _make_review_item(
                    "line_number_conflict",
                    f"component_{component_index:05d}",
                    "review",
                    "Process run has multiple reviewed line numbers.",
                    candidate_line_number_ids=component_line_ids,
                    component_edge_ids=component_edge_ids,
                    component_trace_ids=component_trace_ids,
                )
            )
        elif not component_line_ids:
            review_items.append(
                _make_review_item(
                    "line_number_missing_after_propagation",
                    f"component_{component_index:05d}",
                    "review",
                    "Process run has no reviewed line number after propagation.",
                    component_edge_ids=component_edge_ids,
                    component_trace_ids=component_trace_ids,
                )
            )
    return review_items


def _merge_attachments(primary: dict[str, Any], duplicate: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(primary.get("attachments") or {})
    duplicate_attachments = duplicate.get("attachments") or {}
    if not isinstance(duplicate_attachments, dict):
        return merged
    for group, items in duplicate_attachments.items():
        if not isinstance(items, list):
            continue
        existing_items = merged.setdefault(group, [])
        if not isinstance(existing_items, list):
            merged[group] = existing_items = []
        seen_ids = {
            str(item.get("id") or item.get("source_object_id") or item)
            for item in existing_items
            if isinstance(item, dict)
        }
        for item in items:
            item_id = str(item.get("id") or item.get("source_object_id") or item) if isinstance(item, dict) else str(item)
            if item_id in seen_ids:
                continue
            existing_items.append(deepcopy(item))
            seen_ids.add(item_id)
    return merged


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


def _endpoints_match(
    a_polyline: list[dict[str, float]],
    b_polyline: list[dict[str, float]],
    tolerance_px: float,
) -> tuple[bool, bool]:
    if len(a_polyline) < 2 or len(b_polyline) < 2:
        return False, False
    same = _distance(a_polyline[0], b_polyline[0]) <= tolerance_px and _distance(a_polyline[-1], b_polyline[-1]) <= tolerance_px
    reversed_match = _distance(a_polyline[0], b_polyline[-1]) <= tolerance_px and _distance(a_polyline[-1], b_polyline[0]) <= tolerance_px
    return same, reversed_match


def _collapse_duplicate_trace_edges(
    edges: list[dict[str, Any]],
    *,
    endpoint_tolerance_px: float = 8.0,
) -> dict[str, Any]:
    collapsed: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []

    for edge in edges:
        polyline = _dict_polyline(edge.get("polyline"))
        line_ids = set(_line_number_ids(edge))
        matched = False
        for existing in collapsed:
            existing_polyline = _dict_polyline(existing.get("polyline"))
            same, reversed_match = _endpoints_match(existing_polyline, polyline, endpoint_tolerance_px)
            if not (same or reversed_match):
                continue
            existing_line_ids = set(_line_number_ids(existing))
            if line_ids and existing_line_ids and line_ids != existing_line_ids:
                review_items.append(
                    _make_review_item(
                        "possible_duplicate_conflicting_line_number",
                        str(edge.get("trace_id") or "trace"),
                        "review",
                        "Trace geometrically duplicates another path but line-number evidence conflicts.",
                        duplicate_of=existing.get("trace_id"),
                    )
                )
                continue
            merged_trace_ids = list(existing.get("merged_trace_ids") or [existing.get("trace_id")])
            edge_trace_id = str(edge.get("trace_id") or "")
            if edge_trace_id and edge_trace_id not in merged_trace_ids:
                merged_trace_ids.append(edge_trace_id)
            existing["merged_trace_ids"] = merged_trace_ids
            existing["attachments"] = _merge_attachments(existing, edge)
            existing.setdefault("duplicate_trace_ids", []).append(edge_trace_id)
            events.append(
                {
                    "event": "duplicate_trace_collapsed",
                    "trace_id": edge_trace_id,
                    "kept_trace_id": existing.get("trace_id"),
                    "reversed": reversed_match,
                }
            )
            review_items.append(
                _make_review_item(
                    "duplicate_trace_collapsed",
                    edge_trace_id,
                    "info",
                    "Trace was merged into an existing physical path during Stage 12 normalization.",
                    duplicate_of=existing.get("trace_id"),
                )
            )
            matched = True
            break
        if not matched:
            normalized = deepcopy(edge)
            normalized["merged_trace_ids"] = [str(normalized.get("trace_id") or "")]
            collapsed.append(normalized)

    return {
        "trace_edges": collapsed,
        "events": events,
        "review_items": review_items,
    }


def _find_containing_edge_split(
    target_point: dict[str, float],
    edge: dict[str, Any],
    *,
    split_tolerance_px: float,
) -> dict[str, float] | None:
    polyline = _dict_polyline(edge.get("polyline"))
    if len(polyline) < 2 or _is_endpoint(target_point, polyline, split_tolerance_px):
        return None
    for start, end in zip(polyline, polyline[1:]):
        projected = _point_near_axis_segment(target_point, start, end, split_tolerance_px)
        if projected is not None and not _is_endpoint(projected, polyline, split_tolerance_px):
            return projected
    return None


def _add_split_point(
    split_points_by_index: dict[int, list[dict[str, Any]]],
    edge_index: int,
    point: dict[str, float],
    stable_id: str,
    reason: str,
) -> None:
    existing = split_points_by_index[edge_index]
    for item in existing:
        if str(item.get("stable_id")) == stable_id or _distance(item["point"], point) <= 1.0:
            return
    existing.append({"point": {"x": float(point["x"]), "y": float(point["y"])}, "stable_id": stable_id, "reason": reason})


def _split_point_metadata_for_part_endpoint(
    point: dict[str, float],
    split_points: list[dict[str, Any]],
    tolerance_px: float,
) -> dict[str, Any] | None:
    for item in split_points:
        if _distance(point, item["point"]) <= tolerance_px:
            return item
    return None


def _split_trace_edge(
    edge: dict[str, Any],
    split_points: list[dict[str, Any]],
    *,
    split_tolerance_px: float,
) -> list[dict[str, Any]]:
    polyline = _dict_polyline(edge.get("polyline"))
    points = [item["point"] for item in split_points]
    parts = _split_polyline_at_points(polyline, points, split_tolerance_px)
    if len(parts) <= 1:
        normalized = deepcopy(edge)
        if parts:
            normalized["polyline"] = parts[0]
            normalized["segments"] = _segments_from_polyline(parts[0])
        return [normalized]

    raw_trace_id = str(edge.get("trace_id") or "trace")
    children: list[dict[str, Any]] = []
    for index, part in enumerate(parts, start=1):
        child = deepcopy(edge)
        child["trace_id"] = f"{raw_trace_id}::part_{index:03d}"
        child["original_trace_id"] = raw_trace_id
        child["polyline"] = part
        child["segments"] = _segments_from_polyline(part)
        child["trace_length_px"] = _polyline_length(part)
        child["port"] = {
            "x": part[0]["x"],
            "y": part[0]["y"],
            "direction": _direction_between(part[0], part[1]) if len(part) > 1 else (edge.get("port") or {}).get("direction"),
        }
        child["terminal_xy"] = [part[-1]["x"], part[-1]["y"]]

        source_split = _split_point_metadata_for_part_endpoint(part[0], split_points, split_tolerance_px)
        terminal_split = _split_point_metadata_for_part_endpoint(part[-1], split_points, split_tolerance_px)
        if source_split is not None:
            child["_source_node_override"] = _junction_override(
                source_split["point"],
                str(source_split["stable_id"]),
                reason=str(source_split.get("reason") or "trace_split"),
            )
        else:
            child.pop("_source_node_override", None)
        if terminal_split is not None:
            child["_terminal_node_override"] = _junction_override(
                terminal_split["point"],
                str(terminal_split["stable_id"]),
                reason=str(terminal_split.get("reason") or "trace_split"),
            )
            child["terminal_type"] = "tee_junction"
            child["terminal_obj_id"] = str(terminal_split["stable_id"]).removeprefix("junction::")
        else:
            child.pop("_terminal_node_override", None)
        children.append(child)
    return children


def normalize_stage11_trace_edges(
    trace_edges: list[dict[str, Any]],
    *,
    split_tolerance_px: float = 10.0,
    merge_tolerance_px: float = 12.0,
) -> dict[str, Any]:
    """Split Stage 11 traces at geometric branch/tee junctions before graph assembly."""
    _ = merge_tolerance_px
    edges = [deepcopy(edge) for edge in trace_edges if isinstance(edge, dict)]
    split_points_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    synthetic_branch_source_junction_ids: set[str] = set()

    for edge_index, edge in enumerate(edges):
        trace_kind = _normalize_type(edge.get("trace_kind"))
        source_point = _point_from_xy(edge.get("port"))
        if trace_kind != "branch" or source_point is None:
            continue
        for host_index, host_edge in enumerate(edges):
            if host_index == edge_index:
                continue
            projected = _find_containing_edge_split(source_point, host_edge, split_tolerance_px=split_tolerance_px)
            if projected is None:
                continue
            stable_id = _junction_stable_id(projected)
            synthetic_branch_source_junction_ids.add(stable_id)
            edges[edge_index]["_source_node_override"] = _junction_override(projected, stable_id, reason="branch_source_on_trace")
            _add_split_point(split_points_by_index, host_index, projected, stable_id, "branch_source_on_trace")
            events.append(
                {
                    "event": "branch_source_merged",
                    "branch_trace_id": edge.get("trace_id"),
                    "host_trace_id": host_edge.get("trace_id"),
                    "junction_id": stable_id,
                    "point": projected,
                }
            )
            break

    for edge_index, edge in enumerate(edges):
        if _normalize_type(edge.get("terminal_type")) != "tee_junction":
            continue
        terminal_point = _point_from_xy(edge.get("terminal_xy"))
        if terminal_point is None:
            continue
        stable_id = _junction_stable_id(terminal_point, edge.get("terminal_obj_id"))
        edges[edge_index]["_terminal_node_override"] = _junction_override(terminal_point, stable_id, reason="tee_terminal")
        for host_index, host_edge in enumerate(edges):
            if host_index == edge_index:
                continue
            projected = _find_containing_edge_split(terminal_point, host_edge, split_tolerance_px=split_tolerance_px)
            if projected is None:
                continue
            _add_split_point(split_points_by_index, host_index, projected, stable_id, "tee_terminal_on_trace")
            events.append(
                {
                    "event": "tee_terminal_split_host",
                    "terminal_trace_id": edge.get("trace_id"),
                    "host_trace_id": host_edge.get("trace_id"),
                    "junction_id": stable_id,
                    "point": projected,
                }
            )

    normalized_edges: list[dict[str, Any]] = []
    split_edge_count = 0
    for edge_index, edge in enumerate(edges):
        children = _split_trace_edge(edge, split_points_by_index.get(edge_index, []), split_tolerance_px=split_tolerance_px)
        if len(children) > 1:
            split_edge_count += 1
        normalized_edges.extend(children)
    collapse_result = _collapse_duplicate_trace_edges(
        normalized_edges,
        endpoint_tolerance_px=min(8.0, merge_tolerance_px),
    )
    normalized_edges = collapse_result["trace_edges"]
    duplicate_events = collapse_result["events"]
    duplicate_review_items = collapse_result["review_items"]
    events.extend(duplicate_events)

    return {
        "trace_edges": normalized_edges,
        "review_items": duplicate_review_items,
        "metadata": {
            "source_trace_edge_count": len(edges),
            "normalized_trace_edge_count": len(normalized_edges),
            "split_edge_count": split_edge_count,
            "duplicate_edge_count": len(duplicate_events),
            "event_count": len(events),
            "events": events,
            "split_tolerance_px": split_tolerance_px,
            "synthetic_branch_source_junction_ids": sorted(synthetic_branch_source_junction_ids),
        },
    }


def _downgrade_degree_two_synthetic_tees(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    synthetic_junction_ids: set[str],
) -> list[str]:
    if not synthetic_junction_ids:
        return []
    degree_by_node: Counter[str] = Counter()
    for edge in edges:
        degree_by_node[str(edge.get("source"))] += 1
        degree_by_node[str(edge.get("target"))] += 1

    downgraded: list[str] = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        if node_id not in synthetic_junction_ids:
            continue
        if node.get("type") != "tee_junction":
            continue
        if degree_by_node[node_id] >= 3:
            continue
        node["type"] = "junction"
        node["kind"] = "junction"
        node.setdefault("normalization_notes", []).append("downgraded_synthetic_branch_source_because_degree_below_3")
        downgraded.append(node_id)
    return downgraded


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
    normalization = normalize_stage11_trace_edges(payload.get("trace_edges", []) or [])

    for raw_edge in normalization["trace_edges"]:
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

        source_override_type, source_override_point, source_override_id = _edge_endpoint_override(
            raw_edge,
            "_source_node_override",
            source_point,
        )
        terminal_override_type, terminal_override_point, terminal_override_id = _edge_endpoint_override(
            raw_edge,
            "_terminal_node_override",
            terminal_point,
        )
        source_type = source_override_type or _source_node_type(raw_edge)
        terminal_type = terminal_override_type or _terminal_node_type(raw_edge)
        resolved_source_point = source_override_point or source_point
        resolved_terminal_point = terminal_override_point or terminal_point
        source_node_id = registry.add(
            node_type=source_type,
            position=resolved_source_point,
            stable_id=source_override_id or _stable_source_node_id(raw_edge, source_type),
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
            position=resolved_terminal_point,
            stable_id=terminal_override_id or _stable_terminal_node_id(raw_edge, terminal_type),
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
            "original_trace_id": raw_edge.get("original_trace_id"),
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
            "line_numbers": _line_number_records(raw_edge),
            "warnings": raw_edge.get("warnings") or [],
        }
        if raw_edge.get("merged_trace_ids"):
            edge_payload["merged_trace_ids"] = raw_edge.get("merged_trace_ids")
        if raw_edge.get("duplicate_trace_ids"):
            edge_payload["duplicate_trace_ids"] = raw_edge.get("duplicate_trace_ids")
        graph_edges.append(edge_payload)
        trace_edge_nodes.append(
            {
                "trace_id": trace_id,
                "edge_id": edge_payload["id"],
                "source_node_id": source_node_id,
                "target_node_id": terminal_node_id,
                "source_xy": resolved_source_point,
                "terminal_xy": resolved_terminal_point,
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
                    terminal_xy=resolved_terminal_point,
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
                    terminal_xy=resolved_terminal_point,
                )
            )

    review_queue.extend(normalization.get("review_items") or [])

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

    downgraded_synthetic_tees = _downgrade_degree_two_synthetic_tees(
        registry.nodes,
        graph_edges,
        set(normalization["metadata"].get("synthetic_branch_source_junction_ids") or []),
    )
    review_queue.extend(_apply_line_number_component_propagation(graph_edges))

    node_type_counts = Counter(str(node.get("type")) for node in registry.nodes)
    edge_terminal_counts = Counter(str(edge.get("terminal_type")) for edge in graph_edges)
    review_counts = Counter(str(item.get("issue_type")) for item in review_queue)
    line_groups: dict[str, list[str]] = defaultdict(list)
    for edge in graph_edges:
        for line_id in edge.get("effective_line_number_ids") or edge.get("line_number_ids") or []:
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
            "normalization": normalization["metadata"],
            "downgraded_synthetic_tee_junction_ids": downgraded_synthetic_tees,
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
        "normalized_trace_edge_count": len(normalization["trace_edges"]),
        "normalization_split_edge_count": normalization["metadata"].get("split_edge_count", 0),
        "normalization_duplicate_edge_count": normalization["metadata"].get("duplicate_edge_count", 0),
        "normalization_event_count": normalization["metadata"].get("event_count", 0),
        "downgraded_synthetic_tee_count": len(downgraded_synthetic_tees),
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
    normalization_payload = {
        "image_id": resolved_image_id,
        **normalization["metadata"],
    }
    normalization_summary = {
        "image_id": resolved_image_id,
        "source_trace_edge_count": normalization["metadata"].get("source_trace_edge_count", 0),
        "normalized_trace_edge_count": normalization["metadata"].get("normalized_trace_edge_count", 0),
        "split_edge_count": normalization["metadata"].get("split_edge_count", 0),
        "duplicate_edge_count": normalization["metadata"].get("duplicate_edge_count", 0),
        "event_count": normalization["metadata"].get("event_count", 0),
        "split_tolerance_px": normalization["metadata"].get("split_tolerance_px"),
        "downgraded_synthetic_tee_count": len(downgraded_synthetic_tees),
        "downgraded_synthetic_tee_junction_ids": downgraded_synthetic_tees,
    }
    return {
        "graph_payload": graph_payload,
        "summary": summary,
        "trace_edge_nodes_payload": trace_edge_nodes_payload,
        "review_queue_payload": review_queue_payload,
        "review_queue_summary": review_queue_summary,
        "normalization_payload": normalization_payload,
        "normalization_summary": normalization_summary,
    }


def _as_int_point(point: dict[str, Any]) -> tuple[int, int]:
    return int(round(_as_float(point.get("x")))), int(round(_as_float(point.get("y"))))


def _edge_line_color_key(edge: dict[str, Any]) -> str:
    line_ids = edge.get("effective_line_number_ids") or edge.get("line_number_ids") or []
    if not isinstance(line_ids, list) or not line_ids:
        return ""
    clean_ids = [str(line_id) for line_id in line_ids if str(line_id)]
    if not clean_ids:
        return ""
    return "+".join(sorted(clean_ids))


def _stage12_line_color_map(graph_payload: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    line_keys = sorted(
        {
            key
            for edge in graph_payload.get("edges", []) or []
            for key in [_edge_line_color_key(edge)]
            if key
        }
    )
    return {line_key: STAGE12_LINE_PALETTE[index % len(STAGE12_LINE_PALETTE)] for index, line_key in enumerate(line_keys)}


def _edge_color(edge: dict[str, Any], line_color_by_key: dict[str, tuple[int, int, int]] | None = None) -> tuple[int, int, int]:
    review_state = str(edge.get("review_state") or "").lower()
    terminal_type = _normalize_type(edge.get("terminal_type"))
    line_key = _edge_line_color_key(edge)
    if line_key:
        color = (line_color_by_key or {}).get(line_key)
        if color is not None:
            return color
        return STAGE12_LINE_PALETTE[sum((index + 1) * ord(char) for index, char in enumerate(line_key)) % len(STAGE12_LINE_PALETTE)]
    if terminal_type == "dead_end":
        return (0, 0, 220)
    if review_state == "accepted":
        return (0, 170, 0)
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
    line_color_by_key = _stage12_line_color_map(graph_payload)

    for edge in graph_payload.get("edges", []) or []:
        polyline = edge.get("polyline") or []
        points = [_point_from_xy(point) for point in polyline]
        points = [point for point in points if point is not None]
        if len(points) < 2:
            continue
        color = _edge_color(edge, line_color_by_key)
        for start, end in zip(points, points[1:]):
            cv2.line(overlay, _as_int_point(start), _as_int_point(end), color, 4, lineType=cv2.LINE_AA)
            cv2.line(overlay, _as_int_point(start), _as_int_point(end), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        mid = points[len(points) // 2]
        label = str(edge.get("trace_id") or edge.get("id") or "")
        line_numbers = edge.get("effective_line_number_ids") or edge.get("line_number_ids") or []
        if line_numbers:
            label = f"{label} line:{','.join(str(item) for item in line_numbers[:2])}"
            if len(line_numbers) > 2:
                label = f"{label},+{len(line_numbers) - 2}"
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

    legend_items = [(f"line {key[:18]}", color) for key, color in list(line_color_by_key.items())[:10]]
    if len(line_color_by_key) > 10:
        legend_items.append((f"+{len(line_color_by_key) - 10} more lines", (80, 80, 80)))
    legend_items.extend(
        [
            ("accepted/no line", (0, 170, 0)),
            ("review/missing line", (0, 80, 255)),
            ("dead_end/no line", (0, 0, 220)),
            ("tee/branch", (255, 0, 255)),
            ("equipment", (255, 180, 0)),
        ]
    )
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
