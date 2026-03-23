from __future__ import annotations

import math
from typing import Any


def _normalize_class_name(value: str) -> str:
    lowered = str(value).strip().lower()
    for ch in "-_/":
        lowered = lowered.replace(ch, " ")
    return " ".join(lowered.split())


def _edge_endpoint_vector(edge: dict[str, Any], node_id: str) -> tuple[float, float] | None:
    polyline = edge.get("polyline", [])
    if len(polyline) < 2:
        return None
    if str(edge.get("source", "")) == node_id:
        first, second = polyline[0], polyline[1]
        return (float(second["col"]) - float(first["col"]), float(second["row"]) - float(first["row"]))
    if str(edge.get("target", "")) == node_id:
        last, previous = polyline[-1], polyline[-2]
        return (float(previous["col"]) - float(last["col"]), float(previous["row"]) - float(last["row"]))
    return None


def _edge_alignment(edge: dict[str, Any], node_id: str | None = None) -> str:
    vector = None
    if node_id is not None:
        vector = _edge_endpoint_vector(edge, node_id)
    if vector is None:
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            return "unknown"
        start = polyline[0]
        end = polyline[-1]
        vector = (float(end["col"]) - float(start["col"]), float(end["row"]) - float(start["row"]))
    dx, dy = vector
    if abs(dx) >= abs(dy):
        return "horizontal"
    return "vertical"


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
    )


def _bbox_axis(bbox: dict[str, Any]) -> str:
    width = float(bbox["x_max"]) - float(bbox["x_min"])
    height = float(bbox["y_max"]) - float(bbox["y_min"])
    return "horizontal" if width >= height else "vertical"


def _project_point_to_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[tuple[float, float], float]:
    px, py = point
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return (ax, ay), math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return (proj_x, proj_y), math.hypot(px - proj_x, py - proj_y)


def _closest_edge_point_to_bbox(bbox: dict[str, Any], edge: dict[str, Any]) -> tuple[tuple[float, float] | None, float]:
    polyline = edge.get("polyline", [])
    if len(polyline) < 2:
        return None, float("inf")
    sample = _bbox_center(bbox)
    best_point = None
    best = float("inf")
    for start, end in zip(polyline, polyline[1:]):
        a = (float(start["col"]), float(start["row"]))
        b = (float(end["col"]), float(end["row"]))
        point, distance = _project_point_to_segment(sample, a, b)
        if distance < best:
            best = distance
            best_point = point
    return best_point, best


def _point_side_against_bbox(point: tuple[float, float], bbox: dict[str, Any], *, forced_axis: str | None = None) -> str:
    center_x, center_y = _bbox_center(bbox)
    dx = float(point[0]) - center_x
    dy = float(point[1]) - center_y
    if forced_axis == "horizontal":
        return "left" if dx < 0 else "right"
    if forced_axis == "vertical":
        return "top" if dy < 0 else "bottom"
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "top" if dy < 0 else "bottom"


def _pick_inline_connection_pair(
    candidate_by_side: dict[str, tuple[str, float]],
    *,
    forced_axis: str | None,
) -> tuple[str, str, str] | None:
    if forced_axis == "horizontal":
        first = candidate_by_side.get("left")
        second = candidate_by_side.get("right")
        if first is None or second is None:
            return None
        return first[0], second[0], "horizontal"
    if forced_axis == "vertical":
        first = candidate_by_side.get("top")
        second = candidate_by_side.get("bottom")
        if first is None or second is None:
            return None
        return first[0], second[0], "vertical"

    opposite_pairs = [("left", "right", "horizontal"), ("top", "bottom", "vertical")]
    best_pair: tuple[str, str, str, float] | None = None
    for first_side, second_side, alignment in opposite_pairs:
        first = candidate_by_side.get(first_side)
        second = candidate_by_side.get(second_side)
        if first is None or second is None:
            continue
        score = first[1] + second[1]
        if best_pair is None or score < best_pair[3]:
            best_pair = (first[0], second[0], alignment, score)
    if best_pair is None:
        return None
    return best_pair[0], best_pair[1], best_pair[2]


def _cluster_by_id(node_clusters: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(cluster.get("id", "")): cluster for cluster in node_clusters}


def _incident_edges(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        mapping.setdefault(str(edge.get("source", "")), []).append(edge)
        mapping.setdefault(str(edge.get("target", "")), []).append(edge)
    return mapping


def _junction_connections(edges: list[dict[str, Any]], node_clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cluster_map = _cluster_by_id(node_clusters)
    incident = _incident_edges(edges)
    connections: list[dict[str, Any]] = []
    for node_id, cluster in cluster_map.items():
        if str(cluster.get("kind", "")) != "junction":
            continue
        edge_list = incident.get(node_id, [])
        grouped: dict[str, list[tuple[str, float]]] = {"horizontal": [], "vertical": []}
        for edge in edge_list:
            alignment = _edge_alignment(edge, node_id)
            if alignment in grouped:
                vector = _edge_endpoint_vector(edge, node_id)
                if vector is None:
                    continue
                angle_deg = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
                grouped[alignment].append((str(edge.get("id", "")), angle_deg))
        for alignment, edge_items in grouped.items():
            best_pair: tuple[str, str, float] | None = None
            unique_items = []
            seen_ids: set[str] = set()
            for edge_id, angle_deg in edge_items:
                if edge_id in seen_ids:
                    continue
                seen_ids.add(edge_id)
                unique_items.append((edge_id, angle_deg))
            for idx, (left_id, left_angle) in enumerate(unique_items):
                for right_id, right_angle in unique_items[idx + 1 :]:
                    opposite_error = abs(180.0 - abs((left_angle - right_angle + 180.0) % 360.0 - 180.0))
                    if best_pair is None or opposite_error < best_pair[2]:
                        best_pair = (left_id, right_id, opposite_error)
            if best_pair is None:
                continue
            connections.append(
                {
                    "kind": "junction_alignment",
                    "connector_id": node_id,
                    "alignment": alignment,
                    "source_edge_id": best_pair[0],
                    "target_edge_id": best_pair[1],
                }
            )
    return connections


def _edge_endpoints(edge: dict[str, Any]) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    polyline = edge.get("polyline", [])
    if len(polyline) < 2:
        return []
    start = (float(polyline[0]["col"]), float(polyline[0]["row"]))
    start_next = (float(polyline[1]["col"]), float(polyline[1]["row"]))
    end = (float(polyline[-1]["col"]), float(polyline[-1]["row"]))
    end_prev = (float(polyline[-2]["col"]), float(polyline[-2]["row"]))
    return [
        ("start", start, (start_next[0] - start[0], start_next[1] - start[1])),
        ("end", end, (end_prev[0] - end[0], end_prev[1] - end[1])),
    ]


def _vector_alignment(vector: tuple[float, float]) -> str:
    return "horizontal" if abs(vector[0]) >= abs(vector[1]) else "vertical"


def _opposite_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    la = math.hypot(a[0], a[1])
    lb = math.hypot(b[0], b[1])
    if la == 0 or lb == 0:
        return 999.0
    ax, ay = a[0] / la, a[1] / la
    bx, by = b[0] / lb, b[1] / lb
    dot = ax * bx + ay * by
    return abs(-1.0 - dot)


def _continuation_connections(
    edges: list[dict[str, Any]],
    *,
    max_gap_px: float = 32.0,
    connection_seed_edge_ids: set[str] | None = None,
    seeded_max_gap_px: float = 100.0,
    max_opposite_error: float = 0.35,
) -> list[dict[str, Any]]:
    endpoint_candidates: list[tuple[str, str, tuple[float, float], tuple[float, float]]] = []
    for edge in edges:
        edge_id = str(edge.get("id", ""))
        for endpoint_name, point, vector in _edge_endpoints(edge):
            endpoint_candidates.append((edge_id, endpoint_name, point, vector))
    connections: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    active_seed_edges = set(connection_seed_edge_ids or set())
    expanded = True
    while expanded:
        expanded = False
        best_by_endpoint: dict[tuple[str, str], tuple[float, float, str, str]] = {}
        for idx, (edge_id, endpoint_name, point, vector) in enumerate(endpoint_candidates):
            alignment = _vector_alignment(vector)
            for other_edge_id, other_endpoint_name, other_point, other_vector in endpoint_candidates[idx + 1 :]:
                if edge_id == other_edge_id:
                    continue
                if _vector_alignment(other_vector) != alignment:
                    continue
                gap_px = math.hypot(other_point[0] - point[0], other_point[1] - point[1])
                gap_limit = seeded_max_gap_px if edge_id in active_seed_edges or other_edge_id in active_seed_edges else max_gap_px
                if gap_px > gap_limit:
                    continue
                opposite_error = _opposite_error(vector, other_vector)
                if opposite_error > max_opposite_error:
                    continue
                key_a = (edge_id, endpoint_name)
                key_b = (other_edge_id, other_endpoint_name)
                candidate = (gap_px, opposite_error, other_edge_id, other_endpoint_name)
                reverse_candidate = (gap_px, opposite_error, edge_id, endpoint_name)
                current_a = best_by_endpoint.get(key_a)
                current_b = best_by_endpoint.get(key_b)
                if current_a is None or candidate[:2] < current_a[:2]:
                    best_by_endpoint[key_a] = candidate
                if current_b is None or reverse_candidate[:2] < current_b[:2]:
                    best_by_endpoint[key_b] = reverse_candidate

        for (edge_id, endpoint_name), (gap_px, opposite_error, other_edge_id, other_endpoint_name) in best_by_endpoint.items():
            reciprocal = best_by_endpoint.get((other_edge_id, other_endpoint_name))
            if reciprocal is None or reciprocal[2] != edge_id or reciprocal[3] != endpoint_name:
                continue
            pair = tuple(sorted((edge_id, other_edge_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            seeded = pair[0] in active_seed_edges or pair[1] in active_seed_edges
            if seeded:
                if pair[0] not in active_seed_edges or pair[1] not in active_seed_edges:
                    expanded = True
                active_seed_edges.update(pair)
            connections.append(
                {
                    "kind": "connection_seeded_continuation" if seeded else "gap_continuation",
                    "alignment": _vector_alignment(
                        next(v for e_id, ep_name, _, v in endpoint_candidates if e_id == edge_id and ep_name == endpoint_name)
                    ),
                    "source_edge_id": pair[0],
                    "target_edge_id": pair[1],
                    "gap_px": round(float(gap_px), 3),
                    "opposite_error": round(float(opposite_error), 4),
                }
            )
    return connections


def _inline_connections(
    edges: list[dict[str, Any]],
    object_regions: list[dict[str, Any]],
    *,
    inline_connector_classes: tuple[str, ...],
    inline_match_distance_px: float,
) -> list[dict[str, Any]]:
    allowed = {_normalize_class_name(name) for name in inline_connector_classes}
    connections: list[dict[str, Any]] = []
    for obj in object_regions:
        normalized_class = _normalize_class_name(str(obj.get("class_name", "")))
        if normalized_class not in allowed:
            continue
        bbox = obj.get("bbox", {})
        if not bbox:
            continue
        forced_axis = _bbox_axis(bbox) if normalized_class in {"arrow", "reducer"} else None
        candidate_by_side: dict[str, tuple[str, float]] = {}
        for edge in edges:
            closest_point, distance = _closest_edge_point_to_bbox(bbox, edge)
            if distance <= inline_match_distance_px:
                if closest_point is None:
                    continue
                side = _point_side_against_bbox(closest_point, bbox, forced_axis=forced_axis)
                existing = candidate_by_side.get(side)
                edge_id = str(edge.get("id", ""))
                if existing is None or distance < existing[1]:
                    candidate_by_side[side] = (edge_id, distance)
        if len(candidate_by_side) < 2:
            continue
        picked = _pick_inline_connection_pair(candidate_by_side, forced_axis=forced_axis)
        if picked is None:
            continue
        connections.append(
            {
                "kind": "inline_element",
                "connector_id": str(obj.get("id", "")),
                "connector_class": normalized_class,
                "alignment": picked[2],
                "source_edge_id": picked[0],
                "target_edge_id": picked[1],
            }
        )
    return connections


def build_pipe_edge_connectivity(
    *,
    edges: list[dict[str, Any]],
    node_clusters: list[dict[str, Any]],
    object_regions: list[dict[str, Any]],
    inline_connector_classes: tuple[str, ...],
    inline_match_distance_px: float,
    connection_seed_edge_ids: set[str] | None = None,
) -> dict[str, Any]:
    inline_connections = _inline_connections(
        edges,
        object_regions,
        inline_connector_classes=inline_connector_classes,
        inline_match_distance_px=inline_match_distance_px,
    )
    junction_connections = _junction_connections(edges, node_clusters)
    continuation_connections = _continuation_connections(
        edges,
        connection_seed_edge_ids=connection_seed_edge_ids,
    )
    seen: set[tuple[str, str, str, str]] = set()
    all_connections: list[dict[str, Any]] = []
    for item in inline_connections + junction_connections + continuation_connections:
        pair = tuple(sorted((str(item["source_edge_id"]), str(item["target_edge_id"]))))
        key = (str(item["kind"]), str(item.get("connector_id", "")), str(item.get("alignment", "")), "||".join(pair))
        if key in seen:
            continue
        seen.add(key)
        all_connections.append(item)
    return {
        "connections": all_connections,
        "summary": {
            "edge_connection_count": len(all_connections),
            "inline_element_connection_count": len(inline_connections),
            "junction_alignment_connection_count": len(junction_connections),
            "gap_continuation_connection_count": len(
                [item for item in continuation_connections if item["kind"] == "gap_continuation"]
            ),
            "connection_seeded_continuation_count": len(
                [item for item in continuation_connections if item["kind"] == "connection_seeded_continuation"]
            ),
            "inline_match_distance_px": inline_match_distance_px,
        },
    }
