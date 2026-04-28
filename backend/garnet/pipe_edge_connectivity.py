from __future__ import annotations

import math
from collections import Counter
from typing import Any

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


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


def _cluster_centroid(cluster: dict[str, Any]) -> tuple[float, float] | None:
    centroid = cluster.get("centroid") or {}
    if "x" not in centroid or "y" not in centroid:
        return None
    return (float(centroid["x"]), float(centroid["y"]))


def _edge_junction_geometry(
    edge: dict[str, Any], node_id: str
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    polyline = edge.get("polyline", [])
    if len(polyline) < 2:
        return None
    if str(edge.get("source", "")) == node_id:
        junction = polyline[0]
        for sample in polyline[1:]:
            vector = (float(sample["col"]) - float(junction["col"]), float(sample["row"]) - float(junction["row"]))
            if vector != (0.0, 0.0):
                return (
                    (float(junction["col"]), float(junction["row"])),
                    (float(sample["col"]), float(sample["row"])),
                    vector,
                )
    if str(edge.get("target", "")) == node_id:
        junction = polyline[-1]
        for sample in reversed(polyline[:-1]):
            vector = (float(sample["col"]) - float(junction["col"]), float(sample["row"]) - float(junction["row"]))
            if vector != (0.0, 0.0):
                return (
                    (float(junction["col"]), float(junction["row"])),
                    (float(sample["col"]), float(sample["row"])),
                    vector,
                )
    return None


def _unit_vector(vector: tuple[float, float]) -> tuple[float, float] | None:
    length = math.hypot(vector[0], vector[1])
    if length == 0:
        return None
    return (vector[0] / length, vector[1] / length)


def _junction_pair_metrics(
    *,
    cluster_center: tuple[float, float],
    left_anchor: tuple[float, float],
    left_sample: tuple[float, float],
    left_vector: tuple[float, float],
    right_anchor: tuple[float, float],
    right_sample: tuple[float, float],
    right_vector: tuple[float, float],
) -> dict[str, float] | None:
    left_unit = _unit_vector(left_vector)
    right_unit = _unit_vector(right_vector)
    if left_unit is None or right_unit is None:
        return None
    opposite_error = abs(-1.0 - (left_unit[0] * right_unit[0] + left_unit[1] * right_unit[1]))

    anchor_center = ((left_anchor[0] + right_anchor[0]) / 2.0, (left_anchor[1] + right_anchor[1]) / 2.0)
    if math.hypot(cluster_center[0] - anchor_center[0], cluster_center[1] - anchor_center[1]) <= 6.0:
        center = cluster_center
    else:
        center = anchor_center

    left_side = _unit_vector((left_sample[0] - center[0], left_sample[1] - center[1]))
    right_side = _unit_vector((right_sample[0] - center[0], right_sample[1] - center[1]))
    if left_side is None or right_side is None:
        return None
    side_dot = left_side[0] * right_side[0] + left_side[1] * right_side[1]

    _projection, centerline_error_px = _project_point_to_segment(center, left_sample, right_sample)
    return {
        "opposite_error": opposite_error,
        "centerline_error_px": centerline_error_px,
        "side_dot": side_dot,
        "center_x": center[0],
        "center_y": center[1],
    }


def _junction_connections(edges: list[dict[str, Any]], node_clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = _junction_connection_decisions(edges, node_clusters)
    return result["accepted"]


def _junction_connection_decisions(edges: list[dict[str, Any]], node_clusters: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    max_junction_opposite_error = 0.20
    max_junction_centerline_error_px = 6.0
    max_junction_side_dot = -0.75
    cluster_map = _cluster_by_id(node_clusters)
    incident = _incident_edges(edges)
    connections: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for node_id, cluster in cluster_map.items():
        if str(cluster.get("kind", "")) != "junction":
            continue
        center = _cluster_centroid(cluster)
        if center is None:
            continue
        edge_list = incident.get(node_id, [])
        grouped: dict[str, list[tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]]]] = {
            "horizontal": [],
            "vertical": [],
        }
        for edge in edge_list:
            alignment = _edge_alignment(edge, node_id)
            if alignment in grouped:
                geometry = _edge_junction_geometry(edge, node_id)
                if geometry is None:
                    continue
                anchor_point, sample_point, vector = geometry
                grouped[alignment].append((str(edge.get("id", "")), anchor_point, sample_point, vector))
        for alignment, edge_items in grouped.items():
            best_pair: tuple[str, str, float, float, float] | None = None
            unique_items = []
            seen_ids: set[str] = set()
            for edge_id, anchor_point, sample_point, vector in edge_items:
                if edge_id in seen_ids:
                    continue
                seen_ids.add(edge_id)
                unique_items.append((edge_id, anchor_point, sample_point, vector))
            for idx, (left_id, left_anchor, left_sample, left_vector) in enumerate(unique_items):
                for right_id, right_anchor, right_sample, right_vector in unique_items[idx + 1 :]:
                    metrics = _junction_pair_metrics(
                        cluster_center=center,
                        left_anchor=left_anchor,
                        left_sample=left_sample,
                        left_vector=left_vector,
                        right_anchor=right_anchor,
                        right_sample=right_sample,
                        right_vector=right_vector,
                    )
                    if metrics is None:
                        continue
                    opposite_error = metrics["opposite_error"]
                    centerline_error_px = metrics["centerline_error_px"]
                    side_dot = metrics["side_dot"]
                    base_decision = {
                        "kind": "junction_alignment",
                        "connector_id": node_id,
                        "alignment": alignment,
                        "source_edge_id": left_id,
                        "target_edge_id": right_id,
                        "opposite_error": round(float(opposite_error), 4),
                        "centerline_error_px": round(float(centerline_error_px), 3),
                        "side_dot": round(float(side_dot), 4),
                    }
                    if opposite_error > max_junction_opposite_error:
                        rejected.append({**base_decision, "rejection_reason": "not_opposite_direction"})
                        continue
                    if centerline_error_px > max_junction_centerline_error_px:
                        rejected.append({**base_decision, "rejection_reason": "misses_junction_centerline"})
                        continue
                    if side_dot > max_junction_side_dot:
                        rejected.append({**base_decision, "rejection_reason": "not_opposite_sides"})
                        continue
                    score = (centerline_error_px, opposite_error, side_dot)
                    if best_pair is None or score < (best_pair[3], best_pair[2], best_pair[4]):
                        best_pair = (left_id, right_id, opposite_error, centerline_error_px, side_dot)
            if best_pair is None:
                continue
            connections.append(
                {
                    "kind": "junction_alignment",
                    "connector_id": node_id,
                    "alignment": alignment,
                    "source_edge_id": best_pair[0],
                    "target_edge_id": best_pair[1],
                    "opposite_error": round(float(best_pair[2]), 4),
                    "centerline_error_px": round(float(best_pair[3]), 3),
                    "side_dot": round(float(best_pair[4]), 4),
                }
            )
    accepted_pairs = {
        (
            str(item["connector_id"]),
            str(item["alignment"]),
            tuple(sorted((str(item["source_edge_id"]), str(item["target_edge_id"])))),
        )
        for item in connections
    }
    rejected = [
        item
        for item in rejected
        if (
            str(item["connector_id"]),
            str(item["alignment"]),
            tuple(sorted((str(item["source_edge_id"]), str(item["target_edge_id"])))),
        )
        not in accepted_pairs
    ]
    return {"accepted": connections, "rejected": rejected}


def _edge_endpoints(edge: dict[str, Any]) -> list[tuple[str, str, tuple[float, float], tuple[float, float]]]:
    polyline = edge.get("polyline", [])
    if len(polyline) < 2:
        return []
    start = (float(polyline[0]["col"]), float(polyline[0]["row"]))
    start_next = (float(polyline[1]["col"]), float(polyline[1]["row"]))
    end = (float(polyline[-1]["col"]), float(polyline[-1]["row"]))
    end_prev = (float(polyline[-2]["col"]), float(polyline[-2]["row"]))
    return [
        ("start", str(edge.get("source", "")), start, (start_next[0] - start[0], start_next[1] - start[1])),
        ("end", str(edge.get("target", "")), end, (end_prev[0] - end[0], end_prev[1] - end[1])),
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
    max_gap_px: float = 36.0,
    connection_seed_edge_ids: set[str] | None = None,
    junction_node_ids: set[str] | None = None,
    seeded_max_gap_px: float = 180.0,
    seeded_vertical_max_gap_px: float = 160.0,
    max_opposite_error: float = 0.35,
) -> dict[str, Any]:
    edge_length_map = {str(edge.get("id", "")): float(edge.get("pixel_length", 0.0)) for edge in edges}
    endpoint_candidates: list[tuple[str, str, str, tuple[float, float], tuple[float, float]]] = []
    for edge in edges:
        edge_id = str(edge.get("id", ""))
        for endpoint_name, node_id, point, vector in _edge_endpoints(edge):
            endpoint_candidates.append((edge_id, endpoint_name, node_id, point, vector))
    connections: list[dict[str, Any]] = []
    invalid_shared_junction_fallback_candidate_keys: set[tuple[str, str, str]] = set()
    seen_pairs: set[tuple[str, str]] = set()
    active_seed_edges = set(connection_seed_edge_ids or set())
    blocked_junction_node_ids = set(junction_node_ids or set())
    edge_junction_node_ids = {
        str(edge.get("id", "")): {
            str(edge.get("source", "")),
            str(edge.get("target", "")),
        }
        & blocked_junction_node_ids
        for edge in edges
    }
    expanded = True
    while expanded:
        expanded = False
        best_by_endpoint: dict[tuple[str, str], tuple[float, float, int, float, str, str]] = {}
        for idx, (edge_id, endpoint_name, node_id, point, vector) in enumerate(endpoint_candidates):
            alignment = _vector_alignment(vector)
            for other_edge_id, other_endpoint_name, other_node_id, other_point, other_vector in endpoint_candidates[idx + 1 :]:
                if edge_id == other_edge_id:
                    continue
                shared_junctions = edge_junction_node_ids.get(edge_id, set()) & edge_junction_node_ids.get(other_edge_id, set())
                if shared_junctions:
                    for shared_junction in shared_junctions:
                        sorted_pair = tuple(sorted((edge_id, other_edge_id)))
                        invalid_shared_junction_fallback_candidate_keys.add((shared_junction, sorted_pair[0], sorted_pair[1]))
                    continue
                if _vector_alignment(other_vector) != alignment:
                    continue
                gap_px = math.hypot(other_point[0] - point[0], other_point[1] - point[1])
                if edge_id in active_seed_edges or other_edge_id in active_seed_edges:
                    gap_limit = seeded_vertical_max_gap_px if alignment == "vertical" else seeded_max_gap_px
                else:
                    gap_limit = max_gap_px
                if gap_px > gap_limit:
                    continue
                opposite_error = _opposite_error(vector, other_vector)
                if opposite_error > max_opposite_error:
                    continue
                key_a = (edge_id, endpoint_name)
                key_b = (other_edge_id, other_endpoint_name)
                candidate = (
                    gap_px,
                    opposite_error,
                    0 if other_edge_id in active_seed_edges else 1,
                    -edge_length_map.get(other_edge_id, 0.0),
                    other_edge_id,
                    other_endpoint_name,
                )
                reverse_candidate = (
                    gap_px,
                    opposite_error,
                    0 if edge_id in active_seed_edges else 1,
                    -edge_length_map.get(edge_id, 0.0),
                    edge_id,
                    endpoint_name,
                )
                current_a = best_by_endpoint.get(key_a)
                current_b = best_by_endpoint.get(key_b)
                if current_a is None or candidate[:4] < current_a[:4]:
                    best_by_endpoint[key_a] = candidate
                if current_b is None or reverse_candidate[:4] < current_b[:4]:
                    best_by_endpoint[key_b] = reverse_candidate

        for (edge_id, endpoint_name), (gap_px, opposite_error, _seed_pref, _neg_len, other_edge_id, other_endpoint_name) in best_by_endpoint.items():
            reciprocal = best_by_endpoint.get((other_edge_id, other_endpoint_name))
            if reciprocal is None or reciprocal[4] != edge_id or reciprocal[5] != endpoint_name:
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
                        next(v for e_id, ep_name, _, _, v in endpoint_candidates if e_id == edge_id and ep_name == endpoint_name)
                    ),
                    "source_edge_id": pair[0],
                    "target_edge_id": pair[1],
                    "gap_px": round(float(gap_px), 3),
                    "opposite_error": round(float(opposite_error), 4),
                    "touches_junction_owned_edge": bool(
                        edge_junction_node_ids.get(pair[0], set()) or edge_junction_node_ids.get(pair[1], set())
                    ),
                }
            )
    return {
        "connections": connections,
        "summary": {
            "invalid_shared_junction_fallback_candidate_count": len(invalid_shared_junction_fallback_candidate_keys),
            "junction_touching_continuation_count": len(
                [item for item in connections if item.get("touches_junction_owned_edge")]
            ),
            "junction_touching_gap_continuation_count": len(
                [
                    item
                    for item in connections
                    if item.get("touches_junction_owned_edge") and item.get("kind") == "gap_continuation"
                ]
            ),
            "junction_touching_seeded_continuation_count": len(
                [
                    item
                    for item in connections
                    if item.get("touches_junction_owned_edge") and item.get("kind") == "connection_seeded_continuation"
                ]
            ),
        },
    }


def _polyline_points(edge: dict[str, Any]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for point in edge.get("polyline", []):
        points.append((int(round(float(point["col"]))), int(round(float(point["row"])))))
    return points


def _draw_edge_polyline(image_bgr: Any, edge: dict[str, Any], color: tuple[int, int, int], thickness: int) -> None:
    if cv2 is None:
        return
    points = _polyline_points(edge)
    if len(points) < 2:
        return
    for start, end in zip(points, points[1:]):
        cv2.line(image_bgr, start, end, color, thickness, lineType=cv2.LINE_AA)


def _edge_midpoint(edge: dict[str, Any]) -> tuple[int, int] | None:
    points = _polyline_points(edge)
    if not points:
        return None
    mid = points[len(points) // 2]
    return mid


def render_junction_decision_overlay(
    *,
    image_bgr: Any,
    edges: list[dict[str, Any]],
    edge_connections: list[dict[str, Any]],
    rejected_junction_connections: list[dict[str, Any]],
) -> Any:
    overlay = image_bgr.copy()
    if cv2 is None:
        return overlay
    edge_by_id = {str(edge.get("id", "")): edge for edge in edges}

    for item in rejected_junction_connections:
        for edge_id in (str(item.get("source_edge_id", "")), str(item.get("target_edge_id", ""))):
            edge = edge_by_id.get(edge_id)
            if edge is not None:
                _draw_edge_polyline(overlay, edge, (0, 0, 255), 2)
        source_edge = edge_by_id.get(str(item.get("source_edge_id", "")))
        target_edge = edge_by_id.get(str(item.get("target_edge_id", "")))
        label_point = _edge_midpoint(source_edge or target_edge or {})
        if label_point is not None:
            cv2.putText(
                overlay,
                str(item.get("rejection_reason", "rejected"))[:32],
                label_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    for item in edge_connections:
        if item.get("kind") != "junction_alignment":
            continue
        for edge_id in (str(item.get("source_edge_id", "")), str(item.get("target_edge_id", ""))):
            edge = edge_by_id.get(edge_id)
            if edge is not None:
                _draw_edge_polyline(overlay, edge, (0, 180, 0), 3)
    return overlay


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
    junction_decision_result = _junction_connection_decisions(edges, node_clusters)
    junction_connections = junction_decision_result["accepted"]
    rejected_junction_connections = junction_decision_result["rejected"]
    continuation_result = _continuation_connections(
        edges,
        connection_seed_edge_ids=connection_seed_edge_ids,
        junction_node_ids={
            str(cluster.get("id", ""))
            for cluster in node_clusters
            if str(cluster.get("kind", "")) == "junction"
        },
    )
    continuation_connections = continuation_result["connections"]
    continuation_summary = continuation_result["summary"]
    seen: set[tuple[str, str, str, str]] = set()
    all_connections: list[dict[str, Any]] = []
    for item in inline_connections + junction_connections + continuation_connections:
        pair = tuple(sorted((str(item["source_edge_id"]), str(item["target_edge_id"]))))
        key = (str(item["kind"]), str(item.get("connector_id", "")), str(item.get("alignment", "")), "||".join(pair))
        if key in seen:
            continue
        seen.add(key)
        all_connections.append(item)
    rejection_reason_counts = dict(Counter(str(item.get("rejection_reason", "unknown")) for item in rejected_junction_connections))
    return {
        "connections": all_connections,
        "summary": {
            "edge_connection_count": len(all_connections),
            "accepted_junction_straight_through_count": len(junction_connections),
            "inline_element_connection_count": len(inline_connections),
            "junction_alignment_connection_count": len(junction_connections),
            "rejected_junction_alignment_connection_count": len(rejected_junction_connections),
            "rejected_junction_alignment_reason_counts": rejection_reason_counts,
            "invalid_shared_junction_fallback_candidate_count": continuation_summary[
                "invalid_shared_junction_fallback_candidate_count"
            ],
            "junction_touching_continuation_count": continuation_summary["junction_touching_continuation_count"],
            "junction_touching_gap_continuation_count": continuation_summary["junction_touching_gap_continuation_count"],
            "junction_touching_seeded_continuation_count": continuation_summary[
                "junction_touching_seeded_continuation_count"
            ],
            "gap_continuation_connection_count": len(
                [item for item in continuation_connections if item["kind"] == "gap_continuation"]
            ),
            "connection_seeded_continuation_count": len(
                [item for item in continuation_connections if item["kind"] == "connection_seeded_continuation"]
            ),
            "inline_match_distance_px": inline_match_distance_px,
        },
        "rejected_junction_connections": rejected_junction_connections,
    }
