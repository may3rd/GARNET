from __future__ import annotations

import math
from typing import Any


def _normalize_class_name(value: str) -> str:
    lowered = str(value).strip().lower()
    for ch in "-_/":
        lowered = lowered.replace(ch, " ")
    return " ".join(lowered.split())


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
    )


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _node_centroid(cluster: dict[str, Any]) -> tuple[float, float]:
    centroid = cluster.get("centroid", {})
    return (float(centroid.get("x", 0.0)), float(centroid.get("y", 0.0)))


def _terminal_role_for_class(
    class_name: str,
    *,
    equipment_classes: set[str],
    connection_classes: set[str],
    inline_passthrough_classes: set[str],
) -> str | None:
    normalized = _normalize_class_name(class_name)
    if normalized in equipment_classes:
        return "equipment_terminal"
    if normalized in connection_classes:
        return "connection_terminal"
    if normalized in inline_passthrough_classes:
        return "inline_passthrough"
    return None


def _classify_node_terminal(
    cluster: dict[str, Any],
    *,
    object_regions: list[dict[str, Any]],
    equipment_classes: set[str],
    connection_classes: set[str],
    inline_passthrough_classes: set[str],
    match_distance_px: float,
) -> dict[str, Any]:
    node_id = str(cluster.get("id", ""))
    node_kind = str(cluster.get("kind", "unknown"))
    point = _node_centroid(cluster)
    payload = {
        "node_id": node_id,
        "node_kind": node_kind,
        "point": {"x": point[0], "y": point[1]},
        "terminal_role": "unresolved_terminal",
        "terminal_status": "provisional",
        "matched_object_id": None,
        "matched_object_class": None,
        "matched_object_distance_px": None,
    }
    if node_kind == "junction":
        payload["terminal_role"] = "junction_terminal"
        payload["terminal_status"] = "validated"
        return payload

    best_match: dict[str, Any] | None = None
    best_priority: tuple[int, float] | None = None
    for obj in object_regions:
        role = _terminal_role_for_class(
            str(obj.get("class_name", "")),
            equipment_classes=equipment_classes,
            connection_classes=connection_classes,
            inline_passthrough_classes=inline_passthrough_classes,
        )
        if role is None:
            continue
        bbox = obj.get("bbox", {})
        if not bbox:
            continue
        distance_px = _distance(point, _bbox_center(bbox))
        if distance_px > match_distance_px:
            continue
        role_priority = {
            "equipment_terminal": 0,
            "connection_terminal": 1,
            "inline_passthrough": 2,
        }[role]
        candidate_priority = (role_priority, distance_px)
        if best_priority is None or candidate_priority < best_priority:
            best_priority = candidate_priority
            best_match = {
                "terminal_role": role,
                "terminal_status": "validated" if role != "inline_passthrough" else "provisional",
                "matched_object_id": obj.get("id"),
                "matched_object_class": _normalize_class_name(str(obj.get("class_name", ""))),
                "matched_object_distance_px": round(float(distance_px), 3),
            }

    if best_match is None:
        return payload
    payload.update(best_match)
    return payload


def classify_pipe_edge_terminals(
    *,
    edges: list[dict[str, Any]],
    node_clusters: list[dict[str, Any]],
    object_regions: list[dict[str, Any]],
    equipment_terminal_classes: tuple[str, ...],
    connection_terminal_classes: tuple[str, ...],
    inline_passthrough_classes: tuple[str, ...],
    match_distance_px: float,
) -> dict[str, Any]:
    cluster_map = {str(cluster.get("id", "")): cluster for cluster in node_clusters}
    equipment_classes = {_normalize_class_name(name) for name in equipment_terminal_classes}
    connection_classes = {_normalize_class_name(name) for name in connection_terminal_classes}
    inline_classes = {_normalize_class_name(name) for name in inline_passthrough_classes}

    classified_edges: list[dict[str, Any]] = []
    validated_count = 0
    provisional_count = 0
    unresolved_end_count = 0
    inline_passthrough_end_count = 0

    for edge in edges:
        source_cluster = cluster_map.get(str(edge.get("source", "")))
        target_cluster = cluster_map.get(str(edge.get("target", "")))
        if source_cluster is None or target_cluster is None:
            continue

        source_terminal = _classify_node_terminal(
            source_cluster,
            object_regions=object_regions,
            equipment_classes=equipment_classes,
            connection_classes=connection_classes,
            inline_passthrough_classes=inline_classes,
            match_distance_px=match_distance_px,
        )
        destination_terminal = _classify_node_terminal(
            target_cluster,
            object_regions=object_regions,
            equipment_classes=equipment_classes,
            connection_classes=connection_classes,
            inline_passthrough_classes=inline_classes,
            match_distance_px=match_distance_px,
        )

        accepted_roles = {"equipment_terminal", "connection_terminal", "junction_terminal"}
        is_validated = (
            source_terminal["terminal_role"] in accepted_roles
            and destination_terminal["terminal_role"] in accepted_roles
        )
        if is_validated:
            validated_count += 1
        else:
            provisional_count += 1

        unresolved_end_count += int(source_terminal["terminal_role"] == "unresolved_terminal")
        unresolved_end_count += int(destination_terminal["terminal_role"] == "unresolved_terminal")
        inline_passthrough_end_count += int(source_terminal["terminal_role"] == "inline_passthrough")
        inline_passthrough_end_count += int(destination_terminal["terminal_role"] == "inline_passthrough")

        classified_edges.append(
            {
                "edge_id": str(edge.get("id", "")),
                "source_node_id": str(edge.get("source", "")),
                "destination_node_id": str(edge.get("target", "")),
                "source_terminal": source_terminal,
                "destination_terminal": destination_terminal,
                "terminal_status": "validated" if is_validated else "provisional",
                "provisional_due_to_unresolved_terminal": not is_validated,
                "is_internal_junction_edge": (
                    source_terminal["terminal_role"] == "junction_terminal"
                    and destination_terminal["terminal_role"] == "junction_terminal"
                ),
            }
        )

    return {
        "edge_terminals": classified_edges,
        "summary": {
            "edge_count": len(classified_edges),
            "validated_edge_count": validated_count,
            "provisional_edge_count": provisional_count,
            "unresolved_terminal_end_count": unresolved_end_count,
            "inline_passthrough_end_count": inline_passthrough_end_count,
            "match_distance_px": match_distance_px,
        },
    }
