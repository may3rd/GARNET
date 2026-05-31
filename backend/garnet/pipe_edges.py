from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from garnet.pipe_continuity_helpers import (
    find_near_edges_at_point,
    nearest_point_on_polyline,
    run_post_trace_continuity_check,
    summarize_gaps,
    GAP_THRESHOLD_PX,
    SHORT_EDGE_PX,
)


Point = tuple[int, int]


def _neighbors(pixel: Point, skeleton: np.ndarray) -> list[Point]:
    row, col = pixel
    neighbors: list[Point] = []
    row_max, col_max = skeleton.shape[:2]
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            next_row = row + row_offset
            next_col = col + col_offset
            if next_row < 0 or next_col < 0:
                continue
            if next_row >= row_max or next_col >= col_max:
                continue
            if skeleton[next_row, next_col] > 0:
                neighbors.append((next_row, next_col))
    return neighbors


def _build_node_pixel_map(clusters: list[dict[str, Any]]) -> dict[Point, str]:
    mapping: dict[Point, str] = {}
    for cluster in clusters:
        for member in cluster.get("members", []):
            mapping[(int(member["row"]), int(member["col"]))] = str(cluster["id"])
    return mapping


def _cluster_member_set(cluster: dict[str, Any]) -> set[Point]:
    return {(int(member["row"]), int(member["col"])) for member in cluster.get("members", [])}


def _cluster_exit_angles(cluster: dict[str, Any], skeleton: np.ndarray) -> list[float]:
    centroid_x = float(cluster["centroid"]["x"])
    centroid_y = float(cluster["centroid"]["y"])
    members = _cluster_member_set(cluster)
    angles: list[float] = []
    seen_exits: set[Point] = set()
    for pixel in members:
        for neighbor in _neighbors(pixel, skeleton):
            if neighbor in members or neighbor in seen_exits:
                continue
            seen_exits.add(neighbor)
            dy = float(neighbor[0]) - centroid_y
            dx = float(neighbor[1]) - centroid_x
            angles.append(math.degrees(math.atan2(dy, dx)) % 360.0)
    return sorted(angles)


def _angle_distance_deg(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _angle_groups(angles: list[float], tolerance_deg: float = 20.0) -> list[float]:
    if not angles:
        return []
    groups: list[float] = []
    for angle in sorted(angles):
        if not groups or _angle_distance_deg(angle, groups[-1]) > tolerance_deg:
            groups.append(angle)
            continue
        groups[-1] = (groups[-1] + angle) / 2.0
    if len(groups) > 1 and _angle_distance_deg(groups[0], groups[-1]) <= tolerance_deg:
        merged = (groups[0] + groups[-1]) / 2.0
        groups = [merged] + groups[1:-1]
    return groups


def _passthrough_cluster_ids(clusters: list[dict[str, Any]], skeleton: np.ndarray) -> set[str]:
    passthrough: set[str] = set()
    for cluster in clusters:
        angle_groups = _angle_groups(_cluster_exit_angles(cluster, skeleton))
        if len(angle_groups) != 2:
            continue
        if abs(180.0 - _angle_distance_deg(angle_groups[0], angle_groups[1])) > 20.0:
            continue
        passthrough.add(str(cluster["id"]))
    return passthrough


def _crossing_maps(crossing_resolution: list[dict[str, Any]] | None) -> tuple[dict[str, dict[str, Any]], dict[Point, str]]:
    by_id: dict[str, dict[str, Any]] = {}
    pixel_map: dict[Point, str] = {}
    for item in crossing_resolution or []:
        classification = str(item.get("classification", ""))
        if classification not in {"non_connecting_crossing", "unresolved"}:
            continue
        cluster_id = str(item["id"])
        by_id[cluster_id] = item
        for member in item.get("members", []):
            pixel_map[(int(member["row"]), int(member["col"]))] = cluster_id
    return by_id, pixel_map


def _entry_points(branch: dict[str, Any]) -> list[Point]:
    return [(int(pixel["row"]), int(pixel["col"])) for pixel in branch.get("entry_pixels", [])]


def _paired_branch_id(crossing: dict[str, Any], branch_id: str) -> str | None:
    for left_id, right_id in crossing.get("routing_pairs", []):
        if branch_id == left_id:
            return str(right_id)
        if branch_id == right_id:
            return str(left_id)
    return None


def _nearest_branch_id(crossing: dict[str, Any], pixel: Point) -> str | None:
    best_branch_id = None
    best_distance = None
    for branch in crossing.get("branches", []):
        entry_points = _entry_points(branch)
        if not entry_points:
            continue
        distance = min(math.hypot(pixel[0] - row, pixel[1] - col) for row, col in entry_points)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_branch_id = str(branch["branch_id"])
    return best_branch_id


def _branch_centroid(crossing: dict[str, Any], branch_id: str) -> Point | None:
    for branch in crossing.get("branches", []):
        if str(branch.get("branch_id")) != branch_id:
            continue
        centroid = branch.get("entry_centroid", {})
        return (int(round(float(centroid.get("y", 0.0)))), int(round(float(centroid.get("x", 0.0)))))
    return None


def _candidate_priority(previous: Point, current: Point, candidate: Point) -> tuple[int, float]:
    step_manhattan = abs(candidate[0] - current[0]) + abs(candidate[1] - current[1])
    incoming = (current[0] - previous[0], current[1] - previous[1])
    outgoing = (candidate[0] - current[0], candidate[1] - current[1])
    turn_penalty = abs(incoming[0] - outgoing[0]) + abs(incoming[1] - outgoing[1])
    return step_manhattan, float(turn_penalty)


def _trace_edges(
    skeleton: np.ndarray,
    clusters: list[dict[str, Any]],
    min_edge_length_px: int,
    crossing_resolution: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    crossing_by_id, crossing_pixel_map = _crossing_maps(crossing_resolution)
    passthrough_ids = _passthrough_cluster_ids(clusters, skeleton)
    active_clusters = [
        cluster
        for cluster in clusters
        if str(cluster.get("id")) not in crossing_by_id and str(cluster.get("id")) not in passthrough_ids
    ]
    node_pixel_map = _build_node_pixel_map(active_clusters)
    visited_transitions: set[tuple[Point, Point]] = set()
    edges: list[dict[str, Any]] = []

    for cluster in active_clusters:
        origin_node_id = str(cluster["id"])
        cluster_pixels = [
            (int(member["row"]), int(member["col"]))
            for member in cluster.get("members", [])
        ]
        for start_pixel in cluster_pixels:
            for neighbor in _neighbors(start_pixel, skeleton):
                transition = (start_pixel, neighbor)
                if transition in visited_transitions:
                    continue
                edge = _trace_from_pixel(
                    origin_node_id=origin_node_id,
                    start_pixel=start_pixel,
                    next_pixel=neighbor,
                    skeleton=skeleton,
                    node_pixel_map=node_pixel_map,
                    crossing_by_id=crossing_by_id,
                    crossing_pixel_map=crossing_pixel_map,
                    visited_transitions=visited_transitions,
                    min_edge_length_px=min_edge_length_px,
                )
                if edge is not None:
                    edges.append(edge)
    bridged_edges = _bridge_unmatched_corridors(skeleton, active_clusters, edges, min_edge_length_px=min_edge_length_px)
    if not bridged_edges:
        return edges

    best_by_pair: dict[frozenset[str], dict[str, Any]] = {}
    ordered_pairs: list[frozenset[str]] = []
    for edge in edges + bridged_edges:
        pair = frozenset((str(edge["source"]), str(edge["target"])))
        current = best_by_pair.get(pair)
        if current is None:
            best_by_pair[pair] = edge
            ordered_pairs.append(pair)
            continue
        if int(edge.get("pixel_length", 0)) > int(current.get("pixel_length", 0)):
            best_by_pair[pair] = edge

    return [best_by_pair[pair] for pair in ordered_pairs]


def _component_polyline(points: np.ndarray, orientation: str) -> list[Point]:
    if orientation == "horizontal":
        ordered = sorted(((int(row), int(col)) for row, col in points), key=lambda item: (item[1], item[0]))
    else:
        ordered = sorted(((int(row), int(col)) for row, col in points), key=lambda item: (item[0], item[1]))
    deduped: list[Point] = []
    seen: set[Point] = set()
    for point in ordered:
        if point in seen:
            continue
        seen.add(point)
        deduped.append(point)
    return deduped


def _nearest_cluster_id(point: Point, clusters: list[dict[str, Any]], max_distance_px: float = 20.0) -> str | None:
    best_id = None
    best_distance = None
    row, col = point
    for cluster in clusters:
        cluster_row = float(cluster["centroid"]["y"])
        cluster_col = float(cluster["centroid"]["x"])
        distance = math.hypot(cluster_row - row, cluster_col - col)
        if distance > max_distance_px:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_id = str(cluster["id"])
    return best_id


def _bridge_unmatched_corridors(
    skeleton: np.ndarray,
    clusters: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    min_edge_length_px: int,
) -> list[dict[str, Any]]:
    covered = np.zeros_like(skeleton, dtype=np.uint8)
    for edge in edges:
        for point in edge.get("polyline", []):
            covered[int(point["row"]), int(point["col"])] = 1

    unmatched = ((skeleton > 0) & (covered == 0)).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unmatched, connectivity=8)
    existing_lengths = {
        frozenset((str(edge["source"]), str(edge["target"]))): int(edge.get("pixel_length", 0))
        for edge in edges
    }
    bridged: list[dict[str, Any]] = []

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < max(20, min_edge_length_px * 4):
            continue
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        orientation = None
        if width >= max(40, height * 4):
            orientation = "horizontal"
        elif height >= max(40, width * 4):
            orientation = "vertical"
        if orientation is None:
            continue

        points = np.argwhere(labels == label_idx)
        polyline = _component_polyline(points, orientation=orientation)
        if len(polyline) < max(10, min_edge_length_px * 4):
            continue

        start_pixel = polyline[0]
        end_pixel = polyline[-1]
        source_id = _nearest_cluster_id(start_pixel, clusters)
        target_id = _nearest_cluster_id(end_pixel, clusters)
        if source_id is None or target_id is None or source_id == target_id:
            continue
        pair = frozenset((source_id, target_id))
        if len(polyline) <= existing_lengths.get(pair, 0):
            continue
        edge_id = f"bridge::{source_id}__{target_id}__{len(polyline)}"
        bridged.append(
            {
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "polyline": [{"row": int(row), "col": int(col)} for row, col in polyline],
                "pixel_length": len(polyline),
            }
        )
        existing_lengths[pair] = len(polyline)

    return bridged


def _cast_ray(
    start: Point,
    direction: tuple[float, float],
    skeleton: np.ndarray,
    max_distance: int,
) -> Point | None:
    """Cast a ray from start in direction and return the first skeleton hit."""
    row, col = start
    dy, dx = direction
    for step in range(1, max_distance + 1):
        probe_row = int(round(row + dy * step))
        probe_col = int(round(col + dx * step))
        if probe_row < 0 or probe_col < 0:
            return None
        if probe_row >= skeleton.shape[0] or probe_col >= skeleton.shape[1]:
            return None
        if skeleton[probe_row, probe_col] > 0:
            return (probe_row, probe_col)
    return None


def _ray_polyline(start: Point, end: Point) -> list[dict[str, int]]:
    row0, col0 = start
    row1, col1 = end
    steps = max(abs(row1 - row0), abs(col1 - col0))
    if steps == 0:
        return [{"row": row0, "col": col0}]

    points: list[dict[str, int]] = []
    seen: set[Point] = set()
    for step in range(steps + 1):
        t = step / steps
        point = (int(round(row0 + (row1 - row0) * t)), int(round(col0 + (col1 - col0) * t)))
        if point in seen:
            continue
        seen.add(point)
        points.append({"row": point[0], "col": point[1]})
    return points


def _endpoint_extension_directions(polyline: list[dict[str, Any]], side: str) -> tuple[Point, list[tuple[float, float]]] | None:
    if len(polyline) < 2:
        return None

    endpoint = polyline[0] if side == "start" else polyline[-1]
    adjacent = polyline[1] if side == "start" else polyline[-2]
    ep_row, ep_col = int(endpoint["row"]), int(endpoint["col"])
    adj_row, adj_col = int(adjacent["row"]), int(adjacent["col"])

    # Direction points away from the traced edge endpoint, across the skeleton gap.
    dy = float(ep_row - adj_row)
    dx = float(ep_col - adj_col)
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None

    dy_norm = dy / length
    dx_norm = dx / length
    directions = [(dy_norm, dx_norm)]
    return (ep_row, ep_col), directions


def _extend_endpoints_with_raycasting(
    skeleton: np.ndarray,
    edges: list[dict[str, Any]],
    node_clusters: list[dict[str, Any]],
    *,
    ray_max_distance_px: int = 30,
    min_extension_length_px: int = 10,
) -> list[dict[str, Any]]:
    """
    Recover missed T-junction branches by ray-casting from traced edge dead ends.

    Existing tracing is intentionally left untouched. This post-pass only adds
    short synthetic edge segments when a traced endpoint has no uncovered
    skeleton continuation and a ray finds nearby skeleton across a small gap.
    """
    node_centroids = {
        str(cluster["id"]): (float(cluster["centroid"]["y"]), float(cluster["centroid"]["x"]))
        for cluster in node_clusters
        if cluster.get("kind") == "junction"
    }
    if not node_centroids:
        return []

    covered: set[Point] = set()
    for edge in edges:
        for point in edge.get("polyline", []):
            covered.add((int(point["row"]), int(point["col"])))

    new_edges: list[dict[str, Any]] = []
    seen_pairs: set[frozenset[str]] = set()

    for edge in edges:
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue

        for side in ("start", "end"):
            extension_geometry = _endpoint_extension_directions(polyline, side)
            if extension_geometry is None:
                continue
            endpoint, directions = extension_geometry

            live_neighbors = [neighbor for neighbor in _neighbors(endpoint, skeleton) if neighbor not in covered]
            if live_neighbors:
                continue

            terminal_node_id = str(edge["source"] if side == "start" else edge["target"])
            anchor_node_id = str(edge["target"] if side == "start" else edge["source"])
            for direction_index, direction in enumerate(directions):
                hit = _cast_ray(endpoint, direction, skeleton, max_distance=ray_max_distance_px)
                if hit is None:
                    continue

                hit_distance = math.hypot(hit[0] - endpoint[0], hit[1] - endpoint[1])
                if hit_distance < min_extension_length_px:
                    continue

                nearest_node_id = None
                nearest_distance = None
                for node_id, (node_row, node_col) in node_centroids.items():
                    distance = math.hypot(hit[0] - node_row, hit[1] - node_col)
                    if nearest_distance is None or distance < nearest_distance:
                        nearest_distance = distance
                        nearest_node_id = node_id

                if nearest_node_id is None or nearest_node_id in {terminal_node_id, anchor_node_id}:
                    continue
                if nearest_distance is None or nearest_distance > 20.0:
                    continue

                pair = frozenset((anchor_node_id, nearest_node_id))
                if pair in seen_pairs:
                    continue

                base_polyline = [
                    {"row": int(point["row"]), "col": int(point["col"])}
                    for point in polyline
                ]
                extension_segment = _ray_polyline(endpoint, hit)
                if side == "start":
                    extension_polyline = list(reversed(base_polyline)) + extension_segment[1:]
                else:
                    extension_polyline = base_polyline + extension_segment[1:]

                new_edges.append(
                    {
                        "id": f"extension::{anchor_node_id}__{nearest_node_id}__{len(extension_polyline)}__{direction_index}",
                        "source": anchor_node_id,
                        "target": nearest_node_id,
                        "polyline": extension_polyline,
                        "pixel_length": len(extension_polyline),
                        "extension": True,
                    }
                )
                seen_pairs.add(pair)
                break

    return new_edges


def _trace_from_pixel(
    *,
    origin_node_id: str,
    start_pixel: Point,
    next_pixel: Point,
    skeleton: np.ndarray,
    node_pixel_map: dict[Point, str],
    crossing_by_id: dict[str, dict[str, Any]],
    crossing_pixel_map: dict[Point, str],
    visited_transitions: set[tuple[Point, Point]],
    min_edge_length_px: int,
) -> dict[str, Any] | None:
    polyline: list[Point] = [start_pixel]
    previous = start_pixel
    current = next_pixel

    while True:
        visited_transitions.add((previous, current))
        visited_transitions.add((current, previous))
        polyline.append(current)

        target_node_id = node_pixel_map.get(current)
        if target_node_id is not None and target_node_id != origin_node_id:
            if len(polyline) - 1 < min_edge_length_px:
                return None
            return {
                "id": f"{origin_node_id}__{target_node_id}__{len(polyline)}",
                "source": origin_node_id,
                "target": target_node_id,
                "polyline": [{"row": row, "col": col} for row, col in polyline],
                "pixel_length": len(polyline),
            }

        candidates = [
            pixel
            for pixel in _neighbors(current, skeleton)
            if pixel != previous and (current, pixel) not in visited_transitions
        ]
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda pixel: _candidate_priority(previous, current, pixel))

        crossing_id = crossing_pixel_map.get(current)
        if crossing_id is not None:
            crossing = crossing_by_id[crossing_id]
            incoming_branch_id = _nearest_branch_id(crossing, previous)
            target_branch_id = _paired_branch_id(crossing, incoming_branch_id) if incoming_branch_id is not None else None
            target_centroid = _branch_centroid(crossing, target_branch_id) if target_branch_id is not None else None
            if target_centroid is not None:
                candidates = sorted(
                    candidates,
                    key=lambda pixel: (
                        0 if crossing_pixel_map.get(pixel) == crossing_id or pixel == target_centroid else 1,
                        *_candidate_priority(previous, current, pixel),
                        math.hypot(pixel[0] - target_centroid[0], pixel[1] - target_centroid[1]),
                    ),
                )
            else:
                candidates = sorted(
                    candidates,
                    key=lambda pixel: (
                        0 if crossing_pixel_map.get(pixel) == crossing_id else 1,
                        *_candidate_priority(previous, current, pixel),
                        pixel[0],
                        pixel[1],
                    ),
                )

        if len(candidates) > 1:
            node_at_current = node_pixel_map.get(current)
            if node_at_current is not None and node_at_current == origin_node_id:
                return None

        previous, current = current, candidates[0]


def _draw_overlay(image_bgr: np.ndarray, edges: list[dict[str, Any]]) -> np.ndarray:
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for edge in edges:
        polyline = edge["polyline"]
        for start, end in zip(polyline, polyline[1:]):
            cv2.line(
                overlay,
                (int(start["col"]), int(start["row"])),
                (int(end["col"]), int(end["row"])),
                (255, 255, 0),
                1,
            )
    return overlay


def run_pipe_edge_stage(
    *,
    image_bgr: np.ndarray,
    skeleton_mask: np.ndarray,
    node_clusters: list[dict[str, Any]],
    image_id: str,
    min_edge_length_px: int = 2,
    crossing_resolution: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    edges = _trace_edges(
        skeleton_mask,
        node_clusters,
        min_edge_length_px=min_edge_length_px,
        crossing_resolution=crossing_resolution,
    )

    # ─── Phase 3: ray-casting T-junction recovery ──────────────────────────
    extended_edges = _extend_endpoints_with_raycasting(
        skeleton_mask,
        edges,
        node_clusters,
        ray_max_distance_px=30,
        min_extension_length_px=10,
    )
    existing_pairs = {frozenset((str(edge["source"]), str(edge["target"]))) for edge in edges}
    for extended_edge in extended_edges:
        pair = frozenset((str(extended_edge["source"]), str(extended_edge["target"])))
        if pair in existing_pairs:
            continue
        edges.append(extended_edge)
        existing_pairs.add(pair)

    # ─── Phase 2 continuity check (runs after all edges traced and extended) ─
    continuity_result = run_post_trace_continuity_check(
        edges,
        gap_threshold=GAP_THRESHOLD_PX,
        short_threshold=SHORT_EDGE_PX,
    )
    gap_summary = summarize_gaps(edges, threshold_px=GAP_THRESHOLD_PX)

    return {
        "overlay_image": _draw_overlay(image_bgr, edges),
        "edges_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "edges": edges,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "edge_count": len(edges),
            "min_edge_length_px": min_edge_length_px,
            # Continuity-aware metadata
            "continuity_validated": continuity_result["validated_edges"],
            "continuity_provisional": continuity_result["provisional_edges"],
            "orphan_edges": continuity_result["orphan_edges"],
            "gap_candidate_edges": continuity_result["gap_candidate_edges"],
            "short_provisional_edges": continuity_result["short_provisional_edges"],
            "unique_near_edge_candidates": continuity_result["unique_near_edge_candidates_count"],
            "source_artifacts": [
                "stage7_pipe_skeleton.png",
                "stage9_node_clusters.json",
            ],
        },
        # Extra payloads for downstream stages
        "continuity_result": continuity_result,
        "gap_summary": gap_summary,
    }
