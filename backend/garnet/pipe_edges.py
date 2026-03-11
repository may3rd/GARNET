from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


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
    active_clusters = [
        cluster
        for cluster in clusters
        if str(cluster.get("id")) not in crossing_by_id
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
    return edges


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
            "source_artifacts": [
                "stage7_pipe_skeleton.png",
                "stage9_node_clusters.json",
            ],
        },
    }
