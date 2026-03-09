from __future__ import annotations

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


def _trace_edges(
    skeleton: np.ndarray,
    clusters: list[dict[str, Any]],
    min_edge_length_px: int,
) -> list[dict[str, Any]]:
    node_pixel_map = _build_node_pixel_map(clusters)
    visited_transitions: set[tuple[Point, Point]] = set()
    edges: list[dict[str, Any]] = []

    for cluster in clusters:
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

        candidates = [pixel for pixel in _neighbors(current, skeleton) if pixel != previous]
        if not candidates:
            return None

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
) -> dict[str, Any]:
    edges = _trace_edges(skeleton_mask, node_clusters, min_edge_length_px=min_edge_length_px)
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
