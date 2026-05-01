from __future__ import annotations

import copy
import math
import statistics
from typing import Any


Point = tuple[int, int]


def _point_from_payload(point: dict[str, Any]) -> Point:
    return int(point["row"]), int(point["col"])


def _point_to_payload(point: Point) -> dict[str, int]:
    row, col = point
    return {"row": int(row), "col": int(col)}


def _perpendicular_distance(point: Point, start: Point, end: Point) -> float:
    point_row, point_col = point
    start_row, start_col = start
    end_row, end_col = end

    delta_row = end_row - start_row
    delta_col = end_col - start_col
    segment_length = math.hypot(delta_row, delta_col)
    if segment_length == 0:
        return math.hypot(point_row - start_row, point_col - start_col)

    return abs(delta_col * (start_row - point_row) - (start_col - point_col) * delta_row) / segment_length


def _rdp(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) <= 2:
        return points

    start = points[0]
    end = points[-1]
    max_distance = -1.0
    split_index = 0

    for index in range(1, len(points) - 1):
        distance = _perpendicular_distance(points[index], start, end)
        if distance > max_distance:
            max_distance = distance
            split_index = index

    if max_distance <= epsilon:
        return [start, end]

    left = _rdp(points[: split_index + 1], epsilon)
    right = _rdp(points[split_index:], epsilon)
    return left[:-1] + right


def _simplify_polyline(polyline: list[dict[str, Any]], epsilon: float) -> list[dict[str, int]]:
    if len(polyline) <= 2 or epsilon <= 0:
        return [_point_to_payload(_point_from_payload(point)) for point in polyline]

    points = [_point_from_payload(point) for point in polyline]
    return [_point_to_payload(point) for point in _rdp(points, epsilon)]


def run_polyline_simplification_stage(
    *,
    edges: list[dict[str, Any]],
    image_id: str,
    epsilon: float = 2.0,
) -> dict[str, Any]:
    simplified_edges = copy.deepcopy(edges)
    original_counts: list[int] = []
    simplified_counts: list[int] = []
    per_edge_compression: list[float] = []

    for edge in simplified_edges:
        polyline = edge.get("polyline", [])
        original_count = len(polyline)
        simplified_polyline = _simplify_polyline(polyline, epsilon)
        simplified_count = len(simplified_polyline)

        edge["polyline"] = simplified_polyline
        edge["simplified_pixel_length"] = simplified_count

        original_counts.append(original_count)
        simplified_counts.append(simplified_count)
        per_edge_compression.append(simplified_count / original_count if original_count else 1.0)

    total_original = sum(original_counts)
    total_simplified = sum(simplified_counts)
    compression_ratio = total_simplified / total_original if total_original else 1.0

    return {
        "edges_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "edges": simplified_edges,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "epsilon": float(epsilon),
            "edge_count": len(simplified_edges),
            "edges_simplified_count": sum(
                1 for original, simplified in zip(original_counts, simplified_counts) if simplified < original
            ),
            "total_original_point_count": total_original,
            "total_simplified_point_count": total_simplified,
            "compression_ratio": compression_ratio,
            "mean_compression_per_edge": statistics.fmean(per_edge_compression) if per_edge_compression else 1.0,
            "median_compression_per_edge": statistics.median(per_edge_compression) if per_edge_compression else 1.0,
            "source_artifacts": ["stage10_pipe_edges.json"],
        },
    }
