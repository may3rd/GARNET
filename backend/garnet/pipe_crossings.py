from __future__ import annotations

import math
from collections import deque
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


def _cluster_member_set(cluster: dict[str, Any]) -> set[Point]:
    return {(int(member["row"]), int(member["col"])) for member in cluster.get("members", [])}


def _group_exit_pixels(exit_pixels: set[Point], skeleton: np.ndarray) -> list[list[Point]]:
    remaining = set(exit_pixels)
    groups: list[list[Point]] = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        group = [start]
        while queue:
            row, col = queue.popleft()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                next_row, next_col = neighbor
                if next_row < 0 or next_col < 0:
                    continue
                if next_row >= skeleton.shape[0] or next_col >= skeleton.shape[1]:
                    continue
                if skeleton[next_row, next_col] <= 0:
                    continue
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
                group.append(neighbor)
        groups.append(sorted(group))
    return groups


def _branch_vector(
    cluster_pixels: set[Point],
    entry_pixels: list[Point],
    skeleton: np.ndarray,
    stub_length_px: int,
) -> tuple[float, float]:
    if not entry_pixels:
        return (0.0, 0.0)

    start_row = float(sum(pixel[0] for pixel in entry_pixels) / len(entry_pixels))
    start_col = float(sum(pixel[1] for pixel in entry_pixels) / len(entry_pixels))
    path = list(entry_pixels)
    previous = entry_pixels[0]
    current = entry_pixels[0]

    for _ in range(max(1, stub_length_px)):
        candidates = [pixel for pixel in _neighbors(current, skeleton) if pixel != previous and pixel not in cluster_pixels]
        if not candidates:
            break
        if len(candidates) > 1:
            candidates = sorted(candidates, key=lambda pixel: math.hypot(pixel[0] - start_row, pixel[1] - start_col))
        previous, current = current, candidates[-1]
        path.append(current)

    end_row = float(sum(pixel[0] for pixel in path[-min(len(path), max(2, stub_length_px)) :]) / min(len(path), max(2, stub_length_px)))
    end_col = float(sum(pixel[1] for pixel in path[-min(len(path), max(2, stub_length_px)) :]) / min(len(path), max(2, stub_length_px)))
    return (end_col - start_col, end_row - start_row)


def _angle_deg(vector: tuple[float, float]) -> float:
    return math.degrees(math.atan2(vector[1], vector[0])) % 360.0


def _angle_distance_deg(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _opposite_distance_deg(a: float, b: float) -> float:
    return abs(180.0 - _angle_distance_deg(a, b))


def _extract_branches(
    cluster: dict[str, Any],
    skeleton_mask: np.ndarray,
    stub_length_px: int,
) -> list[dict[str, Any]]:
    cluster_pixels = _cluster_member_set(cluster)
    centroid_x = float(cluster["centroid"]["x"])
    centroid_y = float(cluster["centroid"]["y"])
    exit_pixels: set[Point] = set()
    for pixel in cluster_pixels:
        for neighbor in _neighbors(pixel, skeleton_mask):
            if neighbor not in cluster_pixels:
                exit_pixels.add(neighbor)

    branches: list[dict[str, Any]] = []
    for branch_idx, entry_group in enumerate(_group_exit_pixels(exit_pixels, skeleton_mask)):
        vector = _branch_vector(cluster_pixels, entry_group, skeleton_mask, stub_length_px=stub_length_px)
        entry_row = float(sum(pixel[0] for pixel in entry_group) / len(entry_group))
        entry_col = float(sum(pixel[1] for pixel in entry_group) / len(entry_group))
        angle_deg = _angle_deg((entry_col - centroid_x, entry_row - centroid_y))
        branches.append(
            {
                "branch_id": f"{cluster['id']}::branch_{branch_idx}",
                "entry_pixels": [{"row": int(row), "col": int(col)} for row, col in entry_group],
                "entry_centroid": {"x": float(entry_col), "y": float(entry_row)},
                "angle_deg": round(angle_deg, 3),
                "quality": round(min(1.0, max(0.0, math.hypot(*vector) / max(1.0, float(stub_length_px)))), 3),
            }
        )
    return sorted(branches, key=lambda item: (item["angle_deg"], item["branch_id"]))


def _merge_branches_by_angle(branches: list[dict[str, Any]], merge_tolerance_deg: float) -> list[dict[str, Any]]:
    if not branches:
        return []
    merged: list[dict[str, Any]] = []
    for branch in sorted(branches, key=lambda item: item["angle_deg"]):
        if not merged or _angle_distance_deg(float(branch["angle_deg"]), float(merged[-1]["angle_deg"])) > merge_tolerance_deg:
            merged.append(
                {
                    "branch_id": str(branch["branch_id"]),
                    "entry_pixels": list(branch.get("entry_pixels", [])),
                    "entry_centroid": dict(branch.get("entry_centroid", {})),
                    "angle_deg": float(branch["angle_deg"]),
                    "quality": float(branch.get("quality", 0.0)),
                }
            )
            continue
        prev = merged[-1]
        combined_pixels = prev["entry_pixels"] + list(branch.get("entry_pixels", []))
        avg_x = sum(float(pixel["col"]) for pixel in combined_pixels) / max(1, len(combined_pixels))
        avg_y = sum(float(pixel["row"]) for pixel in combined_pixels) / max(1, len(combined_pixels))
        prev["entry_pixels"] = combined_pixels
        prev["entry_centroid"] = {"x": avg_x, "y": avg_y}
        prev["angle_deg"] = (float(prev["angle_deg"]) + float(branch["angle_deg"])) / 2.0
        prev["quality"] = max(float(prev["quality"]), float(branch.get("quality", 0.0)))
    return merged


def _center_blob_score(sealed_mask: np.ndarray, cluster: dict[str, Any], radius_px: int) -> float:
    center_x = int(round(float(cluster["centroid"]["x"])))
    center_y = int(round(float(cluster["centroid"]["y"])))
    row_min = max(0, center_y - radius_px)
    row_max = min(sealed_mask.shape[0], center_y + radius_px + 1)
    col_min = max(0, center_x - radius_px)
    col_max = min(sealed_mask.shape[1], center_x + radius_px + 1)
    window = sealed_mask[row_min:row_max, col_min:col_max]
    if window.size == 0:
        return 0.0
    return float(np.count_nonzero(window)) / float(window.size)


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
    )


def _collect_stage4_marker_evidence(
    cluster: dict[str, Any],
    topology_markers: list[dict[str, Any]],
    max_distance_px: float,
) -> dict[str, Any]:
    center_x = float(cluster["centroid"]["x"])
    center_y = float(cluster["centroid"]["y"])
    matched: list[dict[str, Any]] = []
    for marker in topology_markers:
        bbox = marker.get("bbox")
        if not isinstance(bbox, dict):
            continue
        marker_x, marker_y = _bbox_center(bbox)
        distance = math.hypot(marker_x - center_x, marker_y - center_y)
        if distance > max_distance_px:
            continue
        matched.append(
            {
                "id": str(marker.get("id", "")),
                "source_object_id": str(marker.get("source_object_id", "")),
                "role": str(marker.get("role", "")),
                "class_name": str(marker.get("class_name", "")),
                "confidence": float(marker.get("confidence", 0.0)),
                "distance_px": round(distance, 3),
            }
        )
    role_counts = {
        role: len([item for item in matched if item["role"] == role])
        for role in {"junction_marker", "connection_marker", "flow_marker"}
    }
    return {
        "supported": bool(matched),
        "matched_object_ids": [item["source_object_id"] for item in matched if item["source_object_id"]],
        "matched_markers": matched,
        "role_counts": role_counts,
    }


def _pair_opposites(branches: list[dict[str, Any]], tolerance_deg: float) -> list[list[str]]:
    unmatched = list(range(len(branches)))
    pairs: list[list[str]] = []
    while len(unmatched) >= 2:
        left_idx = unmatched.pop(0)
        best_pos = None
        best_distance = None
        for pos, right_idx in enumerate(unmatched):
            distance = _opposite_distance_deg(branches[left_idx]["angle_deg"], branches[right_idx]["angle_deg"])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_pos = pos
        if best_pos is None or best_distance is None or best_distance > tolerance_deg:
            continue
        right_idx = unmatched.pop(best_pos)
        pairs.append([branches[left_idx]["branch_id"], branches[right_idx]["branch_id"]])
    return pairs


def _classify_candidate(
    cluster: dict[str, Any],
    branches: list[dict[str, Any]],
    sealed_mask: np.ndarray,
    stage4_marker_evidence: dict[str, Any],
    opposite_angle_tolerance_deg: float,
    blob_radius_px: int,
    blob_threshold: float,
) -> tuple[str, list[list[str]], dict[str, float]]:
    branch_count = len(branches)
    blob_score = _center_blob_score(sealed_mask, cluster, radius_px=blob_radius_px)
    routing_pairs = _pair_opposites(branches, tolerance_deg=opposite_angle_tolerance_deg)
    pair_count = len(routing_pairs)
    crossing_score = 0.0
    junction_score = 0.0
    junction_marker_hits = int(stage4_marker_evidence.get("role_counts", {}).get("junction_marker", 0))
    connection_marker_hits = int(stage4_marker_evidence.get("role_counts", {}).get("connection_marker", 0))
    strong_junction_marker = any(
        item.get("role") == "junction_marker" and float(item.get("confidence", 0.0)) >= 0.85 and float(item.get("distance_px", 999.0)) <= 4.0
        for item in stage4_marker_evidence.get("matched_markers", [])
    )

    if junction_marker_hits > 0:
        junction_score += 0.35
    if connection_marker_hits > 0:
        junction_score += 0.2

    if branch_count == 3:
        junction_score = 0.9
        return "confirmed_junction", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}

    if branch_count == 4:
        if strong_junction_marker:
            junction_score += 0.6
        if pair_count == 2:
            crossing_score += 0.75
        if blob_score >= blob_threshold + 0.15:
            junction_score += 0.75
        elif blob_score >= blob_threshold:
            junction_score += 0.45
        if strong_junction_marker:
            return "confirmed_junction", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}
        if junction_marker_hits > 0 and blob_score >= max(0.6, blob_threshold - 0.1):
            return "confirmed_junction", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}
        if pair_count == 2 and blob_score < blob_threshold + 0.15:
            return "non_connecting_crossing", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}
        if blob_score >= blob_threshold + 0.15 and pair_count < 2:
            return "confirmed_junction", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}
        return "unresolved", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}

    if branch_count > 4:
        crossing_score = 0.35 if pair_count >= 2 else 0.15
        junction_score = 0.75 if blob_score >= blob_threshold + 0.15 else 0.4 if blob_score >= blob_threshold else 0.1
        if junction_score >= 0.75:
            return "confirmed_junction", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}
        return "unresolved", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}

    return "unresolved", routing_pairs, {"crossing": round(crossing_score, 3), "junction": round(junction_score, 3)}


def _draw_overlay(image_bgr: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray:
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    colors = {
        "confirmed_junction": (0, 255, 0),
        "non_connecting_crossing": (255, 255, 0),
        "unresolved": (0, 165, 255),
    }
    for item in candidates:
        color = colors.get(str(item.get("classification", "")), (255, 255, 255))
        center_x = int(round(float(item["centroid"]["x"])))
        center_y = int(round(float(item["centroid"]["y"])))
        cv2.circle(overlay, (center_x, center_y), 4, color, -1)
    return overlay


def run_pipe_crossing_stage(
    *,
    image_bgr: np.ndarray,
    sealed_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    node_clusters: list[dict[str, Any]],
    topology_markers: list[dict[str, Any]] | None = None,
    image_id: str,
    branch_stub_length_px: int = 8,
    branch_merge_angle_tolerance_deg: float = 18.0,
    opposite_angle_tolerance_deg: float = 35.0,
    center_blob_radius_px: int = 4,
    center_blob_threshold: float = 0.5,
    stage4_marker_match_distance_px: float = 24.0,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    confirmed_junctions: list[dict[str, Any]] = []
    non_connecting_crossings: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for cluster in node_clusters:
        if cluster.get("kind") != "junction":
            continue
        raw_branches = _extract_branches(cluster, skeleton_mask, stub_length_px=branch_stub_length_px)
        branches = _merge_branches_by_angle(raw_branches, merge_tolerance_deg=branch_merge_angle_tolerance_deg)
        stage4_marker_evidence = _collect_stage4_marker_evidence(
            cluster,
            topology_markers or [],
            max_distance_px=stage4_marker_match_distance_px,
        )
        classification, routing_pairs, scores = _classify_candidate(
            cluster,
            branches,
            sealed_mask=sealed_mask,
            stage4_marker_evidence=stage4_marker_evidence,
            opposite_angle_tolerance_deg=opposite_angle_tolerance_deg,
            blob_radius_px=center_blob_radius_px,
            blob_threshold=center_blob_threshold,
        )
        candidate = dict(cluster)
        candidate.update(
            {
                "classification": classification,
                "review_state": "accepted" if classification != "unresolved" else "review",
                "branch_count": len(branches),
                "raw_branch_count": len(raw_branches),
                "branches": branches,
                "routing_pairs": routing_pairs,
                "scores": scores,
                "center_blob_score": round(_center_blob_score(sealed_mask, cluster, radius_px=center_blob_radius_px), 4),
                "stage4_object_evidence": stage4_marker_evidence,
            }
        )
        candidates.append(candidate)
        if classification == "confirmed_junction":
            confirmed_junctions.append(candidate)
        elif classification == "non_connecting_crossing":
            non_connecting_crossings.append(candidate)
        else:
            unresolved.append(candidate)

    return {
        "overlay_image": _draw_overlay(image_bgr, candidates),
        "crossings_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "candidates": candidates,
            "confirmed_junctions": confirmed_junctions,
            "non_connecting_crossings": non_connecting_crossings,
            "unresolved_candidates": unresolved,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "candidate_count": len(candidates),
            "confirmed_junction_count": len(confirmed_junctions),
            "non_connecting_crossing_count": len(non_connecting_crossings),
            "unresolved_candidate_count": len(unresolved),
            "branch_stub_length_px": branch_stub_length_px,
            "branch_merge_angle_tolerance_deg": branch_merge_angle_tolerance_deg,
            "opposite_angle_tolerance_deg": opposite_angle_tolerance_deg,
            "center_blob_radius_px": center_blob_radius_px,
            "center_blob_threshold": center_blob_threshold,
            "stage4_marker_match_distance_px": stage4_marker_match_distance_px,
            "source_artifacts": [
                "stage4_topology_markers.json",
                "stage6_pipe_mask_sealed.png",
                "stage7_pipe_skeleton.png",
                "stage9_node_clusters.json",
            ],
        },
    }
