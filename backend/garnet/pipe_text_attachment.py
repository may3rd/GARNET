from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _edge_draw_color(edge: dict[str, Any]) -> tuple[int, int, int]:
    edge_terminal_info = edge.get("edge_terminals") or {}
    if edge_terminal_info.get("provisional_due_to_unresolved_terminal"):
        return (0, 165, 255)
    return (0, 0, 255)


def _edge_bbox(edge: dict[str, Any]) -> tuple[int, int, int, int] | None:
    polyline = edge.get("polyline", [])
    if len(polyline) < 2:
        return None
    xs = [int(point["col"]) for point in polyline]
    ys = [int(point["row"]) for point in polyline]
    return min(xs), min(ys), max(xs), max(ys)


def _filter_border_like_edges(edges: list[dict[str, Any]], image_shape: tuple[int, ...]) -> dict[str, Any]:
    height, width = image_shape[:2]
    kept: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    margin_x = max(20, int(round(width * 0.02)))
    margin_y = max(20, int(round(height * 0.02)))
    right_panel_x = int(round(width * 0.78))
    bottom_panel_y = int(round(height * 0.82))

    for edge in edges:
        bbox = _edge_bbox(edge)
        if bbox is None:
            kept.append(edge)
            continue
        x_min, y_min, x_max, y_max = bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        long_dim = max(bbox_w, bbox_h)
        short_dim = min(bbox_w, bbox_h)
        is_straight = short_dim <= 4
        orientation = "horizontal" if bbox_w >= bbox_h else "vertical"
        reasons: list[str] = []

        if is_straight:
            if orientation == "horizontal" and long_dim >= int(round(width * 0.25)):
                if y_min <= margin_y or y_max >= height - margin_y:
                    reasons.append("page_border")
                elif x_min >= right_panel_x and y_min <= int(round(height * 0.35)):
                    reasons.append("right_panel_border")
                elif y_min >= bottom_panel_y:
                    reasons.append("bottom_title_block_border")
            if orientation == "vertical" and long_dim >= int(round(height * 0.25)):
                if x_min <= margin_x or x_max >= width - margin_x:
                    reasons.append("page_border")
                elif x_min >= right_panel_x:
                    reasons.append("right_panel_border")
                elif y_min >= bottom_panel_y and bbox_h >= int(round(height * 0.08)):
                    reasons.append("bottom_title_block_border")

        if reasons:
            flagged.append(
                {
                    "id": edge["id"],
                    "source": edge.get("source"),
                    "target": edge.get("target"),
                    "pixel_length": edge.get("pixel_length", 0),
                    "bbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    },
                    "orientation": orientation,
                    "reasons": reasons,
                }
            )
            continue
        kept.append(edge)

    return {
        "kept_edges": kept,
        "filtered_edges_payload": {
            "pass_type": "sheet",
            "kept_edge_ids": [str(edge.get("id")) for edge in kept],
            "filtered_edges": flagged,
        },
        "summary": {
            "filtered_edge_count": len(flagged),
            "kept_edge_count": len(kept),
            "page_border_like_edge_count": len([item for item in flagged if "page_border" in item["reasons"]]),
            "panel_border_like_edge_count": len(
                [item for item in flagged if "right_panel_border" in item["reasons"] or "bottom_title_block_border" in item["reasons"]]
            ),
        },
    }


def _center_from_bbox(bbox: dict[str, int]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
    )


def _project_point_to_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[tuple[float, float], float]:
    px, py = point
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return a, math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    proj = (ax + t * abx, ay + t * aby)
    return proj, math.hypot(px - proj[0], py - proj[1])


def _sample_bbox_points(bbox: dict[str, int]) -> list[tuple[float, float]]:
    x_min = float(bbox["x_min"])
    y_min = float(bbox["y_min"])
    x_max = float(bbox["x_max"])
    y_max = float(bbox["y_max"])
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return [
        (x_min, y_min),
        (x_max, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (cx, y_min),
        (cx, y_max),
        (x_min, cy),
        (x_max, cy),
        (cx, cy),
    ]


def _adaptive_attachment_threshold(region: dict[str, Any], base_threshold_px: float, text_class: str) -> float:
    if text_class == "instrument_semantic":
        return float(base_threshold_px) + 5.0
    if text_class != "line_number":
        return float(base_threshold_px)
    bbox = region["bbox"]
    width = max(0.0, float(bbox["x_max"]) - float(bbox["x_min"]))
    height = max(1.0, float(bbox["y_max"]) - float(bbox["y_min"]))
    normalized_text = str(region.get("normalized_text") or region.get("text") or "")
    major_digit_groups = len([m for m in normalized_text.split("-") if any(ch.isdigit() for ch in m) and len("".join(ch for ch in m if ch.isdigit())) >= 3])
    width_bonus = min(70.0, width * 0.45)
    length_bonus = min(35.0, max(0, len(normalized_text) - 18) * 1.8)
    digit_group_bonus = min(20.0, max(0, major_digit_groups - 1) * 10.0)
    slender_bonus = 12.0 if width > height * 4.0 else 0.0
    return min(180.0, float(base_threshold_px) + width_bonus + length_bonus + digit_group_bonus + slender_bonus)


def _nearest_edge(bbox: dict[str, int], edges: list[dict[str, Any]]) -> tuple[str | None, float]:
    best_edge_id = None
    best_dist = float("inf")
    sample_points = _sample_bbox_points(bbox)
    for edge in edges:
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue
        for start, end in zip(polyline, polyline[1:]):
            a = (float(start["col"]), float(start["row"]))
            b = (float(end["col"]), float(end["row"]))
            for point in sample_points:
                _, dist = _project_point_to_segment(point, a, b)
                if dist < best_dist:
                    best_dist = dist
                    best_edge_id = str(edge["id"])
    return best_edge_id, best_dist


def _node_position(node: dict[str, Any]) -> tuple[float, float] | None:
    position = node.get("position") or {}
    if not isinstance(position, dict):
        return None
    if "x" not in position or "y" not in position:
        return None
    return float(position["x"]), float(position["y"])


def _nearest_node(bbox: dict[str, int], nodes: list[dict[str, Any]]) -> tuple[str | None, float]:
    best_node_id = None
    best_dist = float("inf")
    region_center = _center_from_bbox(bbox)
    sample_points = _sample_bbox_points(bbox)
    for node in nodes:
        position = _node_position(node)
        if position is None:
            continue
        distances = [math.hypot(point[0] - position[0], point[1] - position[1]) for point in sample_points]
        distances.append(math.hypot(region_center[0] - position[0], region_center[1] - position[1]))
        dist = min(distances)
        if dist < best_dist:
            best_dist = dist
            best_node_id = str(node["id"])
    return best_node_id, best_dist


def run_node_text_attachment_stage(
    *,
    image_id: str,
    image_bgr: np.ndarray,
    text_regions: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    max_distance_px: float = 80.0,
    text_class: str = "equipment_tag",
) -> dict[str, Any]:
    candidate_regions = [
        item
        for item in text_regions
        if item.get("class") == text_class or item.get("semantic_class") == text_class
    ]
    candidate_regions = [item for item in candidate_regions if str(item.get("text", "")).strip() and item.get("bbox")]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for region in candidate_regions:
        node_id, distance_px = _nearest_node(region["bbox"], nodes)
        payload = {
            "region_id": region.get("id", region.get("source_region_id")),
            "text": region["text"],
            "normalized_text": region.get("normalized_text", ""),
            "semantic_class": region.get("semantic_class", text_class),
            "bbox": region["bbox"],
            "node_id": node_id,
            "distance_px": None if math.isinf(distance_px) else round(float(distance_px), 3),
            "threshold_px": round(float(max_distance_px), 3),
        }
        if node_id is not None and distance_px <= max_distance_px:
            accepted.append(payload)
        else:
            rejected.append(payload)

    overlay = image_bgr.copy()
    for node in nodes:
        position = _node_position(node)
        if position is None:
            continue
        cv2.circle(overlay, (int(round(position[0])), int(round(position[1]))), 4, (0, 200, 0), 1)
    for item in accepted:
        bbox = item["bbox"]
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            (0, 200, 0),
            2,
        )
        label = str(item.get("text", ""))[:32]
        cv2.putText(
            overlay,
            label,
            (int(bbox["x_min"]) + 4, max(12, int(bbox["y_min"]) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

    return {
        "attachments_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "accepted": accepted,
            "rejected": rejected,
            "text_class": text_class,
        },
        "overlay_image": overlay,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "candidate_count": len(candidate_regions),
            "node_candidate_count": len(nodes),
            "accepted_attachment_count": len(accepted),
            "rejected_attachment_count": len(rejected),
            "max_distance_px": max_distance_px,
            "text_class": text_class,
        },
    }


def run_pipe_text_attachment_stage(
    *,
    image_id: str,
    image_bgr: np.ndarray,
    text_regions: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    max_distance_px: float = 80.0,
    text_class: str = "line_number",
) -> dict[str, Any]:
    line_number_regions = [
        item
        for item in text_regions
        if item.get("class") == text_class
        or item.get("semantic_class") == text_class
        or ("source_object_id" in item and text_class == "line_number" and item.get("semantic_class") is None)
    ]
    line_number_regions = [item for item in line_number_regions if str(item.get("text", "")).strip()]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    edge_by_id = {str(edge.get("id", "")): edge for edge in edges}

    for region in line_number_regions:
        edge_id, distance_px = _nearest_edge(region["bbox"], edges)
        threshold_px = _adaptive_attachment_threshold(region, max_distance_px, text_class)
        payload = {
            "region_id": region.get("id", region.get("source_object_id")),
            "text": region["text"],
            "normalized_text": region.get("normalized_text", ""),
            "bbox": region["bbox"],
            "edge_id": edge_id,
            "distance_px": None if math.isinf(distance_px) else round(float(distance_px), 3),
            "threshold_px": round(float(threshold_px), 3),
            "attached_to_provisional_edge": False,
        }
        if edge_id is not None and distance_px <= threshold_px:
            payload["attached_to_provisional_edge"] = bool(
                (edge_by_id.get(str(edge_id), {}).get("edge_terminals") or {}).get("provisional_due_to_unresolved_terminal")
            )
            accepted.append(payload)
        else:
            rejected.append(payload)

    overlay = image_bgr.copy()
    for edge in edges:
        polyline = edge.get("polyline", [])
        for start, end in zip(polyline, polyline[1:]):
            cv2.line(
                overlay,
                (int(start["col"]), int(start["row"])),
                (int(end["col"]), int(end["row"])),
                _edge_draw_color(edge),
                1,
            )
    for item in accepted:
        bbox = item["bbox"]
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            (255, 0, 0),
            2,
        )
        if item["edge_id"] is not None:
            center_x = int(round((bbox["x_min"] + bbox["x_max"]) / 2))
            center_y = int(round((bbox["y_min"] + bbox["y_max"]) / 2))
            cv2.putText(
                overlay,
                str(item["text"])[:32],
                (center_x + 4, center_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return {
        "attachments_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "accepted": accepted,
            "rejected": rejected,
            "text_class": text_class,
        },
        "overlay_image": overlay,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "candidate_count": len(line_number_regions),
            "accepted_attachment_count": len(accepted),
            "rejected_attachment_count": len(rejected),
            "accepted_attachment_on_provisional_edge_count": sum(
                1 for item in accepted if item.get("attached_to_provisional_edge")
            ),
            "max_distance_px": max_distance_px,
            "text_class": text_class,
        },
    }


def render_text_attachment_overlay(
    *,
    image_bgr: np.ndarray,
    edges: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
) -> np.ndarray:
    overlay = image_bgr.copy()
    for edge in edges:
        polyline = edge.get("polyline", [])
        for start, end in zip(polyline, polyline[1:]):
            cv2.line(
                overlay,
                (int(start["col"]), int(start["row"])),
                (int(end["col"]), int(end["row"])),
                _edge_draw_color(edge),
                1,
            )
    for item in attachments:
        bbox = item["bbox"]
        label = str(item.get("text", ""))[:32]
        color = (255, 0, 0)
        if item.get("semantic_class") == "instrument_semantic":
            color = (0, 165, 255)
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            color,
            2,
        )
        center_x = int(round((bbox["x_min"] + bbox["x_max"]) / 2))
        center_y = int(round((bbox["y_min"] + bbox["y_max"]) / 2))
        cv2.putText(
            overlay,
            label,
            (center_x + 4, center_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def render_connection_attachment_overlay(
    *,
    image_bgr: np.ndarray,
    edges: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
    edge_connections: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """
    Render visualization for connection-class objects (page connection, utility connection)
    with anchor points, direction indicators, and pipe connectivity.

    Draws:
    - Bbox outline (magenta)
    - Anchor side label (e.g. "top", "right") at anchor point
    - Anchor dot (blue filled circle) at the anchor xy on the bbox edge
    - Stub line (cyan dashed): anchor → nearest pipe point
    - Pipe polyline (yellow, highlighted) from the connection's edge
    - Direction arrow on the pipe polyline showing travel direction
    - Class name label + det_id text near the bbox
    """
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    # ── Build edge adjacency for highlight propagation ──────────────────
    adjacency: dict[str, set[str]] = {}
    for item in edge_connections or []:
        src = str(item.get("source_edge_id", ""))
        tgt = str(item.get("target_edge_id", ""))
        if src and tgt and src != tgt:
            adjacency.setdefault(src, set()).add(tgt)
            adjacency.setdefault(tgt, set()).add(src)

    attached_edge_ids = {str(item.get("edge_id", "")) for item in attachments if item.get("edge_id")}
    highlighted: set[str] = set()
    for eid in attached_edge_ids:
        if not eid:
            continue
        stack = [eid]
        while stack:
            cur = stack.pop()
            if cur in highlighted:
                continue
            highlighted.add(cur)
            stack.extend(sorted(adjacency.get(cur, set()) - highlighted))

    # ── Draw pipe polylines ─────────────────────────────────────────────
    for edge in edges:
        eid = str(edge.get("id", ""))
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue
        color = (80, 80, 80)
        thickness = 1
        if eid in highlighted:
            color = (0, 255, 255)  # yellow highlight
            thickness = 2
        for start, end in zip(polyline, polyline[1:]):
            cv2.line(
                overlay,
                (int(start["col"]), int(start["row"])),
                (int(end["col"]), int(end["row"])),
                color,
                thickness,
            )

    # ── Draw direction arrows on highlighted edges ──────────────────────
    # (Drawn FIRST so they are beneath stub lines and anchor dots
    # when arrows cross the anchor point region.)
    ANNOTATION_COLOR = (0, 255, 255)
    ARROW_TOLERANCE_PX = 25.0

    for edge in edges:
        eid = str(edge.get("id", ""))
        if eid not in highlighted:
            continue
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue

        total_len = 0.0
        for a, b in zip(polyline, polyline[1:]):
            total_len += math.hypot(float(b["col"]) - float(a["col"]), float(b["row"]) - float(a["row"]))
        if total_len < ARROW_TOLERANCE_PX:
            continue

        mid_idx = len(polyline) // 2
        p_mid = polyline[mid_idx]
        p_next = polyline[min(mid_idx + 1, len(polyline) - 1)]
        dx = float(p_next["col"]) - float(p_mid["col"])
        dy = float(p_next["row"]) - float(p_mid["row"])
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        ux, uy = dx / length, dy / length
        tip_x = int(round(float(p_mid["col"]) + ux * 15))
        tip_y = int(round(float(p_mid["row"]) + uy * 15))
        tail_x = int(round(float(p_mid["col"]) - ux * 15))
        tail_y = int(round(float(p_mid["row"]) - uy * 15))
        cv2.arrowedLine(overlay, (tail_x, tail_y), (tip_x, tip_y), ANNOTATION_COLOR, 2, tipLength=0.3)

    # ── Draw stub lines (dashed cyan) ───────────────────────────────────
    # (Drawn SECOND so they appear above arrows but below anchor dots
    # when the stub line crosses through the anchor region.)
    STUB_COLOR = (255, 255, 0)    # cyan (BGR)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_DETAIL = 0.30
    TEXT_COLOR = (255, 255, 255)  # white

    for item in attachments:
        bbox = item.get("bbox", [])
        if len(bbox) != 4:
            continue
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Class name + det_id label
        class_name = str(item.get("class_name", ""))[:28]
        det_id = str(item.get("det_id", ""))
        label_text = f"{class_name} [{det_id}]"
        cv2.putText(overlay, label_text, (x_min + 4, max(14, y_min - 6)),
                    FONT, FONT_SCALE_DETAIL, TEXT_COLOR, 1, cv2.LINE_AA)

        anchor_xy = item.get("anchor_xy")
        nearest_xy = item.get("nearest_point_xy")
        if anchor_xy and len(anchor_xy) == 2:
            ax, ay = int(round(float(anchor_xy[0]))), int(round(float(anchor_xy[1])))
            if nearest_xy and len(nearest_xy) == 2:
                nx, ny = int(round(float(nearest_xy[0]))), int(round(float(nearest_xy[1])))
                _draw_dashed_line(overlay, (ax, ay), (nx, ny), STUB_COLOR, 2, gap=6)
                cv2.circle(overlay, (nx, ny), 3, STUB_COLOR, -1)

    # ── Draw each attachment anchor dot + bbox ───────────────────────────
    # (Anchor dots drawn LAST so they render on top of arrows + stubs
    # when arrows cross through the anchor point region.)
    ANCHOR_DOT_RADIUS = 7
    ANCHOR_COLOR = (255, 100, 0)  # blue (BGR)
    BBOX_COLOR = (255, 0, 255)    # magenta (BGR)
    LABEL_COLOR = (200, 220, 255) # light blue
    FONT_SCALE_LABEL = 0.42
    THICKNESS_LABEL = 1

    for item in attachments:
        bbox = item.get("bbox", [])
        if len(bbox) != 4:
            continue
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Bbox outline
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), BBOX_COLOR, 2)

        anchor_name = str(item.get("anchor_name", ""))
        anchor_xy = item.get("anchor_xy")

        if anchor_xy and len(anchor_xy) == 2:
            ax, ay = int(round(float(anchor_xy[0]))), int(round(float(anchor_xy[1])))

            # Anchor dot with white ring
            cv2.circle(overlay, (ax, ay), ANCHOR_DOT_RADIUS + 2, TEXT_COLOR, 1)
            cv2.circle(overlay, (ax, ay), ANCHOR_DOT_RADIUS, ANCHOR_COLOR, -1)

            # Anchor side name label
            label_offset_x = 14 if anchor_name in ("left", "right") else -50
            label_offset_y = -10 if anchor_name in ("top", "bottom") else 0
            label_x = ax + label_offset_x
            label_y = ay + label_offset_y
            cv2.rectangle(
                overlay,
                (label_x - 2, label_y - 12),
                (label_x + 50, label_y + 2),
                (30, 30, 30),
                -1,
            )
            cv2.putText(overlay, anchor_name, (label_x, label_y),
                        FONT, FONT_SCALE_LABEL, LABEL_COLOR, THICKNESS_LABEL, cv2.LINE_AA)

        # Override reason annotation (debug / QA marker)
        override_reason = item.get("anchor_override_reason", "")
        if override_reason:
            override_text = f"[DIR] {override_reason}"
            cv2.putText(overlay, override_text,
                        (x_min + 4, y_max + 14),
                        FONT, FONT_SCALE_DETAIL, (180, 180, 180), 1, cv2.LINE_AA)

    return overlay


def _draw_dashed_line(
    canvas: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    *,
    gap: int = 6,
) -> None:
    """Draw a dashed line segment (line-gap-line...) between p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    udx, udy = dx / length, dy / length
    drawn = 0.0
    while drawn < length:
        seg_len = min(gap, length - drawn)
        start = (int(p1[0] + udx * drawn), int(p1[1] + udy * drawn))
        end = (int(p1[0] + udx * (drawn + seg_len)), int(p1[1] + udy * (drawn + seg_len)))
        cv2.line(canvas, start, end, color, thickness)
        drawn += gap * 2
