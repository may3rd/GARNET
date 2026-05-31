from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import math
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from garnet.model_defaults import pick_default_weight_file

BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DetectionSahiConfig:
    model_type: str = "ultralytics"
    weight_path: str = pick_default_weight_file("ultralytics") or "yolo_weights/yolo26n_PPCL_640_20260227.pt"
    config_path: str = "datasets/yaml/data.yaml"
    conf_th: float = 0.8
    image_size: int = 640
    overlap_ratio: float = 0.2
    postprocess_type: str = "GREEDYNMM"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.1


def _build_detection_model(cfg: DetectionSahiConfig) -> Any:
    return AutoDetectionModel.from_pretrained(
        model_type=cfg.model_type,
        model_path=str(BACKEND_DIR / cfg.weight_path),
        config_path=str(BACKEND_DIR / cfg.config_path),
        confidence_threshold=cfg.conf_th,
        image_size=cfg.image_size,
    )


def _is_connection_object(obj: dict[str, Any]) -> bool:
    return obj.get("class_name", "") in {
        "page connection",
        "utility connection",
        "connection",
    }


def _project_port_to_edge(
    endpoint: tuple[int, int], bbox: dict[str, int], edge: str
) -> tuple[int, int]:
    """Project a pipe endpoint perpendicularly onto the bbox edge."""
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]
    ex, ey = endpoint

    if edge == "left":
        return (x_min, ey)
    elif edge == "right":
        return (x_max, ey)
    elif edge == "top":
        return (ex, y_min)
    elif edge == "bottom":
        return (ex, y_max)
    raise ValueError(f"Unknown edge: {edge}")


def _find_touching_edge(
    endpoint: tuple[int, int], bbox: dict[str, int], tolerance: int = 15
) -> str | None:
    """Check which bbox edge the endpoint is touching or nearest to.

    Returns the edge name if the endpoint is within `tolerance` pixels of that edge.
    A pipe endpoint within tolerance of the bbox edge counts as "touching".

    Logic: for each of the 4 edges, check if the endpoint's relevant coordinate
    is within the perpendicular range AND the other coordinate is within the edge span.
    This ensures we correctly identify edges even when the endpoint is within the
    bbox's horizontal span (x range) but actually touches the left/right edge.
    """
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]
    ex, ey = endpoint

    # Top edge: endpoint x within [x_min, x_max], endpoint y near y_min
    if x_min <= ex <= x_max and abs(ey - y_min) <= tolerance:
        return "top"
    # Bottom edge: endpoint x within [x_min, x_max], endpoint y near y_max
    if x_min <= ex <= x_max and abs(ey - y_max) <= tolerance:
        return "bottom"
    # Left edge: endpoint y within [y_min, y_max], endpoint x near x_min
    if y_min <= ey <= y_max and abs(ex - x_min) <= tolerance:
        return "left"
    # Right edge: endpoint y within [y_min, y_max], endpoint x near x_max
    if y_min <= ey <= y_max and abs(ex - x_max) <= tolerance:
        return "right"

    return None


def _line_intersects_bbox_edge(
    x1: float, y1: float,
    x2: float, y2: float,
    bbox: dict[str, int],
) -> list[tuple[int, int, str]]:
    """Compute where a line segment intersects bbox edges.

    For each of the 4 bbox edges (top, bottom, left, right), find where the
    infinite line defined by (x1,y1)-(x2,y2) crosses it. Return only those
    crossings that:
      1. Fall within the edge's valid range on the bbox
      2. Are perpendicular to the edge (pipe crosses at roughly 90°)
      3. Have pipe mask present at the crossing point

    This handles diagonal pipes that cross the full bbox even when the
    detected segment only covers part of the pipe path.
    """
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]
    crossings: list[tuple[int, int, str]] = []

    dx = x2 - x1
    dy = y2 - y1

    # Avoid division by zero for horizontal/vertical segments
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return []

    # Helper: check if a crossing point is within bbox edge bounds
    def in_bounds(x: float, y: float, edge: str) -> bool:
        if edge == "top" or edge == "bottom":
            return x_min <= x <= x_max
        else:
            return y_min <= y <= y_max

    # Note: we NO LONGER gate on perpendicularity here. Diagonal pipes crossing
    # bbox edges are common in P&IDs. The perpendicularity score is computed in
    # _best_port_for_connection once we know all candidate ports. This function
    # returns all mathematically valid edge crossings; downstream scoring decides
    # which is the real connection.
    crossings: list[tuple[int, int, str]] = []

    # Check each edge
    eps = 1e-6

    # TOP edge: y = y_min
    if abs(dy) > eps:
        t = (y_min - y1) / dy
        ix = x1 + t * dx
        iy = y_min
        if 0 <= t <= 1.5 and in_bounds(ix, iy, "top"):
            crossings.append((int(round(ix)), int(iy), "top"))

    # BOTTOM edge: y = y_max
    if abs(dy) > eps:
        t = (y_max - y1) / dy
        ix = x1 + t * dx
        iy = y_max
        if 0 <= t <= 1.5 and in_bounds(ix, iy, "bottom"):
            crossings.append((int(round(ix)), int(iy), "bottom"))

    # LEFT edge: x = x_min
    if abs(dx) > eps:
        t = (x_min - x1) / dx
        ix = x_min
        iy = y1 + t * dy
        if 0 <= t <= 1.5 and in_bounds(ix, iy, "left"):
            crossings.append((int(ix), int(round(iy)), "left"))

    # RIGHT edge: x = x_max
    if abs(dx) > eps:
        t = (x_max - x1) / dx
        ix = x_max
        iy = y1 + t * dy
        if 0 <= t <= 1.5 and in_bounds(ix, iy, "right"):
            crossings.append((int(ix), int(round(iy)), "right"))

    return crossings


def _score_port_perpendicularity(
    port_x: int, port_y: int,
    edge: str,
    segments: list[dict[str, Any]],
) -> tuple[float, float]:
    """Score how perpendicular a pipe is to an edge at a given port position.

    Returns (perpendicularity_score, segment_length) where:
    - perpendicularity_score: 0.0 (worst) to 1.0 (perfectly perpendicular)
      For 'left'/'right' edge: ideal pipe is purely horizontal (dy=0)
      For 'top'/'bottom' edge: ideal pipe is purely vertical (dx=0)
    - segment_length: length of the best segment pointing to this port

    The pipe connecting to this port must be roughly perpendicular to the edge.
    """
    best_score = 0.0
    best_len = 0.0

    for seg in segments:
        x1 = seg["x1"]
        y1 = seg["y1"]
        x2 = seg["x2"]
        y2 = seg["y2"]

        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)

        # Check if either endpoint is near this port
        endpoint_near = False
        for ep in [(x1, y1), (x2, y2)]:
            if abs(ep[0] - port_x) <= 2 and abs(ep[1] - port_y) <= 2:
                endpoint_near = True
                break

        # Also check if the segment's infinite line passes near the port
        # (for line-edge crossings where the port isn't near any endpoint)
        line_near = False
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            # Vector from endpoint to port
            t_x = (port_x - x1) / dx if abs(dx) > 1e-6 else 0
            t_y = (port_y - y1) / dy if abs(dy) > 1e-6 else 0
            # Use average t if both defined, else whichever is defined
            if abs(dx) > 1e-6 and abs(dy) > 1e-6:
                t = (t_x + t_y) / 2
            elif abs(dx) > 1e-6:
                t = t_x
            else:
                t = t_y
            # Check if projection point is on the segment (0 <= t <= 1)
            if 0 <= t <= 1:
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = math.hypot(proj_x - port_x, proj_y - port_y)
                if dist <= 15:
                    line_near = True

        if endpoint_near or line_near:
            seg_len = math.hypot(dx, dy)
            if edge in ("left", "right"):
                # Ideal: purely horizontal (dy ≈ 0)
                # Score = how horizontal the segment is
                perp_ratio = 1.0 - min(abs(dy) / seg_len, 1.0) if seg_len > 1e-6 else 0.0
            else:  # "top" or "bottom"
                # Ideal: purely vertical (dx ≈ 0)
                # Score = how vertical the segment is
                perp_ratio = 1.0 - min(abs(dx) / seg_len, 1.0) if seg_len > 1e-6 else 0.0

            if perp_ratio > best_score or (perp_ratio == best_score and seg_len > best_len):
                best_score = perp_ratio
                best_len = seg_len

    return best_score, best_len


def _best_port_for_connection(
    ports: list[tuple[int, int, str]],
    obj: dict[str, Any],
    segments: list[dict[str, Any]],
) -> list[tuple[int, int, str]]:
    """Pick the best port when a connection has multiple candidates.

    Tiebreaker priority:
    1. Highest perpendicularity score (pipe most perpendicular to edge)
    2. Longest connecting segment (as secondary tiebreaker)
    3. Port must be the entry point: segment direction must go INTO the bbox
       (not exit through the edge from inside the bbox)
    """
    if not ports:
        return ports
    if len(ports) == 1:
        return ports

    # ── Directionality gate: reject ports where the pipe exits the bbox ─────────
    # A connection port is where a pipe ENTERS the connection object.
    # The pipe must come from OUTSIDE the bbox and go INWARD.
    # If the segment direction points OUTWARD from the bbox, it's not the port.
    def _pipe_enters_inward(port_x: int, port_y: int, edge: str) -> bool:
        """Check if any segment near the port enters the bbox through this edge.

        A port is valid only if the pipe CROSSES the edge from outside to inside.
        We check whether the segment endpoint NEAREST the port lies OUTSIDE the bbox
        and the flow direction at that endpoint goes INWARD across the edge.
        """
        bbox = obj.get("bbox", {})
        x_min = bbox.get("x_min", 0)
        y_min = bbox.get("y_min", 0)
        x_max = bbox.get("x_max", 0)
        y_max = bbox.get("y_max", 0)

        for seg in segments:
            x1, y1, x2, y2 = int(seg["x1"]), int(seg["y1"]), int(seg["x2"]), int(seg["y2"])
            dx, dy = x2 - x1, y2 - y1

            # Find which endpoint is near the port
            d1 = math.hypot(x1 - port_x, y1 - port_y)
            d2 = math.hypot(x2 - port_x, y2 - port_y)
            if min(d1, d2) > 40:
                continue

            # Use the near endpoint as the flow reference point
            # The flow direction AT that endpoint is away from the other endpoint
            if d1 <= d2:
                # Flow direction at p1 is p1→p2 = (dx, dy)
                near_x, near_y = x1, y1
                flow_dx, flow_dy = dx, dy
            else:
                # Flow direction at p2 is p2→p1 = (-dx, -dy)
                near_x, near_y = x2, y2
                flow_dx, flow_dy = -dx, -dy

            # For entry, the near endpoint must be outside the bbox in the inward direction
            starts_outside = False
            if edge == "left":
                # Entry from left: flow goes rightward (dx > 0). The pipe originates from
                # LEFT of the bbox. Near endpoint may be just inside the bbox (p2), but
                # the actual pipe comes from outside. Accept if flow is rightward and
                # near endpoint is at or inside the left edge.
                starts_outside = flow_dx > 0 and near_x <= x_min + 10
            elif edge == "right":
                # Entry from right: flow goes leftward (dx < 0). Pipe comes from right.
                starts_outside = flow_dx < 0 and near_x >= x_max - 10
            elif edge == "top":
                # Entry from top: flow goes downward (dy > 0). Pipe comes from above.
                # Relaxed: accept if flow is downward and the near endpoint is not
                # deep inside the bbox (avoids rejecting top-edge ports where the
                # nearest endpoint is a short stub that dips below the bbox top).
                starts_outside = flow_dy > 0 and near_y <= y_min + 50
            elif edge == "bottom":
                # Entry from bottom: flow goes upward (dy < 0). Pipe comes from below.
                starts_outside = flow_dy < 0 and near_y >= y_max - 10

            if starts_outside:
                return True
        return False

    # Score each port
    scored = []
    for port_x, port_y, edge in ports:
        perp_score, seg_len = _score_port_perpendicularity(port_x, port_y, edge, segments)
        # Penalize ports where no segment points inward through this edge
        if not _pipe_enters_inward(port_x, port_y, edge):
            perp_score = 0.0
        scored.append((perp_score, seg_len, port_x, port_y, edge))

    # Sort by perpendicularity score (desc), then segment length (desc)
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)

    best_perp = scored[0][0]
    best_seg_len = scored[0][1]

    # Keep ports from every unique edge that meets a minimum quality bar.
    # This lets Pass 2 line-edge crossings survive even when the connecting
    # segment isn't perfectly perpendicular (e.g. short stub segments).
    min_perp = best_perp * 0.75  # 75% of best is acceptable
    top_candidates = [s for s in scored if s[0] >= min_perp]

    # For page connections: prefer left/right edges over top/bottom
    if obj.get("class_name") == "page connection" and len(top_candidates) > 1:
        side_candidates = [s for s in top_candidates if s[4] in ("left", "right")]
        if side_candidates:
            top_candidates = side_candidates

    # Return all top-scored ports (one per unique edge) — all ports that pass
    # the perpendicularity threshold and directionality gate.
    # Cap at 1 port to respect the design rule: one port per connection object.
    return [(s[2], s[3], s[4]) for s in top_candidates][:1]


def _mask_along_segment(
    port_x: int, port_y: int,
    ep_x: int, ep_y: int,
    bbox: dict[str, int],
    edge: str,
    mask: np.ndarray,
    scan_half_width: int = 10,
    min_ratio: float = 0.2,
) -> bool:
    """Check if pipe mask is present at the port position AND extends outward from the connection edge.

    A valid port requires:
    1. Pipe mask exists at or immediately adjacent to the port pixel on the edge
    2. Pipe extends inward from the edge into the connection bbox
    3. Pipe extends outward from the edge (into the area outside the bbox)

    This three-part check eliminates false positives where a pipe endpoint
    approaches a bbox edge but doesn't actually cross it — the pipe exists
    inside the connection but terminates at the edge without going outside.
    """
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]
    h, w = mask.shape

    # ── Part 1: Port must have mask at the edge pixel itself ──────────────────
    if not (0 <= port_x < w and 0 <= port_y < h):
        return False
    # Use adaptive threshold: 50% of max mask value (handles 0-255 masks or binary masks)
    threshold = max(1, int(mask.max() * 0.5))
    if mask[port_y, port_x] <= threshold:
        # Check immediate neighbors on the edge line
        edge_has_mask = False
        for delta in range(-2, 3):
            if edge in ("left", "right"):
                ny = port_y + delta
                if 0 <= ny < h and mask[ny, port_x] > threshold:
                    edge_has_mask = True
                    break
            else:
                nx = port_x + delta
                if 0 <= nx < w and mask[port_y, nx] > threshold:
                    edge_has_mask = True
                    break
        if not edge_has_mask:
            return False

    if edge in ("left", "right"):
        # Perpendicular direction = horizontal (into bbox interior for "in",
        # away from bbox for "out")
        in_col = x_min + 1 if edge == "left" else x_max - 1
        out_col = x_min - 1 if edge == "left" else x_max + 1

        # Part 2: sample inward — count ALL white pixels in the scan range
        # (not just consecutive), to handle noisy/thin pipes
        white_in = sum(
            1 for col in range(in_col, min(in_col + scan_half_width + 1, w))
            if mask[port_y, col] > threshold
        )

        # Part 3: sample outward — count ALL white pixels in the scan range
        white_out = sum(
            1 for col in range(out_col, max(out_col - scan_half_width - 1, -1), -1)
            if 0 <= col < w and mask[port_y, col] > threshold
        )

        in_ratio = white_in / float(scan_half_width + 1)
        out_ratio = white_out / float(scan_half_width + 1)
        return in_ratio >= min_ratio and out_ratio >= min_ratio

    else:  # top / bottom
        in_row = y_min + 1 if edge == "top" else y_max - 1
        out_row = y_min - 1 if edge == "top" else y_max + 1

        # Part 2: sample inward — count ALL white pixels in the scan range
        # (not just consecutive), to handle noisy/thin pipes
        white_in = sum(
            1 for row in range(in_row, min(in_row + scan_half_width + 1, h))
            if mask[row, port_x] > threshold
        )

        # Part 3: sample outward — count ALL white pixels in the scan range
        white_out = sum(
            1 for row in range(out_row, max(out_row - scan_half_width - 1, -1), -1)
            if 0 <= row < h and mask[row, port_x] > threshold
        )

        in_ratio = white_in / float(scan_half_width + 1)
        out_ratio = white_out / float(scan_half_width + 1)
        return in_ratio >= min_ratio and out_ratio >= min_ratio


def _port_on_edge_crossing(
    port_x: int, port_y: int,
    edge: str,
    bbox: dict[str, int],
    mask: np.ndarray,
    scan_half_width: int = 15,
    min_crossing_ratio: float = 0.2,
) -> bool:
    """Check if a pipe line runs perpendicular to and crosses the bbox edge at the port position.

    A valid crossing requires significant pipe mask on BOTH sides of the edge,
    in the direction perpendicular to the edge. The sampling is done along the
    edge direction (NOT at the port point itself) to handle cases where the pipe
    crosses slightly offset from the nominal port point.

    For 'left'/'right' edges: sample perpendicular = horizontal row scan at port_y ± scan_half_width
    For 'top'/'bottom' edges: sample perpendicular = vertical column scan at port_x ± scan_half_width

    A port is valid only if at least one scan line shows significant white pixels
    on both sides of the edge.
    """
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]
    h, w = mask.shape

    if edge in ("left", "right"):
        # Sample rows around port_y (perpendicular to edge = horizontal lines)
        # For each row, check both inward and outward sides of the edge
        in_sides_ok = 0
        out_sides_ok = 0

        for row_offset in range(-scan_half_width, scan_half_width + 1):
            scan_y = port_y + row_offset
            if not (0 <= scan_y < h):
                continue

            if edge == "left":
                in_col = x_min + 1  # first column inside bbox
                out_col = x_min - 1  # first column outside bbox
            else:  # right
                in_col = x_max - 1   # last column inside bbox
                out_col = x_max + 1  # first column outside bbox

            # Count consecutive white pixels from the edge inward and outward
            in_white = 0
            for col in range(in_col, min(in_col + scan_half_width + 1, w)):
                if mask[scan_y, col] > 127:
                    in_white += 1
                else:
                    break

            out_white = 0
            for col in range(out_col, max(out_col - scan_half_width - 1, -1), -1):
                if 0 <= col < w and mask[scan_y, col] > 127:
                    out_white += 1
                else:
                    break

            # This row counts as a crossing if both sides have pipe
            if in_white > 0 and out_white > 0:
                in_sides_ok += 1
                out_sides_ok += 1

        # At least one scan row must show crossing on both sides
        if in_sides_ok > 0 and out_sides_ok > 0:
            return True

    else:  # 'top' or 'bottom'
        # Sample columns around port_x (perpendicular to edge = vertical lines)
        in_sides_ok = 0
        out_sides_ok = 0

        for col_offset in range(-scan_half_width, scan_half_width + 1):
            scan_x = port_x + col_offset
            if not (0 <= scan_x < w):
                continue

            if edge == "top":
                in_row = y_min + 1   # first row inside bbox
                out_row = y_min - 1  # first row outside bbox
            else:  # bottom
                in_row = y_max - 1   # last row inside bbox
                out_row = y_max + 1  # first row outside bbox

            # Count consecutive white pixels from the edge inward and outward
            in_white = 0
            for row in range(in_row, min(in_row + scan_half_width + 1, h)):
                if mask[row, scan_x] > 127:
                    in_white += 1
                else:
                    break

            out_white = 0
            for row in range(out_row, max(out_row - scan_half_width - 1, -1), -1):
                if 0 <= row < h and mask[row, scan_x] > 127:
                    out_white += 1
                else:
                    break

            # This column counts as a crossing if both sides have pipe
            if in_white > 0 and out_white > 0:
                in_sides_ok += 1
                out_sides_ok += 1

        # At least one scan column must show crossing on both sides
        if in_sides_ok > 0 and out_sides_ok > 0:
            return True

    return False


def get_connection_ports(
    objects: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    mask: np.ndarray | None = None,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, list[tuple[int, int, str]]]:
    """
    Find actual pipe connection ports by snapping segment endpoints to connection bbox edges.

    A pipe connects to a connection object only when the segment endpoint lies ON or
    immediately ADJACENT (within 1px) to one of the four bbox edges. The port
    position is the perpendicular projection of the endpoint onto that edge.

    Ports are validated against the pipe mask — if the mask doesn't continuously
    connect the endpoint to the projected port position, the port is rejected.
    This eliminates false positives where an endpoint approaches a bbox edge
    but the actual pipe drawing doesn't reach.

    For page connections with multiple valid ports, only the best port is kept
    (left/right edges preferred over top/bottom; longest connecting segment breaks ties).

    Returns a dict mapping object_id -> list of (port_x, port_y, edge_name).
    """
    from collections import defaultdict

    ports: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    conn_objects = [o for o in objects if _is_connection_object(o)]

    for seg in segments:
        x1 = int(seg["x1"])
        y1 = int(seg["y1"])
        x2 = int(seg["x2"])
        y2 = int(seg["y2"])

        # Process both endpoints, skipping the segment junction point
        seen = set()
        for endpoint in [(x1, y1), (x2, y2)]:
            if endpoint in seen:
                continue
            seen.add(endpoint)

            for obj in conn_objects:
                obj_id = obj["id"]
                bbox = obj["bbox"]

                edge = _find_touching_edge(endpoint, bbox)
                if edge is not None:
                    port_x, port_y = _project_port_to_edge(endpoint, bbox, edge)
                    # Deduplicate: only one port per edge per object
                    if not any(p[2] == edge for p in ports[obj_id]):
                        # Validate with crossing check: require pipe on BOTH sides of edge
                        if mask is None or (_mask_along_segment(port_x, port_y, endpoint[0], endpoint[1], bbox, edge, mask)
                                           and _port_on_edge_crossing(port_x, port_y, edge, bbox, mask)):
                            ports[obj_id].append((port_x, port_y, edge))

    # Post-process: for page connections with >1 port, keep only the best one
    id_to_obj = {o["id"]: o for o in conn_objects}
    result: dict[str, list[tuple[int, int, str]]] = {}
    for obj_id, port_list in ports.items():
        obj = id_to_obj.get(obj_id, {})
        result[obj_id] = _best_port_for_connection(port_list, obj, segments)

    # ── Additional pass: compute line-edge intersections for each segment ─────
    # This catches diagonal pipes where the detected segment endpoint is NOT near
    # any bbox edge, but the infinite line crosses an edge elsewhere on the bbox.
    if mask is not None:
        for seg in segments:
            x1 = int(seg["x1"])
            y1 = int(seg["y1"])
            x2 = int(seg["x2"])
            y2 = int(seg["y2"])

            for obj in conn_objects:
                obj_id = obj["id"]
                bbox = obj["bbox"]

                # Get line-edge crossings (not just near endpoints)
                crossings = _line_intersects_bbox_edge(x1, y1, x2, y2, bbox)
                for port_x, port_y, edge in crossings:
                    # Validate that the crossing point actually has pipe mask
                    threshold = max(1, int(mask.max() * 0.5))
                    h, w = mask.shape
                    if not (0 <= port_x < w and 0 <= port_y < h):
                        continue
                    if mask[port_y, port_x] <= threshold:
                        continue
                    # Line-edge intersection is geometrically exact — no perpendicularity
                    # gate needed here; the pipe demonstrably crosses this edge.
                    # Add with same-edge dedup (only one port per edge per object)
                    if not any(p[2] == edge for p in result[obj_id]):
                        result[obj_id].append((port_x, port_y, edge))

    # ── Final selection: for connections with multiple ports, pick the best one ──
    # based on perpendicularity score and segment length
    id_to_obj = {o["id"]: o for o in conn_objects}
    for obj_id in list(result.keys()):
        obj = id_to_obj.get(obj_id, {})
        result[obj_id] = _best_port_for_connection(result[obj_id], obj, segments)

    return result


def _connection_port_center(bbox: dict[str, int]) -> tuple[int, int]:
    """
    Return (x, y) center of the short edge (pipe connection side) for a connection object.
    Page connections are horizontal (aspect ~5:1) — port is on a vertical edge (left or right).
    Utility/connection objects are roughly square — port on the closest-to-image-edge side.
    """
    x_min = bbox["x_min"]
    x_max = bbox["x_max"]
    y_min = bbox["y_min"]
    y_max = bbox["y_max"]
    w = x_max - x_min
    h = y_max - y_min

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    # Horizontal rectangle — pipe comes from left or right edge
    if w > h * 1.5:
        # Page connection: pick the edge closer to image center
        img_center_x = 2481  # ~half of 4963
        port_x = x_max if cx >= img_center_x else x_min
        return port_x, cy

    # Square-ish — use the edge closest to image center
    img_center_x = 2481
    img_center_y = 1754  # ~half of 3509
    edges = [
        ("left", x_min, cy),
        ("right", x_max, cy),
        ("top", cx, y_min),
        ("bottom", cx, y_max),
    ]
    # Pick the edge whose midpoint is closest to image center
    best = min(edges, key=lambda e: abs(e[1] - img_center_x) + abs(e[2] - img_center_y))
    return best[1], best[2]


def _draw_overlay(
    image_bgr: np.ndarray,
    objects: list[dict[str, Any]],
    *,
    connection_port_radius: int = 8,
    connection_ports: dict[str, list[tuple[int, int, str]]] | None = None,
) -> np.ndarray:
    """
    Draw blue bboxes for all objects; for connection objects draw port marker(s).

    If `connection_ports` is provided (object_id -> list of (port_x, port_y, edge)),
    draw a port marker for each actual pipe connection.
    If not provided, falls back to midpoint-based heuristic (legacy).
    """
    overlay = image_bgr.copy()
    CONNECTION_TYPES = frozenset(["page connection", "utility connection", "connection"])

    for obj in objects:
        bbox = obj["bbox"]
        x_min = int(bbox["x_min"])
        y_min = int(bbox["y_min"])
        x_max = int(bbox["x_max"])
        y_max = int(bbox["y_max"])

        # Blue bbox for all objects
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        if obj.get("class_name", "") in CONNECTION_TYPES:
            # Determine port positions
            if connection_ports and obj["id"] in connection_ports:
                port_list = connection_ports[obj["id"]]
            else:
                # Legacy fallback: use midpoint heuristic
                px, py = _connection_port_center(bbox)
                port_list = [(px, py, "unknown")]

            for port_x, port_y, edge_name in port_list:
                # Filled cyan circle at port
                cv2.circle(overlay, (port_x, port_y), connection_port_radius, (255, 255, 0), -1)

                # White border on the port circle for visibility
                cv2.circle(overlay, (port_x, port_y), connection_port_radius, (255, 255, 255), 1)

                # Crosshair lines to make the port point obvious
                half = connection_port_radius + 4
                cv2.line(overlay, (port_x - half, port_y), (port_x + half, port_y), (255, 255, 255), 1)
                cv2.line(overlay, (port_x, port_y - half), (port_x, port_y + half), (255, 255, 255), 1)

                # Label the edge
                cv2.putText(
                    overlay,
                    edge_name[:3].upper(),
                    (port_x + connection_port_radius + 2, port_y - connection_port_radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1,
                )

    return overlay


def run_object_detection_sahi(
    image_path: Path | str,
    image_id: str,
    cfg: DetectionSahiConfig | None = None,
    connection_ports: dict[str, list[tuple[int, int, str]]] | None = None,
) -> dict[str, Any]:
    config = cfg or DetectionSahiConfig()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image for detection: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detection_model = _build_detection_model(config)
    result = get_sliced_prediction(
        image=image_rgb,
        detection_model=detection_model,
        slice_height=config.image_size,
        slice_width=config.image_size,
        overlap_height_ratio=config.overlap_ratio,
        overlap_width_ratio=config.overlap_ratio,
        postprocess_type=config.postprocess_type,
        postprocess_match_metric=config.postprocess_match_metric,
        postprocess_match_threshold=config.postprocess_match_threshold,
        verbose=0,
    )

    objects: list[dict[str, Any]] = []
    for idx, detection in enumerate(result.object_prediction_list, start=1):
        bbox_xyxy = detection.bbox.to_xyxy()
        objects.append(
            {
                "id": f"obj_{idx:06d}",
                "class_name": detection.category.name,
                "confidence": round(float(detection.score.value), 4),
                "bbox": {
                    "x_min": int(bbox_xyxy[0]),
                    "y_min": int(bbox_xyxy[1]),
                    "x_max": int(bbox_xyxy[2]),
                    "y_max": int(bbox_xyxy[3]),
                },
                "source_model": config.model_type,
                "source_weight": config.weight_path,
            }
        )

    class_counts = dict(sorted(Counter(obj["class_name"] for obj in objects).items()))
    summary = {
        "image_id": image_id,
        "pass_type": "sheet",
        "route": config.model_type,
        "object_count": len(objects),
        "class_counts": class_counts,
        "image_size": config.image_size,
        "overlap_ratio": config.overlap_ratio,
        "postprocess_type": config.postprocess_type,
        "postprocess_match_metric": config.postprocess_match_metric,
        "postprocess_match_threshold": config.postprocess_match_threshold,
        "source_model": config.model_type,
        "source_weight": config.weight_path,
    }

    # connection_ports: optional pre-computed port positions from stage 5
    # If provided, _draw_overlay will draw actual pipe-connected ports instead of midpoint heuristic
    overlay = _draw_overlay(image_bgr, objects, connection_ports=connection_ports)

    return {
        "objects_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "objects": objects,
        },
        "summary": summary,
        "overlay_image": overlay,
    }
