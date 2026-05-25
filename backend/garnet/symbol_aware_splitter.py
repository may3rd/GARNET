"""
symbol_aware_splitter.py

Split segments at symbol boundary crossings — pipe goes through, symbol
is removed, but the pipe itself survives as two separate traceable segments.

Unlike inpainting (which removes symbols first, then traces through the
cleaned image), this approach cuts segments at symbol boundaries so pipes
remain traceable even after symbol removal.

Integration point: after _extract_contour_segments, before _merge_collinear_segments
in run_line_detection_inpaint.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Tunable — must match line_detection_inpaint.MIN_SEGMENT_LENGTH_PX
MIN_PIECE_PX = 25.0

# Margin added around symbol bboxes to avoid cutting pipes that run
# alongside a symbol (just outside its bounding box). Without this,
# a pipe 2px beside a valve would be cut at the valve's corner.
SYMBOL_MARGIN_PX = 3.0

# Angle tolerance for treating a segment as nearly horizontal/vertical
HV_TOLERANCE_DEG = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal types
# ─────────────────────────────────────────────────────────────────────────────

Segment = dict[str, Any]  # {"x1": int, "y1": int, "x2": int, "y2": int, ...}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────


def _segment_angle_deg(seg: Segment) -> float:
    dx = float(seg["x2"] - seg["x1"])
    dy = float(seg["y2"] - seg["y1"])
    return math.degrees(math.atan2(dy, dx)) % 180.0  # 0-180


def _point_in_rect(
    px: float, py: float,
    x_min: float, y_min: float,
    x_max: float, y_max: float,
) -> bool:
    return (x_min <= px <= x_max) and (y_min <= py <= y_max)


def _segment_endpoints_in_rect(
    seg: Segment,
    x_min: float, y_min: float,
    x_max: float, y_max: float,
) -> str:
    """
    Classify a segment's relationship to a rect.

    Returns:
        "both_inside"  — both endpoints are inside the rect
        "crosses"      — one endpoint inside, one outside
                       — or both endpoints outside but segment passes through (2 intersections)
        "touches"      — both endpoints outside, segment intersects rect at exactly 1 point
                       — (grazes a corner or edge without passing through)
        "outside"      — no intersection with rect at all
    """
    x1, y1 = seg["x1"], seg["y1"]
    x2, y2 = seg["x2"], seg["y2"]

    p1_in = _point_in_rect(x1, y1, x_min, y_min, x_max, y_max)
    p2_in = _point_in_rect(x2, y2, x_min, y_min, x_max, y_max)

    if p1_in and p2_in:
        return "both_inside"
    if p1_in != p2_in:
        return "crosses"

    # Both outside — check for intersection
    pts = _intersection_points_with_rect(seg, x_min, y_min, x_max, y_max)
    if len(pts) >= 2:
        return "crosses"  # passes through rect (enters + exits)
    if len(pts) == 1:
        return "touches"  # grazes corner or is flush along edge
    return "outside"


def _segment_intersects_rect(
    seg: Segment,
    x_min: float, y_min: float,
    x_max: float, y_max: float,
) -> bool:
    """
    Return True if the segment's infinite line intersects the rectangle.
    Uses parametric line intersection with each of the 4 edges.
    """
    x1, y1 = float(seg["x1"]), float(seg["y1"])
    x2, y2 = float(seg["x2"]), float(seg["y2"])

    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return False  # Degenerate

    def _ray_segment_intersects(
        r_x1, r_y1, r_x2, r_y2,
    ) -> bool:
        """Check if segment (x1,y1)-(x2,y2) intersects ray (r_x1,r_y1)-(r_x2,r_y2)."""
        # Parametric: P = p1 + t*(p2-p1), Q = q1 + u*(q2-q1)
        # Solve p1 + t*d = q1 + u*e
        px1, py1 = x1, y1
        qx1, qy1 = r_x1, r_y1
        ddx, ddy = dx, dy
        edx, edy = r_x2 - r_x1, r_y2 - r_y1

        # Parametric: P = p1 + t*d, Q = q1 + u*e
        # Solve P(t) = Q(u):
        #   x1 + t*dx = qx1 + u*edx  (1)
        #   y1 + t*dy = qy1 + u*edy  (2)
        # Cramer's rule:
        #   t = (edx*(y1-qy1) - edy*(x1-qx1)) / denom
        #   u = (dx*(qy1-y1) - dy*(qx1-x1)) / denom
        denom = ddx * edy - ddy * edx
        if abs(denom) < 1e-12:
            return False  # Parallel

        t = (edx * (y1 - r_y1) - edy * (x1 - r_x1)) / denom
        u = (y1 + t * ddy - r_y1) / edy

        eps = 1e-9
        return (-eps <= t <= 1 + eps) and (-eps <= u <= 1 + eps)

    # Check all 4 edges of the rectangle
    return (
        _ray_segment_intersects(x_min, y_min, x_max, y_min) or
        _ray_segment_intersects(x_max, y_min, x_max, y_max) or
        _ray_segment_intersects(x_max, y_max, x_min, y_max) or
        _ray_segment_intersects(x_min, y_max, x_min, y_min)
    )


def _intersection_points_with_rect(
    seg: Segment,
    x_min: float, y_min: float,
    x_max: float, y_max: float,
) -> list[tuple[float, float]]:
    """
    Return list of (x, y) intersection points between the segment's infinite
    line and the rectangle edges. Points are ordered along the segment
    from endpoint 1 → endpoint 2.

    Only returns points where the segment actually crosses the edge (t in (0,1)).
    """
    x1, y1 = float(seg["x1"]), float(seg["y1"])
    x2, y2 = float(seg["x2"]), float(seg["y2"])

    dx, dy = x2 - x1, y2 - y1

    intersections: list[tuple[float, float, float]] = []  # (x, y, t)

    # Helper: check segment (p1→p2) vs rectangle edge (e1→e2)
    def _edge_intersection(
        ex1, ey1, ex2, ey2,
    ) -> tuple[float, float] | None:
        # Parametric: P = x1 + t*dx, Q = ex1 + u*(ex2-ex1)
        edx, edy = ex2 - ex1, ey2 - ey1
        denom = dx * edy - dy * edx
        if abs(denom) < 1e-12:
            return None
        t = (edx * (y1 - ey1) - edy * (x1 - ex1)) / denom
        # u from y-equation: y1 + t*dy = ey1 + u*edy
        # If edge is horizontal (edy≈0), use x-equation instead: x1 + t*dx = ex1 + u*edx
        if abs(edy) > 1e-12:
            u = (y1 + t * dy - ey1) / edy
        elif abs(edx) > 1e-12:
            u = (x1 + t * dx - ex1) / edx
        else:
            return None  # Degenerate edge
        eps = 1e-9
        if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
            return (x1 + t * dx, y1 + t * dy)
        return None

    edges = [
        (x_min, y_min, x_max, y_min),
        (x_max, y_min, x_max, y_max),
        (x_max, y_max, x_min, y_max),
        (x_min, y_max, x_min, y_min),
    ]
    for ex1, ey1, ex2, ey2 in edges:
        pt = _edge_intersection(ex1, ey1, ex2, ey2)
        if pt is not None:
            tx = (pt[0] - x1) / dx if abs(dx) > 1e-9 else (pt[1] - y1) / dy if abs(dy) > 1e-9 else 0.0
            intersections.append((pt[0], pt[1], tx))

    # Sort by t (position along segment from p1 → p2)
    intersections.sort(key=lambda p: p[2])
    return [(ix, iy) for ix, iy, _ in intersections]


def _is_nearly_hv(seg: Segment) -> bool:
    """Return True if segment is clearly horizontal or vertical (within tolerance)."""
    ang = _segment_angle_deg(seg)
    return (ang <= HV_TOLERANCE_DEG or ang >= 180 - HV_TOLERANCE_DEG or
            abs(ang - 90.0) <= HV_TOLERANCE_DEG)


# ─────────────────────────────────────────────────────────────────────────────
# Core splitting logic
# ─────────────────────────────────────────────────────────────────────────────


def _split_segment_at_points(
    seg: Segment,
    points: list[tuple[float, float]],
) -> list[Segment]:
    """
    Split a segment at ordered intersection points.

    Args:
        seg: Original segment
        points: Intersection points sorted from endpoint 1 → endpoint 2

    Returns:
        List of split segments (>= MIN_PIECE_PX in length).
    """
    x1, y1 = float(seg["x1"]), float(seg["y1"])
    x2, y2 = float(seg["x2"]), float(seg["y2"])

    if not points:
        return [seg]

    # Build breakpoints: start point, all intersections, end point
    all_pts = [(x1, y1)] + list(points) + [(x2, y2)]

    results: list[Segment] = []
    for i in range(len(all_pts) - 1):
        px1, py1 = all_pts[i]
        px2, py2 = all_pts[i + 1]
        seg_len = math.hypot(px2 - px1, py2 - py1)
        if seg_len >= MIN_PIECE_PX:
            results.append({
                "x1": int(round(px1)),
                "y1": int(round(py1)),
                "x2": int(round(px2)),
                "y2": int(round(py2)),
                "length": seg_len,
                "area_parent": seg.get("area_parent", 0),
            })

    return results if results else [seg]


def _should_split_segment(
    seg: Segment,
    x_min: float, y_min: float,
    x_max: float, y_max: float,
) -> bool:
    """
    Determine if a segment should be split at a symbol bbox.

    Rules:
    - "both_inside" → don't split (inpainting handles it; segment is fully occluded)
    - "crosses" → split (pipe goes through symbol, cut at entry and exit)
    - "touches" → don't split (segment only grazes corner, not a through-passage)
    - "outside" → don't split
    """
    rel = _segment_endpoints_in_rect(seg, x_min, y_min, x_max, y_max)
    return rel == "crosses"


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol-bbox processing
# ─────────────────────────────────────────────────────────────────────────────


def _split_segments_by_single_bbox(
    segments: list[Segment],
    x_min: float, y_min: float,
    x_max: float, y_max: float,
) -> list[Segment]:
    """
    Process all segments against one symbol bounding box.
    Returns new segment list with splits applied for that bbox.
    """
    result: list[Segment] = []

    for seg in segments:
        if _should_split_segment(seg, x_min, y_min, x_max, y_max):
            pts = _intersection_points_with_rect(seg, x_min, y_min, x_max, y_max)
            split_pieces = _split_segment_at_points(seg, pts)
            result.extend(split_pieces)
        else:
            result.append(seg)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def split_segments_at_symbols(
    segments: list[Segment],
    text_regions: list[dict[str, Any]],
    object_regions: list[dict[str, Any]],
    *,
    margin_px: float = SYMBOL_MARGIN_PX,
) -> list[Segment]:
    """
    Split segments where they cross through symbol bounding boxes.

    Unlike inpainting (which removes symbols then traces pipes), this cuts
    the segment at symbol boundaries so pipes remain traceable as two
    separate pieces even after symbol removal.

    Args:
        segments: Raw segments from _extract_contour_segments.
            Each has keys: x1, y1, x2, y2, length, area_parent.
        text_regions: OCR text regions from Stage 2. Each has bbox.
        object_regions: Object detection results from Stage 4. Each has bbox.
        margin_px: Safety margin around bbox edges (default 3px).
            Pipes running alongside (but not through) a symbol are not cut.

    Returns:
        List of segments with cuts applied at symbol crossings.
        Short pieces (< MIN_PIECE_PX) are dropped.

    Integration:
        Called between _extract_contour_segments and _merge_collinear_segments
        in run_line_detection_inpaint.
    """
    if not segments:
        return []

    # Skip text classes that sit ON pipe lines (line_number, unknown).
    # These are labels like pipe sizes drawn directly on pipes — masking
    # them would cut the pipe at the label rather than at the actual symbol.
    _SKIP_TEXT_CLASSES = frozenset(["line_number", "unknown"])

    # Collect all symbol bounding boxes
    symbol_bboxes: list[tuple[float, float, float, float]] = []

    for region in text_regions:
        if region.get("class") in _SKIP_TEXT_CLASSES:
            continue
        bbox = region.get("bbox")
        if not bbox:
            continue
        x_min = float(bbox["x_min"]) - margin_px
        y_min = float(bbox["y_min"]) - margin_px
        x_max = float(bbox["x_max"]) + margin_px
        y_max = float(bbox["y_max"]) + margin_px
        symbol_bboxes.append((x_min, y_min, x_max, y_max))

    # Object bboxes are NOT added — the existing inpaint mask already handles
    # them via corner-point detection, and adding them here causes pipes that
    # pass through valve/instrument bodies to be over-split.
    # The corner-point grid (CORNER_GRID_CELL_PX=40) already captures most
    # symbol regions. Adding explicit object bboxes here creates double-cutting.
    # See: _assemble_inpaint_mask Phase 3 comment.

    if not symbol_bboxes:
        return segments

    result = list(segments)

    # Sort bboxes by area (smallest first) to minimize over-splitting on
    # nested/overlapping symbol regions
    def _bbox_area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    symbol_bboxes.sort(key=_bbox_area)

    for x_min, y_min, x_max, y_max in symbol_bboxes:
        result = _split_segments_by_single_bbox(result, x_min, y_min, x_max, y_max)

    return result


def split_segments_at_symbols_with_objects(
    segments: list[Segment],
    text_regions: list[dict[str, Any]],
    object_regions: list[dict[str, Any]],
    *,
    margin_px: float = SYMBOL_MARGIN_PX,
    include_objects: bool = True,
) -> list[Segment]:
    """
    Extended variant: optionally include object bboxes in addition to text.

    Use this when the corner-point approach alone is insufficient (e.g., large
    instruments with poor corner detection).

    Args:
        include_objects: If True, also use object detection bboxes for splitting.
            Default False (corner-point approach is preferred to avoid over-cutting).
    """
    if not segments:
        return []

    _SKIP_TEXT_CLASSES = frozenset(["line_number", "unknown"])

    symbol_bboxes: list[tuple[float, float, float, float]] = []

    for region in text_regions:
        if region.get("class") in _SKIP_TEXT_CLASSES:
            continue
        bbox = region.get("bbox")
        if not bbox:
            continue
        x_min = float(bbox["x_min"]) - margin_px
        y_min = float(bbox["y_min"]) - margin_px
        x_max = float(bbox["x_max"]) + margin_px
        y_max = float(bbox["y_max"]) + margin_px
        symbol_bboxes.append((x_min, y_min, x_max, y_max))

    if include_objects:
        for obj in object_regions:
            bbox = obj.get("bbox")
            if not bbox:
                continue
            x_min = float(bbox["x_min"]) - margin_px
            y_min = float(bbox["y_min"]) - margin_px
            x_max = float(bbox["x_max"]) + margin_px
            y_max = float(bbox["y_max"]) + margin_px
            symbol_bboxes.append((x_min, y_min, x_max, y_max))

    if not symbol_bboxes:
        return segments

    result = list(segments)

    def _bbox_area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    symbol_bboxes.sort(key=_bbox_area)

    for x_min, y_min, x_max, y_max in symbol_bboxes:
        result = _split_segments_by_single_bbox(result, x_min, y_min, x_max, y_max)

    return result