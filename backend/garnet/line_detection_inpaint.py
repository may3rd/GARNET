"""
line_detection_inpaint.py

Geometric line-extraction pipeline for P&ID pipe detection.

Replaces the rectangular-suppression mask approach with:
  1. Angle-change feature-point detection to find symbol corners
  2. Telea inpainting over detected symbol regions
  3. Connected-component contour tracing on the cleaned binary image
  4. Collinearity merging + horizontal/vertical split

Based on Ali et al. (2026) geometric pipeline (ssrn-6083108)
and adapted for GARNET Stage 5 → Stage 6 integration.

Phase order:
    A. Adaptive threshold   → binary pipe candidate mask
    B. Corner detection      → Shi-Tomasi angle-change feature points
    C. Symbol mask assembly  → merged bboxes from corner grid-hash + text + objects
    D. Inpaint                → Telea removes symbols, preserves pipes
    E. Contour extraction    → connected-component on cleaned binary
    F. Segment merging       → collinear IoU merge + H / V split
    G. Terminal cleanup      → Endpoint proximity merge + orphan pruning

Output shape mirrors stage5_pipe_mask pipeline but returns explicit line segments
instead of a binary mask to avoid skeleton-gaps.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


# ───────────────────────────────────────────────────────────────
# Tunable parameters (documented, kept central)
# ───────────────────────────────────────────────────────────────

# Thresholding
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 5

# Corner detection (Shi-Tomasi)
CORNER_MAX_CORNERS = 2000
CORNER_QUALITY_LEVEL = 0.02
CORNER_MIN_DISTANCE = 8

# Corner bounding-box: spatial grid cell size in pixels
# Smaller = more fine-grained boxes, larger = coarser grouping
CORNER_GRID_CELL_PX = 40

# Region dilation for inpaint mask
# Tighter kernel (9,9) reduces margin from ~10px to ~4px to avoid
# eating pipe lines that run close to object bounding boxes
# (e.g., line numbers near pipes).
INPAINT_DILATE_KERNEL = (9, 9)
INPAINT_RADIUS = 5

# Connected-component filtering
MIN_COMPONENT_AREA = 15
MAX_COMPONENT_AREA = 500_000

# Segment merging
COLLINEAR_IoU_THRESHOLD = 0.15
COLLINEAR_ANGLE_TOLERANCE_DEG = 15.0

# H / V split threshold (angles closer to 0 or 180 = horizontal)
HV_ANGLE_DEG = 45.0

# Endpoint proximity merge
ENDPOINT_MERGE_PX = 15.0
ENDPOINT_MERGE_ANGLE_TOLERANCE_DEG = 20.0

# Orphan removal
MIN_SEGMENT_LENGTH_PX = 12.0


# ───────────────────────────────────────────────────────────────
# Internal types
# ───────────────────────────────────────────────────────────────

Segment = dict[str, Any]  # {"x1": int, "y1": int, "x2": int, "y2": int, ...}


# ═══════════════════════════════════════════════════════════════
# Phase A — Binary mask from adaptive threshold
# ═══════════════════════════════════════════════════════════════


def _adaptive_threshold_mask(gray: np.ndarray) -> np.ndarray:
    """Return binary mask: 255 = pipe candidate, 0 = background."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )
    return binary


# ═══════════════════════════════════════════════════════════════
# Phase B — Angle-change feature points (Shi-Tomasi)
# ═══════════════════════════════════════════════════════════════


def _detect_corner_points(thresh: np.ndarray) -> np.ndarray:
    """
    Return Nx2 array of (x, y) corner coordinates via Shi-Tomasi.
    These cluster around symbol vertices / intersections.
    """
    corners = cv2.goodFeaturesToTrack(
        thresh,
        maxCorners=CORNER_MAX_CORNERS,
        qualityLevel=CORNER_QUALITY_LEVEL,
        minDistance=CORNER_MIN_DISTANCE,
        blockSize=5,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.int32)
    return corners.reshape(-1, 2).astype(np.int32)


# ═══════════════════════════════════════════════════════════════
# Phase C — Inpaint mask assembly
#   Fast O(n) grid-hash bounding-box assembly:
#     1. Accumulate each corner into its grid cell's bounding rect
#     2. Merge overlapping rects via cv2.groupRectangles
#   This replaces the O(n²) DBSCAN clustering.
# ═══════════════════════════════════════════════════════════════


def _points_to_bboxes_fast(corners: np.ndarray, cell_px: int = 40) -> list[tuple[int, int, int, int]]:
    """
    Convert sparse corner point cloud into bounding boxes using spatial grid hashing.

    Algorithm (O(n)):
      1. For each point, accumulate into grid cell's (min_x, min_y, max_x, max_y).
      2. Emit one (x, y, w, h) bbox per occupied cell.
      3. Merge overlapping boxes via cv2.groupRectangles.

    Args:
        corners: Nx2 array of (x, y) corner coordinates
        cell_px: grid cell size in pixels

    Returns:
        List of (x, y, w, h) rectangles covering all corner points.
    """
    if len(corners) == 0:
        return []

    cells: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    for x, y in corners:
        col = int(x // cell_px)
        row = int(y // cell_px)
        key = (col, row)
        if key in cells:
            cx_min, cy_min, cx_max, cy_max = cells[key]
            cells[key] = (min(cx_min, x), min(cy_min, y),
                          max(cx_max, x), max(cy_max, y))
        else:
            cells[key] = (x, y, x, y)

    if not cells:
        return []

    # Build rect list for groupRectangles — requires int tuples (x, y, w, h)
    rects = [
        (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        for (x_min, y_min, x_max, y_max) in cells.values()
    ]

    # Merge overlapping / adjacent boxes (eps = 10% of cell size)
    eps = cell_px * 0.10
    merged, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=eps)
    return [(int(rx), int(ry), max(1, int(rw)), max(1, int(rh))) for (rx, ry, rw, rh) in merged]


def _assemble_inpaint_mask(
    shape: tuple[int, int],
    corner_points: np.ndarray,
    text_regions: list[dict[str, Any]],
    object_regions: list[dict[str, Any]],
) -> np.ndarray:
    """
    Build binary mask where 255 = region to inpaint (symbol / text).
    Combines corner-derived bboxes with explicit text + object boxes.
    Uses fast grid-hash bounding-box assembly (O(n), no O(n²) clustering).
    """
    h, w = shape[:2]

    # 1. Corner-derived bboxes via fast grid accumulation
    corner_boxes = _points_to_bboxes_fast(corner_points, cell_px=CORNER_GRID_CELL_PX)
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y, bw, bh in corner_boxes:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w - 1, x + bw)
        y2 = min(h - 1, y + bh)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # 2. Text suppression boxes — skip "line_number" class to preserve
    # pipe lines that run close to line number labels.
    for region in text_regions:
        if region.get("class") == "line_number":
            continue
        bbox = region.get("bbox")
        if not bbox:
            continue
        x_min = int(bbox["x_min"])
        y_min = int(bbox["y_min"])
        x_max = int(bbox["x_max"])
        y_max = int(bbox["y_max"])
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

    # 3. Object suppression boxes (symbol interiors) — skip classes that
    # sit ON the pipe line itself. Inpainting them would erase the pipe.
    # Excluded: line number, arrow (drawn on pipe lines).
    # Hybrid approach: page connection symbols are inpainted but with a
    # center strip preserved (where the pipe passes through). This removes
    # the symbol text while keeping the pipe continuity.
    _SKIP_OBJECT_CLASSES = frozenset(["line number", "arrow"])
    _HYBRID_CLASSES = frozenset(["page connection"])
    HYBRID_CENTER_STRIP_PX = 20  # pixels of center strip left un-inpainted

    for region in object_regions:
        cls = region.get("class_name", "")
        if cls in _SKIP_OBJECT_CLASSES:
            continue
        bbox = region.get("bbox")
        if not bbox:
            continue
        x_min = int(bbox["x_min"])
        y_min = int(bbox["y_min"])
        x_max = int(bbox["x_max"])
        y_max = int(bbox["y_max"])

        if cls in _HYBRID_CLASSES:
            # Hybrid: only mask the sides, leave center strip for pipe
            bw = x_max - x_min
            bh = y_max - y_min
            if bh > bw:  # Vertical symbol: pipe runs vertically, shrink width
                half_strip = HYBRID_CENTER_STRIP_PX // 2
                cx = (x_min + x_max) // 2
                # Mask left side
                if cx - x_min > half_strip:
                    cv2.rectangle(mask, (x_min, y_min), (cx - half_strip, y_max), 255, -1)
                # Mask right side
                if x_max - cx > half_strip:
                    cv2.rectangle(mask, (cx + half_strip, y_min), (x_max, y_max), 255, -1)
            else:  # Horizontal symbol: pipe runs horizontally, shrink height
                half_strip = HYBRID_CENTER_STRIP_PX // 2
                cy = (y_min + y_max) // 2
                # Mask top side
                if cy - y_min > half_strip:
                    cv2.rectangle(mask, (x_min, y_min), (x_max, cy - half_strip), 255, -1)
                # Mask bottom side
                if y_max - cy > half_strip:
                    cv2.rectangle(mask, (x_min, cy + half_strip), (x_max, y_max), 255, -1)
        else:
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

    # 4. Dilate to close gaps between adjacent corner boxes
    if INPAINT_DILATE_KERNEL[0] > 0 or INPAINT_DILATE_KERNEL[1] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, INPAINT_DILATE_KERNEL)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ═══════════════════════════════════════════════════════════════
# Phase D — Telea inpainting
# ═══════════════════════════════════════════════════════════════


def _inpaint_masked_region(image_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint regions marked 255 in mask; return cleaned image."""
    bin_mask = (mask > 0).astype(np.uint8) * 255
    inpainted = cv2.inpaint(image_gray, bin_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return inpainted


# ═══════════════════════════════════════════════════════════════
# Phase E — Contour extraction on cleaned binary image
# ═══════════════════════════════════════════════════════════════


def _cleaned_to_binary(cleaned_gray: np.ndarray) -> np.ndarray:
    """Run adaptive threshold on the inpainted image."""
    return _adaptive_threshold_mask(cleaned_gray)


def _extract_contour_segments(binary: np.ndarray) -> list[Segment]:
    """
    Extract straight-ish line segments from connected components.

    Optimised approach (vs. original O(n_comps × image) per-component masks):
      1. connectedComponentsWithStats for bounding-rect + area of each component.
      2. For each component, crop to its bounding-box ROI and find contours there.
         This avoids allocating a full-image mask per component.
      3. Douglas-Peucker approximation on each contour → piecewise segments.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    raw_segments: list[Segment] = []

    h, w = binary.shape[:2]
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < MIN_COMPONENT_AREA or area > MAX_COMPONENT_AREA:
            continue

        # Crop to component bounding-box ROI.
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        # Clamp to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        roi_labels = labels[y1:y2, x1:x2]
        roi_mask = ((roi_labels == label_id) & (binary[y1:y2, x1:x2] > 0)).astype(np.uint8) * 255
        if roi_mask.size == 0:
            continue

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for contour in contours:
            if len(contour) < 2:
                continue
            epsilon = 1.5
            approx = cv2.approxPolyDP(contour, epsilon, closed=False)
            pts = approx.reshape(-1, 2)

            if len(pts) < 2:
                continue

            # Offset points back to full-image coordinates.
            ox, oy = x1, y1
            for i in range(len(pts) - 1):
                x1p, y1p = int(pts[i][0]) + ox, int(pts[i][1]) + oy
                x2p, y2p = int(pts[i + 1][0]) + ox, int(pts[i + 1][1]) + oy
                length = math.hypot(x2p - x1p, y2p - y1p)
                if length < 3:
                    continue
                raw_segments.append({
                    "x1": x1p, "y1": y1p, "x2": x2p, "y2": y2p,
                    "length": length,
                    "area_parent": area,
                })

    return raw_segments


# ═══════════════════════════════════════════════════════════════
# Phase F — Collinearity merging + H / V split
# ═══════════════════════════════════════════════════════════════


def _is_diagonal(seg: Segment, angle_threshold: float = 15.0) -> bool:
    """Return True if segment is NOT clearly horizontal or vertical.

    Uses a tight 15° threshold because P&ID pipes are strictly H/V.
    Any segment >15° off-axis is a contour corner artifact that should
    be decomposed into H+V legs.
    """
    ang = _segment_angle_deg(seg)
    if ang <= angle_threshold or ang >= 180 - angle_threshold:
        return False  # horizontal
    if abs(ang - 90.0) <= angle_threshold:
        return False  # vertical
    return True


def _decompose_diagonal_segments(
    segments: list[Segment], *, min_leg_px: float = 8.0
) -> list[Segment]:
    """
    Replace diagonal segments with H+V pairs at the corner point.

    For a diagonal segment (x1,y1)→(x2,y2), decompose into:
      Horizontal leg: (x1,y1)→(x2,y1)
      Vertical leg:   (x2,y1)→(x2,y2)

    Both legs must be at least min_leg_px to be kept.

    This fixes the common case where contour extraction traces
    diagonally across 90° pipe corners instead of following the
    two orthogonal legs separately.
    """
    result: list[Segment] = []
    for seg in segments:
        if not _is_diagonal(seg):
            result.append(seg)
            continue

        dx = abs(seg["x2"] - seg["x1"])
        dy = abs(seg["y2"] - seg["y1"])

        if dx < min_leg_px and dy < min_leg_px:
            continue  # too short to matter

        # Try both corner choices; pick the one with longer combined legs.
        # Corner A: (x1,y2) — H leg to (x1,y1), V leg to (x2,y2)
        h_a = dx
        v_a = dy
        # Corner B: (x2,y1) — H leg to (x2,y2), V leg to (x1,y1)
        h_b = dx
        v_b = dy
        # Same combined length either way; pick based on which leg
        # direction aligns better with existing H/V segments.
        # Simple approach: use corner at (x2, y1).

        if dx >= min_leg_px:
            result.append({
                "x1": seg["x1"], "y1": seg["y1"],
                "x2": seg["x2"], "y2": seg["y1"],
                "length": float(h_a),
                "area_parent": seg.get("area_parent", 0),
                "decomposed": True,
            })
        if dy >= min_leg_px:
            result.append({
                "x1": seg["x2"], "y1": seg["y1"],
                "x2": seg["x2"], "y2": seg["y2"],
                "length": float(v_a),
                "area_parent": seg.get("area_parent", 0),
                "decomposed": True,
            })

    return result


# ═══════════════════════════════════════════════════════════════
# Internal: angle + AABB helpers
# ═══════════════════════════════════════════════════════════════


def _segment_angle_deg(seg: Segment) -> float:
    dx = float(seg["x2"] - seg["x1"])
    dy = float(seg["y2"] - seg["y1"])
    return math.degrees(math.atan2(dy, dx)) % 180.0  # 0-180


def _aabb(seg: Segment) -> tuple[int, int, int, int]:
    x1, x2 = seg["x1"], seg["x2"]
    y1, y2 = seg["y1"], seg["y2"]
    return (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))


def _segments_are_collinear(seg_a: Segment, seg_b: Segment) -> bool:
    """Check collinearity via angle similarity and enough axial overlap (IoU)."""
    ang_a = _segment_angle_deg(seg_a)
    ang_b = _segment_angle_deg(seg_b)
    angle_diff = min(abs(ang_a - ang_b), 180.0 - abs(ang_a - ang_b))
    if angle_diff > COLLINEAR_ANGLE_TOLERANCE_DEG:
        return False

    ax_min, ax_max, ay_min, ay_max = _aabb(seg_a)
    bx_min, bx_max, by_min, by_max = _aabb(seg_b)

    # Quick reject: no overlap in either axis
    if ax_min > bx_max or bx_min > ax_max:
        return False
    if ay_min > by_max or by_min > ay_max:
        return False

    inter_x = max(0, min(ax_max, bx_max) - max(ax_min, bx_min))
    inter_y = max(0, min(ay_max, by_max) - max(ay_min, by_min))
    union_x = max(ax_max, bx_max) - min(ax_min, bx_min)
    union_y = max(ay_max, by_max) - min(ay_min, by_min)

    # Use the dominant axis (larger extent) for IoU.
    # Handles vertical segments where union_x = 0.
    extent_x = max(union_x, inter_x)  # at least inter_x
    extent_y = max(union_y, inter_y)
    if extent_x >= extent_y:
        iou = inter_x / extent_x if extent_x > 0 else 0.0
    else:
        iou = inter_y / extent_y if extent_y > 0 else 0.0
    return iou >= COLLINEAR_IoU_THRESHOLD


def _merge_segment_pair(seg_a: Segment, seg_b: Segment) -> Segment:
    """Merge two collinear segments by taking extreme endpoints."""
    pts = [
        (seg_a["x1"], seg_a["y1"]), (seg_a["x2"], seg_a["y2"]),
        (seg_b["x1"], seg_b["y1"]), (seg_b["x2"], seg_b["y2"]),
    ]
    # Sort along the dominant axis
    vx = float(seg_a["x2"] - seg_a["x1"])
    vy = float(seg_a["y2"] - seg_a["y1"])
    if abs(vx) >= abs(vy):
        pts_sorted = sorted(pts, key=lambda p: p[0])
    else:
        pts_sorted = sorted(pts, key=lambda p: p[1])
    x1, y1 = pts_sorted[0]
    x2, y2 = pts_sorted[-1]
    return {
        "x1": int(round(x1)), "y1": int(round(y1)),
        "x2": int(round(x2)), "y2": int(round(y2)),
        "length": math.hypot(x2 - x1, y2 - y1),
        "area_parent": max(seg_a.get("area_parent", 0), seg_b.get("area_parent", 0)),
    }


# Gap threshold for collinear segments that are close but not touching
# Bridges small gaps (e.g., at page connections, valve symbols)
# Set to 25px to handle gaps from inpainted symbols like page connections
GAP_BRIDGE_PX = 25.0


def _merge_collinear_segments(segments: list[Segment]) -> list[Segment]:
    """
    Bucket + sweep-line collinear merge, with gap bridging.

    Algorithm:
      1. Bucket segments by angle (5° resolution).
      2. Within each bucket: sort by dominant-axis min, then use a sliding
         window to find overlapping segments. Merge connected groups.
      3. Gap bridging: merge non-overlapping but nearby collinear segments
         (within GAP_BRIDGE_PX) to handle pipes interrupted by symbols.
      4. Flatten all buckets.

    Complexity: O(n log n) for sorting + O(n) for the sweep.
    """
    if not segments:
        return []
    BUCKET_SIZE_DEG = 5.0
    NUM_BUCKETS = int(180.0 / BUCKET_SIZE_DEG)
    buckets: list[list[Segment]] = [[] for _ in range(NUM_BUCKETS)]

    for seg in segments:
        ang = _segment_angle_deg(seg)
        bucket_idx = min(int(ang / BUCKET_SIZE_DEG), NUM_BUCKETS - 1)
        buckets[bucket_idx].append(seg)

    result: list[Segment] = []
    for bucket in buckets:
        if not bucket:
            continue

        # Determine dominant axis for this bucket.
        ang0 = _segment_angle_deg(bucket[0])
        dominant_is_x = (ang0 <= 90 - HV_ANGLE_DEG or ang0 >= 90 + HV_ANGLE_DEG)

        # Union-Find to group connected segments.
        parent = list(range(len(bucket)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Sort by dominant-axis min coordinate.
        bucket.sort(
            key=lambda s: min(s["x1"], s["x2"])
            if dominant_is_x else min(s["y1"], s["y2"])
        )

        # Sweep: for each segment, check forward until we're past the overlap+gap window.
        n = len(bucket)
        for i in range(n):
            si = bucket[i]
            si_min = min(si["x1"], si["x2"]) if dominant_is_x else min(si["y1"], si["y2"])
            si_max = max(si["x1"], si["x2"]) if dominant_is_x else max(si["y1"], si["y2"])
            for j in range(i + 1, n):
                sj = bucket[j]
                sj_min = min(sj["x1"], sj["x2"]) if dominant_is_x else min(sj["y1"], sj["y2"])
                if sj_min > si_max + GAP_BRIDGE_PX:
                    break  # Past the overlap+gap window — no more candidates for i.
                if _segments_are_collinear(si, sj):
                    union(i, j)
                elif sj_min > si_max:
                    # Non-overlapping but within gap threshold — check if close enough
                    # on the perpendicular axis to be part of the same pipe.
                    # For vertical segments: check x-alignment; for horizontal: check y.
                    gap = sj_min - si_max
                    if gap <= GAP_BRIDGE_PX:
                        si_aabb = _aabb(si)
                        sj_aabb = _aabb(sj)
                        if dominant_is_x:
                            # Horizontal segments: check y-alignment
                            y_overlap = min(si_aabb[3], sj_aabb[3]) - max(si_aabb[2], sj_aabb[2])
                            if y_overlap >= -5:  # allow 5px misalignment
                                union(i, j)
                        else:
                            # Vertical segments: check x-alignment
                            x_overlap = min(si_aabb[1], sj_aabb[1]) - max(si_aabb[0], sj_aabb[0])
                            if x_overlap >= -5:  # allow 5px misalignment
                                union(i, j)

        # Merge all segments in each connected component.
        components: dict[int, list[Segment]] = {}
        for idx in range(n):
            root = find(idx)
            components.setdefault(root, []).append(bucket[idx])

        for comp_segs in components.values():
            if len(comp_segs) == 1:
                result.append(comp_segs[0])
                continue
            # Merge all segments in the component.
            all_x1 = [s["x1"] for s in comp_segs]
            all_y1 = [s["y1"] for s in comp_segs]
            all_x2 = [s["x2"] for s in comp_segs]
            all_y2 = [s["y2"] for s in comp_segs]
            # Find the two extreme endpoints along the dominant axis.
            if dominant_is_x:
                # Pick the point with min x and the point with max x.
                min_idx = min(range(len(all_x1)), key=lambda k: min(all_x1[k], all_x2[k]))
                max_idx = max(range(len(all_x1)), key=lambda k: max(all_x1[k], all_x2[k]))
                # Choose the actual endpoint that is min/max.
                if all_x1[min_idx] <= all_x2[min_idx]:
                    x1, y1 = all_x1[min_idx], all_y1[min_idx]
                else:
                    x1, y1 = all_x2[min_idx], all_y2[min_idx]
                if all_x1[max_idx] >= all_x2[max_idx]:
                    x2, y2 = all_x1[max_idx], all_y1[max_idx]
                else:
                    x2, y2 = all_x2[max_idx], all_y2[max_idx]
            else:
                min_idx = min(range(len(all_y1)), key=lambda k: min(all_y1[k], all_y2[k]))
                max_idx = max(range(len(all_y1)), key=lambda k: max(all_y1[k], all_y2[k]))
                if all_y1[min_idx] <= all_y2[min_idx]:
                    x1, y1 = all_x1[min_idx], all_y1[min_idx]
                else:
                    x1, y1 = all_x2[min_idx], all_y2[min_idx]
                if all_y1[max_idx] >= all_y2[max_idx]:
                    x2, y2 = all_x1[max_idx], all_y1[max_idx]
                else:
                    x2, y2 = all_x2[max_idx], all_y2[max_idx]

            result.append({
                "x1": int(round(x1)), "y1": int(round(y1)),
                "x2": int(round(x2)), "y2": int(round(y2)),
                "length": math.hypot(x2 - x1, y2 - y1),
                "area_parent": max((s.get("area_parent", 0) for s in comp_segs), default=0),
            })

    return result


def _split_horizontal_vertical(
    segments: list[Segment], *, angle_threshold: float = 25.0
) -> tuple[list[Segment], list[Segment]]:
    """
    Split segments into horizontal (within angle_threshold of 0° or 180°)
    and vertical (within angle_threshold of 90°).
    Default 25° is tighter than HV_ANGLE_DEG (45°) to avoid misclassifying
    slight-angle segments.
    """
    horiz: list[Segment] = []
    vert: list[Segment] = []
    for seg in segments:
        ang = _segment_angle_deg(seg)
        if ang <= angle_threshold or ang >= 180 - angle_threshold:
            horiz.append(seg)
        elif abs(ang - 90.0) <= angle_threshold:
            vert.append(seg)
        else:
            # Truly diagonal — shouldn't happen after decomposition,
            # but classify by dominant axis as fallback.
            if ang < 45 or ang > 135:
                horiz.append(seg)
            else:
                vert.append(seg)
    return horiz, vert


# ═══════════════════════════════════════════════════════════════
# Phase G — Endpoint proximity merge + orphan pruning
# ═══════════════════════════════════════════════════════════════


def _endpoint_distance(s1: Segment, s2: Segment) -> float:
    """Minimum distance between any endpoint pair of two segments."""
    ps1 = [(s1["x1"], s1["y1"]), (s1["x2"], s1["y2"])]
    ps2 = [(s2["x1"], s2["y1"]), (s2["x2"], s2["y2"])]
    dmin = float("inf")
    for p1 in ps1:
        for p2 in ps2:
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if d < dmin:
                dmin = d
    return dmin


def _segments_share_endpoint(s1: Segment, s2: Segment) -> bool:
    """Check if two segments meet at near-right angle."""
    ang1 = _segment_angle_deg(s1)
    ang2 = _segment_angle_deg(s2)
    angle_between = min(abs(ang1 - ang2), 180.0 - abs(ang1 - ang2))
    return abs(angle_between - 90.0) <= ENDPOINT_MERGE_ANGLE_TOLERANCE_DEG


def _merge_endpoint_pair(s1: Segment, s2: Segment) -> Segment | None:
    """
    Merge two perpendicular segments at a shared endpoint.
    Finds the two outermost endpoints and joins them through the junction.
    Returns None if the merged result is not longer than the better input segment.
    """
    pts = [
        (s1["x1"], s1["y1"]), (s1["x2"], s1["y2"]),
        (s2["x1"], s2["y1"]), (s2["x2"], s2["y2"]),
    ]
    d_max = -1.0
    p_a, p_b = pts[0], pts[0]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > d_max:
                d_max = d
                p_a, p_b = pts[i], pts[j]

    new_len = math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])
    if new_len <= max(s1["length"], s2["length"]):
        return None
    return {
        "x1": int(round(p_a[0])), "y1": int(round(p_a[1])),
        "x2": int(round(p_b[0])), "y2": int(round(p_b[1])),
        "length": new_len,
    }


def _merge_nearby_endpoints(segments: list[Segment]) -> list[Segment]:
    """
    Single-pass union-find merge of perpendicular nearby endpoint pairs (L-joints).

    Replaces the 21-iteration greedy loop with a single scan:
      1. Build spatial hash grid (endpoints → cells).
      2. Scan once to find ALL perpendicular endpoint pairs within merge_px.
      3. Only union if segments actually share an endpoint AND the merge
         would produce a longer segment (prevents over-merging parallel
         segments through perpendicular chains).
      4. For each connected component, merge all segments into one by
         taking the two farthest endpoints.

    Rationale: when A(H)+B(V) merge → diagonal at ~45°. A ~45° segment is
    NOT perpendicular to any H or V segment (within 20° tolerance), so the
    cascade naturally stops. All merges are found in the initial scan.

    Complexity: O(n × c) single scan where c = avg entries per cell.
    Previously: O(n × c × iterations) with ~21 iterations.
    """
    if not segments:
        return []

    cell_px = int(ENDPOINT_MERGE_PX)
    if cell_px < 1:
        cell_px = 1
    merge_thresh_sq = ENDPOINT_MERGE_PX ** 2
    angle_tol = ENDPOINT_MERGE_ANGLE_TOLERANCE_DEG
    n = len(segments)

    # Build spatial hash: cell → list of (seg_idx, px, py)
    grid: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    seg_angles: list[float] = [0.0] * n
    for idx, s in enumerate(segments):
        ang = _segment_angle_deg(s)
        seg_angles[idx] = ang
        for px, py in [(s["x1"], s["y1"]), (s["x2"], s["y2"])]:
            cx, cy = px // cell_px, py // cell_px
            grid.setdefault((cx, cy), []).append((idx, px, py))

    # Union-Find to group segments with nearby perpendicular endpoints.
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression (halving)
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Single grid scan to find all valid perpendicular endpoint pairs.
    # Only union if segments share an endpoint AND produce longer segment.
    neighbour_offsets = [(0, 0), (1, 0), (0, 1), (1, 1), (1, -1)]
    for (cx, cy), cell_entries in grid.items():
        for dx, dy in neighbour_offsets:
            nk = (cx + dx, cy + dy)
            other_cell = grid.get(nk, [])
            if not other_cell:
                continue
            for idx_i, px_i, py_i in cell_entries:
                ang_i = seg_angles[idx_i]
                for idx_j, px_j, py_j in other_cell:
                    if idx_j <= idx_i:
                        continue
                    # Quick distance gate
                    dd = (px_i - px_j) ** 2 + (py_i - py_j) ** 2
                    if dd > merge_thresh_sq:
                        continue
                    # Angle gate (perpendicular)
                    ang_j = seg_angles[idx_j]
                    diff = abs(ang_i - ang_j)
                    if diff > 90:
                        diff = 180 - diff
                    if abs(diff - 90) > angle_tol:
                        continue
                    # Must share an endpoint (inline check, avoids recomputing angles).
                    s1 = segments[idx_i]
                    s2 = segments[idx_j]
                    shared = (
                        (s1["x1"] - s2["x1"]) ** 2 + (s1["y1"] - s2["y1"]) ** 2 <= 1
                        or (s1["x1"] - s2["x2"]) ** 2 + (s1["y1"] - s2["y2"]) ** 2 <= 1
                        or (s1["x2"] - s2["x1"]) ** 2 + (s1["y2"] - s2["y1"]) ** 2 <= 1
                        or (s1["x2"] - s2["x2"]) ** 2 + (s1["y2"] - s2["y2"]) ** 2 <= 1
                    )
                    if not shared:
                        continue
                    # Merge must produce a longer segment.
                    # Inline equivalent of _merge_endpoint_pair (avoids math.hypot calls).
                    all_pts = [
                        (s1["x1"], s1["y1"]), (s1["x2"], s1["y2"]),
                        (s2["x1"], s2["y1"]), (s2["x2"], s2["y2"]),
                    ]
                    max_d_sq = 0
                    for pi in range(4):
                        for pj in range(pi + 1, 4):
                            dx = all_pts[pi][0] - all_pts[pj][0]
                            dy = all_pts[pi][1] - all_pts[pj][1]
                            d_sq = dx * dx + dy * dy
                            if d_sq > max_d_sq:
                                max_d_sq = d_sq
                    max_len_sq = max(s1["length"] ** 2, s2["length"] ** 2)
                    if max_d_sq <= max_len_sq:
                        continue
                    union(idx_i, idx_j)

    # Build connected components: root → list of segment indices.
    components: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        components.setdefault(root, []).append(idx)

    # Merge each component: take the two most distant endpoints.
    result: list[Segment] = []
    for indices in components.values():
        if len(indices) == 1:
            result.append(segments[indices[0]])
            continue
        # Collect all endpoints and find the pair with maximum distance.
        all_pts: list[tuple[int, int]] = []
        max_area = 0
        for idx in indices:
            s = segments[idx]
            all_pts.append((s["x1"], s["y1"]))
            all_pts.append((s["x2"], s["y2"]))
            max_area = max(max_area, s.get("area_parent", 0))
        # Find the pair with maximum distance (squared).
        best_d_sq = -1
        best_pair = (all_pts[0], all_pts[1])
        for i in range(len(all_pts)):
            for j in range(i + 1, len(all_pts)):
                dx = all_pts[i][0] - all_pts[j][0]
                dy = all_pts[i][1] - all_pts[j][1]
                d_sq = dx * dx + dy * dy
                if d_sq > best_d_sq:
                    best_d_sq = d_sq
                    best_pair = (all_pts[i], all_pts[j])
        (x1, y1), (x2, y2) = best_pair
        result.append({
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "length": math.hypot(x2 - x1, y2 - y1),
            "area_parent": max_area,
        })

    return result


def _prune_orphan_segments(
    segments: list[Segment], *, min_length: float = MIN_SEGMENT_LENGTH_PX
) -> list[Segment]:
    """Remove segments shorter than min_length — likely noise."""
    return [s for s in segments if s["length"] >= min_length]


# ═══════════════════════════════════════════════════════════════
# Public entrypoint
# ═══════════════════════════════════════════════════════════════


def run_line_detection_inpaint(
    *,
    stage1_gray: np.ndarray,
    text_regions: list[dict[str, Any]],
    object_regions: list[dict[str, Any]],
    image_id: str = "",
) -> dict[str, Any]:
    """
    Run the full Ali-et-al-inspired geometric line-detection pipeline.

    Args:
        stage1_gray: Grayscale P&ID image from Stage 1 normalization.
        text_regions: Stage 2 OCR text regions list with bboxes.
        object_regions: Stage 4 object detection results with bboxes.
        image_id: Identifier for this image/sheet.

    Returns dict with:
        "segments": list of merged segments (each has x1,y1,x2,y2,length)
        "horizontal_segments": subset classified as horizontal
        "vertical_segments": subset classified as vertical
        "inpaint_mask": binary mask used for Telea inpainting
        "cleaned_gray": inpainted grayscale image
        "cleaned_binary": thresholded cleaned image
        "corner_points": Nx2 array of Shi-Tomasi corners (for debug overlay)
        "summary": json-serializable stats
    """
    if stage1_gray.ndim != 2:
        raise ValueError("stage1_gray must be 2D grayscale")
    shape = stage1_gray.shape[:2]

    # Phase B: corner detection
    thresh = _adaptive_threshold_mask(stage1_gray)
    corner_points = _detect_corner_points(thresh)

    # Phase C: assemble inpaint mask (fast grid-hash bboxes)
    inpaint_mask = _assemble_inpaint_mask(shape, corner_points, text_regions, object_regions)

    # Phase D: Telea inpaint
    cleaned_gray = _inpaint_masked_region(stage1_gray, inpaint_mask)

    # Phase E: binary + contour extraction
    cleaned_binary = _cleaned_to_binary(cleaned_gray)
    raw_segments = _extract_contour_segments(cleaned_binary)

    # Phase F: collinear merge + H/V split
    merged = _merge_collinear_segments(raw_segments)

    # Phase Fb: decompose diagonal segments into H+V pairs at corners
    decomposed = _decompose_diagonal_segments(merged)

    # Phase Fc: re-merge collinear after decomposition. Newly created H+V
    # legs from diagonal decomposition may overlap or be near existing
    # H/V pipe segments. A second collinear merge pass connects them.
    remerged = _merge_collinear_segments(decomposed)
    horiz, vert = _split_horizontal_vertical(remerged)

    # Phase G: endpoint merge (L-joints) — SKIPPED after decomposition.
    # The original endpoint merge re-created diagonals by merging
    # perpendicular H+V pairs at shared corners. With diagonal
    # decomposition in Phase Fb, we no longer need this step:
    # H+V segments already meet at their endpoints.
    filtered = _prune_orphan_segments(remerged)
    final_horiz, final_vert = _split_horizontal_vertical(filtered)

    final_horiz, final_vert = _split_horizontal_vertical(filtered)

    return {
        "segments": filtered,
        "horizontal_segments": final_horiz,
        "vertical_segments": final_vert,
        "inpaint_mask": inpaint_mask,
        "cleaned_gray": cleaned_gray,
        "cleaned_binary": cleaned_binary,
        "corner_points": corner_points,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "raw_segments": len(raw_segments),
            "after_collinear_merge": len(merged),
            "after_decompose_diagonal": len(decomposed),
            "final_segments": len(filtered),
            "horizontal_count": len(final_horiz),
            "vertical_count": len(final_vert),
            "corner_points_detected": len(corner_points),
        },
    }


def render_line_overlay(image_bgr: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """
    Draw colorful overlay of detected line segments on BGR image.
    Horizontal lines in yellow, vertical in magenta.
    """
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for seg in segments:
        ang = _segment_angle_deg(seg)
        if ang <= HV_ANGLE_DEG or ang >= 180 - HV_ANGLE_DEG:
            color = (0, 255, 255)   # yellow for horizontal
        else:
            color = (255, 0, 255)   # magenta for vertical
        cv2.line(
            overlay,
            (seg["x1"], seg["y1"]),
            (seg["x2"], seg["y2"]),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(overlay, (seg["x1"], seg["y1"]), 2, (0, 0, 255), -1)
        cv2.circle(overlay, (seg["x2"], seg["y2"]), 2, (0, 0, 255), -1)
    return overlay