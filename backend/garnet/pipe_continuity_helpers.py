"""
pipe_continuity_helpers.py
Shared continuity-awareness functions used by Stage 10 (edge tracing) and
Stage 12 (graph assembly). Part of Phase 2 of the continuity-aware pipeline.
"""
from __future__ import annotations

import math
from typing import Any


# ─── Tunable constants ──────────────────────────────────────────────────
GAP_THRESHOLD_PX = 20.0       # Rule 8: aligned endpoints this close should connect
MIDSPAN_THRESHOLD_PX = 30.0   # Rule 2: orphan stub detection
SHORT_EDGE_PX = 30.0          # Rule 1: very short edges are likely artifacts
VALIDATED_TERMINAL_ROLES = {"equipment_terminal", "connection_terminal", "junction_terminal"}


# ─── Spatial grid for O(n) gap lookup ───────────────────────────────────
class _SpatialGrid:
    """2D spatial hash grid for O(1) neighbor lookup within a radius."""

    def __init__(self, cell_size: float = 20.0):
        self.cell_size = cell_size
        self.bins: dict[tuple[int, int], list[tuple[float, float, Any]]] = {}

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x / self.cell_size), int(y / self.cell_size))

    def insert(self, x: float, y: float, ref: Any) -> None:
        self.bins.setdefault(self._key(x, y), []).append((x, y, ref))

    def query_radius(self, x: float, y: float, r: float) -> list[tuple[Any, float]]:
        """Return (ref, distance) for all items within r pixels of (x,y)."""
        results: list[tuple[Any, float]] = []
        cell_extent = int(r / self.cell_size) + 1
        bx, by = self._key(x, y)
        for gx in range(bx - cell_extent, bx + cell_extent + 1):
            for gy in range(by - cell_extent, by + cell_extent + 1):
                for px, py, ref in self.bins.get((gx, gy), []):
                    dist = math.hypot(px - x, py - y)
                    if dist <= r:
                        results.append((ref, dist))
        return results


# ─── Point-to-segment projection ────────────────────────────────────────
def project_point_to_segment(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """Project point onto segment [seg_start, seg_end]. Returns (projection, distance)."""
    px, py = point
    ax, ay = seg_start
    bx, by = seg_end
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0.0:
        return seg_start, math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return (proj_x, proj_y), math.hypot(px - proj_x, py - proj_y)


def nearest_point_on_polyline(
    point: tuple[float, float],
    polyline: list[dict[str, float]],
) -> tuple[tuple[float, float], float, int]:
    """Find nearest point on polyline. Returns (nearest_xy, distance_px, segment_index)."""
    best_point: tuple[float, float] | None = None
    best_dist = float("inf")
    best_seg_idx = 0
    for seg_idx in range(len(polyline) - 1):
        a = polyline[seg_idx]
        b = polyline[seg_idx + 1]
        proj, dist = project_point_to_segment(
            point,
            (float(a["col"]), float(a["row"])),
            (float(b["col"]), float(b["row"])),
        )
        if dist < best_dist:
            best_dist = dist
            best_point = proj
            best_seg_idx = seg_idx
    return best_point or point, best_dist, best_seg_idx


# ─── Near-edge detection ────────────────────────────────────────────────
def find_near_edges_at_point(
    point: tuple[float, float],
    all_edges: list[dict[str, Any]],
    exclude_edge_id: str | None = None,
    *,
    threshold_px: float = GAP_THRESHOLD_PX,
) -> list[dict[str, Any]]:
    """Find edges whose polyline is within threshold_px of a point. Returns with distance."""
    candidates: list[dict[str, Any]] = []
    for edge in all_edges:
        edge_id = str(edge.get("id", ""))
        if exclude_edge_id is not None and edge_id == exclude_edge_id:
            continue
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue
        nearest_xy, dist, seg_idx = nearest_point_on_polyline(point, polyline)
        if dist <= threshold_px:
            candidates.append({
                "edge_id": edge_id,
                "nearest_point": {"x": nearest_xy[0], "y": nearest_xy[1]},
                "distance_px": round(dist, 2),
                "segment_index": seg_idx,
            })
    candidates.sort(key=lambda c: c["distance_px"])
    return candidates


def find_midspan_near_edges(
    point: tuple[float, float],
    all_edges: list[dict[str, Any]],
    exclude_edge_id: str | None = None,
    *,
    threshold_px: float = MIDSPAN_THRESHOLD_PX,
) -> list[dict[str, Any]]:
    """Rule 2 orphan stub detection: point near edge MIDSPAN (not near endpoints)."""
    candidates: list[dict[str, Any]] = []
    for edge in all_edges:
        edge_id = str(edge.get("id", ""))
        if exclude_edge_id is not None and edge_id == exclude_edge_id:
            continue
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue
        nearest_xy, dist, _ = nearest_point_on_polyline(point, polyline)
        if dist > threshold_px:
            continue
        start = (float(polyline[0]["col"]), float(polyline[0]["row"]))
        end = (float(polyline[-1]["col"]), float(polyline[-1]["row"]))
        dist_to_start = math.hypot(point[0] - start[0], point[1] - start[1])
        dist_to_end = math.hypot(point[0] - end[0], point[1] - end[1])
        endpoint_threshold = threshold_px * 1.5
        if dist_to_start <= endpoint_threshold or dist_to_end <= endpoint_threshold:
            continue  # normal endpoint connection, not midspan
        candidates.append({
            "edge_id": edge_id,
            "nearest_point": {"x": nearest_xy[0], "y": nearest_xy[1]},
            "distance_px": round(dist, 2),
            "segment_index": 0,
            "type": "midspan",
        })
    candidates.sort(key=lambda c: c["distance_px"])
    return candidates


# ─── Alignment helpers ───────────────────────────────────────────────────
def edge_endpoint_xy(edge: dict[str, Any], which: str = "source") -> tuple[float, float] | None:
    """Get (x, y) of edge source or target endpoint."""
    polyline = edge.get("polyline", [])
    if not polyline:
        return None
    pt = polyline[0] if which == "source" else polyline[-1]
    return float(pt["col"]), float(pt["row"])


def edge_polyline_length(edge: dict[str, Any]) -> float:
    """Compute pixel length of an edge's polyline."""
    polyline = edge.get("polyline", [])
    total = 0.0
    for a, b in zip(polyline, polyline[1:]):
        dx = float(b["col"]) - float(a["col"])
        dy = float(b["row"]) - float(a["row"])
        total += math.hypot(dx, dy)
    return total


# ─── Post-trace edge classification ──────────────────────────────────────
def classify_edge_terminal(
    edge: dict[str, Any],
    all_edges: list[dict[str, Any]],
    *,
    gap_threshold: float = GAP_THRESHOLD_PX,
) -> dict[str, Any]:
    """
    Classify an edge's terminal condition — used during Stage 10 post-trace.
    Returns status: "validated" | "provisional" | "orphan" | "gap_candidate".
    """
    src_xy = edge_endpoint_xy(edge, "source")
    tgt_xy = edge_endpoint_xy(edge, "target")
    src_role = str(edge.get("source_terminal", {}).get("terminal_role", ""))
    dst_role = str(edge.get("destination_terminal", {}).get("terminal_role", ""))
    edge_id = str(edge.get("id", ""))
    seg_len = edge_polyline_length(edge)

    report: dict[str, Any] = {
        "edge_id": edge_id,
        "status": "provisional",
        "reasons": [],
        "near_edge_candidates": [],
        "is_orphan": False,
        "is_gap_candidate": False,
        "is_short_provisional": False,
    }

    src_validated = src_role in VALIDATED_TERMINAL_ROLES
    dst_validated = dst_role in VALIDATED_TERMINAL_ROLES

    if src_validated or dst_validated:
        report["status"] = "validated"
        report["reasons"].append("at_least_one_validated_terminal")

    if seg_len < SHORT_EDGE_PX and not src_validated and not dst_validated:
        report["is_short_provisional"] = True
        report["reasons"].append(f"short_edge_{int(seg_len)}px_both_unresolved")
        if seg_len < 10:
            report["status"] = "provisional"
            report["reasons"].append("likely_tracing_artifact")

    if src_xy is not None:
        near_src = find_near_edges_at_point(src_xy, all_edges, exclude_edge_id=edge_id, threshold_px=gap_threshold)
        if near_src:
            report["near_edge_candidates"].extend(near_src)
            report["is_gap_candidate"] = True
            report["reasons"].append(f"source_near_{len(near_src)}_edge(s)")

    if tgt_xy is not None:
        near_tgt = find_near_edges_at_point(tgt_xy, all_edges, exclude_edge_id=edge_id, threshold_px=gap_threshold)
        if near_tgt:
            report["near_edge_candidates"].extend(near_tgt)
            report["is_gap_candidate"] = True
            report["reasons"].append(f"target_near_{len(near_tgt)}_edge(s)")

    if not src_validated and not dst_validated and not report["near_edge_candidates"]:
        if seg_len < 50:
            report["is_orphan"] = True
            report["status"] = "provisional"
            report["reasons"].append("orphan_stub_both_unresolved_short")

    return report


# ─── Post-trace validation ────────────────────────────────────────────────
def run_post_trace_continuity_check(
    edges: list[dict[str, Any]],
    *,
    gap_threshold: float = GAP_THRESHOLD_PX,
    short_threshold: float = SHORT_EDGE_PX,
) -> dict[str, Any]:
    """
    Run full post-trace continuity validation on all edges.
    Called after _trace_edges completes in Stage 10.
    Adds to each edge: continuity_status, near_edge_candidates, orphan_flag, gap_candidate.
    """
    validated = provisional = orphan_count = gap_candidates = short_provisional = 0
    all_near: list[dict[str, Any]] = []

    for edge in edges:
        report = classify_edge_terminal(edge, edges, gap_threshold=gap_threshold)
        edge["continuity_status"] = report["status"]
        edge["near_edge_candidates"] = report["near_edge_candidates"]
        edge["orphan_flag"] = report["is_orphan"]
        edge["gap_candidate"] = report["is_gap_candidate"]

        if report["status"] == "validated":
            validated += 1
        else:
            provisional += 1
        if report["is_orphan"]:
            orphan_count += 1
        if report["is_gap_candidate"]:
            gap_candidates += 1
        if report["is_short_provisional"]:
            short_provisional += 1
        if report["near_edge_candidates"]:
            all_near.extend(report["near_edge_candidates"])

    # Deduplicate
    seen: set[str] = set()
    unique_near: list[dict[str, Any]] = []
    for cand in all_near:
        eid = cand["edge_id"]
        if eid not in seen:
            seen.add(eid)
            unique_near.append(cand)

    return {
        "total_edges": len(edges),
        "validated_edges": validated,
        "provisional_edges": provisional,
        "orphan_edges": orphan_count,
        "gap_candidate_edges": gap_candidates,
        "short_provisional_edges": short_provisional,
        "unique_near_edge_candidates_count": len(unique_near),
        "near_edge_candidates": unique_near,
    }


# ─── Gap summary using spatial grid ─────────────────────────────────────
def summarize_gaps(
    edges: list[dict[str, Any]],
    threshold_px: float = GAP_THRESHOLD_PX,
) -> list[dict[str, Any]]:
    """
    O(n) gap detection using spatial grid. Finds pairs of edges whose
    endpoints are aligned and within threshold but NOT already connected.
    """
    if not edges:
        return []

    # Build endpoint index
    endpoint_index: list[dict[str, Any]] = []
    for edge in edges:
        eid = str(edge.get("id", ""))
        polyline = edge.get("polyline", [])
        if not polyline:
            continue
        endpoint_index.append({
            "edge_id": eid,
            "source": (float(polyline[0]["col"]), float(polyline[0]["row"])),
            "target": (float(polyline[-1]["col"]), float(polyline[-1]["row"])),
        })

    if not endpoint_index:
        return []

    # Spatial grid
    grid = _SpatialGrid(cell_size=threshold_px)
    for ep in endpoint_index:
        grid.insert(ep["source"][0], ep["source"][1], ("src", ep))
        grid.insert(ep["target"][0], ep["target"][1], ("dst", ep))

    gaps: list[dict[str, Any]] = []
    checked: set[tuple[str, str]] = set()
    search_radius = threshold_px * 1.5

    for ep in endpoint_index:
        eid_a = ep["edge_id"]
        for pt_label, pt_a in [("source", ep["source"]), ("target", ep["target"])]:
            candidates = grid.query_radius(pt_a[0], pt_a[1], search_radius)
            for ref_dist, dist in candidates:
                label, ep_b = ref_dist
                # Skip same edge or same endpoint type
                if label == pt_label and ep_b["edge_id"] == eid_a:
                    continue
                # Skip same edge or same endpoint type
                if label == pt_label and ep_b["edge_id"] == eid_a:
                    continue

                eid_b = ep_b["edge_id"]
                if eid_a >= eid_b:
                    continue
                pair = (eid_a, eid_b)
                if pair in checked:
                    continue

                pt_b = ep_b["target"] if pt_label == "source" else ep_b["source"]
                dx = abs(pt_a[0] - pt_b[0])
                dy = abs(pt_a[1] - pt_b[1])

                if (dx <= threshold_px or dy <= threshold_px):
                    gap_dist = math.hypot(dx, dy)
                    if gap_dist <= threshold_px * 1.5:
                        alignment = "horizontal" if dx <= threshold_px else "vertical"
                        mid_x = (pt_a[0] + pt_b[0]) / 2
                        mid_y = (pt_a[1] + pt_b[1]) / 2
                        checked.add(pair)
                        gaps.append({
                            "edge_a": eid_a, "edge_b": eid_b,
                            "endpoint_a": pt_label,
                            "endpoint_b": "source" if pt_label == "target" else "target",
                            "gap_position": {"x": round(mid_x, 1), "y": round(mid_y, 1)},
                            "gap_distance_px": round(gap_dist, 2),
                            "alignment": alignment,
                            "edge_a_endpoint": pt_a,
                            "edge_b_endpoint": pt_b,
                        })

    return gaps