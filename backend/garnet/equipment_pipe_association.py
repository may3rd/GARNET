from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import KDTree


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    det_id: str
    class_name: str
    bbox: BBox
    score: float = 1.0
    tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center_xy(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.bbox
        return float(max(0, x2 - x1))

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.bbox
        return float(max(0, y2 - y1))


@dataclass
class PipeEdge:
    edge_id: str
    source: str
    target: str
    polyline_xy: List[Point]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnchorPoint:
    name: str
    xy: Point
    weight: float = 1.0


@dataclass
class CandidateAssociation:
    det_id: str
    edge_id: str
    anchor_name: str
    anchor_xy: Point
    nearest_point_xy: Point
    distance_px: float
    score: float
    segment_index: int
    t: float
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssociationResult:
    det_id: str
    class_name: str
    bbox: BBox
    accepted: bool
    reason: str
    anchor_name: Optional[str] = None
    anchor_xy: Optional[Point] = None
    edge_id: Optional[str] = None
    nearest_point_xy: Optional[Point] = None
    distance_px: Optional[float] = None
    score: Optional[float] = None
    segment_index: Optional[int] = None
    t: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_bbox(point_xy: Point, bbox: BBox, margin: float = 0.0) -> bool:
    px, py = point_xy
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin)


def bbox_side_midpoints(bbox: BBox) -> Dict[str, Point]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return {
        "left": (x1, cy),
        "right": (x2, cy),
        "top": (cx, y1),
        "bottom": (cx, y2),
        "center": (cx, cy),
    }


def project_point_to_segment(point_xy: Point, a_xy: Point, b_xy: Point) -> Tuple[Point, float, float]:
    px, py = point_xy
    ax, ay = a_xy
    bx, by = b_xy

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        proj = a_xy
        return proj, 0.0, euclidean_distance(point_xy, proj)

    t = (apx * abx + apy * aby) / ab_len_sq
    t = clamp(t, 0.0, 1.0)

    proj = (ax + t * abx, ay + t * aby)
    dist = euclidean_distance(point_xy, proj)
    return proj, t, dist


def segment_direction(a_xy: Point, b_xy: Point) -> Point:
    dx = b_xy[0] - a_xy[0]
    dy = b_xy[1] - a_xy[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


def cosine_alignment(vec_a: Point, vec_b: Point) -> float:
    ax, ay = vec_a
    bx, by = vec_b
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na == 0 or nb == 0:
        return 0.0
    return (ax * bx + ay * by) / (na * nb)


class AnchorGenerator:
    def generate(self, det: Detection) -> List[AnchorPoint]:
        mids = bbox_side_midpoints(det.bbox)
        cls = det.class_name.lower()

        if cls in {"valve", "control_valve", "check_valve", "flowmeter", "inline_instrument"}:
            return [
                AnchorPoint("left", mids["left"], weight=1.25),
                AnchorPoint("right", mids["right"], weight=1.25),
                AnchorPoint("center", mids["center"], weight=0.60),
                AnchorPoint("top", mids["top"], weight=0.30),
                AnchorPoint("bottom", mids["bottom"], weight=0.30),
            ]

        if cls in {"vessel", "column", "tank", "separator", "drum"}:
            return [
                AnchorPoint("left", mids["left"], weight=1.00),
                AnchorPoint("right", mids["right"], weight=1.00),
                AnchorPoint("top", mids["top"], weight=1.00),
                AnchorPoint("bottom", mids["bottom"], weight=1.00),
                AnchorPoint("center", mids["center"], weight=0.25),
            ]

        if cls in {"pump", "compressor", "blower", "fan"}:
            return [
                AnchorPoint("left", mids["left"], weight=1.20),
                AnchorPoint("right", mids["right"], weight=1.20),
                AnchorPoint("top", mids["top"], weight=0.50),
                AnchorPoint("bottom", mids["bottom"], weight=0.50),
                AnchorPoint("center", mids["center"], weight=0.40),
            ]

        return [
            AnchorPoint("left", mids["left"], weight=1.00),
            AnchorPoint("right", mids["right"], weight=1.00),
            AnchorPoint("top", mids["top"], weight=1.00),
            AnchorPoint("bottom", mids["bottom"], weight=1.00),
            AnchorPoint("center", mids["center"], weight=0.50),
        ]


class PipeEdgeIndex:
    def __init__(self, edges: Sequence[PipeEdge]) -> None:
        self.edges = list(edges)
        self.edge_map = {e.edge_id: e for e in self.edges}

        vertex_rows = []
        vertex_to_edge_ids: List[str] = []

        for edge in self.edges:
            for pt in edge.polyline_xy:
                vertex_rows.append([pt[0], pt[1]])
                vertex_to_edge_ids.append(edge.edge_id)

        if not vertex_rows:
            raise ValueError("No pipe polyline vertices found")

        self.vertex_points_xy = np.asarray(vertex_rows, dtype=float)
        self.vertex_to_edge_ids = vertex_to_edge_ids
        self.tree = KDTree(self.vertex_points_xy)

    def query_candidate_edges(self, point_xy: Point, k_vertices: int = 12) -> List[str]:
        k_vertices = min(k_vertices, len(self.vertex_points_xy))
        _, indices = self.tree.query([point_xy[0], point_xy[1]], k=k_vertices)

        if np.isscalar(indices):
            indices = [int(indices)]
        else:
            indices = [int(i) for i in indices]

        edge_ids = []
        seen = set()
        for idx in indices:
            edge_id = self.vertex_to_edge_ids[idx]
            if edge_id not in seen:
                seen.add(edge_id)
                edge_ids.append(edge_id)
        return edge_ids

    def nearest_point_on_edge(self, edge: PipeEdge, point_xy: Point) -> Tuple[Point, int, float, float]:
        poly = edge.polyline_xy
        if len(poly) == 1:
            return poly[0], 0, 0.0, euclidean_distance(point_xy, poly[0])

        best_proj = None
        best_segment_index = None
        best_t = None
        best_dist = float("inf")

        for i in range(len(poly) - 1):
            proj, t, dist = project_point_to_segment(point_xy, poly[i], poly[i + 1])
            if dist < best_dist:
                best_proj = proj
                best_segment_index = i
                best_t = t
                best_dist = dist

        assert best_proj is not None
        assert best_segment_index is not None
        assert best_t is not None

        return best_proj, best_segment_index, best_t, best_dist


class CandidateScorer:
    def __init__(
        self,
        max_distance_px: float = 60.0,
        reject_inside_bbox: bool = False,
        inside_bbox_margin_px: float = 2.0,
    ) -> None:
        self.max_distance_px = float(max_distance_px)
        self.reject_inside_bbox = bool(reject_inside_bbox)
        self.inside_bbox_margin_px = float(inside_bbox_margin_px)

    def score_candidate(
        self,
        det: Detection,
        anchor: AnchorPoint,
        edge: PipeEdge,
        nearest_point_xy: Point,
        segment_index: int,
        t: float,
        distance_px: float,
    ) -> Tuple[float, Dict[str, Any], Optional[str]]:
        if distance_px > self.max_distance_px:
            return (
                -1e9,
                {"distance_px": distance_px},
                f"distance too large ({distance_px:.2f}px > {self.max_distance_px:.2f}px)",
            )

        if self.reject_inside_bbox and point_in_bbox(nearest_point_xy, det.bbox, self.inside_bbox_margin_px):
            return (
                -1e9,
                {"distance_px": distance_px},
                "nearest edge point lies inside equipment bbox",
            )

        distance_norm = distance_px / max(self.max_distance_px, 1e-6)
        distance_score = 1.0 - distance_norm

        poly = edge.polyline_xy
        seg_dir = segment_direction(poly[segment_index], poly[min(segment_index + 1, len(poly) - 1)])
        conn_vec = (nearest_point_xy[0] - anchor.xy[0], nearest_point_xy[1] - anchor.xy[1])
        parallelness = abs(cosine_alignment(seg_dir, conn_vec))
        orthogonality_score = 1.0 - parallelness

        center_dist = euclidean_distance(det.center_xy, anchor.xy)
        size_ref = max(det.width, det.height, 1.0)
        boundary_bias = clamp(center_dist / size_ref, 0.0, 1.0)

        total = (
            2.0 * distance_score
            + 1.5 * anchor.weight
            + 0.5 * orthogonality_score
            + 0.3 * boundary_bias
        )

        debug = {
            "distance_px": distance_px,
            "distance_score": distance_score,
            "anchor_weight": anchor.weight,
            "orthogonality_score": orthogonality_score,
            "boundary_bias": boundary_bias,
            "segment_index": segment_index,
            "t": t,
        }
        return total, debug, None


class EquipmentPipeAssociatorV2:
    def __init__(
        self,
        pipe_edges: Sequence[PipeEdge],
        anchor_generator: Optional[AnchorGenerator] = None,
        scorer: Optional[CandidateScorer] = None,
        k_candidate_edges: int = 10,
        max_results_per_det: int = 1,
    ) -> None:
        self.pipe_edges = list(pipe_edges)
        self.edge_index = PipeEdgeIndex(self.pipe_edges)
        self.anchor_generator = anchor_generator or AnchorGenerator()
        self.scorer = scorer or CandidateScorer()
        self.k_candidate_edges = int(k_candidate_edges)
        self.max_results_per_det = int(max_results_per_det)

    def associate_one(self, det: Detection) -> AssociationResult:
        anchors = self.anchor_generator.generate(det)
        all_candidates: List[CandidateAssociation] = []
        rejection_reasons = []

        for anchor in anchors:
            candidate_edge_ids = self.edge_index.query_candidate_edges(
                anchor.xy,
                k_vertices=self.k_candidate_edges,
            )

            for edge_id in candidate_edge_ids:
                edge = self.edge_index.edge_map[edge_id]
                nearest_xy, seg_idx, t, dist = self.edge_index.nearest_point_on_edge(edge, anchor.xy)
                score, debug, rejection_reason = self.scorer.score_candidate(
                    det=det,
                    anchor=anchor,
                    edge=edge,
                    nearest_point_xy=nearest_xy,
                    segment_index=seg_idx,
                    t=t,
                    distance_px=dist,
                )
                if rejection_reason is not None:
                    rejection_reasons.append(
                        {
                            "anchor_name": anchor.name,
                            "edge_id": edge_id,
                            "reason": rejection_reason,
                            "distance_px": dist,
                        }
                    )
                    continue

                all_candidates.append(
                    CandidateAssociation(
                        det_id=det.det_id,
                        edge_id=edge_id,
                        anchor_name=anchor.name,
                        anchor_xy=anchor.xy,
                        nearest_point_xy=nearest_xy,
                        distance_px=dist,
                        score=score,
                        segment_index=seg_idx,
                        t=t,
                        debug=debug,
                    )
                )

        if not all_candidates:
            return AssociationResult(
                det_id=det.det_id,
                class_name=det.class_name,
                bbox=det.bbox,
                accepted=False,
                reason="no valid pipe candidate found",
                debug={"rejections": rejection_reasons},
            )

        all_candidates.sort(key=lambda c: c.score, reverse=True)
        best = all_candidates[0]
        return AssociationResult(
            det_id=det.det_id,
            class_name=det.class_name,
            bbox=det.bbox,
            accepted=True,
            reason="best candidate selected",
            anchor_name=best.anchor_name,
            anchor_xy=best.anchor_xy,
            edge_id=best.edge_id,
            nearest_point_xy=best.nearest_point_xy,
            distance_px=best.distance_px,
            score=best.score,
            segment_index=best.segment_index,
            t=best.t,
            debug={
                "best_candidate": best.debug,
                "num_candidates": len(all_candidates),
                "top_candidates": [
                    {
                        "edge_id": c.edge_id,
                        "anchor_name": c.anchor_name,
                        "distance_px": c.distance_px,
                        "score": c.score,
                    }
                    for c in all_candidates[:5]
                ],
            },
        )

    def associate_many(self, detections: Sequence[Detection]) -> List[AssociationResult]:
        return [self.associate_one(det) for det in detections]


def create_attachment_node_payload(result: AssociationResult) -> Dict[str, Any]:
    if not result.accepted:
        raise ValueError("Cannot create attachment node for rejected association")

    return {
        "node_id": f"attach::{result.det_id}",
        "type": "equipment_attachment",
        "equipment_id": result.det_id,
        "edge_id": result.edge_id,
        "xy": result.nearest_point_xy,
        "segment_index": result.segment_index,
        "t": result.t,
        "anchor_name": result.anchor_name,
    }


def group_associations_by_edge(results: Sequence[AssociationResult]) -> Dict[str, List[AssociationResult]]:
    grouped: Dict[str, List[AssociationResult]] = defaultdict(list)
    for result in results:
        if result.accepted and result.edge_id is not None:
            grouped[result.edge_id].append(result)
    return dict(grouped)
