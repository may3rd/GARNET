"""
Topology reconstruction pipeline for P&ID pipe networks.

This module implements the canonical flow described in the repository AGENTS:
pipe mask -> skeleton -> node detection -> equipment association ->
node consolidation -> edge tracing -> NetworkX graph construction.

The implementation is intentionally modular so existing project code can be
plugged into individual stages later without rewriting the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from scipy import ndimage as nd
from sklearn.cluster import DBSCAN

from .equipment_pipe_association import (
    AssociationResult,
    CandidateScorer,
    Detection,
    EquipmentPipeAssociatorV2,
    PipeEdge,
    create_attachment_node_payload,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from skimage.morphology import skeletonize
except Exception as exc:  # pragma: no cover
    skeletonize = None
    _SKELETONIZE_IMPORT_ERROR = exc
else:
    _SKELETONIZE_IMPORT_ERROR = None


Point = Tuple[int, int]


@dataclass(frozen=True)
class EquipmentDetection:
    id: str
    category: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyNode:
    id: str
    kind: str
    position: Tuple[float, float]
    pixels: Tuple[Point, ...] = ()
    tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyEdge:
    source: str
    target: str
    polyline: Tuple[Point, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyArtifacts:
    binary_mask: np.ndarray
    skeleton: np.ndarray
    raw_endpoints: Tuple[Point, ...]
    raw_junctions: Tuple[Point, ...]
    consolidated_nodes: Tuple[TopologyNode, ...]
    edges: Tuple[TopologyEdge, ...]
    equipment_associations: Tuple[AssociationResult, ...]
    graph: nx.Graph


@dataclass
class TopologyConfig:
    equipment_snap_threshold: float = 30.0
    node_cluster_eps: float = 6.0
    node_cluster_min_samples: int = 1
    min_edge_length_px: int = 2
    mask_threshold: int = 0
    preserve_mask_values: bool = False
    reject_edge_inside_equipment_bbox: bool = False
    inside_bbox_margin_px: float = 2.0
    candidate_edge_k: int = 10


class TopologyReconstructionPipeline:
    def __init__(self, config: Optional[TopologyConfig] = None) -> None:
        self.config = config or TopologyConfig()

    def run(
        self,
        pipe_mask: np.ndarray,
        equipment: Optional[Sequence[EquipmentDetection]] = None,
    ) -> TopologyArtifacts:
        binary_mask = self._normalize_mask(pipe_mask)
        skeleton = self.extract_skeleton(binary_mask)
        raw_endpoints, raw_junctions = self.detect_nodes(skeleton)
        consolidated_nodes = self.consolidate_nodes(
            raw_endpoints=raw_endpoints,
            raw_junctions=raw_junctions,
        )
        edges = self.trace_edges(skeleton=skeleton, nodes=consolidated_nodes)
        equipment_associations = self.associate_equipment(equipment=equipment or (), edges=edges)
        graph = self.build_graph(
            nodes=consolidated_nodes,
            edges=edges,
            equipment=equipment or (),
            equipment_associations=equipment_associations,
        )
        return TopologyArtifacts(
            binary_mask=binary_mask,
            skeleton=skeleton,
            raw_endpoints=tuple(raw_endpoints),
            raw_junctions=tuple(raw_junctions),
            consolidated_nodes=tuple(consolidated_nodes),
            edges=tuple(edges),
            equipment_associations=tuple(equipment_associations),
            graph=graph,
        )

    def _normalize_mask(self, pipe_mask: np.ndarray) -> np.ndarray:
        if pipe_mask.ndim == 3:
            if cv2 is None:
                pipe_mask = pipe_mask[..., 0]
            else:
                pipe_mask = cv2.cvtColor(pipe_mask, cv2.COLOR_BGR2GRAY)
        if self.config.preserve_mask_values:
            return pipe_mask.copy()
        return (pipe_mask > self.config.mask_threshold).astype(np.uint8)

    def extract_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        if skeletonize is None:  # pragma: no cover
            raise RuntimeError(
                "scikit-image skeletonize is required for topology reconstruction"
            ) from _SKELETONIZE_IMPORT_ERROR
        return skeletonize(binary_mask > 0).astype(np.uint8)

    def detect_nodes(self, skeleton: np.ndarray) -> Tuple[List[Point], List[Point]]:
        kernel = np.array(
            [
                [1, 1, 1],
                [1, 10, 1],
                [1, 1, 1],
            ],
            dtype=np.int32,
        )
        neighbor_count = nd.convolve((skeleton > 0).astype(np.int32), kernel, mode="constant")
        endpoints = np.argwhere(neighbor_count == 11)
        junctions = np.argwhere(neighbor_count >= 13)
        return self._to_points(endpoints), self._to_points(junctions)

    def consolidate_nodes(
        self,
        raw_endpoints: Sequence[Point],
        raw_junctions: Sequence[Point],
    ) -> List[TopologyNode]:
        nodes: List[TopologyNode] = []
        nodes.extend(
            self._cluster_pixel_nodes(
                pixels=raw_endpoints,
                kind="endpoint",
                prefix="endpoint",
            )
        )
        nodes.extend(
            self._cluster_pixel_nodes(
                pixels=raw_junctions,
                kind="junction",
                prefix="junction",
            )
        )
        return nodes

    def associate_equipment(
        self,
        equipment: Sequence[EquipmentDetection],
        edges: Sequence[TopologyEdge],
    ) -> List[AssociationResult]:
        if not equipment or not edges:
            return []

        associator = EquipmentPipeAssociatorV2(
            pipe_edges=[self._to_pipe_edge(edge) for edge in edges],
            scorer=CandidateScorer(
                max_distance_px=self.config.equipment_snap_threshold,
                reject_inside_bbox=self.config.reject_edge_inside_equipment_bbox,
                inside_bbox_margin_px=self.config.inside_bbox_margin_px,
            ),
            k_candidate_edges=self.config.candidate_edge_k,
        )
        return associator.associate_many(
            [self._to_detection(detection) for detection in equipment]
        )

    def trace_edges(
        self,
        skeleton: np.ndarray,
        nodes: Sequence[TopologyNode],
    ) -> List[TopologyEdge]:
        node_pixel_map = self._build_node_pixel_map(nodes)
        visited_transitions: Set[Tuple[Point, Point]] = set()
        edges: List[TopologyEdge] = []

        for node in nodes:
            for start_pixel in node.pixels:
                for neighbor in self._skeleton_neighbors(start_pixel, skeleton):
                    transition = (start_pixel, neighbor)
                    if transition in visited_transitions:
                        continue
                    edge = self._trace_from_pixel(
                        origin_node_id=node.id,
                        start_pixel=start_pixel,
                        next_pixel=neighbor,
                        skeleton=skeleton,
                        node_pixel_map=node_pixel_map,
                        visited_transitions=visited_transitions,
                    )
                    if edge is not None:
                        edges.append(edge)
        return edges

    def _trace_from_pixel(
        self,
        origin_node_id: str,
        start_pixel: Point,
        next_pixel: Point,
        skeleton: np.ndarray,
        node_pixel_map: Dict[Point, str],
        visited_transitions: Set[Tuple[Point, Point]],
    ) -> Optional[TopologyEdge]:
        polyline: List[Point] = [start_pixel]
        previous = start_pixel
        current = next_pixel

        while True:
            visited_transitions.add((previous, current))
            visited_transitions.add((current, previous))
            polyline.append(current)

            target_node_id = node_pixel_map.get(current)
            if target_node_id is not None and target_node_id != origin_node_id:
                if len(polyline) - 1 < self.config.min_edge_length_px:
                    return None
                return TopologyEdge(
                    source=origin_node_id,
                    target=target_node_id,
                    polyline=tuple(polyline),
                )

            candidates = [
                pixel
                for pixel in self._skeleton_neighbors(current, skeleton)
                if pixel != previous
            ]

            if not candidates:
                return None

            if len(candidates) > 1:
                node_at_current = node_pixel_map.get(current)
                if node_at_current is not None and node_at_current == origin_node_id:
                    return None

            previous, current = current, candidates[0]

    def build_graph(
        self,
        nodes: Sequence[TopologyNode],
        edges: Sequence[TopologyEdge],
        equipment: Sequence[EquipmentDetection],
        equipment_associations: Sequence[AssociationResult],
    ) -> nx.Graph:
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(
                node.id,
                type=node.kind,
                position=node.position,
                tag=node.tag,
                pixels=node.pixels,
                **node.metadata,
            )
        for edge in edges:
            if edge.source == edge.target:
                continue
            graph.add_edge(
                edge.source,
                edge.target,
                polyline=edge.polyline,
                **edge.metadata,
            )
        equipment_lookup = {item.id: item for item in equipment}
        for association in equipment_associations:
            if not association.accepted:
                continue
            attachment_payload = create_attachment_node_payload(association)
            equipment_item = equipment_lookup.get(association.det_id)
            attachment_node_id = attachment_payload["node_id"]
            graph.add_node(
                attachment_node_id,
                type=attachment_payload["type"],
                position=attachment_payload["xy"],
                equipment_id=attachment_payload["equipment_id"],
                edge_id=attachment_payload["edge_id"],
                segment_index=attachment_payload["segment_index"],
                t=attachment_payload["t"],
                anchor_name=attachment_payload["anchor_name"],
                equipment_class=association.class_name,
                equipment_bbox=repr(association.bbox),
            )
            if equipment_item is not None:
                equipment_node_id = f"equipment::{equipment_item.id}"
                graph.add_node(
                    equipment_node_id,
                    type=equipment_item.category,
                    position=equipment_item.center,
                    tag=equipment_item.tag,
                    bbox=repr(equipment_item.bbox),
                    **equipment_item.metadata,
                )
                graph.add_edge(
                    equipment_node_id,
                    attachment_node_id,
                    type="equipment_to_attachment",
                )
        return graph

    def export_graphml(self, graph: nx.Graph, output_path: str) -> None:
        graph_copy = nx.Graph()
        for node_id, attrs in graph.nodes(data=True):
            graph_copy.add_node(
                node_id,
                **self._graphml_safe_attrs(attrs),
            )
        for source, target, attrs in graph.edges(data=True):
            graph_copy.add_edge(
                source,
                target,
                **self._graphml_safe_attrs(attrs),
            )
        nx.write_graphml(graph_copy, output_path)

    def _cluster_pixel_nodes(
        self,
        pixels: Sequence[Point],
        kind: str,
        prefix: str,
    ) -> List[TopologyNode]:
        if not pixels:
            return []
        data = np.array(pixels, dtype=float)
        clustering = DBSCAN(
            eps=self.config.node_cluster_eps,
            min_samples=self.config.node_cluster_min_samples,
        ).fit(data)
        clusters: Dict[int, List[Point]] = {}
        for pixel, label in zip(pixels, clustering.labels_):
            clusters.setdefault(int(label), []).append(pixel)

        nodes: List[TopologyNode] = []
        for cluster_idx, cluster_pixels in sorted(clusters.items()):
            coords = np.array(cluster_pixels, dtype=float)
            centroid_row, centroid_col = coords.mean(axis=0)
            nodes.append(
                TopologyNode(
                    id=f"{prefix}_{cluster_idx}",
                    kind=kind,
                    position=(float(centroid_col), float(centroid_row)),
                    pixels=tuple(cluster_pixels),
                )
            )
        return nodes

    def _build_node_pixel_map(self, nodes: Sequence[TopologyNode]) -> Dict[Point, str]:
        mapping: Dict[Point, str] = {}
        for node in nodes:
            for pixel in node.pixels:
                mapping[pixel] = node.id
        return mapping

    def _skeleton_neighbors(self, pixel: Point, skeleton: np.ndarray) -> List[Point]:
        row, col = pixel
        neighbors: List[Point] = []
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

    def _to_points(self, pixels: Iterable[np.ndarray]) -> List[Point]:
        return [(int(row), int(col)) for row, col in pixels]

    def _graphml_safe_attrs(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for key, value in attrs.items():
            if value is None:
                safe[key] = ""
                continue
            if isinstance(value, (tuple, list, dict)):
                safe[key] = repr(value)
            else:
                safe[key] = value
        return safe

    def _to_detection(self, detection: EquipmentDetection) -> Detection:
        x1, y1, x2, y2 = detection.bbox
        return Detection(
            det_id=detection.id,
            class_name=detection.category,
            bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            score=float(detection.metadata.get("score", 1.0)),
            tag=detection.tag,
            metadata=dict(detection.metadata),
        )

    def _to_pipe_edge(self, edge: TopologyEdge) -> PipeEdge:
        return PipeEdge(
            edge_id=f"{edge.source}::{edge.target}::{abs(hash(edge.polyline))}",
            source=edge.source,
            target=edge.target,
            polyline_xy=[(float(col), float(row)) for row, col in edge.polyline],
            metadata=dict(edge.metadata),
        )
