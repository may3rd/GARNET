from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from garnet.topology_pipeline import EquipmentDetection, TopologyConfig, TopologyReconstructionPipeline


Color = Tuple[int, int, int]

WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
RED: Color = (0, 0, 255)
GREEN: Color = (0, 200, 0)
BLUE: Color = (255, 0, 0)
ORANGE: Color = (0, 165, 255)
MAGENTA: Color = (255, 0, 255)
CYAN: Color = (255, 255, 0)


def build_synthetic_case() -> tuple[np.ndarray, list[EquipmentDetection]]:
    mask = np.zeros((420, 640), dtype=np.uint8)

    cv2.line(mask, (60, 210), (580, 210), 255, 9)
    cv2.line(mask, (220, 70), (220, 340), 255, 9)
    cv2.line(mask, (420, 210), (520, 120), 255, 9)
    cv2.line(mask, (420, 210), (520, 300), 255, 9)

    detections = [
        EquipmentDetection(
            id="P101",
            category="pump",
            bbox=(80, 165, 150, 245),
            center=(115, 205),
            tag="P-101",
            metadata={"score": 0.96},
        ),
        EquipmentDetection(
            id="V201",
            category="valve",
            bbox=(250, 180, 320, 240),
            center=(285, 210),
            tag="V-201",
            metadata={"score": 0.93},
        ),
        EquipmentDetection(
            id="TK301",
            category="tank",
            bbox=(520, 70, 610, 165),
            center=(565, 118),
            tag="TK-301",
            metadata={"score": 0.97},
        ),
        EquipmentDetection(
            id="C401",
            category="compressor",
            bbox=(520, 255, 610, 345),
            center=(565, 300),
            tag="C-401",
            metadata={"score": 0.92},
        ),
    ]
    return mask, detections


def to_bgr(binary_or_bool: np.ndarray) -> np.ndarray:
    image = (binary_or_bool > 0).astype(np.uint8) * 255
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def draw_equipment(canvas: np.ndarray, detections: Iterable[EquipmentDetection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), ORANGE, 2)
        cv2.circle(canvas, (int(det.center[0]), int(det.center[1])), 4, ORANGE, -1)
        label = f"{det.id}:{det.category}"
        cv2.putText(canvas, label, (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ORANGE, 1, cv2.LINE_AA)


def draw_nodes_and_edges(
    canvas: np.ndarray,
    artifacts,
) -> None:
    for edge in artifacts.edges:
        xy_polyline = [(int(col), int(row)) for row, col in edge.polyline]
        for start, end in zip(xy_polyline, xy_polyline[1:]):
            cv2.line(canvas, start, end, CYAN, 2)

    for node in artifacts.consolidated_nodes:
        x = int(round(node.position[0]))
        y = int(round(node.position[1]))
        color = RED if node.kind == "junction" else GREEN
        radius = 6 if node.kind == "junction" else 5
        cv2.circle(canvas, (x, y), radius, color, -1)
        cv2.putText(canvas, node.id, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


def draw_raw_node_pixels(canvas: np.ndarray, artifacts) -> None:
    for row, col in artifacts.raw_endpoints:
        cv2.circle(canvas, (int(col), int(row)), 2, GREEN, -1)
    for row, col in artifacts.raw_junctions:
        cv2.circle(canvas, (int(col), int(row)), 2, RED, -1)


def draw_associations(
    canvas: np.ndarray,
    detections: Iterable[EquipmentDetection],
    artifacts,
) -> None:
    draw_equipment(canvas, detections)
    draw_nodes_and_edges(canvas, artifacts)

    for result in artifacts.equipment_associations:
        if not result.accepted or result.anchor_xy is None or result.nearest_point_xy is None:
            continue
        anchor_xy = (int(round(result.anchor_xy[0])), int(round(result.anchor_xy[1])))
        attach_xy = (int(round(result.nearest_point_xy[0])), int(round(result.nearest_point_xy[1])))
        cv2.circle(canvas, anchor_xy, 5, MAGENTA, -1)
        cv2.circle(canvas, attach_xy, 6, BLUE, -1)
        cv2.line(canvas, anchor_xy, attach_xy, MAGENTA, 2)
        label = f"{result.det_id}->{result.edge_id} ({result.distance_px:.1f}px)"
        cv2.putText(
            canvas,
            label,
            (attach_xy[0] + 8, attach_xy[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            MAGENTA,
            1,
            cv2.LINE_AA,
        )


def save_outputs(output_dir: Path, detections: list[EquipmentDetection], artifacts) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_bgr = to_bgr(artifacts.binary_mask)
    skeleton_bgr = to_bgr(artifacts.skeleton)
    raw_nodes_bgr = skeleton_bgr.copy()
    draw_raw_node_pixels(raw_nodes_bgr, artifacts)

    nodes_edges_bgr = mask_bgr.copy()
    draw_nodes_and_edges(nodes_edges_bgr, artifacts)
    draw_equipment(nodes_edges_bgr, detections)

    node_clusters_bgr = skeleton_bgr.copy()
    draw_nodes_and_edges(node_clusters_bgr, artifacts)

    equipment_bgr = mask_bgr.copy()
    draw_equipment(equipment_bgr, detections)

    association_bgr = mask_bgr.copy()
    draw_associations(association_bgr, detections, artifacts)

    cv2.imwrite(str(output_dir / "topology_step_01_mask.png"), mask_bgr)
    cv2.imwrite(str(output_dir / "topology_step_02_skeleton.png"), skeleton_bgr)
    cv2.imwrite(str(output_dir / "topology_step_03_raw_nodes.png"), raw_nodes_bgr)
    cv2.imwrite(str(output_dir / "topology_step_04_equipment.png"), equipment_bgr)
    cv2.imwrite(str(output_dir / "topology_step_05_consolidated_nodes.png"), node_clusters_bgr)
    cv2.imwrite(str(output_dir / "topology_step_06_traced_edges.png"), nodes_edges_bgr)
    cv2.imwrite(str(output_dir / "topology_step_07_associations.png"), association_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual test harness for the topology reconstruction pipeline.")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "output"),
        help="Output directory for generated overlays and graph artifacts.",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    mask, detections = build_synthetic_case()

    pipeline = TopologyReconstructionPipeline(
        TopologyConfig(
            equipment_snap_threshold=90.0,
            candidate_edge_k=8,
            node_cluster_eps=6.0,
            min_edge_length_px=2,
        )
    )
    artifacts = pipeline.run(mask, detections)

    save_outputs(output_dir, detections, artifacts)
    pipeline.export_graphml(artifacts.graph, str(output_dir / "topology_step_08_graph.graphml"))

    print(f"Output directory: {output_dir}")
    print(f"Detected nodes: {len(artifacts.consolidated_nodes)}")
    print(f"Traced edges: {len(artifacts.edges)}")
    accepted = [result for result in artifacts.equipment_associations if result.accepted]
    print(f"Accepted equipment attachments: {len(accepted)} / {len(artifacts.equipment_associations)}")
    for result in artifacts.equipment_associations:
        print(
            f"- {result.det_id}: accepted={result.accepted}, "
            f"edge={result.edge_id}, distance_px={result.distance_px}, reason={result.reason}"
        )


if __name__ == "__main__":
    main()
