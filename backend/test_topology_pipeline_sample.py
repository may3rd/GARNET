from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Tuple

import cv2
import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

from garnet.topology_pipeline import EquipmentDetection, TopologyConfig, TopologyReconstructionPipeline


Color = Tuple[int, int, int]

RED: Color = (0, 0, 255)
GREEN: Color = (0, 200, 0)
BLUE: Color = (255, 0, 0)
ORANGE: Color = (0, 165, 255)
MAGENTA: Color = (255, 0, 255)
CYAN: Color = (255, 255, 0)
YELLOW: Color = (0, 255, 255)


TEXT_LIKE_CLASSES = {
    "instrument dcs",
    "instrument logic",
    "instrument tag",
    "line number",
    "trip function",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def coco_bbox_to_xyxy(bbox_xywh) -> tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def to_bgr(binary_or_bool: np.ndarray) -> np.ndarray:
    image = (binary_or_bool > 0).astype(np.uint8) * 255
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def load_equipment_detections(coco_path: Path, min_score: float = 0.25) -> list[EquipmentDetection]:
    data = load_json(coco_path)
    detections: list[EquipmentDetection] = []
    for idx, ann in enumerate(data.get("annotations", [])):
        class_name = ann.get("category_name", "").strip().lower()
        score = float(ann.get("score", 1.0))
        if score < min_score:
            continue
        if class_name in TEXT_LIKE_CLASSES:
            continue
        bbox = coco_bbox_to_xyxy(ann["bbox"])
        x1, y1, x2, y2 = bbox
        detections.append(
            EquipmentDetection(
                id=f"det_{idx}",
                category=class_name.replace(" ", "_"),
                bbox=bbox,
                center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                tag=ann.get("category_name"),
                metadata={
                    "score": score,
                    "category_id": ann.get("category_id"),
                    "source": "coco_annotations",
                },
            )
        )
    return detections


def draw_boxes(canvas: np.ndarray, boxes: Iterable[tuple[int, int, int, int]], color: Color, thickness: int) -> None:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)


def build_pipe_mask(
    image_bgr: np.ndarray,
    coco_path: Path,
    arrows_path: Path,
    ocr_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    binary_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )

    masked = binary_inv.copy()

    coco_data = load_json(coco_path)
    arrow_data = load_json(arrows_path)
    ocr_data = load_json(ocr_path)

    text_boxes = []
    symbol_boxes = []

    for ann in coco_data.get("annotations", []):
        bbox = coco_bbox_to_xyxy(ann["bbox"])
        cls = ann.get("category_name", "").strip().lower()
        if cls in TEXT_LIKE_CLASSES:
            text_boxes.append(bbox)
        else:
            symbol_boxes.append(bbox)

    for ann in arrow_data.get("annotations", []):
        symbol_boxes.append(coco_bbox_to_xyxy(ann["bbox"]))

    for ann in ocr_data.get("annotations", []):
        text_boxes.append(coco_bbox_to_xyxy(ann["bbox"]))

    for x1, y1, x2, y2 in text_boxes:
        cv2.rectangle(masked, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 0, -1)

    for x1, y1, x2, y2 in symbol_boxes:
        cv2.rectangle(masked, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), 0, -1)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    line_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))

    cleaned = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel_h)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_v)

    # Preserve thin pipe strokes that a generic denoise/open step would erase.
    thin_horizontal = cv2.morphologyEx(masked, cv2.MORPH_OPEN, line_kernel_h)
    thin_vertical = cv2.morphologyEx(masked, cv2.MORPH_OPEN, line_kernel_v)
    cleaned = cv2.bitwise_or(cleaned, thin_horizontal)
    cleaned = cv2.bitwise_or(cleaned, thin_vertical)
    cleaned = cv2.bitwise_or(cleaned, masked)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= 8:
            filtered[labels == label_idx] = 255

    return binary_inv, masked, cleaned, filtered


def draw_equipment(canvas: np.ndarray, detections: Iterable[EquipmentDetection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), ORANGE, 2)
        cv2.circle(canvas, (int(det.center[0]), int(det.center[1])), 3, ORANGE, -1)
        cv2.putText(
            canvas,
            det.category,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            ORANGE,
            1,
            cv2.LINE_AA,
        )


def draw_pipeline(canvas: np.ndarray, artifacts) -> None:
    for edge in artifacts.edges:
        xy_polyline = [(int(col), int(row)) for row, col in edge.polyline]
        for start, end in zip(xy_polyline, xy_polyline[1:]):
            cv2.line(canvas, start, end, CYAN, 2)

    for node in artifacts.consolidated_nodes:
        x = int(round(node.position[0]))
        y = int(round(node.position[1]))
        color = RED if node.kind == "junction" else GREEN
        cv2.circle(canvas, (x, y), 5, color, -1)


def draw_raw_nodes(canvas: np.ndarray, artifacts) -> None:
    for row, col in artifacts.raw_endpoints:
        cv2.circle(canvas, (int(col), int(row)), 1, GREEN, -1)
    for row, col in artifacts.raw_junctions:
        cv2.circle(canvas, (int(col), int(row)), 1, RED, -1)


def draw_associations(canvas: np.ndarray, artifacts) -> None:
    for result in artifacts.equipment_associations:
        if not result.accepted or result.anchor_xy is None or result.nearest_point_xy is None:
            continue
        anchor_xy = (int(round(result.anchor_xy[0])), int(round(result.anchor_xy[1])))
        nearest_xy = (int(round(result.nearest_point_xy[0])), int(round(result.nearest_point_xy[1])))
        cv2.circle(canvas, anchor_xy, 4, MAGENTA, -1)
        cv2.circle(canvas, nearest_xy, 5, BLUE, -1)
        cv2.line(canvas, anchor_xy, nearest_xy, MAGENTA, 2)


def save_outputs(
    out_dir: Path,
    image_bgr: np.ndarray,
    binary_inv: np.ndarray,
    masked: np.ndarray,
    cleaned: np.ndarray,
    filtered: np.ndarray,
    detections: list[EquipmentDetection],
    artifacts,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    original_vis = image_bgr.copy()
    draw_equipment(original_vis, detections)

    threshold_vis = to_bgr(binary_inv)
    masked_vis = to_bgr(masked)
    cleaned_vis = to_bgr(cleaned)
    filtered_vis = to_bgr(filtered)
    skeleton_vis = to_bgr(artifacts.skeleton)

    raw_nodes_vis = skeleton_vis.copy()
    draw_raw_nodes(raw_nodes_vis, artifacts)

    graph_vis = image_bgr.copy()
    draw_pipeline(graph_vis, artifacts)
    draw_equipment(graph_vis, detections)

    assoc_vis = graph_vis.copy()
    draw_associations(assoc_vis, artifacts)

    cv2.imwrite(str(out_dir / "sample_topology_step_01_original.png"), original_vis)
    cv2.imwrite(str(out_dir / "sample_topology_step_02_threshold.png"), threshold_vis)
    cv2.imwrite(str(out_dir / "sample_topology_step_03_masked_text_symbols.png"), masked_vis)
    cv2.imwrite(str(out_dir / "sample_topology_step_04_cleaned.png"), cleaned_vis)
    cv2.imwrite(str(out_dir / "sample_topology_step_05_pipe_mask.png"), filtered_vis)
    cv2.imwrite(str(out_dir / "sample_topology_step_06_skeleton.png"), skeleton_vis)
    cv2.imwrite(str(out_dir / "sample_topology_step_07_raw_nodes.png"), raw_nodes_vis)
    cv2.imwrite(str(out_dir / "sample_topology_step_08_graph.png"), graph_vis)
    cv2.imwrite(str(out_dir / "sample_topology_step_09_associations.png"), assoc_vis)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topology reconstruction test on backend/sample.png.")
    parser.add_argument("--image", default=str(BACKEND_ROOT / "sample.png"))
    parser.add_argument("--coco", default=str(BACKEND_ROOT / "coco_annotations.json"))
    parser.add_argument("--arrows", default=str(BACKEND_ROOT / "coco_arrows.json"))
    parser.add_argument("--ocr", default=str(BACKEND_ROOT / "ocr_results.json"))
    parser.add_argument("--out", default=str(BACKEND_ROOT / "output"))
    args = parser.parse_args()

    image_path = Path(args.image)
    coco_path = Path(args.coco)
    arrows_path = Path(args.arrows)
    ocr_path = Path(args.ocr)
    out_dir = Path(args.out)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    detections = load_equipment_detections(coco_path)
    binary_inv, masked, cleaned, pipe_mask = build_pipe_mask(
        image_bgr=image_bgr,
        coco_path=coco_path,
        arrows_path=arrows_path,
        ocr_path=ocr_path,
    )

    pipeline = TopologyReconstructionPipeline(
        TopologyConfig(
            equipment_snap_threshold=48.0,
            candidate_edge_k=12,
            node_cluster_eps=6.0,
            min_edge_length_px=2,
        )
    )
    artifacts = pipeline.run(pipe_mask, detections)

    save_outputs(
        out_dir=out_dir,
        image_bgr=image_bgr,
        binary_inv=binary_inv,
        masked=masked,
        cleaned=cleaned,
        filtered=pipe_mask,
        detections=detections,
        artifacts=artifacts,
    )
    pipeline.export_graphml(artifacts.graph, str(out_dir / "sample_topology_step_10_graph.graphml"))

    accepted = [result for result in artifacts.equipment_associations if result.accepted]
    print(f"image: {image_path}")
    print(f"equipment detections used: {len(detections)}")
    print(f"graph nodes: {artifacts.graph.number_of_nodes()}")
    print(f"graph edges: {artifacts.graph.number_of_edges()}")
    print(f"accepted equipment attachments: {len(accepted)} / {len(artifacts.equipment_associations)}")
    print(f"outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
