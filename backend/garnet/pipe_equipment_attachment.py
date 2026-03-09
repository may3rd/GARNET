from __future__ import annotations

from typing import Any

from garnet.equipment_pipe_association import (
    AssociationResult,
    CandidateScorer,
    Detection,
    EquipmentPipeAssociatorV2,
    PipeEdge,
)


def _to_detection(obj: dict[str, Any]) -> Detection:
    bbox = obj["bbox"]
    return Detection(
        det_id=str(obj["id"]),
        class_name=str(obj["class_name"]),
        bbox=(
            int(bbox["x_min"]),
            int(bbox["y_min"]),
            int(bbox["x_max"]),
            int(bbox["y_max"]),
        ),
        score=float(obj.get("confidence", 1.0)),
        tag=obj.get("class_name"),
        metadata=dict(obj),
    )


def _to_pipe_edge(edge: dict[str, Any]) -> PipeEdge:
    polyline = [(float(point["col"]), float(point["row"])) for point in edge.get("polyline", [])]
    return PipeEdge(
        edge_id=str(edge["id"]),
        source=str(edge["source"]),
        target=str(edge["target"]),
        polyline_xy=polyline,
        metadata=dict(edge),
    )


def run_pipe_equipment_attachment_stage(
    *,
    image_id: str,
    objects: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    attachment_classes: tuple[str, ...] = ("pump", "heat exchanger", "tank", "vessel", "column", "compressor", "blower", "fan"),
    max_distance_px: float = 48.0,
    k_candidate_edges: int = 10,
) -> dict[str, Any]:
    normalized_allow = {item.lower() for item in attachment_classes}
    equipment_objects = [obj for obj in objects if str(obj.get("class_name", "")).lower() in normalized_allow]
    if not equipment_objects or not edges:
        return {
            "attachments_payload": {
                "image_id": image_id,
                "pass_type": "sheet",
                "accepted": [],
                "rejected": [],
                "equipment_detection_stage": "stage4 (provisional until Stage 4.1 exists)",
            },
            "summary": {
                "image_id": image_id,
                "pass_type": "sheet",
                "equipment_candidates": len(equipment_objects),
                "accepted_attachment_count": 0,
                "rejected_attachment_count": len(equipment_objects),
                "attachment_classes": list(attachment_classes),
                "equipment_detection_stage": "stage4 (provisional until Stage 4.1 exists)",
            },
        }

    associator = EquipmentPipeAssociatorV2(
        pipe_edges=[_to_pipe_edge(edge) for edge in edges],
        scorer=CandidateScorer(max_distance_px=max_distance_px),
        k_candidate_edges=k_candidate_edges,
    )
    results = associator.associate_many([_to_detection(obj) for obj in equipment_objects])
    accepted = [result for result in results if result.accepted]
    rejected = [result for result in results if not result.accepted]

    def _serialize(result: AssociationResult) -> dict[str, Any]:
        return {
            "det_id": result.det_id,
            "class_name": result.class_name,
            "bbox": result.bbox,
            "accepted": result.accepted,
            "reason": result.reason,
            "anchor_name": result.anchor_name,
            "anchor_xy": result.anchor_xy,
            "edge_id": result.edge_id,
            "nearest_point_xy": result.nearest_point_xy,
            "distance_px": result.distance_px,
            "score": result.score,
            "segment_index": result.segment_index,
            "t": result.t,
        }

    return {
        "attachments_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "accepted": [_serialize(result) for result in accepted],
            "rejected": [_serialize(result) for result in rejected],
            "equipment_detection_stage": "stage4 (provisional until Stage 4.1 exists)",
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "equipment_candidates": len(equipment_objects),
            "accepted_attachment_count": len(accepted),
            "rejected_attachment_count": len(rejected),
            "attachment_classes": list(attachment_classes),
            "equipment_detection_stage": "stage4 (provisional until Stage 4.1 exists)",
        },
    }
