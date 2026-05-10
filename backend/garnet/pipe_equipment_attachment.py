from __future__ import annotations

from typing import Any

from garnet.equipment_pipe_association import (
    AssociationResult,
    CandidateScorer,
    Detection,
    EquipmentPipeAssociatorV2,
    PipeEdge,
)


def _nearest_anchor_on_bbox(bbox: tuple[int, int, int, int], point_xy: tuple[float, float] | None) -> tuple[float, float] | None:
    if point_xy is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    px, py = float(point_xy[0]), float(point_xy[1])
    side_points = [
        (float(x_min), (float(y_min) + float(y_max)) / 2.0),
        (float(x_max), (float(y_min) + float(y_max)) / 2.0),
        ((float(x_min) + float(x_max)) / 2.0, float(y_min)),
        ((float(x_min) + float(x_max)) / 2.0, float(y_max)),
    ]
    return min(side_points, key=lambda item: (item[0] - px) ** 2 + (item[1] - py) ** 2)


def _connection_anchor_for_edge(
    bbox: tuple[int, int, int, int],
    edge_polyline: list[tuple[float, float]],
) -> tuple[str, tuple[float, float]] | None:
    """
    Compute the anchor for connection-class objects (page connection, utility connection)
    using pipe travel direction, not proximity.

    A page/utility connection symbol on a P&ID is a terminus — the pipe exits the sheet
    boundary through it. The anchor must reflect which side of the symbol the pipe
    enters from and exits toward, based on pipe direction.

    Logic:
      Horizontal pipe (|dx| > |dy|): exits RIGHT if dx>0, LEFT if dx<0
      Vertical pipe   (|dy| >= |dx|): exits BOTTOM if dy>0, TOP if dy<0

    This overrides any proximity-based anchor that the scoring pass may have picked.
    """
    if len(edge_polyline) < 2:
        return None
    x_min, y_min, x_max, y_max = bbox
    bbox_cx = (x_min + x_max) / 2.0
    bbox_cy = (y_min + y_max) / 2.0
    p1, p2 = edge_polyline[0], edge_polyline[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) >= abs(dy):
        # Horizontal
        if dx >= 0:
            return "right", (float(x_max), bbox_cy)
        else:
            return "left", (float(x_min), bbox_cy)
    else:
        # Vertical
        if dy >= 0:
            return "bottom", (bbox_cx, float(y_max))
        else:
            return "top", (bbox_cx, float(y_min))


def _is_connection_class(class_name: str) -> bool:
    normalized = str(class_name or "").lower().strip()
    return normalized in {
        "connection",
        "page connection",
        "utility connection",
    }


def _exit_side_anchor(
    bbox: tuple[int, int, int, int],
    edge_polyline: list[tuple[float, float]],
) -> tuple[str, tuple[float, float]] | None:
    """
    Compute the pipe exit side anchor for equipment attachments.

    Three-tier strategy:
    1. Pipe-through-bbox: if pipe midpoint is inside bbox projection,
       exit side = direction of pipe travel (existing logic)
    2. Pipe-approaches-from-outside: if pipe passes outside bbox,
       exit side = direction pipe travels (new — fixes off-page connector bug)
    3. Fallback: return None (keep original anchor_name from AnchorGenerator)

    Returns (anchor_name, anchor_xy) or None.
    """
    if len(edge_polyline) < 2:
        return None

    x_min, y_min, x_max, y_max = bbox
    bbox_cx = (x_min + x_max) / 2.0
    bbox_cy = (y_min + y_max) / 2.0

    p1, p2 = edge_polyline[0], edge_polyline[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    pipe_mid_x = (p1[0] + p2[0]) / 2.0
    pipe_mid_y = (p1[1] + p2[1]) / 2.0

    pipe_horiz_inside = x_min <= pipe_mid_x <= x_max
    pipe_vert_inside = y_min <= pipe_mid_y <= y_max

    # ---- Tier 1: pipe passes THROUGH the bbox (existing logic) ----
    if abs(dx) > abs(dy):
        if pipe_horiz_inside:
            if dx > 0:
                return "right", (float(x_max), bbox_cy)
            else:
                return "left", (float(x_min), bbox_cy)
    elif abs(dy) > abs(dx):
        if pipe_vert_inside:
            if dy > 0:
                return "bottom", (bbox_cx, float(y_max))
            else:
                return "top", (bbox_cx, float(y_min))

    # ---- Tier 2: pipe approaches FROM OUTSIDE — use travel direction ----
    if abs(dx) > abs(dy):
        # Horizontal pipe: exits in the direction it's traveling
        if dx > 0:
            return "right", (float(x_max), bbox_cy)
        else:
            return "left", (float(x_min), bbox_cy)
    elif abs(dy) > abs(dx):
        # Vertical pipe: exits in the direction it's traveling
        if dy > 0:
            return "bottom", (bbox_cx, float(y_max))
        else:
            return "top", (bbox_cx, float(y_min))

    # ---- Tier 3: cannot determine ----
    return None


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

    # edges_by_id: used in _serialize to look up edge polyline for exit-side anchor
    edges_by_id: dict[str, dict[str, Any]] = {str(e["id"]): e for e in edges}

    def _serialize(result: AssociationResult) -> dict[str, Any]:
        # Look up the edge polyline for this result's edge_id
        edge_polyline_raw: list[dict[str, Any]] | None = None
        if result.edge_id and edges_by_id:
            edge = edges_by_id.get(result.edge_id)
            if edge:
                edge_polyline_raw = edge.get("polyline", [])
        edge_polyline: list[tuple[float, float]] = [
            (float(pt["col"]), float(pt["row"])) for pt in edge_polyline_raw
        ] if edge_polyline_raw else []

        # ── Connection-class: use pipe direction, not proximity ────────────
        # Connection symbols (page connection, utility connection) are pipe
        # terminations — the anchor must reflect which side of the symbol the
        # pipe enters from / exits toward, based on pipe travel direction.
        if _is_connection_class(result.class_name) and edge_polyline:
            conn_anchor = _connection_anchor_for_edge(result.bbox, edge_polyline)
            if conn_anchor:
                conn_anchor_name, conn_anchor_xy = conn_anchor
                return {
                    "det_id": result.det_id,
                    "class_name": result.class_name,
                    "bbox": result.bbox,
                    "accepted": result.accepted,
                    "reason": result.reason,
                    "anchor_name": conn_anchor_name,
                    "anchor_xy": conn_anchor_xy,
                    "edge_id": result.edge_id,
                    "nearest_point_xy": result.nearest_point_xy,
                    "connection_anchor_xy": conn_anchor_xy,
                    "attachment_stub_xy": None
                    if result.nearest_point_xy is None or conn_anchor_xy is None
                    else [result.nearest_point_xy, conn_anchor_xy],
                    "distance_px": result.distance_px,
                    "score": result.score,
                    "segment_index": result.segment_index,
                    "t": result.t,
                    "anchor_override_reason": "pipe_direction_connection",
                }

        # ── Equipment / other: proximity-based anchor with exit-side override ─
        connection_anchor_xy = _nearest_anchor_on_bbox(result.bbox, result.nearest_point_xy)
        exit_anchor = _exit_side_anchor(result.bbox, edge_polyline) if edge_polyline else None
        exit_anchor_name = exit_anchor[0] if exit_anchor else None
        exit_anchor_xy = exit_anchor[1] if exit_anchor else None

        # Use exit-side anchor if computed; otherwise fall back to association result
        final_anchor_name = exit_anchor_name if exit_anchor_name else result.anchor_name
        final_anchor_xy = exit_anchor_xy if exit_anchor_xy is not None else (
            connection_anchor_xy if connection_anchor_xy is not None else result.anchor_xy
        )

        return {
            "det_id": result.det_id,
            "class_name": result.class_name,
            "bbox": result.bbox,
            "accepted": result.accepted,
            "reason": result.reason,
            "anchor_name": final_anchor_name,
            "anchor_xy": final_anchor_xy,
            "edge_id": result.edge_id,
            "nearest_point_xy": result.nearest_point_xy,
            "connection_anchor_xy": connection_anchor_xy,
            "attachment_stub_xy": None
            if result.nearest_point_xy is None or final_anchor_xy is None
            else [result.nearest_point_xy, final_anchor_xy],
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
