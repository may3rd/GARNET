"""Stage 6 trace association helpers.

This module keeps semantic attachment logic out of the pipeline orchestrator.
It is intentionally data-oriented: callers provide already-loaded Stage 4/5b
artifacts and receive the payloads that `pid_extractor` writes to disk.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Optional

import numpy as np

LINE_NUMBER_REVIEW_ASSUMPTION = "accepted_line_numbers_are_human_reviewed"


def apply_stage6_line_number_review(
    trace_associations_payload: dict[str, Any],
    line_number_review_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return trace associations with reviewed Stage 6 line numbers applied.

    The raw Stage 6 trace association artifact remains system evidence. This
    helper overlays the HITL line-number review artifact for downstream graph
    construction and exports.
    """
    if not line_number_review_payload:
        return trace_associations_payload

    reviewed = deepcopy(trace_associations_payload)
    accepted = [
        dict(item)
        for item in line_number_review_payload.get("accepted", []) or []
        if str(item.get("trace_id") or "")
    ]
    needs_review = [
        dict(item)
        for item in line_number_review_payload.get("needs_review", []) or []
        if str(item.get("trace_id") or item.get("id") or "")
    ]
    missing_trace_ids = {
        str(trace_id)
        for trace_id in line_number_review_payload.get("traces_without_line_number", []) or []
        if str(trace_id)
    }

    accepted_by_trace: dict[str, list[dict[str, Any]]] = {}
    for item in accepted:
        trace_id = str(item.get("trace_id") or "")
        item.setdefault("review_state", "accepted")
        item.setdefault("review_source", "human")
        item.setdefault("review_required", False)
        accepted_by_trace.setdefault(trace_id, []).append(item)

    reviewed_trace_ids = set(accepted_by_trace) | missing_trace_ids
    for item in needs_review:
        trace_id = str(item.get("trace_id") or item.get("id") or "")
        if trace_id:
            reviewed_trace_ids.add(trace_id)

    for edge in reviewed.get("trace_edges", []) or []:
        trace_id = str(edge.get("trace_id") or "")
        if trace_id not in reviewed_trace_ids:
            continue
        attachments = edge.setdefault("attachments", {})
        attachments["line_numbers"] = accepted_by_trace.get(trace_id, [])

    associations = reviewed.setdefault("associations", {})
    line_assoc = associations.setdefault("line_numbers", {})
    line_assoc["accepted"] = accepted
    line_assoc["rejected"] = needs_review

    unresolved = reviewed.setdefault("unresolved", {})
    unresolved["traces_without_line_number"] = sorted(missing_trace_ids)
    unresolved["unattached_line_numbers"] = needs_review
    reviewed["line_number_review_applied"] = True
    reviewed["line_number_review_assumption"] = line_number_review_payload.get(
        "review_assumption",
        LINE_NUMBER_REVIEW_ASSUMPTION,
    )
    return reviewed


def _mark_line_number_review_state(association: dict[str, Any], *, accepted: bool) -> dict[str, Any]:
    result = dict(association)
    if accepted:
        result.update(
            {
                "review_state": "accepted",
                "review_source": "human_assumed",
                "review_required": False,
            }
        )
    else:
        result.update(
            {
                "review_state": "needs_review",
                "review_source": "system",
                "review_required": True,
            }
        )
    return result


def build_stage6_line_number_review_payload(
    *,
    image_id: str,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    traces_without_line_number: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "image_id": image_id,
        "review_assumption": LINE_NUMBER_REVIEW_ASSUMPTION,
        "accepted": accepted,
        "needs_review": rejected,
        "traces_without_line_number": traces_without_line_number,
    }
    summary = {
        "image_id": image_id,
        "accepted_count": len(accepted),
        "needs_review_count": len(rejected),
        "trace_without_line_number_count": len(traces_without_line_number),
        "simulated_assignment_count": len([item for item in accepted if item.get("source") == "simulated_hitl"]),
        "review_assumption": LINE_NUMBER_REVIEW_ASSUMPTION,
    }
    return payload, summary


def _stable_choice_index(key: str, count: int) -> int:
    if count <= 0:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(key)) % count


def _bbox_point_distance(bbox: dict[str, Any] | None, point: tuple[float, float] | None) -> float:
    """Distance from a point to a bbox (0.0 when inside); inf when either is missing."""
    if not bbox or point is None:
        return float("inf")
    try:
        x_min = float(bbox["x_min"])
        y_min = float(bbox["y_min"])
        x_max = float(bbox["x_max"])
        y_max = float(bbox["y_max"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    px, py = point
    dx = max(x_min - px, 0.0, px - x_max)
    dy = max(y_min - py, 0.0, py - y_max)
    return float(math.hypot(dx, dy))


def _trace_reference_points(edge: dict[str, Any]) -> list[tuple[float, float]]:
    """Endpoints that anchor a trace geometrically: polyline ends and terminal."""
    points: list[tuple[float, float]] = []
    polyline = edge.get("polyline") or []
    if polyline:
        for index in (0, -1):
            point = polyline[index]
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    terminal = edge.get("terminal_xy")
    if isinstance(terminal, (list, tuple)) and len(terminal) >= 2:
        points.append((float(terminal[0]), float(terminal[1])))
    return points


def _nearest_template_index(edge: dict[str, Any], reviewed_pool: list[dict[str, Any]], trace_id: str) -> int:
    """Index of the pool line number geometrically nearest this trace.

    Prefers the accepted line number closest to the trace's endpoints (ports
    and terminal); falls back to the stable hash pick when either side lacks
    usable geometry so the assignment stays deterministic.
    """
    fallback = _stable_choice_index(trace_id, len(reviewed_pool))
    points = _trace_reference_points(edge)
    if not points:
        return fallback
    best_index, best_distance = fallback, float("inf")
    for index, template in enumerate(reviewed_pool):
        distance = min(_bbox_point_distance(template.get("bbox"), point) for point in points)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def simulate_line_number_hitl_for_missing_traces(
    edges: list[dict[str, Any]],
    reviewed_line_numbers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Temporary deterministic stand-in for human line-number correction."""
    reviewed_pool = [
        item
        for item in reviewed_line_numbers
        if str(item.get("review_state") or "") == "accepted" and str(item.get("id") or item.get("source_object_id") or "")
    ]
    if not reviewed_pool:
        return []

    assignments: list[dict[str, Any]] = []
    for edge in edges:
        attachments = edge.setdefault("attachments", {})
        line_numbers = attachments.setdefault("line_numbers", [])
        if line_numbers:
            continue
        # Only sheet-boundary connectors need a line number (they feed the
        # off-page connector key used by strict multi-sheet merge). Equipment
        # ports and branch stubs without a nearby label stay empty and are
        # reported in traces_without_line_number instead of getting a
        # fabricated number drawn onto the wrong pipe.
        source_type = str(edge.get("source_obj_type") or "").casefold()
        if "connection" not in source_type:
            continue
        trace_id = str(edge.get("trace_id") or "")
        template = reviewed_pool[_nearest_template_index(edge, reviewed_pool, trace_id)]
        line_id = str(template.get("id") or template.get("source_object_id") or "")
        assignment = {
            "id": line_id,
            "source_object_id": template.get("source_object_id", line_id),
            "class_name": template.get("class_name", ""),
            "bbox": template.get("bbox"),
            "text": template.get("text", ""),
            "normalized_text": template.get("normalized_text", template.get("text", "")),
            "confidence": template.get("confidence"),
            "trace_id": trace_id,
            "trace_kind": edge.get("trace_kind"),
            "source": "simulated_hitl",
            "review_state": "accepted",
            "review_source": "human_simulated",
            "review_required": False,
            "simulated_from_line_number_id": line_id,
        }
        line_numbers.append(assignment)
        assignments.append(assignment)
    return assignments


def _point_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, float, float]:
    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq <= 0:
        return ax, ay, 0.0, math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    qx = ax + t * abx
    qy = ay + t * aby
    return qx, qy, t, math.hypot(px - qx, py - qy)


def _bbox_points(bbox: dict[str, Any]) -> list[tuple[float, float]]:
    x_min = float(bbox["x_min"])
    y_min = float(bbox["y_min"])
    x_max = float(bbox["x_max"])
    y_max = float(bbox["y_max"])
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return [
        (cx, cy),
        (x_min, y_min),
        (x_max, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (cx, y_min),
        (cx, y_max),
        (x_min, cy),
        (x_max, cy),
    ]


def _polyline_from_segments(segments: list[dict[str, Any]]) -> list[list[int]]:
    polyline: list[list[int]] = []
    for segment in segments:
        p1 = [int(segment["x1"]), int(segment["y1"])]
        p2 = [int(segment["x2"]), int(segment["y2"])]
        if not polyline or polyline[-1] != p1:
            polyline.append(p1)
        if polyline[-1] != p2:
            polyline.append(p2)
    return polyline


def _source_metadata(source_obj_id: str, objects_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    obj = objects_by_id.get(source_obj_id)
    if obj is not None:
        return {
            "source_obj_id": source_obj_id,
            "source_obj_type": obj.get("class_name"),
            "source_bbox": obj.get("bbox"),
        }
    if source_obj_id.startswith("equip_"):
        return {
            "source_obj_id": source_obj_id,
            "source_obj_type": "equipment",
            "source_bbox": None,
        }
    if source_obj_id.startswith("branch_"):
        return {
            "source_obj_id": source_obj_id,
            "source_obj_type": "branch_candidate",
            "source_bbox": None,
        }
    return {
        "source_obj_id": source_obj_id,
        "source_obj_type": None,
        "source_bbox": None,
    }


def load_stage5b_trace_edges(
    trace_payload: dict[str, Any],
    branch_payload: dict[str, Any],
    objects_by_id: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    objects_by_id = objects_by_id or {}
    edges: list[dict[str, Any]] = []
    for trace_id, trace in trace_payload.items():
        segments = trace.get("segments", [])
        if not segments:
            continue
        source_obj_id = str(trace.get("source_obj_id", trace_id))
        edges.append({
            "trace_id": str(trace_id),
            "trace_kind": "port",
            **_source_metadata(source_obj_id, objects_by_id),
            "port_index": trace.get("port_index"),
            "port": trace.get("port"),
            "terminal_type": trace.get("terminal_type"),
            "terminal_obj_id": trace.get("terminal_obj_id"),
            "terminal_xy": [trace.get("terminal_x"), trace.get("terminal_y")],
            "segments": segments,
            "polyline": _polyline_from_segments(segments),
            "turns": trace.get("turns", []),
            "hits": trace.get("hits", []),
            "trace_length_px": trace.get("trace_length_px", 0),
            "status": trace.get("status", "ok"),
            "attachments": {},
            "warnings": [],
        })

    for branch_id, branch in branch_payload.get("branches", {}).items():
        if branch.get("status") != "traced" or not branch.get("segments"):
            continue
        segments = branch.get("segments", [])
        source_obj_id = str(branch_id)
        edges.append({
            "trace_id": str(branch_id),
            "trace_kind": "branch",
            **_source_metadata(source_obj_id, objects_by_id),
            "candidate": branch.get("candidate", {}),
            "port": branch.get("port"),
            "terminal_type": branch.get("terminal_type"),
            "terminal_obj_id": branch.get("terminal_obj_id"),
            "terminal_xy": [branch.get("terminal_x"), branch.get("terminal_y")],
            "segments": segments,
            "polyline": _polyline_from_segments(segments),
            "turns": branch.get("turns", []),
            "hits": branch.get("hits", []),
            "trace_length_px": branch.get("trace_length_px", 0),
            "status": branch.get("status", "traced"),
            "paired_branch_id": branch.get("paired_branch_id"),
            "attachments": {},
            "warnings": [],
        })
    return edges


def _nearest_segment(point: tuple[float, float], edges: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    best: Optional[dict[str, Any]] = None
    px, py = point
    for edge in edges:
        cumulative = 0.0
        for index, segment in enumerate(edge.get("segments", [])):
            ax = float(segment["x1"])
            ay = float(segment["y1"])
            bx = float(segment["x2"])
            by = float(segment["y2"])
            qx, qy, t, distance = _point_to_segment(px, py, ax, ay, bx, by)
            seg_len = max(abs(bx - ax), abs(by - ay))
            along = cumulative + t * seg_len
            if best is None or distance < best["distance_px"]:
                best = {
                    "trace_id": edge["trace_id"],
                    "trace_kind": edge["trace_kind"],
                    "segment_index": index,
                    "projected_xy": [round(qx, 2), round(qy, 2)],
                    "distance_px": round(distance, 2),
                    "t": round(t, 4),
                    "trace_distance_px": round(along, 2),
                }
            cumulative += seg_len
    return best


def _nearest_bbox(bbox: dict[str, Any], edges: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    best: Optional[dict[str, Any]] = None
    for point in _bbox_points(bbox):
        candidate = _nearest_segment(point, edges)
        if candidate is None:
            continue
        if best is None or candidate["distance_px"] < best["distance_px"]:
            best = candidate
    return best


_LABEL_ORIENTATION_MIN_RATIO = 1.5


def _label_orientation(bbox: dict[str, Any]) -> Optional[str]:
    """Aspect orientation of a text label: 'horizontal', 'vertical', or None.

    Near-square labels are orientation-ambiguous and get no preference.
    """
    try:
        width = float(bbox["x_max"]) - float(bbox["x_min"])
        height = float(bbox["y_max"]) - float(bbox["y_min"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    if width >= height * _LABEL_ORIENTATION_MIN_RATIO:
        return "horizontal"
    if height >= width * _LABEL_ORIENTATION_MIN_RATIO:
        return "vertical"
    return None


def _segment_orientation(segment: dict[str, Any]) -> str:
    dx = abs(float(segment["x2"]) - float(segment["x1"]))
    dy = abs(float(segment["y2"]) - float(segment["y1"]))
    return "horizontal" if dx > dy else "vertical"


def _nearest_bbox_oriented(
    bbox: dict[str, Any],
    edges: list[dict[str, Any]],
    orientation: str,
    max_distance_px: float,
) -> Optional[dict[str, Any]]:
    """Nearest segment whose direction matches the label orientation.

    Mirrors the corner-sampling metric of `_nearest_bbox`, but only considers
    candidates within `max_distance_px`; returns None when no orientation-
    matching candidate qualifies so the caller can fall back to the pure
    nearest segment.
    """
    best: Optional[dict[str, Any]] = None
    for edge in edges:
        cumulative = 0.0
        for index, segment in enumerate(edge.get("segments", [])):
            if _segment_orientation(segment) != orientation:
                cumulative += max(abs(float(segment["x2"]) - float(segment["x1"])), abs(float(segment["y2"]) - float(segment["y1"])))
                continue
            ax = float(segment["x1"])
            ay = float(segment["y1"])
            bx = float(segment["x2"])
            by = float(segment["y2"])
            seg_len = max(abs(bx - ax), abs(by - ay))
            for point in _bbox_points(bbox):
                qx, qy, t, distance = _point_to_segment(point[0], point[1], ax, ay, bx, by)
                if best is None or distance < best["distance_px"]:
                    best = {
                        "trace_id": edge["trace_id"],
                        "trace_kind": edge["trace_kind"],
                        "segment_index": index,
                        "projected_xy": [round(qx, 2), round(qy, 2)],
                        "distance_px": round(distance, 2),
                        "t": round(t, 4),
                        "trace_distance_px": round(cumulative + t * seg_len, 2),
                    }
            cumulative += seg_len
    if best is not None and best["distance_px"] <= max_distance_px:
        return best
    return None


def _add_attachment(
    edges_by_id: dict[str, dict[str, Any]],
    trace_id: str,
    group: str,
    association: dict[str, Any],
) -> None:
    edge = edges_by_id.get(trace_id)
    if edge is None:
        return
    edge.setdefault("attachments", {}).setdefault(group, []).append(association)


def _attach_bbox_items(
    *,
    edges: list[dict[str, Any]],
    edges_by_id: dict[str, dict[str, Any]],
    group: str,
    items: list[dict[str, Any]],
    max_distance_px: float,
    id_key: str = "id",
    class_key: str = "class_name",
    prefer_orientation_match: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in items:
        bbox = item.get("bbox")
        item_id = str(item.get(id_key, ""))
        if not bbox:
            rejected.append({"id": item_id, "reason": "missing_bbox", "source": item})
            continue
        nearest = _nearest_bbox(bbox, edges)
        if prefer_orientation_match and nearest is not None:
            # A text label annotates the line it is written along: when the
            # globally nearest segment runs perpendicular to the label's long
            # axis (e.g. a crossing pipe clipping the label corner), prefer a
            # direction-matching segment within threshold.
            orientation = _label_orientation(bbox)
            if orientation is not None:
                aligned = _nearest_bbox_oriented(bbox, edges, orientation, max_distance_px)
                if aligned is not None:
                    nearest = aligned
        if nearest is None:
            rejected.append({"id": item_id, "reason": "no_trace_edges", "source": item})
            continue
        association = {
            "id": item_id,
            "source_object_id": item.get("source_object_id", item_id),
            "class_name": item.get(class_key, item.get("semantic_class", "")),
            "bbox": bbox,
            "text": item.get("text", ""),
            "normalized_text": item.get("normalized_text", ""),
            "confidence": item.get("confidence", item.get("fused_confidence", item.get("detection_confidence"))),
            **nearest,
        }
        if nearest["distance_px"] <= max_distance_px:
            if group == "line_numbers":
                association = _mark_line_number_review_state(association, accepted=True)
            accepted.append(association)
            _add_attachment(edges_by_id, nearest["trace_id"], group, association)
        else:
            rejected_association = {
                **association,
                "reason": "distance_over_threshold",
                "max_distance_px": max_distance_px,
            }
            if group == "line_numbers":
                rejected_association = _mark_line_number_review_state(rejected_association, accepted=False)
            rejected.append(rejected_association)
    return accepted, rejected


def render_trace_association_overlay(
    image: np.ndarray,
    edges: list[dict[str, Any]],
    associations: dict[str, Any],
) -> np.ndarray:
    import cv2 as _cv2

    overlay = image.copy()
    for edge in edges:
        color = (0, 180, 0) if edge.get("trace_kind") == "port" else (0, 0, 220)
        for segment in edge.get("segments", []):
            _cv2.line(
                overlay,
                (int(segment["x1"]), int(segment["y1"])),
                (int(segment["x2"]), int(segment["y2"])),
                color,
                2,
            )

    draw_specs = [
        ("equipment_ports", (255, 255, 0), 4),
        ("inline_objects", (0, 165, 255), 4),
        ("line_numbers", (255, 0, 255), 3),
        ("instrument_tags", (0, 255, 255), 3),
        ("flow_arrows", (255, 0, 0), 4),
        ("terminals", (180, 0, 180), 5),
    ]
    for group, color, radius in draw_specs:
        for item in associations.get(group, {}).get("accepted", []):
            projected = item.get("projected_xy")
            if not projected:
                continue
            x = int(round(float(projected[0])))
            y = int(round(float(projected[1])))
            _cv2.circle(overlay, (x, y), radius, color, -1)
            if group == "line_numbers":
                label = str(item.get("normalized_text") or item.get("text") or item.get("id") or "")
                if len(label) > 48:
                    label = label[:45] + "..."
                if label:
                    _cv2.putText(
                        overlay,
                        label,
                        (x + 8, y - 8),
                        _cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        _cv2.LINE_AA,
                    )
    return overlay


def build_trace_associations(
    *,
    image_id: str,
    objects: list[dict[str, Any]],
    trace_payload: dict[str, Any],
    branch_payload: dict[str, Any],
    ports_payload: dict[str, Any],
    line_numbers: list[dict[str, Any]],
    instrument_tags: list[dict[str, Any]],
    equipment_port_max_distance_px: float,
    inline_object_max_distance_px: float,
    text_max_distance_px: float,
    instrument_max_distance_px: float,
    arrow_max_distance_px: float,
) -> dict[str, Any]:
    objects_by_id = {str(obj.get("id", "")): obj for obj in objects}
    edges = load_stage5b_trace_edges(trace_payload, branch_payload, objects_by_id)
    edges_by_id = {edge["trace_id"]: edge for edge in edges}

    associations: dict[str, Any] = {
        "equipment_ports": {"accepted": [], "rejected": []},
        "inline_objects": {"accepted": [], "rejected": []},
        "line_numbers": {"accepted": [], "rejected": []},
        "instrument_tags": {"accepted": [], "rejected": []},
        "flow_arrows": {"accepted": [], "rejected": []},
        "terminals": {"accepted": [], "rejected": []},
    }

    for obj_id, port_list in ports_payload.items():
        for port_index, port in enumerate(port_list, start=1):
            if len(port) < 3:
                continue
            trace_id = obj_id if len(port_list) == 1 else f"{obj_id}:port_{port_index:02d}"
            point = (float(port[0]), float(port[1]))
            nearest = _nearest_segment(point, edges)
            if trace_id in edges_by_id:
                own_nearest = _nearest_segment(point, [edges_by_id[trace_id]])
                if own_nearest is not None:
                    nearest = own_nearest
            if nearest is None:
                associations["equipment_ports"]["rejected"].append({
                    "id": f"{obj_id}:port_{port_index:02d}",
                    "reason": "no_trace_edges",
                    "source_obj_id": obj_id,
                    "port_index": port_index,
                    "port": port,
                })
                continue
            association = {
                "id": f"{obj_id}:port_{port_index:02d}",
                "source_obj_id": obj_id,
                "port_index": port_index,
                "port_xy": [int(port[0]), int(port[1])],
                "direction": str(port[2]),
                **nearest,
            }
            if nearest["distance_px"] <= equipment_port_max_distance_px:
                associations["equipment_ports"]["accepted"].append(association)
                _add_attachment(edges_by_id, nearest["trace_id"], "equipment_ports", association)
            else:
                associations["equipment_ports"]["rejected"].append({
                    **association,
                    "reason": "distance_over_threshold",
                    "max_distance_px": equipment_port_max_distance_px,
                })

    inline_classes = {
        "gate_valve", "globe_valve", "check_valve", "ball_valve",
        "butterfly_valve", "control_valve", "pressure_relief_valve",
        "reducer", "spectacle_blind", "strainer",
        "gate valve", "globe valve", "check valve", "ball valve",
        "butterfly valve", "control valve", "pressure relief valve",
        "spectacle blind",
    }
    inline_objects = [obj for obj in objects if obj.get("class_name") in inline_classes]
    accepted, rejected = _attach_bbox_items(
        edges=edges,
        edges_by_id=edges_by_id,
        group="inline_objects",
        items=inline_objects,
        max_distance_px=inline_object_max_distance_px,
    )
    associations["inline_objects"]["accepted"].extend(accepted)
    associations["inline_objects"]["rejected"].extend(rejected)

    seen_inline = {
        (item.get("trace_id"), item.get("class_name"), int(round(float(item.get("projected_xy", [0, 0])[0]))), int(round(float(item.get("projected_xy", [0, 0])[1]))))
        for item in associations["inline_objects"]["accepted"]
    }
    for edge in edges:
        for hit_index, hit in enumerate(edge.get("hits", []), start=1):
            point = (float(hit.get("x", 0)), float(hit.get("y", 0)))
            nearest = _nearest_segment(point, [edge])
            if nearest is None:
                continue
            key = (
                edge["trace_id"],
                hit.get("class", hit.get("class_name", "")),
                int(round(float(nearest["projected_xy"][0]))),
                int(round(float(nearest["projected_xy"][1]))),
            )
            if key in seen_inline:
                continue
            association = {
                "id": f"{edge['trace_id']}:hit_{hit_index:03d}",
                "class_name": hit.get("class", hit.get("class_name", "")),
                "hit_xy": [int(point[0]), int(point[1])],
                "source": "stage5b_hit",
                **nearest,
            }
            associations["inline_objects"]["accepted"].append(association)
            _add_attachment(edges_by_id, edge["trace_id"], "inline_objects", association)
            seen_inline.add(key)

    accepted, rejected = _attach_bbox_items(
        edges=edges,
        edges_by_id=edges_by_id,
        group="line_numbers",
        items=line_numbers,
        max_distance_px=text_max_distance_px,
        prefer_orientation_match=True,
    )
    associations["line_numbers"]["accepted"] = accepted
    associations["line_numbers"]["rejected"] = rejected

    accepted, rejected = _attach_bbox_items(
        edges=edges,
        edges_by_id=edges_by_id,
        group="instrument_tags",
        items=instrument_tags,
        max_distance_px=instrument_max_distance_px,
    )
    associations["instrument_tags"]["accepted"] = accepted
    associations["instrument_tags"]["rejected"] = rejected

    arrows = [obj for obj in objects if obj.get("class_name") == "arrow"]
    accepted, rejected = _attach_bbox_items(
        edges=edges,
        edges_by_id=edges_by_id,
        group="flow_arrows",
        items=arrows,
        max_distance_px=arrow_max_distance_px,
    )
    associations["flow_arrows"]["accepted"] = accepted
    associations["flow_arrows"]["rejected"] = rejected

    for edge in edges:
        terminal_xy = edge.get("terminal_xy") or []
        if len(terminal_xy) != 2 or terminal_xy[0] is None or terminal_xy[1] is None:
            continue
        nearest = _nearest_segment((float(terminal_xy[0]), float(terminal_xy[1])), [edge])
        if nearest is None:
            continue
        association = {
            "id": f"{edge['trace_id']}:terminal",
            "terminal_type": edge.get("terminal_type"),
            "terminal_obj_id": edge.get("terminal_obj_id"),
            "terminal_xy": terminal_xy,
            **nearest,
        }
        associations["terminals"]["accepted"].append(association)
        _add_attachment(edges_by_id, edge["trace_id"], "terminals", association)

    skipped_branches = [
        {"id": branch_id, **branch}
        for branch_id, branch in branch_payload.get("branches", {}).items()
        if branch.get("status") != "traced"
    ]
    # Missing line numbers are unresolved evidence.  The former deterministic
    # HITL stand-in assigned an arbitrary accepted label to connector traces,
    # which could silently corrupt cross-sheet line identity.  Keep the helper
    # available for legacy experiments, but never invoke it in production.
    simulated_line_number_assignments: list[dict[str, Any]] = []
    traces_without_line_number = [
        edge["trace_id"]
        for edge in edges
        if not edge.get("attachments", {}).get("line_numbers")
    ]
    dead_end_traces = [
        edge["trace_id"]
        for edge in edges
        if edge.get("terminal_type") == "dead_end"
    ]
    line_number_review_payload, line_number_review_summary = build_stage6_line_number_review_payload(
        image_id=image_id,
        accepted=associations["line_numbers"]["accepted"],
        rejected=associations["line_numbers"]["rejected"],
        traces_without_line_number=traces_without_line_number,
    )

    payload = {
        "image_id": image_id,
        "trace_source": "stage5b",
        "trace_edges": edges,
        "associations": associations,
        "unresolved": {
            "skipped_branches": skipped_branches,
            "traces_without_line_number": traces_without_line_number,
            "dead_end_traces": dead_end_traces,
            "unattached_line_numbers": associations["line_numbers"]["rejected"],
            "unattached_instrument_tags": associations["instrument_tags"]["rejected"],
        },
    }
    summary = {
        "image_id": image_id,
        "trace_edge_count": len(edges),
        "port_trace_count": len([edge for edge in edges if edge.get("trace_kind") == "port"]),
        "branch_trace_count": len([edge for edge in edges if edge.get("trace_kind") == "branch"]),
        "skipped_branch_count": len(skipped_branches),
        "dead_end_trace_count": len(dead_end_traces),
        "trace_without_line_number_count": len(traces_without_line_number),
        "accepted_counts": {
            key: len(value.get("accepted", []))
            for key, value in associations.items()
        },
        "simulated_line_number_assignment_count": len(simulated_line_number_assignments),
        "rejected_counts": {
            key: len(value.get("rejected", []))
            for key, value in associations.items()
        },
    }

    return {
        "trace_edges": edges,
        "associations": associations,
        "trace_associations_payload": payload,
        "trace_association_summary": summary,
        "line_number_review_payload": line_number_review_payload,
        "line_number_review_summary": line_number_review_summary,
    }
