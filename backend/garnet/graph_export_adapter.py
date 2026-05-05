from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NODE_TYPES = [
    "equipment_general",
    "tank_vessel",
    "pump_compressor",
    "valve",
    "instrumentation",
    "inlet_outlet",
    "arrow",
    "crossing",
    "ankle",
    "border",
]
EDGE_TYPES = ["solid", "non_solid"]


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _bbox_to_xywh(bbox: Any) -> dict[str, float]:
    if isinstance(bbox, dict):
        if {"x", "y", "w", "h"}.issubset(bbox):
            return {
                "x": float(bbox.get("x", 0.0)),
                "y": float(bbox.get("y", 0.0)),
                "w": max(0.0, float(bbox.get("w", 0.0))),
                "h": max(0.0, float(bbox.get("h", 0.0))),
            }
        if {"x_min", "y_min", "x_max", "y_max"}.issubset(bbox):
            x_min = float(bbox.get("x_min", 0.0))
            y_min = float(bbox.get("y_min", 0.0))
            x_max = float(bbox.get("x_max", x_min))
            y_max = float(bbox.get("y_max", y_min))
            return {"x": x_min, "y": y_min, "w": max(0.0, x_max - x_min), "h": max(0.0, y_max - y_min)}
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x_min, y_min, x_max, y_max = [float(item) for item in bbox[:4]]
        return {"x": x_min, "y": y_min, "w": max(0.0, x_max - x_min), "h": max(0.0, y_max - y_min)}
    return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def _center_from_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {"x": float(bbox["x"] + bbox["w"] / 2.0), "y": float(bbox["y"] + bbox["h"] / 2.0)}


def _fallback_bbox_from_position(position: Any) -> dict[str, float]:
    if not isinstance(position, dict):
        return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    x = float(position.get("x", 0.0))
    y = float(position.get("y", 0.0))
    return {"x": x - 0.5, "y": y - 0.5, "w": 1.0, "h": 1.0}


def _object_key_from_node_id(node_id: str) -> str | None:
    if "::" not in node_id:
        return None
    return node_id.split("::", 1)[1]


def _page_connector_labels_by_object_id(page_connector_labels_payload: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    labels_by_object_id: dict[str, list[dict[str, Any]]] = {}
    for connector in (page_connector_labels_payload or {}).get("connectors", []):
        object_id = str(connector.get("object_id") or "")
        if object_id:
            labels_by_object_id[object_id] = list(connector.get("labels", []))
    return labels_by_object_id


def get_bbox_from_objects(objects: list[dict[str, Any]], object_id: str | None = None) -> dict[str, dict[str, float]] | dict[str, float] | None:
    bboxes: dict[str, dict[str, float]] = {}
    for obj in objects:
        obj_id = str(obj.get("id") or obj.get("det_id") or "")
        if not obj_id:
            continue
        bboxes[obj_id] = _bbox_to_xywh(obj.get("bbox"))
    if object_id is None:
        return bboxes
    return bboxes.get(str(object_id))


def reproject_polyline(polyline: list[dict[str, Any]]) -> list[dict[str, float]]:
    projected: list[dict[str, float]] = []
    for point in polyline or []:
        if not isinstance(point, dict):
            continue
        x = point.get("x", point.get("col", 0.0))
        y = point.get("y", point.get("row", 0.0))
        projected.append({"x": float(x), "y": float(y)})
    return projected


def map_node_type(source_type: str | None) -> str:
    normalized = str(source_type or "").strip().lower().replace("_", " ")
    if normalized in {"tank", "vessel", "column", "heat exchanger"}:
        return "tank_vessel"
    if normalized in {"pump", "compressor", "blower", "fan"}:
        return "pump_compressor"
    if "valve" in normalized or normalized == "reducer":
        return "valve"
    if normalized in {"instrument", "instrumentation", "instrument semantic", "instrument tag"}:
        return "instrumentation"
    if normalized in {"connection", "page connection", "utility connection", "inlet", "outlet", "inlet outlet"}:
        return "inlet_outlet"
    if normalized == "arrow":
        return "arrow"
    if normalized in {"junction", "crossing", "node"}:
        return "crossing"
    if normalized in {"endpoint", "equipment attachment", "inline", "ankle"}:
        return "ankle"
    if normalized == "border":
        return "border"
    return "equipment_general"


def map_edge_type(edge: dict[str, Any]) -> str:
    if str(edge.get("line_style", "")).lower() in {"dashed", "non_solid", "non-solid"}:
        return "non_solid"
    return "solid"


def compute_node_confidence(node: dict[str, Any], source_object: dict[str, Any] | None = None) -> float:
    if source_object is not None:
        return _clamp_confidence(source_object.get("confidence"), default=0.5)
    review_state = str(node.get("review_state", "")).lower()
    if review_state == "accepted":
        return 0.9
    if review_state == "unresolved":
        return 0.3
    return 0.6


def compute_edge_confidence(edge: dict[str, Any]) -> float:
    direction_confidence = edge.get("flow_direction_confidence")
    if direction_confidence is not None:
        return _clamp_confidence(direction_confidence, default=0.6)
    review_state = str(edge.get("review_state", "")).lower()
    if review_state == "accepted":
        return 0.9
    if review_state in {"unresolved", "rejected"}:
        return 0.3
    return 0.6


def build_provenance(notes: str = "") -> dict[str, str]:
    return {
        "annotated_by": "garnet.pipeline",
        "annotated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": "auto",
        "notes": notes,
    }


def _bbox_intersects(a: dict[str, float], b: dict[str, float]) -> bool:
    return not (
        a["x"] + a["w"] < b["x"]
        or b["x"] + b["w"] < a["x"]
        or a["y"] + a["h"] < b["y"]
        or b["y"] + b["h"] < a["y"]
    )


def attach_text_to_nodes(nodes: list[dict[str, Any]], text_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for node in nodes:
        node_bbox = node.get("bbox", {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
        for region in text_regions or []:
            region_bbox = _bbox_to_xywh(region.get("bbox"))
            if _bbox_intersects(node_bbox, region_bbox):
                node["text"] = {
                    "raw": str(region.get("text", "")),
                    "normalized": str(region.get("normalized_text") or region.get("text") or ""),
                    "confidence": _clamp_confidence(region.get("confidence"), default=0.5),
                }
                break
    return nodes


def attach_tags_to_nodes(nodes: list[dict[str, Any]], tag_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for node in nodes:
        node_bbox = node.get("bbox", {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
        for region in tag_regions or []:
            region_bbox = _bbox_to_xywh(region.get("bbox"))
            if not _bbox_intersects(node_bbox, region_bbox):
                continue
            normalized = str(region.get("normalized_text") or region.get("text") or "")
            tags = node.setdefault("tags", {"pid_tag": "", "line_tag": "", "service": ""})
            if str(region.get("text_class", "")).lower() == "line_number" or "-" in normalized:
                tags["line_tag"] = normalized
            else:
                tags["pid_tag"] = normalized
            break
    return nodes


def _source_object_for_node(node: dict[str, Any], object_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    node_id = str(node.get("id", ""))
    object_key = _object_key_from_node_id(node_id)
    if object_key and object_key in object_by_id:
        return object_by_id[object_key]
    if node_id in object_by_id:
        return object_by_id[node_id]
    return None


def _node_bbox(node: dict[str, Any], source_object: dict[str, Any] | None) -> dict[str, float]:
    if source_object is not None:
        return _bbox_to_xywh(source_object.get("bbox"))
    if "bbox" in node:
        return _bbox_to_xywh(node.get("bbox"))
    return _fallback_bbox_from_position(node.get("position"))


def _document_payload(image_id: str, image_dimensions: dict[str, Any] | None) -> dict[str, Any]:
    width = int((image_dimensions or {}).get("width", 1) or 1)
    height = int((image_dimensions or {}).get("height", 1) or 1)
    suffix = Path(image_id).suffix.lower().lstrip(".") or "other"
    file_type = suffix if suffix in {"pdf", "png", "jpg", "tif"} else "other"
    return {
        "doc_id": image_id or "unknown",
        "source": {
            "file_name": image_id or "",
            "file_type": file_type,
            "page_index": 0,
            "render_dpi": 300,
            "notes": "Generated from stage12_graph.json",
        },
        "image": {"width": width, "height": height},
    }


def _tiling_payload(image_dimensions: dict[str, Any] | None) -> dict[str, Any]:
    width = int((image_dimensions or {}).get("width", 1) or 1)
    height = int((image_dimensions or {}).get("height", 1) or 1)
    return {
        "is_patch": False,
        "tile_engine": "sahi",
        "tile": {
            "tile_id": "full_sheet",
            "tile_row": 0,
            "tile_col": 0,
            "tile_width": width,
            "tile_height": height,
            "overlap_x": 0,
            "overlap_y": 0,
            "offset_x": 0,
            "offset_y": 0,
        },
        "global_image": {"width": width, "height": height},
    }



def _exit_terminal_for_anchor(anchor_name: str) -> str:
    """Map anchor_name to exit_terminal value."""
    # anchor_name corresponds to which edge terminal the off-page connector is attached to
    # top/bottom anchors → destination terminal (pipe going out the sheet top/bottom)
    # left/right anchors → source terminal (pipe coming from left/right boundary)
    if anchor_name in ("top", "bottom"):
        return "destination"
    return "source"


def _build_off_page_connector_map(
    connection_attachments_payload: dict[str, Any] | None,
    page_connector_labels_payload: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build a map from edge_id → off_page_connector dict for page-connection edges.

    Only page connection attachments with resolved labels (non-empty labels list)
    get off_page_connector fields. Edges without labels remain unmapped.
    """
    if not connection_attachments_payload:
        return {}

    result: dict[str, dict[str, Any]] = {}
    pc_labels = _page_connector_labels_by_object_id(page_connector_labels_payload)
    for att in connection_attachments_payload.get("accepted", []):
        if att.get("class_name") != "page connection":
            continue
        edge_id = str(att.get("edge_id", ""))
        if not edge_id or edge_id.startswith("attach_edge::"):
            continue
        obj_id = str(att.get("det_id") or att.get("object_id") or "")
        labels = pc_labels.get(obj_id, [])
        if not labels:
            continue

        first_label = labels[0]
        ref = first_label.get("page_reference")
        if not ref:
            continue

        ref_type = str(ref.get("reference_type", "sheet") or "sheet")
        ref_value = str(ref.get("reference_value") or "")
        if not ref_value:
            continue

        anchor_name = str(att.get("anchor_name", ""))
        exit_terminal = _exit_terminal_for_anchor(anchor_name)
        direction = "output" if exit_terminal == "source" else "input"

        result[edge_id] = {
            "reference_type": ref_type,
            "reference_value": ref_value,
            "direction": direction,
            "exit_terminal": exit_terminal,
            "local_edge_id": edge_id,
        }

    return result


def build_graph_v1_payload(
    stage12_graph: dict[str, Any],
    objects_payload: dict[str, Any] | None = None,
    line_numbers_payload: dict[str, Any] | None = None,
    instrument_tags_payload: dict[str, Any] | None = None,
    page_connector_labels_payload: dict[str, Any] | None = None,
    connection_attachments_payload: dict[str, Any] | None = None,
    image_dimensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    objects = (objects_payload or {}).get("objects", [])
    object_by_id = {str(obj.get("id") or obj.get("det_id")): obj for obj in objects if obj.get("id") or obj.get("det_id")}
    page_connector_labels = _page_connector_labels_by_object_id(page_connector_labels_payload)
    image_id = str(stage12_graph.get("image_id") or (objects_payload or {}).get("image_id") or "")

    nodes: list[dict[str, Any]] = []
    page_connector_node_ids: set[str] = set()
    for source_node in stage12_graph.get("nodes", []):
        source_object = _source_object_for_node(source_node, object_by_id)
        bbox = _node_bbox(source_node, source_object)
        center = source_node.get("position") if isinstance(source_node.get("position"), dict) else _center_from_bbox(bbox)
        source_type = (
            source_object.get("class_name")
            if source_object is not None and source_object.get("class_name")
            else source_node.get("type") or source_node.get("kind") or ""
        )
        node = {
            "id": str(source_node.get("id", "")),
            "type": map_node_type(str(source_type)),
            "bbox": bbox,
            "confidence": compute_node_confidence(source_node, source_object),
            "text": {"raw": "", "normalized": "", "confidence": 0.0},
            "role": {
                "is_symbol": bool(source_object),
                "is_topology": str(source_node.get("kind") or source_node.get("type") or "").lower()
                in {"endpoint", "junction", "crossing", "equipment_attachment", "inline"},
            },
            "provenance": build_provenance(f"stage12 node type={source_node.get('type', '')}"),
            "geometry": {"center": {"x": float(center.get("x", 0.0)), "y": float(center.get("y", 0.0))}},
            "patch_link": {"global_bbox_xywh": bbox, "tile_id": "full_sheet"},
            "tags": {"pid_tag": "", "line_tag": "", "service": "", "page_reference": None},
        }
        if str(source_type).strip().lower() == "page connection":
            page_connector_node_ids.add(node["id"])
        nodes.append(node)

    attach_text_to_nodes(nodes, (line_numbers_payload or {}).get("line_numbers", []))
    attach_tags_to_nodes(nodes, (line_numbers_payload or {}).get("line_numbers", []))
    attach_tags_to_nodes(nodes, (instrument_tags_payload or {}).get("instrument_tags", []))
    for node in nodes:
        if node["id"] not in page_connector_node_ids:
            continue
        labels = page_connector_labels.get(_object_key_from_node_id(node["id"]) or "", [])
        first_label = labels[0] if labels else None
        node["text"] = first_label.get("normalized_text") if first_label else None
        node.setdefault("tags", {})["page_reference"] = first_label.get("page_reference") if first_label else None

    edges: list[dict[str, Any]] = []
    off_page_by_edge = _build_off_page_connector_map(
        connection_attachments_payload,
        page_connector_labels_payload,
    )
    for source_edge in stage12_graph.get("edges", []):
        edge_id = str(source_edge.get("id", ""))
        edge_node = {
            "id": edge_id,
            "src": str(source_edge.get("source", "")),
            "dst": str(source_edge.get("target", "")),
            "type": map_edge_type(source_edge),
            "confidence": compute_edge_confidence(source_edge),
            "directed": source_edge.get("flow_direction") is not None,
            "provenance": build_provenance(f"stage12 edge review_state={source_edge.get('review_state', '')}"),
            "geometry": {"polyline": reproject_polyline(source_edge.get("polyline", []))},
        }
        if edge_id in off_page_by_edge:
            edge_node["off_page_connector"] = off_page_by_edge[edge_id]
        edges.append(edge_node)

    return {
        "schema_version": "graph_v1",
        "description": "Graph annotation payload exported from GARNET Stage 12.",
        "coordinate_system": {
            "image_origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "units": "pixels",
            "bbox_format": "xywh",
            "bbox_xywh_definition": {"x": "left", "y": "top", "w": "width", "h": "height"},
        },
        "document": _document_payload(image_id, image_dimensions),
        "tiling": _tiling_payload(image_dimensions),
        "classes": {"node_types": NODE_TYPES, "edge_types": EDGE_TYPES},
        "nodes": nodes,
        "edges": edges,
        "constraints": {
            "node_id_unique": True,
            "edge_id_unique": True,
            "edge_endpoints_exist": True,
            "self_loops_disallowed": True,
            "border_nodes_allowed_only_when_is_patch_true": True,
            "edge_type_required": True,
            "bbox_w_h_positive": True,
        },
        "recommended_defaults": {
            "edges": {"directed": False, "type_when_unknown": "solid"},
            "nodes": {"confidence_for_ground_truth": 1.0, "text_confidence_default": 0.0},
        },
    }
