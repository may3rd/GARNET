from __future__ import annotations

import copy
import math
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


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
        x_value, y_value = float(x), float(y)
        if math.isfinite(x_value) and math.isfinite(y_value):
            projected.append({"x": x_value, "y": y_value})
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



def _build_off_page_connector_map(
    stage12_graph: dict[str, Any],
    page_connector_labels_payload: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build deterministic graph-native off-page connector metadata by edge ID."""
    from garnet.page_connector import select_connector_metadata

    edges = stage12_graph.get("edges", [])
    pc_labels = _page_connector_labels_by_object_id(page_connector_labels_payload)
    connector_payloads = {
        str(connector.get("object_id") or ""): connector
        for connector in (page_connector_labels_payload or {}).get("connectors", [])
        if str(connector.get("object_id") or "")
    }
    result: dict[str, dict[str, Any]] = {}
    for obj_id, labels in pc_labels.items():
        selected_metadata = select_connector_metadata(labels)
        selected_metadata.update(
            {
                key: value
                for key, value in connector_payloads.get(obj_id, {}).items()
                if key in {"page_reference", "target_sheet_reference", "raw_reference_text", "connector_key"}
                and value not in (None, "")
            }
        )
        ref = selected_metadata.get("page_reference") or {}
        connector_key = str(selected_metadata.get("connector_key") or "").strip()
        ref_type = str(ref.get("reference_type", "sheet") or "sheet")
        ref_value = str(ref.get("reference_value") or "")
        node_id = f"connection::{obj_id}"
        source_edges = sorted(
            (edge for edge in edges if str(edge.get("source", "")) == node_id),
            key=lambda edge: str(edge.get("id", "")),
        )
        target_edges = sorted(
            (edge for edge in edges if str(edge.get("target", "")) == node_id),
            key=lambda edge: str(edge.get("id", "")),
        )
        selected = (source_edges or target_edges or [None])[0]
        if selected is None:
            continue
        if str(selected.get("line_number_review_state") or "") == "human_reviewed":
            corrected_records = selected.get("effective_line_numbers") or selected.get("line_numbers") or []
            if isinstance(corrected_records, list) and len(corrected_records) == 1 and isinstance(corrected_records[0], dict):
                corrected_key = str(
                    corrected_records[0].get("normalized_text") or corrected_records[0].get("display_text") or corrected_records[0].get("text") or ""
                ).strip()
                connector_key = corrected_key
        edge_id = str(selected.get("id", ""))
        result[edge_id] = {
            "reference_type": ref_type,
            "reference_value": ref_value,
            "target_sheet_reference": str(selected_metadata.get("target_sheet_reference") or ref_value),
            "connector_key": connector_key,
            "raw_reference_text": str(selected_metadata.get("raw_reference_text") or ""),
            "direction": "bidirectional",
            "exit_terminal": "source" if str(selected.get("source", "")) == node_id else "destination",
            "local_edge_id": edge_id,
        }
    return result


def _attachment_order_key(item: dict[str, Any], edge: dict[str, Any], index: int) -> tuple[float, int]:
    """Return a stable position along a traced route for an attachment."""
    for key in ("trace_distance_px", "distance_along_route_px", "route_position_px", "projected_distance_px"):
        try:
            if item.get(key) is not None:
                value = float(item[key])
                if math.isfinite(value):
                    return value, index
        except (TypeError, ValueError):
            pass
    # Associations normally carry trace_distance_px.  Project onto the actual
    # polyline for reviewed/manual records without that field.
    projected = item.get("projected_xy")
    if isinstance(projected, (list, tuple)) and len(projected) >= 2:
        try:
            points = reproject_polyline(edge.get("polyline", []))
            best: tuple[float, float] | None = None
            cumulative = 0.0
            px, py = float(projected[0]), float(projected[1])
            for start, end in zip(points, points[1:]):
                dx, dy = end["x"] - start["x"], end["y"] - start["y"]
                length_sq = dx * dx + dy * dy
                t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - start["x"]) * dx + (py - start["y"]) * dy) / length_sq))
                qx, qy = start["x"] + t * dx, start["y"] + t * dy
                distance_sq = (px - qx) ** 2 + (py - qy) ** 2
                candidate = (distance_sq, cumulative + (length_sq ** 0.5) * t)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                cumulative += length_sq ** 0.5
            if best is not None:
                return best[1], index
        except (TypeError, ValueError):
            pass
    return float("inf"), index


def _record_text(record: dict[str, Any]) -> tuple[str, str]:
    display = str(record.get("display_text") or record.get("text") or record.get("normalized_text") or "").strip()
    normalized = str(record.get("normalized_text") or display).strip()
    return display, normalized


def _build_semantic_projection(
    edges: list[dict[str, Any]],
    instrument_tags_payload: dict[str, Any] | None = None,
    drawing_scope: str = "unknown",
) -> dict[str, Any]:
    """Project edge evidence into stable line/object/instrument identities.

    These fields are additive to graph-v1.  Occurrences intentionally retain
    the original attachment/line records so OCR evidence and human review
    decisions remain auditable when a canonical identity is reused on routes.
    """
    lines_by_id: dict[str, dict[str, Any]] = {}
    inline_by_id: dict[str, dict[str, Any]] = {}
    instruments_by_id: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    relationship_keys: set[tuple[str, str, str]] = set()

    def add_line(line_id: str, record: dict[str, Any], edge: dict[str, Any], occurrence_index: int) -> None:
        display, normalized = _record_text(record)
        canonical_key = str(record.get("canonical_line_id") or normalized or line_id).strip()
        if not canonical_key:
            return
        line = lines_by_id.setdefault(canonical_key, {
            "id": f"line::{drawing_scope}::{canonical_key}",
            "drawing_id": drawing_scope,
            "canonical_line_id": canonical_key,
            "display_text": "",
            "normalized_text": "",
            "occurrences": [],
            "review_state": None,
            "provenance": build_provenance("canonical line identity projected from edge evidence"),
        })
        if not line["display_text"] and display:
            line["display_text"] = display
        if not line["normalized_text"] and normalized:
            line["normalized_text"] = normalized
        occurrence = {
            "occurrence_id": str(record.get("occurrence_id") or record.get("id") or record.get("source_object_id") or f"{edge.get('id')}::line_{occurrence_index}"),
            "ocr_id": line_id or None,
            "edge_id": str(edge.get("id") or ""),
            "record": _json_safe(copy.deepcopy(record)),
            "review_state": edge.get("line_number_review_state") or edge.get("review_state"),
            "provenance": copy.deepcopy(edge.get("provenance") or {}),
        }
        line["occurrences"].append(occurrence)
        state = occurrence["review_state"]
        if state:
            line["review_state"] = state

    for edge in edges:
        edge_id = str(edge.get("id") or "")
        records = edge.get("line_numbers") or []
        ids = [str(value) for value in edge.get("line_number_ids", []) if str(value)]
        edge_canonical_ids = []
        record_ids: set[str] = set()
        if isinstance(records, list):
            for index, record in enumerate(records):
                if isinstance(record, dict):
                    record_id = str(record.get("id") or record.get("source_object_id") or "")
                    if record_id:
                        record_ids.add(record_id)
                    add_line(record_id or (ids[index] if index < len(ids) else ""), record, edge, index)
                    _, normalized = _record_text(record)
                    canonical_key = str(record.get("canonical_line_id") or normalized or record_id or (ids[index] if index < len(ids) else "")).strip()
                    if canonical_key and canonical_key not in edge_canonical_ids:
                        edge_canonical_ids.append(canonical_key)
        # Reviewed IDs can remain useful even when the selected record has no
        # text payload (for example a UI correction submitted by ID only).
        for line_id in ids:
            if record_ids and line_id in record_ids:
                continue
            if line_id in edge_canonical_ids:
                continue
            lines_by_id.setdefault(line_id, {
                "id": f"line::{drawing_scope}::{line_id}", "canonical_line_id": line_id, "drawing_id": drawing_scope,
                "display_text": "", "normalized_text": "", "occurrences": [],
                "review_state": edge.get("line_number_review_state"),
                "provenance": build_provenance("canonical line identity projected from reviewed ID"),
            })
            edge_canonical_ids.append(line_id)
        edge["canonical_line_ids"] = [f"line::{drawing_scope}::{value}" for value in edge_canonical_ids]

        attachments = edge.get("attachments") or {}
        for group, store, relation_type, prefix in (
            ("inline_objects", inline_by_id, "has_inline_object", "inline"),
            ("instrument_tags", instruments_by_id, "instrument_association", "instrument"),
        ):
            values = attachments.get(group, []) if isinstance(attachments, dict) else []
            if not isinstance(values, list):
                continue
            for index, raw in enumerate(values):
                if not isinstance(raw, dict):
                    continue
                source_id = str(raw.get("source_object_id") or raw.get("id") or "").strip()
                if prefix == "instrument":
                    _, normalized_tag = _record_text(raw)
                    key = str(raw.get("canonical_instrument_id") or normalized_tag or source_id).strip()
                else:
                    key = source_id
                key = key or f"{edge_id}::{prefix}_{index}"
                identity_id = f"{prefix}::{drawing_scope}::{key}"
                entity = store.setdefault(key, {
                    "id": identity_id,
                    "drawing_id": drawing_scope,
                    "source_object_id": source_id or None,
                    "occurrences": [],
                    "edge_ids": [],
                    "review_state": raw.get("review_state") or edge.get("review_state"),
                    "provenance": build_provenance(f"canonical {prefix} identity projected from edge attachment"),
                })
                if prefix == "inline":
                    entity.setdefault("class_name", str(raw.get("class_name") or "inline_object"))
                else:
                    display, normalized = _record_text(raw)
                    entity.setdefault("tag", display)
                    entity.setdefault("normalized_tag", normalized)
                route_position, _ = _attachment_order_key(raw, edge, index)
                occurrence = {
                    "occurrence_id": str(raw.get("occurrence_id") or raw.get("id") or raw.get("source_object_id") or f"{edge_id}::{prefix}_{index}"),
                    "source_record_id": str(raw.get("id") or raw.get("source_object_id") or "") or None,
                    "edge_id": edge_id,
                    "route_position_px": None if route_position == float("inf") else route_position,
                    "record": _json_safe(copy.deepcopy(raw)),
                    "review_state": raw.get("review_state") or edge.get("review_state"),
                    "provenance": copy.deepcopy(raw.get("provenance") or edge.get("provenance") or {}),
                }
                entity["occurrences"].append(occurrence)
                if edge_id and edge_id not in entity["edge_ids"]:
                    entity["edge_ids"].append(edge_id)
                explicit_relation = str(raw.get("relationship_type") or raw.get("association_type") or "").strip().lower()
                typed_relation = explicit_relation if relation_type == "instrument_association" and explicit_relation in {"measures", "controls", "actuates"} else relation_type
                relationship_source = identity_id if relation_type == "instrument_association" else edge_id
                relationship_target = edge_id if relation_type == "instrument_association" else identity_id
                relationship_key = (typed_relation, relationship_source, relationship_target)
                if relationship_key in relationship_keys:
                    continue
                relationship_keys.add(relationship_key)
                relationships.append({
                    "id": f"{typed_relation}::{edge_id}::{identity_id}",
                    "type": typed_relation,
                    "source": relationship_source,
                    "target": relationship_target,
                    "edge_id": edge_id,
                    "semantic_state": "observed" if typed_relation != "instrument_association" and relation_type == "instrument_association" else ("unresolved" if relation_type == "instrument_association" else "observed"),
                    "provenance": copy.deepcopy(raw.get("provenance") or edge.get("provenance") or {}),
                    "review_state": raw.get("review_state") or edge.get("review_state"),
                })

    # Instrument detector records with no route attachment remain available as
    # canonical instruments for downstream review, with no fabricated edge.
    for raw in (instrument_tags_payload or {}).get("instrument_tags", []):
        if not isinstance(raw, dict):
            continue
        source_id = str(raw.get("source_object_id") or raw.get("id") or "").strip()
        _, normalized_tag = _record_text(raw)
        key = str(raw.get("canonical_instrument_id") or normalized_tag or source_id).strip()
        if not key:
            continue
        display, normalized = _record_text(raw)
        entity = instruments_by_id.get(key)
        if entity is not None:
            occurrence_id = str(raw.get("occurrence_id") or source_id or f"unattached::{key}")
            if any(str(item.get("occurrence_id") or "") == occurrence_id or (source_id and str(item.get("source_record_id") or "") == source_id) for item in entity.get("occurrences", [])):
                continue
            entity.setdefault("occurrences", []).append({
                "occurrence_id": occurrence_id,
                "source_record_id": source_id or None, "edge_id": None,
                "route_position_px": None, "record": _json_safe(copy.deepcopy(raw)),
                "review_state": raw.get("review_state"), "provenance": copy.deepcopy(raw.get("provenance") or {}),
            })
            continue
        instruments_by_id[key] = {
            "id": f"instrument::{drawing_scope}::{key}",
            "drawing_id": drawing_scope,
            "source_object_id": source_id,
            "tag": display,
            "normalized_tag": normalized,
            "occurrences": [],
            "edge_ids": [],
            "review_state": raw.get("review_state"),
            "provenance": copy.deepcopy(raw.get("provenance") or build_provenance("unattached instrument detector record")),
        }
        instruments_by_id[key]["occurrences"].append({
            "occurrence_id": str(raw.get("occurrence_id") or source_id or f"unattached::{key}"),
            "source_record_id": source_id or None, "edge_id": None,
            "route_position_px": None, "record": _json_safe(copy.deepcopy(raw)),
            "review_state": raw.get("review_state"), "provenance": copy.deepcopy(raw.get("provenance") or {}),
        })

    for entity in inline_by_id.values():
        entity["occurrences"].sort(key=lambda item: (item.get("edge_id") or "", float("inf") if item["route_position_px"] is None else item["route_position_px"]))
    for entity in instruments_by_id.values():
        entity["occurrences"].sort(key=lambda item: (item.get("edge_id") or "", float("inf") if item["route_position_px"] is None else item["route_position_px"]))
    for line in lines_by_id.values():
        line["occurrences"].sort(key=lambda item: item["edge_id"])
    return {
        "lines": [lines_by_id[key] for key in sorted(lines_by_id)],
        "inline_objects": [inline_by_id[key] for key in sorted(inline_by_id)],
        "instruments": [instruments_by_id[key] for key in sorted(instruments_by_id)],
        "relationships": sorted(relationships, key=lambda item: item["id"]),
    }


def build_graph_v1_payload(
    stage12_graph: dict[str, Any],
    objects_payload: dict[str, Any] | None = None,
    line_numbers_payload: dict[str, Any] | None = None,
    instrument_tags_payload: dict[str, Any] | None = None,
    page_connector_labels_payload: dict[str, Any] | None = None,
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
        reference_label = next((label for label in labels if label.get("page_reference")), None)
        node["text"] = reference_label.get("normalized_text") if reference_label else None
        node.setdefault("tags", {})["page_reference"] = reference_label.get("page_reference") if reference_label else None

    edges: list[dict[str, Any]] = []
    off_page_by_edge = _build_off_page_connector_map(
        stage12_graph,
        page_connector_labels_payload,
    )
    for source_edge in stage12_graph.get("edges", []):
        edge_id = str(source_edge.get("id", ""))
        attachments = source_edge.get("attachments")
        if not isinstance(attachments, dict):
            attachments = {}
        line_numbers = source_edge.get("effective_line_numbers") or source_edge.get("line_numbers")
        if not isinstance(line_numbers, list):
            line_numbers = attachments.get("line_numbers") if isinstance(attachments.get("line_numbers"), list) else []
        line_number_ids = source_edge.get("effective_line_number_ids") or source_edge.get("line_number_ids")
        if not isinstance(line_number_ids, list):
            line_number_ids = [
                str(record.get("id") or record.get("source_object_id"))
                for record in line_numbers
                if isinstance(record, dict) and str(record.get("id") or record.get("source_object_id") or "")
            ]
        legacy_flow_direction = source_edge.get("flow_direction_state") is None and source_edge.get("flow_direction") is not None
        edge_node = {
            "id": edge_id,
            "src": str(source_edge.get("legacy_source", source_edge.get("source", ""))),
            "dst": str(source_edge.get("legacy_target", source_edge.get("target", ""))),
            "canonical_src": str(source_edge.get("source", "")),
            "canonical_dst": str(source_edge.get("target", "")),
            "type": map_edge_type(source_edge),
            "confidence": compute_edge_confidence(source_edge),
            "directed": str(source_edge.get("flow_direction_state") or source_edge.get("flow_direction") or "unknown").lower() in {"forward", "reverse", "source_to_target", "target_to_source"},
            "flow_direction_state": {"source_to_target": "forward", "target_to_source": "reverse"}.get(str(source_edge.get("flow_direction_state") or source_edge.get("flow_direction") or "unknown").lower(), str(source_edge.get("flow_direction_state") or source_edge.get("flow_direction") or "unknown").lower()),
            "flow_direction_confidence": source_edge.get("flow_direction_confidence"),
            "flow_direction_evidence": _json_safe(copy.deepcopy(source_edge.get("flow_direction_evidence") or [])),
            "flow_direction_review_state": source_edge.get("flow_direction_review_state"),
            "provenance": build_provenance(f"stage12 edge review_state={source_edge.get('review_state', '')}"),
            "geometry": {"polyline": reproject_polyline(source_edge.get("polyline", []))},
            # Preserve semantic evidence additively.  Existing consumers only
            # require the fields above; downstream line/package workflows need
            # the reviewed line identity and attachment evidence as well.
            "line_number_ids": [str(value) for value in line_number_ids if str(value)],
            "direct_line_number_ids": [str(value) for value in source_edge.get("direct_line_number_ids", []) if str(value)],
            "inferred_line_number_ids": [str(value) for value in source_edge.get("inferred_line_number_ids", []) if str(value)],
            "direct_line_numbers": copy.deepcopy(source_edge.get("direct_line_numbers", [])),
            "inferred_line_numbers": copy.deepcopy(source_edge.get("inferred_line_numbers", [])),
            "line_number_assignment_state": source_edge.get("line_number_assignment_state"),
            "line_numbers": copy.deepcopy(line_numbers),
            "attachments": _json_safe(copy.deepcopy(attachments)),
            "review_state": source_edge.get("review_state"),
            "line_number_review_state": source_edge.get("line_number_review_state"),
        }
        direction_state = edge_node["flow_direction_state"]
        if direction_state in {"forward", "reverse"}:
            edge_node["flow_src"] = edge_node["canonical_dst"] if direction_state == "reverse" else edge_node["canonical_src"]
            edge_node["flow_dst"] = edge_node["canonical_src"] if direction_state == "reverse" else edge_node["canonical_dst"]
        if direction_state == "reverse" and not legacy_flow_direction:
            edge_node["directed"] = False
        inline_ids = []
        for item in attachments.get("inline_objects", []) if isinstance(attachments.get("inline_objects"), list) else []:
            if isinstance(item, dict):
                value = str(item.get("source_object_id") or item.get("id") or "")
                if value and value not in inline_ids:
                    inline_ids.append(value)
        instrument_ids = []
        for item in attachments.get("instrument_tags", []) if isinstance(attachments.get("instrument_tags"), list) else []:
            if isinstance(item, dict):
                _, normalized = _record_text(item)
                value = str(item.get("canonical_instrument_id") or normalized or item.get("source_object_id") or item.get("id") or "")
                if value and value not in instrument_ids:
                    instrument_ids.append(value)
        inline_items = attachments.get("inline_objects", []) if isinstance(attachments.get("inline_objects"), list) else []
        inline_order = sorted(enumerate(inline_items), key=lambda pair: _attachment_order_key(pair[1], source_edge, pair[0]))
        inline_ids = []
        for _, item in inline_order:
            value = str(item.get("source_object_id") or item.get("id") or "") if isinstance(item, dict) else ""
            if value and value not in inline_ids:
                inline_ids.append(value)
        edge_node["inline_object_ids"] = [f"inline::{image_id or 'unknown'}::{value}" for value in inline_ids]
        edge_node["ordered_inline_object_ids"] = list(edge_node["inline_object_ids"])
        edge_node["ordered_inline_objects"] = [
            {"inline_object_id": f"inline::{image_id or 'unknown'}::{item.get('source_object_id') or item.get('id')}",
             "route_position_px": (None if _attachment_order_key(item, source_edge, index)[0] == float("inf") else _attachment_order_key(item, source_edge, index)[0])}
            for index, item in inline_order if isinstance(item, dict) and str(item.get("source_object_id") or item.get("id") or "")
        ]
        seen_inline_ids: set[str] = set()
        edge_node["ordered_inline_objects"] = [item for item in edge_node["ordered_inline_objects"] if not (item["inline_object_id"] in seen_inline_ids or seen_inline_ids.add(item["inline_object_id"]))]
        edge_node["instrument_ids"] = list(dict.fromkeys(f"instrument::{image_id or 'unknown'}::{value}" for value in instrument_ids))
        for source_key in ("source_equipment_id", "target_equipment_id", "terminal_equipment_id", "source_port_id", "target_port_id", "terminal_port_id"):
            if source_edge.get(source_key) is not None:
                edge_node[source_key] = str(source_edge[source_key])
        if edge_id in off_page_by_edge:
            edge_node["off_page_connector"] = off_page_by_edge[edge_id]
        edges.append(edge_node)

    line_to_edges: dict[str, list[str]] = {}
    for edge in edges:
        for line_id in edge.get("line_number_ids", []):
            line_to_edges.setdefault(str(line_id), []).append(edge["id"])

    semantic_projection = _build_semantic_projection(edges, instrument_tags_payload, image_id or "unknown")
    canonical_line_to_edges: dict[str, list[str]] = {}
    for edge in edges:
        for line_id in edge.get("canonical_line_ids", []):
            canonical_line_to_edges.setdefault(str(line_id), []).append(edge["id"])
    semantic_projection["canonical_line_to_edges"] = {
        key: sorted(value) for key, value in sorted(canonical_line_to_edges.items())
    }
    equipment_catalog = copy.deepcopy(stage12_graph.get("equipment_catalog") or stage12_graph.get("equipment") or [])
    port_catalog = copy.deepcopy(stage12_graph.get("equipment_ports") or stage12_graph.get("ports") or [])
    if not equipment_catalog:
        equipment_catalog = []
        referenced = {str(edge.get(key)) for edge in stage12_graph.get("edges", []) for key in ("source_equipment_id", "source_obj_id", "target_equipment_id", "target_obj_id", "terminal_equipment_id", "terminal_obj_id") if edge.get(key)}
        for obj in objects:
            obj_id = str(obj.get("id") or obj.get("det_id") or "")
            if referenced and obj_id not in referenced:
                continue
            if not obj_id:
                continue
            equipment_catalog.append({
                "id": obj_id,
                "class_name": obj.get("class_name"),
                "bbox": copy.deepcopy(obj.get("bbox")),
                "confidence": obj.get("confidence"),
                "provenance": copy.deepcopy(obj.get("provenance") or build_provenance("equipment catalog from object detection")),
                "review_state": obj.get("review_state"),
            })
    for equipment in equipment_catalog if isinstance(equipment_catalog, list) else []:
        if not isinstance(equipment, dict):
            continue
        source = object_by_id.get(str(equipment.get("source_object_id") or equipment.get("id") or ""))
        if source:
            for key in ("class_name", "bbox", "confidence", "review_state"):
                if source.get(key) is not None:
                    equipment[key] = copy.deepcopy(source[key])
            equipment.setdefault("provenance", copy.deepcopy(source.get("provenance") or build_provenance("equipment catalog enriched from object detection")))
    # Normalize catalogs before emitting relationships.  Persisted review data
    # can contain duplicate rows or endpoint references absent from catalogs;
    # retain those references as explicit unresolved records instead of
    # producing dangling graph edges.
    equipment_catalog = list({str(item.get("id")): item for item in equipment_catalog if isinstance(item, dict) and item.get("id")}.values())
    port_catalog = list({str(item.get("id")): item for item in port_catalog if isinstance(item, dict) and item.get("id")}.values())
    equipment_ids = {str(item["id"]) for item in equipment_catalog}
    port_ids = {str(item["id"]) for item in port_catalog}
    for edge in edges:
        for equipment_key in ("source_equipment_id", "target_equipment_id", "terminal_equipment_id"):
            value = edge.get(equipment_key)
            if value and str(value) not in equipment_ids:
                equipment_catalog.append({"id": str(value), "review_state": "unresolved", "provenance": build_provenance("synthesized from edge endpoint reference")})
                equipment_ids.add(str(value))
        for port_key, equipment_key in (("source_port_id", "source_equipment_id"), ("target_port_id", "target_equipment_id"), ("terminal_port_id", "terminal_equipment_id")):
            value = edge.get(port_key)
            if value and str(value) not in port_ids:
                owner = str(edge.get(equipment_key) or f"unresolved_equipment::{value}")
                port_catalog.append({"id": str(value), "equipment_id": owner, "review_state": "unresolved", "provenance": build_provenance("synthesized from edge port reference")})
                port_ids.add(str(value))
    for port in port_catalog:
        equipment_id = str(port.get("equipment_id") or "")
        if equipment_id and equipment_id not in equipment_ids:
            equipment_catalog.append({"id": equipment_id, "review_state": "unresolved", "provenance": build_provenance("synthesized from port equipment reference")})
            equipment_ids.add(equipment_id)
    port_owner = {str(port.get("id")): str(port.get("equipment_id")) for port in port_catalog if port.get("id") and port.get("equipment_id")}
    for port in port_catalog:
        if not isinstance(port, dict) or not port.get("id"):
            continue
        if not str(port.get("equipment_id") or "").strip():
            owner = f"unresolved_equipment::{port['id']}"
            port["equipment_id"] = owner
            port["review_state"] = "unresolved"
            port.setdefault("conflict_evidence", []).append({"reason": "missing_equipment_owner", "assigned_owner": owner})
            port["provenance"] = copy.deepcopy(port.get("provenance") or build_provenance("synthesized unresolved port owner"))
            port_owner[str(port["id"])] = owner
            if owner not in equipment_ids:
                equipment_catalog.append({"id": owner, "review_state": "unresolved", "provenance": build_provenance("synthesized from port with missing owner")})
                equipment_ids.add(owner)
    for edge in edges:
        conflicts = []
        for equipment_key, port_key in (("source_equipment_id", "source_port_id"), ("target_equipment_id", "target_port_id"), ("terminal_equipment_id", "terminal_port_id")):
            port_id = edge.get(port_key)
            owner = port_owner.get(str(port_id)) if port_id else None
            if not owner:
                continue
            current = str(edge.get(equipment_key) or "")
            if current and current != owner:
                conflicts.append({"field": equipment_key, "port_id": str(port_id), "observed_equipment_id": current, "canonical_equipment_id": owner})
                edge["review_state"] = "unresolved"
                edge["line_number_assignment_state"] = edge.get("line_number_assignment_state")
            edge[equipment_key] = owner
        if conflicts:
            edge["endpoint_conflicts"] = conflicts
    for port in port_catalog if isinstance(port_catalog, list) else []:
        if not isinstance(port, dict) or not port.get("id") or not port.get("equipment_id"):
            continue
        semantic_projection["relationships"].append({
            "id": f"has_port::{port['equipment_id']}::{port['id']}", "type": "has_port",
            "source": str(port["equipment_id"]), "target": str(port["id"]),
            "semantic_state": "unresolved" if str(port.get("review_state") or "") == "unresolved" else "observed", "provenance": copy.deepcopy(port.get("provenance") or build_provenance("equipment port catalog")),
        })
    for edge in edges:
        seen_endpoint_ports = set()
        for side, equipment_key, port_key in (("source", "source_equipment_id", "source_port_id"), ("target", "target_equipment_id", "target_port_id"), ("terminal", "terminal_equipment_id", "terminal_port_id")):
            port_id = edge.get(port_key)
            if port_id is None:
                continue
            if str(port_id) in seen_endpoint_ports:
                continue
            seen_endpoint_ports.add(str(port_id))
            semantic_projection["relationships"].append({
                "id": f"connects_to::{edge['id']}::{side}::{port_id}", "type": "connects_to",
                "source": edge["id"], "target": str(port_id), "edge_id": edge["id"],
                "endpoint": side, "equipment_id": port_owner.get(str(port_id)) or edge.get(equipment_key),
                "semantic_state": "unresolved" if edge.get("endpoint_conflicts") or str(port_owner.get(str(port_id)) or "").startswith("unresolved_equipment::") else "observed", "provenance": copy.deepcopy(edge.get("provenance") or {}),
            })
    semantic_projection["relationships"] = list({item["id"]: item for item in semantic_projection["relationships"]}.values())
    semantic_projection["relationships"].sort(key=lambda item: item["id"])

    payload = {
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
        "equipment": copy.deepcopy(equipment_catalog) if isinstance(equipment_catalog, list) else [],
        "equipment_ports": copy.deepcopy(port_catalog) if isinstance(port_catalog, list) else [],
        "line_to_edges": {key: sorted(value) for key, value in sorted(line_to_edges.items())},
        # Additive semantic identity projection.  The original edge and
        # attachment fields above remain the compatibility surface for older
        # consumers; these entities make OCR occurrences and route semantics
        # explicit for downstream package/HAZOP workflows.
        **semantic_projection,
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
    return _json_safe(payload)
