from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

_EQUIPMENT_NODE_TYPES = {"equipment", "equipment_port"}


def _edge_length(edge: dict[str, Any]) -> float:
    try:
        return float(edge.get("trace_length_px") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _line_ids(edge: dict[str, Any]) -> list[str]:
    values = edge.get("effective_line_number_ids") or edge.get("line_number_ids") or []
    ids = [str(value) for value in values if str(value)]
    return ids or ["unassigned"]


def _line_records(edge: dict[str, Any]) -> list[dict[str, Any]]:
    records = edge.get("effective_line_numbers") or edge.get("line_numbers") or []
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _direct_line_records(edge: dict[str, Any]) -> list[dict[str, Any]]:
    records = edge.get("line_numbers") or []
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _line_texts(edge: dict[str, Any], key: str) -> list[str]:
    values: list[str] = []
    for record in _line_records(edge):
        value = str(record.get(key) or "")
        if value and value not in values:
            values.append(value)
    return values


def _merge_unique(values: list[list[str]]) -> list[str]:
    result: list[str] = []
    for group in values:
        for value in group:
            if value and value not in result:
                result.append(value)
    return result


def _line_record_id(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("source_object_id") or "")


def _normalize_line_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _line_record_id(record),
        "source_object_id": record.get("source_object_id"),
        "display_text": record.get("display_text") or record.get("text") or record.get("normalized_text") or "",
        "normalized_text": record.get("normalized_text") or record.get("display_text") or record.get("text") or "",
    }


def _candidate_line_records(edge: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for record in _direct_line_records(edge) + _line_records(edge):
        normalized = _normalize_line_record(record)
        if normalized["id"]:
            candidates.setdefault(normalized["id"], normalized)
    return list(candidates.values())


def _choose_line_for_inline_occurrence(edge: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    direct = [_normalize_line_record(record) for record in _direct_line_records(edge)]
    direct = [record for record in direct if record["id"]]
    candidates = _candidate_line_records(edge)
    if direct:
        return sorted(direct, key=lambda record: record["id"])[0], candidates
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def _select_mto_line(item: dict[str, Any]) -> None:
    selected = item.pop("_selected_line_record", None)
    candidates = item.pop("_candidate_line_records", {})
    item["candidate_line_numbers"] = sorted(candidates.values(), key=lambda record: record["id"])
    if selected is None:
        item["line_number_ids"] = []
        item["line_number_texts"] = []
        item["normalized_line_number_texts"] = []
        item["line_number_assignment_state"] = "ambiguous" if item["candidate_line_numbers"] else "missing"
        return
    item["line_number_ids"] = [selected["id"]]
    item["line_number_texts"] = [selected["display_text"]] if selected.get("display_text") else []
    item["normalized_line_number_texts"] = [selected["normalized_text"]] if selected.get("normalized_text") else []
    item["line_number_assignment_state"] = "selected"


def _is_synthetic_inline_observation(obj: dict[str, Any]) -> bool:
    item_id = str(obj.get("id") or "")
    source = str(obj.get("source") or "")
    return source == "stage5b_hit" or ":hit_" in item_id


def _physical_inline_id(obj: dict[str, Any]) -> str:
    return str(obj.get("source_object_id") or obj.get("id") or "")


def _pending_property_basis(line_number_ids: list[str], property_name: str) -> dict[str, Any]:
    return {
        "status": "pending_line_property_data",
        "property": property_name,
        "line_number_ids": line_number_ids,
        "source": "future_line_property_table",
    }


def _nodes_by_id(graph_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("id")): node
        for node in graph_payload.get("nodes", []) or []
        if isinstance(node, dict) and str(node.get("id") or "")
    }


def _bbox_xyxy(bbox: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, dict):
        return None
    if {"x_min", "y_min", "x_max", "y_max"}.issubset(bbox):
        return (
            int(round(float(bbox.get("x_min", 0)))),
            int(round(float(bbox.get("y_min", 0)))),
            int(round(float(bbox.get("x_max", 0)))),
            int(round(float(bbox.get("y_max", 0)))),
        )
    if {"x", "y", "w", "h"}.issubset(bbox):
        x = int(round(float(bbox.get("x", 0))))
        y = int(round(float(bbox.get("y", 0))))
        w = int(round(float(bbox.get("w", 0))))
        h = int(round(float(bbox.get("h", 0))))
        return x, y, x + w, y + h
    return None


def _build_line_list(image_id: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        for line_id in _line_ids(edge):
            grouped[line_id].append(edge)

    lines = []
    for line_id in sorted(grouped):
        line_edges = sorted(grouped[line_id], key=lambda edge: str(edge.get("id") or ""))
        node_ids = sorted(
            {
                str(value)
                for edge in line_edges
                for value in (edge.get("source"), edge.get("target"))
                if str(value or "")
            }
        )
        lines.append(
            {
                "line_number_id": line_id,
                "assignment_state": "missing" if line_id == "unassigned" else "assigned",
                "edge_ids": [str(edge.get("id")) for edge in line_edges],
                "node_ids": node_ids,
                "display_texts": _merge_unique([_line_texts(edge, "display_text") for edge in line_edges]),
                "normalized_texts": _merge_unique([_line_texts(edge, "normalized_text") for edge in line_edges]),
                "total_length_px": round(sum(_edge_length(edge) for edge in line_edges), 3),
                "edge_count": len(line_edges),
            }
        )
    return {"image_id": image_id, "source": "stage15_process_exports", "lines": lines}


def _build_equipment_connectivity(image_id: str, nodes_by_id: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    by_line: dict[str, set[str]] = defaultdict(set)
    direct_connections = []
    for edge in edges:
        endpoints = [str(edge.get("source") or ""), str(edge.get("target") or "")]
        equipment_nodes = [node_id for node_id in endpoints if str(nodes_by_id.get(node_id, {}).get("type") or "") in _EQUIPMENT_NODE_TYPES]
        for line_id in _line_ids(edge):
            by_line[line_id].update(equipment_nodes)
        if equipment_nodes:
            direct_connections.append(
                {
                    "edge_id": str(edge.get("id") or ""),
                    "line_number_ids": _line_ids(edge),
                    "line_number_texts": _line_texts(edge, "display_text"),
                    "normalized_line_number_texts": _line_texts(edge, "normalized_text"),
                    "equipment_node_ids": equipment_nodes,
                    "source": endpoints[0],
                    "target": endpoints[1],
                }
            )

    line_connections = []
    for line_id in sorted(by_line):
        equipment_node_ids = sorted(by_line[line_id])
        if equipment_node_ids:
            line_connections.append({"line_number_id": line_id, "equipment_node_ids": equipment_node_ids})
    return {
        "image_id": image_id,
        "source": "stage15_process_exports",
        "connections": line_connections,
        "direct_edge_connections": direct_connections,
    }


def _build_inline_mto(image_id: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
    items_by_id: dict[str, dict[str, Any]] = {}
    for edge in edges:
        edge_id = str(edge.get("id") or "")
        for obj in (edge.get("attachments") or {}).get("inline_objects", []) or []:
            if not isinstance(obj, dict):
                continue
            if _is_synthetic_inline_observation(obj):
                continue
            item_id = _physical_inline_id(obj)
            if not item_id:
                continue
            item = items_by_id.get(item_id)
            if item is None:
                item = {
                    "id": item_id,
                    "source_object_id": obj.get("source_object_id", item_id),
                    "class_name": str(obj.get("class_name") or "inline_object"),
                    "edge_ids": [],
                    "_candidate_line_records": {},
                    "_selected_line_record": None,
                    "bbox": obj.get("bbox"),
                    "confidence": obj.get("confidence"),
                }
                items_by_id[item_id] = item
            if edge_id and edge_id not in item["edge_ids"]:
                item["edge_ids"].append(edge_id)
            selected, candidates = _choose_line_for_inline_occurrence(edge)
            for candidate in candidates:
                item["_candidate_line_records"].setdefault(candidate["id"], candidate)
            if item["_selected_line_record"] is None and selected is not None:
                item["_selected_line_record"] = selected
    items = list(items_by_id.values())
    for item in items:
        _select_mto_line(item)
        item["material_basis"] = _pending_property_basis(item["line_number_ids"], "material")
        item["design_condition_basis"] = _pending_property_basis(item["line_number_ids"], "design_conditions")
    class_counts = Counter(item["class_name"] for item in items)
    return {
        "image_id": image_id,
        "source": "stage15_process_exports",
        "scope": "unique_physical_inline_objects_only",
        "items": sorted(items, key=lambda item: (item["class_name"], item["id"])),
        "class_counts": dict(class_counts),
    }


def _build_inline_observations(image_id: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for edge in edges:
        edge_id = str(edge.get("id") or "")
        for obj in (edge.get("attachments") or {}).get("inline_objects", []) or []:
            if not isinstance(obj, dict):
                continue
            items.append(
                {
                    "id": str(obj.get("id") or obj.get("source_object_id") or f"{edge_id}::inline"),
                    "source_object_id": obj.get("source_object_id"),
                    "class_name": str(obj.get("class_name") or "inline_object"),
                    "edge_id": edge_id,
                    "line_number_ids": _line_ids(edge),
                    "line_number_texts": _line_texts(edge, "display_text"),
                    "normalized_line_number_texts": _line_texts(edge, "normalized_text"),
                    "is_synthetic": _is_synthetic_inline_observation(obj),
                    "source": obj.get("source"),
                    "bbox": obj.get("bbox"),
                    "hit_xy": obj.get("hit_xy"),
                    "projected_xy": obj.get("projected_xy"),
                }
            )
    return {
        "image_id": image_id,
        "source": "stage15_process_exports",
        "scope": "all_inline_graph_observations_for_qa",
        "items": items,
    }


def _build_instrument_index(image_id: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
    items_by_id: dict[str, dict[str, Any]] = {}
    for edge in edges:
        edge_id = str(edge.get("id") or "")
        for inst in (edge.get("attachments") or {}).get("instrument_tags", []) or []:
            if not isinstance(inst, dict):
                continue
            instrument_id = str(inst.get("id") or inst.get("source_object_id") or f"{edge_id}::instrument")
            item = items_by_id.get(instrument_id)
            if item is None:
                item = {
                    "instrument_id": instrument_id,
                    "source_object_id": inst.get("source_object_id"),
                    "edge_ids": [],
                    "line_number_ids": [],
                    "line_number_texts": [],
                    "normalized_line_number_texts": [],
                    "bbox": inst.get("bbox"),
                    "text": inst.get("text"),
                    "normalized_text": inst.get("normalized_text"),
                }
                items_by_id[instrument_id] = item
            if edge_id and edge_id not in item["edge_ids"]:
                item["edge_ids"].append(edge_id)
            for line_id in _line_ids(edge):
                if line_id not in item["line_number_ids"]:
                    item["line_number_ids"].append(line_id)
            for text in _line_texts(edge, "display_text"):
                if text not in item["line_number_texts"]:
                    item["line_number_texts"].append(text)
            for text in _line_texts(edge, "normalized_text"):
                if text not in item["normalized_line_number_texts"]:
                    item["normalized_line_number_texts"].append(text)
    items = list(items_by_id.values())
    return {"image_id": image_id, "source": "stage15_process_exports", "items": sorted(items, key=lambda item: item["instrument_id"])}


def build_stage15_process_exports(*, image_id: str, corrected_graph_payload: dict[str, Any]) -> dict[str, Any]:
    edges = [edge for edge in corrected_graph_payload.get("edges", []) or [] if isinstance(edge, dict)]
    nodes_by_id = _nodes_by_id(corrected_graph_payload)
    line_list_payload = _build_line_list(image_id, edges)
    equipment_connectivity_payload = _build_equipment_connectivity(image_id, nodes_by_id, edges)
    inline_mto_payload = _build_inline_mto(image_id, edges)
    inline_observations_payload = _build_inline_observations(image_id, edges)
    instrument_index_payload = _build_instrument_index(image_id, edges)
    summary = {
        "image_id": image_id,
        "line_count": len(line_list_payload["lines"]),
        "equipment_connection_count": len(equipment_connectivity_payload["connections"]),
        "direct_equipment_edge_count": len(equipment_connectivity_payload["direct_edge_connections"]),
        "inline_item_count": len(inline_mto_payload["items"]),
        "inline_observation_count": len(inline_observations_payload["items"]),
        "instrument_item_count": len(instrument_index_payload["items"]),
        "unassigned_line_edge_count": sum(1 for edge in edges if _line_ids(edge) == ["unassigned"]),
    }
    return {
        "line_list_payload": line_list_payload,
        "equipment_connectivity_payload": equipment_connectivity_payload,
        "inline_mto_payload": inline_mto_payload,
        "inline_observations_payload": inline_observations_payload,
        "instrument_index_payload": instrument_index_payload,
        "summary": summary,
    }


def render_stage15_inline_mto_overlay(image_bgr: Any, inline_mto_payload: dict[str, Any]) -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to render stage15_inline_mto_overlay") from exc

    overlay = image_bgr.copy()
    color = (0, 128, 255)
    for item in inline_mto_payload.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        bbox = _bbox_xyxy(item.get("bbox"))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        label = f"{item.get('id', '')} {item.get('class_name', '')}".strip()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.circle(overlay, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 3, color, -1)
        cv2.putText(
            overlay,
            label[:80],
            (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _edge_lookup(graph_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(edge.get("id")): edge
        for edge in graph_payload.get("edges", []) or []
        if isinstance(edge, dict) and str(edge.get("id") or "")
    }


def _polyline_points(edge: dict[str, Any]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for point in edge.get("polyline", []) or []:
        if not isinstance(point, dict):
            continue
        points.append((int(round(float(point.get("x", 0)))), int(round(float(point.get("y", 0))))))
    return points


def render_stage15_line_number_overlay(
    image_bgr: Any,
    line_list_payload: dict[str, Any],
    corrected_graph_payload: dict[str, Any],
) -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to render stage15_line_number_overlay") from exc

    overlay = image_bgr.copy()
    edges_by_id = _edge_lookup(corrected_graph_payload)
    palette = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]
    for index, line in enumerate(line_list_payload.get("lines", []) or []):
        if not isinstance(line, dict):
            continue
        color = palette[index % len(palette)]
        label_point: tuple[int, int] | None = None
        for edge_id in line.get("edge_ids", []) or []:
            edge = edges_by_id.get(str(edge_id))
            if edge is None:
                continue
            points = _polyline_points(edge)
            if len(points) < 2:
                continue
            for start, end in zip(points, points[1:]):
                cv2.line(overlay, start, end, color, 2, cv2.LINE_AA)
            if label_point is None:
                label_point = points[len(points) // 2]
        if label_point is None:
            continue
        display_texts = [str(text) for text in line.get("display_texts", []) or [] if str(text)]
        first_text = display_texts[0] if display_texts else str(line.get("line_number_id") or "unassigned")
        extra_count = max(0, len(display_texts) - 1)
        suffix = f" +{extra_count}" if extra_count else ""
        label = f"{line.get('line_number_id', '')}: {first_text}{suffix}"
        cv2.circle(overlay, label_point, 4, color, -1)
        cv2.putText(
            overlay,
            label[:100],
            (label_point[0] + 6, label_point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay
