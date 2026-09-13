"""Deterministic, evidence-bound contexts for downstream LLM workflows.

This module deliberately stops at structured input preparation.  It does not
write process prose, infer chemistry, or fill in HAZOP causes, consequences,
or safeguards.  Callers may pass either a graph-v1 payload or the combined
graph-v2 payload produced by :mod:`pipe_sheet_merge`.
"""

from __future__ import annotations

import copy
import math
from typing import Any


_NODE_TYPE_BY_ROLE = {
    "equipment": "equipment",
    "equipment_port": "port",
    "instrumentation": "instrument",
    "inlet_outlet": "boundary_terminal",
    "ankle": "topology",
    "crossing": "topology",
}
_DEVIATION_DIMENSIONS = (
    "flow",
    "pressure",
    "temperature",
    "level",
    "composition",
    "phase",
    "utility",
)


def _release_state(graph_payload: dict[str, Any]) -> tuple[bool, str, list[str]]:
    payloads: list[dict[str, Any]] = []
    if isinstance(graph_payload, dict):
        for value in (graph_payload.get("release_gate"), graph_payload.get("stage9_release_gate")):
            if isinstance(value, dict) and value not in payloads:
                payloads.append(value)
        combined = graph_payload.get("combined_graph")
        if isinstance(combined, dict) and isinstance(combined.get("release_gate"), dict):
            if combined["release_gate"] not in payloads:
                payloads.append(combined["release_gate"])
    if not payloads:
        return False, "blocked", ["missing_release_gate"]
    reasons: set[str] = set()
    for payload in payloads:
        if payload.get("release_ready") is not True:
            raw_reasons = payload.get("blocking_reasons") or payload.get("blocking_issues") or []
            if not isinstance(raw_reasons, (list, tuple, set, frozenset)):
                raw_reasons = [raw_reasons]
            reasons.update(_text(reason) for reason in raw_reasons if _text(reason))
            if not raw_reasons:
                reasons.add("release_gate_not_ready")
    return (False, "blocked", sorted(reasons)) if reasons else (True, "released", [])


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe(value: Any) -> Any:
    """Return a JSON-safe deep copy while keeping input ordering irrelevant."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {
            str(key): _safe(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_safe(item) for item in sorted(value, key=lambda item: repr(item))]
    return copy.deepcopy(value)


def _id(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _state(record: dict[str, Any], default: str = "unresolved") -> str:
    # Candidate/released status describes the record's lifecycle; review_state
    # is retained separately when present and must not hide an explicit
    # candidate state from downstream consumers.
    for key in ("state", "semantic_state", "status", "review_state"):
        value = _text(record.get(key))
        if value:
            return value
    return default


def _confidence(record: dict[str, Any]) -> float | None:
    for key in ("confidence", "flow_direction_confidence"):
        value = _finite(record.get(key))
        if value is not None:
            return max(0.0, min(1.0, value))
    return None


def _flow_state(value: Any) -> str:
    aliases = {
        "source_to_target": "forward",
        "target_to_source": "reverse",
        "both": "bidirectional",
    }
    normalized = _text(value).lower()
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"forward", "reverse", "bidirectional", "unknown", "conflicting"} else "unknown"


def _provenance(record: dict[str, Any]) -> Any:
    value = record.get("provenance")
    return _safe(value) if value is not None else {"source": "input_record"}


def _graph_view(graph_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(graph_payload, dict):
        return {}
    combined = graph_payload.get("combined_graph")
    if isinstance(combined, dict) and (combined.get("nodes") or combined.get("edges")):
        return combined
    return graph_payload


def _catalog(graph: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = graph.get(key)
        if isinstance(value, dict):
            value = list(value.values())
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _drawing_ids(graph: dict[str, Any]) -> list[str]:
    values = graph.get("sheets") or []
    if isinstance(values, list):
        result = [_text(value) for value in values if _text(value)]
        if result:
            return sorted(set(result))
    drawings = graph.get("drawings") or []
    if isinstance(drawings, list):
        result = []
        for drawing in drawings:
            if isinstance(drawing, dict):
                value = _id(drawing, "sheet", "drawing_id", "doc_id")
                if value:
                    result.append(value)
        if result:
            return sorted(set(result))
    document = graph.get("document")
    value = _id(document, "doc_id", "drawing_id") if isinstance(document, dict) else ""
    return [value] if value else []


def _pixel_evidence(record: dict[str, Any], *, entity_id: str, kind: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    polyline = record.get("polyline")
    if polyline is None and isinstance(record.get("geometry"), dict):
        polyline = record["geometry"].get("polyline")
    if polyline is None:
        for key in ("pixel_route", "route", "trace"):
            if record.get(key) is not None:
                polyline = record[key]
                break
    if _has_pixel_route(polyline):
        evidence.append({
            "kind": "pixel_route",
            "entity_id": entity_id,
            "source": f"{kind}.geometry.polyline",
            "polyline": _safe(polyline),
        })
    position = record.get("position")
    if position is None and isinstance(record.get("geometry"), dict):
        position = record["geometry"].get("center")
    if position is not None:
        evidence.append({
            "kind": "pixel_position",
            "entity_id": entity_id,
            "source": f"{kind}.position",
            "position": _safe(position),
        })
    bbox = record.get("bbox")
    if bbox is not None:
        evidence.append({
            "kind": "pixel_bbox",
            "entity_id": entity_id,
            "source": f"{kind}.bbox",
            "bbox": _safe(bbox),
        })
    return evidence


def _has_pixel_route(polyline: Any) -> bool:
    if not isinstance(polyline, list) or not polyline:
        return False
    for point in polyline:
        if isinstance(point, dict):
            x, y = _finite(point.get("x", point.get("col"))), _finite(point.get("y", point.get("row")))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = _finite(point[0]), _finite(point[1])
        else:
            continue
        if x is not None and y is not None:
            return True
    return False


def _entity(record: dict[str, Any], *, entity_type: str, entity_id: str | None = None) -> dict[str, Any]:
    value = entity_id or _id(record, "id", "canonical_id", "source_object_id")
    item = {
        "id": value,
        "type": entity_type,
        "state": _state(record),
        "confidence": _confidence(record),
        "provenance": _provenance(record),
        "pixel_evidence": _pixel_evidence(record, entity_id=value, kind=entity_type),
    }
    for key in ("sheet", "drawing_id", "local_id", "class_name", "tag", "normalized_tag", "display_text", "normalized_text"):
        if record.get(key) is not None and record.get(key) != "":
            item[key] = _safe(record[key])
    return item


def _list_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = _id(value, "id", "canonical_id", "source_object_id", "line_number_id", "instrument_id", "inline_object_id")
        value = _text(value)
        if value and value not in result:
            result.append(value)
    return result


def _value_ids(*values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate = _id(candidate, "id", "canonical_id", "source_object_id")
            candidate = _text(candidate)
            if candidate and candidate not in result:
                result.append(candidate)
    return result


def _attachment_ids(edge: dict[str, Any], group: str, direct_keys: tuple[str, ...]) -> list[str]:
    result = _value_ids(edge.get(direct_keys[0])) if direct_keys else []
    for key in direct_keys[1:]:
        result.extend(value for value in _value_ids(edge.get(key)) if value not in result)
    attachments = edge.get("attachments")
    if isinstance(attachments, dict):
        for raw in attachments.get(group, []) or []:
            if isinstance(raw, dict):
                value = _id(raw, "canonical_id", "canonical_instrument_id", "source_object_id", "id")
                if value and value not in result:
                    result.append(value)
    return result


def _route(edge: dict[str, Any], index: int) -> dict[str, Any]:
    edge_id = _id(edge, "id", "edge_id") or f"edge::{index}"
    source = _id(edge, "source", "canonical_src", "src")
    target = _id(edge, "target", "canonical_dst", "dst")
    line_ids = _list_ids(edge.get("canonical_line_ids"))
    line_ids.extend(value for value in _list_ids(edge.get("line_number_ids")) if value not in line_ids)
    equipment_ids = _value_ids(edge.get("equipment_ids"), edge.get("source_equipment_id"), edge.get("target_equipment_id"), edge.get("terminal_equipment_id"))
    instrument_ids = _attachment_ids(edge, "instrument_tags", ("instrument_ids",))
    inline_ids = _attachment_ids(edge, "inline_objects", ("ordered_inline_object_ids", "inline_object_ids"))
    flow_state = _flow_state(edge.get("flow_direction_state") or edge.get("flow_direction") or "unknown")
    return {
        "id": edge_id,
        "order": index,
        "source": source,
        "target": target,
        "state": _state(edge),
        "confidence": _confidence(edge),
        "provenance": _provenance(edge),
        "line_ids": line_ids,
        "line_number_ids": _list_ids(edge.get("line_number_ids")),
        "equipment_ids": equipment_ids,
        "instrument_ids": instrument_ids,
        "inline_object_ids": inline_ids,
        "flow": {
            "state": flow_state or "unknown",
            "source": _id(edge, "flow_src") or source,
            "target": _id(edge, "flow_dst") or target,
            "confidence": _confidence({"confidence": edge.get("flow_direction_confidence")}),
            "evidence": _safe(edge.get("flow_direction_evidence") or []),
            "review_state": _safe(edge.get("flow_direction_review_state")),
        },
        "trace_length_px": _finite(edge.get("trace_length_px")),
        "pixel_evidence": _pixel_evidence(edge, entity_id=edge_id, kind="edge"),
    }


def _relationship(record: dict[str, Any], index: int) -> dict[str, Any]:
    rel_id = _id(record, "id", "relationship_id") or f"relationship::{index}"
    item = {
        "id": rel_id,
        "type": _text(record.get("type") or record.get("relationship_type") or "unknown"),
        "source": _id(record, "source", "from"),
        "target": _id(record, "target", "to"),
        "state": _state(record),
        "confidence": _confidence(record),
        "provenance": _provenance(record),
    }
    for key in ("edge_id", "connector_ids", "evidence", "match_evidence", "boundary_flow"):
        if record.get(key) is not None:
            item[key] = _safe(record[key])
    return item


def _input_entities(payload: dict[str, Any] | None, *, key: str, entity_type: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    values = payload.get(key)
    nested_key = "process_boundaries" if key == "boundaries" else "test_packages"
    if values is None:
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            values = nested.get("candidates")
            if values is None:
                values = nested.get(key)
        elif isinstance(nested, list):
            values = nested
    if values is None:
        values = payload.get("candidates")
    if values is None:
        values = payload.get("items")
    if isinstance(values, dict):
        values = values.get("candidates") if "candidates" in values else list(values.values())
    if not isinstance(values, list):
        return []
    result = []
    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            continue
        value = _id(raw, "id", "boundary_id", "test_package_id", "package_id") or f"{entity_type}::{index}"
        item = _entity(raw, entity_type=entity_type, entity_id=value)
        for field in (
            "kind", "state", "review_state", "member_ids", "member_node_ids", "member_edge_ids",
            "member_line_ids", "member_line_number_ids", "member_equipment_ids",
            "member_inline_object_ids", "member_instrument_ids", "member_relationship_ids",
            "members", "routes", "cut_points", "isolation_elements", "exclusions", "uncertainty",
            "line_ids", "line_number_ids", "edge_ids", "equipment_ids", "relationship_ids",
            "flow_direction_state", "review_questions",
        ):
            if raw.get(field) is not None:
                item[field] = _safe(raw[field])
        result.append(item)
    return sorted(result, key=lambda item: item["id"])


def _candidate_questions(items: list[dict[str, Any]], entity_type: str) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for item in items:
        candidate_id = item["id"]
        raw_questions = item.get("review_questions") or []
        raw_uncertainty = item.get("uncertainty") or []
        if not isinstance(raw_questions, list):
            raw_questions = [raw_questions]
        if not isinstance(raw_uncertainty, list):
            raw_uncertainty = [raw_uncertainty]
        for index, raw in enumerate([*raw_questions, *raw_uncertainty]):
            if isinstance(raw, dict):
                kind = _text(raw.get("type") or raw.get("kind") or "candidate_uncertainty")
                reference = _id(raw, "edge_id", "route_id", "node_id", "line_id", "connector_id") or str(index)
                question = {
                    "kind": kind,
                    "status": "unresolved",
                    "candidate_id": candidate_id,
                    "evidence": _safe(raw),
                }
            else:
                kind = "candidate_review_question"
                reference = str(index)
                question = {"kind": kind, "status": "unresolved", "candidate_id": candidate_id, "question": _text(raw)}
            question["id"] = f"gap::{entity_type}::{candidate_id}::{kind}::{reference}"
            questions.append(question)
    unique = {item["id"]: item for item in questions}
    return [unique[key] for key in sorted(unique)]


def _graph_context(graph_payload: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_view(graph_payload)
    nodes = _catalog(graph, "nodes")
    edges = _catalog(graph, "edges")
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = _id(node, "id", "node_id")
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        raw_type = _text(node.get("type") or node.get("kind")).lower()
        entity_type = _NODE_TYPE_BY_ROLE.get(raw_type, raw_type or "node")
        entities.append(_entity(node, entity_type=entity_type, entity_id=node_id))
    for key, entity_type in (("equipment", "equipment"), ("equipment_catalog", "equipment"), ("equipment_ports", "port"), ("lines", "line"), ("inline_objects", "inline_object"), ("instruments", "instrument"), ("connectors", "off_page_connector")):
        for record in _catalog(graph, key):
            record_id = _id(record, "id", "canonical_id", "equipment_id", "port_id", "line_id", "instrument_id")
            if not record_id or record_id in seen:
                continue
            seen.add(record_id)
            entities.append(_entity(record, entity_type=entity_type, entity_id=record_id))
    routes = [_route(edge, index) for index, edge in enumerate(sorted(edges, key=lambda item: _id(item, "id", "edge_id")))]
    relationships = [_relationship(record, index) for index, record in enumerate(sorted(_catalog(graph, "relationships"), key=lambda item: _id(item, "id", "relationship_id")))]
    return _safe({
        "graph_schema_version": _text(graph.get("schema_version")) or "unknown",
        "drawing_ids": _drawing_ids(graph),
        "entities": sorted(entities, key=lambda item: (item["type"], item["id"])),
        "routes": routes,
        "relationships": relationships,
        "graph_issues": _safe(graph.get("issues") or []),
        "release_gate": _safe(
            graph.get("release_gate")
            or graph.get("stage9_release_gate")
            or graph_payload.get("release_gate")
            or graph_payload.get("stage9_release_gate")
            or {}
        ),
    })


def _empty_graph_context() -> dict[str, Any]:
    """Return a redacted graph container for an unreleased graph.

    The release gate itself is surfaced by the parent projection.  Keeping the
    graph container empty prevents a caller from accidentally treating
    provisional routes, entities, relationships, or drawing metadata as LLM
    input when the graph has not passed review.
    """
    return {
        "graph_schema_version": "unreleased",
        "drawing_ids": [],
        "entities": [],
        "routes": [],
        "relationships": [],
        "graph_issues": [],
        "release_gate": {},
    }


def _evidence_gaps(context: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for route in context["routes"]:
        if not route["line_ids"]:
            gaps.append({"id": f"gap::line::{route['id']}", "kind": "line_identity", "status": "unresolved", "route_id": route["id"]})
        if route["flow"]["state"] in {"unknown", "conflicting", ""}:
            gaps.append({"id": f"gap::flow::{route['id']}", "kind": "flow_direction", "status": "unresolved", "route_id": route["id"]})
        if not any(item.get("kind") == "pixel_route" for item in route["pixel_evidence"]):
            gaps.append({"id": f"gap::pixel_route::{route['id']}", "kind": "pixel_route", "status": "unresolved", "route_id": route["id"]})
    return gaps


def build_process_description_context(
    *,
    graph_payload: dict[str, Any],
    boundary_payload: dict[str, Any] | None = None,
    test_package_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured facts for a downstream process-description LLM."""
    release_ready, release_state, release_reasons = _release_state(graph_payload)
    if release_ready:
        context = _graph_context(graph_payload)
        boundaries = _input_entities(boundary_payload, key="boundaries", entity_type="boundary")
        packages = _input_entities(test_package_payload, key="test_packages", entity_type="test_package")
        context["entities"].extend(boundaries + packages)
        context["entities"] = sorted(context["entities"], key=lambda item: (item["type"], item["id"]))
    else:
        context = _empty_graph_context()
        boundaries = []
        packages = []
    gaps = _evidence_gaps(context)
    if not release_ready:
        gaps.append({
            "id": "gap::release_gate",
            "kind": "release_gate",
            "status": "unresolved",
            "reasons": release_reasons,
        })
    gaps.extend(_candidate_questions(boundaries, "boundary"))
    gaps.extend(_candidate_questions(packages, "test_package"))
    gaps = {item["id"]: item for item in gaps}
    return {
        "schema_version": "llm_process_context_v1",
        "context_type": "process_description",
        "state": release_state,
        "release_ready": release_ready,
        "blocked_reasons": release_reasons,
        "generation_policy": {
            "structured_facts_only": True,
            "prose_generation": "downstream_only",
            "chemistry_inference": "forbidden",
            "flow_inference_from_route_order": "forbidden",
        },
        "graph": context,
        "boundaries": boundaries,
        "test_packages": packages,
        "unresolved_assumptions": [],
        "unresolved_questions": [gaps[key] for key in sorted(gaps)],
    }


def build_hazop_context(
    *,
    graph_payload: dict[str, Any],
    boundary_payload: dict[str, Any] | None = None,
    test_package_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build candidate HAZOP inputs without producing engineering judgments."""
    process = build_process_description_context(
        graph_payload=graph_payload,
        boundary_payload=boundary_payload,
        test_package_payload=test_package_payload,
    )
    gaps = list(process["unresolved_questions"])
    if not process["release_ready"]:
        return _safe({
            "schema_version": "llm_hazop_context_v1",
            "context_type": "hazop_input_scaffold",
            "state": process["state"],
            "release_ready": False,
            "blocked_reasons": process["blocked_reasons"],
            "generation_policy": {
                "candidate_scaffold_only": True,
                "causes": "not_generated",
                "consequences": "not_generated",
                "safeguards": "not_generated",
                "design_conditions": "not_generated",
                "chemistry_inference": "forbidden",
            },
            "graph": process["graph"],
            "boundaries": [],
            "test_packages": [],
            "candidate_nodes": [],
            "candidate_segments": [],
            "deviation_dimensions": [],
            "evidence_gaps": gaps,
            "unresolved_assumptions": [],
            "unresolved_questions": gaps,
        })
    dimensions = []
    for name in _DEVIATION_DIMENSIONS:
        gap_id = f"gap::design_data::{name}"
        dimensions.append({
            "id": f"deviation::{name}",
            "dimension": name,
            "status": "candidate",
            "review_state": "unresolved",
            "evidence_ids": [],
            "unresolved_question_id": gap_id,
        })
        gaps.append({
            "id": gap_id,
            "kind": "design_condition_or_process_data",
            "dimension": name,
            "status": "unresolved",
        })
    return _safe({
        "schema_version": "llm_hazop_context_v1",
        "context_type": "hazop_input_scaffold",
        "state": process["state"],
        "release_ready": process["release_ready"],
        "blocked_reasons": process["blocked_reasons"],
        "generation_policy": {
            "candidate_scaffold_only": True,
            "causes": "not_generated",
            "consequences": "not_generated",
            "safeguards": "not_generated",
            "design_conditions": "not_generated",
            "chemistry_inference": "forbidden",
        },
        "graph": process["graph"],
        "boundaries": process["boundaries"],
        "test_packages": process["test_packages"],
        "candidate_nodes": [item for item in process["graph"]["entities"] if item["type"] in {"equipment", "boundary", "boundary_terminal"}],
        "candidate_segments": process["graph"]["routes"],
        "deviation_dimensions": dimensions,
        "evidence_gaps": gaps,
        "unresolved_assumptions": [],
        "unresolved_questions": gaps,
    })


def build_llm_projections(
    *,
    graph_payload: dict[str, Any],
    boundary_payload: dict[str, Any] | None = None,
    test_package_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return both supported downstream LLM contexts from one graph snapshot."""
    return _safe({
        "schema_version": "llm_projections_v1",
        "process_description": build_process_description_context(
            graph_payload=graph_payload,
            boundary_payload=boundary_payload,
            test_package_payload=test_package_payload,
        ),
        "hazop": build_hazop_context(
            graph_payload=graph_payload,
            boundary_payload=boundary_payload,
            test_package_payload=test_package_payload,
        ),
    })


__all__ = [
    "build_hazop_context",
    "build_llm_projections",
    "build_process_description_context",
]
