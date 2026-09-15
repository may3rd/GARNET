"""Phase 8 engineering views derived from a released PID graph.

The views in this module deliberately remain candidate views.  They preserve
the graph entity IDs and route geometry needed by a reviewer while keeping
missing line numbers, unknown flow, and unresolved connector semantics
explicit.  No topology or engineering fact is invented here.
"""

from __future__ import annotations

import math
import copy
from collections import defaultdict
from typing import Any, Iterable


_RELEASED_STATES = {"released", "release_ready", "ready", "accepted", "human_reviewed", "reviewed"}
_TERMINAL_TYPES = {
    "connection",
    "connection_terminal",
    "dead_end",
    "feed",
    "inlet",
    "inlet_outlet",
    "off_page_connector",
    "outlet",
    "page_connection",
    "terminal",
    "utility_connection",
}
_ISOLATION_WORDS = ("isolation", "block", "gate", "ball", "butterfly", "plug", "knife")
_NON_ISOLATION_WORDS = ("check", "control", "relief", "safety", "regulating")


def _as_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = list(value.values())
    return [item for item in value or [] if isinstance(item, dict)]


def _safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {
            str(key): _safe_json(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_safe_json(item) for item in sorted(value, key=lambda item: repr(item))]
    return copy.deepcopy(value)


def _entity_id(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _graph_scope(graph: dict[str, Any]) -> str:
    if str(graph.get("schema_version") or "").startswith("graph_v2_combined"):
        return "combined"
    document = graph.get("document")
    if isinstance(document, dict):
        return str(document.get("doc_id") or graph.get("image_id") or "drawing")
    return str(graph.get("image_id") or "drawing")


def _release_state(graph: dict[str, Any], gate: dict[str, Any] | None) -> tuple[bool, str, list[str]]:
    payloads: list[dict[str, Any]] = []
    if isinstance(gate, dict):
        payloads.append(gate)
    for candidate in (graph.get("release_gate"), graph.get("stage9_release_gate")):
        if isinstance(candidate, dict) and candidate not in payloads:
            payloads.append(candidate)
    if not payloads:
        return False, "blocked", ["missing_release_gate"]

    # An explicitly supplied gate is allowed for callers that keep the gate in
    # a separate artifact, but a stale blocked gate already embedded in the
    # graph must not be bypassed by that argument.
    reasons: set[str] = set()
    for payload in payloads:
        if payload.get("release_ready") is not True:
            raw_reasons = payload.get("blocking_reasons") or payload.get("blocking_issues") or []
            if not isinstance(raw_reasons, (list, tuple, set, frozenset)):
                raw_reasons = [raw_reasons]
            reasons.update(str(reason) for reason in raw_reasons if str(reason).strip())
            if not raw_reasons:
                reasons.add("release_gate_not_ready")
    if reasons:
        return False, "blocked", sorted(reasons)
    return True, "released", []


def _node_records(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for node in _as_records(graph.get("nodes")):
        node_id = _entity_id(node, "id", "node_id")
        if node_id:
            result[node_id] = node
    return result


def _edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    return (
        _entity_id(edge, "source", "src", "canonical_src"),
        _entity_id(edge, "target", "dst", "canonical_dst"),
    )


def _edge_records(graph: dict[str, Any]) -> list[dict[str, Any]]:
    edges = _as_records(graph.get("edges"))
    return sorted(edges, key=lambda edge: _entity_id(edge, "id", "edge_id"))


def _polyline(edge: dict[str, Any]) -> list[dict[str, float]]:
    raw = edge.get("polyline")
    if not isinstance(raw, list):
        raw = (edge.get("geometry") or {}).get("polyline") if isinstance(edge.get("geometry"), dict) else []
    points: list[dict[str, float]] = []
    for point in raw or []:
        if isinstance(point, dict):
            try:
                x = float(point.get("x"))
                y = float(point.get("y"))
            except (TypeError, ValueError):
                continue
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                continue
        else:
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append({"x": x, "y": y})
    return points


def _line_ids(edge: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("canonical_line_ids", "effective_line_number_ids", "line_number_ids", "line_ids"):
        candidate = edge.get(key)
        if isinstance(candidate, list):
            values.extend(candidate)
            if values:
                break
    result = sorted({str(value) for value in values if str(value).strip() and str(value) != "unassigned"})
    return result


def _line_number_ids(edge: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("effective_line_number_ids", "line_number_ids", "line_ids"):
        candidate = edge.get(key)
        if isinstance(candidate, list):
            values.extend(candidate)
    return sorted({str(value) for value in values if str(value).strip() and str(value) != "unassigned"})


def _line_records(edge: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for key in ("effective_line_numbers", "line_numbers", "direct_line_numbers", "inferred_line_numbers"):
        for record in _as_records(edge.get(key)):
            if record not in values:
                values.append(record)
    return sorted(
        [_safe_json(record) for record in values],
        key=lambda item: _entity_id(item, "id", "source_object_id", "canonical_line_id", "normalized_text", "display_text"),
    )


def _relationship_ids(graph: dict[str, Any], edge_ids: set[str], node_ids: set[str]) -> list[str]:
    result: list[str] = []
    for relationship in _as_records(graph.get("relationships")):
        relationship_id = _entity_id(relationship, "id", "relationship_id")
        refs = {_entity_id(relationship, key) for key in ("source", "target", "edge_id", "node_id")}
        if relationship_id and refs & (edge_ids | node_ids):
            result.append(relationship_id)
    return sorted(set(result))


def _line_missing(edge: dict[str, Any]) -> bool:
    if _line_ids(edge):
        return False
    return True


def _flow_state(edge: dict[str, Any]) -> str:
    value = str(edge.get("flow_direction_state") or edge.get("flow_direction") or "unknown").strip().lower()
    aliases = {"source_to_target": "forward", "target_to_source": "reverse", "both": "bidirectional"}
    value = aliases.get(value, value)
    return value if value in {"forward", "reverse", "bidirectional", "unknown", "conflicting"} else "unknown"


def _confidence(edge: dict[str, Any]) -> float | None:
    for key in ("confidence", "trace_confidence", "flow_direction_confidence"):
        try:
            value = float(edge.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return max(0.0, min(1.0, value))
    return None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _record_refs(edge: dict[str, Any], kind: str) -> list[str]:
    attachments = edge.get("attachments") if isinstance(edge.get("attachments"), dict) else {}
    keys = {
        "inline": ("inline_object_ids", "ordered_inline_object_ids"),
        "instrument": ("instrument_ids",),
    }[kind]
    values: list[Any] = []
    for key in keys:
        if isinstance(edge.get(key), list):
            values.extend(edge[key])
    attachment_key = "inline_objects" if kind == "inline" else "instrument_tags"
    for item in _as_records(attachments.get(attachment_key)):
        value = _entity_id(item, "canonical_id", "canonical_instrument_id", "source_object_id", "id")
        if value:
            values.append(value)
    return sorted({str(value) for value in values if str(value).strip()})


def _connector_for_edge(graph: dict[str, Any], edge: dict[str, Any]) -> list[dict[str, Any]]:
    edge_id = _entity_id(edge, "id", "edge_id")
    found: list[dict[str, Any]] = []
    direct = edge.get("off_page_connector")
    if isinstance(direct, dict):
        found.append({**direct, "id": _entity_id(direct, "id", "connector_id") or f"connector::{edge_id}"})
    for connector in _as_records(graph.get("connectors")):
        if _entity_id(connector, "edge_id", "local_edge_id") in {edge_id, _entity_id(edge, "local_id")}:
            found.append(connector)
    unique: dict[str, dict[str, Any]] = {}
    for connector in found:
        connector_id = _entity_id(connector, "id", "connector_id")
        if connector_id:
            unique[connector_id] = connector
    return [unique[key] for key in sorted(unique)]


def _inline_records(graph: dict[str, Any], edge: dict[str, Any]) -> list[dict[str, Any]]:
    records = _as_records((edge.get("attachments") or {}).get("inline_objects")) if isinstance(edge.get("attachments"), dict) else []
    global_records = { _entity_id(item, "id", "inline_object_id", "source_object_id"): item for item in _as_records(graph.get("inline_objects")) }
    result: list[dict[str, Any]] = []
    for item in records:
        item_id = _entity_id(item, "canonical_id", "source_object_id", "id")
        if item_id:
            result.append({**global_records.get(item_id, {}), **item, "id": item_id})
    for item_id in _record_refs(edge, "inline"):
        if item_id in global_records and not any(_entity_id(item, "id") == item_id for item in result):
            result.append({**global_records[item_id], "id": item_id})
    return sorted(result, key=lambda item: (_entity_id(item, "id"), str(item.get("route_position_px") or "")))


def _instrument_records(graph: dict[str, Any], edge: dict[str, Any]) -> list[dict[str, Any]]:
    records = _as_records((edge.get("attachments") or {}).get("instrument_tags")) if isinstance(edge.get("attachments"), dict) else []
    global_records = { _entity_id(item, "id", "instrument_id", "canonical_instrument_id"): item for item in _as_records(graph.get("instruments")) }
    result: list[dict[str, Any]] = []
    for item in records:
        item_id = _entity_id(item, "canonical_id", "canonical_instrument_id", "source_object_id", "id")
        if item_id:
            result.append({**global_records.get(item_id, {}), **item, "id": item_id})
    for item_id in _record_refs(edge, "instrument"):
        if item_id in global_records and not any(_entity_id(item, "id") == item_id for item in result):
            result.append({**global_records[item_id], "id": item_id})
    return sorted(result, key=lambda item: _entity_id(item, "id"))


def _class_name(item: dict[str, Any]) -> str:
    return str(item.get("class_name") or item.get("type") or item.get("service") or "").strip().lower()


def _isolation_candidate(item: dict[str, Any]) -> bool:
    explicit = str(item.get("isolation_role") or item.get("valve_role") or "").strip().lower()
    if explicit in {"isolation", "isolation_valve", "block", "blocking"}:
        return True
    name = _class_name(item)
    return "valve" in name and any(word in name for word in _ISOLATION_WORDS) and not any(word in name for word in _NON_ISOLATION_WORDS)


def _position(item: dict[str, Any]) -> dict[str, float] | None:
    value = item.get("projected_xy") or item.get("position") or item.get("hit_xy")
    if isinstance(value, dict):
        try:
            x, y = float(value.get("x")), float(value.get("y"))
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x, y = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    else:
        return None
    return {"x": x, "y": y} if math.isfinite(x) and math.isfinite(y) else None


def _union_find(ids: Iterable[str], edges: list[dict[str, Any]]) -> dict[str, str]:
    parent = {item: item for item in ids}

    def find(item: str) -> str:
        while parent.get(item, item) != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        if left and right:
            parent.setdefault(left, left)
            parent.setdefault(right, right)
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

    for edge in edges:
        source, target = _edge_endpoints(edge)
        union(source, target)
    return {item: find(item) for item in parent}


def _route_view(graph: dict[str, Any], edge: dict[str, Any]) -> dict[str, Any]:
    edge_id = _entity_id(edge, "id", "edge_id")
    source, target = _edge_endpoints(edge)
    inline = _inline_records(graph, edge)
    instruments = _instrument_records(graph, edge)
    line_ids = _line_ids(edge)
    route = {
        "edge_id": edge_id,
        "source_node_id": source,
        "target_node_id": target,
        "line_ids": line_ids,
        "line_number_ids": _line_number_ids(edge),
        "line_number_records": _line_records(edge),
        "line_assignment_state": "assigned" if line_ids else "missing",
        "flow_direction_state": _flow_state(edge),
        "polyline": _polyline(edge),
        "polyline_reference": {"edge_id": edge_id, "coordinate_system": "image_pixel_origin_top_left"},
        "inline_object_ids": sorted({_entity_id(item, "id") for item in inline if _entity_id(item, "id")}),
        "instrument_ids": sorted({_entity_id(item, "id") for item in instruments if _entity_id(item, "id")}),
        "provenance": {"source": "released_graph", "edge_id": edge_id},
    }
    if not route["polyline"]:
        route["uncertainty"] = [{"type": "missing_route_geometry", "edge_id": edge_id}]
    return route


def _component_views(graph: dict[str, Any]) -> list[dict[str, Any]]:
    edges = _edge_records(graph)
    node_map = _node_records(graph)
    union = _union_find(node_map.keys(), edges)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        source, target = _edge_endpoints(edge)
        root = union.get(source) or union.get(target) or _entity_id(edge, "id", "edge_id")
        groups[root].append(edge)
    return [groups[key] for key in sorted(groups)]


def _boundary_candidate(graph: dict[str, Any], component: list[dict[str, Any]], node_map: dict[str, dict[str, Any]], scope: str) -> dict[str, Any]:
    edge_ids = sorted(_entity_id(edge, "id", "edge_id") for edge in component)
    node_ids = sorted({node_id for edge in component for node_id in _edge_endpoints(edge) if node_id})
    lines = sorted({line_id for edge in component for line_id in _line_ids(edge)})
    equipment_values = {
        value
        for edge in component
        for value in (
            _entity_id(edge, "source_equipment_id"),
            _entity_id(edge, "terminal_equipment_id"),
            _entity_id(edge, "target_equipment_id"),
        )
        if value
    }
    # Some graph-v1 exports retain equipment only as endpoint nodes.  Resolve
    # those node IDs here so downstream package and boundary views do not lose
    # the equipment identity merely because an additive edge reference is
    # absent.
    for node_id in node_ids:
        node_type = str(node_map.get(node_id, {}).get("type") or node_map.get(node_id, {}).get("node_type") or "").lower()
        if node_type in {"equipment", "equipment_general", "tank_vessel", "pump_compressor", "valve", "instrumentation"}:
            equipment_values.add(node_id)
    equipment = sorted(equipment_values)
    inline = sorted({item_id for edge in component for item_id in _record_refs(edge, "inline") if item_id})
    instruments = sorted({item_id for edge in component for item_id in _record_refs(edge, "instrument") if item_id})
    relationship_ids = _relationship_ids(graph, set(edge_ids), set(node_ids))
    degree: defaultdict[str, int] = defaultdict(int)
    for edge in component:
        source, target = _edge_endpoints(edge)
        if source:
            degree[source] += 1
        if target:
            degree[target] += 1
    connectors = [connector for edge in component for connector in _connector_for_edge(graph, edge)]
    cut_points: list[dict[str, Any]] = []
    for node_id in node_ids:
        node = node_map.get(node_id, {})
        node_type = str(node.get("type") or node.get("node_type") or "").lower()
        if degree[node_id] <= 1 or node_type in _TERMINAL_TYPES:
            cut_points.append({
                "kind": "node_terminal",
                "node_id": node_id,
                "position": _position(node),
                "state": "observed" if node_type in _TERMINAL_TYPES else "candidate",
            })
    for connector in sorted(connectors, key=lambda item: _entity_id(item, "id", "connector_id")):
        cut_points.append({
            "kind": "off_page_connector",
            "connector_id": _entity_id(connector, "id", "connector_id"),
            "edge_id": _entity_id(connector, "edge_id", "local_edge_id"),
            "reference": _safe_json(connector.get("reference_value") or connector.get("target_sheet_id")),
            "exit_terminal": connector.get("exit_terminal"),
            "state": str(connector.get("review_state") or "unresolved"),
        })
    isolation: list[dict[str, Any]] = []
    for edge in component:
        edge_id = _entity_id(edge, "id", "edge_id")
        for item in _inline_records(graph, edge):
            if _isolation_candidate(item):
                route_position = item.get("route_position_px")
                if route_position is None:
                    route_position = item.get("trace_distance_px")
                isolation.append({
                    "id": _entity_id(item, "id"),
                    "edge_id": edge_id,
                    "class_name": _safe_json(item.get("class_name") or item.get("type")),
                    "route_position_px": _finite(route_position),
                    "position": _position(item),
                    "state": "observed" if item.get("isolation_role") or "isolation" in _class_name(item) else "inferred",
                })
    uncertainty: list[dict[str, Any]] = []
    for edge in component:
        edge_id = _entity_id(edge, "id", "edge_id")
        if _line_missing(edge):
            uncertainty.append({"type": "missing_line_number", "edge_id": edge_id})
        if _flow_state(edge) in {"unknown", "conflicting"}:
            uncertainty.append({"type": "uncertain_flow_direction", "edge_id": edge_id, "state": _flow_state(edge)})
    confidence_values = [_confidence(edge) for edge in component]
    confidence_values = [value for value in confidence_values if value is not None]
    candidate_id = f"boundary::{scope}::{edge_ids[0] if edge_ids else node_ids[0] if node_ids else 'empty'}"
    return {
        "id": candidate_id,
        "kind": "system" if connectors else "process",
        "state": "candidate",
        "review_state": "derived_from_released_graph",
        "confidence": round(min(confidence_values), 3) if confidence_values else None,
        "member_node_ids": node_ids,
        "member_edge_ids": edge_ids,
        "member_line_ids": lines,
        "member_line_number_ids": sorted({line_id for edge in component for line_id in _line_number_ids(edge)}),
        "member_equipment_ids": equipment,
        "member_inline_object_ids": inline,
        "member_instrument_ids": instruments,
        "member_relationship_ids": relationship_ids,
        "routes": [_route_view(graph, edge) for edge in component],
        "cut_points": sorted(cut_points, key=lambda item: (str(item.get("kind")), str(item.get("node_id") or item.get("connector_id") or ""))),
        "isolation_elements": sorted(isolation, key=lambda item: (str(item.get("edge_id")), str(item.get("id")))),
        "exclusions": [],
        "uncertainty": sorted(uncertainty, key=lambda item: (str(item.get("type")), str(item.get("edge_id")))),
        "provenance": {"source": "released_graph", "scope": scope, "edge_ids": edge_ids},
    }


def build_process_boundary_candidates(
    graph_payload: dict[str, Any],
    *,
    release_gate_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic connected-component process/system boundary candidates."""
    graph = graph_payload if isinstance(graph_payload, dict) else {}
    released, state, reasons = _release_state(graph, release_gate_payload)
    scope = _graph_scope(graph)
    if not released:
        return _safe_json({"schema_version": "phase8_boundary_candidates_v1", "scope": scope, "state": state, "release_ready": False, "blocked_reasons": reasons, "candidates": []})
    node_map = _node_records(graph)
    candidates = [_boundary_candidate(graph, component, node_map, scope) for component in _component_views(graph)]
    return _safe_json({"schema_version": "phase8_boundary_candidates_v1", "scope": scope, "state": state, "release_ready": True, "blocked_reasons": [], "candidates": candidates})


def _package_groups(graph: dict[str, Any], edges: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    node_ids = {node_id for edge in edges for node_id in _edge_endpoints(edge) if node_id}
    union = _union_find(node_ids, edges)

    def component_key(edge: dict[str, Any]) -> str:
        source, target = _edge_endpoints(edge)
        return union.get(source) or union.get(target) or _entity_id(edge, "id", "edge_id")

    # The same canonical line text can occur on separate disconnected drawing
    # components.  Keep those candidates separate unless the graph itself
    # proves that they share a connected component.  A line-number match alone
    # is insufficient evidence for one hydrotest package.
    roots_by_line: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        root = component_key(edge)
        for line_id in _line_ids(edge):
            roots_by_line[line_id].add(root)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        line_ids = _line_ids(edge)
        if not line_ids:
            keys = [f"edge::{_entity_id(edge, 'id', 'edge_id')}"]
        else:
            root = component_key(edge)
            keys = [
                f"{line_id}::component::{root}"
                if len(roots_by_line[line_id]) > 1
                else line_id
                for line_id in line_ids
            ]
        for key in keys:
            groups[key].append(edge)
    return [(key, sorted(groups[key], key=lambda edge: _entity_id(edge, "id", "edge_id"))) for key in sorted(groups)]


def build_test_package_candidates(
    graph_payload: dict[str, Any],
    *,
    release_gate_payload: dict[str, Any] | None = None,
    boundary_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build line-linked test-package candidates while preserving route ambiguity."""
    graph = graph_payload if isinstance(graph_payload, dict) else {}
    released, state, reasons = _release_state(graph, release_gate_payload)
    scope = _graph_scope(graph)
    if not released:
        return _safe_json({"schema_version": "phase8_test_package_candidates_v1", "scope": scope, "state": state, "release_ready": False, "blocked_reasons": reasons, "candidates": []})
    boundaries: list[dict[str, Any]] = []
    if isinstance(boundary_payload, dict):
        raw_boundaries = boundary_payload.get("candidates")
        if raw_boundaries is None and isinstance(boundary_payload.get("process_boundaries"), dict):
            raw_boundaries = boundary_payload["process_boundaries"].get("candidates")
        boundaries = [item for item in raw_boundaries or [] if isinstance(item, dict)]
    boundary_for_edge = {edge_id: boundary for boundary in boundaries for edge_id in boundary.get("member_edge_ids", [])}
    edges = _edge_records(graph)
    node_map = _node_records(graph)
    degree: defaultdict[str, int] = defaultdict(int)
    endpoint_pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for edge in edges:
        source, target = _edge_endpoints(edge)
        degree[source] += 1
        degree[target] += 1
        endpoint_pair_counts[tuple(sorted((source, target)))] += 1
    candidates: list[dict[str, Any]] = []
    for key, group in _package_groups(graph, edges):
        routes = [_route_view(graph, edge) for edge in group]
        edge_ids = [route["edge_id"] for route in routes]
        line_ids = sorted({line_id for route in routes for line_id in route["line_ids"]})
        line_number_ids = sorted({line_id for route in routes for line_id in route["line_number_ids"]})
        line_record_map: dict[str, dict[str, Any]] = {}
        for route in routes:
            for record in route["line_number_records"]:
                record_key = _entity_id(record, "id", "source_object_id", "canonical_line_id", "normalized_text", "display_text")
                if record_key:
                    line_record_map.setdefault(record_key, record)
        states = {route["flow_direction_state"] for route in routes}
        flow = next(iter(states)) if len(states) == 1 else "conflicting" if states - {"unknown"} else "unknown"
        group_boundary = _boundary_candidate(graph, group, node_map, scope)
        isolation = group_boundary["isolation_elements"]
        cut_points = group_boundary["cut_points"]
        uncertainty = [item for route in routes for item in route.get("uncertainty", [])]
        for route in routes:
            if len(route["line_ids"]) > 1:
                uncertainty.append({
                    "type": "multiple_line_assignments",
                    "edge_id": route["edge_id"],
                    "line_ids": route["line_ids"],
                })
        route_kinds: set[str] = set()
        for edge in group:
            edge_id = _entity_id(edge, "id", "edge_id")
            if _line_missing(edge):
                uncertainty.append({"type": "missing_line_number", "edge_id": edge_id})
            if _flow_state(edge) in {"unknown", "conflicting"}:
                uncertainty.append({"type": "uncertain_flow_direction", "edge_id": edge_id, "state": _flow_state(edge)})
            source, target = _edge_endpoints(edge)
            if endpoint_pair_counts[tuple(sorted((source, target)))] > 1:
                route_kinds.add("parallel_or_bypass")
            elif degree[source] > 2 or degree[target] > 2:
                route_kinds.add("branch")
            else:
                route_kinds.add("main_or_single")
        route_kind = (
            "parallel_or_bypass"
            if "parallel_or_bypass" in route_kinds
            else "branch"
            if "branch" in route_kinds
            else "main_or_single"
        )
        confidence_values = [_confidence(edge) for edge in group]
        confidence_values = [value for value in confidence_values if value is not None]
        linked_boundaries = sorted({boundary_for_edge[edge_id]["id"] for edge_id in edge_ids if edge_id in boundary_for_edge})
        relationship_ids = _relationship_ids(graph, set(edge_ids), {route["source_node_id"] for route in routes} | {route["target_node_id"] for route in routes})
        candidate_id = f"test_package::{scope}::{key}"
        candidates.append({
            "id": candidate_id,
            "state": "candidate",
            "review_state": "derived_from_released_graph",
            "confidence": round(min(confidence_values), 3) if confidence_values else None,
            "line_ids": line_ids,
            "line_number_ids": line_number_ids,
            "line_number_records": [_safe_json(line_record_map[key]) for key in sorted(line_record_map)],
            "line_assignment_state": "assigned" if line_ids else "missing",
            "edge_ids": edge_ids,
            "equipment_ids": group_boundary["member_equipment_ids"],
            "inline_object_ids": sorted({item_id for route in routes for item_id in route["inline_object_ids"]}),
            "instrument_ids": sorted({item_id for route in routes for item_id in route["instrument_ids"]}),
            "route_kind": route_kind,
            "routes": routes,
            "flow_direction_state": flow,
            "boundary_candidate_ids": linked_boundaries,
            "relationship_ids": relationship_ids,
            "cut_points": sorted(cut_points, key=lambda item: (str(item.get("kind")), str(item.get("node_id") or item.get("connector_id") or ""))),
            "isolation_elements": sorted(isolation, key=lambda item: (str(item.get("edge_id")), str(item.get("id")))),
            "exclusions": [],
            "uncertainty": sorted({(str(item.get("type")), str(item.get("edge_id")), str(item.get("state"))): item for item in uncertainty}.values(), key=lambda item: (str(item.get("type")), str(item.get("edge_id")))),
            "provenance": {"source": "released_graph", "scope": scope, "line_key": key.split("::component::", 1)[0], "edge_ids": edge_ids},
        })
    return _safe_json({"schema_version": "phase8_test_package_candidates_v1", "scope": scope, "state": state, "release_ready": True, "blocked_reasons": [], "candidates": candidates})


def build_phase8_engineering_views(
    graph_payload: dict[str, Any],
    *,
    release_gate_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build all Phase 8 candidate projections from one release-gated graph."""
    boundaries = build_process_boundary_candidates(graph_payload, release_gate_payload=release_gate_payload)
    packages = build_test_package_candidates(graph_payload, release_gate_payload=release_gate_payload, boundary_payload=boundaries)
    return _safe_json({
        "schema_version": "phase8_engineering_views_v1",
        "scope": _graph_scope(graph_payload if isinstance(graph_payload, dict) else {}),
        "release_ready": bool(boundaries.get("release_ready") and packages.get("release_ready")),
        "state": "released" if boundaries.get("release_ready") and packages.get("release_ready") else "blocked",
        "blocked_reasons": sorted(set(boundaries.get("blocked_reasons", []) + packages.get("blocked_reasons", []))),
        "process_boundaries": boundaries,
        "test_packages": packages,
    })


__all__ = [
    "build_phase8_engineering_views",
    "build_process_boundary_candidates",
    "build_test_package_candidates",
]
