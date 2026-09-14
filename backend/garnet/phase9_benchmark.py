"""Deterministic Phase 9 benchmark checks for exported P&ID graph payloads.

The benchmark measures whether an export preserves the evidence and structure
needed by a reviewer.  It deliberately does not score detector accuracy: the
representative fixtures contain invariants, rather than a fabricated pixel
level gold annotation.  Unresolved evidence is reported separately and only
blocks a case when its expectations require the fact to be resolved.

The module accepts a graph-v1 payload, a graph-v2 combined payload, or the
``{"sheets": [...], "combined_graph": ...}`` export envelope used by the
multi-sheet pipeline.  Reports contain only JSON-compatible, sorted data so a
report can be checked into a benchmark run or compared byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .versioned_export import validate_downstream_export


BENCHMARK_SCHEMA_VERSION = "phase9_benchmark_v1"
DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "phase9_benchmark" / "benchmark.json"

_RELATIONSHIP_TYPES = {
    "connects_to",
    "branches_from",
    "cross_sheet_continues",
    "has_inline_object",
    "measures",
    "controls",
    "actuates",
    "has_port",
    "instrument_association",
}
_UNRESOLVED_STATES = {"unresolved", "unknown", "conflicting", "candidate", "pending", "review_pending"}


class BenchmarkContractError(ValueError):
    """Raised by strict benchmark runs when a contract invariant is broken."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        violations = report.get("violations", [])
        super().__init__(f"Phase 9 benchmark contract failed ({len(violations)} violation(s))")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _point(point: Any) -> bool:
    if isinstance(point, dict):
        return _finite(point.get("x", point.get("col"))) and _finite(point.get("y", point.get("row")))
    return isinstance(point, (list, tuple)) and len(point) >= 2 and _finite(point[0]) and _finite(point[1])


def _polyline(edge: dict[str, Any]) -> list[Any]:
    value = edge.get("polyline") or edge.get("path") or edge.get("points")
    return value if isinstance(value, list) else []


def _endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    physical = edge.get("physical") if isinstance(edge.get("physical"), dict) else {}
    return _text(edge.get("source") or edge.get("src") or physical.get("source")), _text(edge.get("target") or edge.get("dst") or physical.get("target"))


def _state(item: dict[str, Any]) -> str:
    flow = item.get("flow") if isinstance(item.get("flow"), dict) else {}
    return _text(item.get("state") or item.get("semantic_state") or item.get("review_state") or flow.get("state") or flow.get("review_state")).lower()


def _flow_state(item: dict[str, Any]) -> str:
    flow = item.get("flow") if isinstance(item.get("flow"), dict) else {}
    value = item.get("flow_direction_state") or item.get("flow_direction") or flow.get("state")
    aliases = {"source_to_target": "forward", "target_to_source": "reverse", "both": "bidirectional"}
    value = aliases.get(_text(value).lower(), _text(value).lower())
    return value if value in {"forward", "reverse", "bidirectional", "unknown", "conflicting"} else "unknown"


def _record_id(item: Any) -> str:
    if not isinstance(item, dict):
        return _text(item)
    return _text(item.get("id") or item.get("occurrence_id") or item.get("segment_id") or item.get("boundary_id") or item.get("package_id"))


def _sheet_id(graph: dict[str, Any]) -> str:
    meta = _drawing_metadata(graph)
    return meta["drawing_id"]


def _catalog_ids(graph: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("nodes", "equipment", "equipment_ports", "ports", "segments", "edges", "routes", "lines", "line_number_occurrences", "inline_objects", "instruments", "connectors", "process_boundaries", "test_packages", "boundaries"):
        for item in _as_list(graph.get(key)):
            if isinstance(item, dict) and _text(item.get("id") or item.get("segment_id") or item.get("boundary_id") or item.get("package_id")):
                ids.add(_text(item.get("id") or item.get("segment_id") or item.get("boundary_id") or item.get("package_id")))
    return ids


def _graphs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract local graph payloads, preserving deterministic input order."""
    result: list[dict[str, Any]] = []
    sheets = _as_list(payload.get("sheets"))
    if sheets:
        for sheet in sheets:
            graph = sheet.get("graph_v1") if isinstance(sheet, dict) else None
            if not isinstance(graph, dict) and isinstance(sheet, dict):
                graph = sheet.get("graph") if isinstance(sheet.get("graph"), dict) else sheet
            if isinstance(graph, dict):
                result.append(graph)
    if not result and isinstance(payload.get("graph"), dict):
        # The versioned downstream export keeps graph catalogs in an envelope
        # and Phase 8 views beside them.  Evaluate a shallow normalized copy so
        # callers can benchmark an export without changing it.
        graph = dict(payload["graph"])
        drawings = _as_list(payload.get("drawings"))
        if drawings and "drawing" not in graph:
            graph["drawing"] = drawings[0]
        views = payload.get("engineering_views") if isinstance(payload.get("engineering_views"), dict) else {}
        if "boundaries" in views and "process_boundaries" not in graph:
            graph["process_boundaries"] = views["boundaries"]
        if "test_packages" in views and "test_packages" not in graph:
            graph["test_packages"] = views["test_packages"]
        result.append(graph)
    if not result and ("edges" in payload or "segments" in payload or "nodes" in payload):
        result.append(payload)
    return result


def _routes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    routes = [item for item in _as_list(graph.get("edges") or graph.get("routes")) if isinstance(item, dict)]
    if routes:
        return routes
    # Canonical fixture records use ``segments``.  Supporting them makes the
    # harness useful for fixture review as well as release exports.
    return [item for item in _as_list(graph.get("segments")) if isinstance(item, dict)]


def _drawing_metadata(graph: dict[str, Any]) -> dict[str, Any]:
    drawing = graph.get("drawing") if isinstance(graph.get("drawing"), dict) else {}
    if not drawing and _as_list(graph.get("drawings")):
        drawing = graph["drawings"][0] if isinstance(graph["drawings"][0], dict) else {}
    document = graph.get("document") if isinstance(graph.get("document"), dict) else {}
    dimensions = drawing.get("pixel_dimensions") or drawing.get("image") or document.get("image") or graph.get("pixel_dimensions")
    dimensions = dimensions if isinstance(dimensions, dict) else {}
    return {
        "drawing_id": _text(drawing.get("drawing_id") or drawing.get("doc_id") or document.get("doc_id") or graph.get("drawing_id")),
        "width": dimensions.get("width"),
        "height": dimensions.get("height"),
        "coordinate_system": _text(drawing.get("coordinate_system") or graph.get("coordinate_system")),
    }


def _ratio(covered: int, total: int) -> float | None:
    return round(covered / total, 6) if total else None


def _metric(covered: int, total: int, *, status: str | None = None) -> dict[str, Any]:
    value = {"covered": covered, "total": total, "ratio": _ratio(covered, total)}
    if status:
        value["status"] = status
    return value


def _check(checks: list[dict[str, Any]], violations: list[dict[str, Any]], check_id: str, passed: bool, *, observed: Any = None, expected: Any = None, message: str = "") -> None:
    item = {"id": check_id, "status": "passed" if passed else "failed"}
    if observed is not None:
        item["observed"] = observed
    if expected is not None:
        item["expected"] = expected
    if message:
        item["message"] = message
    checks.append(item)
    if not passed:
        violations.append({"check": check_id, "message": message or "contract invariant failed", "observed": observed, "expected": expected})


def _unresolved(report: dict[str, Any], kind: str, item: dict[str, Any], *, expected: bool = False) -> None:
    report.setdefault("unresolved_evidence", []).append({"kind": kind, "id": _text(item.get("id")), "state": _state(item), "expected": expected})


def evaluate_graph_payload(payload: dict[str, Any], *, expectations: dict[str, Any] | None = None, case_id: str = "payload", source_graph: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one export-like payload and return a machine-readable report.

    ``expectations`` describes fixture invariants only.  It is never treated as
    detector gold truth; for example, an expected unresolved connector remains
    an informational unresolved item rather than a failure.
    """
    if not isinstance(payload, dict):
        raise TypeError("benchmark payload must be an object")
    expectations = expectations if isinstance(expectations, dict) else {}
    graphs = _graphs(payload)
    combined = payload.get("combined_graph") if isinstance(payload.get("combined_graph"), dict) else None
    if combined is None and isinstance(payload.get("combined_continuity"), dict):
        continuity = payload["combined_continuity"].get("continuity_relationships")
        continuity = continuity if isinstance(continuity, list) else []
        endpoint_ids = {
            _text(item.get(side))
            for item in continuity
            if isinstance(item, dict)
            for side in ("source", "target")
            if _text(item.get(side))
        }
        combined = {
            "schema_version": "graph_v2_combined",
            "connectors": [{"id": value, "state": "reviewed"} for value in sorted(endpoint_ids)],
            "relationships": continuity,
            "issues": payload["combined_continuity"].get("issues") or [],
        }
    if combined is None and _text(payload.get("schema_version")).startswith("graph_v2_combined"):
        combined = payload
    report: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "case_id": case_id,
        "passed": True,
        "violations": [],
        "unresolved_evidence": [],
        "checks": [],
        "metrics": {"coverage_validity": {}, "detector_accuracy": {"status": "not_scored", "reason": "No annotated gold truth is included in the benchmark fixtures"}},
    }
    checks = report["checks"]
    violations = report["violations"]

    if _text(payload.get("schema_version")) == "garnet_downstream_export_v1":
        export_result = validate_downstream_export(payload, source_graph=source_graph)
        _check(checks, violations, "versioned_export_contract", export_result["valid"], observed=export_result["issues"], expected=[], message="versioned export schema, release gate, hash, or typed reference contract failed")

    _check(checks, violations, "payload_object", bool(payload), observed=type(payload).__name__, expected="object")
    _check(checks, violations, "graph_present", bool(graphs or combined), observed=bool(graphs or combined), expected=True)

    all_routes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    metadata_ok = 0
    for graph in graphs:
        if "physical_multigraph" in graph:
            _check(checks, violations, "physical_graph_contract", graph.get("physical_multigraph") is True and graph.get("flow_direction_separate") is True and graph.get("directed_multigraph") is False, observed={"physical_multigraph": graph.get("physical_multigraph"), "flow_direction_separate": graph.get("flow_direction_separate"), "directed_multigraph": graph.get("directed_multigraph")}, expected={"physical_multigraph": True, "flow_direction_separate": True, "directed_multigraph": False}, message="physical connectivity must be an undirected multigraph with flow stored separately")
        meta = _drawing_metadata(graph)
        dimensions_ok = _finite(meta["width"]) and _finite(meta["height"]) and float(meta["width"]) > 0 and float(meta["height"]) > 0
        identity_ok = bool(meta["drawing_id"] and meta["coordinate_system"])
        if dimensions_ok and identity_ok:
            metadata_ok += 1
        else:
            _check(checks, violations, "drawing_metadata", False, observed=meta, expected="drawing id, positive pixel dimensions, coordinate system", message="drawing metadata is incomplete")
        for route in _routes(graph):
            all_routes.append((graph, route))
            if _state(route) in _UNRESOLVED_STATES:
                _unresolved(report, "route", route)
            if _flow_state(route) in {"unknown", "conflicting"}:
                _unresolved(report, "flow_direction", route)

    if graphs and metadata_ok == len(graphs):
        _check(checks, violations, "drawing_metadata", True, observed=metadata_ok, expected=len(graphs))
    elif not graphs:
        _check(checks, violations, "drawing_metadata", False, observed=0, expected=1, message="no graph drawing metadata found")

    valid_geometry = 0
    valid_connectivity = 0
    for graph, route in all_routes:
        points = _polyline(route)
        geometry_ok = len(points) >= 2 and all(_point(item) for item in points)
        if geometry_ok:
            valid_geometry += 1
        source, target = _endpoints(route)
        refs = _catalog_ids(graph)
        connectivity_ok = bool(source and target and source in refs and target in refs and source != target)
        if connectivity_ok:
            valid_connectivity += 1
        if not geometry_ok:
            _check(checks, violations, "route_pixel_geometry", False, observed={"id": _text(route.get("id")), "points": len(points)}, expected=">=2 finite pixel points", message="route is missing ordered pixel geometry")
        if not connectivity_ok:
            _check(checks, violations, "physical_connectivity", False, observed={"id": _text(route.get("id")), "source": source, "target": target}, expected="existing distinct endpoint references", message="route endpoint does not resolve to a graph entity")

    total_routes = len(all_routes)
    report["metrics"]["coverage_validity"].update({
        "route_pixel_geometry": _metric(valid_geometry, total_routes),
        "physical_connectivity": _metric(valid_connectivity, total_routes),
    })
    if total_routes and valid_geometry == total_routes:
        _check(checks, violations, "route_pixel_geometry", True, observed=valid_geometry, expected=total_routes)
    if total_routes and valid_connectivity == total_routes:
        _check(checks, violations, "physical_connectivity", True, observed=valid_connectivity, expected=total_routes)

    # Line identity, assignment and OCR occurrence traceability.
    lines: list[dict[str, Any]] = []
    assigned: set[str] = set()
    occurrence_total = occurrence_valid = 0
    for graph in graphs:
        local_routes = _routes(graph)
        for route in local_routes:
            route_id = _text(route.get("id"))
            if route.get("line_id") or route.get("canonical_line_id") or route.get("line_number") or route.get("line_tag") or route.get("line_ids"):
                assigned.add(route_id)
        for line in _as_list(graph.get("lines")):
            if not isinstance(line, dict):
                continue
            lines.append(line)
            refs = line.get("line_edges") or line.get("edge_ids") or line.get("segment_ids") or []
            for ref in refs:
                ref_id = _text(ref.get("edge_id") if isinstance(ref, dict) else ref)
                if ref_id:
                    assigned.add(ref_id)
            occurrences = line.get("occurrences") or line.get("line_number_occurrences") or []
            line_id = _text(line.get("id") or line.get("canonical_line_id"))
            for occurrence in occurrences:
                occurrence_total += 1
                occurrence_id = _text(occurrence.get("id") if isinstance(occurrence, dict) else occurrence)
                if occurrence_id and occurrence_id != line_id:
                    occurrence_valid += 1
        # Versioned exports flatten occurrences into a dedicated catalog and
        # retain the canonical line reference on each occurrence.
        for occurrence in _as_list(graph.get("line_occurrences") or graph.get("line_number_occurrences")):
            if not isinstance(occurrence, dict):
                continue
            occurrence_total += 1
            occurrence_id = _text(occurrence.get("id") or occurrence.get("occurrence_id"))
            line_id = _text(occurrence.get("line_id") or occurrence.get("canonical_line_id"))
            if occurrence_id and line_id and occurrence_id != line_id:
                occurrence_valid += 1
    assigned_total = len(all_routes)
    assigned_count = sum(1 for _, route in all_routes if _text(route.get("id")) in assigned)
    report["metrics"]["coverage_validity"].update({
        "line_assignment": _metric(assigned_count, assigned_total),
        "line_occurrence_traceability": _metric(occurrence_valid, occurrence_total),
    })
    if lines and occurrence_total and occurrence_valid != occurrence_total:
        _check(checks, violations, "line_occurrence_traceability", False, observed=occurrence_valid, expected=occurrence_total, message="line occurrence is not distinct from or traceable to its canonical line")
    elif not lines:
        _check(checks, violations, "line_catalog", False, observed=0, expected=">=1 line identity", message="no canonical line identities are present")

    # Equipment, ports and typed instrument relationships.
    equipment = [item for graph in graphs for item in _as_list(graph.get("equipment")) if isinstance(item, dict)]
    ports = [item for graph in graphs for item in _as_list(graph.get("ports") or graph.get("equipment_ports")) if isinstance(item, dict)]
    instruments = [item for graph in graphs for item in _as_list(graph.get("instruments")) if isinstance(item, dict)]
    equipment_ids = {_text(item.get("id")) for item in equipment if _text(item.get("id"))}
    port_ids = {_text(item.get("id")) for item in ports if _text(item.get("id"))}
    instrument_ids = {_text(item.get("id")) for item in instruments if _text(item.get("id"))}
    port_valid = sum(1 for item in ports if _text(item.get("equipment_id")) in equipment_ids and isinstance(item.get("position"), dict) and _finite(item["position"].get("x")) and _finite(item["position"].get("y")))
    relationship_total = relationship_valid = 0
    typed_instrument = 0
    for graph in graphs:
        refs = _catalog_ids(graph) | equipment_ids | port_ids | instrument_ids
        for relation in _as_list(graph.get("relationships")):
            if not isinstance(relation, dict):
                continue
            relationship_total += 1
            source, target = _text(relation.get("source")), _text(relation.get("target"))
            if source in refs and target in refs:
                relationship_valid += 1
            if _text(relation.get("type")) in {"measures", "controls", "actuates"} and source in instrument_ids:
                typed_instrument += 1
            if _state(relation) in _UNRESOLVED_STATES:
                _unresolved(report, "relationship", relation)
    report["metrics"]["coverage_validity"].update({
        "equipment_port_relationships": _metric(port_valid, len(ports)),
        "typed_instrument_relationships": _metric(typed_instrument, len(instruments)),
        "relationship_endpoint_validity": _metric(relationship_valid, relationship_total),
    })
    if ports and port_valid != len(ports):
        _check(checks, violations, "equipment_port_relationships", False, observed=port_valid, expected=len(ports), message="port lacks a drawing-scoped equipment reference or pixel position")
    if relationship_total and relationship_valid != relationship_total:
        _check(checks, violations, "relationship_endpoint_validity", False, observed=relationship_valid, expected=relationship_total, message="typed relationship endpoint is not resolvable")

    # Boundaries and test packages are candidates.  Their unresolved state is
    # carried into the report, while links to known routes/entities are checked.
    candidates = [item for graph in graphs for key in ("process_boundaries", "boundaries", "test_packages") for item in _as_list(graph.get(key)) if isinstance(item, dict)]
    candidate_ids = {_text(item.get("id") or item.get("boundary_id") or item.get("package_id")) for item in candidates}
    route_ids = {_text(route.get("id")) for _, route in all_routes}
    line_ids = {_text(line.get("id") or line.get("canonical_line_id")) for line in lines}
    candidate_links = candidate_valid = 0
    for item in candidates:
        if _state(item) in _UNRESOLVED_STATES:
            _unresolved(report, "engineering_candidate", item, expected=True)
        links = item.get("members") or item.get("edge_ids") or item.get("line_ids") or item.get("route_ids") or []
        if item.get("boundary_id"):
            links = [*links, item["boundary_id"]]
        for link in links:
            candidate_links += 1
            link_id = _text(link.get("edge_id") if isinstance(link, dict) else link)
            if link_id in route_ids or link_id in line_ids or link_id in candidate_ids:
                candidate_valid += 1
    report["metrics"]["coverage_validity"]["boundary_test_package_links"] = _metric(candidate_valid, candidate_links)
    if candidate_links and candidate_valid != candidate_links:
        _check(checks, violations, "boundary_test_package_links", False, observed=candidate_valid, expected=candidate_links, message="boundary or test-package link does not resolve to a route or line")

    # Parallel route preservation is an explicit invariant, not detector
    # accuracy.  Group by ordered endpoints and compare complete polylines.
    expected_parallel = expectations.get("parallel_routes") or []
    for group in expected_parallel:
        source, target = _text(group.get("source")), _text(group.get("target"))
        expected_count = int(group.get("count", 2))
        actual = [route for _, route in all_routes if _endpoints(route) == (source, target) or _endpoints(route) == (target, source)]
        distinct = {json.dumps(_polyline(route), sort_keys=True) for route in actual}
        _check(checks, violations, "parallel_route_preservation", len(actual) >= expected_count and len(distinct) >= expected_count, observed={"count": len(actual), "distinct_geometry": len(distinct)}, expected={"count": expected_count, "distinct_geometry": expected_count}, message="parallel or bypass routes were collapsed")

    # Topology classifications are reviewed evidence.  The benchmark checks
    # that crossing and tee decisions survive export; it does not judge a
    # detector's ability to discover them from pixels.
    topology_records = [
        item for graph in graphs
        for key in ("crossings", "crossing_resolution", "topology_events")
        for item in _as_list(graph.get(key)) if isinstance(item, dict)
    ]
    topology_by_id = {_record_id(item): item for item in topology_records if _record_id(item)}
    for expected in expectations.get("crossings", []) or []:
        item = topology_by_id.get(_text(expected.get("id")))
        expected_class = _text(expected.get("classification") or expected.get("expected_classification"))
        _check(checks, violations, f"crossing_classification:{_text(expected.get('id'))}", bool(item) and _text(item.get("classification") or item.get("state")) == expected_class, observed=(item or {}).get("classification") if item else None, expected=expected_class, message="crossing/junction classification was not preserved")
    for expected in expectations.get("marked_tees", []) or []:
        item = topology_by_id.get(_text(expected.get("id")))
        incident = item.get("incident_route_ids") or item.get("route_ids") or [] if item else []
        expected_count = int(expected.get("incident_count", 3))
        _check(checks, violations, f"marked_tee:{_text(expected.get('id'))}", bool(item) and _text(item.get("classification") or item.get("state")) == "confirmed_junction" and len(incident) >= expected_count and bool(item.get("junction_marker") or item.get("marker")), observed={"classification": (item or {}).get("classification") if item else None, "incident_count": len(incident), "junction_marker": bool(item and (item.get("junction_marker") or item.get("marker")))}, expected={"classification": "confirmed_junction", "incident_count": expected_count, "junction_marker": True}, message="marked tee evidence or its incident routes were lost")

    for expected in expectations.get("flow_states", []) or []:
        route = next((route for _, route in all_routes if _text(route.get("id")) == _text(expected.get("route_id"))), None)
        expected_state = _text(expected.get("state")).lower()
        _check(checks, violations, f"flow_state:{_text(expected.get('route_id'))}", bool(route) and _flow_state(route) == expected_state and bool(_endpoints(route)[0]) and bool(_endpoints(route)[1]), observed=_flow_state(route) if route else None, expected=expected_state, message="unknown or conflicting flow state was not retained independently of physical endpoints")

    for expected in expectations.get("unresolved_instrument_associations", []) or []:
        relation_id = _text(expected.get("id") if isinstance(expected, dict) else expected)
        relation = next((relation for graph in graphs for relation in _as_list(graph.get("relationships")) if isinstance(relation, dict) and _text(relation.get("id")) == relation_id), None)
        evidence = relation.get("evidence") if isinstance(relation, dict) else None
        _check(checks, violations, f"unresolved_instrument_association:{relation_id}", bool(relation) and _text(relation.get("type")) == "instrument_association" and _state(relation) in _UNRESOLVED_STATES and (not evidence or _text(relation.get("association_method") or relation.get("method")) in {"proximity", "proximity_only", ""}), observed={"type": (relation or {}).get("type") if relation else None, "state": _state(relation) if relation else None, "method": (relation or {}).get("association_method") if relation else None}, expected={"type": "instrument_association", "state": "unresolved"}, message="proximity-only instrument association was promoted to functional truth")

    # Repeated labels are drawing-scoped evidence: identical text on separate
    # sheets must retain separate line identities and occurrences.
    for expected in expectations.get("repeated_line_labels", []) or []:
        wanted = _text(expected.get("normalized_text") or expected.get("text"))
        sheet_set = set(_text(value) for value in expected.get("sheets", []) if _text(value))
        found: dict[str, set[str]] = {}
        for graph in graphs:
            sheet = _sheet_id(graph)
            for line in _as_list(graph.get("lines")):
                if not isinstance(line, dict):
                    continue
                occurrences = line.get("occurrences") or line.get("line_number_occurrences") or []
                texts = {_text(line.get("normalized_text") or line.get("number") or line.get("text"))}
                texts.update(_text(item.get("normalized_text") or item.get("text") or item.get("raw_text")) for item in occurrences if isinstance(item, dict))
                if wanted in texts:
                    found.setdefault(sheet, set()).add(_text(line.get("id") or line.get("line_id")))
        scoped_ok = (not sheet_set or sheet_set.issubset(found)) and len({line_id for values in found.values() for line_id in values if line_id}) >= len(sheet_set or found)
        _check(checks, violations, f"repeated_line_label:{wanted}", scoped_ok, observed={sheet: sorted(ids) for sheet, ids in sorted(found.items())}, expected={"sheets": sorted(sheet_set), "sheet_scoped": True}, message="repeated line labels lost drawing scope or OCR occurrence evidence")

    # Multi-sheet continuity is only promoted by an explicit relationship.
    if combined is not None:
        combined_ids = _catalog_ids(combined)
        connectors = [item for item in _as_list(combined.get("connectors")) if isinstance(item, dict)]
        continuity = [item for item in _as_list(combined.get("relationships")) if isinstance(item, dict) and _text(item.get("type")) == "cross_sheet_continues"]
        continuity_valid = sum(1 for item in continuity if _text(item.get("source")) in combined_ids and _text(item.get("target")) in combined_ids and _text(item.get("source")) != _text(item.get("target")))
        report["metrics"]["coverage_validity"]["multi_sheet_connector_continuity"] = _metric(continuity_valid, len(continuity))
        for connector in connectors:
            if _state(connector) in _UNRESOLVED_STATES or _text(connector.get("status")) in _UNRESOLVED_STATES:
                _unresolved(report, "connector", connector, expected=True)
        required = int(expectations.get("required_cross_sheet_relationships", 0) or 0)
        required_resolved = int(expectations.get("required_resolved_cross_sheet_relationships", 0) or 0)
        resolved_continuity = sum(1 for item in continuity if _state(item) in {"observed", "reviewed", "accepted", "resolved"})
        _check(checks, violations, "multi_sheet_connector_continuity", continuity_valid >= required and resolved_continuity >= required_resolved, observed={"valid": continuity_valid, "resolved": resolved_continuity}, expected={"valid": required, "resolved": required_resolved}, message="cross-sheet continuation is missing, unresolved, or has invalid endpoints")
        exact_count = expectations.get("exact_cross_sheet_continuations")
        if exact_count is not None:
            _check(checks, violations, "exact_cross_sheet_continuity", len(continuity) == int(exact_count), observed=len(continuity), expected=int(exact_count), message="the combined graph contains an unexpected number of continuity relationships")
        reviewed_count = expectations.get("reviewed_cross_sheet_continuations")
        if reviewed_count is not None:
            reviewed = sum(1 for item in continuity if _state(item) in {"reviewed", "accepted", "resolved"})
            _check(checks, violations, "reviewed_cross_sheet_continuity", reviewed == int(reviewed_count), observed=reviewed, expected=int(reviewed_count), message="cross-sheet continuity lacks the required reviewed provenance")
        for connector_id in expectations.get("unresolved_connector_ids", []) or []:
            connector = next((item for item in connectors if _text(item.get("id")) == _text(connector_id)), None)
            connected = any(_text(item.get("source")) == _text(connector_id) or _text(item.get("target")) == _text(connector_id) for item in continuity)
            if connector is not None:
                _unresolved(report, "connector", connector, expected=True)
            _check(checks, violations, f"unresolved_connector_no_continuity:{_text(connector_id)}", connector is not None and not connected and (_state(connector) in _UNRESOLVED_STATES or _text(connector.get("status")) in _UNRESOLVED_STATES), observed={"present": connector is not None, "connected": connected, "state": _state(connector) if connector else None}, expected={"present": True, "connected": False, "state": "unresolved"}, message="an unresolved connector created cross-sheet physical continuity")

        # A combined graph must retain every local entity and identify its
        # source sheet.  ``local_id`` is the preferred audit field; qualified
        # IDs are accepted as a compatibility fallback.
        collection_pairs = (("nodes", "nodes"), ("edges", "edges"), ("equipment", "equipment"), ("equipment_ports", "ports"), ("ports", "ports"), ("lines", "lines"), ("inline_objects", "inline_objects"), ("instruments", "instruments"), ("connectors", "connectors"))
        completeness_missing: list[str] = []
        provenance_missing: list[str] = []
        combined_sheets = {_text(value) for value in combined.get("sheets", []) if _text(value)}
        if not combined_sheets:
            combined_sheets = {_text(item.get("sheet") or item.get("drawing_id")) for item in _as_list(combined.get("drawings")) if isinstance(item, dict) and _text(item.get("sheet") or item.get("drawing_id"))}
        for graph in graphs:
            sheet = _sheet_id(graph)
            combined_sheets_ok = not combined_sheets or sheet in combined_sheets
            if not combined_sheets_ok:
                completeness_missing.append(f"drawing:{sheet}")
            for local_key, combined_key in collection_pairs:
                local_records = [item for item in _as_list(graph.get(local_key)) if isinstance(item, dict)]
                combined_records = [item for item in _as_list(combined.get(combined_key)) if isinstance(item, dict)]
                for item in local_records:
                    local_id = _record_id(item)
                    if not local_id:
                        continue
                    match = next((candidate for candidate in combined_records if (_text(candidate.get("local_id")) == local_id and _text(candidate.get("sheet") or candidate.get("drawing_id")) == sheet) or (_text(candidate.get("id")).endswith(f"::{sheet}::{local_id}") or _text(candidate.get("id")).endswith(f"::{local_id}"))), None)
                    if match is None:
                        completeness_missing.append(f"{sheet}:{combined_key}:{local_id}")
                    elif not _text(match.get("sheet") or match.get("drawing_id")) or not isinstance(match.get("provenance"), dict):
                        provenance_missing.append(f"{combined_key}:{local_id}")
        if expectations.get("require_combined_completeness"):
            _check(checks, violations, "combined_graph_completeness", not completeness_missing, observed=sorted(completeness_missing), expected=[], message="combined graph dropped local entities or drawings")
            _check(checks, violations, "combined_graph_provenance", not provenance_missing, observed=sorted(provenance_missing), expected=[], message="combined entities lack per-sheet provenance")
    elif expectations.get("required_cross_sheet_relationships"):
        _check(checks, violations, "multi_sheet_connector_continuity", False, observed=0, expected=expectations["required_cross_sheet_relationships"], message="multi-sheet continuity was expected but no combined graph was exported")

    # Expectations for required relationships and explicit unresolved counts.
    relation_types = {_text(item.get("type")) for graph in graphs for item in _as_list(graph.get("relationships")) if isinstance(item, dict)}
    for relation_type in sorted(set(expectations.get("required_relationship_types", []))):
        _check(checks, violations, f"relationship_type:{relation_type}", relation_type in relation_types, observed=sorted(relation_types), expected=relation_type, message=f"required relationship type {relation_type!r} is missing")
    for kind, minimum in (expectations.get("minimum_unresolved") or {}).items():
        actual = sum(1 for item in report["unresolved_evidence"] if item["kind"] == kind)
        _check(checks, violations, f"unresolved_retained:{kind}", actual >= int(minimum), observed=actual, expected=int(minimum), message="expected unresolved evidence was discarded")

    # Geometry/topology contract cases are deliberately expressed as fixture
    # invariants.  They test how a graph records ambiguity without claiming a
    # detector score: a crossing remains two physical routes, while a tee has
    # an explicit junction endpoint.
    route_by_id = {_text(route.get("id")): route for _graph, route in all_routes if _text(route.get("id"))}
    for item in expectations.get("crossing_without_junction", []) or []:
        first = route_by_id.get(_text(item.get("route_a")), {})
        second = route_by_id.get(_text(item.get("route_b")), {})
        junction = _text(item.get("junction_id"))
        first_endpoints, second_endpoints = _endpoints(first), _endpoints(second)
        passed = bool(first and second and junction and junction not in first_endpoints and junction not in second_endpoints)
        _check(checks, violations, "crossing_without_junction", passed, observed={"route_a": first_endpoints, "route_b": second_endpoints}, expected=f"{junction!r} is not a route endpoint", message="a visual crossing was promoted to a physical tee")
    for item in expectations.get("tee_junctions", []) or []:
        junction = _text(item.get("junction_id"))
        degree = sum(1 for _graph, route in all_routes if junction in _endpoints(route))
        _check(checks, violations, "explicit_tee_junction", degree >= int(item.get("minimum_degree", 3)), observed=degree, expected=f">={int(item.get('minimum_degree', 3))} routes", message="tee connectivity is missing an explicit junction endpoint")
    repeated = expectations.get("repeated_sheet_scoped_occurrences")
    if isinstance(repeated, dict):
        occurrence_ids = {
            _text(occurrence.get("id") or occurrence.get("occurrence_id"))
            for graph in graphs
            for occurrence in _as_list(graph.get("line_occurrences") or graph.get("line_number_occurrences"))
            if isinstance(occurrence, dict)
        }
        occurrence_ids.update(
            _text(occurrence.get("id") or occurrence.get("occurrence_id"))
            for graph in graphs
            for line in _as_list(graph.get("lines"))
            if isinstance(line, dict)
            for occurrence in _as_list(line.get("occurrences") or line.get("line_number_occurrences"))
            if isinstance(occurrence, dict)
        )
        expected_count = int(repeated.get("minimum_distinct", 2))
        _check(checks, violations, "sheet_scoped_occurrence_identity", len(occurrence_ids) >= expected_count, observed=len(occurrence_ids), expected=expected_count, message="repeated line labels lost drawing-scoped occurrence identities")
    unresolved_connector = expectations.get("unresolved_connector_no_continuity")
    if isinstance(unresolved_connector, dict):
        connector_ids = {
            _text(item.get("id"))
            for item in _as_list((combined or {}).get("connectors"))
            if isinstance(item, dict) and _state(item) in _UNRESOLVED_STATES
        }
        continuity = [item for item in _as_list((combined or {}).get("relationships")) if isinstance(item, dict) and _text(item.get("type")) == "cross_sheet_continues"]
        connected_unresolved = any(_text(item.get("source")) in connector_ids or _text(item.get("target")) in connector_ids for item in continuity)
        _check(checks, violations, "unresolved_connector_no_continuity", not connected_unresolved, observed=connected_unresolved, expected=False, message="an unresolved connector created cross-sheet continuity")
    instrument_semantics = expectations.get("required_instrument_semantics")
    if instrument_semantics:
        semantic_types = {
            _text(item.get("type"))
            for graph in graphs
            for item in _as_list(graph.get("relationships"))
            if isinstance(item, dict) and _text(item.get("type")) in {"measures", "controls", "actuates", "instrument_association"}
        }
        expected_types = {str(value) for value in instrument_semantics}
        _check(checks, violations, "explicit_instrument_semantics", expected_types.issubset(semantic_types), observed=sorted(semantic_types), expected=sorted(expected_types), message="instrument semantics were collapsed into an untyped association")

    report["violations"] = sorted(report["violations"], key=lambda item: (str(item.get("check")), str(item.get("message"))))
    report["unresolved_evidence"] = sorted(report["unresolved_evidence"], key=lambda item: (str(item.get("kind")), str(item.get("id"))))
    report["checks"] = sorted(report["checks"], key=lambda item: str(item.get("id")))
    report["passed"] = not report["violations"]
    return report


def validate_benchmark_case(case: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Validate one fixture case, optionally raising on contract violations."""
    if not isinstance(case, dict) or not isinstance(case.get("payload"), dict):
        raise ValueError("benchmark case requires an object payload")
    source_graph = case.get("source_graph") if isinstance(case.get("source_graph"), dict) else None
    report = evaluate_graph_payload(case["payload"], expectations=case.get("expectations"), case_id=_text(case.get("id")) or "case", source_graph=source_graph)
    if strict and not report["passed"]:
        raise BenchmarkContractError(report)
    return report


def run_phase9_benchmark(source: dict[str, Any] | str | Path = DEFAULT_FIXTURE_PATH, *, strict: bool = False) -> dict[str, Any]:
    """Run all representative cases and return a deterministic aggregate report."""
    if isinstance(source, (str, Path)):
        bundle = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        bundle = source
    if not isinstance(bundle, dict) or not isinstance(bundle.get("cases"), list):
        raise ValueError("benchmark bundle requires a cases array")
    reports = [validate_benchmark_case(case, strict=False) for case in bundle["cases"]]
    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "fixture_schema_version": _text(bundle.get("schema_version")),
        "case_count": len(reports),
        "passed": all(item["passed"] for item in reports),
        "violations": sum((item["violations"] for item in reports), []),
        "cases": sorted(reports, key=lambda item: str(item["case_id"])),
        "acceptance_thresholds": {
            "contract_violations": 0,
            "route_pixel_geometry_ratio": 1.0,
            "physical_connectivity_ratio": 1.0,
            "line_occurrence_traceability_ratio": 1.0,
            "relationship_endpoint_validity_ratio": 1.0,
            "detector_accuracy": "not_scored",
        },
        "detector_accuracy": {"status": "not_scored", "reason": "No annotated gold truth is included in the benchmark fixtures"},
    }
    metric_names = sorted({
        name
        for item in reports
        for name in item.get("metrics", {}).get("coverage_validity", {})
    })
    aggregate_metrics: dict[str, Any] = {}
    for name in metric_names:
        covered = total = 0
        for item in reports:
            metric = item.get("metrics", {}).get("coverage_validity", {}).get(name)
            if isinstance(metric, dict) and isinstance(metric.get("covered"), int) and isinstance(metric.get("total"), int):
                covered += metric["covered"]
                total += metric["total"]
        aggregate_metrics[name] = _metric(covered, total)
    report["metrics"] = {"coverage_validity": aggregate_metrics, "detector_accuracy": report["detector_accuracy"]}
    report["violations"] = sorted(report["violations"], key=lambda item: (str(item.get("check")), str(item.get("message"))))
    if strict and not report["passed"]:
        raise BenchmarkContractError(report)
    return report


# Short aliases for callers that prefer the generic benchmark terminology.
evaluate_benchmark = evaluate_graph_payload
run_benchmark = run_phase9_benchmark


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--report", type=Path, help="write the JSON report to this path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_phase9_benchmark(args.fixture, strict=False)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
