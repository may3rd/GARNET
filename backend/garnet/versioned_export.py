"""Versioned, evidence-preserving downstream export.

The existing ``graph_v1`` payload is intentionally left untouched.  This
module is an additive projection for consumers that need stable identities,
ordered pixel routes, and an explicit distinction between physical
connectivity and process-flow evidence.

The projection is deliberately conservative: records are copied from the
graph and optional Stage 10 views, and missing facts remain missing.  The
validator is useful both before writing an export and when accepting one from
another process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any


SCHEMA_VERSION = "garnet_downstream_export_v1"
EXPORT_KIND = "pid_downstream_graph"
VERSIONED_EXPORT_SCHEMA_VERSION = SCHEMA_VERSION
DOWNSTREAM_EXPORT_SCHEMA_VERSION = SCHEMA_VERSION
FLOW_STATES = {"forward", "reverse", "bidirectional", "unknown", "conflicting"}
RELATIONSHIP_TYPES = {
    "connects_to", "branches_from", "cross_sheet_continues", "has_port",
    "has_inline_object", "measures", "controls", "actuates",
    "instrument_association", "edge_to_equipment",
}
RESOLVED_STATES = {"accepted", "observed", "reviewed", "resolved", "inferred"}
ENTITY_COLLECTIONS = (
    "nodes",
    "routes",
    "lines",
    "line_number_occurrences",
    "equipment",
    "ports",
    "instruments",
    "inline_objects",
    "connectors",
    "relationships",
    "boundaries",
    "test_packages",
)

# A compact machine-readable descriptor is kept beside the projection code so
# downstream adapters can advertise the contract without importing a JSON
# Schema package.  Detailed invariants (references, hashes, and finite values)
# are enforced by ``validate_downstream_export``.
EXPORT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GARNET downstream P&ID export",
    "type": "object",
    "required": ["schema_version", "source", "drawings", "graph", "engineering_views", "uncertainty", "review", "release_ready"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "graph_content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "source": {"type": "object", "required": ["graph_content_sha256"]},
        "drawings": {"type": "array"},
        "graph": {"type": "object", "required": ["physical_multigraph", "flow_direction_separate", "nodes", "routes", "lines", "line_number_occurrences", "equipment", "ports", "instruments", "inline_objects", "relationships"]},
        "engineering_views": {"type": "object", "required": ["boundaries", "test_packages"]},
    },
}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _assert_finite(value: Any, path: str = "$ ".strip()) -> None:
    """Reject NaN and infinity before they can be changed into null."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non_finite_number at {path}: use a finite number or omit the value")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _safe_copy(value: Any) -> Any:
    _assert_finite(value)
    if isinstance(value, dict):
        return {str(key): _safe_copy(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_safe_copy(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_safe_copy(item) for item in sorted(value, key=lambda item: repr(item))]
    return copy.deepcopy(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


_UNORDERED_GRAPH_LISTS = {
    "nodes", "edges", "routes", "lines", "canonical_lines", "equipment", "equipment_catalog",
    "equipment_ports", "ports", "instruments", "instrument_catalog", "inline_objects",
    "inline_object_catalog", "connectors", "relationships", "issues", "sheets", "drawings",
}


def _canonical_graph(value: Any, key: str = "") -> Any:
    """Normalize only record collections; preserve ordered evidence and routes."""
    if isinstance(value, dict):
        return {str(name): _canonical_graph(item, str(name)) for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        items = [_canonical_graph(item, key) for item in value]
        if key in _UNORDERED_GRAPH_LISTS and all(isinstance(item, dict) for item in items):
            return sorted(items, key=lambda item: _canonical_json(item))
        if key in {"sheets"} and all(isinstance(item, (str, int, float)) for item in items):
            return sorted(items, key=lambda item: str(item))
        return items
    if isinstance(value, tuple):
        return [_canonical_graph(item, key) for item in value]
    return copy.deepcopy(value)


def graph_content_sha256(graph_payload: dict[str, Any]) -> str:
    """Return the stable content hash used to bind an export to its graph."""
    if not isinstance(graph_payload, dict):
        raise ValueError("graph_payload must be an object")
    return hashlib.sha256(_canonical_json(_canonical_graph(_safe_copy(graph_payload))).encode("utf-8")).hexdigest()


def release_gate_sha256(release_gate: dict[str, Any]) -> str:
    """Return the deterministic binding hash for a release-gate payload."""
    if not isinstance(release_gate, dict):
        raise ValueError("release_gate must be an object")
    return hashlib.sha256(_canonical_json(_safe_copy(release_gate)).encode("utf-8")).hexdigest()


# Descriptive alias used by downstream contract adapters.
release_gate_content_sha256 = release_gate_sha256


def _as_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = list(value.values())
    return [item for item in value or [] if isinstance(item, dict)] if isinstance(value, list) else []


def _records(graph: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        if key in graph:
            return _as_records(graph.get(key))
    return []


def _record_id(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _drawing_id(graph: dict[str, Any]) -> str:
    document = graph.get("document") if isinstance(graph.get("document"), dict) else {}
    return _record_id(graph, "drawing_id", "image_id") or _record_id(document, "drawing_id", "doc_id", "sheet")


def _source_graph(graph_payload: dict[str, Any]) -> dict[str, Any]:
    combined = graph_payload.get("combined_graph")
    if isinstance(combined, dict):
        return combined
    return graph_payload


def _drawing_records(graph_payload: dict[str, Any], graph: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _records(graph, "drawings")
    if not raw and graph is not graph_payload:
        raw = _records(graph_payload, "drawings")
    result: list[dict[str, Any]] = []
    for item in raw:
        document = item.get("document") if isinstance(item.get("document"), dict) else item
        drawing_id = _record_id(item, "drawing_id", "sheet", "doc_id") or _record_id(document, "drawing_id", "doc_id", "sheet")
        if not drawing_id:
            continue
        source = document.get("source") if isinstance(document.get("source"), dict) else {}
        image = document.get("image") if isinstance(document.get("image"), dict) else {}
        dimensions = item.get("pixel_dimensions") or document.get("pixel_dimensions")
        if not dimensions and image:
            dimensions = {key: image[key] for key in ("width", "height") if key in image}
        result.append({
            **copy.deepcopy(item),
            "drawing_id": drawing_id,
            "revision": item.get("revision", document.get("revision")),
            "content_sha256": item.get("content_sha256", document.get("content_sha256")),
            "pixel_dimensions": dimensions,
            "coordinate_system": item.get("coordinate_system", document.get("coordinate_system")),
            "sheet": item.get("sheet", drawing_id),
            "document": copy.deepcopy(document),
            "source": source,
            "provenance": item.get("provenance", document.get("provenance")),
        })
    if result:
        return result
    source_drawing = graph_payload.get("drawing") if isinstance(graph_payload.get("drawing"), dict) else {}
    document = graph_payload.get("document") if isinstance(graph_payload.get("document"), dict) else {}
    drawing_id = _record_id(source_drawing, "drawing_id", "doc_id", "sheet") or _drawing_id(graph_payload)
    if not drawing_id:
        return []
    image = document.get("image") if isinstance(document.get("image"), dict) else {}
    dimensions = source_drawing.get("pixel_dimensions") or document.get("pixel_dimensions") or ({key: image[key] for key in ("width", "height") if key in image} or None)
    return [{
        "drawing_id": drawing_id,
        "revision": source_drawing.get("revision", graph_payload.get("revision", document.get("revision"))),
        "content_sha256": source_drawing.get("content_sha256", graph_payload.get("content_sha256", document.get("content_sha256"))),
        "pixel_dimensions": dimensions,
        "coordinate_system": source_drawing.get("coordinate_system", graph_payload.get("coordinate_system", document.get("coordinate_system"))),
        "source": source_drawing.get("source", document.get("source")),
        "provenance": source_drawing.get("provenance", graph_payload.get("provenance")),
    }]


def _polyline(edge: dict[str, Any]) -> Any:
    geometry = edge.get("geometry") if isinstance(edge.get("geometry"), dict) else {}
    if "polyline" in geometry:
        return copy.deepcopy(geometry["polyline"])
    return copy.deepcopy(edge.get("polyline"))


def _flow_state(edge: dict[str, Any]) -> str:
    aliases = {"source_to_target": "forward", "target_to_source": "reverse", "both": "bidirectional"}
    state = _text(edge.get("flow_direction_state") or edge.get("flow_direction")).lower()
    state = aliases.get(state, state)
    return state if state in FLOW_STATES else "unknown"


def _endpoint(edge: dict[str, Any], side: str) -> str:
    if side == "source":
        return _text(edge.get("source") or edge.get("canonical_src") or edge.get("src"))
    return _text(edge.get("target") or edge.get("canonical_dst") or edge.get("dst"))


def _route(edge: dict[str, Any], drawing_id: str) -> dict[str, Any]:
    route_id = _record_id(edge, "id", "edge_id")
    source, target = _endpoint(edge, "source"), _endpoint(edge, "target")
    state = _flow_state(edge)
    flow: dict[str, Any] = {
        "state": state,
        "confidence": edge.get("flow_direction_confidence"),
        "evidence": copy.deepcopy(edge.get("flow_direction_evidence") or edge.get("direction_evidence") or []),
        "review_state": edge.get("flow_direction_review_state") or edge.get("direction_review_state"),
    }
    if state in {"forward", "reverse"}:
        flow["source"] = _text(edge.get("flow_src")) or (target if state == "reverse" else source)
        flow["target"] = _text(edge.get("flow_dst")) or (source if state == "reverse" else target)
    attachments = edge.get("attachments") if isinstance(edge.get("attachments"), dict) else {}
    inline_ids = []
    for item in _as_records(attachments.get("inline_objects")):
        value = _record_id(item, "canonical_inline_object_id", "inline_object_id", "source_object_id", "id")
        if value and value not in inline_ids:
            inline_ids.append(value)
    return {
        "id": route_id,
        "drawing_id": _text(edge.get("drawing_id") or edge.get("sheet") or edge.get("drawing")) or drawing_id,
        "physical": {"source": source, "target": target},
        # Convenience aliases keep route consumers simple while the nested
        # physical object makes the separation from flow explicit.
        "source": source,
        "target": target,
        "polyline": _polyline(edge),
        "flow": flow,
        "line_ids": ([edge.get("canonical_line_id")] if _text(edge.get("canonical_line_id")) else
                     copy.deepcopy(edge.get("canonical_line_ids") or [])),
        "line_number_occurrence_ids": copy.deepcopy(edge.get("line_number_ids") or []),
        "inline_object_ids": inline_ids,
        "provenance": copy.deepcopy(edge.get("provenance")),
        "confidence": edge.get("confidence"),
        "review_state": edge.get("review_state"),
        "source_record": copy.deepcopy(edge),
    }


# Catalog collections the Stage 9 corrected graph never carries at top level,
# mapped to the edge-attachment key that holds the same records.
_ATTACHMENT_CATALOG_KEYS = {"inline_objects": "inline_objects"}


def _attachment_catalog(graph: dict[str, Any], attachment_key: str) -> list[dict[str, Any]]:
    """Rebuild a catalog from the edge attachments that reference it.

    The Stage 9 corrected graph carries inline objects and instrument tags
    only as per-edge attachments -- it has no top-level catalog collection.
    Deriving the catalog from those same attachment records keeps the ids
    identical to the ones ``_route`` emits, so released routes never dangle.
    """
    catalog: dict[str, dict[str, Any]] = {}
    for edge in _records(graph, "edges", "routes"):
        attachments = edge.get("attachments") if isinstance(edge.get("attachments"), dict) else {}
        for item in _as_records(attachments.get(attachment_key)):
            item_id = _record_id(item, f"canonical_{attachment_key[:-1]}_id", f"{attachment_key[:-1]}_id", "source_object_id", "id")
            if item_id and item_id not in catalog:
                catalog[item_id] = {**copy.deepcopy(item), "id": item_id}
    return list(catalog.values())


def _catalog_records(graph: dict[str, Any], drawing_id: str, kind: str) -> list[dict[str, Any]]:
    keys = {
        # Raw detector ``objects`` are provisional evidence and are not
        # promoted to equipment merely because a catalog is absent.
        "equipment": ("equipment", "equipment_catalog"),
        "ports": ("ports", "equipment_ports"),
        "instruments": ("instruments", "instrument_catalog"),
        "inline_objects": ("inline_objects", "inline_object_catalog"),
        "lines": ("lines", "canonical_lines"),
        "connectors": ("connectors",),
    }
    source_records = _records(graph, *keys[kind])
    if not source_records and kind in _ATTACHMENT_CATALOG_KEYS:
        source_records = _attachment_catalog(graph, _ATTACHMENT_CATALOG_KEYS[kind])
    result = []
    for item in source_records:
        item = copy.deepcopy(item)
        if kind == "ports":
            item["id"] = _record_id(item, "id", "port_id")
            item.setdefault("equipment_id", _record_id(item, "equipment_id"))
        elif kind == "lines":
            item["id"] = _record_id(item, "id", "line_id", "canonical_line_id")
        else:
            item["id"] = _record_id(item, "id", f"{kind[:-1]}_id", "source_object_id", "canonical_id")
        if item.get("id"):
            item_drawing_id = _text(item.get("drawing_id") or item.get("sheet") or item.get("drawing"))
            item.setdefault("drawing_id", item_drawing_id or drawing_id)
            result.append(item)
    return result


def _line_occurrences(graph: dict[str, Any], lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in lines:
        for occurrence in _as_records(line.get("occurrences") or line.get("line_occurrences")):
            item = copy.deepcopy(occurrence)
            occurrence_id = _record_id(item, "occurrence_id", "id", "source_record_id")
            if not occurrence_id or occurrence_id in seen:
                continue
            seen.add(occurrence_id)
            item["id"] = occurrence_id
            item["line_id"] = line["id"]
            item.setdefault("text", item.get("raw_text") or item.get("normalized_text"))
            result.append(item)
    # Some graph-v1 payloads keep OCR observations in a flat catalog.
    for occurrence in _records(graph, "line_number_occurrences", "line_occurrences"):
        occurrence_id = _record_id(occurrence, "occurrence_id", "id", "source_record_id")
        if not occurrence_id or occurrence_id in seen:
            continue
        item = copy.deepcopy(occurrence)
        item["id"] = occurrence_id
        result.append(item)
        seen.add(occurrence_id)
    # Other graph-v1 payloads keep OCR observations on edges rather than lines.
    for edge in _records(graph, "edges"):
        for occurrence in _as_records(edge.get("effective_line_numbers") or edge.get("line_numbers")):
            occurrence_id = _record_id(occurrence, "occurrence_id", "id", "source_object_id")
            if not occurrence_id or occurrence_id in seen:
                continue
            item = copy.deepcopy(occurrence)
            item["id"] = occurrence_id
            item.setdefault("line_id", _record_id(occurrence, "canonical_line_id"))
            result.append(item)
            seen.add(occurrence_id)
    return result


def _view_records(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if isinstance(value, dict):
                # Stage 10 uses {schema_version, candidates: [...]}; accept
                # that wrapper without promoting its metadata as a candidate.
                if isinstance(value.get("candidates"), list):
                    return _as_records(value["candidates"])
                if isinstance(value.get("boundaries"), list):
                    return _as_records(value["boundaries"])
                if isinstance(value.get("test_packages"), list):
                    return _as_records(value["test_packages"])
            return _as_records(value)
    return []


def _relationships(graph: dict[str, Any], routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [copy.deepcopy(item) for item in _records(graph, "relationships") if _record_id(item, "id", "relationship_id")]
    existing = {_record_id(item, "id", "relationship_id") for item in result}
    for route in routes:
        source, target = route["physical"]["source"], route["physical"]["target"]
        if not source or not target:
            continue
        rel_id = f"connects_to::{route['id']}"
        if rel_id in existing:
            continue
        result.append({
            "id": rel_id,
            "type": "connects_to",
            "source": source,
            "target": target,
            "route_id": route["id"],
            "state": route.get("review_state") or "unresolved",
            "provenance": copy.deepcopy(route.get("provenance")),
        })
        existing.add(rel_id)
    return result


def _relationship_state(item: dict[str, Any]) -> str:
    for key in ("review_state", "state", "semantic_state", "status"):
        value = _text(item.get(key)).lower()
        if value:
            return value
    return ""


def _resolved_cross_sheet_relationships(relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one explicitly resolved continuity record per connector pair."""
    result: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for item in relationships:
        if _text(item.get("type")) != "cross_sheet_continues":
            continue
        if _relationship_state(item) not in RESOLVED_STATES:
            continue
        source, target = _text(item.get("source")), _text(item.get("target"))
        if not source or not target or source == target:
            continue
        pair = tuple(sorted((source, target)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        result.append(item)
    return result


def build_downstream_export(
    graph_payload: dict[str, Any],
    *,
    boundary_payload: dict[str, Any] | None = None,
    test_package_payload: dict[str, Any] | None = None,
    engineering_views: dict[str, Any] | None = None,
    process_exports: dict[str, Any] | None = None,
    phase8_views: dict[str, Any] | None = None,
    release_gate: dict[str, Any] | None = None,
    source_graph_artifact: str | None = None,
    source_release_gate_artifact: str | None = None,
    source_release_gate_sha256: str | None = None,
    scope: str = "page",
    blocked_reasons: list[Any] | None = None,
    source_graph_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic ``garnet_downstream_export_v1`` envelope."""
    if not isinstance(graph_payload, dict):
        raise ValueError("graph_payload must be an object")
    _assert_finite(graph_payload)
    if boundary_payload is not None:
        _assert_finite(boundary_payload)
    if test_package_payload is not None:
        _assert_finite(test_package_payload)
    if engineering_views is not None:
        _assert_finite(engineering_views)
    for value in (process_exports, phase8_views, release_gate, blocked_reasons):
        if value is not None:
            _assert_finite(value)

    graph = _source_graph(graph_payload)
    drawing_id = _drawing_id(graph_payload) or _drawing_id(graph)
    drawings = _drawing_records(graph_payload, graph)
    if drawing_id and (not isinstance(graph_payload.get("combined_graph"), dict) or not drawings) and not any(item.get("drawing_id") == drawing_id for item in drawings):
        # The combined graph can have only qualified drawing records; retain
        # the input document identity as a separate drawing when it is known.
        drawings.append({"drawing_id": drawing_id, "revision": None, "content_sha256": None,
                         "pixel_dimensions": None, "coordinate_system": None, "source": None, "provenance": None})
    drawings.sort(key=lambda item: _text(item.get("drawing_id")))

    routes = [_route(edge, _text(edge.get("drawing_id") or edge.get("sheet") or edge.get("drawing")) or drawing_id) for edge in _records(graph, "edges", "routes")]
    routes.sort(key=lambda item: item["id"])
    nodes = [copy.deepcopy(item) for item in _records(graph, "nodes") if _record_id(item, "id", "node_id")]
    for item in nodes:
        item["id"] = _record_id(item, "id", "node_id")
        item.setdefault("drawing_id", _text(item.get("drawing_id") or item.get("sheet")) or drawing_id)
    nodes.sort(key=lambda item: item["id"])

    catalogs = {kind: _catalog_records(graph, drawing_id, kind) for kind in ("equipment", "ports", "instruments", "inline_objects", "lines", "connectors")}
    for values in catalogs.values():
        values.sort(key=lambda item: item["id"])
    occurrences = _line_occurrences(graph, catalogs["lines"])
    engineering_views = engineering_views or {}
    boundaries = _view_records(boundary_payload or engineering_views, "boundaries", "process_boundaries", "candidates")
    packages = _view_records(test_package_payload or engineering_views, "test_packages", "candidates")
    relationships = _relationships(graph, routes)
    boundaries.sort(key=lambda item: _record_id(item, "id", "boundary_id"))
    packages.sort(key=lambda item: _record_id(item, "id", "test_package_id"))
    relationships.sort(key=lambda item: _record_id(item, "id", "relationship_id"))

    combined_relationships = _resolved_cross_sheet_relationships(relationships)
    combined = None
    if isinstance(graph_payload.get("combined_graph"), dict) or combined_relationships:
        combined_graph = graph_payload.get("combined_graph") if isinstance(graph_payload.get("combined_graph"), dict) else graph
        combined = {
            "is_combined": True,
            "drawings": copy.deepcopy(combined_graph.get("drawings") or drawings),
            "relationships": combined_relationships,
            "continuity_relationships": combined_relationships,
            "issues": copy.deepcopy(combined_graph.get("issues") or []),
            "source_schema_version": combined_graph.get("schema_version"),
        }

    release_gate = release_gate or graph_payload.get("release_gate") or graph_payload.get("stage9_release_gate") or graph.get("release_gate") or graph.get("stage9_release_gate")
    release_gate = copy.deepcopy(release_gate) if isinstance(release_gate, dict) else None
    computed_gate_sha256 = release_gate_sha256(release_gate) if release_gate is not None else None
    uncertainty = copy.deepcopy(graph.get("uncertainty") or graph.get("issues") or [])
    for candidate in [*boundaries, *packages]:
        uncertainty.extend(copy.deepcopy(candidate.get("uncertainty") or []))
    # A blocked envelope may redact the projected collections, but its source
    # binding must still identify the canonical graph that was evaluated.
    content_hash = graph_content_sha256(source_graph_payload or graph_payload)
    export = {
        "schema_version": SCHEMA_VERSION,
        "export_kind": EXPORT_KIND,
        "schema_semver": "1.0.0",
        "graph_content_sha256": content_hash,
        "source": {
            "schema_version": graph_payload.get("schema_version"),
            "graph_content_sha256": content_hash,
            "source_graph_artifact": source_graph_artifact or graph_payload.get("source_graph_artifact"),
            "source_release_gate_artifact": source_release_gate_artifact,
            "release_gate_sha256": source_release_gate_sha256 or computed_gate_sha256,
            "release_gate_content_sha256": source_release_gate_sha256 or computed_gate_sha256,
            "scope": scope,
        },
        "drawings": drawings,
        "graph": {
            # Physical pipe connectivity is undirected and may contain
            # parallel routes.  Process flow is evidence carried separately
            # on each route, so consumers must not infer direction from the
            # physical edge endpoints.
            "directed_multigraph": False,
            "physical_multigraph": True,
            "flow_direction_separate": True,
            "nodes": nodes,
            "routes": routes,
            "lines": catalogs["lines"],
            "line_number_occurrences": occurrences,
            "equipment": catalogs["equipment"],
            "ports": catalogs["ports"],
            "instruments": catalogs["instruments"],
            "inline_objects": catalogs["inline_objects"],
            "connectors": catalogs["connectors"],
            "relationships": relationships,
        },
        "engineering_views": {"boundaries": boundaries, "test_packages": packages},
        "uncertainty": uncertainty,
        "review": {
            "release_gate": copy.deepcopy(release_gate),
            "release_gate_sha256": computed_gate_sha256,
            "release_gate_content_sha256": computed_gate_sha256,
            "state": graph_payload.get("review_state"),
            "release_ready": bool(release_gate and release_gate.get("release_ready") is True),
        },
        "provenance": copy.deepcopy(graph_payload.get("provenance")),
    }
    export["release_ready"] = bool(release_gate and release_gate.get("release_ready") is True)
    export["blocked_reasons"] = [str(item) for item in (blocked_reasons or []) if str(item)]
    if process_exports is not None:
        export["legacy_process_exports"] = copy.deepcopy(process_exports)
    if phase8_views is not None:
        export["phase8_views"] = copy.deepcopy(phase8_views)
    if combined is not None:
        export["combined_continuity"] = combined
        combined_graph = graph_payload.get("combined_graph")
        if isinstance(combined_graph, dict):
            # Retain the complete qualified graph for system consumers.  The
            # continuity summary above is intentionally only a convenience
            # view and must never replace this source graph.
            export["combined_graph"] = copy.deepcopy(combined_graph)
    # A downstream envelope is public only after an explicit Stage 9 release
    # gate.  This applies even to node-only or catalog-only payloads: absence
    # of routes is not evidence that the graph is safe to release.
    if release_gate is None or release_gate.get("release_ready") is not True:
        # Retain identity, gate, and uncertainty evidence but keep blocked
        # graph facts out of the public downstream envelope.
        export["drawings"] = [
            {"drawing_id": item.get("drawing_id"), "sheet": item.get("sheet", item.get("drawing_id")), "redacted": True}
            for item in export["drawings"] if item.get("drawing_id")
        ]
        export["graph"] = {
            "redacted": True,
            "redaction_reason": "release_gate_blocked",
            "directed_multigraph": False,
            "physical_multigraph": True,
            "flow_direction_separate": True,
            **{collection: [] for collection in ENTITY_COLLECTIONS},
        }
        export["engineering_views"] = {"boundaries": [], "test_packages": []}
        export["uncertainty"] = []
        export.pop("legacy_process_exports", None)
        export.pop("phase8_views", None)
        export.pop("combined_continuity", None)
        export.pop("combined_graph", None)
        export["review"]["blocked_redaction"] = True
    export = _safe_copy(export)
    issues = validate_downstream_export(export)
    if issues["issues"]:
        # The builder may emit unresolved candidates, but structural defects
        # such as dangling required references are still caller errors.
        structural = [item for item in issues["issues"] if item.get("severity") == "error"]
        if structural:
            raise ValueError("invalid downstream export: " + "; ".join(item["message"] for item in structural))
    return export


def build_blocked_export(
    graph_payload: dict[str, Any],
    *,
    release_gate: dict[str, Any] | None = None,
    blocked_reasons: list[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a redacted envelope when the release gate is not satisfied.

    Identities and source provenance remain available for diagnostics, while
    all released graph collections are empty.  This prevents a stale or
    unresolved graph from being mistaken for a downstream release.
    """
    release_gate = release_gate or graph_payload.get("release_gate") or graph_payload.get("stage9_release_gate")
    redacted = copy.deepcopy(graph_payload)
    graph = redacted.get("combined_graph") if isinstance(redacted.get("combined_graph"), dict) else redacted
    if isinstance(graph, dict):
        graph["redacted"] = True
        graph["redaction_reason"] = "release_gate_blocked"
        for key in ("nodes", "edges", "routes", "lines", "canonical_lines", "equipment", "equipment_catalog", "equipment_ports", "ports", "instruments", "instrument_catalog", "inline_objects", "inline_object_catalog", "connectors", "relationships"):
            if key in graph:
                graph[key] = []
    kwargs.setdefault("source_graph_payload", graph_payload)
    return build_downstream_export(
        redacted,
        release_gate=release_gate,
        blocked_reasons=blocked_reasons or ["release_gate_not_ready"],
        **kwargs,
    )


def validate_downstream_export(payload: Any, *, source_graph: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``{valid, issues}`` with actionable structural diagnostics."""
    issues: list[dict[str, Any]] = []

    def add(path: str, code: str, message: str, suggestion: str | None = None, severity: str = "error") -> None:
        item = {"path": path, "code": code, "message": message, "severity": severity}
        if suggestion:
            item["suggestion"] = suggestion
        issues.append(item)

    try:
        _assert_finite(payload)
    except ValueError as exc:
        add("$", "non_finite_number", str(exc), "Replace NaN or infinity with a measured value or omit the field")
        return {"valid": False, "issues": issues}
    try:
        _canonical_json(payload)
    except (TypeError, ValueError) as exc:
        add("$", "non_json_value", f"export contains a value that strict JSON cannot encode: {exc}", "Convert sets and custom objects to JSON arrays or objects")
        return {"valid": False, "issues": issues}
    if not isinstance(payload, dict):
        add("$", "wrong_type", "export must be a JSON object", "Pass the versioned export envelope")
        return {"valid": False, "issues": issues}
    if payload.get("schema_version") != SCHEMA_VERSION:
        add("$.schema_version", "unsupported_schema_version", f"expected {SCHEMA_VERSION!r}", "Migrate the payload before downstream use")
    source = payload.get("source")
    if not isinstance(source, dict):
        add("$.source", "missing_source", "source graph binding is required", "Include source.graph_content_sha256")
    else:
        declared = _text(source.get("graph_content_sha256"))
        if len(declared) != 64:
            add("$.source.graph_content_sha256", "invalid_hash", "graph content hash must be a 64-character SHA-256", "Use lowercase hexadecimal SHA-256")
        if source_graph is not None:
            try:
                actual = graph_content_sha256(source_graph)
                if declared != actual:
                    add("$.source.graph_content_sha256", "hash_mismatch", f"declared {declared!r} does not match source graph {actual!r}", "Rebuild the export from the corrected graph revision")
            except ValueError as exc:
                add("$.source", "invalid_source_graph", str(exc))
        top_level_hash = _text(payload.get("graph_content_sha256"))
        if top_level_hash and not re.fullmatch(r"[0-9a-f]{64}", top_level_hash):
            add("$.graph_content_sha256", "invalid_hash", "top-level graph content hash must be lowercase hexadecimal SHA-256", "Use the source graph binding hash")
        if top_level_hash and declared and top_level_hash != declared:
            add("$.graph_content_sha256", "hash_mismatch", "top-level and source graph hashes differ", "Copy the same corrected-graph SHA-256 into both bindings")
        gate_hash = source.get("release_gate_sha256") or source.get("release_gate_content_sha256")
        if gate_hash is not None and (not isinstance(gate_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", gate_hash)):
            add("$.source.release_gate_sha256", "invalid_hash", "release gate hash must be a 64-character SHA-256", "Use lowercase hexadecimal SHA-256")
        source_gate = None
        if isinstance(source_graph, dict):
            source_gate = source_graph.get("release_gate") or source_graph.get("stage9_release_gate")
        review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
        review_gate = review.get("release_gate") if isinstance(review.get("release_gate"), dict) else None
        gate_for_check = source_gate or review_gate
        if gate_hash and isinstance(gate_for_check, dict):
            try:
                actual_gate_hash = release_gate_sha256(gate_for_check)
                if gate_hash != actual_gate_hash:
                    add("$.source.release_gate_sha256", "gate_hash_mismatch", "release gate hash does not match the bound release gate", "Rebuild the export from the same release-gate artifact")
            except ValueError as exc:
                add("$.source.release_gate_sha256", "invalid_release_gate", str(exc))
        review_gate_hash = review.get("release_gate_sha256") or review.get("release_gate_content_sha256")
        if review_gate_hash and gate_hash and review_gate_hash != gate_hash:
            add("$.review.release_gate_sha256", "gate_hash_mismatch", "review and source release-gate hashes differ", "Bind both fields to the same release-gate artifact")
    drawings = payload.get("drawings")
    if not isinstance(drawings, list) or not drawings:
        add("$.drawings", "missing_drawings", "at least one drawing identity is required", "Provide drawing_id and coordinate metadata")
    else:
        seen_drawings: set[str] = set()
        for index, item in enumerate(drawings):
            path = f"$.drawings[{index}]"
            if not isinstance(item, dict):
                add(path, "wrong_type", "drawing must be an object")
                continue
            drawing_id = _text(item.get("drawing_id"))
            if not drawing_id:
                add(path + ".drawing_id", "missing_drawing_id", "drawing_id is required", "Use the stable document or sheet identity")
            elif drawing_id in seen_drawings:
                add(path + ".drawing_id", "duplicate_typed_id", f"duplicate drawing id {drawing_id!r}", "Emit one drawing record per drawing_id")
            seen_drawings.add(drawing_id)
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        add("$.graph", "missing_graph", "graph catalog is required", "Include graph.nodes, graph.routes, and graph.relationships")
        return {"valid": False, "issues": issues}
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    release_gate = review.get("release_gate") if isinstance(review.get("release_gate"), dict) else None
    gate_ready = bool(release_gate and release_gate.get("release_ready") is True)
    top_release_ready = payload.get("release_ready")
    if top_release_ready is not gate_ready:
        add("$.release_ready", "release_ready_mismatch", "top-level release_ready must agree with review.release_gate.release_ready", "Derive release_ready from the bound release gate")
    route_present = bool(graph.get("routes"))
    if route_present and release_gate is None:
        add("$.review.release_gate", "missing_release_gate", "route-containing exports require an explicit release gate", "Use a blocked redacted export until Stage 9 review is available")
    source_binding = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    source_gate_hash = source_binding.get("release_gate_sha256")
    review_gate_hash = review.get("release_gate_sha256")
    if gate_ready:
        if not isinstance(source_gate_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_gate_hash):
            add("$.source.release_gate_sha256", "missing_release_gate_hash", "released exports require a valid release-gate SHA-256", "Bind the export to the exact release-gate artifact")
        if not isinstance(review_gate_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", review_gate_hash):
            add("$.review.release_gate_sha256", "missing_release_gate_hash", "released exports require a review release-gate SHA-256", "Bind review metadata to the exact release-gate artifact")
    physical_multigraph = graph.get("physical_multigraph")
    if physical_multigraph is not True:
        add("$.graph.physical_multigraph", "invalid_physical_graph", "physical connectivity must be an undirected multigraph", "Set physical_multigraph=true")
    if graph.get("flow_direction_separate") is not True:
        add("$.graph.flow_direction_separate", "flow_not_separate", "flow direction must be represented separately from physical connectivity", "Set flow_direction_separate=true and keep route.flow independent")
    if graph.get("directed_multigraph") is not False:
        add("$.graph.directed_multigraph", "directed_physical_graph", "physical connectivity cannot be directed by route storage order", "Set directed_multigraph=false")
    # Missing and negative gates are both blocked states.  Require the same
    # redaction contract for either case so a node/equipment/ports-only graph
    # cannot bypass the route-specific release check.
    if not gate_ready:
        if graph.get("redacted") is not True:
            add("$.graph", "blocked_graph_not_redacted", "a blocked release must not expose graph facts", "Use build_blocked_export or redact the graph collections")
        if any(graph.get(collection) for collection in ENTITY_COLLECTIONS):
            add("$.graph", "blocked_graph_contains_entities", "blocked exports must contain empty entity collections", "Retain blocked evidence under review or uncertainty")
    typed_ids: dict[str, set[str]] = {}
    records_by_type: dict[str, dict[str, dict[str, Any]]] = {}
    collection_types = {"nodes": "node", "routes": "route", "lines": "line", "line_number_occurrences": "line_number_occurrence",
                        "equipment": "equipment", "ports": "port", "instruments": "instrument", "inline_objects": "inline_object",
                        "connectors": "connector", "relationships": "relationship", "boundaries": "boundary", "test_packages": "test_package"}
    for collection, kind in collection_types.items():
        values = graph.get(collection)
        if collection in {"boundaries", "test_packages"}:
            values = (payload.get("engineering_views") or {}).get(collection, []) if values is None else values
        if not isinstance(values, list):
            add(f"$.graph.{collection}", "missing_collection", f"{collection} must be an array", "Emit an empty array when there are no records")
            values = []
        typed_ids[kind] = set()
        records_by_type[kind] = {}
        for index, item in enumerate(values):
            path = f"$.graph.{collection}[{index}]"
            if not isinstance(item, dict):
                add(path, "wrong_type", f"{collection} entries must be objects")
                continue
            item_id = _record_id(item, "id", f"{kind}_id")
            if not item_id:
                add(path + ".id", "missing_typed_id", f"{kind} id is required", "Preserve the source stable id")
                continue
            if item_id in typed_ids[kind]:
                add(path + ".id", "duplicate_typed_id", f"duplicate {kind} id {item_id!r}", f"Give each {kind} one stable id")
            typed_ids[kind].add(item_id)
            records_by_type[kind][item_id] = item

    id_owners: dict[str, set[str]] = {}
    for kind, ids in typed_ids.items():
        for item_id in ids:
            id_owners.setdefault(item_id, set()).add(kind)
    for item_id, owners in sorted(id_owners.items()):
        if owners == {"node", "port"} and _text(records_by_type["port"].get(item_id, {}).get("source_node_id")) == item_id:
            # An equipment port and its graph node are one physical thing. The
            # port record says so explicitly via source_node_id, so the shared
            # id is a declared identity, not an unresolvable collision.
            continue
        if len(owners) > 1:
            add("$.graph", "ambiguous_typed_id", f"id {item_id!r} is used by multiple typed collections: {sorted(owners)}", "Use globally distinct typed identifiers so physical endpoints resolve unambiguously")

    all_ids = {item_id for values in typed_ids.values() for item_id in values}
    node_or_attachable = typed_ids.get("node", set()) | typed_ids.get("equipment", set()) | typed_ids.get("port", set()) | typed_ids.get("connector", set())
    route_ids = typed_ids.get("route", set())
    for index, route in enumerate(graph.get("routes", []) if isinstance(graph.get("routes"), list) else []):
        if not isinstance(route, dict):
            continue
        path = f"$.graph.routes[{index}]"
        physical = route.get("physical") if isinstance(route.get("physical"), dict) else {}
        source = _text(physical.get("source") or route.get("source"))
        target = _text(physical.get("target") or route.get("target"))
        if not source or not target:
            add(path + ".physical", "missing_route_endpoint", "physical source and target are required", "Keep route endpoints even when flow is unknown")
        for endpoint, label in ((source, "source"), (target, "target")):
            if endpoint and endpoint not in node_or_attachable:
                add(path + f".physical.{label}", "dangling_reference", f"route endpoint {endpoint!r} does not resolve to a node, equipment, port, or connector", "Preserve the referenced entity or mark the route unresolved before export")
        polyline = route.get("polyline")
        if not isinstance(polyline, list) or len(polyline) < 2:
            add(path + ".polyline", "missing_route_geometry", "an ordered pixel polyline with at least two points is required", "Retain the traced pixel route or leave the route out of a released export")
        flow = route.get("flow")
        if not isinstance(flow, dict) or _text(flow.get("state")) not in FLOW_STATES:
            add(path + ".flow.state", "invalid_flow_state", "flow state must be one of forward, reverse, bidirectional, unknown, conflicting", "Use unknown when direction evidence is absent")
        for ref in route.get("inline_object_ids") or []:
            if _text(ref) and _text(ref) not in typed_ids.get("inline_object", set()):
                add(path + ".inline_object_ids", "dangling_reference", f"inline object {ref!r} is not present in the inline_objects catalog", "Retain the object catalog record")
        for ref in route.get("line_ids") or []:
            if _text(ref) and _text(ref) not in typed_ids.get("line", set()):
                add(path + ".line_ids", "dangling_reference", f"canonical line {ref!r} is not present in the lines catalog", "Export the canonical line record or remove the reference")
        for ref in route.get("line_number_occurrence_ids") or []:
            if _text(ref) and _text(ref) not in typed_ids.get("line_number_occurrence", set()):
                add(path + ".line_number_occurrence_ids", "dangling_reference", f"line-number occurrence {ref!r} is not present in the occurrence catalog", "Export the OCR occurrence record")
    for index, relationship in enumerate(graph.get("relationships", []) if isinstance(graph.get("relationships"), list) else []):
        if not isinstance(relationship, dict):
            continue
        path = f"$.graph.relationships[{index}]"
        source, target = _text(relationship.get("source")), _text(relationship.get("target"))
        relationship_type = _text(relationship.get("type"))
        if not relationship_type:
            add(path + ".type", "missing_relationship_type", "typed relationships require type", "Use a contract relationship type such as connects_to or measures")
        elif relationship_type not in RELATIONSHIP_TYPES:
            add(path + ".type", "unsupported_relationship_type", f"relationship type {relationship_type!r} is not in the downstream contract", "Use a supported typed relationship or retain it as unresolved evidence")
        if not source or not target:
            add(path, "missing_relationship_endpoint", "typed relationships require source and target", "Keep unresolved evidence in uncertainty instead of a partial relationship")
        for endpoint, label in ((source, "source"), (target, "target")):
            if endpoint and endpoint not in all_ids and endpoint not in route_ids:
                add(path + f".{label}", "dangling_reference", f"relationship endpoint {endpoint!r} does not resolve to an exported entity", "Export the referenced typed entity or remove the relationship")
        expected_types = {
            "has_port": ({"equipment"}, {"port"}),
            "has_inline_object": ({"route"}, {"inline_object"}),
            "measures": ({"instrument"}, {"route"}),
            "controls": ({"instrument"}, {"route"}),
            "actuates": ({"instrument"}, {"route"}),
            "instrument_association": ({"instrument"}, {"route"}),
            "cross_sheet_continues": ({"connector"}, {"connector"}),
        }.get(relationship_type)
        if expected_types:
            source_matches = any(source in typed_ids.get(kind, set()) for kind in expected_types[0])
            target_matches = any(target in typed_ids.get(kind, set()) for kind in expected_types[1])
            if not source_matches or not target_matches:
                add(path, "typed_reference_mismatch", f"{relationship_type} endpoints must be {sorted(expected_types[0])} -> {sorted(expected_types[1])}", "Preserve typed relationship endpoint identities")
    for index, port in enumerate(records_by_type.get("port", {}).values()):
        owner = _text(port.get("equipment_id"))
        if not owner:
            add(f"$.graph.ports[{index}].equipment_id", "missing_port_owner", "port equipment_id is required", "Export the owning equipment identity")
        elif owner not in typed_ids.get("equipment", set()):
            add(f"$.graph.ports[{index}].equipment_id", "dangling_reference", f"port owner {owner!r} is not in equipment", "Export the owning equipment record")
    for index, occurrence in enumerate(records_by_type.get("line_number_occurrence", {}).values()):
        line_id = _text(occurrence.get("line_id"))
        if line_id and line_id not in typed_ids.get("line", set()):
            add(f"$.graph.line_number_occurrences[{index}].line_id", "dangling_reference", f"occurrence line {line_id!r} is not in lines", "Keep ambiguous occurrences unresolved or export their canonical line")
    combined = payload.get("combined_continuity") if isinstance(payload.get("combined_continuity"), dict) else None
    if combined is not None:
        connectors = {_text(item.get("id")) for item in graph.get("connectors", []) if isinstance(item, dict)}
        continuity = [item for item in combined.get("relationships", []) if isinstance(item, dict) and _text(item.get("type")) == "cross_sheet_continues"]
        seen_pairs: set[tuple[str, str]] = set()
        for index, relation in enumerate(continuity):
            source, target = _text(relation.get("source")), _text(relation.get("target"))
            pair = tuple(sorted((source, target)))
            if source not in connectors or target not in connectors or source == target:
                add(f"$.combined_continuity.relationships[{index}]", "invalid_cross_sheet_continuity", "cross-sheet continuity must connect two distinct exported connectors", "Retain unresolved connector matches as issues without continuity")
            if _relationship_state(relation) not in RESOLVED_STATES:
                add(f"$.combined_continuity.relationships[{index}]", "unresolved_cross_sheet_continuity", "unresolved connector matches must not be promoted to continuity", "Keep the connector unresolved and omit this relationship")
            if pair in seen_pairs:
                add(f"$.combined_continuity.relationships[{index}]", "duplicate_cross_sheet_continuity", "each resolved connector pair may have only one continuity relationship", "Deduplicate the connector pair")
            seen_pairs.add(pair)
    return {"valid": not any(item.get("severity") == "error" for item in issues), "issues": issues}


def serialize_downstream_export(payload: dict[str, Any]) -> str:
    """Serialize after validation, with strict JSON settings."""
    result = validate_downstream_export(payload)
    if not result["valid"]:
        raise ValueError("invalid downstream export: " + "; ".join(item["message"] for item in result["issues"]))
    return _canonical_json(payload)


# Friendly aliases for callers that use the roadmap vocabulary.
build_versioned_export = build_downstream_export
validate_versioned_export = validate_downstream_export


__all__ = [
    "SCHEMA_VERSION",
    "VERSIONED_EXPORT_SCHEMA_VERSION",
    "DOWNSTREAM_EXPORT_SCHEMA_VERSION",
    "EXPORT_KIND",
    "EXPORT_SCHEMA",
    "build_downstream_export",
    "build_versioned_export",
    "graph_content_sha256",
    "release_gate_sha256",
    "release_gate_content_sha256",
    "serialize_downstream_export",
    "validate_downstream_export",
    "validate_versioned_export",
    "build_blocked_export",
]
