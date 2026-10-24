"""Release-gated process knowledge projection for GARNET graphs.

Consumes the released downstream envelope (``garnet_downstream_export_v1``) and
produces the layers a process-knowledge consumer needs, kept explicitly apart
so neither query style corrupts the other:

    physical_topology    undirected equipment/port/route graph. Process flow is
                         carried as evidence on each route and is never turned
                         into an edge direction.
    process_knowledge    HAZOP study nodes projected from released topology,
                         each carrying the line properties guide-word reasoning
                         depends on.
    deviation_graph      cause -> deviation -> consequence -> safeguard, linked
                         to study nodes by id. Empty until a workshop fills it.

Nothing leaves this module unless the upstream release gate already said
released. A blocked envelope yields a redacted, empty stack rather than a
partial one, so a stale graph can never be mistaken for a released one.

    .venv/bin/python -m garnet.knowledge_graph stage10_final_export.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "garnet_process_knowledge_v1"
JSONLD_CONTEXT = "https://garnet.local/schema/process-knowledge/v1.jsonld"

# Export relationship types that carry a real graph edge.
RELATIONSHIP_TYPES = {
    "has_port",
    "has_inline_object",
    "measures",
    "controls",
    "actuates",
    "instrument_association",
    "edge_to_equipment",
    "cross_sheet_continues",
    "branches_from",
}

# Line facts copied onto a study node. Missing facts stay missing.
LINE_PROPERTY_KEYS = (
    "line_number",
    "size",
    "spec",
    "service",
    "fluid",
    "insulation",
    "design_pressure",
    "design_temperature",
    "operating_pressure",
    "operating_temperature",
    "material",
)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _record_id(record: dict[str, Any]) -> str:
    for key in (
        "id", "node_id", "route_id", "relationship_id", "line_id",
        "equipment_id", "instrument_id", "port_id", "occurrence_id",
    ):
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _graph(payload: dict[str, Any]) -> dict[str, Any]:
    graph = payload.get("graph")
    return graph if isinstance(graph, dict) else {}


def _sort_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)


def release_state(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (released, reasons). Both the flag and the gate must agree."""
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    gate = review.get("release_gate") if isinstance(review.get("release_gate"), dict) else None
    reasons = [_text(item) for item in payload.get("blocked_reasons") or [] if _text(item)]
    if payload.get("release_ready") is True and isinstance(gate, dict) and gate.get("release_ready") is True:
        return True, []
    return False, sorted(set(reasons or ["release_gate_not_ready"]))


def content_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _route_pairs(payload: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(source, target, route) using physical endpoints only."""
    pairs: list[tuple[str, str, dict[str, Any]]] = []
    graph = _graph(payload)
    for route in _records(graph.get("routes")) + _records(graph.get("edges")):
        physical = route.get("physical") if isinstance(route.get("physical"), dict) else route
        source = _text(physical.get("source"))
        target = _text(physical.get("target"))
        if source and target:
            pairs.append((source, target, route))
    return pairs


def _port_owners(payload: dict[str, Any]) -> dict[str, str]:
    """Port id -> equipment id, from the catalog and from has_port relationships."""
    owners: dict[str, str] = {}
    for port in _records(_graph(payload).get("ports")):
        port_id = _record_id(port)
        equipment_id = _text(port.get("equipment_id") or port.get("owner_id"))
        if port_id and equipment_id:
            owners.setdefault(port_id, equipment_id)
    for relationship in _records(_graph(payload).get("relationships")):
        if _text(relationship.get("type")) != "has_port":
            continue
        port_id = _text(relationship.get("target"))
        equipment_id = _text(relationship.get("source"))
        if port_id and equipment_id:
            owners.setdefault(port_id, equipment_id)
    return owners


def _endpoint_equipment(payload: dict[str, Any], owners: dict[str, str]) -> dict[str, str]:
    """Resolve every route endpoint to its equipment identity."""
    nodes = {_record_id(node): node for node in _records(_graph(payload).get("nodes")) if _record_id(node)}
    resolved: dict[str, str] = {}
    for source, target, _ in _route_pairs(payload):
        for endpoint in (source, target):
            if endpoint in resolved:
                continue
            if endpoint in owners:
                resolved[endpoint] = owners[endpoint]
                continue
            node = nodes.get(endpoint, {})
            resolved[endpoint] = (
                _text(node.get("equipment_id") or node.get("owner_id") or node.get("parent_id")) or endpoint
            )
    return resolved


def _route_lines(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Route id -> line ids, from explicit route facts only."""
    attached: dict[str, list[str]] = defaultdict(list)
    graph = _graph(payload)
    for route in _records(graph.get("routes")) + _records(graph.get("edges")):
        route_id = _record_id(route)
        if not route_id:
            continue
        candidates = (
            route.get("canonical_line_ids")
            or route.get("line_ids")
            or route.get("effective_line_number_ids")
            or []
        )
        if not isinstance(candidates, list):
            candidates = [candidates]
        for value in candidates:
            line_id = _text(value)
            if line_id and line_id not in attached[route_id]:
                attached[route_id].append(line_id)
    return attached


def _line_properties(line: dict[str, Any]) -> dict[str, Any]:
    properties = {key: line[key] for key in LINE_PROPERTY_KEYS if line.get(key) not in (None, "", [], {})}
    if "line_number" not in properties:
        for occurrence in _records(line.get("occurrences")):
            text = _text(occurrence.get("normalized_text") or occurrence.get("text"))
            if text:
                properties["line_number"] = text
                break
    return properties


def build_physical_topology(payload: dict[str, Any]) -> dict[str, Any]:
    """The undirected physical layer, with flow kept as evidence."""
    graph = _graph(payload)
    owners = _port_owners(payload)
    endpoints = _endpoint_equipment(payload, owners)
    route_lines = _route_lines(payload)

    edges: list[dict[str, Any]] = []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target, route in _route_pairs(payload):
        route_id = _record_id(route) or f"route::{source}::{target}"
        flow = route.get("flow") if isinstance(route.get("flow"), dict) else {}
        left = endpoints.get(source, source)
        right = endpoints.get(target, target)
        edges.append({
            "id": route_id,
            "type": "connects_to",
            "undirected": True,
            "source": source,
            "target": target,
            "source_equipment_id": left,
            "target_equipment_id": right,
            "line_ids": route_lines.get(route_id, []),
            "flow_state": _text(flow.get("state") or route.get("flow_direction_state")) or "unknown",
            "flow_source": _text(flow.get("source")) or None,
            "flow_target": _text(flow.get("target")) or None,
            "confidence": route.get("confidence"),
        })
        adjacency[left].add(right)
        adjacency[right].add(left)

    equipment = _records(graph.get("equipment")) or [
        {"id": equipment_id, "label": "Equipment"} for equipment_id in sorted(adjacency, key=_sort_key)
    ]
    lines = [
        {"id": _record_id(line), "label": "PipingLine", "process_properties": _line_properties(line)}
        for line in _records(graph.get("lines"))
        if _record_id(line)
    ]
    return {
        "layer": "physical_topology",
        "directed": False,
        "flow_direction_separate": True,
        "equipment": equipment,
        "lines": lines,
        "ports": _records(graph.get("ports")),
        "edges": edges,
        "adjacency": {key: sorted(values, key=_sort_key) for key, values in sorted(adjacency.items(), key=lambda kv: _sort_key(kv[0]))},
    }


def build_process_knowledge(payload: dict[str, Any]) -> dict[str, Any]:
    """Project HAZOP study nodes from released physical topology."""
    topology = build_physical_topology(payload)
    equipment_meta = {_record_id(item): item for item in topology["equipment"] if _record_id(item)}
    line_properties = {item["id"]: item["process_properties"] for item in topology["lines"]}

    neighbours: dict[str, set[str]] = defaultdict(set)
    equipment_routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    instrument_ids: dict[str, list[str]] = defaultdict(list)
    for edge in topology["edges"]:
        left = edge["source_equipment_id"] or edge["source"]
        right = edge["target_equipment_id"] or edge["target"]
        neighbours[left].add(right)
        neighbours[right].add(left)
        equipment_routes[left].append(edge)
        equipment_routes[right].append(edge)
    for relationship in _records(_graph(payload).get("relationships")):
        kind = _text(relationship.get("type"))
        if kind not in {"measures", "controls", "actuates", "instrument_association"}:
            continue
        target = _text(relationship.get("target"))
        for endpoint in (_text(relationship.get("source")), _text(relationship.get("target"))):
            if endpoint in equipment_routes and target:
                instrument_ids[endpoint].append(target)

    study_nodes: list[dict[str, Any]] = []
    for equipment_id in sorted(equipment_routes, key=_sort_key):
        routes = equipment_routes[equipment_id]
        line_ids: list[str] = []
        for edge in routes:
            for line_id in edge["line_ids"]:
                if line_id not in line_ids:
                    line_ids.append(line_id)
        properties: dict[str, Any] = {}
        for line_id in line_ids:
            for key, value in line_properties.get(line_id, {}).items():
                properties.setdefault(key, value)
        study_nodes.append({
            "id": f"studynode::{equipment_id}",
            "label": _text(equipment_meta.get(equipment_id, {}).get("tag")) or equipment_id,
            "kind": "equipment",
            "equipment_ids": [equipment_id],
            "line_ids": line_ids,
            "instrument_ids": sorted(set(instrument_ids.get(equipment_id, [])), key=_sort_key),
            "route_ids": [edge["id"] for edge in routes],
            "upstream_node_ids": [],
            "downstream_node_ids": sorted(neighbours[equipment_id], key=_sort_key),
            "evidence": [{"kind": "route", "id": edge["id"]} for edge in routes],
            "uncertainty": [],
            "process_properties": properties,
        })
    return {"layer": "process_knowledge", "study_nodes": study_nodes}


def build_deviation_graph(deviations: list[dict[str, Any]] | None, study_node_ids: set[str]) -> dict[str, Any]:
    """HAZOP worksheet as a graph. Unknown study-node references are reported."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    issues: list[str] = []
    for deviation in _records(deviations):
        deviation_id = _text(deviation.get("id"))
        if not deviation_id:
            issues.append("deviation_missing_id")
            continue
        study_node_id = _text(deviation.get("study_node_id"))
        if study_node_id not in study_node_ids:
            issues.append(f"unknown_study_node:{study_node_id or '<empty>'}")
            continue
        nodes.append({
            "id": deviation_id,
            "label": "Deviation",
            "guide_word": _text(deviation.get("guide_word")) or None,
            "parameter": _text(deviation.get("parameter")) or None,
            "cause": _text(deviation.get("cause")) or None,
            "consequence": _text(deviation.get("consequence")) or None,
            "safeguard": _text(deviation.get("safeguard")) or None,
            "risk": deviation.get("risk"),
        })
        edges.append({
            "id": f"applies_to::{deviation_id}::{study_node_id}",
            "type": "applies_to",
            "source": deviation_id,
            "target": study_node_id,
        })
        for safeguard_id in deviation.get("safeguard_ids") or []:
            edges.append({
                "id": f"protects::{deviation_id}::{_text(safeguard_id)}",
                "type": "protects",
                "source": _text(safeguard_id),
                "target": deviation_id,
            })
    return {"layer": "deviation_graph", "nodes": nodes, "edges": edges, "issues": issues}


def build_jsonld(topology: dict[str, Any], knowledge: dict[str, Any], content_hash: str) -> dict[str, Any]:
    """Graph-shaped JSON-LD: one wire format for storage, diffing and loading."""
    nodes: list[dict[str, Any]] = []
    for equipment in topology["equipment"]:
        equipment_id = _record_id(equipment)
        nodes.append({
            "@id": f"garnet:equipment/{equipment_id}",
            "@type": "Equipment",
            "tag": equipment.get("tag") or equipment.get("label"),
        })
    for line in topology["lines"]:
        nodes.append({
            "@id": f"garnet:line/{line['id']}",
            "@type": "PipingLine",
            **line["process_properties"],
        })
    for route in topology["edges"]:
        nodes.append({
            "@id": f"garnet:route/{route['id']}",
            "@type": "Route",
            "connectsTo": [{"@id": f"garnet:port/{route['source']}"}, {"@id": f"garnet:port/{route['target']}"}],
            "carriesLine": [{"@id": f"garnet:line/{line_id}"} for line_id in route["line_ids"]],
            "flowState": route["flow_state"],
            "flowDirectionEvidenceOnly": True,
        })
    for study_node in knowledge["study_nodes"]:
        nodes.append({
            "@id": f"garnet:{study_node['id']}",
            "@type": "StudyNode",
            "label": study_node["label"],
            "includesEquipment": [{"@id": f"garnet:equipment/{item}"} for item in study_node["equipment_ids"]],
            "includesLine": [{"@id": f"garnet:line/{item}"} for item in study_node["line_ids"]],
            "connectsTo": [{"@id": f"garnet:studynode::{item}"} for item in study_node["downstream_node_ids"]],
        })
    return {
        "@context": JSONLD_CONTEXT,
        "@graph": nodes,
        "source": {"graph_content_sha256": content_hash, "schema_version": SCHEMA_VERSION},
    }


def build_knowledge_stack(
    payload: Any, *, deviations: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build the release-gated knowledge stack from a downstream export envelope."""
    if not isinstance(payload, dict):
        raise ValueError("knowledge stack input must be the downstream export envelope")
    if _text(payload.get("schema_version")) != "garnet_downstream_export_v1":
        raise ValueError("knowledge stack requires garnet_downstream_export_v1")

    released, reasons = release_state(payload)
    content_hash = content_sha256(payload)
    review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
    source = {
        "downstream_export_schema_version": payload.get("schema_version"),
        "downstream_export_content_sha256": content_hash,
        "graph_content_sha256": (payload.get("source") or {}).get("graph_content_sha256"),
        "release_gate_sha256": review.get("release_gate_sha256"),
        "drawing_ids": [_text(item.get("drawing_id")) for item in _records(payload.get("drawings")) if _text(item.get("drawing_id"))],
    }

    if not released:
        return {
            "schema_version": SCHEMA_VERSION,
            "layer": "release_gate",
            "release_ready": False,
            "redacted": True,
            "blocked_reasons": reasons,
            "source": source,
            "physical_topology": {"layer": "physical_topology", "equipment": [], "lines": [], "ports": [], "edges": [], "adjacency": {}},
            "process_knowledge": {"layer": "process_knowledge", "study_nodes": []},
            "deviation_graph": {"layer": "deviation_graph", "nodes": [], "edges": [], "issues": []},
            "jsonld": {"@context": JSONLD_CONTEXT, "@graph": []},
            "issues": [],
        }

    topology = build_physical_topology(payload)
    knowledge = build_process_knowledge(payload)
    study_node_ids = {node["id"] for node in knowledge["study_nodes"]}
    deviation_graph = build_deviation_graph(deviations, study_node_ids)

    issues = list(deviation_graph["issues"])
    for edge in topology["edges"]:
        if not edge["line_ids"]:
            issues.append(f"route_without_line:{edge['id']}")
        if edge["flow_state"] == "unknown":
            issues.append(f"unknown_flow_direction:{edge['id']}")

    source["counts"] = {
        "equipment": len(topology["equipment"]),
        "lines": len(topology["lines"]),
        "routes": len(topology["edges"]),
        "study_nodes": len(knowledge["study_nodes"]),
        "deviations": len(deviation_graph["nodes"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "knowledge_stack",
        "release_ready": True,
        "redacted": False,
        "blocked_reasons": [],
        "source": source,
        "physical_topology": topology,
        "process_knowledge": knowledge,
        "deviation_graph": deviation_graph,
        "jsonld": build_jsonld(topology, knowledge, content_hash),
        "issues": sorted(set(issues)),
    }


def serialize_knowledge_stack(stack: dict[str, Any]) -> str:
    """Strict, deterministic JSON for storage or handoff."""
    return json.dumps(stack, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def validate_knowledge_stack(stack: Any, *, source_export: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``{valid, issues}``; every issue carries path, code and message."""
    issues: list[dict[str, Any]] = []

    def add(path: str, code: str, message: str, suggestion: str | None = None) -> None:
        item = {"path": path, "code": code, "message": message, "severity": "error"}
        if suggestion:
            item["suggestion"] = suggestion
        issues.append(item)

    if not isinstance(stack, dict):
        add("$", "wrong_type", "knowledge stack must be an object")
        return {"valid": False, "issues": issues}
    if stack.get("schema_version") != SCHEMA_VERSION:
        add("$.schema_version", "unsupported_schema_version", f"expected {SCHEMA_VERSION!r}")
        return {"valid": False, "issues": issues}
    try:
        serialize_knowledge_stack(stack)
    except (TypeError, ValueError) as exc:
        add("$", "non_json_value", str(exc))
        return {"valid": False, "issues": issues}

    released = stack.get("release_ready") is True
    if released and stack.get("redacted") is True:
        add("$", "inconsistent_release_state", "a released stack must not be redacted")
    if not released and stack.get("redacted") is not True:
        add("$.redacted", "inconsistent_release_state", "a blocked stack must be redacted")

    if source_export is not None:
        declared = _text((stack.get("source") or {}).get("downstream_export_content_sha256"))
        if declared != content_sha256(source_export):
            add("$.source.downstream_export_content_sha256", "hash_mismatch",
                "stack is not bound to this downstream export", "Rebuild from the current export")

    topology = stack.get("physical_topology") or {}
    equipment_ids = {_record_id(item) for item in _records(topology.get("equipment"))}
    route_ids = {_record_id(item) for item in _records(topology.get("edges"))}
    line_ids = {_record_id(item) for item in _records(topology.get("lines"))}
    for index, edge in enumerate(_records(topology.get("edges"))):
        path = f"$.physical_topology.edges[{index}]"
        if edge.get("undirected") is not True:
            add(path, "directed_physical_edge", "physical connectivity must stay undirected")
        for line_id in edge.get("line_ids") or []:
            if line_id not in line_ids:
                add(path, "dangling_line_reference", f"unknown line id {line_id!r}")

    study_ids: set[str] = set()
    for index, node in enumerate(_records((stack.get("process_knowledge") or {}).get("study_nodes"))):
        path = f"$.process_knowledge.study_nodes[{index}]"
        study_ids.add(_text(node.get("id")))
        for route_id in node.get("route_ids") or []:
            if route_id not in route_ids:
                add(path, "dangling_route_reference", f"unknown route id {route_id!r}")
        for equipment_id in node.get("equipment_ids") or []:
            if equipment_ids and equipment_id not in equipment_ids:
                add(path, "dangling_equipment_reference", f"unknown equipment id {equipment_id!r}")

    deviation_graph = stack.get("deviation_graph") or {}
    for index, node in enumerate(_records(deviation_graph.get("nodes"))):
        for edge in _records(deviation_graph.get("edges")):
            if edge.get("type") == "applies_to" and _text(edge.get("source")) == _text(node.get("id")):
                if _text(edge.get("target")) not in study_ids:
                    add(f"$.deviation_graph.nodes[{index}]", "dangling_study_node_reference",
                        f"deviation targets unknown study node {edge.get('target')!r}")

    jsonld = stack.get("jsonld")
    if not isinstance(jsonld, dict) or not isinstance(jsonld.get("@graph"), list):
        add("$.jsonld", "missing_jsonld_graph", "a released stack must carry a JSON-LD @graph")
    elif released and not jsonld["@graph"]:
        add("$.jsonld.@graph", "empty_jsonld_graph", "a released stack must not serialize an empty graph")

    return {"valid": not issues, "issues": issues}


def main(argv: list[str] | None = None) -> int:
    """Project one or more released exports to knowledge stacks."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__.strip().splitlines()[-1])
        print("usage: python -m garnet.knowledge_graph <export.json> [...] [--out DIR]")
        return 0 if args else 2

    out_dir = None
    if "--out" in args:
        index = args.index("--out")
        try:
            out_dir = Path(args[index + 1])
        except IndexError:
            print("--out needs a directory", file=sys.stderr)
            return 2
        del args[index:index + 2]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in args:
        path = Path(name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stack = build_knowledge_stack(payload)
        except (OSError, ValueError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        report = validate_knowledge_stack(stack, source_export=payload)
        target = (out_dir / f"{path.stem}_knowledge_stack.json") if out_dir else None
        if target is not None:
            target.write_text(serialize_knowledge_stack(stack), encoding="utf-8")
        state = "released" if stack["release_ready"] else "blocked"
        print(
            f"{path}: {state} valid={report['valid']} "
            f"study_nodes={len(stack['process_knowledge']['study_nodes'])} "
            f"deviations={len(stack['deviation_graph']['nodes'])}"
            + (f" -> {target}" if target else "")
        )
        for issue in report["issues"]:
            print(f"  {issue['path']}: {issue['code']}: {issue['message']}", file=sys.stderr)
        failures += 0 if report["valid"] else 1
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "JSONLD_CONTEXT",
    "build_deviation_graph",
    "build_jsonld",
    "build_knowledge_stack",
    "build_physical_topology",
    "build_process_knowledge",
    "content_sha256",
    "release_state",
    "serialize_knowledge_stack",
    "validate_knowledge_stack",
]
