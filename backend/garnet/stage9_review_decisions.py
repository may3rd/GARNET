from __future__ import annotations

import copy
from collections import Counter
from typing import Any

_NOOP_DECISIONS = {"accept_as_is", "false_positive", "defer"}
_TOPOLOGY_DECISIONS = {"merge_nodes", "reconnect_edge", "split_edge", "delete_edge", "set_node_type"}
_FLOW_DIRECTION_DECISION_STATES = {"forward", "reverse", "bidirectional", "unknown"}
_OFF_PAGE_EXIT_TERMINALS = {"source", "src", "destination", "target", "dst", "terminal"}
_RELEASE_BLOCKING_WARNINGS = {
    "unknown_review_item_id",
    "duplicate_decision",
    "missing_review_item_id",
    "invalid_off_page_connector",
}


def _decision_index(decisions_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decisions_by_id: dict[str, dict[str, Any]] = {}
    for decision in decisions_payload.get("decisions", []) or []:
        if not isinstance(decision, dict):
            continue
        review_item_id = str(decision.get("review_item_id") or "").strip()
        if review_item_id:
            decisions_by_id[review_item_id] = decision
    return decisions_by_id


def _resolution_for_item(item: dict[str, Any], decision: dict[str, Any] | None) -> dict[str, Any]:
    review_item_id = item.get("id")
    category = item.get("category")
    if decision is None:
        return {
            "review_item_id": review_item_id,
            "category": category,
            "resolution_state": "unresolved",
            "decision_source": "stage9_review_required",
            "graph_changed": False,
        }

    decision_name = str(decision.get("decision") or "")
    reviewer = str(decision.get("reviewer") or "unspecified")
    if decision_name == "set_line_number":
        resolution_state = "set_line_number"
    elif decision_name == "set_flow_direction":
        resolution_state = "set_flow_direction"
    elif decision_name in _NOOP_DECISIONS:
        resolution_state = "unresolved" if decision_name == "defer" else decision_name
    elif decision_name in _TOPOLOGY_DECISIONS:
        resolution_state = "pending"
    else:
        resolution_state = "unsupported_decision"
    resolution = {
        "review_item_id": review_item_id,
        "category": category,
        "resolution_state": resolution_state,
        "decision": decision_name,
        "decision_source": reviewer,
        "graph_changed": False,
    }
    if decision.get("note"):
        resolution["note"] = decision["note"]
    return resolution


def _edges_by_id(graph_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(edge.get("id")): edge
        for edge in graph_payload.get("edges", []) or []
        if isinstance(edge, dict) and str(edge.get("id") or "")
    }


def _line_record_catalog(graph_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for edge in graph_payload.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        records = (edge.get("line_numbers") or []) + (edge.get("effective_line_numbers") or [])
        for record in records:
            if not isinstance(record, dict):
                continue
            key = str(record.get("id") or record.get("source_object_id") or "")
            if key:
                catalog.setdefault(key, copy.deepcopy(record))
    return catalog


def _flow_direction_state_from_decision(decision: dict[str, Any]) -> str:
    """Read the canonical state while accepting the field aliases used by clients."""
    for key in ("flow_direction_state", "flow_direction", "direction_state", "direction", "state"):
        value = decision.get(key)
        if value is not None:
            return str(value).strip().lower().replace("-", "_")
    return ""


def _flow_direction_evidence(edge: dict[str, Any]) -> list[Any]:
    evidence = edge.get("flow_direction_evidence")
    if not evidence:
        evidence = edge.get("direction_evidence")
    if isinstance(evidence, list):
        return copy.deepcopy(evidence)
    if evidence is None:
        return []
    return [copy.deepcopy(evidence)]


def _flow_direction_provenance(edge: dict[str, Any]) -> list[Any]:
    provenance = edge.get("flow_direction_provenance")
    if isinstance(provenance, list):
        return copy.deepcopy(provenance)
    if provenance is None:
        return []
    return [copy.deepcopy(provenance)]


def _flow_edge_ids(decision: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    if decision.get("edge_ids") is not None:
        edge_values = decision.get("edge_ids")
        if isinstance(edge_values, (list, tuple, set)):
            values.extend(edge_values)
        else:
            values.append(edge_values)
    elif decision.get("edge_id") is not None:
        values.append(decision.get("edge_id"))
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _apply_set_flow_direction(
    *,
    corrected_graph_payload: dict[str, Any],
    decision: dict[str, Any],
    review_item_id: str,
    image_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Apply an explicit human direction override atomically across edge IDs."""
    state = _flow_direction_state_from_decision(decision)
    edge_ids = _flow_edge_ids(decision)
    warnings: list[dict[str, Any]] = []
    if state not in _FLOW_DIRECTION_DECISION_STATES:
        warnings.append(
            {
                "review_item_id": review_item_id,
                "warning": "invalid_flow_direction",
                "provided_state": state or None,
                "message": "set_flow_direction accepts only forward, reverse, bidirectional, or unknown.",
            }
        )
        return None, warnings

    edges = _edges_by_id(corrected_graph_payload)
    missing_edge_ids = [edge_id for edge_id in edge_ids if edge_id not in edges]
    if not edge_ids:
        warnings.append(
            {
                "review_item_id": review_item_id,
                "warning": "missing_edge",
                "message": "set_flow_direction decision did not reference an edge in the corrected graph.",
            }
        )
    elif missing_edge_ids:
        # Direction overrides are atomic: a stale review item must not produce
        # a partially updated graph when one of a batch of edge IDs vanished.
        for edge_id in missing_edge_ids:
            warnings.append(
                {
                    "review_item_id": review_item_id,
                    "warning": "missing_edge",
                    "edge_id": edge_id,
                    "message": "set_flow_direction decision referenced an edge that is not in the corrected graph.",
                }
            )
    if not edge_ids or missing_edge_ids:
        return None, warnings

    reviewer = str(decision.get("reviewer") or "human_review")
    note = decision.get("note")
    prior_states: dict[str, Any] = {}
    prior_evidence: dict[str, list[Any]] = {}
    for edge_id in edge_ids:
        edge = edges[edge_id]
        prior_states[edge_id] = copy.deepcopy(edge.get("flow_direction_state", edge.get("flow_direction")))
        prior_evidence[edge_id] = _flow_direction_evidence(edge)
        observed_evidence = edge.get("observed_flow_direction_evidence")
        if not isinstance(observed_evidence, list):
            observed_evidence = prior_evidence[edge_id]
        edge["observed_flow_direction_evidence"] = copy.deepcopy(observed_evidence)
        edge["flow_direction_state"] = state
        edge["flow_direction_confidence"] = 1.0
        edge["flow_direction_confidence_source"] = "human_review"
        # Keep the original detector/geometry evidence intact and record the
        # correction separately as provenance rather than relabeling it as an
        # observation.
        edge["flow_direction_evidence"] = copy.deepcopy(prior_evidence[edge_id])
        edge["direction_evidence"] = copy.deepcopy(prior_evidence[edge_id])
        review_evidence = {
            "source": "human_review",
            "state": state,
            "review_item_id": review_item_id,
            "reviewer": reviewer,
        }
        edge["flow_direction_review_evidence"] = review_evidence
        provenance = _flow_direction_provenance(edge)
        provenance.append(
            {
                **review_evidence,
                "source": "stage9_review_decisions",
                "state": "human_reviewed",
                "flow_direction_state": state,
            }
        )
        edge["flow_direction_provenance"] = provenance
        edge["flow_direction_review_state"] = "human_reviewed"

    correction: dict[str, Any] = {
        "id": f"correction::set_flow_direction::{review_item_id}",
        "image_id": image_id,
        "review_item_id": review_item_id,
        "decision": "set_flow_direction",
        "flow_direction_state": state,
        "flow_direction": state,
        "affected_edge_ids": edge_ids,
        "prior_flow_direction_states": prior_states,
        "prior_flow_direction_evidence": prior_evidence,
        "reviewer": decision.get("reviewer"),
    }
    if note:
        correction["note"] = note
    return correction, warnings


def _apply_set_line_number(
    *,
    corrected_graph_payload: dict[str, Any],
    decision: dict[str, Any],
    review_item_id: str,
    image_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    line_number_id = str(decision.get("line_number_id") or "")
    edge_ids = [str(edge_id) for edge_id in decision.get("edge_ids", []) or [] if str(edge_id)]
    edges = _edges_by_id(corrected_graph_payload)
    line_catalog = _line_record_catalog(corrected_graph_payload)
    affected_edge_ids: list[str] = []
    warnings: list[dict[str, Any]] = []

    if not line_number_id:
        warnings.append(
            {
                "review_item_id": review_item_id,
                "warning": "missing_line_number_id",
                "message": "set_line_number decision did not include line_number_id.",
            }
        )
        return None, warnings

    missing_edge_ids = [edge_id for edge_id in edge_ids if edge_id not in edges]
    if missing_edge_ids:
        warnings.extend({"review_item_id": review_item_id, "warning": "missing_edge", "edge_id": edge_id,
                         "message": "set_line_number decision referenced an edge that is not in the corrected graph."}
                        for edge_id in missing_edge_ids)
        return None, warnings

    for edge_id in edge_ids:
        edge = edges.get(edge_id)
        if edge is None:
            warnings.append(
                {
                    "review_item_id": review_item_id,
                    "warning": "missing_edge",
                    "edge_id": edge_id,
                    "message": "set_line_number decision referenced an edge that is not in the corrected graph.",
                }
            )
            continue
        # Keep the effective id, direct records, and effective records in
        # lockstep.  Stage 10 and graph-v1 consume different views of these
        # fields, so updating only the id leaves stale display text behind.
        attachments = edge.setdefault("attachments", {})
        if not isinstance(attachments, dict):
            attachments = {}
            edge["attachments"] = attachments
        source_records = attachments.get("line_numbers")
        if not isinstance(source_records, list):
            source_records = []
        top_records = edge.get("line_numbers")
        if not isinstance(top_records, list):
            top_records = []
        prior_observed = attachments.get("observed_line_numbers")
        if isinstance(prior_observed, list):
            observed_source = prior_observed
        else:
            observed_source = list(source_records) + list(top_records)
        observed_records = []
        observed_keys: set[str] = set()
        for record in observed_source:
            if not isinstance(record, dict):
                continue
            key = str(record.get("id") or record.get("source_object_id") or repr(record))
            if key not in observed_keys:
                observed_keys.add(key)
                observed_records.append(copy.deepcopy(record))
        records = [record for record in source_records if isinstance(record, dict)]
        selected_record = next(
            (
                record
                for record in records
                if str(record.get("id") or record.get("source_object_id") or "") == line_number_id
            ),
            None,
        )
        if selected_record is None:
            selected_record = line_catalog.get(line_number_id)
        if selected_record is None:
            selected_record = {
                "id": line_number_id,
                "source_object_id": None,
                "text": str(decision.get("line_number_text") or line_number_id),
                "normalized_text": str(decision.get("line_number_text") or line_number_id),
            }
            records.append(selected_record)
        selected_record["review_state"] = "accepted"
        selected_record["review_source"] = str(decision.get("reviewer") or "human_review")
        # Preserve all OCR/association evidence under an explicit observed
        # collection, while making the effective/direct view unambiguous for
        # Stage 10's line selection logic.
        attachments["observed_line_numbers"] = observed_records
        attachments["line_numbers"] = [selected_record]
        edge["line_numbers"] = [selected_record]
        edge["line_number_ids"] = [line_number_id]
        edge["effective_line_numbers"] = [selected_record]
        edge["effective_line_number_ids"] = [line_number_id]
        edge["direct_line_number_ids"] = [line_number_id]
        edge["inferred_line_number_ids"] = []
        edge["direct_line_numbers"] = [selected_record]
        edge["inferred_line_numbers"] = []
        edge["line_number_assignment_state"] = "human_reviewed"
        edge["reviewed_line_number_id"] = line_number_id
        edge["line_number_review_state"] = "human_reviewed"
        affected_edge_ids.append(edge_id)

    if not affected_edge_ids:
        return None, warnings

    line_to_edges: dict[str, list[str]] = {}
    for edge in corrected_graph_payload.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        for value in edge.get("effective_line_number_ids") or edge.get("line_number_ids") or []:
            key = str(value)
            if key:
                line_to_edges.setdefault(key, []).append(str(edge.get("id") or ""))
    corrected_graph_payload["line_to_edges"] = {
        key: sorted(edge_ids) for key, edge_ids in sorted(line_to_edges.items()) if edge_ids
    }
    corrected_graph_payload["line_groups"] = [
        {"line_number_id": key, "edge_ids": value}
        for key, value in corrected_graph_payload["line_to_edges"].items()
    ]

    correction = {
        "id": f"correction::set_line_number::{review_item_id}",
        "image_id": image_id,
        "review_item_id": review_item_id,
        "decision": "set_line_number",
        "line_number_id": line_number_id,
        "affected_edge_ids": affected_edge_ids,
        "reviewer": decision.get("reviewer"),
    }
    return correction, warnings


def _topology_ids(decision: dict[str, Any], key: str) -> list[str]:
    value = decision.get(key)
    if value is None:
        value = decision.get(f"{key}s")
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if value is not None and str(value) else []


_SOURCE_ENDPOINT_METADATA = {
    "source_obj_id",
    "source_obj_type",
    "source_port_index",
    "source_port_id",
    "source_port_xy",
    "source_equipment_id",
    "source_node_id",
    "source_node_type",
}
_TARGET_ENDPOINT_METADATA = {
    "terminal_obj_id",
    "terminal_type",
    "terminal_port_index",
    "terminal_port_id",
    "terminal_port_xy",
    "terminal_equipment_id",
    "terminal_node_id",
    "terminal_node_type",
    "target_obj_id",
    "target_obj_type",
    "target_port_index",
    "target_port_id",
    "target_port_xy",
    "target_equipment_id",
    "target_node_id",
    "target_node_type",
}


def _endpoint_ids(edge: dict[str, Any]) -> tuple[str, str]:
    """Read endpoint aliases without allowing a present null alias to win."""
    source = edge.get("src") if edge.get("src") not in (None, "") else edge.get("source")
    target = edge.get("dst") if edge.get("dst") not in (None, "") else edge.get("target")
    return str(source or ""), str(target or "")


def _set_endpoint_ids(edge: dict[str, Any], source: str, target: str) -> None:
    """Keep the graph's legacy and canonical endpoint aliases in lockstep."""
    edge["src"], edge["source"] = source, source
    edge["dst"], edge["target"] = target, target


def _clear_endpoint_metadata(edge: dict[str, Any], endpoint: str) -> None:
    """Remove identity evidence that belonged to a replaced graph endpoint."""
    fields = _SOURCE_ENDPOINT_METADATA if endpoint == "source" else _TARGET_ENDPOINT_METADATA
    for key in fields:
        if key in edge:
            edge[key] = None


def _set_endpoint_node_metadata(edge: dict[str, Any], endpoint: str, node_id: str) -> None:
    key = "source_node_id" if endpoint == "source" else "target_node_id"
    if key in edge:
        edge[key] = node_id
    if endpoint == "target" and "terminal_node_id" in edge:
        edge["terminal_node_id"] = node_id


def _copy_off_page_connector_for_split(
    edge: dict[str, Any], left: dict[str, Any], right: dict[str, Any]
) -> None:
    connector = edge.get("off_page_connector")
    if not isinstance(connector, dict):
        return
    exit_terminal = str(connector.get("exit_terminal") or "").strip().lower()
    if exit_terminal in {"source", "src"}:
        right.pop("off_page_connector", None)
    elif exit_terminal in _OFF_PAGE_EXIT_TERMINALS - {"source", "src"}:
        left.pop("off_page_connector", None)
    else:
        # A connector without a declared exit cannot be assigned safely to a
        # child after a split.
        left.pop("off_page_connector", None)
        right.pop("off_page_connector", None)


def _off_page_connector_warning(
    edge: dict[str, Any], item_id: str,
) -> dict[str, Any] | None:
    """Reject connector metadata that cannot be assigned to an edge terminal."""
    if "off_page_connector" not in edge:
        return None
    connector = edge.get("off_page_connector")
    exit_terminal = str(connector.get("exit_terminal") or "").strip().lower() if isinstance(connector, dict) else ""
    if exit_terminal in _OFF_PAGE_EXIT_TERMINALS:
        return None
    return {
        "warning": "invalid_off_page_connector",
        "review_item_id": item_id,
        "edge_id": str(edge.get("id") or ""),
        "message": "off_page_connector must be an object with a valid exit_terminal.",
    }


def _unique_split_trace_ids(
    edge: dict[str, Any],
    existing_edges: list[dict[str, Any]],
) -> tuple[str, str, str]:
    """Return unique child trace IDs and the root trace ID for a split.

    A Stage 9 decision can split an edge that was itself created by an earlier
    split.  Reusing ``original_trace_id`` as the child ID prefix would then
    produce duplicate ``::part_000``/``::part_001`` IDs.  Use the current edge
    trace as the prefix and retain the root trace separately for provenance.
    """
    current_trace_id = str(edge.get("trace_id") or edge.get("original_trace_id") or edge.get("id") or "trace")
    root_trace_id = str(edge.get("original_trace_id") or edge.get("trace_id") or edge.get("id") or "trace")
    used = {
        str(candidate.get("trace_id"))
        for candidate in existing_edges
        if isinstance(candidate, dict) and str(candidate.get("trace_id") or "")
    }
    child_ids: list[str] = []
    suffix = 0
    for part_index in range(2):
        candidate = f"{current_trace_id}::part_{part_index:03d}"
        while candidate in used or candidate in child_ids:
            suffix += 1
            candidate = f"{current_trace_id}::part_{part_index:03d}::split_{suffix:03d}"
        child_ids.append(candidate)
    return child_ids[0], child_ids[1], root_trace_id


def _apply_topology_decision(graph: dict[str, Any], decision: dict[str, Any], item_id: str, image_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate a topology edit completely, then apply it to the private copy."""
    name = str(decision.get("decision") or "")
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    node_map = {str(node.get("id")): node for node in nodes}
    edge_map = {str(edge.get("id")): edge for edge in edges}
    affected: list[str] = []
    before: dict[str, Any] = {}

    if name == "merge_nodes":
        ids = _topology_ids(decision, "node_id")
        ids = list(dict.fromkeys(ids))
        keep = str(decision.get("keep_node_id") or (ids[0] if ids else ""))
        if len(ids) < 2 or keep not in node_map or any(node_id not in node_map for node_id in ids):
            return None, {"warning": "invalid_merge_nodes", "review_item_id": item_id}
        removed_nodes = [node_id for node_id in ids if node_id != keep]
        for node_id in removed_nodes:
            before[node_id] = copy.deepcopy(node_map[node_id])
        original_edge_records = {str(edge.get("id")): copy.deepcopy(edge) for edge in edges}
        rewired_edges: list[str] = []
        collapsed: list[str] = []
        incident_edges: list[str] = []
        for edge in edges:
            edge_id = str(edge.get("id") or "")
            old_source, old_target = _endpoint_ids(edge)
            if old_source in ids or old_target in ids:
                connector_warning = _off_page_connector_warning(edge, item_id)
                if connector_warning is not None:
                    return None, connector_warning
            new_source = keep if old_source in ids else old_source
            new_target = keep if old_target in ids else old_target
            # Include every edge incident to a merged node in the audit,
            # including edges whose endpoint already used the retained node
            # and therefore has no rewritten endpoint value.
            if old_source in ids or old_target in ids:
                before.setdefault(edge_id, copy.deepcopy(edge))
                incident_edges.append(edge_id)
            if new_source != old_source or new_target != old_target:
                if new_source != old_source:
                    _set_endpoint_ids(edge, new_source, new_target)
                    if old_source != keep:
                        _clear_endpoint_metadata(edge, "source")
                    _set_endpoint_node_metadata(edge, "source", new_source)
                elif new_target != old_target:
                    _set_endpoint_ids(edge, new_source, new_target)
                if new_target != old_target:
                    _set_endpoint_ids(edge, new_source, new_target)
                    if old_target != keep:
                        _clear_endpoint_metadata(edge, "target")
                    _set_endpoint_node_metadata(edge, "target", new_target)
                if "legacy_source" in edge and old_source in ids:
                    edge["legacy_source"] = new_source
                if "legacy_target" in edge and old_target in ids:
                    edge["legacy_target"] = new_target
                rewired_edges.append(edge_id)
        kept_edges = []
        for edge in edges:
            src, dst = _endpoint_ids(edge)
            old_source, old_target = _endpoint_ids(original_edge_records[str(edge.get("id"))])
            if src == dst and old_source != old_target:
                collapsed.append(str(edge.get("id")))
            else:
                kept_edges.append(edge)
        graph["edges"] = kept_edges
        graph["nodes"] = [node for node in nodes if str(node.get("id")) == keep or str(node.get("id")) not in ids]
        affected = list(dict.fromkeys(removed_nodes + incident_edges + rewired_edges + collapsed))
    elif name == "reconnect_edge":
        edge_id = str(decision.get("edge_id") or "")
        src = str(decision.get("src") or decision.get("source") or "")
        dst = str(decision.get("dst") or decision.get("target") or "")
        endpoint = str(decision.get("endpoint") or "").lower()
        old_source, old_target = _endpoint_ids(edge_map.get(edge_id, {}))
        if endpoint in {"source", "target"}:
            node_id = str(decision.get("node_id") or "")
            old = edge_map.get(edge_id, {})
            src = node_id if endpoint == "source" else _endpoint_ids(old)[0]
            dst = node_id if endpoint == "target" else _endpoint_ids(old)[1]
        else:
            src = src or old_source
            dst = dst or old_target
        if edge_id not in edge_map or src not in node_map or dst not in node_map:
            return None, {"warning": "invalid_reconnect_edge", "review_item_id": item_id}
        if src == dst:
            return None, {"warning": "self_loop_not_supported", "review_item_id": item_id}
        edge = edge_map[edge_id]
        connector_warning = _off_page_connector_warning(edge, item_id)
        if connector_warning is not None:
            return None, connector_warning
        before[edge_id] = copy.deepcopy(edge)
        _set_endpoint_ids(edge, src, dst)
        if src != old_source:
            _clear_endpoint_metadata(edge, "source")
            _set_endpoint_node_metadata(edge, "source", src)
            if str((edge.get("off_page_connector") or {}).get("exit_terminal") or "").lower() in {"source", "src"}:
                edge.pop("off_page_connector", None)
        if dst != old_target:
            _clear_endpoint_metadata(edge, "target")
            _set_endpoint_node_metadata(edge, "target", dst)
            if str((edge.get("off_page_connector") or {}).get("exit_terminal") or "").lower() in {"destination", "target", "dst", "terminal"}:
                edge.pop("off_page_connector", None)
        affected = [edge_id]
    elif name == "delete_edge":
        ids = _topology_ids(decision, "edge_id")
        if not ids or any(edge_id not in edge_map for edge_id in ids):
            return None, {"warning": "invalid_delete_edge", "review_item_id": item_id}
        for edge_id in ids:
            before[edge_id] = copy.deepcopy(edge_map[edge_id])
        graph["edges"] = [edge for edge in edges if str(edge.get("id")) not in ids]
        affected = ids
    elif name == "set_node_type":
        node_id = str(decision.get("node_id") or "")
        node_type = str(decision.get("node_type") or decision.get("type") or "").strip()
        if node_id not in node_map or not node_type:
            return None, {"warning": "invalid_set_node_type", "review_item_id": item_id}
        before[node_id] = copy.deepcopy(node_map[node_id])
        node_map[node_id]["type"] = node_type
        affected = [node_id]
    elif name == "split_edge":
        edge_id = str(decision.get("edge_id") or "")
        new_node_id = str(decision.get("new_node_id") or "")
        first_id = str(decision.get("first_edge_id") or f"{edge_id}::a")
        second_id = str(decision.get("second_edge_id") or f"{edge_id}::b")
        if edge_id not in edge_map or not new_node_id or new_node_id in node_map or first_id == second_id or first_id in edge_map or second_id in edge_map:
            return None, {"warning": "invalid_split_edge", "review_item_id": item_id}
        edge = edge_map[edge_id]
        polyline = edge.get("polyline")
        if not isinstance(polyline, list) or len(polyline) < 3:
            return None, {"warning": "invalid_split_route", "review_item_id": item_id}
        try:
            if decision.get("split_index") is not None:
                split_index = int(decision.get("split_index"))
            else:
                point = decision.get("split_point") or decision.get("point")
                if isinstance(point, dict):
                    point = [point.get("x"), point.get("y")]
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    raise ValueError
                distances = []
                for vertex in polyline:
                    coords = (vertex.get("x"), vertex.get("y")) if isinstance(vertex, dict) else vertex[:2]
                    distances.append((float(coords[0]) - float(point[0])) ** 2 + (float(coords[1]) - float(point[1])) ** 2)
                split_index = min(range(len(polyline)), key=lambda index: distances[index])
                if distances[split_index] > float(decision.get("split_tolerance_px", 5.0)) ** 2:
                    raise ValueError
            if split_index <= 0 or split_index >= len(polyline) - 1:
                raise ValueError
        except (TypeError, ValueError):
            return None, {"warning": "invalid_split_route", "review_item_id": item_id}
        connector_warning = _off_page_connector_warning(edge, item_id)
        if connector_warning is not None:
            return None, connector_warning
        before[edge_id] = copy.deepcopy(edge)
        left, right = copy.deepcopy(edge), copy.deepcopy(edge)
        def _xy(value: Any) -> tuple[float, float]:
            return (float(value.get("x")), float(value.get("y"))) if isinstance(value, dict) else (float(value[0]), float(value[1]))
        left["id"], right["id"] = first_id, second_id
        left["polyline"] = copy.deepcopy(polyline[: split_index + 1])
        right["polyline"] = copy.deepcopy(polyline[split_index:])
        split_distance = sum(((_xy(polyline[i + 1])[0] - _xy(polyline[i])[0]) ** 2 + (_xy(polyline[i + 1])[1] - _xy(polyline[i])[1]) ** 2) ** 0.5 for i in range(split_index))
        for group, values in (edge.get("attachments") or {}).items():
            if not isinstance(values, list):
                continue
            left_values, right_values = [], []
            for value in values:
                route_distance = value.get("trace_distance_px") if isinstance(value, dict) else None
                copied = copy.deepcopy(value)
                if route_distance is not None and float(route_distance) > split_distance:
                    copied["trace_distance_px"] = round(float(route_distance) - split_distance, 2)
                    right_values.append(copied)
                else:
                    left_values.append(copied)
            left.setdefault("attachments", {})[group] = left_values
            right.setdefault("attachments", {})[group] = right_values
        original_source, original_target = _endpoint_ids(edge)
        _set_endpoint_ids(left, original_source, new_node_id)
        _set_endpoint_ids(right, new_node_id, original_target)
        _clear_endpoint_metadata(left, "target")
        _clear_endpoint_metadata(right, "source")
        _set_endpoint_node_metadata(left, "target", new_node_id)
        _set_endpoint_node_metadata(right, "source", new_node_id)
        if "legacy_source" in left:
            left["legacy_source"] = original_source
        if "legacy_target" in left:
            left["legacy_target"] = new_node_id
        if "legacy_source" in right:
            right["legacy_source"] = new_node_id
        if "legacy_target" in right:
            right["legacy_target"] = original_target
        left_trace_id, right_trace_id, root_trace_id = _unique_split_trace_ids(edge, edges)
        left["trace_id"] = left_trace_id
        right["trace_id"] = right_trace_id
        left["original_trace_id"] = root_trace_id
        right["original_trace_id"] = root_trace_id
        left["terminal_type"] = "junction"
        right["source_obj_type"] = None
        _copy_off_page_connector_for_split(edge, left, right)
        for child in (left, right):
            points = child.get("polyline", [])
            child["trace_length_px"] = sum(((
                (_xy(points[i + 1])[0] - _xy(points[i])[0]) ** 2
                + (_xy(points[i + 1])[1] - _xy(points[i])[1]) ** 2
            ) ** 0.5) for i in range(len(points) - 1))
            if edge.get("flow_direction_review_state") == "human_reviewed":
                child["flow_direction_state"] = edge.get("flow_direction_state")
                child["flow_direction_review_state"] = "human_reviewed"
            else:
                child["flow_direction_state"] = "unknown"
                child["flow_direction_review_state"] = "unresolved"
        graph["edges"] = [item for item in edges if str(item.get("id")) != edge_id] + [left, right]
        vertex = polyline[split_index]
        position = vertex if isinstance(vertex, dict) else {"x": vertex[0], "y": vertex[1]}
        graph.setdefault("nodes", []).append({"id": new_node_id, "type": str(decision.get("node_type") or "junction"), "position": position})
        affected = [edge_id, first_id, second_id, new_node_id]
    else:
        return None, {"warning": "unsupported_decision", "review_item_id": item_id}

    affected = list(dict.fromkeys(str(value) for value in affected if str(value)))
    for key in affected:
        before.setdefault(key, None)
    after_objects = {str(obj.get("id")): obj for obj in graph.get("nodes", []) + graph.get("edges", []) if isinstance(obj, dict) and str(obj.get("id") or "")}
    audit = {"id": f"audit::{name}::{item_id}", "image_id": image_id, "review_item_id": item_id,
             "decision": name, "affected_ids": sorted(affected), "before": before,
             "after": {key: copy.deepcopy(after_objects.get(key)) for key in sorted(affected)}}
    # Keep review attribution fields present on every topology audit so audit
    # consumers can rely on one shape even when a client omitted an optional
    # value.
    for field in ("reviewer", "note", "reviewed_at"):
        audit[field] = copy.deepcopy(decision.get(field))
    return audit, None


def _rebuild_graph_indexes(graph: dict[str, Any]) -> None:
    if "line_to_edges" not in graph and "line_groups" not in graph and not any(
        isinstance(edge, dict) and (edge.get("effective_line_number_ids") or edge.get("line_number_ids"))
        for edge in graph.get("edges", []) or []
    ):
        return
    line_to_edges: dict[str, list[str]] = {}
    for edge in graph.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        for value in edge.get("effective_line_number_ids") or edge.get("line_number_ids") or []:
            line_to_edges.setdefault(str(value), []).append(str(edge.get("id") or ""))
    graph["line_to_edges"] = {key: sorted(ids) for key, ids in sorted(line_to_edges.items()) if ids}
    graph["line_groups"] = [{"line_number_id": key, "edge_ids": ids} for key, ids in graph["line_to_edges"].items()]


def apply_stage9_review_decisions(
    *,
    image_id: str,
    graph_payload: dict[str, Any],
    review_items_payload: dict[str, Any],
    decisions_payload: dict[str, Any],
) -> dict[str, Any]:
    corrected_graph_payload = copy.deepcopy(graph_payload)
    review_items = [item for item in review_items_payload.get("review_items", []) or [] if isinstance(item, dict)]
    decisions_by_id = _decision_index(decisions_payload)
    raw_decision_entries = decisions_payload.get("decisions", []) or []
    raw_decisions = [decision for decision in raw_decision_entries if isinstance(decision, dict)]
    duplicate_ids = sorted(
        review_item_id
        for review_item_id, count in Counter(
            str(d.get("review_item_id") or "").strip() for d in raw_decisions
        ).items()
        if review_item_id and count > 1
    )
    review_ids = {str(item.get("id") or "") for item in review_items}
    malformed_decision_indexes = [
        index for index, decision in enumerate(raw_decision_entries)
        if not isinstance(decision, dict) or not str(decision.get("review_item_id") or "").strip()
    ]

    resolutions = []
    corrections: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    warnings.extend({"warning": "duplicate_decision", "review_item_id": item_id} for item_id in duplicate_ids)
    warnings.extend({"warning": "unknown_review_item_id", "review_item_id": item_id} for item_id in sorted(set(decisions_by_id) - review_ids))
    warnings.extend(
        {
            "warning": "missing_review_item_id",
            "decision_index": index,
            "message": "Stage 9 decision record is malformed because review_item_id is missing.",
        }
        for index in malformed_decision_indexes
    )
    for item in review_items:
        review_item_id = str(item.get("id") or "")
        decision = decisions_by_id.get(review_item_id)
        if review_item_id in duplicate_ids:
            decision = None
        resolution = _resolution_for_item(item, decision)
        if decision is not None and str(decision.get("decision") or "") == "set_line_number":
            correction, decision_warnings = _apply_set_line_number(
                corrected_graph_payload=corrected_graph_payload,
                decision=decision,
                review_item_id=review_item_id,
                image_id=image_id,
            )
            warnings.extend(decision_warnings)
            if correction is not None:
                corrections.append(correction)
                resolution["graph_changed"] = True
            else:
                resolution["resolution_state"] = "unresolved"
        elif decision is not None and str(decision.get("decision") or "") == "set_flow_direction":
            flow_decision = dict(decision)
            if not _flow_edge_ids(flow_decision):
                item_evidence = item.get("evidence")
                if isinstance(item_evidence, dict) and item_evidence.get("edge_id") is not None:
                    flow_decision["edge_id"] = item_evidence["edge_id"]
            correction, decision_warnings = _apply_set_flow_direction(
                corrected_graph_payload=corrected_graph_payload,
                decision=flow_decision,
                review_item_id=review_item_id,
                image_id=image_id,
            )
            warnings.extend(decision_warnings)
            if correction is not None:
                corrections.append(correction)
                resolution["graph_changed"] = True
                resolution["flow_direction_state"] = correction["flow_direction_state"]
            elif decision_warnings and any(warning.get("warning") == "invalid_flow_direction" for warning in decision_warnings):
                resolution["resolution_state"] = "unsupported_flow_direction"
            elif correction is None:
                resolution["resolution_state"] = "unresolved"
        elif decision is not None and str(decision.get("decision") or "") in _TOPOLOGY_DECISIONS:
            audit, topology_warning = _apply_topology_decision(corrected_graph_payload, decision, review_item_id, image_id)
            if audit is not None:
                corrections.append(audit)
                resolution["resolution_state"] = "applied"
                resolution["graph_changed"] = True
            else:
                resolution["resolution_state"] = "unresolved"
                if topology_warning:
                    topology_warning["review_item_id"] = review_item_id
                    warnings.append(topology_warning)
        resolutions.append(resolution)

    _rebuild_graph_indexes(corrected_graph_payload)
    state_counts = Counter(str(item.get("resolution_state") or "unknown") for item in resolutions)
    explicit_resolution_count = sum(1 for item in resolutions if item.get("resolution_state") in {"accept_as_is", "false_positive", "set_line_number", "set_flow_direction", "applied"})
    blocking_ids = []
    for item, resolution in zip(review_items, resolutions):
        category = str(item.get("category") or "").lower()
        if "release_blocking" in item:
            blocking = bool(item.get("release_blocking"))
        elif "blocking" in item:
            blocking = bool(item.get("blocking"))
        else:
            blocking = not (category.startswith("info") or category in {"informational", "context"})
        if blocking and resolution.get("resolution_state") not in {"accept_as_is", "false_positive", "set_line_number", "set_flow_direction", "applied"}:
            blocking_ids.append(str(item.get("id") or ""))

    release_blocked_by_warning = any(w.get("warning") in _RELEASE_BLOCKING_WARNINGS for w in warnings)
    return {
        "corrected_graph_payload": corrected_graph_payload,
        "review_resolution_payload": {
            "image_id": image_id,
            "source": "stage9_review_decisions",
            "resolutions": resolutions,
        },
        "correction_audit_payload": {
            "image_id": image_id,
            "source": "stage9_review_decisions",
            "corrections": corrections,
            "warnings": warnings,
        },
        "summary": {
            "image_id": image_id,
            "input_review_item_count": len(review_items),
            "decision_count": len(decisions_payload.get("decisions", []) or []),
            "explicit_resolution_count": explicit_resolution_count,
            "correction_count": len(corrections),
            "assumed_resolved_count": state_counts.get("accepted_by_assumption", 0),
            "unsupported_decision_count": state_counts.get("unsupported_decision", 0),
            "unsupported_flow_direction_count": state_counts.get("unsupported_flow_direction", 0),
            "resolution_state_counts": dict(state_counts),
            "warning_count": len(warnings),
        },
        "release_gate_payload": {
            "release_ready": not blocking_ids and not release_blocked_by_warning,
            "status": "ready" if not blocking_ids and not release_blocked_by_warning else "blocked",
            "blocking_count": len(blocking_ids),
            "blocking_review_item_ids": sorted(blocking_ids),
            "blocking_reasons": (
                (["unresolved_review_items"] if blocking_ids else [])
                + (["unknown_review_item_id"] if any(w.get("warning") == "unknown_review_item_id" for w in warnings) else [])
                + (["duplicate_decision"] if any(w.get("warning") == "duplicate_decision" for w in warnings) else [])
                + (["missing_review_item_id"] if any(w.get("warning") == "missing_review_item_id" for w in warnings) else [])
                + (["invalid_off_page_connector"] if any(w.get("warning") == "invalid_off_page_connector" for w in warnings) else [])
            ),
        },
    }
