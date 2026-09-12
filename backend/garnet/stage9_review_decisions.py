from __future__ import annotations

import copy
from collections import Counter
from typing import Any

_NOOP_DECISIONS = {"accept_as_is", "false_positive", "defer"}
_FLOW_DIRECTION_DECISION_STATES = {"forward", "reverse", "bidirectional", "unknown"}


def _decision_index(decisions_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decisions_by_id: dict[str, dict[str, Any]] = {}
    for decision in decisions_payload.get("decisions", []) or []:
        if not isinstance(decision, dict):
            continue
        review_item_id = str(decision.get("review_item_id") or "")
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
            "resolution_state": "accepted_by_assumption",
            "decision_source": "stage9_identity_pass",
            "graph_changed": False,
        }

    decision_name = str(decision.get("decision") or "")
    reviewer = str(decision.get("reviewer") or "unspecified")
    if decision_name == "set_line_number":
        resolution_state = "set_line_number"
    elif decision_name == "set_flow_direction":
        resolution_state = "set_flow_direction"
    elif decision_name in _NOOP_DECISIONS:
        resolution_state = decision_name
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

    resolutions = []
    corrections: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for item in review_items:
        review_item_id = str(item.get("id") or "")
        decision = decisions_by_id.get(review_item_id)
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
        resolutions.append(resolution)

    state_counts = Counter(str(item.get("resolution_state") or "unknown") for item in resolutions)
    explicit_resolution_count = sum(
        1 for item in resolutions if item.get("resolution_state") not in {"accepted_by_assumption"}
    )

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
    }
