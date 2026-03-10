from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from garnet.pipe_graph_qa import run_pipe_graph_qa_stage


def _decision_map(review_state: dict[str, Any]) -> dict[tuple[str, str], str]:
    decisions: dict[tuple[str, str], str] = {}
    for item in review_state.get("items", []):
        bucket = str(item.get("bucket", ""))
        entity_id = str(item.get("entity_id") or item.get("item_id") or "")
        decision = str(item.get("decision", "deferred"))
        if bucket and entity_id:
            decisions[(bucket, entity_id)] = decision
    return decisions


def _attachment_allowed(
    region_id: str,
    *,
    primary_bucket: str,
    fallback_bucket: str | None,
    decisions: dict[tuple[str, str], str],
) -> bool:
    primary = decisions.get((primary_bucket, region_id))
    if primary is not None:
        return primary != "rejected"
    if fallback_bucket is not None:
        fallback = decisions.get((fallback_bucket, region_id))
        if fallback is not None:
            return fallback != "rejected"
    return True


def build_reviewed_graph_payload(graph_payload: dict[str, Any], review_state: dict[str, Any]) -> dict[str, Any]:
    decisions = _decision_map(review_state)

    text_attachments = [
        attachment
        for attachment in graph_payload.get("text_attachments", [])
        if _attachment_allowed(
            str(attachment.get("region_id", "")),
            primary_bucket="stage12_line_attachment",
            fallback_bucket="stage4_line_number",
            decisions=decisions,
        )
    ]
    instrument_attachments = [
        attachment
        for attachment in graph_payload.get("instrument_tag_attachments", [])
        if _attachment_allowed(
            str(attachment.get("region_id", "")),
            primary_bucket="stage12_instrument_attachment",
            fallback_bucket="stage4_instrument",
            decisions=decisions,
        )
    ]

    allowed_text_ids = {str(item.get("region_id", "")) for item in text_attachments}
    allowed_instrument_ids = {str(item.get("region_id", "")) for item in instrument_attachments}

    reviewed_edges: list[dict[str, Any]] = []
    for edge in graph_payload.get("edges", []):
        next_edge = dict(edge)
        next_edge["line_texts"] = [
            item for item in edge.get("line_texts", []) if str(item.get("region_id", "")) in allowed_text_ids
        ]
        next_edge["instrument_tags"] = [
            item for item in edge.get("instrument_tags", []) if str(item.get("region_id", "")) in allowed_instrument_ids
        ]
        reviewed_edges.append(next_edge)

    return {
        **graph_payload,
        "edges": reviewed_edges,
        "text_attachments": text_attachments,
        "instrument_tag_attachments": instrument_attachments,
        "review_state_version": review_state.get("version", 1),
        "review_state_updated_at": review_state.get("updated_at"),
    }


def build_reviewed_graph_summary(graph_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": graph_payload.get("image_id"),
        "pass_type": graph_payload.get("pass_type", "sheet"),
        "node_count": len(graph_payload.get("nodes", [])),
        "edge_count": len(graph_payload.get("edges", [])),
        "unresolved_junction_count": len(graph_payload.get("unresolved_junction_ids", [])),
        "accepted_attachment_count": len(graph_payload.get("equipment_attachments", [])),
        "accepted_text_attachment_count": len(graph_payload.get("text_attachments", [])),
        "accepted_instrument_tag_attachment_count": len(graph_payload.get("instrument_tag_attachments", [])),
        "source_artifacts": [
            "stage12_graph.json",
            "stage_review_state.json",
        ],
    }


def generate_reviewed_outputs(job_dir: str | Path) -> dict[str, Any]:
    base = Path(job_dir)
    graph_path = base / "stage12_graph.json"
    review_state_path = base / "stage_review_state.json"
    if not graph_path.exists():
      raise FileNotFoundError(f"Missing graph artifact: {graph_path}")
    if not review_state_path.exists():
      raise FileNotFoundError(f"Missing review state artifact: {review_state_path}")

    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    review_state = json.loads(review_state_path.read_text(encoding="utf-8"))

    reviewed_graph = build_reviewed_graph_payload(graph_payload, review_state)
    reviewed_graph_summary = build_reviewed_graph_summary(reviewed_graph)
    reviewed_qa = run_pipe_graph_qa_stage(
        image_id=str(reviewed_graph.get("image_id", "")),
        graph_payload=reviewed_graph,
    )

    (base / "stage12_graph_reviewed.json").write_text(json.dumps(reviewed_graph, indent=2), encoding="utf-8")
    (base / "stage12_graph_reviewed_summary.json").write_text(json.dumps(reviewed_graph_summary, indent=2), encoding="utf-8")
    (base / "stage13_graph_anomalies_reviewed.json").write_text(json.dumps(reviewed_qa["anomaly_report"], indent=2), encoding="utf-8")
    (base / "stage13_review_queue_reviewed.json").write_text(json.dumps(reviewed_qa["review_queue"], indent=2), encoding="utf-8")
    (base / "stage13_graph_qa_reviewed_summary.json").write_text(json.dumps(reviewed_qa["summary"], indent=2), encoding="utf-8")

    return {
        "graph_payload": reviewed_graph,
        "graph_summary": reviewed_graph_summary,
        "qa": reviewed_qa,
    }
