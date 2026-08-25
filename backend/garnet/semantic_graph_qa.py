"""Semantic (chemical-engineering) QA pass over an assembled process graph.

This stage is *advisory*: it does not modify the graph. It prompts a language model
with a compact node/edge summary plus a document of process-design rules
(`semantic_qa_rules.md`) and returns a list of semantic anomaly candidates in the
same issue shape used by `trace_graph_qa`, so they can flow into the existing review
queue. Geometry remains authoritative; semantics are only ever a flag.

Status: **not yet wired into the pipeline.** This module has no callers in
`pid_extractor.py` or `api.py`; it is retained as advisory scaffolding for future
work. See `tests/test_semantic_graph_qa.py` for the locked-in fail-soft contract.

Model-agnostic and configurable via `SemanticQaConfig`. No fine-tuning; domain
knowledge is injected as prompt context only. The stage is designed to fail soft:
if no API key is present or the model call errors, it returns an empty issue list
with a `skipped` marker rather than aborting the pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).resolve().parent / "semantic_qa_rules.md"
ROOT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

_SEVERITIES = {"high", "medium", "low"}
_MAX_ANOMALIES = 64


@dataclass(frozen=True)
class SemanticQaConfig:
    model_name: str = "google/gemini-3-flash-preview"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    rules_path: Path = RULES_PATH
    openrouter_api_key: str | None = None
    max_anomalies: int = _MAX_ANOMALIES


def _resolve_api_key(explicit_key: str | None) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    env_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if env_key:
        return env_key
    if ROOT_ENV_PATH.exists():
        try:
            from dotenv import dotenv_values

            root_key = str(dotenv_values(ROOT_ENV_PATH).get("OPENROUTER_API_KEY") or "").strip()
            if root_key:
                return root_key
        except Exception:  # pragma: no cover - dotenv is optional here
            pass
    return ""


def _load_rules(rules_path: Path) -> str:
    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing semantic QA rules file: {path}")
    return path.read_text(encoding="utf-8")


def _node_type(node: dict[str, Any]) -> str:
    return str(node.get("type") or node.get("kind") or node.get("node_type") or "")


def _node_text(node: dict[str, Any]) -> str:
    tags = node.get("tags")
    if isinstance(tags, dict):
        for key in ("pid_tag", "line_tag", "service"):
            value = str(tags.get(key) or "").strip()
            if value:
                return value
    for key in ("normalized_text", "display_text", "text", "label"):
        value = str(node.get(key) or "").strip()
        if value:
            return value
    return ""


def _node_position(node: dict[str, Any]) -> dict[str, float] | None:
    position = node.get("position")
    if isinstance(position, dict) and {"x", "y"}.issubset(position):
        return {"x": round(float(position["x"]), 3), "y": round(float(position["y"]), 3)}
    center = node.get("geometry", {}).get("center")
    if isinstance(center, dict) and {"x", "y"}.issubset(center):
        return {"x": round(float(center["x"]), 3), "y": round(float(center["y"]), 3)}
    bbox = node.get("bbox")
    if isinstance(bbox, dict):
        try:
            x = _safe_float(bbox.get("x"))
            y = _safe_float(bbox.get("y"))
            w = _safe_float(bbox.get("w"))
            h = _safe_float(bbox.get("h"))
            if x is not None and y is not None:
                w = w or 0.0
                h = h or 0.0
                return {"x": round(x + w / 2.0, 3), "y": round(y + h / 2.0, 3)}
        except (TypeError, ValueError):
            pass
    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    source = str(edge.get("source") or edge.get("src") or "")
    target = str(edge.get("target") or edge.get("dst") or "")
    return source, target


def _edge_line_style(edge: dict[str, Any]) -> str:
    return str(edge.get("line_style") or edge.get("type") or "")


def _edge_line_number_texts(edge: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for record in edge.get("line_numbers") or []:
        if isinstance(record, dict):
            text = str(record.get("display_text") or record.get("normalized_text") or record.get("text") or "").strip()
            if text:
                texts.append(text)
    return texts


def _build_summary(graph_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (summary, nodes_by_id). The summary is the compact input to the model."""
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in graph_payload.get("nodes", []) or []:
        node_id = str(node.get("id") or "")
        if node_id:
            nodes_by_id[node_id] = node

    node_summaries = [
        {"id": node_id, "type": _node_type(node), "text": _node_text(node)}
        for node_id, node in nodes_by_id.items()
    ]

    edge_summaries: list[dict[str, Any]] = []
    for edge in graph_payload.get("edges", []) or []:
        source, target = _edge_endpoints(edge)
        edge_summaries.append(
            {
                "id": str(edge.get("id") or ""),
                "source": source,
                "target": target,
                "source_text": _node_text(nodes_by_id.get(source, {})),
                "target_text": _node_text(nodes_by_id.get(target, {})),
                "line_style": _edge_line_style(edge),
                "line_numbers": _edge_line_number_texts(edge),
            }
        )

    return {
        "nodes": node_summaries,
        "edges": edge_summaries,
    }, nodes_by_id


def _anomaly_geometry(
    *,
    node_id: str | None,
    edge_id: str | None,
    nodes_by_id: dict[str, dict[str, Any]],
    edges_by_id: dict[str, dict[str, Any]],
) -> dict[str, float] | None:
    if node_id and node_id in nodes_by_id:
        return _node_position(nodes_by_id[node_id])
    if edge_id and edge_id in edges_by_id:
        edge = edges_by_id[edge_id]
        source, target = _edge_endpoints(edge)
        source_pos = _node_position(nodes_by_id.get(source, {}))
        target_pos = _node_position(nodes_by_id.get(target, {}))
        if source_pos is not None and target_pos is not None:
            return {
                "x": round((source_pos["x"] + target_pos["x"]) / 2.0, 3),
                "y": round((source_pos["y"] + target_pos["y"]) / 2.0, 3),
            }
        return source_pos or target_pos
    return None


def _issue(
    *,
    category: str,
    severity: str,
    message: str,
    node_id: str | None = None,
    edge_id: str | None = None,
    geometry: dict[str, float] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parts = [category]
    if edge_id:
        parts.append(edge_id)
    elif node_id:
        parts.append(node_id)
    issue_id = "qa::" + "::".join(parts)
    payload: dict[str, Any] = {
        "id": issue_id,
        "category": category,
        "severity": severity,
        "message": message,
    }
    if node_id is not None:
        payload["node_id"] = node_id
    if edge_id is not None:
        payload["edge_id"] = edge_id
    if geometry is not None:
        payload["geometry"] = geometry
    if evidence is not None:
        payload["evidence"] = evidence
    return payload


def _normalise_anomaly(
    raw: dict[str, Any],
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    edges_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    node_id = str(raw.get("node_id") or "").strip() or None
    edge_id = str(raw.get("edge_id") or "").strip() or None
    if node_id and node_id not in nodes_by_id:
        node_id = None
    if edge_id and edge_id not in edges_by_id:
        edge_id = None
    if not node_id and not edge_id:
        # The model must anchor every anomaly to an existing node or edge.
        return None

    severity = str(raw.get("severity") or "low").strip().lower()
    if severity not in _SEVERITIES:
        severity = "low"
    rule_id = str(raw.get("rule_id") or "").strip() or "semantic"
    message = str(raw.get("message") or "").strip()
    if not message:
        return None

    confidence = raw.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None

    return _issue(
        category=f"semantic::{rule_id}",
        severity=severity,
        message=message,
        node_id=node_id,
        edge_id=edge_id,
        geometry=_anomaly_geometry(
            node_id=node_id, edge_id=edge_id, nodes_by_id=nodes_by_id, edges_by_id=edges_by_id
        ),
        evidence={"confidence": confidence},
    )


def _call_model(cfg: SemanticQaConfig, rules: str, summary: dict[str, Any], api_key: str) -> dict[str, Any]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(base_url=cfg.base_url, api_key=api_key)
    system_prompt = (
        "You are a senior chemical process engineer auditing the topology of a "
        "digitised P&ID graph. Apply the process-design rules below to the graph "
        "summary and return ONLY the anomalies you are confident about, as a JSON "
        "object. Reference only existing node/edge ids. Flag fewer, higher-confidence "
        "anomalies over many weak ones.\n\n"
        "Process-design rules:\n\n"
        f"{rules}"
    )
    user_prompt = (
        "Here is the graph summary (nodes and edges with their tag text and line "
        "style). Return a JSON object of the form "
        '{"anomalies": [{"rule_id": "...", "severity": "high|medium|low", '
        '"node_id": "optional existing id", "edge_id": "optional existing id", '
        '"message": "...", "confidence": 0.0}], "notes": "optional"}.\n\n'
        f"Graph summary:\n{json.dumps(summary, ensure_ascii=False)}"
    )

    response = client.chat.completions.create(
        model=cfg.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("Semantic QA model response was not a string")
    return json.loads(content)


def run_semantic_graph_qa(
    *,
    image_id: str,
    graph_payload: dict[str, Any],
    cfg: SemanticQaConfig | None = None,
) -> dict[str, Any]:
    """Run the semantic QA pass. Never raises on model/network failure — advisory only.

    Returns a payload with `issues` (trace_graph_qa-compatible), `summary`,
    `skipped`, and (when skipped or errored) `skip_reason`.
    """
    cfg = cfg or SemanticQaConfig()

    summary, nodes_by_id = _build_summary(graph_payload)
    edges_by_id = {
        str(edge.get("id") or ""): edge
        for edge in graph_payload.get("edges", []) or []
        if str(edge.get("id") or "")
    }

    api_key = _resolve_api_key(cfg.openrouter_api_key)
    if not api_key:
        return {
            "image_id": image_id,
            "issues": [],
            "summary": summary,
            "skipped": True,
            "skip_reason": "OPENROUTER_API_KEY not configured",
        }

    try:
        raw_response = _call_model(cfg, _load_rules(cfg.rules_path), summary, api_key)
    except Exception as exc:  # noqa: BLE001 - advisory stage must not abort the pipeline
        return {
            "image_id": image_id,
            "issues": [],
            "summary": summary,
            "skipped": True,
            "skip_reason": f"model call failed: {exc}",
        }

    issues: list[dict[str, Any]] = []
    for raw in (raw_response.get("anomalies") or [])[: cfg.max_anomalies]:
        if not isinstance(raw, dict):
            continue
        issue = _normalise_anomaly(raw, nodes_by_id=nodes_by_id, edges_by_id=edges_by_id)
        if issue is not None:
            issues.append(issue)

    return {
        "image_id": image_id,
        "issues": issues,
        "summary": summary,
        "skipped": False,
        "skip_reason": "",
        "notes": raw_response.get("notes", ""),
        "raw_response": raw_response,
    }
