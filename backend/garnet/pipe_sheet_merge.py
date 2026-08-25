"""
Multi-sheet P&ID merge engine.

Reads a list of graph_v1 payloads (one per sheet), resolves off-page connector
pairs using the (reference_type, reference_value) merge key, and produces a
`MergeResult` with cross-sheet virtual edges and a list of issues requiring
human review.

Usage
-----
    from garnet.pipe_sheet_merge import resolve_merge_pairs

    merged = resolve_merge_pairs([sheet_a_graph_v1, sheet_b_graph_v1])
    for edge in merged.cross_sheet_edges:
        print(edge["id"], edge["merge_key"], edge["status"])
    for issue in merged.merge_issues:
        print(issue["issue_id"], issue["type"])
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

MergeKey = tuple[str, str]  # (reference_type, reference_value)

Direction = Literal["output", "input", "bidirectional"]

CrossSheetEdgeStatus = Literal["merged", "pending_human_review"]
IssueType = Literal[
    "ambiguous_merge",
    "ambiguous_match",
    "intra_sheet_duplicate",
    "dangling_connector",
    "direction_conflict",
    "missing_connector_key",
    "missing_target_sheet",
    "unknown_target_sheet",
    "non_reciprocal_reference",
    "rejected_connector",
    "stale_review_reference",
]


@dataclass
class CrossSheetEdge:
    """A virtual edge that connects two off-page connectors on different sheets."""

    id: str
    merge_key: MergeKey
    reference_type: str
    reference_value: str
    sheets: list[str]  # exactly 2 sheet doc_ids
    terminals: list[dict[str, Any]]
    direction_a: str
    direction_b: str
    status: CrossSheetEdgeStatus = "merged"
    connector_key: str | None = None
    match_method: str = "legacy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "merge_key": {
                "reference_type": self.merge_key[0],
                "reference_value": self.merge_key[1],
            },
            "reference_type": self.reference_type,
            "reference_value": self.reference_value,
            "sheets": self.sheets,
            "terminals": self.terminals,
            "direction_pair": (self.direction_a, self.direction_b),
            "status": self.status,
            "connector_key": self.connector_key or self.reference_value,
            "match_method": self.match_method,
        }


@dataclass
class MergeIssue:
    """An issue detected during the merge that requires human review."""

    issue_id: str
    type: IssueType
    merge_key: MergeKey
    sheets_involved: list[str]
    connectors: list[dict[str, Any]] = field(default_factory=list)
    resolution: str = "pending_human_review"

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "type": self.type,
            "merge_key": {
                "reference_type": self.merge_key[0],
                "reference_value": self.merge_key[1],
            },
            "sheets_involved": self.sheets_involved,
            "connectors": self.connectors,
            "resolution": self.resolution,
        }


@dataclass
class SheetResolved:
    """Per-sheet summary of how many connectors were resolved vs left dangling."""

    doc_id: str
    resolved_count: int = 0
    dangling_count: int = 0
    resolved_references: list[str] = field(default_factory=list)
    dangling_references: list[str] = field(default_factory=list)


@dataclass
class MergeResult:
    """Output of the merge engine."""

    schema_version: str = "graph_v2"
    cross_sheet_edges: list[CrossSheetEdge] = field(default_factory=list)
    merge_issues: list[MergeIssue] = field(default_factory=list)
    per_sheet_resolved: list[SheetResolved] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cross_sheet_edges": [e.to_dict() for e in self.cross_sheet_edges],
            "merge_issues": [i.to_dict() for i in self.merge_issues],
            "per_sheet_resolved": [
                {
                    "doc_id": s.doc_id,
                    "resolved_count": s.resolved_count,
                    "dangling_count": s.dangling_count,
                    "resolved_references": s.resolved_references,
                    "dangling_references": s.dangling_references,
                }
                for s in self.per_sheet_resolved
            ],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VIRTUAL_EDGE_RE = re.compile(r"^xs::(.+?)::(.+?)::(.+)$")


def _normalize_doc_id(doc_id: str | None) -> str:
    if not doc_id or not str(doc_id).strip():
        return "UNKNOWN"
    return str(doc_id).strip()


def _normalize_ref_type(ref_type: str | None) -> str:
    if ref_type in ("sheet", "pid", "drawing", "figure"):
        return ref_type
    return "sheet"


def _normalize_direction(direction: str | None) -> Direction:
    if direction in ("output", "input", "bidirectional"):
        return direction  # type: ignore[return-value]
    return "bidirectional"


def _is_complementary(a: Direction, b: Direction) -> bool:
    """Return True if a and b can safely be paired."""
    if a == "bidirectional" or b == "bidirectional":
        return True
    return a != b


def _make_virtual_edge_id(doc_id_a: str, doc_id_b: str, ref_value: str) -> str:
    """Canonical ID for a cross-sheet virtual edge."""
    # Sort so A→B and B→A produce the same ID
    a, b = sorted([doc_id_a, doc_id_b])
    safe_ref = re.sub(r"[^a-zA-Z0-9_\-]", "_", ref_value)
    return f"xs::{a}::{safe_ref}::{b}"


# ---------------------------------------------------------------------------
# Off-page connector extraction
# ---------------------------------------------------------------------------

def _extract_off_page_connectors(
    graph: dict[str, Any]
) -> list[dict[str, Any]]:
    """Walk graph_v1 and collect all edges with off_page_connector set."""
    connectors = []
    doc_id = _normalize_doc_id(graph.get("document", {}).get("doc_id"))
    schema_version = str(graph.get("schema_version", "graph_v1"))

    for edge in graph.get("edges", []):
        opc = edge.get("off_page_connector")
        if not opc:
            continue
        ref_type = _normalize_ref_type(opc.get("reference_type"))
        ref_value = str(opc.get("reference_value") or "").strip()
        if not ref_value:
            continue

        connectors.append(
            {
                "doc_id": doc_id,
                "schema_version": schema_version,
                "local_edge_id": str(edge.get("id", "")),
                "edge": edge,
                "reference_type": ref_type,
                "reference_value": ref_value,
                "direction": _normalize_direction(opc.get("direction")),
                "exit_terminal": str(opc.get("exit_terminal") or "source"),
            }
        )

    return connectors


# ---------------------------------------------------------------------------
# Pairing logic
# ---------------------------------------------------------------------------

def _build_index(
    connectors: list[dict[str, Any]]
) -> dict[MergeKey, list[dict[str, Any]]]:
    """Group connectors by merge key."""
    index: dict[MergeKey, list[dict[str, Any]]] = {}
    for conn in connectors:
        key: MergeKey = (conn["reference_type"], conn["reference_value"])
        index.setdefault(key, []).append(conn)
    return index


def _pair_connectors(
    key: MergeKey,
    conns: list[dict[str, Any]],
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
]:
    """
    Pair connectors by (reference_type, reference_value).

    Returns (pairs, unpaired) where each pair is two connectors from
    different sheets with complementary or bidirectional directions.

    Rules
    -----
    - A connector can appear in at most one pair
    - If >2 connectors share the same key → all are unpaired (AMBIGUOUS)
    - If 2 connectors are from the same sheet → INTRA_SHEET_DUPLICATE
    - If directions conflict (both OUTPUT or both INPUT with neither
      bidirectional) → both unpaired (DIRECTION_CONFLICT)
    """
    if len(conns) > 2:
        # Ambiguous — flag all, do not pair
        return [], conns

    if len(conns) < 2:
        return [], conns

    a, b = conns[0], conns[1]

    # Same sheet → intra-sheet duplicate
    if a["doc_id"] == b["doc_id"]:
        return [], conns

    # Direction compatibility check
    if not _is_complementary(a["direction"], b["direction"]):
        return [], conns

    # Prefer (output, input) over bidirectional pairings
    # Both bidirectional is OK
    return [(a, b)], []


def _normalize_match_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _strict_connectors(
    graphs: list[dict[str, Any]],
    connector_overrides: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    connectors: list[dict[str, Any]] = []
    connector_ids: set[str] = set()
    for graph in graphs:
        sheet_id = _normalize_doc_id(graph.get("document", {}).get("doc_id"))
        for edge in graph.get("edges", []):
            opc = edge.get("off_page_connector")
            if not isinstance(opc, dict):
                continue
            local_edge_id = str(opc.get("local_edge_id") or edge.get("id") or "")
            if not local_edge_id:
                continue
            connector_id = f"{sheet_id}::{local_edge_id}"
            override = connector_overrides.get(connector_id, {})
            target_sheet_id = str(
                override.get("target_sheet_id")
                or opc.get("target_sheet_reference")
                or opc.get("reference_value")
                or ""
            ).strip()
            connector_key = str(override.get("connector_key") or opc.get("connector_key") or "").strip()
            state = str(override.get("review_state") or "accepted").strip().lower()
            connectors.append(
                {
                    "connector_id": connector_id,
                    "doc_id": sheet_id,
                    "local_edge_id": local_edge_id,
                    "edge": edge,
                    "reference_type": _normalize_ref_type(opc.get("reference_type")),
                    "reference_value": str(opc.get("reference_value") or "").strip(),
                    "raw_reference_text": str(opc.get("raw_reference_text") or ""),
                    "target_sheet_id": target_sheet_id,
                    "connector_key": connector_key,
                    "direction": "bidirectional",
                    "exit_terminal": str(opc.get("exit_terminal") or "source"),
                    "review_state": "rejected" if state == "rejected" else "accepted",
                }
            )
            connector_ids.add(connector_id)
    connectors.sort(key=lambda item: item["connector_id"])
    return connectors, connector_ids


def _strict_issue(issue_type: IssueType, connector: dict[str, Any]) -> MergeIssue:
    connector_id = connector["connector_id"]
    return MergeIssue(
        issue_id=f"{issue_type.upper()}::{connector_id}",
        type=issue_type,
        merge_key=("connector", connector_id),
        sheets_involved=[connector["doc_id"]],
        connectors=[connector],
    )


def _strict_edge(a: dict[str, Any], b: dict[str, Any], match_method: str) -> CrossSheetEdge:
    connector_key = a.get("connector_key") or b.get("connector_key") or "manual"
    terminals = [
        {
            "connector_id": connector["connector_id"],
            "sheet": connector["doc_id"],
            "local_edge_id": connector["local_edge_id"],
            "exit_terminal": connector["exit_terminal"],
            "direction": connector["direction"],
        }
        for connector in (a, b)
    ]
    terminals.sort(key=lambda item: item["connector_id"])
    return CrossSheetEdge(
        id=_make_virtual_edge_id(a["doc_id"], b["doc_id"], f"{a['local_edge_id']}::{b['local_edge_id']}"),
        merge_key=("line_or_tag", connector_key),
        reference_type="line_or_tag",
        reference_value=connector_key,
        sheets=sorted([a["doc_id"], b["doc_id"]]),
        terminals=terminals,
        direction_a=a["direction"],
        direction_b=b["direction"],
        connector_key=connector_key,
        match_method=match_method,
    )


def _resolve_strict_merge(
    graphs: list[dict[str, Any]],
    connector_overrides: dict[str, dict[str, Any]],
    manual_pairs: list[dict[str, str]],
) -> MergeResult:
    connectors, connector_ids = _strict_connectors(graphs, connector_overrides)
    by_id = {connector["connector_id"]: connector for connector in connectors}
    sheet_by_normalized = {
        _normalize_match_text(graph.get("document", {}).get("doc_id")): _normalize_doc_id(
            graph.get("document", {}).get("doc_id")
        )
        for graph in graphs
    }
    issues: list[MergeIssue] = []
    edges: list[CrossSheetEdge] = []
    used: set[str] = set()

    stale_ids = sorted(set(connector_overrides) - connector_ids)
    for connector_id in stale_ids:
        issues.append(
            MergeIssue(
                issue_id=f"STALE_REVIEW_REFERENCE::{connector_id}",
                type="stale_review_reference",
                merge_key=("connector", connector_id),
                sheets_involved=[],
                connectors=[],
            )
        )

    for pair in manual_pairs:
        left_id = str(pair.get("left_connector_id") or "")
        right_id = str(pair.get("right_connector_id") or "")
        if left_id not in by_id or right_id not in by_id:
            stale = left_id if left_id not in by_id else right_id
            issues.append(
                MergeIssue(
                    issue_id=f"STALE_REVIEW_REFERENCE::{stale}",
                    type="stale_review_reference",
                    merge_key=("connector", stale),
                    sheets_involved=[],
                    connectors=[],
                )
            )
            continue
        left, right = by_id[left_id], by_id[right_id]
        if (
            left_id == right_id
            or left["doc_id"] == right["doc_id"]
            or left_id in used
            or right_id in used
            or left["review_state"] == "rejected"
            or right["review_state"] == "rejected"
        ):
            issues.append(_strict_issue("ambiguous_match", left))
            continue
        edges.append(_strict_edge(left, right, "manual"))
        used.update((left_id, right_id))

    candidates: dict[str, list[str]] = {connector["connector_id"]: [] for connector in connectors}
    eligible = [
        connector
        for connector in connectors
        if connector["connector_id"] not in used
        and connector["review_state"] != "rejected"
        and _normalize_match_text(connector["connector_key"])
        and _normalize_match_text(connector["target_sheet_id"]) in sheet_by_normalized
    ]
    for index, left in enumerate(eligible):
        for right in eligible[index + 1 :]:
            if left["doc_id"] == right["doc_id"]:
                continue
            if _normalize_match_text(left["connector_key"]) != _normalize_match_text(right["connector_key"]):
                continue
            if _normalize_match_text(left["target_sheet_id"]) != _normalize_match_text(right["doc_id"]):
                continue
            if _normalize_match_text(right["target_sheet_id"]) != _normalize_match_text(left["doc_id"]):
                continue
            candidates[left["connector_id"]].append(right["connector_id"])
            candidates[right["connector_id"]].append(left["connector_id"])

    for connector in eligible:
        connector_id = connector["connector_id"]
        if connector_id in used or len(candidates[connector_id]) != 1:
            continue
        partner_id = candidates[connector_id][0]
        if partner_id in used or len(candidates[partner_id]) != 1:
            continue
        edges.append(_strict_edge(connector, by_id[partner_id], "automatic"))
        used.update((connector_id, partner_id))

    for connector in connectors:
        connector_id = connector["connector_id"]
        if connector_id in used:
            continue
        if connector["review_state"] == "rejected":
            issues.append(_strict_issue("rejected_connector", connector))
            continue
        key = _normalize_match_text(connector["connector_key"])
        target = _normalize_match_text(connector["target_sheet_id"])
        if not key:
            issues.append(_strict_issue("missing_connector_key", connector))
        elif not target:
            issues.append(_strict_issue("missing_target_sheet", connector))
        elif target not in sheet_by_normalized:
            issues.append(_strict_issue("unknown_target_sheet", connector))
        elif len(candidates[connector_id]) > 1:
            issues.append(_strict_issue("ambiguous_match", connector))
        else:
            same_key_on_target = any(
                other["doc_id"] == sheet_by_normalized[target]
                and _normalize_match_text(other["connector_key"]) == key
                for other in connectors
            )
            issues.append(
                _strict_issue(
                    "non_reciprocal_reference" if same_key_on_target else "dangling_connector",
                    connector,
                )
            )

    resolved_by_sheet: dict[str, list[str]] = {}
    dangling_by_sheet: dict[str, list[str]] = {}
    for edge in edges:
        for terminal in edge.terminals:
            resolved_by_sheet.setdefault(terminal["sheet"], []).append(edge.connector_key or "")
    for issue in issues:
        for connector in issue.connectors:
            dangling_by_sheet.setdefault(connector["doc_id"], []).append(connector.get("connector_key") or connector["connector_id"])
    sheet_ids = sorted({_normalize_doc_id(graph.get("document", {}).get("doc_id")) for graph in graphs})
    per_sheet = [
        SheetResolved(
            doc_id=sheet_id,
            resolved_count=len(resolved_by_sheet.get(sheet_id, [])),
            dangling_count=len(dangling_by_sheet.get(sheet_id, [])),
            resolved_references=resolved_by_sheet.get(sheet_id, []),
            dangling_references=dangling_by_sheet.get(sheet_id, []),
        )
        for sheet_id in sheet_ids
    ]
    edges.sort(key=lambda edge: edge.id)
    issues.sort(key=lambda issue: issue.issue_id)
    return MergeResult(cross_sheet_edges=edges, merge_issues=issues, per_sheet_resolved=per_sheet)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def resolve_merge_pairs(
    graphs: list[dict[str, Any]],
    *,
    strict: bool = False,
    connector_overrides: dict[str, dict[str, Any]] | None = None,
    manual_pairs: list[dict[str, str]] | None = None,
) -> MergeResult:
    """
    Resolve off-page connector pairs across multiple sheets.

    Parameters
    ----------
    graphs
        List of graph_v1 payloads, one per P&ID sheet. Each must contain
        a ``document.doc_id`` and a list of ``edges`` with ``off_page_connector``
        fields (as produced by stage12b_graph_export after S6-03).

    Returns
    -------
    MergeResult
        ``cross_sheet_edges``
            Virtual edges connecting paired off-page connectors across sheets.
        ``merge_issues``
            Ambiguous, dangling, or duplicate connectors requiring human review.
        ``per_sheet_resolved``
            Per-sheet summary of resolved vs dangling connectors.

    Notes
    -----
    - Connectors with the same (reference_type, reference_value) but on
      different sheets form a **pair** and become one cross-sheet edge.
    - Ambiguous merges (>2 connectors with the same key) are flagged, not auto-merged.
    - Dangling connectors (no matching partner anywhere) are flagged.
    - Intra-sheet duplicates (same key, same sheet) are flagged.
    - The merge key is ``(reference_type, reference_value)`` —
      e.g. ``("sheet", "A-3")``.
    """
    if strict:
        return _resolve_strict_merge(graphs, connector_overrides or {}, manual_pairs or [])

    all_connectors: list[dict[str, Any]] = []
    for g in graphs:
        all_connectors.extend(_extract_off_page_connectors(g))

    index = _build_index(all_connectors)

    cross_sheet_edges: list[CrossSheetEdge] = []
    merge_issues: list[MergeIssue] = []
    sheet_stats: dict[str, dict[str, list[str]]] = {}

    def _stats(doc_id: str) -> dict[str, list[str]]:
        if doc_id not in sheet_stats:
            sheet_stats[doc_id] = {"resolved": [], "dangling": []}
        return sheet_stats[doc_id]

    for (ref_type, ref_value), conns in index.items():
        key: MergeKey = (ref_type, ref_value)
        pairs, unpaired = _pair_connectors(key, conns)

        # Record all connectors as seen
        for conn in conns:
            doc_id = conn["doc_id"]
            _stats(doc_id)  # ensure entry exists

        if pairs:
            for a, b in pairs:
                edge_id = _make_virtual_edge_id(a["doc_id"], b["doc_id"], ref_value)
                terminals = [
                    {
                        "sheet": a["doc_id"],
                        "local_edge_id": a["local_edge_id"],
                        "exit_terminal": a["exit_terminal"],
                        "direction": a["direction"],
                    },
                    {
                        "sheet": b["doc_id"],
                        "local_edge_id": b["local_edge_id"],
                        "exit_terminal": b["exit_terminal"],
                        "direction": b["direction"],
                    },
                ]
                cross_sheet_edges.append(
                    CrossSheetEdge(
                        id=edge_id,
                        merge_key=key,
                        reference_type=ref_type,
                        reference_value=ref_value,
                        sheets=sorted([a["doc_id"], b["doc_id"]]),
                        terminals=terminals,
                        direction_a=a["direction"],
                        direction_b=b["direction"],
                        status="merged",
                    )
                )
                _stats(a["doc_id"])["resolved"].append(ref_value)
                _stats(b["doc_id"])["resolved"].append(ref_value)

        # Classify unpaired
        unique_sheets = {c["doc_id"] for c in unpaired}

        if len(unpaired) == 1:
            # Dangling
            conn = unpaired[0]
            issue_id = f"DANGLING::{ref_type}::{ref_value}::{conn['doc_id']}"
            merge_issues.append(
                MergeIssue(
                    issue_id=issue_id,
                    type="dangling_connector",
                    merge_key=key,
                    sheets_involved=[conn["doc_id"]],
                    connectors=[conn],
                    resolution="pending_human_review",
                )
            )
            _stats(conn["doc_id"])["dangling"].append(ref_value)

        elif len(unique_sheets) > 2:
            # Ambiguous: >2 sheets
            issue_id = f"AMBIGUOUS_MERGE::{ref_type}::{ref_value}"
            merge_issues.append(
                MergeIssue(
                    issue_id=issue_id,
                    type="ambiguous_merge",
                    merge_key=key,
                    sheets_involved=sorted(list(unique_sheets)),
                    connectors=unpaired,
                    resolution="pending_human_review",
                )
            )
            for conn in unpaired:
                _stats(conn["doc_id"])["dangling"].append(ref_value)

        elif len(unpaired) == 2:
            # Same sheet or direction conflict
            a, b = unpaired
            if a["doc_id"] == b["doc_id"]:
                issue_type: IssueType = "intra_sheet_duplicate"
                issue_id = f"INTRA_SHEET_DUP::{ref_type}::{ref_value}::{a['doc_id']}"
            else:
                issue_type = "direction_conflict"
                issue_id = f"DIRECTION_CONFLICT::{ref_type}::{ref_value}"

            merge_issues.append(
                MergeIssue(
                    issue_id=issue_id,
                    type=issue_type,
                    merge_key=key,
                    sheets_involved=sorted(list(unique_sheets)),
                    connectors=unpaired,
                    resolution="pending_human_review",
                )
            )
            for conn in unpaired:
                _stats(conn["doc_id"])["dangling"].append(ref_value)

    per_sheet_resolved = [
        SheetResolved(
            doc_id=doc_id,
            resolved_count=len(stats["resolved"]),
            dangling_count=len(stats["dangling"]),
            resolved_references=stats["resolved"],
            dangling_references=stats["dangling"],
        )
        for doc_id, stats in sorted(sheet_stats.items())
    ]

    return MergeResult(
        schema_version="graph_v2",
        cross_sheet_edges=cross_sheet_edges,
        merge_issues=merge_issues,
        per_sheet_resolved=per_sheet_resolved,
    )
