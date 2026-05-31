"""
continuity_aware_connections.py
Post-processing for Stage 12 edge connectivity — validates and enriches
connection decisions using Stage 10 continuity data (gap_summary + continuity_result).
Part of Phase 2.

Implements:
  - Gap connection validation: gaps from Stage 10 that Stage 12 missed
  - Orphan branch detection: branches that should attach to parent but didn't
  - Terminal role enrichment using Stage 10 continuity metadata
"""
from __future__ import annotations

from typing import Any


def validate_connections_against_gaps(
    edges: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    gap_summary: list[dict[str, Any]],
    *,
    gap_threshold_px: float = 20.0,
) -> dict[str, Any]:
    """
    Compare Stage 12 connections against Stage 10 gap_summary.
    Report: (a) gaps that Stage 12 correctly connected, (b) gaps that were
    missed, (c) connections made to gap-proximate edges that weren't in gap_summary.

    Stage 10's gap_summary is the ground-truth for geometric alignment gaps.
    Stage 12's _continuation_connections already handles most gaps, but
    Stage 10's near-edge detection may find additional candidates that
    _continuation_connections missed.

    Returns:
        - connected_gaps: gaps that have a Stage 12 connection
        - missed_gaps: gaps with no Stage 12 connection (recommend adding)
        - extra_connections: Stage 12 connections that bridge gaps not in gap_summary
    """
    if not gap_summary:
        return {
            "connected_gaps": [],
            "missed_gaps": [],
            "extra_connections": [],
            "gap_connection_summary": {"total": 0, "connected": 0, "missed": 0},
        }

    # Build a set of (edge_a, edge_b) pairs that are connected by Stage 12
    connected_pairs: set[frozenset[str]] = set()
    for conn in connections:
        pair = frozenset((str(conn.get("source_edge_id", "")), str(conn.get("target_edge_id", ""))))
        connected_pairs.add(pair)

    connected_gaps: list[dict[str, Any]] = []
    missed_gaps: list[dict[str, Any]] = []

    for gap in gap_summary:
        edge_a = str(gap.get("edge_a", ""))
        edge_b = str(gap.get("edge_b", ""))
        gap_pair = frozenset((edge_a, edge_b))

        if gap_pair in connected_pairs:
            connected_gaps.append(gap)
        else:
            missed_gaps.append(gap)

    # Extra connections: Stage 12 made connections not predicted by gap_summary
    gap_edge_pairs: set[frozenset[str]] = set()
    for gap in gap_summary:
        gap_edge_pairs.add(frozenset((str(gap.get("edge_a", "")), str(gap.get("edge_b", "")))))

    extra_connections: list[dict[str, Any]] = []
    for conn in connections:
        pair = frozenset((str(conn.get("source_edge_id", "")), str(conn.get("target_edge_id", ""))))
        if pair not in gap_edge_pairs:
            extra_connections.append(conn)

    return {
        "connected_gaps": connected_gaps,
        "missed_gaps": missed_gaps,
        "extra_connections": extra_connections,
        "gap_connection_summary": {
            "total_gaps_in_summary": len(gap_summary),
            "connected_by_stage12": len(connected_gaps),
            "missed_by_stage12": len(missed_gaps),
            "extra_connections_made": len(extra_connections),
            "gap_coverage_pct": round(len(connected_gaps) / len(gap_summary) * 100, 1) if gap_summary else 100.0,
        },
    }


def enrich_edge_terminals_with_continuity(
    edges: list[dict[str, Any]],
    continuity_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Enrich edge terminal records with Stage 10 continuity metadata.
    This makes downstream stages (S12, S13, S14) aware of orphan/gap status.

    continuity_result is the output from run_post_trace_continuity_check() in pipe_edges.py.
    It contains orphan_edges, gap_candidate_edges, near_edge_candidates.

    For each edge, adds:
      - orphan_flag: bool
      - gap_candidate: bool
      - near_edge_candidate_count: int
      - continuity_status: "validated" | "provisional"

    Returns enriched edge terminal records (list of dicts, one per edge).
    """
    edge_map: dict[str, dict[str, Any]] = {}
    for edge in edges:
        eid = str(edge.get("id", ""))
        edge_map[eid] = edge

    # Build continuity status from Stage 10 metadata
    near_candidates_by_edge: dict[str, list[dict[str, Any]]] = {}
    orphan_edge_ids: set[str] = set()
    gap_candidate_edge_ids: set[str] = set()
    provisional_edge_ids: set[str] = set()
    validated_edge_ids: set[str] = set()

    for edge in edges:
        eid = str(edge.get("id", ""))
        status = str(edge.get("continuity_status", "provisional"))
        orphan = bool(edge.get("orphan_flag", False))
        gap_cand = bool(edge.get("gap_candidate", False))
        near = edge.get("near_edge_candidates", [])

        near_candidates_by_edge[eid] = near or []
        if orphan:
            orphan_edge_ids.add(eid)
        if gap_cand:
            gap_candidate_edge_ids.add(eid)
        if status == "validated":
            validated_edge_ids.add(eid)
        else:
            provisional_edge_ids.add(eid)

    # Enrich edge terminal records
    enriched_terminals: list[dict[str, Any]] = []
    for edge in edges:
        eid = str(edge.get("id", ""))
        enriched_terminals.append({
            "edge_id": eid,
            "source": str(edge.get("source", "")),
            "target": str(edge.get("target", "")),
            "orphan_flag": eid in orphan_edge_ids,
            "gap_candidate": eid in gap_candidate_edge_ids,
            "near_edge_candidate_count": len(near_candidates_by_edge.get(eid, [])),
            "near_edge_candidates": near_candidates_by_edge.get(eid, []),
            "continuity_status": "validated" if eid in validated_edge_ids else "provisional",
            "polyline_length_approx": _approx_edge_length(edge),
        })

    return enriched_terminals


def _approx_edge_length(edge: dict[str, Any]) -> int:
    import math
    polyline = edge.get("polyline", [])
    if not polyline:
        return 0
    total = 0.0
    for a, b in zip(polyline, polyline[1:]):
        dx = float(b["col"]) - float(a["col"])
        dy = float(b["row"]) - float(a["row"])
        total += math.hypot(dx, dy)
    return int(total)


def merge_continuity_into_graph(
    edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    continuity_result: dict[str, Any],
    gap_summary: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Merge Stage 10 continuity data into the graph before Stage 12 assembly.
    This ensures that Stage 12's edge_connectivity uses the most complete
    geometric awareness available.

    Returns enriched (edges, nodes) with continuity metadata embedded.
    """
    enriched_edges = []
    for edge in edges:
        eid = str(edge.get("id", ""))

        # Mark orphan/gap edges
        is_orphan = bool(edge.get("orphan_flag", False))
        is_gap = bool(edge.get("gap_candidate", False))
        near_candidates = edge.get("near_edge_candidates", [])

        enriched = {**edge}
        enriched["_continuity"] = {
            "orphan": is_orphan,
            "gap_candidate": is_gap,
            "near_edge_count": len(near_candidates),
            "near_edge_candidates": near_candidates,
            "status": str(edge.get("continuity_status", "provisional")),
        }

        # Add continuation_flag: this edge is part of a gap that should connect
        # Stage 12 will use this to prioritize its continuation connection logic
        if is_gap and near_candidates:
            enriched["_continuation_priority"] = min(
                c.get("distance_px", 999) for c in near_candidates
            )

        enriched_edges.append(enriched)

    # Nodes: mark nodes that are part of orphan edges
    orphan_edge_ids = {str(ee.get("id", "")) for ee in edges if ee.get("orphan_flag")}
    orphan_node_ids: set[str] = set()
    for edge in edges:
        if str(edge.get("id", "")) in orphan_edge_ids:
            orphan_node_ids.add(str(edge.get("source", "")))
            orphan_node_ids.add(str(edge.get("target", "")))

    enriched_nodes = []
    for node in nodes:
        nid = str(node.get("id", ""))
        enriched_nodes.append({
            **node,
            "_continuity": {
                "part_of_orphan_edge": nid in orphan_node_ids,
                "orphan_edge_count": sum(
                    1 for e in edges
                    if str(e.get("id", "")) in orphan_edge_ids
                    and nid in (str(e.get("source", "")), str(e.get("target", "")))
                ),
            },
        })

    return enriched_edges, enriched_nodes


def generate_continuity_review_items(
    edges: list[dict[str, Any]],
    gap_summary: list[dict[str, Any]],
    orphan_threshold_px: float = 50.0,
) -> list[dict[str, Any]]:
    """
    Generate review queue items specifically from Stage 10 continuity data
    that Stage 14's continuity checker can merge into its unified review queue.

    These are the high-priority items that Stage 10 detected but Stage 12
    may not have resolved automatically.
    """
    items: list[dict[str, Any]] = []

    # Orphan edges
    for edge in edges:
        if not edge.get("orphan_flag"):
            continue
        polyline = edge.get("polyline", [])
        mid_x = float(polyline[len(polyline) // 2]["col"]) if polyline else 0.0
        mid_y = float(polyline[len(polyline) // 2]["row"]) if polyline else 0.0
        items.append({
            "category": "orphan_branch_from_s10",
            "rule": 1,
            "priority": "high",
            "edge_id": str(edge.get("id", "")),
            "source": str(edge.get("source", "")),
            "target": str(edge.get("target", "")),
            "geometry": {"x": round(mid_x, 1), "y": round(mid_y, 1)},
            "near_edge_candidates": edge.get("near_edge_candidates", []),
            "polyline_length_approx": _approx_edge_length(edge),
            "continuity_status": str(edge.get("continuity_status", "provisional")),
        })

    # Gap candidates with no Stage 12 connection yet
    connected_edge_pairs: set[frozenset[str]] = set()  # populated externally by caller
    for gap in gap_summary:
        edge_a = str(gap.get("edge_a", ""))
        edge_b = str(gap.get("edge_b", ""))
        pair = frozenset((edge_a, edge_b))
        if pair in connected_edge_pairs:
            continue
        items.append({
            "category": "unresolved_gap_from_s10",
            "rule": 8,
            "priority": "medium",
            "edge_a": edge_a,
            "edge_b": edge_b,
            "gap_position": gap.get("gap_position", {}),
            "gap_distance_px": gap.get("gap_distance_px", 0),
            "alignment": gap.get("alignment", "unknown"),
        })

    return items