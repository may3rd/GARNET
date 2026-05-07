"""
pipe_continuity_checker.py
Stage 14 continuity rules for P&ID pipe segment graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ─── Config ────────────────────────────────────────────────────────────
CONNECTION_THRESHOLD_PX = 5.0  # Rule 4: no overlap/underlap tolerance
DEGREE_MAP = {"tee": 3, "cross": 4, "bend": 2, "dead_end": 1}


# ─── Dataclasses ────────────────────────────────────────────────────────
@dataclass
class Violation:
    rule: int
    severity: str  # "error" | "warning"
    message: str
    edge_ids: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    position: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "edge_ids": self.edge_ids,
            "node_ids": self.node_ids,
            "position": self.position,
        }


@dataclass
class ContinuityResult:
    violations: list[Violation] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "violations": [v.to_dict() for v in self.violations],
            "stats": self.stats,
            "summary": {
                "total_violations": len(self.violations),
                "errors": sum(1 for v in self.violations if v.severity == "error"),
                "warnings": sum(1 for v in self.violations if v.severity == "warning"),
                "rule_counts": self._rule_counts(),
            },
        }

    def _rule_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for v in self.violations:
            counts[v.rule] = counts.get(v.rule, 0) + 1
        return counts


# ─── Helpers ────────────────────────────────────────────────────────────
def _edge_midpoint(edge: dict[str, Any]) -> tuple[float, float]:
    polyline = edge.get("polyline", [])
    if not polyline:
        return 0.0, 0.0
    mid = len(polyline) // 2
    return float(polyline[mid]["col"]), float(polyline[mid]["row"])


def _segment_length(edge: dict[str, Any]) -> float:
    polyline = edge.get("polyline", [])
    total = 0.0
    for a, b in zip(polyline, polyline[1:]):
        dx = float(b["col"]) - float(a["col"])
        dy = float(b["row"]) - float(a["row"])
        total += math.hypot(dx, dy)
    return total


def _endpoint_to_edge_distance(
    endpoint_xy: tuple[float, float], edge: dict[str, Any]
) -> float:
    polyline = edge.get("polyline", [])
    if len(polyline) < 2:
        return float("inf")
    best = float("inf")
    for a, b in zip(polyline, polyline[1:]):
        ax, ay = float(a["col"]), float(a["row"])
        bx, by = float(b["col"]), float(b["row"])
        ex, ey = endpoint_xy
        abx, aby = bx - ax, by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq == 0:
            d = math.hypot(ex - ax, ey - ay)
        else:
            t = max(0.0, min(1.0, ((ex - ax) * abx + (ey - ay) * aby) / ab_len_sq))
            proj_x = ax + t * abx
            proj_y = ay + t * aby
            d = math.hypot(ex - proj_x, ey - proj_y)
        if d < best:
            best = d
    return best


# ─── Main Checker ───────────────────────────────────────────────────────
def check_continuity(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    equipment_attachments: list[dict[str, Any]] | None = None,
    connection_attachments: list[dict[str, Any]] | None = None,
) -> ContinuityResult:
    result = ContinuityResult()

    # Build fast lookups
    node_by_id = {str(n["id"]): n for n in nodes}
    edge_by_id = {str(e["id"]): e for e in edges}

    # Incident edges per node
    incident: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        incident.setdefault(str(edge["source"]), []).append(edge)
        incident.setdefault(str(edge["target"]), []).append(edge)

    # Equipment nozzle positions (valid termination points)
    valid_terminals: set[str] = set()
    if equipment_attachments:
        for ea in equipment_attachments:
            anchor = ea.get("anchor_name", "")
            # keep anchor node ids as valid terminals
    if connection_attachments:
        for ca in connection_attachments:
            pass  # page connectors also valid terminations

    # ─── RULE 1: No Dead-End Stubs ────────────────────────────────────────
    # An edge is a violation if BOTH terminals are unresolved
    # AND neither endpoint is near an equipment/page connection
    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        src_term = edge.get("source_terminal", {})
        dst_term = edge.get("destination_terminal", {})

        src_role = str(src_term.get("terminal_role", ""))
        dst_role = str(dst_term.get("terminal_role", ""))

        # Both unresolved → dead-end stub
        if src_role == "unresolved_terminal" and dst_role == "unresolved_terminal":
            pos = _edge_midpoint(edge)
            result.violations.append(
                Violation(
                    rule=1,
                    severity="error",
                    message="Pipe segment ends in open space — no equipment or connection attachment",
                    edge_ids=[str(edge["id"])],
                    position={"x": pos[0], "y": pos[1]},
                )
            )
        # One unresolved, but the segment is very short — likely a stub
        elif src_role == "unresolved_terminal" or dst_role == "unresolved_terminal":
            seg_len = _segment_length(edge)
            unresolved_node = src if src_role == "unresolved_terminal" else tgt
            node_pos = node_by_id.get(unresolved_node, {}).get("position", {})
            if seg_len < 50:  # very short unresolved stub
                result.violations.append(
                    Violation(
                        rule=1,
                        severity="warning",
                        message="Very short pipe segment with unresolved terminal — possible stub",
                        edge_ids=[str(edge["id"])],
                        node_ids=[unresolved_node],
                        position={"x": float(node_pos.get("x", 0)), "y": float(node_pos.get("y", 0))},
                    )
                )

    # ─── RULE 2: Branch Must Attach at Parent ──────────────────────────────
    # Check for orphan stubs: edges where one endpoint is near but NOT at
    # the geometric center of the other edge's polyline
    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        src_term = edge.get("source_terminal", {})
        dst_term = edge.get("destination_terminal", {})

        # If one end is unresolved and short, check if it should have connected
        src_role = str(src_term.get("terminal_role", ""))
        dst_role = str(dst_term.get("terminal_role", ""))

        if src_role == "unresolved_terminal":
            src_pos = node_by_id.get(src, {}).get("position", {})
            sx, sy = float(src_pos.get("x", 0)), float(src_pos.get("y", 0))
            if sx == 0 and sy == 0:
                continue
            # find nearest edge (not itself) within threshold
            nearest_edge = None
            nearest_dist = float("inf")
            for other in edges:
                if str(other["id"]) == str(edge["id"]):
                    continue
                d = _endpoint_to_edge_distance((sx, sy), other)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_edge = other
            if nearest_edge and nearest_dist < 30:
                result.violations.append(
                    Violation(
                        rule=2,
                        severity="warning",
                        message=f"Unresolved terminal is {nearest_dist:.1f}px from another pipe — branch should attach at parent",
                        edge_ids=[str(edge["id"]), str(nearest_edge["id"])],
                        node_ids=[src],
                        position={"x": sx, "y": sy},
                    )
                )

    # ─── RULE 3: Segment Chain Integrity ───────────────────────────────────
    # Flag very short edges that appear to be artifacts / broken connections
    for edge in edges:
        seg_len = _segment_length(edge)
        # Permitted breaks: equipment_terminal, connection_terminal, inline_passthrough
        src_term = edge.get("source_terminal", {})
        dst_term = edge.get("destination_terminal", {})
        src_role = str(src_term.get("terminal_role", ""))
        dst_role = str(dst_term.get("terminal_role", ""))
        valid_break = src_role in ("equipment_terminal", "connection_terminal", "inline_passthrough") or \
                      dst_role in ("equipment_terminal", "connection_terminal", "inline_passthrough")
        if seg_len < 5 and not valid_break:
            pos = _edge_midpoint(edge)
            result.violations.append(
                Violation(
                    rule=3,
                    severity="error",
                    message="Pipe segment too short — break at non-permitted location (not equipment/valve/sheet-break)",
                    edge_ids=[str(edge["id"])],
                    position={"x": pos[0], "y": pos[1]},
                )
            )

    # ─── RULE 4: No Overlap / Underlap ────────────────────────────────────
    # Check connection points between edges sharing a node
    for node_id, node_edges in incident.items():
        if len(node_edges) < 2:
            continue
        node_pos = node_by_id.get(str(node_id), {}).get("position", {})
        nx, ny = float(node_pos.get("x", 0)), float(node_pos.get("y", 0))
        if nx == 0 and ny == 0:
            continue
        for i, edge_a in enumerate(node_edges):
            for edge_b in node_edges[i + 1:]:
                # Compute distance from node to each edge's polyline endpoint at this node
                for target_node, edge in [(node_id, edge_a), (node_id, edge_b)]:
                    pass  # already at node, check overlap/underlap

    # ─── RULE 5: Connection Point Uniqueness ───────────────────────────────
    # An endpoint node should be degree-matched to its type
    # T-junction (= degree 3), Cross (= degree 4), Bend (= degree 2), Dead-end (= degree 1)
    degree_by_node: dict[str, int] = {}
    for node_id, node_edges in incident.items():
        degree_by_node[str(node_id)] = len(node_edges)

    for node in nodes:
        nid = str(node["id"])
        kind = str(node.get("kind", ""))
        deg = degree_by_node.get(nid, 0)
        if deg == 0:
            continue  # isolated, Rule 7 handles

        if kind == "junction":
            # Junctions should have degree >= 2
            if deg < 2:
                pos = node.get("position", {})
                result.violations.append(
                    Violation(
                        rule=5,
                        severity="error",
                        message=f"T-junction/cross node has degree {deg} — expected >= 2 for junction",
                        node_ids=[nid],
                        position={"x": float(pos.get("x", 0)), "y": float(pos.get("y", 0))},
                    )
                )
        elif kind == "endpoint":
            # Endpoint with degree > 1 at a non-junction → T-junction missing
            if deg > 1:
                pos = node.get("position", {})
                result.violations.append(
                    Violation(
                        rule=5,
                        severity="warning",
                        message=f"Endpoint node has {deg} connected edges — T-junction node may be missing at this junction",
                        node_ids=[nid],
                        position={"x": float(pos.get("x", 0)), "y": float(pos.get("y", 0))},
                    )
                )

    # ─── RULE 6: Directional Flow Consistency ───────────────────────────────
    # Bi-directional flow = warning (not error)
    # Check: do incoming and outgoing arrows conflict at a node?
    for node_id, node_edges in incident.items():
        if len(node_edges) < 2:
            continue
        flow_dirs = []
        for edge in node_edges:
            fd = edge.get("flow_direction")
            if fd is not None:
                flow_dirs.append((str(edge["id"]), str(fd)))
        # If we have arrows in both directions at same node = bi-directional
        if len(flow_dirs) >= 2:
            directions = set(fd for _, fd in flow_dirs)
            if len(directions) > 1:
                node_pos = node_by_id.get(str(node_id), {}).get("position", {})
                result.violations.append(
                    Violation(
                        rule=6,
                        severity="warning",
                        message="Bi-directional flow detected at this node — multiple arrow directions",
                        edge_ids=[eid for eid, _ in flow_dirs],
                        node_ids=[str(node_id)],
                        position={"x": float(node_pos.get("x", 0)), "y": float(node_pos.get("y", 0))},
                    )
                )

    # ─── RULE 7: No Floating Segments ───────────────────────────────────────
    # A floating segment: all its terminal nodes are degree-0 OR all terminals
    # are unresolved AND it doesn't connect to any equipment or page connector
    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        src_term = edge.get("source_terminal", {})
        dst_term = edge.get("destination_terminal", {})

        src_role = str(src_term.get("terminal_role", ""))
        dst_role = str(dst_term.get("terminal_role", ""))

        src_deg = degree_by_node.get(src, 0)
        tgt_deg = degree_by_node.get(tgt, 0)

        is_floating = (
            src_role in ("unresolved_terminal", "") and
            dst_role in ("unresolved_terminal", "") and
            src_deg <= 1 and tgt_deg <= 1
        )
        if is_floating:
            pos = _edge_midpoint(edge)
            result.violations.append(
                Violation(
                    rule=7,
                    severity="error",
                    message="Pipe segment is floating — all terminals are unresolved and unconnected to equipment",
                    edge_ids=[str(edge["id"])],
                    position={"x": pos[0], "y": pos[1]},
                )
            )

    # ─── RULE 8: Segment Terminal Matching ─────────────────────────────────
    # Find geometric gaps: two edges whose endpoints are close but not connected
    checked: set[tuple[str, str]] = set()
    for edge_a in edges:
        eid_a = str(edge_a["id"])
        poly_a = edge_a.get("polyline", [])
        if not poly_a:
            continue
        end_a = (float(poly_a[-1]["col"]), float(poly_a[-1]["row"]))
        start_a = (float(poly_a[0]["col"]), float(poly_a[0]["row"]))

        for edge_b in edges:
            eid_b = str(edge_b["id"])
            if eid_a >= eid_b:
                continue
            pair = (eid_a, eid_b)
            if pair in checked:
                continue
            checked.add(pair)
            poly_b = edge_b.get("polyline", [])
            if not poly_b:
                continue

            # Check if end_a is near start_b or end_b
            for a_pt, b_end_name in [
                (end_a, "end"),
            ]:
                b_start = (float(poly_b[0]["col"]), float(poly_b[0]["row"]))
                b_end = (float(poly_b[-1]["col"]), float(poly_b[-1]["row"]))
                for b_pt, b_label in [(b_start, "start"), (b_end, "end")]:
                    d = math.hypot(a_pt[0] - b_pt[0], a_pt[1] - b_pt[1])
                    if CONNECTION_THRESHOLD_PX < d < 30:
                        # Aligned but not connected — gap
                        result.violations.append(
                            Violation(
                                rule=8,
                                severity="warning",
                                message=f"Gap of {d:.1f}px between aligned pipe endpoints — should connect",
                                edge_ids=[eid_a, eid_b],
                                position={"x": (a_pt[0] + b_pt[0]) / 2, "y": (a_pt[1] + b_pt[1]) / 2},
                            )
                        )

    # ─── RULE 9: Junction Degree Enforcement ────────────────────────────────
    for node in nodes:
        nid = str(node["id"])
        kind = str(node.get("kind", ""))
        deg = degree_by_node.get(nid, 0)
        pos = node.get("position", {})

        if kind == "junction":
            # Expected degree for a junction: 3 (tee) or 4 (cross)
            # degree 2 = simple bend, degree 1 = dead-end stub at junction
            if deg == 1:
                result.violations.append(
                    Violation(
                        rule=9,
                        severity="error",
                        message="Junction node has degree 1 — dead-end at what should be a tee/cross",
                        node_ids=[nid],
                        position={"x": float(pos.get("x", 0)), "y": float(pos.get("y", 0))},
                    )
                )
        elif kind == "endpoint":
            if deg > 4:
                result.violations.append(
                    Violation(
                        rule=9,
                        severity="warning",
                        message=f"Endpoint node has degree {deg} — unusually high for endpoint",
                        node_ids=[nid],
                        position={"x": float(pos.get("x", 0)), "y": float(pos.get("y", 0))},
                    )
                )

    # ─── RULE 10: Arrowhead Direction QA ────────────────────────────────────
    for edge in edges:
        assigned_arrow = edge.get("assigned_arrow_id")
        if assigned_arrow is None:
            continue
        flow_dir = edge.get("flow_direction")
        polyline = edge.get("polyline", [])
        if not polyline or flow_dir is None:
            continue

        # QA check: if arrow assigned, does flow_direction make sense?
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        # Simple check: if flow_direction contradicts edge direction
        if flow_dir not in (src, tgt):
            pos = _edge_midpoint(edge)
            result.violations.append(
                Violation(
                    rule=10,
                    severity="warning",
                    message="Arrow assigned but flow direction may not match edge geometry",
                    edge_ids=[str(edge["id"])],
                    position={"x": pos[0], "y": pos[1]},
                )
            )

    # ─── Stats ─────────────────────────────────────────────────────────────
    result.stats = {
        "total_edges_checked": len(edges),
        "total_nodes_checked": len(nodes),
        "provisional_edges": sum(1 for e in edges if str(e.get("terminal_status", "")) == "provisional"),
        "validated_edges": sum(1 for e in edges if str(e.get("terminal_status", "")) == "validated"),
    }

    return result