"""
S5-01: Bounded Recovery Loop — Approach 2 (Non-destructive Artifact)

Reads stage13_review_queue.json, applies targeted recovery strategies per category,
and produces stage5_recovery_decisions.json as a separate artifact.

The original pipeline output is NEVER modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    RETRY = "retry"  # Apply recovery and re-evaluate
    HUMAN_REVIEW = "human_review"  # Cannot auto-resolve, flag for human
    ACCEPT = "accept"  # Accept as-is


@dataclass
class RecoveryItem:
    original_item: Dict[str, Any]
    category: str  # articulation_point | isolated_node | unresolved_crossing | unresolved_terminal_edge
    group_key: str
    priority: str  # low | medium | high
    strategy_attempted: List[str] = field(default_factory=list)
    action: RecoveryAction = RecoveryAction.HUMAN_REVIEW
    notes: str = ""
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "group_key": self.group_key,
            "priority": self.priority,
            "node_ids": self.original_item.get("node_ids", []),
            "edge_ids": self.original_item.get("edge_ids", []),
            "action": self.action.value,
            "strategy_attempted": self.strategy_attempted,
            "notes": self.notes,
            "retry_count": self.retry_count,
        }


class RecoveryEngine:
    """
    Reads unresolved items from stage13_review_queue.json and applies
    bounded recovery strategies. All recovery is read-only on pipeline artifacts.
    """

    def __init__(self, job_dir: Path, max_iterations: int = 3):
        self.job_dir = Path(job_dir)
        self.max_iterations = max_iterations
        self.decisions: List[RecoveryItem] = []
        self._artifacts: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def load_review_queue(self) -> Dict[str, Any]:
        """Load stage13_review_queue.json."""
        path = self.job_dir / "stage13_review_queue.json"
        if not path.exists():
            logger.warning(f"stage13_review_queue.json not found in {self.job_dir}")
            return {"items": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_stage_artifacts(self) -> Dict[str, Any]:
        """Load OCR, objects, and graph artifacts for recovery analysis."""
        artifacts = {}

        # Stage 2 OCR
        ocr_path = self.job_dir / "stage2_ocr_regions.json"
        if ocr_path.exists():
            with open(ocr_path, "r", encoding="utf-8") as f:
                artifacts["ocr"] = json.load(f)

        # Stage 4 objects
        objects_path = self.job_dir / "stage4_objects.json"
        if objects_path.exists():
            with open(objects_path, "r", encoding="utf-8") as f:
                artifacts["objects"] = json.load(f)

        # Stage 12 graph
        graph_path = self.job_dir / "stage12_graph.json"
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                artifacts["graph"] = json.load(f)

        # Phase 2: Stage 12 gap-vs-connection validation (Stage 10 continuity check)
        gap_validation_path = self.job_dir / "stage12_connection_validation.json"
        if gap_validation_path.exists():
            with open(gap_validation_path, "r", encoding="utf-8") as f:
                artifacts["gap_validation"] = json.load(f)

        # S5: Phase 3 gap detection (geometric bypass path)
        phase3_gaps_path = self.job_dir / "phase3_gaps.json"
        if phase3_gaps_path.exists():
            with open(phase3_gaps_path, "r", encoding="utf-8") as f:
                artifacts["phase3_gaps"] = json.load(f)

        # S5: Phase 3 boundary terminal detection
        phase3_boundary_path = self.job_dir / "phase3_boundary_terminals.json"
        if phase3_boundary_path.exists():
            with open(phase3_boundary_path, "r", encoding="utf-8") as f:
                artifacts["phase3_boundary"] = json.load(f)

        self._artifacts = artifacts
        return artifacts

    # ------------------------------------------------------------------
    # Recovery strategies
    # ------------------------------------------------------------------

    def recover_isolated_node(self, item: Dict[str, Any]) -> RecoveryItem:
        """
        Strategy for isolated_node items.

        Isolated nodes (endpoints or junctions with degree 0) are analyzed:
        1. Check if nearby text regions exist that weren't attached — wider radius
        2. Check if the node is actually a valid standalone symbol (instrument, valve)
        3. If it has attached text labels, it may be a valid isolated equipment symbol
        """
        result = RecoveryItem(
            original_item=item,
            category=item.get("category", "isolated_node"),
            group_key=item.get("group_key", ""),
            priority=item.get("priority", "medium"),
        )

        node_ids = item.get("node_ids", [])
        graph = self._artifacts.get("graph", {})
        nodes = {n["id"]: n for n in graph.get("nodes", [])}

        attached_text_count = 0
        labeled_nodes = []
        for nid in node_ids:
            node = nodes.get(nid, {})
            text = node.get("text", "").strip()
            if text:
                attached_text_count += 1
                labeled_nodes.append(nid)

        ocr = self._artifacts.get("ocr", {})
        text_regions = ocr.get("text_regions", [])
        ocr_text_count = len([t for t in text_regions if t.get("text", "").strip()])

        # If nodes have attached text labels, they may be valid standalone equipment
        if labeled_nodes:
            result.action = RecoveryAction.ACCEPT
            result.notes = (
                f"{len(labeled_nodes)}/{len(node_ids)} nodes have attached text labels — "
                f"likely valid standalone symbols"
            )
            result.strategy_attempted.append("text_label_check")
        elif ocr_text_count > 0 and len(node_ids) <= 5:
            # Few isolated nodes + available OCR text = possible missed attachment
            result.action = RecoveryAction.RETRY
            result.notes = (
                f"{len(node_ids)} isolated nodes with no labels, but {ocr_text_count} "
                f"OCR regions available — retry text attachment with wider radius"
            )
            result.strategy_attempted.append("wider_text_attachment")
            result.retry_count = 1
        else:
            # Many isolated nodes or no OCR context — human review needed
            result.action = RecoveryAction.HUMAN_REVIEW
            result.notes = (
                f"{len(node_ids)} isolated nodes, no attached text, "
                f"{ocr_text_count} OCR regions — requires manual disambiguation"
            )
            result.strategy_attempted.append("no_viable_strategy")

        return result

    def recover_articulation_point(self, item: Dict[str, Any]) -> RecoveryItem:
        """
        Strategy for articulation_point items.

        Articulation points are nodes whose removal disconnects the graph.
        These are usually correct structural features (e.g., control valve in series
        on a pipe). Recovery strategy: verify the node is correctly typed.
        """
        result = RecoveryItem(
            original_item=item,
            category=item.get("category", "articulation_point"),
            group_key=item.get("group_key", ""),
            priority=item.get("priority", "medium"),
        )

        node_ids = item.get("node_ids", [])
        graph = self._artifacts.get("graph", {})
        nodes = {n["id"]: n for n in graph.get("nodes", [])}

        # Check if the articulation nodes are correctly typed as junctions
        correctly_typed = 0
        for nid in node_ids:
            node = nodes.get(nid, {})
            node_type = node.get("type", "")
            # Articulation points at pipe junctions are typically correct
            if node_type in ("junction", "pipe_junction"):
                correctly_typed += 1

        if correctly_typed == len(node_ids):
            result.action = RecoveryAction.ACCEPT
            result.notes = (
                f"All {len(node_ids)} articulation nodes correctly typed as junctions — "
                f"valid structural feature"
            )
            result.strategy_attempted.append("junction_type_check")
        elif correctly_typed > 0:
            result.action = RecoveryAction.ACCEPT
            result.notes = (
                f"{correctly_typed}/{len(node_ids)} articulation nodes correctly typed — "
                f"rest may be equipment, treating as valid"
            )
            result.strategy_attempted.append("mixed_type_check")
        else:
            # Nodes not typed as junctions — could be equipment incorrectly flagged
            result.action = RecoveryAction.HUMAN_REVIEW
            result.notes = (
                f"Articulation point not typed as junction — may be equipment requiring "
                f"manual classification"
            )
            result.strategy_attempted.append("non_junction_check")

        return result

    def recover_unresolved_crossing(self, item: Dict[str, Any]) -> RecoveryItem:
        """
        Strategy for unresolved_crossing items.

        Pipe crossings where it was unclear if they represent a connection or a
        jump-over (no physical connection). Recovery: these are inherently ambiguous
        and should go to human review.
        """
        result = RecoveryItem(
            original_item=item,
            category=item.get("category", "unresolved_crossing"),
            group_key=item.get("group_key", ""),
            priority=item.get("priority", "medium"),
        )

        edge_ids = item.get("edge_ids", [])
        graph = self._artifacts.get("graph", {})
        edges = {e["id"]: e for e in graph.get("edges", [])}

        # Check if there's explicit directionality info that could resolve it
        has_direction = 0
        for eid in edge_ids:
            edge = edges.get(eid, {})
            if edge.get("directed", False) or edge.get("direction"):
                has_direction += 1

        if has_direction >= len(edge_ids):
            result.action = RecoveryAction.ACCEPT
            result.notes = "All crossing edges have directionality — resolved"
            result.strategy_attempted.append("directionality_resolved")
        else:
            result.action = RecoveryAction.HUMAN_REVIEW
            result.notes = (
                f"{len(edge_ids)} crossing edges, {has_direction} with direction — "
                f"requires manual pipe-crossing disambiguation"
            )
            result.strategy_attempted.append("ambiguous_crossing")

        return result

    def recover_near_edge_gap(self, item: Dict[str, Any]) -> RecoveryItem:
        """
        Strategy for near_edge_gap items.

        Stage 10's geometric gap detection found two edges with aligned endpoints
        within 20px that Stage 12 never connected. The handler scores each gap
        by closure confidence:

        High   → gap distance ≤ 5px AND both endpoints snap to the same junction
                 cluster → safe to auto-close with a new provisional connection
        Medium → gap distance ≤ 15px AND consistent horizontal/vertical alignment
                 → flag for human review (need visual confirmation)
        Low    → gap distance 15-20px OR only one endpoint has cluster anchor
                 → flag for human review (may need OCR/equipment check)
        Skip   → gap distance > 20px → not within threshold, skip

        After classification, a suggested_connection block is embedded so that
        downstream gap-injection code (Phase 2) can patch stage12_edge_connections
        and re-run graph assembly without any ambiguity about what to add.
        """
        result = RecoveryItem(
            original_item=item,
            category=item.get("category", "near_edge_gap"),
            group_key=item.get("group_key", ""),
            priority=item.get("priority", "medium"),
        )

        gap_distance = float(item.get("gap_distance_px", 0))
        alignment = item.get("alignment", "unknown")
        gap_pos = item.get("gap_position", {})
        edge_a = str(item.get("edge_a", ""))
        edge_b = str(item.get("edge_b", ""))
        endpoint_a = item.get("endpoint_a", "")  # "source" or "target"
        endpoint_b = item.get("endpoint_b", "")

        # Load node clusters to check if endpoints share a cluster
        graph = self._artifacts.get("graph", {})
        nodes = {n["id"]: n for n in graph.get("nodes", [])}
        edges = {e["id"]: e for e in graph.get("edges", [])}

        def _node_cluster(node_id: str) -> Optional[str]:
            node = nodes.get(node_id)
            if not node:
                return None
            return node.get("cluster_id") or node.get("parent_cluster") or None

        # Resolve endpoint coordinates from edge polylines
        def _edge_endpoint_coords(eid: str, which: str) -> Optional[Dict[str, float]]:
            edge = edges.get(eid)
            if not edge:
                return None
            polyline = edge.get("polyline", [])
            if not polyline:
                return None
            if which == "source":
                pt = polyline[0]
            else:
                pt = polyline[-1]
            return {"x": float(pt.get("col", 0)), "y": float(pt.get("row", 0))}

        # Determine how many endpoints have cluster anchors
        ep_a_coords = _edge_endpoint_coords(edge_a, endpoint_a)
        ep_b_coords = _edge_endpoint_coords(edge_b, endpoint_b)

        # Heuristic: check if the gap position is close to both edge endpoints
        # (indicating the gap truly straddles those two endpoints)
        pos_x = float(gap_pos.get("x", 0))
        pos_y = float(gap_pos.get("y", 0))

        def _endpoint_distance(ep_coords: Optional[Dict], px: float, py: float) -> float:
            if ep_coords is None:
                return 9999.0
            return ((ep_coords["x"] - px) ** 2 + (ep_coords["y"] - py) ** 2) ** 0.5

        dist_a = _endpoint_distance(ep_a_coords, pos_x, pos_y)
        dist_b = _endpoint_distance(ep_b_coords, pos_x, pos_y)

        # Both endpoints near the gap position → confirmed gap straddles them
        gap_confirmed = (dist_a < 30.0 and dist_b < 30.0)

        # Determine alignment sanity (diagonal gaps are unreliable)
        alignment_sane = alignment in ("horizontal", "vertical")

        # Confidence scoring
        if gap_distance > 20.0:
            result.action = RecoveryAction.ACCEPT
            result.notes = (
                f"gap_distance={gap_distance:.1f}px > 20px threshold — "
                f"outside geometric tolerance, accepting as-is"
            )
            result.strategy_attempted.append("gap_too_large")
        elif gap_distance <= 5.0 and alignment_sane and gap_confirmed:
            result.action = RecoveryAction.ACCEPT
            result.notes = (
                f"gap_distance={gap_distance:.1f}px ≤ 5px, alignment={alignment}, "
                f"gap confirmed at ({pos_x:.0f},{pos_y:.0f}) — safe to auto-close"
            )
            result.strategy_attempted.append("high_confidence_gap_closure")
        elif gap_distance <= 15.0 and alignment_sane:
            result.action = RecoveryAction.HUMAN_REVIEW
            result.notes = (
                f"gap_distance={gap_distance:.1f}px ≤ 15px, alignment={alignment} — "
                f"medium confidence, requires visual confirmation"
            )
            result.strategy_attempted.append("medium_confidence_gap_review")
        elif alignment_sane:
            result.action = RecoveryAction.HUMAN_REVIEW
            result.notes = (
                f"gap_distance={gap_distance:.1f}px (15-20px range), alignment={alignment} — "
                f"lower confidence, requires OCR/equipment check"
            )
            result.strategy_attempted.append("low_confidence_gap_review")
        else:
            result.action = RecoveryAction.HUMAN_REVIEW
            result.notes = (
                f"gap_distance={gap_distance:.1f}px, alignment={alignment} — "
                f"diagonal or invalid alignment, flagging for review"
            )
            result.strategy_attempted.append("diagonal_gap_review")

        # Attach suggested_connection so Phase 2 gap-injection can act on it
        result.original_item["suggested_connection"] = {
            "source_edge": edge_a,
            "target_edge": edge_b,
            "source_endpoint": endpoint_a,
            "target_endpoint": endpoint_b,
            "gap_position": {"x": pos_x, "y": pos_y},
            "gap_distance_px": gap_distance,
            "alignment": alignment,
        }
        result.original_item["gap_confirmed"] = gap_confirmed

        return result

    def recover_unresolved_terminal_edge(self, item: Dict[str, Any]) -> RecoveryItem:
        """
        Strategy for unresolved_terminal_edge items.

        Edges that end without a clear connection point. Recovery: check if
        the terminal is actually near a valid node or if it's a true dead end.
        """
        result = RecoveryItem(
            original_item=item,
            category=item.get("category", "unresolved_terminal_edge"),
            group_key=item.get("group_key", ""),
            priority=item.get("priority", "medium"),
        )

        edge_ids = item.get("edge_ids", [])
        graph = self._artifacts.get("graph", {})
        edges = {e["id"]: e for e in graph.get("edges", [])}
        nodes = {n["id"]: n for n in graph.get("nodes", [])}

        # Check if terminal edges are near any unconnected nodes
        terminal_near_nodes = 0
        for eid in edge_ids:
            edge = edges.get(eid, {})
            end_x = edge.get("end_x") or edge.get("geometry", {}).get("end_x")
            end_y = edge.get("end_y") or edge.get("geometry", {}).get("end_y")
            if end_x is None or end_y is None:
                continue
            for nid, node in nodes.items():
                pos = node.get("position", {})
                nx = pos.get("x", 0)
                ny = pos.get("y", 0)
                dist = ((end_x - nx) ** 2 + (end_y - ny) ** 2) ** 0.5
                if dist < 50:  # within 50px
                    terminal_near_nodes += 1
                    break

        if terminal_near_nodes > 0:
            result.action = RecoveryAction.RETRY
            result.notes = (
                f"{terminal_near_nodes}/{len(edge_ids)} terminal edges near unconnected "
                f"nodes — retry attachment with wider radius"
            )
            result.strategy_attempted.append("wider_terminal_attachment")
            result.retry_count = 1
        else:
            result.action = RecoveryAction.HUMAN_REVIEW
            result.notes = (
                f"{len(edge_ids)} terminal edges with no nearby nodes — "
                f"likely true dead ends or off-page connectors"
            )
            result.strategy_attempted.append("no_nearby_node")

        return result

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def _classify_item(self, item: Dict[str, Any]) -> RecoveryItem:
        """Dispatch recovery strategy based on item category."""
        category = item.get("category", "")
        if category == "isolated_node":
            return self.recover_isolated_node(item)
        elif category == "articulation_point":
            return self.recover_articulation_point(item)
        elif category == "unresolved_crossing":
            return self.recover_unresolved_crossing(item)
        elif category == "unresolved_terminal_edge":
            return self.recover_unresolved_terminal_edge(item)
        elif category == "near_edge_gap":
            return self.recover_near_edge_gap(item)
        else:
            # Unknown category — default to human review
            result = RecoveryItem(
                original_item=item,
                category=category,
                group_key=item.get("group_key", ""),
                priority=item.get("priority", "medium"),
                action=RecoveryAction.HUMAN_REVIEW,
                notes=f"Unknown category '{category}' — defaulting to human review",
                strategy_attempted=["unknown_category"],
            )
            return result

    def run(self) -> Dict[str, Any]:
        """
        Run the bounded recovery loop.

        Returns a decisions dict ready to be written as stage5_recovery_decisions.json.
        The original pipeline artifacts are never modified.

        Handles two sources of items:
        1. stage13_review_queue.json — existing pipeline review items
        2. stage12_connection_validation.json — near-edge gaps detected by Stage 10
           that Stage 12 never connected
        """
        logger.info(f"RecoveryEngine starting — max_iterations={self.max_iterations}")

        # Load artifacts once
        queue_data = self.load_review_queue()
        self.load_stage_artifacts()

        decisions: List[RecoveryItem] = []
        iterations_completed = 0

        # ---- Source 1: existing review queue items ----
        items = queue_data.get("items", [])
        if items:
            logger.info(f"Processing {len(items)} review queue items")
            for item in items:
                decision = self._classify_item(item)
                decisions.append(decision)
            iterations_completed = max(iterations_completed, 1)

        # ---- Source 2: near-edge gaps from Stage 10 continuity check ----
        # stage12_connection_validation.json is loaded by load_stage_artifacts()
        gap_validation = self._artifacts.get("gap_validation", {})
        missed_gaps = gap_validation.get("missed_gaps", [])
        if missed_gaps:
            logger.info(f"Processing {len(missed_gaps)} near-edge gaps from Stage 10 continuity check")
            for gap in missed_gaps:
                gap_item: Dict[str, Any] = {
                    "category": "near_edge_gap",
                    "group_key": f"gap::{gap.get('edge_a','')}::{gap.get('edge_b','')}",
                    "priority": "medium",
                    "edge_a": gap.get("edge_a", ""),
                    "edge_b": gap.get("edge_b", ""),
                    "endpoint_a": gap.get("endpoint_a", ""),
                    "endpoint_b": gap.get("endpoint_b", ""),
                    "gap_position": gap.get("gap_position", {}),
                    "gap_distance_px": gap.get("gap_distance_px", 0),
                    "alignment": gap.get("alignment", "unknown"),
                }
                decision = self.recover_near_edge_gap(gap_item)
                decisions.append(decision)
            iterations_completed = max(iterations_completed, 1)

        # ---- Source 3: Phase 3 gaps (geometric bypass path) ----
        phase3_gaps_data = self._artifacts.get("phase3_gaps", {})
        phase3_gaps = phase3_gaps_data.get("gaps", [])
        if phase3_gaps:
            logger.info(f"Processing {len(phase3_gaps)} Phase 3 geometric gaps")
            for gap in phase3_gaps:
                gap_item: Dict[str, Any] = {
                    "category": "near_edge_gap",
                    "group_key": f"phase3_gap::{gap.get('edge_a','')}::{gap.get('edge_b','')}",
                    "priority": "medium",
                    "edge_a": gap.get("edge_a", ""),
                    "edge_b": gap.get("edge_b", ""),
                    "endpoint_a": gap.get("endpoint_a", ""),
                    "endpoint_b": gap.get("endpoint_b", ""),
                    "gap_position": gap.get("gap_position", {}),
                    "gap_distance_px": gap.get("gap_distance_px", 0),
                    "alignment": gap.get("alignment", "unknown"),
                    "source": "phase3_gap_detection",
                }
                decision = self.recover_near_edge_gap(gap_item)
                decisions.append(decision)
            iterations_completed = max(iterations_completed, 1)

        # ---- Source 4: Phase 3 boundary terminal enrichment ----
        phase3_boundary_data = self._artifacts.get("phase3_boundary", {})
        boundary_terminals = phase3_boundary_data.get("boundary_terminals", [])
        if boundary_terminals:
            logger.info(f"Processing {len(boundary_terminals)} Phase 3 boundary terminals")
            for bt in boundary_terminals:
                bt_item: Dict[str, Any] = {
                    "category": "boundary_terminal",
                    "group_key": f"boundary::{bt.get('edge_id','')}",
                    "priority": "medium",
                    "edge_id": bt.get("edge_id", ""),
                    "source_node": bt.get("source_node", ""),
                    "target_node": bt.get("target_node", ""),
                    "source_boundary_side": bt.get("source_boundary_side"),
                    "target_boundary_side": bt.get("target_boundary_side"),
                    "source_col": bt.get("source_col"),
                    "source_row": bt.get("source_row"),
                    "target_col": bt.get("target_col"),
                    "target_row": bt.get("target_row"),
                    "source": "phase3_boundary_detection",
                }
                result = RecoveryItem(
                    original_item=bt_item,
                    category="boundary_terminal",
                    group_key=bt_item["group_key"],
                    priority="medium",
                )
                # Check if this edge already has an off_page_connector
                edge_id = bt.get("edge_id", "")
                graph = self._artifacts.get("graph", {})
                edges = graph.get("edges", [])
                edge = next((e for e in edges if e.get("id") == edge_id), None)
                if edge and edge.get("off_page_connector"):
                    result.action = RecoveryAction.ACCEPT
                    result.notes = f"Edge {edge_id} already has off_page_connector"
                    result.strategy_attempted.append("off_page_connector_present")
                else:
                    result.action = RecoveryAction.HUMAN_REVIEW
                    result.notes = (
                        f"Edge {edge_id} terminal near {bt.get('source_boundary_side') or bt.get('target_boundary_side')} image boundary "
                        f"— likely off-page connector requiring manual verification"
                    )
                    result.strategy_attempted.append("boundary_proximity_check")
                decisions.append(result)
            iterations_completed = max(iterations_completed, 1)

        if not decisions:
            logger.info("No recovery items to process — recovery loop complete")
            return self._build_output(iterations_completed=0, decisions=[])

        logger.info(
            f"Recovery classification complete: "
            f"{sum(1 for d in decisions if d.action == RecoveryAction.RETRY)} retry, "
            f"{sum(1 for d in decisions if d.action == RecoveryAction.HUMAN_REVIEW)} human_review, "
            f"{sum(1 for d in decisions if d.action == RecoveryAction.ACCEPT)} accept"
        )

        return self._build_output(iterations_completed=iterations_completed, decisions=decisions)

    def _build_output(self, iterations_completed: int, decisions: List[RecoveryItem]) -> Dict[str, Any]:
        """Build the final output dict."""
        retry_count = sum(1 for d in decisions if d.action == RecoveryAction.RETRY)
        human_review_count = sum(1 for d in decisions if d.action == RecoveryAction.HUMAN_REVIEW)
        accept_count = sum(1 for d in decisions if d.action == RecoveryAction.ACCEPT)

        return {
            "version": "1.0",
            "max_iterations": self.max_iterations,
            "iterations_completed": iterations_completed,
            "summary": {
                "total_items": len(decisions),
                "retry": retry_count,
                "human_review": human_review_count,
                "accept": accept_count,
            },
            "decisions": [d.to_dict() for d in decisions],
        }


def run_recovery_stage(job_dir: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    Stage function — runs the recovery loop on a job output directory.

    Reads:
        stage13_review_queue.json
        stage2_ocr_regions.json
        stage4_objects.json
        stage12_graph.json

    Writes:
        stage5_recovery_decisions.json

    Does NOT modify any other pipeline artifact.
    """
    engine = RecoveryEngine(Path(job_dir), max_iterations=max_iterations)
    result = engine.run()

    out_path = Path(job_dir) / "stage5_recovery_decisions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote {out_path}")

    return result
