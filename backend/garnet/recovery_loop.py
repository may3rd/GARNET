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
        """
        logger.info(f"RecoveryEngine starting — max_iterations={self.max_iterations}")

        # Load artifacts once
        queue_data = self.load_review_queue()
        self.load_stage_artifacts()

        items = queue_data.get("items", [])
        if not items:
            logger.info("No items in review queue — recovery loop complete")
            return self._build_output(iterations_completed=0, decisions=[])

        logger.info(f"Processing {len(items)} review queue items")

        # Bounded iteration: run strategies once per item
        # (Approach 2 = non-destructive; we classify and route, not reprocess in-place)
        decisions: List[RecoveryItem] = []
        for item in items:
            decision = self._classify_item(item)
            decisions.append(decision)

        iterations_completed = 1
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
