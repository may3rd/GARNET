import importlib.util
import os
from pathlib import Path
import unittest

EVALUATOR_PATH = (
    Path(__file__).resolve().parents[2] / "autoresearch" / "graph_loop" / "evaluate_graph.py"
)


def _load_evaluator_module():
    spec = importlib.util.spec_from_file_location("graph_loop_evaluator", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load graph loop evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(EVALUATOR_PATH.is_file(), "autoresearch graph loop evaluator not present (experiment branch only)")
class GraphLoopEvaluatorTests(unittest.TestCase):
    def test_graph_loop_score_rewards_junction_correctness_counters(self) -> None:
        evaluator = _load_evaluator_module()
        base_stage12 = {"edge_component_count": 100, "edge_count": 200, "accepted_attachment_count": 20}
        base_stage13 = {"unresolved_terminal_edge_count": 5, "review_queue_count": 10}
        weak = {
            "connection_seeded_continuation_count": 0,
            "accepted_junction_straight_through_count": 0,
            "rejected_junction_alignment_connection_count": 0,
            "invalid_shared_junction_fallback_candidate_count": 20,
            "junction_touching_continuation_count": 0,
        }
        strong = {
            "connection_seeded_continuation_count": 0,
            "accepted_junction_straight_through_count": 20,
            "rejected_junction_alignment_connection_count": 40,
            "invalid_shared_junction_fallback_candidate_count": 0,
            "junction_touching_continuation_count": 10,
        }

        self.assertLess(
            evaluator.graph_loop_score(base_stage12, base_stage13, strong),
            evaluator.graph_loop_score(base_stage12, base_stage13, weak),
        )


if __name__ == "__main__":
    unittest.main()
