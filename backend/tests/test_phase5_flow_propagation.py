import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.stage10_process_exports import _aggregate_flow_direction
from garnet.trace_graph_builder import _recompute_flow_direction_from_attachments


class Phase5FlowPropagationTests(unittest.TestCase):
    def test_local_arrow_resolves_only_edge_with_local_evidence(self):
        first = {"segments": [{"x1": 0, "y1": 0, "x2": 50, "y2": 0}], "attachments": {"flow_arrows": [{"id": "a", "vector": [1, 0], "projected_xy": [20, 0], "confidence": .9}]}}
        second = {"segments": [{"x1": 50, "y1": 0, "x2": 100, "y2": 0}], "attachments": {"flow_arrows": []}}
        self.assertEqual(_recompute_flow_direction_from_attachments(first)["flow_direction_state"], "forward")
        self.assertEqual(_recompute_flow_direction_from_attachments(second)["flow_direction_state"], "unknown")

    def test_reversed_duplicate_vectors_are_interpreted_per_retained_orientation(self):
        forward = {"segments": [{"x1": 0, "y1": 0, "x2": 100, "y2": 0}], "attachments": {"flow_arrows": [{"vector": [1, 0], "projected_xy": [50, 0]}]}}
        reverse = {"segments": [{"x1": 100, "y1": 0, "x2": 0, "y2": 0}], "attachments": {"flow_arrows": [{"vector": [-1, 0], "projected_xy": [50, 0]}]}}
        self.assertEqual(_recompute_flow_direction_from_attachments(forward)["flow_direction_state"], "forward")
        self.assertEqual(_recompute_flow_direction_from_attachments(reverse)["flow_direction_state"], "forward")

    def test_arrow_uses_nearest_segment_not_segment_midpoint(self):
        edge = {"segments": [{"x1": 0, "y1": 0, "x2": 1000, "y2": 0}, {"x1": 1000, "y1": 0, "x2": 1000, "y2": 10}], "attachments": {"flow_arrows": [{"vector": [1, 0], "projected_xy": [900, 0]}]}}
        self.assertEqual(_recompute_flow_direction_from_attachments(edge)["flow_direction_state"], "forward")

    def test_opposing_duplicate_evidence_is_conflicting(self):
        edge = {"segments": [{"x1": 0, "y1": 0, "x2": 100, "y2": 0}], "attachments": {"flow_arrows": [{"vector": [1, 0]}, {"vector": [-1, 0]}]}}
        self.assertEqual(_recompute_flow_direction_from_attachments(edge)["flow_direction_state"], "conflicting")

    def test_stage10_direction_aggregation_ignores_invalid_confidence(self):
        result = _aggregate_flow_direction([{"flow_direction_state": "forward", "flow_direction_confidence": float("nan")}, {"flow_direction_state": "forward", "flow_direction_confidence": "bad"}])
        self.assertEqual(result["flow_direction_state"], "forward")
        self.assertIsNone(result["flow_direction_confidence"])
        json.dumps(result, allow_nan=False)

    def test_stage10_mixed_edge_storage_orders_remain_conservative(self):
        result = _aggregate_flow_direction([{"flow_direction_state": "forward"}, {"flow_direction_state": "reverse"}])
        self.assertEqual(result["flow_direction_state"], "unknown")


if __name__ == "__main__":
    unittest.main()
