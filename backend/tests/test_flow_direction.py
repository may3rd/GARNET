import unittest
import numpy as np

from garnet.flow_direction import (
    aggregate_edge_direction,
    infer_arrow_route_direction,
    normalize_arrow_evidence,
)
from garnet.trace_associations import build_trace_associations


class FlowDirectionTests(unittest.TestCase):
    EDGE = {"segments": [{"x1": 0, "y1": 10, "x2": 100, "y2": 10}]}

    def test_explicit_horizontal_and_vertical_vectors(self):
        right = normalize_arrow_evidence({"vector": [4, 0], "confidence": 0.9})
        right["segment_index"] = 0
        self.assertEqual(infer_arrow_route_direction(right, self.EDGE), "forward")
        vertical = normalize_arrow_evidence({"tip": [10, 0], "tail": [10, 20], "confidence": 0.8})
        vertical["segment_index"] = 0
        self.assertEqual(infer_arrow_route_direction(vertical, self.EDGE), "conflicting")

    def test_raster_ambiguity_and_no_arrow_are_unknown(self):
        evidence = normalize_arrow_evidence({"bbox": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}}, image=None)
        self.assertIsNone(evidence["vector"])
        self.assertEqual(infer_arrow_route_direction(evidence, self.EDGE), "unknown")
        self.assertEqual(aggregate_edge_direction([]), ("unknown", None))

    def test_agreement_and_conflict(self):
        a = {"route_direction": "forward", "confidence": 0.8}
        b = {"route_direction": "forward", "confidence": 0.6}
        self.assertEqual(aggregate_edge_direction([a, b]), ("forward", 0.7))
        self.assertEqual(aggregate_edge_direction([a, {"route_direction": "reverse"}])[0], "conflicting")

    def test_bidirectional_requires_review(self):
        unresolved = normalize_arrow_evidence({"direction": "bidirectional"})
        self.assertNotIn("explicit_state", unresolved)
        reviewed = normalize_arrow_evidence({"direction": "bidirectional", "review_state": "accepted"})
        self.assertEqual(reviewed["explicit_state"], "bidirectional")
        self.assertEqual(aggregate_edge_direction([reviewed])[0], "bidirectional")

    def test_clear_raster_cardinal_arrows_and_symmetric_blob(self):
        for direction in ("right", "left", "up", "down"):
            image = np.full((20, 20), 255, dtype=np.uint8)
            if direction == "right":
                image[9:11, 3:15] = 0
                for i in range(6): image[10-i:11+i, 14+i] = 0
            elif direction == "left":
                image[9:11, 5:17] = 0
                for i in range(6): image[10-i:11+i, 5-i] = 0
            elif direction == "down":
                image[3:15, 9:11] = 0
                for i in range(6): image[14+i, 10-i:11+i] = 0
            else:
                image[5:17, 9:11] = 0
                for i in range(6): image[5-i, 10-i:11+i] = 0
            evidence = normalize_arrow_evidence(
                {"bbox": {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20}},
                image=image, raster_confidence_threshold=.2, raster_asymmetry_threshold=.05,
            )
            self.assertEqual(evidence["vector"], {"right": [1.0, 0.0], "left": [-1.0, 0.0], "up": [0.0, -1.0], "down": [0.0, 1.0]}[direction])
        blob = np.full((20, 20), 255, dtype=np.uint8)
        blob[9:11, 4:16] = 0
        self.assertIsNone(normalize_arrow_evidence({"bbox": {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20}}, image=blob)["vector"])

    def test_build_associations_enriches_shared_arrow_record(self):
        result = build_trace_associations(
            image_id="x", objects=[{"id": "a", "class_name": "arrow", "bbox": {"x_min": 10, "y_min": 5, "x_max": 20, "y_max": 15}, "vector": [1, 0], "confidence": .9}],
            trace_payload={"t": {"segments": [{"x1": 0, "y1": 10, "x2": 100, "y2": 10}]}}, branch_payload={"branches": {}}, ports_payload={}, line_numbers=[], instrument_tags=[],
            equipment_port_max_distance_px=10, inline_object_max_distance_px=10, text_max_distance_px=10, instrument_max_distance_px=10, arrow_max_distance_px=20,
        )
        edge = result["trace_edges"][0]
        arrow = result["associations"]["flow_arrows"]["accepted"][0]
        self.assertIs(edge["attachments"]["flow_arrows"][0], arrow)
        self.assertEqual(edge["flow_direction_state"], "forward")
        self.assertEqual(edge["flow_direction_review_state"], "inferred")
        self.assertEqual(arrow["flow_direction_evidence"]["vector"], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
