import unittest

from garnet.geometric_graph_builder import (
    build_graph_from_runs_and_junctions,
    chain_geometric_segments,
    detect_junctions_from_runs,
)


class GeometricGraphBuilderTests(unittest.TestCase):
    def test_chain_geometric_segments_merges_collinear_segments_with_small_gap(self) -> None:
        runs = chain_geometric_segments(
            [
                {"id": "a", "x1": 0, "y1": 10, "x2": 70, "y2": 10, "length": 70},
                {"id": "b", "x1": 75, "y1": 12, "x2": 130, "y2": 12, "length": 55},
                {"id": "c", "x1": 200, "y1": 10, "x2": 310, "y2": 10, "length": 110},
            ]
        )

        self.assertEqual(len(runs), 2)
        merged = next(run for run in runs if run["member_segment_ids"] == ["a", "b"])
        self.assertEqual(merged["orientation"], "H")
        self.assertEqual(merged["x1"], 0)
        self.assertEqual(merged["x2"], 130)

    def test_detect_junctions_classifies_l_corner_and_terminals(self) -> None:
        runs = [
            {"id": "run_h", "orientation": "H", "x1": 0, "y1": 10, "x2": 50, "y2": 10, "length": 50},
            {"id": "run_v", "orientation": "V", "x1": 50, "y1": 10, "x2": 50, "y2": 60, "length": 50},
        ]

        junctions = detect_junctions_from_runs(runs)

        l_junctions = [j for j in junctions if j["junction_subtype"] == "L"]
        terminals = [j for j in junctions if j["type"] == "terminal"]
        self.assertEqual(len(l_junctions), 1)
        self.assertEqual(set(l_junctions[0]["connected_runs"]), {"run_h", "run_v"})
        self.assertEqual(len(terminals), 2)

    def test_build_graph_from_runs_and_junctions_emits_stage12_shape(self) -> None:
        runs = [
            {"id": "run_h", "orientation": "H", "x1": 0, "y1": 10, "x2": 50, "y2": 10, "length": 50},
            {"id": "run_v", "orientation": "V", "x1": 50, "y1": 10, "x2": 50, "y2": 60, "length": 50},
        ]
        junctions = detect_junctions_from_runs(runs)

        result = build_graph_from_runs_and_junctions(runs, junctions, image_id="unit.png")
        payload = result["graph_payload"]

        self.assertEqual(payload["image_id"], "unit.png")
        self.assertIn("nodes", payload)
        self.assertIn("edges", payload)
        self.assertEqual(payload["unresolved_junction_ids"], [])
        self.assertEqual(payload["crossings"], [])
        self.assertEqual(len(payload["edges"]), 2)
        for edge in payload["edges"]:
            self.assertIn("id", edge)
            self.assertIn("source", edge)
            self.assertIn("target", edge)
            self.assertIn("pixel_length", edge)
            self.assertIn("polyline", edge)
            self.assertEqual(edge["review_state"], "provisional")


if __name__ == "__main__":
    unittest.main()
