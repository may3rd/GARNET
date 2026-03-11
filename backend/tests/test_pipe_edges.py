import unittest

import numpy as np

from garnet.pipe_edges import run_pipe_edge_stage


class PipeEdgeTests(unittest.TestCase):
    def test_run_pipe_edge_stage_traces_simple_segment(self) -> None:
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        skeleton = np.zeros((12, 12), dtype=np.uint8)
        skeleton[6, 2:10] = 255

        clusters = [
            {
                "id": "endpoint_0",
                "kind": "endpoint",
                "centroid": {"x": 2.0, "y": 6.0},
                "member_count": 1,
                "members": [{"row": 6, "col": 2}],
            },
            {
                "id": "endpoint_1",
                "kind": "endpoint",
                "centroid": {"x": 9.0, "y": 6.0},
                "member_count": 1,
                "members": [{"row": 6, "col": 9}],
            },
        ]

        result = run_pipe_edge_stage(
            image_bgr=image,
            skeleton_mask=skeleton,
            node_clusters=clusters,
            image_id="sample.png",
            min_edge_length_px=2,
        )

        self.assertIn("edges_payload", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["edge_count"], 1)
        self.assertEqual(len(result["edges_payload"]["edges"]), 1)

    def test_run_pipe_edge_stage_routes_through_non_connecting_crossing(self) -> None:
        image = np.zeros((15, 15, 3), dtype=np.uint8)
        skeleton = np.zeros((15, 15), dtype=np.uint8)
        skeleton[7, 2:13] = 255
        skeleton[2:13, 7] = 255

        clusters = [
            {"id": "endpoint_left", "kind": "endpoint", "centroid": {"x": 2.0, "y": 7.0}, "member_count": 1, "members": [{"row": 7, "col": 2}]},
            {"id": "endpoint_right", "kind": "endpoint", "centroid": {"x": 12.0, "y": 7.0}, "member_count": 1, "members": [{"row": 7, "col": 12}]},
            {"id": "endpoint_top", "kind": "endpoint", "centroid": {"x": 7.0, "y": 2.0}, "member_count": 1, "members": [{"row": 2, "col": 7}]},
            {"id": "endpoint_bottom", "kind": "endpoint", "centroid": {"x": 7.0, "y": 12.0}, "member_count": 1, "members": [{"row": 12, "col": 7}]},
            {"id": "junction_0", "kind": "junction", "centroid": {"x": 7.0, "y": 7.0}, "member_count": 1, "members": [{"row": 7, "col": 7}]},
        ]
        crossing_resolution = [
            {
                "id": "junction_0",
                "classification": "non_connecting_crossing",
                "members": [{"row": 7, "col": 7}],
                "branches": [
                    {"branch_id": "left", "entry_pixels": [{"row": 7, "col": 6}], "entry_centroid": {"x": 6.0, "y": 7.0}},
                    {"branch_id": "right", "entry_pixels": [{"row": 7, "col": 8}], "entry_centroid": {"x": 8.0, "y": 7.0}},
                    {"branch_id": "top", "entry_pixels": [{"row": 6, "col": 7}], "entry_centroid": {"x": 7.0, "y": 6.0}},
                    {"branch_id": "bottom", "entry_pixels": [{"row": 8, "col": 7}], "entry_centroid": {"x": 7.0, "y": 8.0}},
                ],
                "routing_pairs": [["left", "right"], ["top", "bottom"]],
            }
        ]

        result = run_pipe_edge_stage(
            image_bgr=image,
            skeleton_mask=skeleton,
            node_clusters=clusters,
            image_id="sample.png",
            min_edge_length_px=2,
            crossing_resolution=crossing_resolution,
        )

        edges = result["edges_payload"]["edges"]
        endpoints = {frozenset((edge["source"], edge["target"])) for edge in edges}
        self.assertEqual(len(edges), 2)
        self.assertIn(frozenset(("endpoint_left", "endpoint_right")), endpoints)
        self.assertIn(frozenset(("endpoint_top", "endpoint_bottom")), endpoints)


if __name__ == "__main__":
    unittest.main()
