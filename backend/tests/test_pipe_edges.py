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


if __name__ == "__main__":
    unittest.main()
