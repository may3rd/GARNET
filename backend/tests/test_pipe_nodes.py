import unittest

import numpy as np

from garnet.pipe_nodes import _degree_map, run_pipe_node_stage


class PipeNodeTests(unittest.TestCase):
    def test_degree_map_marks_endpoint_and_junction_values(self) -> None:
        skeleton = np.zeros((9, 9), dtype=np.uint8)
        skeleton[4, 2:7] = 255
        skeleton[2:7, 4] = 255

        degree = _degree_map(skeleton)

        self.assertEqual(int(degree[4, 2]), 11)
        self.assertGreaterEqual(int(degree[4, 4]), 14)

    def test_run_pipe_node_stage_returns_endpoint_and_junction_maps(self) -> None:
        image = np.zeros((9, 9, 3), dtype=np.uint8)
        skeleton = np.zeros((9, 9), dtype=np.uint8)
        skeleton[4, 2:7] = 255
        skeleton[2:7, 4] = 255

        result = run_pipe_node_stage(
            image_bgr=image,
            skeleton_mask=skeleton,
            image_id="sample.png",
        )

        self.assertIn("endpoint_image", result)
        self.assertIn("junction_image", result)
        self.assertIn("overlay_image", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["endpoint_count"], 4)
        self.assertGreaterEqual(result["summary"]["junction_count"], 1)


if __name__ == "__main__":
    unittest.main()
