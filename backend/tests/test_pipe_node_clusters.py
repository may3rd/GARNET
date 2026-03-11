import unittest

import numpy as np

from garnet.pipe_node_clusters import run_pipe_node_cluster_stage


class PipeNodeClusterTests(unittest.TestCase):
    def test_run_pipe_node_cluster_stage_clusters_nearby_points(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        endpoint_mask = np.zeros((20, 20), dtype=np.uint8)
        junction_mask = np.zeros((20, 20), dtype=np.uint8)

        endpoint_mask[5, 5] = 255
        endpoint_mask[6, 5] = 255
        endpoint_mask[15, 15] = 255
        junction_mask[10, 10] = 255
        junction_mask[11, 10] = 255

        result = run_pipe_node_cluster_stage(
            image_bgr=image,
            endpoint_mask=endpoint_mask,
            junction_mask=junction_mask,
            image_id="sample.png",
            cluster_eps=2.0,
            cluster_min_samples=1,
        )

        self.assertIn("clusters_payload", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["endpoint_cluster_count"], 2)
        self.assertEqual(result["summary"]["junction_cluster_count"], 1)
        self.assertIn("raw_junction_cluster_count", result["summary"])
        self.assertIn("merged_junction_cluster_count", result["summary"])


if __name__ == "__main__":
    unittest.main()
