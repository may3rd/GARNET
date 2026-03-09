import unittest

import numpy as np

from garnet.pipe_junctions import run_pipe_junction_stage


class PipeJunctionTests(unittest.TestCase):
    def test_run_pipe_junction_stage_splits_confirmed_and_unresolved(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        skeleton = np.zeros((20, 20), dtype=np.uint8)
        skeleton[10, 5:16] = 255
        skeleton[5:16, 10] = 255
        skeleton[4:8, 15] = 255

        clusters = [
            {
                "id": "junction_ok",
                "kind": "junction",
                "centroid": {"x": 10.0, "y": 10.0},
                "member_count": 1,
                "members": [{"row": 10, "col": 10}],
            },
            {
                "id": "junction_bad",
                "kind": "junction",
                "centroid": {"x": 15.0, "y": 7.0},
                "member_count": 1,
                "members": [{"row": 7, "col": 15}],
            },
        ]

        result = run_pipe_junction_stage(
            image_bgr=image,
            skeleton_mask=skeleton,
            node_clusters=clusters,
            image_id="sample.png",
        )

        self.assertIn("junctions_payload", result)
        self.assertEqual(result["summary"]["confirmed_junction_count"], 1)
        self.assertEqual(result["summary"]["unresolved_junction_count"], 1)


if __name__ == "__main__":
    unittest.main()
