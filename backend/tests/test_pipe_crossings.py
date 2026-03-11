import unittest

import numpy as np

from garnet.pipe_crossings import run_pipe_crossing_stage


class PipeCrossingTests(unittest.TestCase):
    def test_run_pipe_crossing_stage_classifies_crossing(self) -> None:
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        sealed = np.zeros((21, 21), dtype=np.uint8)
        skeleton = np.zeros((21, 21), dtype=np.uint8)
        sealed[10, 4:17] = 255
        sealed[4:17, 10] = 255
        skeleton[10, 4:17] = 255
        skeleton[4:17, 10] = 255

        result = run_pipe_crossing_stage(
            image_bgr=image,
            sealed_mask=sealed,
            skeleton_mask=skeleton,
            node_clusters=[
                {
                    "id": "junction_0",
                    "kind": "junction",
                    "centroid": {"x": 10.0, "y": 10.0},
                    "member_count": 1,
                    "members": [{"row": 10, "col": 10}],
                }
            ],
            image_id="sample.png",
        )

        self.assertEqual(result["summary"]["candidate_count"], 1)
        candidate = result["crossings_payload"]["candidates"][0]
        self.assertEqual(candidate["classification"], "non_connecting_crossing")
        self.assertEqual(len(candidate["routing_pairs"]), 2)
        self.assertIn("stage4_object_evidence", candidate)

    def test_run_pipe_crossing_stage_classifies_tee_as_junction(self) -> None:
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        sealed = np.zeros((21, 21), dtype=np.uint8)
        skeleton = np.zeros((21, 21), dtype=np.uint8)
        sealed[10, 4:17] = 255
        sealed[4:11, 10] = 255
        skeleton[10, 4:17] = 255
        skeleton[4:11, 10] = 255

        result = run_pipe_crossing_stage(
            image_bgr=image,
            sealed_mask=sealed,
            skeleton_mask=skeleton,
            node_clusters=[
                {
                    "id": "junction_0",
                    "kind": "junction",
                    "centroid": {"x": 10.0, "y": 10.0},
                    "member_count": 1,
                    "members": [{"row": 10, "col": 10}],
                }
            ],
            image_id="sample.png",
        )

        self.assertEqual(result["crossings_payload"]["candidates"][0]["classification"], "confirmed_junction")


if __name__ == "__main__":
    unittest.main()
