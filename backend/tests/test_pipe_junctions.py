import unittest

import numpy as np

from garnet.pipe_junctions import run_pipe_junction_stage


class PipeJunctionTests(unittest.TestCase):
    def test_run_pipe_junction_stage_splits_confirmed_and_unresolved(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        candidates = [
            {
                "id": "junction_ok",
                "classification": "confirmed_junction",
                "centroid": {"x": 10.0, "y": 10.0},
            },
            {
                "id": "junction_bad",
                "classification": "unresolved",
                "centroid": {"x": 15.0, "y": 7.0},
            },
            {
                "id": "crossing_0",
                "classification": "non_connecting_crossing",
                "centroid": {"x": 8.0, "y": 8.0},
            },
        ]

        result = run_pipe_junction_stage(
            image_bgr=image,
            crossing_candidates=candidates,
            image_id="sample.png",
        )

        self.assertIn("junctions_payload", result)
        self.assertEqual(result["summary"]["confirmed_junction_count"], 1)
        self.assertEqual(result["summary"]["unresolved_junction_count"], 1)
        self.assertEqual(result["summary"]["non_connecting_crossing_count"], 1)


if __name__ == "__main__":
    unittest.main()
