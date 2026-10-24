import os
import tempfile
import unittest

import cv2
import numpy as np

from garnet.graph.ln_candidates import find_candidates, render_crop


class T(unittest.TestCase):
    box = [250, 150, 350, 170]
    traces = {"near": [[100, 210], [500, 210]],   # 40 px below box
              "far": [[100, 320], [500, 320]],    # 150 px
              "gone": [[100, 570], [500, 570]]}   # 400 px

    def test_order_and_exclusion(self):
        c = find_candidates(self.box, self.traces)
        self.assertEqual([(x["letter"], x["trace_id"]) for x in c], [("A", "near"), ("B", "far")])
        self.assertAlmostEqual(c[0]["dist_px"], 40, delta=2)

    def test_pick_included_even_if_far(self):
        c = find_candidates(self.box, self.traces, "gone")
        self.assertEqual([x["trace_id"] for x in c], ["near", "far", "gone"])

    def test_render(self):
        img = np.full((400, 600, 3), 255, np.uint8)
        lab = {"id": "line_number_0001", "text": "4\"-X", "bbox": self.box, "geometric_pick": None,
               "candidates": find_candidates(self.box, self.traces)}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.png")
            render_crop(img, lab, self.traces, p)
            self.assertIsNotNone(cv2.imread(p))


if __name__ == "__main__":
    unittest.main()
