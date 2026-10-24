import os, tempfile, unittest

import numpy as np

from garnet.eval.render_golden import render


class T(unittest.TestCase):
    def test_render(self):
        img = np.full((600, 800, 3), 200, np.uint8)
        graph = {"nodes": [{"id": "S1/V-1/p1", "xy_px": [100, 100]}, {"id": "S1/V-2/p1", "xy_px": [600, 400]},
                           {"id": "S1/PC/x@1", "xy_px": [50, 500]}, {"id": "S1/PC/x@2", "xy_px": [700, 50]}]}
        g = {"sheet": "a-b-S1", "connections": [
            {"a": "V-1/p1", "b": "V-2/p1", "line_core": "25-1", "confidence": "high",
             "path_px": [[100, 100], [100, 400], [600, 400]]},
            {"a": "V-1/p1", "b": "NOPE/p9", "line_core": "25-2", "confidence": "low"},
            {"a": "PC/x", "b": "V-2/p1", "line_core": "25-3", "confidence": "low"}]}
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(render(g, graph, img, d), 3)
            for n in (1, 2, 3):
                for k in ("pair", "path"):
                    self.assertTrue(os.path.getsize(f"{d}/{k}_{n:02d}.png") > 0)
            self.assertIn("UNRESOLVED NOPE/p9", open(f"{d}/index.html").read())


if __name__ == "__main__":
    unittest.main()
