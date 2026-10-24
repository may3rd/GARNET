import json, os, tempfile, unittest

import numpy as np

from garnet.eval.render_golden import render


class T(unittest.TestCase):
    def test_render(self):
        img = np.full((600, 800, 3), 200, np.uint8)
        graph = {"nodes": [{"id": "S1/V-1/p1", "xy_px": [100, 100]}, {"id": "S1/V-2/p1", "xy_px": [600, 400]},
                           {"id": "S1/PC/x@1", "xy_px": [50, 500]}, {"id": "S1/PC/x@2", "xy_px": [700, 50]}],
                 "edges": [{"kind": "pipe", "u": "S1/V-1/p1", "v": "S1/V-2/p1", "line_core": "25-9",
                            "polyline": [[100, 100], [100, 380], [600, 380]]}]}
        g = {"sheet": "a-b-S1", "connections": [
            {"a": "V-1/p1", "b": "V-2/p1", "line_core": "25-1", "confidence": "high",
             "path_px": [[100, 100], [100, 400], [600, 400]]},
            {"a": "V-1/p1", "b": "NOPE/p9", "line_core": "25-2", "confidence": "low"},
            {"a": "PC/x", "b": "V-2/p1", "line_core": "25-3", "confidence": "low"}]}
        with tempfile.TemporaryDirectory() as d:
            open(f"{d}/pair_01.png", "w").close()
            self.assertEqual(len(render(g, graph, img, d)), 3)
            for n in (1, 2, 3):
                self.assertTrue(os.path.getsize(f"{d}/conn_{n:02d}.png") > 0)
            self.assertFalse(os.path.exists(f"{d}/pair_01.png"))
            self.assertIn("NO PATH TRACED", open(f"{d}/index.html").read())
            r = json.load(open(f"{d}/review.json"))["connections"]
            self.assertEqual([x["has_path"] for x in r], [True, False, False])
            self.assertTrue(r[0]["tracer_found"] and not r[0]["tracer_line_ok"])


if __name__ == "__main__":
    unittest.main()
