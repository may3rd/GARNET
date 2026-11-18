"""Checks for the two alternative backends: vector fast-path and skeleton-graph routing.

Both replace the same thing (the greedy pixel walk), so both get asserted on synthetic input
where the right answer is known: a rotation that must land strokes on the raster frame, a chain
of strokes that must collapse into one run, and a routing that must follow the ink rather than
cut across a gap.

Run: .venv/bin/python -m unittest discover -s garnet/tests -p "test_backends.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from garnet.path_tracer import skeleton_graph as sg  # noqa: E402
from garnet.path_tracer import vector_trace as vt  # noqa: E402


class VectorPathTest(unittest.TestCase):
    def test_270_degree_rotation_maps_media_box_onto_the_rendered_page(self):
        # A3 media box 842x1191 rotated 270 -> 1191x842 page; a point on the media-box left
        # edge must end up on the rendered page's TOP edge, not half a page away.
        x, y = vt._rotate_to_page(0.0, 0.0, 270, 842.0, 1191.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 842.0)
        x, y = vt._rotate_to_page(842.0, 0.0, 270, 842.0, 1191.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)

    def test_fragmented_strokes_chain_into_one_run(self):
        # Three collinear strokes with 8 px holes: one pipe, drawn in pieces by the plotter.
        segs = [(0, 0, 100, 0), (108, 0, 200, 0), (208, 0, 300, 0)]
        net = vt.build_network(segs, (0, 0), snap_px=12)
        self.assertEqual(len(net["polylines"]), 1)
        self.assertGreater(net["polylines"][0]["length_px"], 280)
        traces = vt.to_trace_results(net, min_length_px=50)
        self.assertEqual(len(traces), 1)
        # every emitted segment must be real stroke geometry, never a jump between fragments
        for seg in next(iter(traces.values()))["segments"]:
            self.assertAlmostEqual(seg["y1"], 0, places=3)
            self.assertAlmostEqual(seg["y2"], 0, places=3)

    def test_connect_filter_keeps_only_runs_between_items(self):
        boxes = [{"bbox": {"x_min": -20, "y_min": -20, "x_max": 20, "y_max": 20}},
                 {"bbox": {"x_min": 280, "y_min": -20, "x_max": 320, "y_max": 20}}]
        between = {"segments": [{"x1": 20, "y1": 0, "x2": 280, "y2": 0}]}
        dangling = {"segments": [{"x1": 20, "y1": 400, "x2": 280, "y2": 400}]}
        self.assertTrue(vt.connects_items(between, boxes))
        self.assertFalse(vt.connects_items(dangling, boxes))


class SkeletonGraphTest(unittest.TestCase):
    def test_a_tee_reads_as_three_edges_and_routing_follows_the_ink(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[98:102, 10:190] = 255          # horizontal pipe
        mask[98:190, 98:102] = 255         # vertical stub -> tee at (100, 100)
        g = sg.skeleton_graph(mask)
        self.assertGreaterEqual(g.number_of_nodes(), 3)     # two ends + the junction
        start = sg._nearest_node(g, 15, 100, 20)
        goal = sg._nearest_node(g, 185, 100, 20)
        self.assertIsNotNone(start)
        self.assertIsNotNone(goal)
        points = sg.route(g, start, goal)
        self.assertIsNotNone(points)
        # the route must travel along y=100, not cut diagonally through empty paper
        self.assertTrue(all(abs(y - 100) <= 3 for _, y in points))

    def test_route_returns_none_when_the_ink_does_not_reach(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[98:102, 10:90] = 255
        mask[98:102, 120:190] = 255        # a 30 px gap, wider than any snap
        g = sg.skeleton_graph(mask)
        start = sg._nearest_node(g, 15, 100, 20)
        goal = sg._nearest_node(g, 185, 100, 20)
        self.assertIsNone(sg.route(g, start, goal))


if __name__ == "__main__":
    unittest.main()
