"""Checks for the trace verification pass — the endpoint classifier is what decides HOLD.

The classifier is the load-bearing part: a walk that stops in empty space must be flagged, and
a walk that stops on an item must not be. Both are pure geometry over the trace dicts, so they
are asserted directly instead of through a full pipeline run.

Run: .venv/bin/python -m unittest discover -s garnet/tests -p "test_trace_verify.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from garnet.path_tracer.trace_verify import _bbox_distance, _endpoint, _try_repair  # noqa: E402

BBOX = {"x_min": 100, "y_min": 100, "x_max": 160, "y_max": 140}


class TraceVerifyTest(unittest.TestCase):
    def test_distance_to_an_item_bbox_is_zero_inside_and_euclidean_outside(self):
        self.assertEqual(_bbox_distance(BBOX, 120, 120), 0.0)          # inside
        self.assertEqual(_bbox_distance(BBOX, 160, 100), 0.0)          # on the corner
        self.assertEqual(_bbox_distance(BBOX, 170, 120), 10.0)         # 10 px right of the edge
        self.assertAlmostEqual(_bbox_distance(BBOX, 163, 144), 5.0)    # 3 px + 4 px diagonal

    def test_endpoint_comes_from_the_last_segment(self):
        trace = {"segments": [
            {"x1": 0, "y1": 0, "x2": 50, "y2": 0, "direction": "RIGHT"},
            {"x1": 50, "y1": 0, "x2": 50, "y2": 80, "direction": "DOWN"},
        ]}
        self.assertEqual(_endpoint(trace), (50, 80, "DOWN"))
        self.assertIsNone(_endpoint({"segments": []}))

    def test_repair_finds_nothing_when_the_mask_is_empty(self):
        empty = np.zeros((300, 300), np.uint8)
        self.assertIsNone(_try_repair(
            (150, 200, "RIGHT"),
            pipe_mask=empty, image=empty,
            buckets={"page": [], "instrument": [], "equipment": [], "inline": []},
            markers=[], params={}, source_id="t0", straight_only=False,
        ))

    def test_repair_rejects_a_walk_longer_than_a_repair(self):
        # A full-width pipe run means the repaired walk is a whole new line, not a continuation.
        mask = np.zeros((300, 900), np.uint8)
        mask[198:202, 0:900] = 255
        self.assertIsNone(_try_repair(
            (150, 200, "RIGHT"),
            pipe_mask=mask, image=mask,
            buckets={"page": [], "instrument": [], "equipment": [], "inline": []},
            markers=[], params={}, source_id="t0", straight_only=True, max_add_px=100,
        ))


if __name__ == "__main__":
    unittest.main()
