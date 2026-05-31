"""Integration-style tests for Agent 2 — port computation and segment assembly."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from garnet.visual_primitives.agent2_pipeline_tracer import (
    SHEET_EDGE_THRESHOLD_PX,
    compute_port_from_bbox,
)
from garnet.visual_primitives.schemas import TraceDirection, TraceSegment, TraceStep, TraceTokenType


class TestComputePort(unittest.TestCase):
    IMG_W = 3961
    IMG_H = 3224

    def test_left_edge_page_connection(self):
        bbox = {"x_min": 208, "y_min": 3100, "x_max": 416, "y_max": 3220}
        x, y, direction = compute_port_from_bbox(bbox, self.IMG_W, self.IMG_H)
        self.assertEqual(direction, "RIGHT")
        self.assertEqual(x, 416)  # x_max

    def test_right_edge_page_connection(self):
        bbox = {"x_min": 3700, "y_min": 2450, "x_max": 3990, "y_max": 2500}
        x, y, direction = compute_port_from_bbox(bbox, self.IMG_W, self.IMG_H)
        self.assertEqual(direction, "LEFT")
        self.assertEqual(x, 3700)  # x_min

    def test_not_at_edge_wide_box(self):
        # Box in middle, wide → guess horizontal
        bbox = {"x_min": 1500, "y_min": 1500, "x_max": 1800, "y_max": 1550}
        x, y, direction = compute_port_from_bbox(bbox, self.IMG_W, self.IMG_H)
        self.assertIn(direction, ("LEFT", "RIGHT"))

    def test_not_at_edge_tall_box(self):
        # Box in middle, tall → guess vertical
        bbox = {"x_min": 1500, "y_min": 800, "x_max": 1550, "y_max": 1200}
        x, y, direction = compute_port_from_bbox(bbox, self.IMG_W, self.IMG_H)
        self.assertIn(direction, ("UP", "DOWN"))

    def test_top_edge(self):
        bbox = {"x_min": 1500, "y_min": 0, "x_max": 1600, "y_max": 50}
        x, y, direction = compute_port_from_bbox(bbox, self.IMG_W, self.IMG_H)
        self.assertEqual(direction, "DOWN")


class TestTraceResultSerialisation(unittest.TestCase):
    def test_segment_to_json(self):
        seg = TraceSegment(
            anchor_id="obj_000118",
            anchor_bbox_global=[950, 760, 999, 775],
            start_point_global=[999, 767],
            start_direction=TraceDirection.LEFT,
            steps=[
                TraceStep(token_type=TraceTokenType.STEP, direction=TraceDirection.LEFT, distance_px=100),
                TraceStep(token_type=TraceTokenType.HIT, symbol_class="gate_valve", symbol_bbox_view=[50, 10, 80, 40]),
                TraceStep(token_type=TraceTokenType.TERM, symbol_class="pump", symbol_bbox_view=[200, 100, 250, 160]),
            ],
            terminal_class="pump",
            terminal_tag="P-2512",
            terminal_point_global=[100, 400],
            total_length_px=120,
        )
        d = seg.model_dump()
        self.assertEqual(d["anchor_id"], "obj_000118")
        self.assertEqual(d["terminal_class"], "pump")
        self.assertEqual(len(d["steps"]), 3)

    def test_empty_trace_result(self):
        from garnet.visual_primitives.schemas import TraceResult

        tr = TraceResult(
            source_image="test.jpg",
            model="test-model",
            source_dimensions=[100, 200],
        )
        self.assertEqual(tr.total_segments, 0)


if __name__ == "__main__":
    unittest.main()
