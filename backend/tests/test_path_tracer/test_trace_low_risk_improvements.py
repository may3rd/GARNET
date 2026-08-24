"""Tests for the Stage 5b low-risk tracing improvements.

Covers:
- raycast lateral-snap guard (prevents jumping onto a neighbouring parallel pipe)
- spatial index `_check_terminals` equivalence with the previous linear scan
- PipelineConfig trace knobs and their wiring into CVPipeTracer / branch loop
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from garnet.path_tracer.stage5b_pipeline import Stage5bPipelineMixin
from garnet.path_tracer.cv_pipe_tracer import (
    CVPipeTracer,
    TerminalType,
    _check_bbox_hit,
    _is_inside_bbox_exact,
)


class _FakeMixin(Stage5bPipelineMixin):
    """Minimal mixin holder so `_trace_cv_params` can be exercised."""

    def __init__(self, cfg):
        self.cfg = cfg


def _mask_with_vline(mask, x, y0, y1):
    for y in range(y0, y1 + 1):
        mask[y, x] = 255


class RaycastSnapGuardTest(unittest.TestCase):
    def test_raycast_does_not_jump_to_parallel_pipe(self):
        # Two vertical parallel pipes: x=50 (with a mid gap) and x=60.
        mask = np.zeros((220, 220), dtype=np.uint8)
        _mask_with_vline(mask, 50, 0, 200)
        for y in range(30, 51):  # gap on the left pipe
            mask[y, 50] = 0
        _mask_with_vline(mask, 60, 0, 200)

        tracer = CVPipeTracer(pipe_mask=mask)
        res = tracer._find_straight_raycast_candidate(50, 20, "DOWN", "")
        self.assertIsNotNone(res)
        # Must continue along the LEFT pipe (x=50) past the gap, not snap to x=60.
        self.assertEqual(res[0], 50)

    def test_default_snap_shift_is_instance_value(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        tracer = CVPipeTracer(pipe_mask=mask)
        self.assertEqual(tracer.raycast_max_snap_shift_px, 4)


class SpatialIndexTerminalEquivalenceTest(unittest.TestCase):
    """`_check_terminals` must reproduce the original linear-scan results exactly."""

    @staticmethod
    def _old_check_terminals(t, x, y, direction, source_obj_id="", look_ahead=0):
        dxy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
        dx, dy = dxy.get(direction, (0, 0))
        cpm, ctm, cdm, cem = 2, 4, 6, 4
        apm, aem, atm = 2, 4, 2
        for pc in t.page_connections:
            if pc["id"] == source_obj_id:
                continue
            if _check_bbox_hit(x, y, pc["bbox"], margin=cpm):
                return (TerminalType.PAGE_CONNECTION.value, pc["id"])
        for eq in t.equipment_objects:
            if eq["id"] == source_obj_id:
                continue
            if _is_inside_bbox_exact(x, y, eq["bbox"]):
                return (TerminalType.EQUIPMENT.value, eq["id"])
        for tag in t.instrument_tags:
            if tag["id"] == source_obj_id:
                continue
            m = cdm if tag["class_name"] == "instrument dcs" else ctm
            if _check_bbox_hit(x, y, tag["bbox"], margin=m):
                return (TerminalType.INSTRUMENT_TAG.value, tag["id"])
        for eq in t.equipment_objects:
            if eq["id"] == source_obj_id:
                continue
            b = eq["bbox"]
            if (b["x_min"] - cem <= x <= b["x_max"] + cem
                    and b["y_min"] - cem <= y <= b["y_max"] + cem):
                return (TerminalType.EQUIPMENT.value, eq["id"])
        for off in range(0, look_ahead, 5):
            tx = x + off * dx
            ty = y + off * dy
            for pc in t.page_connections:
                if pc["id"] == source_obj_id:
                    continue
                if _check_bbox_hit(tx, ty, pc["bbox"], margin=apm):
                    return (TerminalType.PAGE_CONNECTION.value, pc["id"])
            for eq in t.equipment_objects:
                if eq["id"] == source_obj_id:
                    continue
                b = eq["bbox"]
                if (b["x_min"] - aem <= tx <= b["x_max"] + aem
                        and b["y_min"] - aem <= ty <= b["y_max"] + aem):
                    return (TerminalType.EQUIPMENT.value, eq["id"])
            for tag in t.instrument_tags:
                if tag["id"] == source_obj_id:
                    continue
                if _check_bbox_hit(tx, ty, tag["bbox"], margin=atm):
                    return (TerminalType.INSTRUMENT_TAG.value, tag["id"])
        return None

    def test_terminal_lookup_matches_linear_scan(self):
        rng = random.Random(42)
        mask = np.zeros((200, 200), dtype=np.uint8)

        def box():
            x1 = rng.randint(0, 190)
            y1 = rng.randint(0, 190)
            return {"x_min": x1, "y_min": y1,
                    "x_max": x1 + rng.randint(1, 40), "y_max": y1 + rng.randint(1, 40)}

        mismatch = 0
        for _ in range(500):
            pc = [{"id": f"pc{i}", "class_name": "page connection", "bbox": box()}
                  for i in range(rng.randint(0, 10))]
            eq = [{"id": f"eq{i}", "class_name": "pump", "bbox": box()}
                  for i in range(rng.randint(0, 10))]
            tags = [{"id": f"tag{i}", "class_name": "instrument tag", "bbox": box()}
                    for i in range(rng.randint(0, 10))]
            t = CVPipeTracer(pipe_mask=mask, page_connections=pc,
                             equipment_objects=eq, instrument_tags=tags)
            x, y = rng.randint(0, 199), rng.randint(0, 199)
            direction = rng.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            look = rng.choice([0, 5, 10, 30])
            sources = [o["id"] for o in (pc + eq + tags)] + [""]
            src = rng.choice(sources)
            old = self._old_check_terminals(t, x, y, direction, src, look)
            new = t._check_terminals(x, y, direction, src, look)
            if old != new:
                mismatch += 1
        self.assertEqual(mismatch, 0)


class PipelineConfigTraceKnobsTest(unittest.TestCase):
    def test_defaults(self):
        from garnet.pid_extractor import PipelineConfig

        cfg = PipelineConfig()
        self.assertEqual(cfg.trace_max_steps, 5000)
        self.assertEqual(cfg.trace_min_step, 5)
        self.assertEqual(cfg.trace_straight_min_step, 10)
        self.assertEqual(cfg.trace_turn_min_step, 3)
        self.assertEqual(cfg.trace_lookahead_px, 30)
        self.assertEqual(cfg.trace_raycast_max_snap_shift_px, 4)
        self.assertEqual(cfg.trace_branch_min_run_px, 25)
        self.assertEqual(cfg.trace_branch_candidate_sample_step_px, 5)
        self.assertEqual(cfg.trace_branch_cluster_radius_px, 8)
        self.assertEqual(cfg.trace_branch_max_iterations, 5)

    def test_extended_trace_threshold_defaults(self):
        from garnet.pid_extractor import PipelineConfig

        cfg = PipelineConfig()
        expected = {
            "trace_centerline_radius_px": 8,
            "trace_side_path_inline_probe_px": 60,
            "trace_raycast_start_px": 20,
            "trace_raycast_max_px": 50,
            "trace_raycast_step_px": 2,
            "trace_anchor_turn_max_gap_px": 60,
            "trace_turn_terminal_scan_px": 70,
            "trace_turn_terminal_scan_far_px": 160,
            "trace_axis_rewind_px": 12,
            "trace_sheet_edge_margin_px": 10,
            "trace_warmup_steps": 20,
            "trace_turn_gap_max_px": 60,
            "trace_branch_side_turn_probe_px": 8,
            "trace_branch_side_turn_terminal_px": 90,
            "trace_turn_probe_px": 8,
            "trace_tee_search_px": 8,
            "trace_terminal_current_margin_px": 2,
            "trace_terminal_current_tag_margin_px": 4,
            "trace_terminal_current_dcs_margin_px": 6,
            "trace_terminal_current_equipment_margin_px": 4,
            "trace_terminal_ahead_margin_px": 2,
            "trace_terminal_ahead_equipment_margin_px": 4,
            "trace_inline_hit_margin_px": 2,
            "trace_inline_exit_margin_px": 6,
            "trace_branch_point_tolerance_px": 10,
            "trace_branch_turn_tolerance_px": 8,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(cfg, name), value)

    def test_trace_cv_params_wiring(self):
        from garnet.pid_extractor import PipelineConfig

        cfg = PipelineConfig(trace_min_step=7, trace_raycast_max_snap_shift_px=6)
        mixin = _FakeMixin(cfg)
        params = mixin._trace_cv_params()
        self.assertEqual(params["min_step"], 7)
        self.assertEqual(params["raycast_max_snap_shift_px"], 6)

        mask = np.zeros((64, 64), dtype=np.uint8)
        tracer = CVPipeTracer(pipe_mask=mask, **params)
        self.assertEqual(tracer.min_step, 7)
        self.assertEqual(tracer.raycast_max_snap_shift_px, 6)

    def test_extended_trace_cv_params_wiring(self):
        from garnet.pid_extractor import PipelineConfig

        cfg = PipelineConfig(
            trace_centerline_radius_px=12,
            trace_raycast_max_px=90,
            trace_inline_exit_margin_px=9,
            trace_terminal_ahead_equipment_margin_px=7,
        )
        mixin = _FakeMixin(cfg)
        params = mixin._trace_cv_params()
        self.assertEqual(params["centerline_radius_px"], 12)
        self.assertEqual(params["raycast_max_px"], 90)
        self.assertEqual(params["inline_exit_margin_px"], 9)
        self.assertEqual(params["terminal_ahead_equipment_margin_px"], 7)

        mask = np.zeros((64, 64), dtype=np.uint8)
        tracer = CVPipeTracer(pipe_mask=mask, **params)
        self.assertEqual(tracer.centerline_radius_px, 12)
        self.assertEqual(tracer.raycast_max_px, 90)
        self.assertEqual(tracer.inline_exit_margin_px, 9)
        self.assertEqual(tracer.terminal_ahead_equipment_margin_px, 7)


if __name__ == "__main__":
    unittest.main()