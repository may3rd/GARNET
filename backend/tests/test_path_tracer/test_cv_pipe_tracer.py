import unittest

import numpy as np

from garnet.path_tracer.cv_pipe_tracer import CVPipeTracer, TerminalType, TraceResult


def _plus_mask() -> np.ndarray:
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[40, 10:91] = 255
    mask[10:71, 50] = 255
    return mask


def _tee_stem_mask() -> np.ndarray:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[40, 10:71] = 255
    mask[40:71, 40] = 255
    return mask


class TestCvPipeTracerTeeThroughTurns(unittest.TestCase):
    def test_bidirectional_turn_leg_detects_tee_through_not_elbow(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[40, 10:51] = 255
        mask[10:41, 50] = 255
        tee = CVPipeTracer(mask)

        self.assertFalse(tee._has_bidirectional_turn_leg(50, 40, "UP"))

        mask[41:71, 50] = 255
        tee = CVPipeTracer(mask)

        self.assertTrue(tee._has_bidirectional_turn_leg(50, 40, "UP"))


class TestRewoundAxisContinuation(unittest.TestCase):
    def test_axis_continues_past_cross_not_tee_bar(self):
        tracer = CVPipeTracer(_plus_mask())
        self.assertTrue(tracer._axis_continues_past(50, 40, "RIGHT"))
        self.assertFalse(tracer._axis_continues_past(40, 40, "UP"))

    def test_cross_walk_does_not_stop_as_tee(self):
        tracer = CVPipeTracer(_plus_mask(), min_step=5, straight_min_step=10)
        result = tracer.trace(12, 40, "RIGHT")
        self.assertNotEqual(result.terminal_type, TerminalType.TEE_JUNCTION.value)
        self.assertGreaterEqual(result.terminal_x, 85)

    def test_tee_from_stem_still_stops(self):
        tracer = CVPipeTracer(_tee_stem_mask(), min_step=5, straight_min_step=10)
        result = tracer.trace(40, 68, "UP")
        self.assertEqual(result.terminal_type, TerminalType.TEE_JUNCTION.value)
        self.assertLessEqual(abs(result.terminal_y - 40), 8)

    def test_thick_cross_walk_does_not_stop_as_tee(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[39:42, 10:91] = 255
        mask[10:71, 49:52] = 255
        tracer = CVPipeTracer(mask, min_step=5, straight_min_step=10)
        result = tracer.trace(12, 40, "RIGHT")
        self.assertNotEqual(result.terminal_type, TerminalType.TEE_JUNCTION.value)
        self.assertGreaterEqual(result.terminal_x, 85)

    def test_explicit_junction_marker_stops_cross_walk(self):
        mask = np.zeros((100, 220), dtype=np.uint8)
        mask[49:52, 10:211] = 255
        mask[10:91, 107:114] = 255
        marker = {
            "id": "node_001",
            "bbox": {"x_min": 105, "y_min": 45, "x_max": 115, "y_max": 55},
        }
        tracer = CVPipeTracer(
            mask,
            junction_markers=[marker],
            min_step=5,
            straight_min_step=10,
        )

        result = tracer.trace(12, 50, "RIGHT", source_obj_id="branch_001")

        self.assertEqual(result.terminal_type, TerminalType.TEE_JUNCTION.value)
        self.assertEqual(result.terminal_obj_id, "node_001")

    def test_axis_resume_records_continuous_segments(self):
        tracer = CVPipeTracer(_plus_mask(), min_step=5, straight_min_step=10)
        result = TraceResult()

        resumed = tracer._continue_if_axis_open(result, 12, 40, 50, 40, "RIGHT")

        self.assertIsNotNone(resumed)
        self.assertGreaterEqual(len(result.segments), 2)
        self.assertTrue(all(
            (first.x2, first.y2) == (second.x1, second.y1)
            for first, second in zip(result.segments, result.segments[1:])
        ))
        self.assertEqual(result.trace_length_px, sum(segment.length_px for segment in result.segments))


class TestCvPipeTracerInlineSymbols(unittest.TestCase):
    def test_trace_walks_through_inline_symbol_and_continues(self):
        # Continuous pipe with a valve bbox overlapping the middle of the line.
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[40, 10:91] = 255
        inline = [{
            "id": "valve_1",
            "class_name": "gate valve",
            "bbox": {"x_min": 40, "x_max": 60, "y_min": 35, "y_max": 45},
        }]
        tracer = CVPipeTracer(mask, min_step=5, straight_min_step=10)
        tracer.set_inline_symbols(inline)

        result = tracer.trace(12, 40, "RIGHT")

        # The inline symbol is recorded as a hit, not a terminal.
        self.assertEqual(
            [(h.class_name, h.x, h.y) for h in result.hits],
            [("gate valve", 42, 40)],
        )
        # The trace continues past the valve to the far end of the pipe.
        self.assertEqual(result.terminal_type, TerminalType.DEAD_END.value)
        self.assertGreaterEqual(result.terminal_x, 85)

    def test_trace_jumps_inline_gap_and_continues(self):
        # The valve replaces the pipe line, leaving a gap in the mask.
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[40, 10:91] = 255
        mask[40, 40:61] = 0
        inline = [{
            "id": "valve_1",
            "class_name": "gate valve",
            "bbox": {"x_min": 40, "x_max": 60, "y_min": 35, "y_max": 45},
        }]
        tracer = CVPipeTracer(mask, min_step=5, straight_min_step=10)
        tracer.set_inline_symbols(inline)

        result = tracer.trace(12, 40, "RIGHT")

        # The inline symbol is recorded as a hit and the trace resumes on the
        # far side of the valve rather than stopping at the gap.
        self.assertTrue(any(h.class_name == "gate valve" for h in result.hits))
        self.assertEqual(result.terminal_type, TerminalType.DEAD_END.value)
        self.assertGreaterEqual(result.terminal_x, 85)


if __name__ == "__main__":
    unittest.main()
