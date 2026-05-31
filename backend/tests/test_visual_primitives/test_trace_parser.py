"""Tests for trace_parser — parsing VLM trace responses into TraceStep lists."""
from __future__ import annotations

import unittest

from garnet.visual_primitives.schemas import TraceDirection, TraceStep, TraceTokenType
from garnet.visual_primitives.trace_parser import (
    has_terminal,
    last_terminal,
    parse_trace_response,
    total_trace_distance,
)


class TestParseStep(unittest.TestCase):
    def test_single_step_right(self):
        response = "I see the pipe. <|step|> RIGHT 45 then it bends."
        steps = parse_trace_response(response)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].token_type, TraceTokenType.STEP)
        self.assertEqual(steps[0].direction, TraceDirection.RIGHT)
        self.assertEqual(steps[0].distance_px, 45)

    def test_step_with_px_suffix(self):
        response = "<|step|> DOWN 30px"
        steps = parse_trace_response(response)
        self.assertEqual(steps[0].distance_px, 30)

    def test_multiple_steps(self):
        response = "<|step|> RIGHT 50 <|step|> DOWN 30 <|step|> RIGHT 25"
        steps = parse_trace_response(response)
        self.assertEqual(len(steps), 3)
        self.assertTrue(all(s.token_type == TraceTokenType.STEP for s in steps))

    def test_step_case_insensitive(self):
        response = "<|step|> up 20"
        steps = parse_trace_response(response)
        self.assertEqual(steps[0].direction, TraceDirection.UP)


class TestParseHit(unittest.TestCase):
    def test_single_hit(self):
        response = "Found a <|hit|>gate_valve<|box|>[[10,20,40,60]]<|/box|><|/hit|> on the line."
        steps = parse_trace_response(response)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].token_type, TraceTokenType.HIT)
        self.assertEqual(steps[0].symbol_class, "gate_valve")
        self.assertEqual(steps[0].symbol_bbox_view, [10, 20, 40, 60])

    def test_hit_with_spaces_in_class(self):
        response = "<|hit|>gate valve<|box|>[[1,2,3,4]]<|/box|><|/hit|>"
        steps = parse_trace_response(response)
        self.assertEqual(steps[0].symbol_class, "gate_valve")

    def test_hit_mixed_with_steps(self):
        response = (
            "<|step|> RIGHT 30 "
            "<|hit|>reducer<|box|>[[50,50,70,70]]<|/box|><|/hit|> "
            "<|step|> RIGHT 20"
        )
        steps = parse_trace_response(response)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].token_type, TraceTokenType.STEP)
        self.assertEqual(steps[1].token_type, TraceTokenType.HIT)
        self.assertEqual(steps[2].token_type, TraceTokenType.STEP)


class TestParseTerm(unittest.TestCase):
    def test_term_with_box(self):
        response = "Reached <|term|>pump<|box|>[[100,200,150,260]]<|/box|><|/term|>"
        steps = parse_trace_response(response)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].token_type, TraceTokenType.TERM)
        self.assertEqual(steps[0].symbol_class, "pump")
        self.assertEqual(steps[0].symbol_bbox_view, [100, 200, 150, 260])

    def test_term_no_box(self):
        response = "Line goes off <|term|>crop_edge<|/term|>"
        steps = parse_trace_response(response)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].token_type, TraceTokenType.TERM)
        self.assertEqual(steps[0].symbol_class, "crop_edge")
        self.assertIsNone(steps[0].symbol_bbox_view)

    def test_term_with_tag(self):
        response = "<|term|>pump<|box|>[[10,10,50,50]]<|/box|> tag=P-2512<|/term|>"
        steps = parse_trace_response(response)
        self.assertEqual(steps[0].symbol_tag, "P-2512")

    def test_term_no_pipe_found(self):
        response = "I cannot see a pipe here. <|term|>no_pipe_found<|/term|>"
        steps = parse_trace_response(response)
        self.assertEqual(steps[0].token_type, TraceTokenType.TERM)
        self.assertEqual(steps[0].symbol_class, "no_pipe_found")


class TestParseEmpty(unittest.TestCase):
    def test_empty_response(self):
        steps = parse_trace_response("")
        self.assertEqual(steps, [])

    def test_no_tokens(self):
        steps = parse_trace_response("The line appears to go right for about 50 pixels then bends down.")
        self.assertEqual(steps, [])

    def test_malformed_token(self):
        response = "<|step|> RIGHT (no distance)"
        steps = parse_trace_response(response)
        self.assertEqual(steps, [])  # shouldn't match


class TestHelpers(unittest.TestCase):
    def test_has_terminal_true(self):
        steps = [
            TraceStep(token_type=TraceTokenType.STEP, direction=TraceDirection.RIGHT, distance_px=10),
            TraceStep(token_type=TraceTokenType.TERM, symbol_class="pump"),
        ]
        self.assertTrue(has_terminal(steps))

    def test_has_terminal_false(self):
        steps = [
            TraceStep(token_type=TraceTokenType.STEP, direction=TraceDirection.RIGHT, distance_px=10),
        ]
        self.assertFalse(has_terminal(steps))

    def test_last_terminal(self):
        steps = [
            TraceStep(token_type=TraceTokenType.HIT, symbol_class="valve"),
            TraceStep(token_type=TraceTokenType.TERM, symbol_class="pump"),
        ]
        self.assertEqual(last_terminal(steps).symbol_class, "pump")

    def test_last_terminal_none(self):
        steps = [TraceStep(token_type=TraceTokenType.STEP, direction=TraceDirection.RIGHT, distance_px=10)]
        self.assertIsNone(last_terminal(steps))

    def test_total_trace_distance(self):
        steps = [
            TraceStep(token_type=TraceTokenType.STEP, direction=TraceDirection.RIGHT, distance_px=30),
            TraceStep(token_type=TraceTokenType.HIT, symbol_class="valve"),
            TraceStep(token_type=TraceTokenType.STEP, direction=TraceDirection.DOWN, distance_px=20),
            TraceStep(token_type=TraceTokenType.TERM, symbol_class="pump"),
        ]
        self.assertEqual(total_trace_distance(steps), 50)


if __name__ == "__main__":
    unittest.main()
