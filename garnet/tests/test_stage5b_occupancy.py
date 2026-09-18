"""CV walker unit tests: the occupancy cases that decide topology.

Each test builds a synthetic 1-px mask and asserts what trace() must do at a
junction.  These encode the decision table in pid-raster-pipe-tracing:
forward-open + side stub = keep going (seed, not terminal); a tee from the
stem stops as tee_junction; an elbow turns exactly once.
"""
import unittest
import numpy as np
from garnet.stage5b.cv_pipe_tracer import CVPipeTracer


def mask_cross(w=201, h=201, cy=100, cx=100):
    m = np.zeros((h, w), dtype=np.uint8)
    m[cy, :] = 255
    m[:, cx] = 255
    return m


class CrossTests(unittest.TestCase):
    def test_walk_does_not_stop_at_cross_center(self):
        tracer = CVPipeTracer(mask_cross())
        r = tracer.trace(0, 100, 'RIGHT', source_obj_id='equip_a:port_01')
        self.assertEqual(r.status, 'ok')
        # Keep going through the 4-way crossing: terminal is the far sheet
        # edge / end of ink, not the crossing pixel.
        self.assertNotEqual(r.terminal_x, 100)
        self.assertGreater(r.terminal_x, 150)
        # Two unvisited branch seeds must remain (up + down arms).
        self.assertEqual(int(tracer.visited[0:100, 100].sum()), 0)
        self.assertEqual(int(tracer.visited[101:, 100].sum()), 0)


class TeeTests(unittest.TestCase):
    def test_walk_from_stem_stops_as_tee(self):
        # A true T: stem below the crossbar only.  (A full-height column plus a
        # full-width bar is a 4-way CROSS, whose walk correctly continues to the
        # sheet edge -- see CrossTests.)
        m = np.zeros((201, 201), dtype=np.uint8)
        m[100:, 100] = 255       # stem (vertical), below the crossbar
        m[100, :] = 255          # crossbar
        tracer = CVPipeTracer(m)
        r = tracer.trace(100, 200, 'UP', source_obj_id='equip_b:port_01')
        self.assertEqual(r.terminal_type, 'tee_junction')
        self.assertEqual((r.terminal_x, r.terminal_y), (100, 100))
        self.assertIsNone(r.terminal_obj_id)
        # Two seeds remain on the crossbar arms.
        self.assertEqual(int(tracer.visited[100, :100].sum()), 0)
        self.assertEqual(int(tracer.visited[100, 101:].sum()), 0)

    def test_crossbar_walk_continue_to_sheet_edge(self):
        # Same geometry approached from an arm: the stem is a side stub, so the
        # walk keeps going rather than stopping as a tee.
        m = np.zeros((201, 201), dtype=np.uint8)
        m[100:, 100] = 255
        m[100, :] = 255
        r = CVPipeTracer(m).trace(0, 100, 'RIGHT', source_obj_id='equip_b:port_02')
        self.assertEqual(r.status, 'ok')
        self.assertNotEqual(r.terminal_type, 'tee_junction')
        self.assertGreater(r.terminal_x, 150)


class ElbowTests(unittest.TestCase):
    def test_elbow_turns_once_and_no_tee(self):
        m = np.zeros((201, 201), dtype=np.uint8)
        m[100, 0:100] = 255      # horizontal leg
        m[0:100, 100] = 255      # vertical leg (elbow at (100,100))
        tracer = CVPipeTracer(m)
        r = tracer.trace(0, 100, 'RIGHT', source_obj_id='equip_c:port_01')
        self.assertEqual(r.status, 'ok')
        self.assertNotEqual(r.terminal_type, 'tee_junction')
        self.assertEqual(len(r.turns), 1)
        self.assertEqual(r.terminal_x, 100)
        self.assertLess(r.terminal_y, 50)


class FlangeTickTests(unittest.TestCase):
    def test_short_flange_tick_is_not_a_turn(self):
        m = np.zeros((201, 401), dtype=np.uint8)
        m[100, 0:400] = 255
        m[92:108, 150] = 255     # short perpendicular tick (flange mark)
        tracer = CVPipeTracer(m)
        r = tracer.trace(0, 100, 'RIGHT', source_obj_id='equip_d:port_01')
        self.assertEqual(r.status, 'ok')
        self.assertEqual(len(r.turns), 0)
        self.assertGreater(r.terminal_x, 350)


class ValvePassThroughTests(unittest.TestCase):
    def test_inline_valve_bbox_jump_keeps_single_polyline(self):
        m = np.zeros((201, 401), dtype=np.uint8)
        m[100, 0:400] = 255
        tracer = CVPipeTracer(m)
        tracer.set_inline_symbols([
            {'class_name': 'gate valve', 'Left': 180, 'Top': 85,
             'Width': 41, 'Height': 31, 'id': 'obj_valve1'},
        ])
        r = tracer.trace(0, 100, 'RIGHT', source_obj_id='equip_d:port_01')
        self.assertEqual(r.status, 'ok')
        self.assertGreater(r.terminal_x, 350)
        # One polyline through the valve: segments stay on one axis run.
        self.assertTrue(all(
            (s.y1 == s.y2) for s in r.segments
        ) or all((s.x1 == s.x2) for s in r.segments))


if __name__ == '__main__':
    unittest.main()
