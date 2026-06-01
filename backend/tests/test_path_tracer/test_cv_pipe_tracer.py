import unittest

import numpy as np

from garnet.path_tracer.cv_pipe_tracer import CVPipeTracer


class TestCvPipeTracerTeeThroughTurns(unittest.TestCase):
    def test_bidirectional_turn_leg_detects_tee_through_not_elbow(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[40, 10:51] = 255
        mask[30:41, 50] = 255
        tee = CVPipeTracer(mask)

        self.assertFalse(tee._has_bidirectional_turn_leg(50, 40, "UP"))

        mask[41:61, 50] = 255
        tee = CVPipeTracer(mask)

        self.assertTrue(tee._has_bidirectional_turn_leg(50, 40, "UP"))


if __name__ == "__main__":
    unittest.main()
