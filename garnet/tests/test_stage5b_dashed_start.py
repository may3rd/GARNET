"""Regression: a periodic signal lead must not seed a process-pipe walk."""
import unittest
import numpy as np
from garnet.stage5b.cv_pipe_tracer import CVPipeTracer


class DashedStartTests(unittest.TestCase):
    def test_dashed_start_does_not_claim_signal_pixels(self):
        # Dash geometry is taken from the real sheet: instrument leads measure
        # 3-15px dashes (p95=15) with ~6px gaps.  The previous 31px "dashes" were
        # longer than any dash the sheet produces and collided with the run
        # length of real pipe between two symbol glyphs.
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                mask = np.zeros((100, 500), dtype=np.uint8)
                for x in range(20, 450, 26):
                    mask[49:52, x:x + 12] = 255     # 12px dash, 14px gap
                if vertical:
                    mask = mask.T.copy()
                tracer = CVPipeTracer(mask)
                result = tracer.trace(50 if vertical else 20,
                                      20 if vertical else 50,
                                      'DOWN' if vertical else 'RIGHT')
                self.assertEqual(result.status, 'no_pipe')
                self.assertEqual(result.segments, [])
                self.assertFalse(tracer.visited.any())

    def test_solid_and_single_gap_are_not_rejected(self):
        for gap in (False, True):
            mask = np.zeros((100, 500), dtype=np.uint8)
            mask[49:52, 20:450] = 255
            if gap:
                mask[49:52, 70:80] = 0
            result = CVPipeTracer(mask).trace(20, 50, 'RIGHT')
            self.assertEqual(result.status, 'ok')
            self.assertGreater(result.terminal_x, 400)

    def test_pipe_with_dense_inline_symbols_is_not_rejected(self):
        # Regression: three inline-symbol gaps inside the 160px screen window,
        # each separated by a substantial run of real pipe ink.  Counting gaps
        # alone flagged this pipe as a dashed signal lead; the ink-run gate must
        # let it through.
        #
        # Note the spacing is the point: gaps only 25px apart behind 25px ink
        # runs are genuinely indistinguishable from a dash train and stay
        # rejected -- that residual is documented on _looks_like_dashed_lead.
        mask = np.zeros((120, 900), dtype=np.uint8)
        mask[58:61, 20:880] = 255
        for x in (100, 155, 210):
            mask[58:61, x:x + 3] = 0       # 3px gap, >48px ink between gaps
        tracer = CVPipeTracer(mask)
        result = tracer.trace(20, 59, 'RIGHT')
        self.assertEqual(result.status, 'ok')
        self.assertGreater(result.terminal_x, 800)
        self.assertTrue(tracer.visited.any())

    def test_dash_train_short_ink_still_rejected(self):
        # A proper dash train: 12px dashes with 12px gaps, repeatedly.
        mask = np.zeros((120, 900), dtype=np.uint8)
        for x in range(100, 400, 24):          # 12px dash, 12px gap
            mask[58:61, x:x + 12] = 255
        tracer = CVPipeTracer(mask)
        self.assertTrue(tracer._looks_like_dashed_lead(100, 59, 1, 0))

    def test_long_ink_run_with_short_gaps_is_pipe_not_lead(self):
        # The discriminator itself: long ink between short gaps = pipe.
        mask = np.zeros((120, 900), dtype=np.uint8)
        mask[58:61, 20:880] = 255
        for x in (120, 240, 360):                    # gaps 20px, ink 100px
            mask[58:61, x:x + 20] = 0
        tracer = CVPipeTracer(mask)
        sx, sy = 20, 59
        self.assertFalse(tracer._looks_like_dashed_lead(sx, sy, 1, 0))

    def test_short_dashes_with_short_gaps_is_lead(self):
        # Same gap count, dash-sized ink runs = signal lead.
        mask = np.zeros((120, 900), dtype=np.uint8)
        for x in range(20, 260, 24):                 # 10px dash, 14px gap
            mask[58:61, x:x + 10] = 255
        tracer = CVPipeTracer(mask)
        self.assertTrue(tracer._looks_like_dashed_lead(20, 59, 1, 0))

    def test_equipment_port_start_is_never_screened(self):
        # An equipment nozzle start is real pipe by construction, even when the
        # ink immediately beyond it happens to look dashed.
        mask = np.zeros((120, 900), dtype=np.uint8)
        for x in range(20, 500, 26):                 # dash train from x=20
            mask[58:61, x:x + 10] = 255
        tracer = CVPipeTracer(mask)
        result = tracer.trace(20, 59, 'RIGHT', source_obj_id='equip_v_2501:port_01')
        self.assertNotEqual(result.status, 'no_pipe')


if __name__ == '__main__':
    unittest.main()
