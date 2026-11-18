"""Checks for Stage 5b inline-symbol seeding — the mask ring scan that gives a PSV a walk start.

`_inline_symbol_seeds` is the only thing standing between a relief valve and an untraced
pressure-side line (0002A L23 went 0.02 -> 0.97 GT path coverage once it existed), so the two
behaviours that matter get asserted on synthetic masks: it finds pipe on every side of a
symbol, and it rejects a tag-leader stub that dies after a few px.

Run: .venv/bin/python -m unittest discover -s garnet/tests -p "test_inline_symbol_seeds.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from garnet.path_tracer.stage5b_pipeline import Stage5bPipelineMixin  # noqa: E402


class Seeder(Stage5bPipelineMixin):
    """The mixin carries pipeline state; the seeder needs none of it."""


BBOX = {"x_min": 100, "y_min": 100, "x_max": 140, "y_max": 140}


class InlineSymbolSeedsTest(unittest.TestCase):
    def test_pipe_leaving_both_flanks_gives_a_port_per_side(self):
        mask = np.zeros((300, 300), np.uint8)
        mask[118:122, 0:300] = 255          # horizontal run straight through the glyph
        seed = Seeder()._inline_symbol_seeds(BBOX, mask)
        got = sorted(s[2] for s in seed)
        self.assertEqual(got, ["LEFT", "RIGHT"])
        for x, y, _ in seed:
            self.assertLessEqual(y, 145, "a horizontal run must seed at its own height")

    def test_a_stub_that_dies_after_a_few_px_is_rejected(self):
        mask = np.zeros((300, 300), np.uint8)
        mask[118:122, 0:300] = 255          # the real pipe (left/right)
        mask[60:64, 120:126] = 255          # 6 px leader above the glyph: tag leader, not pipe
        dirs = [s[2] for s in Seeder()._inline_symbol_seeds(BBOX, mask)]
        self.assertNotIn("UP", dirs)
        self.assertIn("LEFT", dirs)

    def test_no_pipe_near_the_glyph_returns_nothing(self):
        self.assertEqual(Seeder()._inline_symbol_seeds(BBOX, np.zeros((300, 300), np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
