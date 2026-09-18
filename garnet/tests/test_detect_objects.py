"""Checks for garnet/detect_objects.py — output-path guard, overlay label placement, class colours.

Run: cd garnet && ../.venv/bin/python -m unittest discover -s tests -p 'test_detect_objects.py' -v
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "detect_objects.py"
_spec = importlib.util.spec_from_file_location("detect_objects", MODULE_PATH)
detect_objects = importlib.util.module_from_spec(_spec)
sys.modules["detect_objects"] = detect_objects
_spec.loader.exec_module(detect_objects)


def _overlay_ink(image: np.ndarray, region: tuple[int, int, int, int]) -> int:
    y0, y1, x0, x1 = region
    return int((image[y0:y1, x0:x1] > 0).sum())


class OutputGuardTests(unittest.TestCase):
    """A reserved output path aliasing the input raster must be refused, or the sheet is lost.

    Driven through `main()` because that is the contract: exit code 1 and an untouched input
    file. The guard runs before `detect()`, so no weight or torch load happens here — the
    weight is a placeholder and the raster is not even a readable image.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._stderr = contextlib.redirect_stderr(io.StringIO())
        self._stderr.__enter__()
        self.weight = self.tmp / "placeholder.pt"
        self.weight.write_bytes(b"not a real weight")

    def tearDown(self) -> None:
        self._stderr.__exit__(None, None, None)
        self._tmp.cleanup()

    def _run(self, image: Path, out: Path) -> int:
        return detect_objects.main(
            ["--image", str(image), "--weight", str(self.weight), "--out", str(out)]
        )

    def test_refuses_when_out_path_is_the_image(self) -> None:
        image = self.tmp / "sheet.png"
        image.write_bytes(b"pretend raster")
        self.assertEqual(self._run(image, image), 1)
        self.assertEqual(image.read_bytes(), b"pretend raster")

    def test_refuses_when_overlay_path_is_the_image(self) -> None:
        # `--out sheet.json` implies overlay `sheet_overlay.png`, so name the raster that.
        image = self.tmp / "sheet_overlay.png"
        image.write_bytes(b"pretend raster")
        self.assertEqual(self._run(image, self.tmp / "sheet.json"), 1)
        self.assertEqual(image.read_bytes(), b"pretend raster")

    def test_refuses_when_out_path_is_a_hardlink_to_the_image(self) -> None:
        image = self.tmp / "sheet.png"
        image.write_bytes(b"pretend raster")
        link = self.tmp / "alias.png"
        link.hardlink_to(image)
        self.assertEqual(self._run(image, link), 1)
        self.assertEqual(image.read_bytes(), b"pretend raster")

    def test_distinct_output_paths_are_not_refused(self) -> None:
        # Not a readable raster, so detect() raises — which is the proof the guard let it
        # through instead of short-circuiting on a false positive.
        image = self.tmp / "sheet.png"
        image.write_bytes(b"pretend raster")
        with self.assertRaises(FileNotFoundError):
            self._run(image, self.tmp / "objects.json")

    def test_skipped_overlay_is_not_reserved(self) -> None:
        # With --no-overlay the overlay path is never written, so it must not trigger the guard.
        image = self.tmp / "sheet_overlay.png"
        image.write_bytes(b"pretend raster")
        with self.assertRaises(FileNotFoundError):
            detect_objects.main(
                ["--image", str(image), "--weight", str(self.weight),
                 "--out", str(self.tmp / "sheet.json"), "--no-overlay"]
            )
        self.assertEqual(image.read_bytes(), b"pretend raster")


class DrawOverlayTests(unittest.TestCase):
    """Labels are the only annotation; they must land inside the canvas."""

    def test_label_for_tall_bottom_edge_box_is_not_drawn_off_canvas(self) -> None:
        # A box reaching both edges leaves no room above OR below: pre-fix the label banner
        # was placed at y2 + th + pad, entirely off-image, so no white label text was drawn.
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        objs = [{"class_name": "instrument tag", "confidence": 0.90,
                 "bbox": {"x_min": 20, "y_min": 5, "x_max": 300, "y_max": 399}}]
        overlay = detect_objects.draw_overlay(image, objs)
        white_text = ((overlay[:, :, 0] > 240) & (overlay[:, :, 1] > 240) & (overlay[:, :, 2] > 240))
        self.assertGreater(int(white_text.sum()), 0)

    def test_label_for_right_edge_box_stays_in_canvas(self) -> None:
        # The label banner is as wide as its text; pre-fix it started at x1 and ran off the
        # right edge, so only the box outline was drawn in the label's band.
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        objs = [{"class_name": "instrument tag", "confidence": 0.9,
                 "bbox": {"x_min": 580, "y_min": 150, "x_max": 599, "y_max": 200}}]
        overlay = detect_objects.draw_overlay(image, objs)
        self.assertGreater(_overlay_ink(overlay, (130, 155, 0, 600)), 2000)

    def test_box_outlines_are_drawn_for_every_object(self) -> None:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        objs = [
            {"class_name": "node", "confidence": 0.9, "bbox": {"x_min": 10, "y_min": 10, "x_max": 60, "y_max": 60}},
            {"class_name": "arrow", "confidence": 0.8, "bbox": {"x_min": 100, "y_min": 100, "x_max": 150, "y_max": 150}},
        ]
        overlay = detect_objects.draw_overlay(image, objs)
        self.assertGreater(_overlay_ink(overlay, (10, 62, 10, 62)), 0)
        self.assertGreater(_overlay_ink(overlay, (100, 152, 100, 152)), 0)


class ClassColorTests(unittest.TestCase):
    def test_distinct_classes_get_distinct_colors(self) -> None:
        names = ["arrow", "node", "gate valve", "instrument tag", "line number"]
        colors = [detect_objects._class_color(n) for n in names]
        self.assertEqual(len(set(colors)), len(names))


if __name__ == "__main__":
    unittest.main()
