"""Unit tests for the visual-primitives canvas module."""

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from garnet.visual_primitives.canvas import (
    load_canvas,
    normalize,
    denormalize,
    make_global_view,
    pixel_to_norm_bbox,
    norm_to_pixel_bbox,
    CanvasConfig,
    CanvasMeta,
)


class TestLoadCanvas(unittest.TestCase):
    """Tests for load_canvas."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_test_image(self, w=800, h=600) -> str:
        """Create a simple test PNG and return its path."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[100:200, 150:250] = (0, 255, 0)  # green square
        path = Path(self.tmpdir) / "test.png"
        cv2.imwrite(str(path), img)
        return str(path)

    def test_load_png(self):
        path = self._make_test_image()
        img, meta = load_canvas(path)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (600, 800, 3))
        self.assertEqual(meta.width, 800)
        self.assertEqual(meta.height, 600)
        self.assertEqual(meta.channels, 3)

    def test_load_nonexistent(self):
        with self.assertRaises(FileNotFoundError):
            load_canvas("/nonexistent/path.png")

    def test_load_grayscale(self):
        """Grayscale images should be converted to 3-channel BGR."""
        path = Path(self.tmpdir) / "gray.png"
        img = np.zeros((100, 200), dtype=np.uint8)
        cv2.imwrite(str(path), img)
        _, meta = load_canvas(str(path))
        self.assertEqual(meta.channels, 3)


class TestCoordinateNormalizer(unittest.TestCase):
    """Tests for normalize / denormalize round-trip."""

    def test_normalize_origin(self):
        self.assertEqual(normalize(0, 0, 1000, 1000), (0, 0))

    def test_normalize_max(self):
        self.assertEqual(normalize(1000, 1000, 1000, 1000), (999, 999))

    def test_normalize_midpoint(self):
        self.assertEqual(normalize(500, 500, 1000, 1000), (500, 500))

    def test_roundtrip(self):
        for px, py, w, h in [
            (0, 0, 1000, 1000),
            (500, 300, 800, 600),
            (799, 599, 800, 600),
            (1234, 567, 2000, 1500),
        ]:
            nx, ny = normalize(px, py, w, h)
            rx, ry = denormalize(nx, ny, w, h)
            # Round-trip should be close (within 2px due to rounding).
            self.assertLess(abs(rx - px), 3, f"px={px},py={py},w={w},h={h} -> nx={nx},ny={ny} -> rx={rx},ry={ry}")
            self.assertLess(abs(ry - py), 3)

    def test_clamp(self):
        """Out-of-bounds input should be clamped to [0, 999]."""
        cfg = CanvasConfig(normalized_range=999)
        self.assertEqual(normalize(-100, -100, 1000, 1000, cfg=cfg), (0, 0))
        self.assertEqual(normalize(2000, 2000, 1000, 1000, cfg=cfg), (999, 999))


class TestBBoxConversion(unittest.TestCase):
    """Tests for pixel_to_norm_bbox and norm_to_pixel_bbox."""

    def test_full_canvas(self):
        bbox = (0, 0, 1000, 1000)
        norm = pixel_to_norm_bbox(bbox, 1000, 1000)
        self.assertEqual(norm, [0, 0, 999, 999])

    def test_mid_canvas(self):
        bbox = (200, 150, 600, 450)
        norm = pixel_to_norm_bbox(bbox, 800, 600)
        self.assertEqual(norm, [250, 250, 749, 749])


class TestGlobalView(unittest.TestCase):
    """Tests for make_global_view downsampling."""

    def test_small_image_passthrough(self):
        """Image smaller than max_dim should pass through unchanged."""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cfg = CanvasConfig(global_view_max_dim=1200)
        result = make_global_view(img, cfg=cfg)
        self.assertEqual(result.shape, (300, 400, 3))

    def test_large_image_downsample(self):
        """Image larger than max_dim should be downsampled."""
        img = np.zeros((2400, 3600, 3), dtype=np.uint8)
        cfg = CanvasConfig(global_view_max_dim=1200)
        result = make_global_view(img, cfg=cfg)
        # Should be max 1200 on the longer edge.
        self.assertLessEqual(max(result.shape[:2]), 1200)
        # Should maintain aspect ratio approximately.
        ratio = result.shape[1] / result.shape[0]
        orig_ratio = 3600 / 2400
        self.assertAlmostEqual(ratio, orig_ratio, delta=0.02)

    def test_adaptive_to_size(self):
        """Different sheet sizes should get appropriately scaled views."""
        cfg = CanvasConfig(global_view_max_dim=1200)
        small = np.zeros((600, 800, 3), dtype=np.uint8)  # already small — passthrough
        large = np.zeros((4800, 6400, 3), dtype=np.uint8)  # large — downsampled
        v_small = make_global_view(small, cfg=cfg)
        v_large = make_global_view(large, cfg=cfg)
        # Small image passes through unchanged.
        self.assertEqual(v_small.shape[:2], (600, 800))
        # Large image is downsampled (max dim <= 1200, smaller than original).
        self.assertLessEqual(max(v_large.shape[:2]), 1200)
        self.assertLess(v_large.shape[0], large.shape[0])
        self.assertLess(v_large.shape[1], large.shape[1])


if __name__ == "__main__":
    unittest.main()
