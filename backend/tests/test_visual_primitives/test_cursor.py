"""Tests for cursor module — PipelineCursor state, cropping, markers, movement."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from garnet.visual_primitives.cursor import PipelineCursor


class TestPipelineCursorInit(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((500, 600, 3), dtype=np.uint8)
        self.img[:, :] = [255, 255, 255]  # white background

    def test_init_sets_position(self):
        c = PipelineCursor(image=self.img, x=150, y=200, direction="RIGHT")
        self.assertEqual(c.x, 150)
        self.assertEqual(c.y, 200)
        self.assertEqual(c.direction, "RIGHT")

    def test_init_records_start_in_visited(self):
        c = PipelineCursor(image=self.img, x=100, y=100, direction="RIGHT")
        self.assertIn((100, 100), c._visited)

    def test_default_crop_size(self):
        c = PipelineCursor(image=self.img, x=200, y=200, direction="RIGHT")
        self.assertEqual(c.crop_size, 300)


class TestPipelineCursorCrop(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((500, 600, 3), dtype=np.uint8)
        self.img[:, :] = [255, 255, 255]

    def test_crop_centered(self):
        c = PipelineCursor(image=self.img, x=300, y=250, direction="RIGHT", crop_size=200)
        crop, meta = c.crop_view()
        self.assertEqual(meta["crop_w"], 200)
        self.assertEqual(meta["crop_h"], 200)
        self.assertEqual(meta["cursor_x_view"], 100)  # 300 - 200
        self.assertEqual(meta["cursor_y_view"], 100)  # 250 - 150

    def test_crop_near_left_edge(self):
        c = PipelineCursor(image=self.img, x=50, y=200, direction="RIGHT", crop_size=200)
        crop, meta = c.crop_view()
        self.assertEqual(meta["crop_x1"], 0)
        self.assertEqual(meta["cursor_x_view"], 50)

    def test_crop_near_right_edge(self):
        c = PipelineCursor(image=self.img, x=580, y=200, direction="RIGHT", crop_size=200)
        crop, meta = c.crop_view()
        self.assertEqual(meta["crop_x2"], 600)
        self.assertEqual(meta["cursor_x_view"], 580 - meta["crop_x1"])

    def test_crop_returns_pil_image(self):
        c = PipelineCursor(image=self.img, x=300, y=250, direction="RIGHT")
        crop, meta = c.crop_view()
        self.assertIsInstance(crop, Image.Image)

    def test_crop_metadata_correct(self):
        c = PipelineCursor(image=self.img, x=300, y=250, direction="RIGHT")
        crop, meta = c.crop_view()
        self.assertIn("crop_x1", meta)
        self.assertIn("crop_y1", meta)
        self.assertIn("cursor_x_view", meta)
        self.assertIn("cursor_y_view", meta)
        self.assertIn("crop_w", meta)
        self.assertIn("crop_h", meta)


class TestPipelineCursorMarkers(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((500, 600, 3), dtype=np.uint8)
        self.img[:, :] = [255, 255, 255]

    def test_draw_cursor_marker_returns_image(self):
        c = PipelineCursor(image=self.img, x=200, y=200, direction="RIGHT")
        crop, meta = c.crop_view()
        marked = c.draw_cursor_marker(crop, meta)
        self.assertIsInstance(marked, Image.Image)
        self.assertEqual(marked.size, crop.size)

    def test_draw_visited_path_new(self):
        c = PipelineCursor(image=self.img, x=200, y=200, direction="RIGHT")
        crop, meta = c.crop_view()
        marked = c.draw_visited_path(crop, meta)
        self.assertIsInstance(marked, Image.Image)

    def test_draw_visited_with_multiple_points(self):
        c = PipelineCursor(image=self.img, x=200, y=200, direction="RIGHT")
        c.advance("RIGHT", 50)
        c.advance("RIGHT", 30)
        crop, meta = c.crop_view()
        marked = c.draw_visited_path(crop, meta)
        self.assertIsInstance(marked, Image.Image)


class TestPipelineCursorMovement(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((500, 600, 3), dtype=np.uint8)
        self.img[:, :] = [255, 255, 255]

    def test_advance_right(self):
        c = PipelineCursor(image=self.img, x=100, y=100, direction="RIGHT")
        c.advance("RIGHT", 50)
        self.assertEqual(c.x, 150)
        self.assertEqual(c.y, 100)

    def test_advance_left(self):
        c = PipelineCursor(image=self.img, x=100, y=100, direction="LEFT")
        c.advance("LEFT", 30)
        self.assertEqual(c.x, 70)

    def test_advance_down(self):
        c = PipelineCursor(image=self.img, x=100, y=100, direction="DOWN")
        c.advance("DOWN", 40)
        self.assertEqual(c.y, 140)

    def test_advance_up(self):
        c = PipelineCursor(image=self.img, x=100, y=200, direction="UP")
        c.advance("UP", 40)
        self.assertEqual(c.y, 160)

    def test_advance_records_path(self):
        c = PipelineCursor(image=self.img, x=100, y=100, direction="RIGHT")
        self.assertEqual(c.path_length, 0)
        c.advance("RIGHT", 10)
        self.assertEqual(c.path_length, 10)

    def test_advance_adds_to_visited(self):
        c = PipelineCursor(image=self.img, x=100, y=100, direction="RIGHT")
        initial = c.visited_count
        c.advance("RIGHT", 5)
        self.assertGreater(c.visited_count, initial)


class TestPipelineCursorCoordinates(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((500, 600, 3), dtype=np.uint8)
        self.img[:, :] = [255, 255, 255]

    def test_global_coords(self):
        c = PipelineCursor(image=self.img, x=123, y=456, direction="RIGHT")
        self.assertEqual(c.global_coords(), (123, 456))

    def test_norm_coords(self):
        c = PipelineCursor(image=self.img, x=300, y=250, direction="RIGHT")
        norm = c.norm_coords(600, 500)
        # 300/600*999 ≈ 499.5 → round to 500; 250/500*999 ≈ 499.5 → round to 500
        self.assertEqual(norm, [500, 500])

    def test_view_to_global(self):
        c = PipelineCursor(image=self.img, x=200, y=200, direction="RIGHT")
        _, meta = c.crop_view()
        gx, gy = c.view_to_global(100, 50, meta)
        self.assertEqual(gx, 100 + meta["crop_x1"])
        self.assertEqual(gy, 50 + meta["crop_y1"])

    def test_is_near_edge(self):
        c = PipelineCursor(image=self.img, x=5, y=200, direction="RIGHT")
        self.assertTrue(c.is_near_edge())
        c2 = PipelineCursor(image=self.img, x=300, y=250, direction="RIGHT")
        self.assertFalse(c2.is_near_edge())


if __name__ == "__main__":
    unittest.main()
