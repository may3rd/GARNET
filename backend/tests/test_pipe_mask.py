import unittest

import numpy as np

from garnet.pipe_mask import _draw_overlay, _filter_small_components, _suppress_boxes, run_pipe_mask_stage


class PipeMaskTests(unittest.TestCase):
    def test_suppress_boxes_clears_masked_regions(self) -> None:
        mask = np.ones((20, 20), dtype=np.uint8) * 255
        boxes = [{"bbox": {"x_min": 5, "y_min": 6, "x_max": 10, "y_max": 12}}]

        suppressed, removed = _suppress_boxes(mask, boxes, padding=0)

        self.assertEqual(removed, 42)
        self.assertTrue(np.all(suppressed[6:13, 5:11] == 0))

    def test_filter_small_components_removes_obvious_specks(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        mask[10:18, 10:18] = 255

        filtered, removed = _filter_small_components(mask, min_area=10)

        self.assertEqual(removed, 1)
        self.assertEqual(int(filtered[1:3, 1:3].sum()), 0)
        self.assertGreater(int(filtered[10:18, 10:18].sum()), 0)

    def test_draw_overlay_marks_pipe_mask_in_blue(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 6:11] = 255

        overlay = _draw_overlay(image, mask)

        self.assertTrue(np.array_equal(overlay[5, 6], np.array([255, 0, 0], dtype=np.uint8)))

    def test_run_pipe_mask_stage_returns_summary_and_images(self) -> None:
        image_bgr = np.full((30, 30, 3), 255, dtype=np.uint8)
        image_bgr[15, 3:27] = 0
        image_bgr[4:8, 4:9] = 0

        result = run_pipe_mask_stage(
            image_bgr=image_bgr,
            gray_image=np.mean(image_bgr, axis=2).astype(np.uint8),
            adaptive_mask=(np.mean(image_bgr, axis=2) < 200).astype(np.uint8) * 255,
            otsu_mask=(np.mean(image_bgr, axis=2) < 200).astype(np.uint8) * 255,
            ocr_regions=[{"bbox": {"x_min": 4, "y_min": 4, "x_max": 8, "y_max": 7}}],
            object_regions=[{"bbox": {"x_min": 20, "y_min": 12, "x_max": 24, "y_max": 18}}],
            image_id="sample.png",
            ocr_padding=0,
            object_inset=1,
            min_component_area=5,
        )

        self.assertIn("mask_image", result)
        self.assertIn("overlay_image", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["image_id"], "sample.png")
        self.assertIn("mask_pixel_count", result["summary"])
        self.assertIn("ocr_suppression_pixel_count", result["summary"])
        self.assertIn("object_suppression_pixel_count", result["summary"])


if __name__ == "__main__":
    unittest.main()
