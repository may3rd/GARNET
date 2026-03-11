import unittest

import numpy as np

from garnet.pipe_mask import (
    _draw_overlay,
    _filter_small_components,
    _select_object_regions_for_suppression,
    _suppress_boxes,
    run_pipe_mask_stage,
)


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

    def test_select_object_regions_for_suppression_preserves_topology_classes(self) -> None:
        selected, suppressed_counts, preserved_counts = _select_object_regions_for_suppression(
            [
                {"class_name": "pump"},
                {"class_name": "arrow"},
                {"class_name": "node"},
                {"class_name": "connection"},
            ],
            preserve_classes=("arrow", "node", "connection"),
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["class_name"], "pump")
        self.assertEqual(suppressed_counts, {"pump": 1})
        self.assertEqual(preserved_counts, {"arrow": 1, "connection": 1, "node": 1})

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

    def test_run_pipe_mask_stage_preserves_arrow_and_node_objects(self) -> None:
        image_bgr = np.full((30, 30, 3), 255, dtype=np.uint8)
        binary = np.zeros((30, 30), dtype=np.uint8)
        binary[10:20, 10:20] = 255

        result = run_pipe_mask_stage(
            image_bgr=image_bgr,
            gray_image=np.mean(image_bgr, axis=2).astype(np.uint8),
            adaptive_mask=binary,
            otsu_mask=binary,
            ocr_regions=[],
            object_regions=[
                {"class_name": "pump", "bbox": {"x_min": 10, "y_min": 10, "x_max": 14, "y_max": 14}},
                {"class_name": "arrow", "bbox": {"x_min": 15, "y_min": 10, "x_max": 19, "y_max": 14}},
                {"class_name": "node", "bbox": {"x_min": 10, "y_min": 15, "x_max": 14, "y_max": 19}},
            ],
            image_id="sample.png",
            object_inset=0,
            min_component_area=1,
            preserve_object_classes=("arrow", "node"),
        )

        mask = result["mask_image"]
        self.assertEqual(int(mask[10:15, 10:15].sum()), 0)
        self.assertGreater(int(mask[10:15, 15:20].sum()), 0)
        self.assertGreater(int(mask[15:20, 10:15].sum()), 0)
        self.assertEqual(result["summary"]["suppressed_object_class_counts"], {"pump": 1})
        self.assertEqual(result["summary"]["preserved_object_class_counts"], {"arrow": 1, "node": 1})


if __name__ == "__main__":
    unittest.main()
