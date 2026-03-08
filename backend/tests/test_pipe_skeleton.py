import unittest

import numpy as np

from garnet.pipe_skeleton import _draw_overlay, _skeletonize_mask, run_pipe_skeleton_stage


class PipeSkeletonTests(unittest.TestCase):
    def test_skeletonize_mask_thins_linework(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[8:12, 2:18] = 255

        skeleton = _skeletonize_mask(mask)

        self.assertGreater(int(np.count_nonzero(mask)), int(np.count_nonzero(skeleton)))
        self.assertGreater(int(np.count_nonzero(skeleton)), 0)

    def test_draw_overlay_marks_skeleton_in_green(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        skeleton = np.zeros((10, 10), dtype=np.uint8)
        skeleton[4, 5] = 255

        overlay = _draw_overlay(image, skeleton)

        self.assertTrue(np.array_equal(overlay[4, 5], np.array([0, 255, 0], dtype=np.uint8)))

    def test_run_pipe_skeleton_stage_returns_summary_and_images(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 2:18] = 255
        mask[2:18, 10] = 255

        result = run_pipe_skeleton_stage(
            image_bgr=image,
            sealed_mask=mask,
            image_id="sample.png",
        )

        self.assertIn("skeleton_image", result)
        self.assertIn("overlay_image", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["image_id"], "sample.png")
        self.assertIn("skeleton_pixel_count", result["summary"])
        self.assertIn("pixel_reduction", result["summary"])


if __name__ == "__main__":
    unittest.main()
