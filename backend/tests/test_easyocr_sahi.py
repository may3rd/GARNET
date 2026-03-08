import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from garnet.easyocr_sahi import (
    EasyOcrSahiConfig,
    EasyOcrSahiDetectionModel,
    _draw_overlay,
    _merge_same_line_regions,
    _rotate_quad_back,
    run_easyocr_sahi,
)


class EasyOcrSahiHelperTests(unittest.TestCase):
    def test_rotate_quad_back_restores_clockwise_rotation_to_original_coordinates(self) -> None:
        original_width = 100
        original_height = 40
        rotated_quad = [
            [5.0, 10.0],
            [5.0, 30.0],
            [15.0, 30.0],
            [15.0, 10.0],
        ]

        restored = _rotate_quad_back(rotated_quad, "cw", original_width, original_height)

        self.assertEqual(restored, [[10.0, 34.0], [30.0, 34.0], [30.0, 24.0], [10.0, 24.0]])

    def test_merge_same_line_regions_combines_adjacent_text_boxes(self) -> None:
        regions = [
            {
                "id": "",
                "text": "CAUSTIC",
                "normalized_text": "CAUSTIC",
                "class": "unknown",
                "confidence": 0.91,
                "bbox": {"x_min": 10, "y_min": 10, "x_max": 90, "y_max": 30},
                "rotation": 0,
                "reading_direction": "horizontal",
                "legibility": "clear",
            },
            {
                "id": "",
                "text": "WASH",
                "normalized_text": "WASH",
                "class": "unknown",
                "confidence": 0.92,
                "bbox": {"x_min": 98, "y_min": 11, "x_max": 150, "y_max": 31},
                "rotation": 0,
                "reading_direction": "horizontal",
                "legibility": "clear",
            },
            {
                "id": "",
                "text": "FEED",
                "normalized_text": "FEED",
                "class": "unknown",
                "confidence": 0.94,
                "bbox": {"x_min": 159, "y_min": 10, "x_max": 215, "y_max": 30},
                "rotation": 0,
                "reading_direction": "horizontal",
                "legibility": "clear",
            },
        ]

        merged = _merge_same_line_regions(regions, line_merge_gap_px=16, line_merge_y_tolerance_px=8)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "CAUSTIC WASH FEED")
        self.assertEqual(merged[0]["bbox"], {"x_min": 10, "y_min": 10, "x_max": 215, "y_max": 31})

    def test_detection_model_emits_object_predictions_from_reader_regions(self) -> None:
        cfg = EasyOcrSahiConfig()
        reader = MagicMock()
        model = EasyOcrSahiDetectionModel(cfg=cfg, reader=reader)
        model.load_model()

        with patch(
            "garnet.easyocr_sahi._read_tile_with_orientations",
            return_value=[
                (
                    "none",
                    [
                        (
                            [[10.0, 10.0], [10.0, 30.0], [90.0, 30.0], [90.0, 10.0]],
                            "ITEM No.",
                            0.94,
                        )
                    ],
                )
            ],
        ):
            model.perform_inference(np.zeros((64, 128), dtype=np.uint8))
            model.convert_original_predictions(shift_amount=[[5, 7]], full_shape=[[64, 128]])

        self.assertEqual(len(model.object_prediction_list), 1)
        self.assertEqual(model.all_candidates[0]["text"], "ITEM No.")
        self.assertEqual(model.all_candidates[0]["bbox"], {"x_min": 15, "y_min": 17, "x_max": 95, "y_max": 37})

    @patch("garnet.easyocr_sahi._draw_overlay", return_value=np.zeros((32, 32, 3), dtype=np.uint8))
    @patch("garnet.easyocr_sahi.get_sliced_prediction")
    @patch("garnet.easyocr_sahi._get_reader")
    @patch("garnet.easyocr_sahi.cv2.imread", return_value=np.zeros((32, 32, 3), dtype=np.uint8))
    def test_run_easyocr_sahi_uses_sahi_postprocess_threshold(
        self,
        mock_imread: MagicMock,
        mock_get_reader: MagicMock,
        mock_get_sliced_prediction: MagicMock,
        mock_draw_overlay: MagicMock,
    ) -> None:
        mock_get_reader.return_value = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.object_prediction_list = []
        mock_get_sliced_prediction.return_value = mock_prediction

        cfg = EasyOcrSahiConfig(postprocess_match_threshold=0.1)
        run_easyocr_sahi("dummy.png", image_id="sample", cfg=cfg)

        kwargs = mock_get_sliced_prediction.call_args.kwargs
        self.assertEqual(kwargs["slice_height"], cfg.slice_height)
        self.assertEqual(kwargs["slice_width"], cfg.slice_width)
        self.assertEqual(kwargs["postprocess_match_metric"], cfg.postprocess_match_metric)
        self.assertEqual(kwargs["postprocess_match_threshold"], 0.1)
        self.assertEqual(kwargs["postprocess_type"], cfg.postprocess_type)
        self.assertFalse(kwargs["auto_slice_resolution"])
        self.assertFalse(kwargs["perform_standard_pred"])

    def test_draw_overlay_uses_blue_bounding_boxes(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        overlay = _draw_overlay(
            image,
            [
                {
                    "bbox": {"x_min": 2, "y_min": 3, "x_max": 12, "y_max": 13},
                }
            ],
        )

        self.assertTrue(np.array_equal(overlay[3, 2], np.array([255, 0, 0], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
