import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from garnet.object_detection_sahi import DetectionSahiConfig, _draw_overlay, run_object_detection_sahi


class ObjectDetectionSahiTests(unittest.TestCase):
    @patch("garnet.object_detection_sahi._build_detection_model")
    @patch("garnet.object_detection_sahi.get_sliced_prediction")
    @patch("garnet.object_detection_sahi.cv2.imread", return_value=np.zeros((64, 64, 3), dtype=np.uint8))
    def test_run_object_detection_sahi_returns_pipeline_contract(
        self,
        mock_imread: MagicMock,
        mock_get_sliced_prediction: MagicMock,
        mock_build_detection_model: MagicMock,
    ) -> None:
        del mock_imread
        mock_build_detection_model.return_value = MagicMock()

        detection = MagicMock()
        detection.bbox.to_xyxy.return_value = [10, 20, 40, 60]
        detection.category.name = "valve"
        detection.score.value = 0.91
        mock_get_sliced_prediction.return_value = MagicMock(object_prediction_list=[detection])

        result = run_object_detection_sahi("dummy.png", image_id="sample.png")

        self.assertEqual(result["objects_payload"]["image_id"], "sample.png")
        self.assertEqual(result["objects_payload"]["pass_type"], "sheet")
        self.assertEqual(len(result["objects_payload"]["objects"]), 1)
        obj = result["objects_payload"]["objects"][0]
        self.assertEqual(obj["id"], "obj_000001")
        self.assertEqual(obj["class_name"], "valve")
        self.assertEqual(obj["bbox"], {"x_min": 10, "y_min": 20, "x_max": 40, "y_max": 60})
        self.assertEqual(obj["source_model"], "ultralytics")
        self.assertEqual(obj["source_weight"], "yolo_weights/yolo26n_PPCL_640_20260227.pt")
        self.assertEqual(result["summary"]["object_count"], 1)
        self.assertEqual(result["summary"]["source_weight"], "yolo_weights/yolo26n_PPCL_640_20260227.pt")

    def test_draw_overlay_uses_blue_boxes(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        overlay = _draw_overlay(
            image,
            [{"bbox": {"x_min": 2, "y_min": 3, "x_max": 12, "y_max": 13}}],
        )

        self.assertTrue(np.array_equal(overlay[3, 2], np.array([255, 0, 0], dtype=np.uint8)))

    @patch("garnet.object_detection_sahi._build_detection_model")
    @patch("garnet.object_detection_sahi.get_sliced_prediction")
    @patch("garnet.object_detection_sahi.cv2.imread", return_value=np.zeros((64, 64, 3), dtype=np.uint8))
    def test_run_object_detection_sahi_uses_fixed_baseline(
        self,
        mock_imread: MagicMock,
        mock_get_sliced_prediction: MagicMock,
        mock_build_detection_model: MagicMock,
    ) -> None:
        del mock_imread
        mock_build_detection_model.return_value = MagicMock()
        mock_get_sliced_prediction.return_value = MagicMock(object_prediction_list=[])

        cfg = DetectionSahiConfig()
        result = run_object_detection_sahi("dummy.png", cfg=cfg)

        mock_build_detection_model.assert_called_once_with(cfg)
        kwargs = mock_get_sliced_prediction.call_args.kwargs
        self.assertEqual(kwargs["slice_height"], 640)
        self.assertEqual(kwargs["slice_width"], 640)
        self.assertEqual(kwargs["postprocess_type"], "GREEDYNMM")
        self.assertEqual(kwargs["postprocess_match_metric"], "IOS")
        self.assertEqual(kwargs["postprocess_match_threshold"], 0.1)
        self.assertEqual(result["summary"]["source_model"], "ultralytics")


if __name__ == "__main__":
    unittest.main()
