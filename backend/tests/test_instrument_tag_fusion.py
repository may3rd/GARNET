import unittest
from unittest.mock import patch

import numpy as np

from garnet.instrument_tag_fusion import run_instrument_tag_fusion_stage


class InstrumentTagFusionTests(unittest.TestCase):
    def test_run_instrument_tag_fusion_stage_matches_simple_tag(self) -> None:
        object_regions = [
            {
                "id": "obj_1",
                "class_name": "instrument tag",
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 40, "y_max": 20},
            }
        ]
        text_regions = [
            {
                "id": "ocr_1",
                "text": "PI 0201A",
                "bbox": {"x_min": 2, "y_min": 2, "x_max": 35, "y_max": 18},
            }
        ]

        result = run_instrument_tag_fusion_stage(
            image_id="sample.png",
            image_bgr=np.zeros((30, 50, 3), dtype=np.uint8),
            object_regions=object_regions,
            text_regions=text_regions,
            max_distance_px=30.0,
        )

        self.assertEqual(result["summary"]["matched_instrument_semantic_count"], 1)

    @patch("garnet.instrument_tag_fusion._confirm_with_crop_ocr", return_value="PI-0201A")
    def test_run_instrument_tag_fusion_stage_uses_crop_confirmation_for_two_line_tag(self, _mock_crop) -> None:
        object_regions = [
            {
                "id": "obj_1",
                "class_name": "instrument tag",
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 40, "y_max": 40},
                "confidence": 0.9,
            }
        ]
        text_regions = []

        result = run_instrument_tag_fusion_stage(
            image_id="sample.png",
            image_bgr=np.zeros((50, 50, 3), dtype=np.uint8),
            object_regions=object_regions,
            text_regions=text_regions,
            max_distance_px=30.0,
        )

        self.assertEqual(result["summary"]["matched_instrument_semantic_count"], 1)
        self.assertEqual(result["instrument_tags_payload"]["instrument_tags"][0]["normalized_text"], "PI-0201A")


if __name__ == "__main__":
    unittest.main()
