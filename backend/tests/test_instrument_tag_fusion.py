import unittest

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

        self.assertEqual(result["summary"]["matched_instrument_tag_count"], 1)


if __name__ == "__main__":
    unittest.main()
