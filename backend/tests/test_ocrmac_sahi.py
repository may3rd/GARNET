import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from garnet.ocrmac_sahi import OcrMacSahiConfig, run_ocrmac_sahi


class OcrMacSahiTests(unittest.TestCase):
    @patch("garnet.ocrmac_sahi.platform.system", return_value="Darwin")
    @patch("garnet.ocrmac_sahi._get_ocrmac_module")
    def test_run_ocrmac_sahi_returns_shared_stage2_contract(self, mock_module, _mock_platform) -> None:
        class FakeOCR:
            def __init__(self, *_args, **_kwargs):
                pass

            def recognize(self):
                return [
                    ("PI-0201A", 0.94, [0.1, 0.6, 0.3, 0.2]),
                ]

        mock_module.return_value.OCR = FakeOCR

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            image = np.full((100, 200, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            result = run_ocrmac_sahi(
                image_path,
                image_id="sample.png",
                cfg=OcrMacSahiConfig(slice_height=200, slice_width=200, enable_rotated_ocr=False),
            )

        self.assertIn("regions_payload", result)
        self.assertIn("summary", result)
        self.assertIn("exception_candidates", result)
        self.assertIn("overlay_image", result)
        self.assertEqual(result["summary"]["route"], "ocrmac")
        self.assertEqual(len(result["regions_payload"]["text_regions"]), 1)

    @patch("garnet.ocrmac_sahi.platform.system", return_value="Darwin")
    @patch("garnet.ocrmac_sahi._get_ocrmac_module")
    def test_run_ocrmac_sahi_restores_rotated_detections(self, mock_module, _mock_platform) -> None:
        class FakeOCR:
            calls = 0

            def __init__(self, image, *_args, **_kwargs):
                self.image = image

            def recognize(self):
                FakeOCR.calls += 1
                if FakeOCR.calls == 2:
                    return [
                        ("6\"-P-20012-A1", 0.96, [0.20, 0.15, 0.45, 0.20]),
                    ]
                return []

        mock_module.return_value.OCR = FakeOCR

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            image = np.full((100, 200, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            result = run_ocrmac_sahi(
                image_path,
                image_id="sample.png",
                cfg=OcrMacSahiConfig(slice_height=200, slice_width=200, enable_rotated_ocr=True),
            )

        self.assertEqual(len(result["regions_payload"]["text_regions"]), 1)
        region = result["regions_payload"]["text_regions"][0]
        self.assertEqual(region["text"], '6"-P-20012-A1')
        self.assertEqual(region["rotation"], 90)
        self.assertEqual(region["reading_direction"], "rotated")
        self.assertLess(region["bbox"]["x_min"], region["bbox"]["x_max"])
        self.assertLess(region["bbox"]["y_min"], region["bbox"]["y_max"])
        self.assertTrue(result["summary"]["rotated_ocr_enabled"])


if __name__ == "__main__":
    unittest.main()
