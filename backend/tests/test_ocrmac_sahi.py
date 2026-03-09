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
                cfg=OcrMacSahiConfig(slice_height=200, slice_width=200),
            )

        self.assertIn("regions_payload", result)
        self.assertIn("summary", result)
        self.assertIn("exception_candidates", result)
        self.assertIn("overlay_image", result)
        self.assertEqual(result["summary"]["route"], "ocrmac")
        self.assertEqual(len(result["regions_payload"]["text_regions"]), 1)


if __name__ == "__main__":
    unittest.main()
