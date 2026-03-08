import tempfile
import unittest
from pathlib import Path

import numpy as np

from garnet.gemini_ocr_sahi import (
    GeminiOcrSahiConfig,
    _choose_best_candidate,
    _build_patch_grid,
    _draw_overlay,
    _extract_patch,
    _load_prompt_bundle,
    _map_bbox_to_sheet,
    run_gemini_ocr_sahi,
)


class GeminiOcrSahiTests(unittest.TestCase):
    def test_load_prompt_bundle_reads_expected_files(self) -> None:
        bundle = _load_prompt_bundle()

        self.assertIn("full-page discovery pass", bundle["full_page_system"])
        self.assertIn("Process this full engineering drawing image", bundle["full_page_user"])
        self.assertIn("crop refinement pass", bundle["crop_system"])
        self.assertIn("Process this cropped region", bundle["crop_user"])

    def test_extract_patch_outputs_1024_square_and_transform(self) -> None:
        image = np.zeros((300, 500), dtype=np.uint8)

        patch, transform = _extract_patch(image, (100, 50, 260, 140), patch_size=1024)

        self.assertEqual(patch.shape[:2], (1024, 1024))
        self.assertGreater(transform["scale"], 0)
        self.assertGreaterEqual(transform["source_box"]["x_min"], 0)
        self.assertGreaterEqual(transform["source_box"]["y_min"], 0)

    def test_map_bbox_to_sheet_reverses_patch_transform(self) -> None:
        image = np.zeros((300, 500), dtype=np.uint8)
        _, transform = _extract_patch(image, (100, 50, 260, 140), patch_size=1024)

        mapped = _map_bbox_to_sheet(
            {
                "x_min": transform["pad_x"],
                "y_min": transform["pad_y"],
                "x_max": transform["pad_x"] + transform["resized_width"],
                "y_max": transform["pad_y"] + transform["resized_height"],
            },
            transform,
            image.shape[1],
            image.shape[0],
        )

        self.assertEqual(mapped, transform["source_box"])

    def test_run_gemini_ocr_sahi_triggers_crop_fallback_for_low_confidence(self) -> None:
        image = np.zeros((256, 256), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            try:
                import cv2  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise unittest.SkipTest(f"cv2 unavailable: {exc}") from exc
            cv2.imwrite(str(image_path), image)

            calls: list[str] = []

            def fake_infer(patch_image, *, pass_kind, prompt_bundle, image_id):
                del patch_image, prompt_bundle, image_id
                calls.append(pass_kind)
                if pass_kind == "full_page":
                    return {
                        "image_id": "",
                        "pass_type": "full_page",
                        "text_regions": [
                            {
                                "id": "patch_1",
                                "text": "P-1001",
                                "normalized_text": "P-1001",
                                "class": "line_number",
                                "confidence": 0.2,
                                "bbox": {"x_min": 100, "y_min": 120, "x_max": 300, "y_max": 180},
                                "rotation": 0,
                                "reading_direction": "horizontal",
                                "legibility": "clear",
                            }
                        ],
                    }
                return {
                    "image_id": "",
                    "pass_type": "crop",
                    "text_regions": [
                        {
                            "id": "crop_1",
                            "text": "P-1001",
                            "normalized_text": "P-1001",
                            "class": "line_number",
                            "confidence": 0.82,
                            "bbox": {"x_min": 90, "y_min": 110, "x_max": 310, "y_max": 190},
                            "rotation": 0,
                            "reading_direction": "horizontal",
                            "legibility": "clear",
                        }
                    ],
                }

            result = run_gemini_ocr_sahi(
                image_path,
                cfg=GeminiOcrSahiConfig(patch_size=1024, low_confidence_threshold=0.3),
                infer_fn=fake_infer,
            )

        self.assertEqual(calls, ["full_page", "crop"])
        self.assertEqual(result["regions_payload"]["pass_type"], "sheet")
        self.assertEqual(len(result["regions_payload"]["text_regions"]), 1)
        self.assertEqual(result["summary"]["route"], "gemini")
        self.assertEqual(result["summary"]["crop_fallback_count"], 1)

    def test_build_patch_grid_covers_image(self) -> None:
        patches = _build_patch_grid(2200, 1800, patch_size=1024, overlap=128)

        self.assertGreaterEqual(len(patches), 4)
        self.assertEqual(patches[0], (0, 0, 1024, 1024))
        self.assertTrue(any(x2 == 2200 for _, _, x2, _ in patches))
        self.assertTrue(any(y2 == 1800 for _, _, _, y2 in patches))

    def test_choose_best_candidate_prefers_highest_confidence(self) -> None:
        merged_bbox = {"x_min": 10, "y_min": 10, "x_max": 110, "y_max": 40}
        candidates = [
            {
                "text": "LOW",
                "class": "unknown",
                "confidence": 0.35,
                "bbox": {"x_min": 12, "y_min": 12, "x_max": 108, "y_max": 38},
            },
            {
                "text": "HIGH",
                "class": "line_number",
                "confidence": 0.91,
                "bbox": {"x_min": 14, "y_min": 14, "x_max": 106, "y_max": 36},
            },
        ]

        chosen = _choose_best_candidate(merged_bbox, candidates, match_threshold=0.1)

        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen["text"], "HIGH")
        self.assertEqual(chosen["class"], "line_number")

    def test_run_gemini_ocr_sahi_calls_sahi_with_postprocess_threshold(self) -> None:
        image = np.zeros((256, 256), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            try:
                import cv2  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise unittest.SkipTest(f"cv2 unavailable: {exc}") from exc
            cv2.imwrite(str(image_path), image)

            class FakeResult:
                object_prediction_list = []

            from unittest.mock import patch

            with patch("garnet.gemini_ocr_sahi.get_sliced_prediction", return_value=FakeResult()) as mock_sahi:
                result = run_gemini_ocr_sahi(
                    image_path,
                    cfg=GeminiOcrSahiConfig(postprocess_match_threshold=0.1),
                    infer_fn=lambda *_args, **_kwargs: {"image_id": "", "pass_type": "full_page", "text_regions": []},
                )

        self.assertEqual(result["summary"]["postprocess_match_threshold"], 0.1)
        self.assertEqual(mock_sahi.call_args.kwargs["postprocess_match_threshold"], 0.1)

    def test_draw_overlay_uses_blue_bounding_boxes(self) -> None:
        image = np.zeros((20, 20), dtype=np.uint8)
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
