import unittest
from unittest.mock import patch

import numpy as np

from garnet.line_number_fusion import run_line_number_fusion_stage
from garnet.pipe_text_attachment import run_pipe_text_attachment_stage


class PipeTextAttachmentTests(unittest.TestCase):
    def test_run_line_number_fusion_stage_fuses_nearby_fragments(self) -> None:
        object_regions = [
            {
                "id": "obj_1",
                "class_name": "line number",
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 120, "y_max": 20},
            }
        ]
        text_regions = [
            {
                "id": "ocr_1",
                "text": '3"-PL-25',
                "bbox": {"x_min": 2, "y_min": 2, "x_max": 45, "y_max": 18},
                "confidence": 0.9,
            },
            {
                "id": "ocr_2",
                "text": "002013-B1A2-NI",
                "bbox": {"x_min": 48, "y_min": 2, "x_max": 118, "y_max": 18},
                "confidence": 0.92,
            },
        ]

        result = run_line_number_fusion_stage(
            image_id="sample.png",
            image_bgr=np.zeros((30, 140, 3), dtype=np.uint8),
            object_regions=object_regions,
            text_regions=text_regions,
            max_distance_px=40.0,
        )

        self.assertEqual(result["summary"]["matched_line_number_count"], 1)
        self.assertIn('3"-PL-25', result["line_numbers_payload"]["line_numbers"][0]["text"])

    @patch("garnet.line_number_fusion._confirm_with_crop_ocr", return_value=("crop_ocr", '6"-NAS-25-003003-B2A2-NI'))
    def test_run_line_number_fusion_stage_uses_crop_confirmation_when_sheet_ocr_misses(self, _mock_crop) -> None:
        object_regions = [
            {
                "id": "obj_1",
                "class_name": "line number",
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 160, "y_max": 24},
                "confidence": 0.72,
            }
        ]

        result = run_line_number_fusion_stage(
            image_id="sample.png",
            image_bgr=np.zeros((40, 180, 3), dtype=np.uint8),
            object_regions=object_regions,
            text_regions=[],
            max_distance_px=40.0,
        )

        self.assertEqual(result["summary"]["matched_line_number_count"], 1)
        self.assertEqual(result["summary"]["ocr_confirmed_line_number_count"], 1)
        self.assertEqual(result["summary"]["od_only_line_number_count"], 0)
        self.assertEqual(
            result["line_numbers_payload"]["line_numbers"][0]["normalized_text"],
            '6"-NAS-25-003003-B2A2-NI',
        )
        self.assertEqual(result["line_numbers_payload"]["line_numbers"][0]["ocr_region_id"], "crop_ocr")
        self.assertEqual(result["line_numbers_payload"]["line_numbers"][0]["ocr_source"], "crop_ocr")
        self.assertEqual(result["line_numbers_payload"]["line_numbers"][0]["review_state"], "ocr_confirmed")

    def test_run_pipe_text_attachment_stage_attaches_line_number_to_edge(self) -> None:
        text_regions = [
            {
                "id": "ocr_1",
                "text": "3\"-PL-25-002013",
                "normalized_text": "3\"-PL-25-002013",
                "class": "line_number",
                "bbox": {"x_min": 5, "y_min": 5, "x_max": 25, "y_max": 15},
            }
        ]
        edges = [
            {
                "id": "edge_1",
                "source": "n1",
                "target": "n2",
                "polyline": [
                    {"row": 10, "col": 0},
                    {"row": 10, "col": 30},
                ],
            }
        ]

        result = run_pipe_text_attachment_stage(
            image_id="sample.png",
            image_bgr=np.zeros((30, 40, 3), dtype=np.uint8),
            text_regions=text_regions,
            edges=edges,
            max_distance_px=20.0,
        )

        self.assertEqual(result["summary"]["candidate_count"], 1)
        self.assertEqual(result["summary"]["accepted_attachment_count"], 1)
        self.assertEqual(len(result["attachments_payload"]["accepted"]), 1)

    def test_run_pipe_text_attachment_stage_uses_bbox_distance_not_center_only(self) -> None:
        text_regions = [
            {
                "id": "ocr_1",
                "text": '6"-NAS-25-003003-B2A2-NI',
                "normalized_text": '6"-NAS-25-003003-B2A2-NI',
                "class": "line_number",
                "bbox": {"x_min": 10, "y_min": 0, "x_max": 150, "y_max": 20},
            }
        ]
        edges = [
            {
                "id": "edge_1",
                "source": "n1",
                "target": "n2",
                "polyline": [
                    {"row": 24, "col": 10},
                    {"row": 24, "col": 150},
                ],
            }
        ]

        result = run_pipe_text_attachment_stage(
            image_id="sample.png",
            image_bgr=np.zeros((40, 160, 3), dtype=np.uint8),
            text_regions=text_regions,
            edges=edges,
            max_distance_px=10.0,
        )

        self.assertEqual(result["summary"]["accepted_attachment_count"], 1)


if __name__ == "__main__":
    unittest.main()
