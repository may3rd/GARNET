import unittest
from unittest.mock import patch

import numpy as np

from garnet.line_number_fusion import run_line_number_fusion_stage
from garnet.pipe_text_attachment import _filter_border_like_edges, render_text_attachment_overlay, run_pipe_text_attachment_stage


class PipeTextAttachmentTests(unittest.TestCase):
    def test_filter_border_like_edges_excludes_page_and_title_block_borders(self) -> None:
        edges = [
            {
                "id": "edge_border_top",
                "source": "n1",
                "target": "n2",
                "pixel_length": 300,
                "polyline": [{"row": 5, "col": 10}, {"row": 5, "col": 390}],
            },
            {
                "id": "edge_title_right",
                "source": "n3",
                "target": "n4",
                "pixel_length": 240,
                "polyline": [{"row": 30, "col": 360}, {"row": 250, "col": 360}],
            },
            {
                "id": "edge_process",
                "source": "n5",
                "target": "n6",
                "pixel_length": 120,
                "polyline": [{"row": 150, "col": 100}, {"row": 150, "col": 240}],
            },
        ]

        result = _filter_border_like_edges(edges, (300, 400, 3))

        kept_ids = {edge["id"] for edge in result["kept_edges"]}
        filtered_ids = {item["id"] for item in result["filtered_edges_payload"]["filtered_edges"]}
        self.assertIn("edge_process", kept_ids)
        self.assertIn("edge_border_top", filtered_ids)
        self.assertIn("edge_title_right", filtered_ids)
        self.assertEqual(result["summary"]["filtered_edge_count"], 2)

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

    @patch("garnet.line_number_fusion._confirm_with_crop_ocr", return_value=("crop_ocr", '6"-NAS-25-003003-B2A2-NI'))
    def test_run_line_number_fusion_stage_prefers_fuller_crop_text_over_partial_sheet_text(self, _mock_crop) -> None:
        object_regions = [
            {
                "id": "obj_1",
                "class_name": "line number",
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 220, "y_max": 24},
                "confidence": 0.72,
            }
        ]
        text_regions = [
            {
                "id": "ocr_1",
                "text": '6"-NAS-25-003003-E',
                "bbox": {"x_min": 2, "y_min": 2, "x_max": 180, "y_max": 20},
                "confidence": 0.9,
                "class": "line_number",
            }
        ]

        result = run_line_number_fusion_stage(
            image_id="sample.png",
            image_bgr=np.zeros((40, 240, 3), dtype=np.uint8),
            object_regions=object_regions,
            text_regions=text_regions,
            max_distance_px=40.0,
        )

        self.assertEqual(
            result["line_numbers_payload"]["line_numbers"][0]["normalized_text"],
            '6"-NAS-25-003003-B2A2-NI',
        )
        self.assertEqual(result["line_numbers_payload"]["line_numbers"][0]["ocr_source"], "crop_ocr")

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

    def test_run_pipe_text_attachment_stage_marks_attachment_on_provisional_edge(self) -> None:
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
                "edge_terminals": {
                    "provisional_due_to_unresolved_terminal": True,
                },
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

        self.assertTrue(result["attachments_payload"]["accepted"][0]["attached_to_provisional_edge"])
        self.assertEqual(result["summary"]["accepted_attachment_on_provisional_edge_count"], 1)

    def test_render_text_attachment_overlay_uses_orange_for_provisional_edges(self) -> None:
        overlay = render_text_attachment_overlay(
            image_bgr=np.zeros((20, 20, 3), dtype=np.uint8),
            edges=[
                {
                    "id": "edge_1",
                    "source": "n1",
                    "target": "n2",
                    "edge_terminals": {"provisional_due_to_unresolved_terminal": True},
                    "polyline": [{"row": 10, "col": 2}, {"row": 10, "col": 17}],
                }
            ],
            attachments=[],
        )

        self.assertTrue(np.array_equal(overlay[10, 5], np.array([0, 165, 255], dtype=np.uint8)))

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

    def test_run_pipe_text_attachment_stage_uses_adaptive_threshold_for_long_line_number(self) -> None:
        text_regions = [
            {
                "id": "ocr_1",
                "text": '-6"-NAS-25-003003-B2A2-NI',
                "normalized_text": '-6"-NAS-25-003003-B2A2-NI',
                "class": "line_number",
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 160, "y_max": 28},
            }
        ]
        edges = [
            {
                "id": "edge_1",
                "source": "n1",
                "target": "n2",
                "polyline": [
                    {"row": 160, "col": 200},
                    {"row": 180, "col": 220},
                ],
            }
        ]

        result = run_pipe_text_attachment_stage(
            image_id="sample.png",
            image_bgr=np.zeros((240, 260, 3), dtype=np.uint8),
            text_regions=text_regions,
            edges=edges,
            max_distance_px=80.0,
        )

        self.assertEqual(result["summary"]["accepted_attachment_count"], 1)
        self.assertGreater(result["attachments_payload"]["accepted"][0]["threshold_px"], 140.0)

    def test_run_pipe_text_attachment_stage_uses_small_tolerance_for_instrument_semantic(self) -> None:
        text_regions = [
            {
                "id": "inst_1",
                "text": "XC-2504",
                "normalized_text": "XC-2504",
                "semantic_class": "instrument_semantic",
                "bbox": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100},
            }
        ]
        edges = [
            {
                "id": "edge_1",
                "source": "n1",
                "target": "n2",
                "polyline": [
                    {"row": 181, "col": 100},
                    {"row": 181, "col": 120},
                ],
            }
        ]

        result = run_pipe_text_attachment_stage(
            image_id="sample.png",
            image_bgr=np.zeros((240, 240, 3), dtype=np.uint8),
            text_regions=text_regions,
            edges=edges,
            max_distance_px=80.0,
            text_class="instrument_semantic",
        )

        self.assertEqual(result["summary"]["accepted_attachment_count"], 1)
        self.assertEqual(result["attachments_payload"]["accepted"][0]["threshold_px"], 85.0)

    def test_run_line_number_fusion_stage_rejects_tiny_detection_only_fragment(self) -> None:
        object_regions = [
            {
                "id": "obj_1",
                "class_name": "line number",
                "bbox": {"x_min": 5, "y_min": 5, "x_max": 35, "y_max": 35},
                "confidence": 0.9,
            }
        ]

        result = run_line_number_fusion_stage(
            image_id="sample.png",
            image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            object_regions=object_regions,
            text_regions=[],
            max_distance_px=40.0,
        )

        self.assertEqual(result["summary"]["matched_line_number_count"], 0)
        self.assertEqual(result["summary"]["rejected_line_number_count"], 1)

    def test_run_line_number_fusion_stage_rejects_extreme_border_vertical_detection_only(self) -> None:
        object_regions = [
            {
                "id": "obj_1",
                "class_name": "line number",
                "bbox": {"x_min": 0, "y_min": 10, "x_max": 70, "y_max": 1200},
                "confidence": 0.9,
            }
        ]

        result = run_line_number_fusion_stage(
            image_id="sample.png",
            image_bgr=np.zeros((1400, 1000, 3), dtype=np.uint8),
            object_regions=object_regions,
            text_regions=[],
            max_distance_px=40.0,
        )

        self.assertEqual(result["summary"]["matched_line_number_count"], 0)
        self.assertEqual(result["summary"]["rejected_line_number_count"], 1)


if __name__ == "__main__":
    unittest.main()
