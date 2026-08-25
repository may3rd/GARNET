import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.page_connector import (
    classify_off_page_reference,
    find_nearby_text,
    line_number_by_trace_id,
    select_connector_metadata,
)


class PageConnectorTests(unittest.TestCase):
    def test_select_connector_metadata_uses_reference_and_line_labels_independently(self) -> None:
        metadata = select_connector_metadata(
            [
                {
                    "text": "10-P-100-A",
                    "normalized_text": "10-P-100-A",
                    "semantic_class": "line_number",
                    "distance_px": 5.0,
                    "page_reference": None,
                },
                {
                    "text": "SEE DWG 200-02",
                    "normalized_text": "SEE DWG 200-02",
                    "semantic_class": "reference",
                    "distance_px": 12.0,
                    "page_reference": {
                        "reference_type": "drawing",
                        "reference_value": "200-02",
                        "matched_text": "DWG 200-02",
                    },
                },
            ]
        )

        self.assertEqual(metadata["connector_key"], "10-P-100-A")
        self.assertEqual(metadata["target_sheet_reference"], "200-02")
        self.assertEqual(metadata["raw_reference_text"], "SEE DWG 200-02")
        self.assertEqual(metadata["page_reference"]["reference_type"], "drawing")

    def test_classify_sheet_pattern(self) -> None:
        self.assertEqual(
            classify_off_page_reference("SHEET P-101"),
            {"reference_type": "sheet", "reference_value": "P-101", "matched_text": "SHEET P-101"},
        )

    def test_classify_pid_pattern(self) -> None:
        self.assertEqual(
            classify_off_page_reference("PID-3"),
            {"reference_type": "pid", "reference_value": "3", "matched_text": "PID-3"},
        )

    def test_classify_fig_pattern(self) -> None:
        self.assertEqual(
            classify_off_page_reference("fig 12"),
            {"reference_type": "figure", "reference_value": "12", "matched_text": "fig 12"},
        )

    def test_classify_page_pattern(self) -> None:
        self.assertEqual(
            classify_off_page_reference("page 5"),
            {"reference_type": "sheet", "reference_value": "5", "matched_text": "page 5"},
        )

    def test_classify_dwg_pattern(self) -> None:
        self.assertEqual(
            classify_off_page_reference("DWG 8-137"),
            {"reference_type": "drawing", "reference_value": "8-137", "matched_text": "DWG 8-137"},
        )

    def test_classify_no_match(self) -> None:
        self.assertIsNone(classify_off_page_reference("random text"))

    def test_find_nearby_text_attaches_closest(self) -> None:
        labels = find_nearby_text(
            {"x_min": 100, "y_min": 100, "x_max": 120, "y_max": 120},
            [
                {
                    "id": "far",
                    "text": "SHEET P-101",
                    "normalized_text": "SHEET P-101",
                    "bbox": {"x_min": 170, "y_min": 100, "x_max": 190, "y_max": 120},
                },
                {
                    "id": "near",
                    "text": "TO CONTROL VALVE",
                    "bbox": {"x_min": 115, "y_min": 100, "x_max": 135, "y_max": 120},
                },
                {
                    "id": "outside",
                    "text": "PAGE 5",
                    "bbox": {"x_min": 300, "y_min": 100, "x_max": 320, "y_max": 120},
                },
            ],
        )

        self.assertEqual([label["region_id"] for label in labels], ["near", "far"])
        self.assertEqual(labels[0]["semantic_class"], "label")
        self.assertEqual(labels[1]["semantic_class"], "reference")

    def test_find_nearby_text_accepts_list_bbox(self) -> None:
        labels = find_nearby_text(
            [100, 100, 120, 120],
            [
                {
                    "id": "near",
                    "text": "PAGE 5",
                    "bbox": [115, 100, 135, 120],
                }
            ],
        )

        self.assertEqual(labels[0]["page_reference"]["reference_value"], "5")

    def test_select_connector_metadata_uses_fallback_line_number_when_no_line_label(self) -> None:
        """When no line-number OCR label is near the connector, the fallback
        (the line number attached to the traced pipe) becomes the connector key."""
        metadata = select_connector_metadata(
            [
                {
                    "text": "SEE DWG 200-02",
                    "normalized_text": "SEE DWG 200-02",
                    "semantic_class": "reference",
                    "distance_px": 12.0,
                    "page_reference": {
                        "reference_type": "drawing",
                        "reference_value": "200-02",
                        "matched_text": "DWG 200-02",
                    },
                }
            ],
            fallback_line_number="2NAS-25-003004-B2A2-NI",
        )

        self.assertEqual(metadata["connector_key"], "2NAS-25-003004-B2A2-NI")
        self.assertEqual(metadata["target_sheet_reference"], "200-02")

    def test_select_connector_metadata_prefers_nearby_line_label_over_fallback(self) -> None:
        """A real line-number label near the connector wins over the fallback."""
        metadata = select_connector_metadata(
            [
                {
                    "text": "10-P-100-A",
                    "normalized_text": "10-P-100-A",
                    "semantic_class": "line_number",
                    "distance_px": 5.0,
                    "page_reference": None,
                }
            ],
            fallback_line_number="2NAS-25-003004-B2A2-NI",
        )

        self.assertEqual(metadata["connector_key"], "10-P-100-A")

    def test_line_number_by_trace_id_picks_highest_confidence(self) -> None:
        payload = {
            "associations": {
                "line_numbers": {
                    "accepted": [
                        {"trace_id": "obj_000195", "text": "2NAS-25-003004-B2A2-NI", "confidence": 0.9},
                        {"trace_id": "obj_000195", "text": "2NAS-25-003004-B2A2-NI", "confidence": 0.95},
                        {"trace_id": "obj_000191", "text": "4\"-CUL-25-002017-B1A2-NI", "confidence": 0.8},
                        {"trace_id": "", "text": "ignored-no-trace", "confidence": 0.9},
                        {"trace_id": "obj_000200", "text": "", "confidence": 0.9},
                    ]
                }
            }
        }

        result = line_number_by_trace_id(payload)

        self.assertEqual(result["obj_000195"], "2NAS-25-003004-B2A2-NI")
        self.assertEqual(result["obj_000191"], "4\"-CUL-25-002017-B1A2-NI")
        self.assertNotIn("", result)
        self.assertNotIn("obj_000200", result)

    def test_line_number_by_trace_id_handles_empty_payload(self) -> None:
        self.assertEqual(line_number_by_trace_id(None), {})
        self.assertEqual(line_number_by_trace_id({}), {})


if __name__ == "__main__":
    unittest.main()
