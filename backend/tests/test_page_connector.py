import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.page_connector import classify_off_page_reference, find_nearby_text


class PageConnectorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
