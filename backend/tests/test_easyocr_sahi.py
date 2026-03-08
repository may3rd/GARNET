import unittest

from garnet.easyocr_sahi import _merge_same_line_regions, _rotate_quad_back


class EasyOcrSahiHelperTests(unittest.TestCase):
    def test_rotate_quad_back_restores_clockwise_rotation_to_original_coordinates(self) -> None:
        original_width = 100
        original_height = 40
        rotated_quad = [
            [5.0, 10.0],
            [5.0, 30.0],
            [15.0, 30.0],
            [15.0, 10.0],
        ]

        restored = _rotate_quad_back(rotated_quad, "cw", original_width, original_height)

        self.assertEqual(restored, [[10.0, 34.0], [30.0, 34.0], [30.0, 24.0], [10.0, 24.0]])

    def test_merge_same_line_regions_combines_adjacent_text_boxes(self) -> None:
        regions = [
            {
                "id": "",
                "text": "CAUSTIC",
                "normalized_text": "CAUSTIC",
                "class": "unknown",
                "confidence": 0.91,
                "bbox": {"x_min": 10, "y_min": 10, "x_max": 90, "y_max": 30},
                "rotation": 0,
                "reading_direction": "horizontal",
                "legibility": "clear",
            },
            {
                "id": "",
                "text": "WASH",
                "normalized_text": "WASH",
                "class": "unknown",
                "confidence": 0.92,
                "bbox": {"x_min": 98, "y_min": 11, "x_max": 150, "y_max": 31},
                "rotation": 0,
                "reading_direction": "horizontal",
                "legibility": "clear",
            },
            {
                "id": "",
                "text": "FEED",
                "normalized_text": "FEED",
                "class": "unknown",
                "confidence": 0.94,
                "bbox": {"x_min": 159, "y_min": 10, "x_max": 215, "y_max": 30},
                "rotation": 0,
                "reading_direction": "horizontal",
                "legibility": "clear",
            },
        ]

        merged = _merge_same_line_regions(regions, line_merge_gap_px=16, line_merge_y_tolerance_px=8)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "CAUSTIC WASH FEED")
        self.assertEqual(merged[0]["bbox"], {"x_min": 10, "y_min": 10, "x_max": 215, "y_max": 31})


if __name__ == "__main__":
    unittest.main()
