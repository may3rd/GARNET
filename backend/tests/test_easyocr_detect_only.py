import unittest

import numpy as np

try:
    from garnet.easyocr_sahi import EasyOcrSahiConfig, _bbox_from_quad, _read_tile_with_orientations

    EASYOCR_IMPORTABLE = True
except Exception:  # pragma: no cover - exercised only where easyocr/cv2 are absent
    EASYOCR_IMPORTABLE = False


class _FakeReader:
    """Stub reader: records calls, never runs a real model."""

    def __init__(self) -> None:
        self.detect_calls = 0
        self.readtext_calls = 0

    def detect(self, image, **kwargs):
        self.detect_calls += 1
        horizontal_list = [[[10, 50, 20, 30]]]
        free_list = [[[[5, 5], [15, 5], [15, 15], [5, 15]]]]
        return horizontal_list, free_list

    def readtext(self, image, **kwargs):
        self.readtext_calls += 1
        raise AssertionError("readtext must not be called in detect-only mode")


@unittest.skipUnless(EASYOCR_IMPORTABLE, "easyocr/cv2 not importable in this environment")
class EasyOcrDetectOnlyTests(unittest.TestCase):
    def test_detect_only_uses_detect_not_readtext(self) -> None:
        reader = _FakeReader()
        cfg = EasyOcrSahiConfig(detect_only=True, enable_rotated_ocr=False)
        tile = np.zeros((40, 60, 3), dtype=np.uint8)

        oriented_results = _read_tile_with_orientations(reader, tile, cfg)

        self.assertEqual(reader.readtext_calls, 0)
        self.assertEqual(reader.detect_calls, 1)
        self.assertEqual(len(oriented_results), 1)
        orientation, results = oriented_results[0]
        self.assertEqual(orientation, "none")
        for quad, text, score in results:
            self.assertEqual(text, "")
            self.assertEqual(score, 1.0)

    def test_horizontal_entry_ordering_maps_to_correct_bbox(self) -> None:
        reader = _FakeReader()
        cfg = EasyOcrSahiConfig(detect_only=True, enable_rotated_ocr=False)
        tile = np.zeros((40, 60, 3), dtype=np.uint8)

        _, results = _read_tile_with_orientations(reader, tile, cfg)[0]
        horizontal_quad = results[0][0]
        bbox = _bbox_from_quad(horizontal_quad)
        self.assertEqual(bbox, {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 30})

    def test_free_list_quad_passes_through_unchanged(self) -> None:
        reader = _FakeReader()
        cfg = EasyOcrSahiConfig(detect_only=True, enable_rotated_ocr=False)
        tile = np.zeros((40, 60, 3), dtype=np.uint8)

        _, results = _read_tile_with_orientations(reader, tile, cfg)[0]
        free_quad = results[1][0]
        self.assertEqual(free_quad, [[5, 5], [15, 5], [15, 15], [5, 15]])

    def test_all_three_orientations_attempted_when_rotated_ocr_enabled(self) -> None:
        reader = _FakeReader()
        cfg = EasyOcrSahiConfig(detect_only=True, enable_rotated_ocr=True)
        tile = np.zeros((40, 60, 3), dtype=np.uint8)

        oriented_results = _read_tile_with_orientations(reader, tile, cfg)

        self.assertEqual([o for o, _ in oriented_results], ["none", "cw", "ccw"])
        self.assertEqual(reader.detect_calls, 3)
        self.assertEqual(reader.readtext_calls, 0)


if __name__ == "__main__":
    unittest.main()
