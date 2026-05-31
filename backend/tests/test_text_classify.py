import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.text_classify import classify_text_region


class TextClassifyTests(unittest.TestCase):
    def test_equipment_tag_pattern_V_style(self) -> None:
        self.assertEqual(classify_text_region({"text": "V-101"}), "equipment_tag")

    def test_equipment_tag_pattern_P_style(self) -> None:
        self.assertEqual(classify_text_region({"text": "P-100A"}), "equipment_tag")

    def test_nozzle_tag_pattern(self) -> None:
        self.assertEqual(classify_text_region({"text": "N-1"}), "nozzle_tag")

    def test_pipe_spec_pattern(self) -> None:
        self.assertEqual(classify_text_region({"text": '6"-CS-150'}), "pipe_spec")

    def test_note_class_passthrough(self) -> None:
        self.assertEqual(classify_text_region({"class": "note", "text": "V-101"}), "note")

    def test_unknown_non_matching(self) -> None:
        self.assertEqual(classify_text_region({"text": "SEE DETAIL A"}), "unknown")


if __name__ == "__main__":
    unittest.main()
