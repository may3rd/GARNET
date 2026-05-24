"""Unit tests for the visual-primitives response parser."""

import unittest

from garnet.visual_primitives.response_parser import (
    parse_response,
    _extract_boxes,
    _extract_thinking,
    _map_class,
    _iou,
)
from garnet.visual_primitives.schemas import EquipmentClass


class TestThinkingExtraction(unittest.TestCase):

    def test_extract_thinking_before_json(self):
        text = """Scanning the drawing. I see a column: <|ref|>distillation_column<|/ref|><|box|>[[100,50,200,600]]<|/box|>

```json
{"equipment": [{"tag": "C-201", "equipment_class": "distillation_column", "bbox": [100,50,200,600], "confidence": "high"}]}
```"""
        thinking = _extract_thinking(text)
        self.assertIn("Scanning", thinking)
        self.assertNotIn("```json", thinking)
        self.assertNotIn('"equipment"', thinking)

    def test_no_json_block(self):
        text = "Just some raw thinking with no JSON."
        thinking = _extract_thinking(text)
        self.assertEqual(thinking, "Just some raw thinking with no JSON.")


class TestBoxExtraction(unittest.TestCase):
    """Tests for <box> primitive extraction."""

    def test_single_box(self):
        text = 'Found column: <|ref|>distillation_column<|/ref|><|box|>[[100,50,200,600]]<|/box|> tag=C-201 confidence=high'
        entries = _extract_boxes(text, 1000, 1000)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e.tag, "C-201")
        self.assertEqual(e.equipment_class, EquipmentClass.DISTILLATION_COLUMN)
        self.assertEqual(e.confidence.value, "high")
        # View is 1000×1000, so coords stay the same after [0,999] normalisation.
        # 200/1000*999=199.8->200, 600/1000*999=599.4->599 (rounding)
        self.assertEqual(e.global_bbox, [100, 50, 200, 599])

    def test_coordinate_scaling(self):
        """Coordinates should be scaled from view-space to [0,999]."""
        text = '<|ref|>pump<|/ref|><|box|>[[500,250,600,350]]<|/box|> tag=P-101 confidence=medium'
        # View is 800×400 — coordinates should scale.
        entries = _extract_boxes(text, 800, 400)
        e = entries[0]
        # 500/800*999 ≈ 624, 250/400*999 ≈ 624
        self.assertEqual(e.global_bbox, [624, 624, 749, 874])

    def test_multiple_boxes(self):
        text = """
        Column: <|ref|>distillation_column<|/ref|><|box|>[[10,20,30,40]]<|/box|> tag=C-201 confidence=high
        Pump: <|ref|>pump<|/ref|><|box|>[[50,60,70,80]]<|/box|> tag=P-101 confidence=medium
        """
        entries = _extract_boxes(text, 100, 100)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].tag, "C-201")
        self.assertEqual(entries[1].tag, "P-101")

    def test_duplicate_bbox_filtered(self):
        text = """
        <|ref|>vessel<|/ref|><|box|>[[10,20,30,40]]<|/box|>
        <|ref|>vessel<|/ref|><|box|>[[10,20,30,40]]<|/box|>
        """
        entries = _extract_boxes(text, 100, 100)
        self.assertEqual(len(entries), 1)

    def test_missing_tag_defaults_to_unknown(self):
        text = '<|ref|>compressor<|/ref|><|box|>[[10,20,30,40]]<|/box|>'
        entries = _extract_boxes(text, 100, 100)
        self.assertEqual(entries[0].tag, "unknown")

    def test_missing_confidence_defaults_to_medium(self):
        text = '<|ref|>reactor<|/ref|><|box|>[[10,20,30,40]]<|/box|> tag=R-501'
        entries = _extract_boxes(text, 100, 100)
        self.assertEqual(entries[0].confidence.value, "medium")

    def test_no_boxes_returns_empty(self):
        text = "No equipment found in this drawing."
        entries = _extract_boxes(text, 100, 100)
        self.assertEqual(len(entries), 0)


class TestClassMapping(unittest.TestCase):

    def test_known_classes(self):
        cases = [
            ("distillation_column", EquipmentClass.DISTILLATION_COLUMN),
            ("column", EquipmentClass.DISTILLATION_COLUMN),
            ("tower", EquipmentClass.DISTILLATION_COLUMN),
            ("pressure_vessel", EquipmentClass.PRESSURE_VESSEL),
            ("vessel", EquipmentClass.PRESSURE_VESSEL),
            ("drum", EquipmentClass.PRESSURE_VESSEL),
            ("separator", EquipmentClass.PRESSURE_VESSEL),
            ("heat_exchanger", EquipmentClass.HEAT_EXCHANGER),
            ("exchanger", EquipmentClass.HEAT_EXCHANGER),
            ("shell_and_tube", EquipmentClass.HEAT_EXCHANGER),
            ("reboiler", EquipmentClass.HEAT_EXCHANGER),
            ("condenser", EquipmentClass.HEAT_EXCHANGER),
            ("storage_tank", EquipmentClass.STORAGE_TANK),
            ("tank", EquipmentClass.STORAGE_TANK),
            ("pump", EquipmentClass.PUMP),
            ("compressor", EquipmentClass.COMPRESSOR),
            ("reactor", EquipmentClass.REACTOR),
        ]
        for name, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(_map_class(name), expected)

    def test_unknown_class(self):
        self.assertEqual(_map_class("giraffe"), EquipmentClass.OTHER)
        self.assertEqual(_map_class(""), EquipmentClass.OTHER)


class TestIoU(unittest.TestCase):

    def test_perfect_overlap(self):
        self.assertAlmostEqual(_iou(0, 0, 100, 100, 0, 0, 100, 100), 1.0)

    def test_no_overlap(self):
        self.assertEqual(_iou(0, 0, 10, 10, 20, 20, 30, 30), 0.0)

    def test_partial_overlap(self):
        # 100×100 at origin, 100×100 at (50,50) -> 50% overlap
        iou = _iou(0, 0, 100, 100, 50, 50, 150, 150)
        self.assertAlmostEqual(iou, 0.1428, delta=0.01)


class TestFullParseResponse(unittest.TestCase):
    """Integration test: full parse_response pipeline."""

    def test_full_parse(self):
        raw = """Reasoning: I see a large column here <|ref|>distillation_column<|/ref|><|box|>[[100,50,200,600]]<|/box|> tag=C-201 confidence=high. Also a pump <|ref|>pump<|/ref|><|box|>[[700,800,750,820]]<|/box|> tag=P-101 confidence=medium.

```json
{
  "equipment": [
    {"tag": "C-201", "equipment_class": "distillation_column", "bbox": [100, 50, 200, 600], "confidence": "high"},
    {"tag": "P-101", "equipment_class": "pump", "bbox": [700, 800, 750, 820], "confidence": "medium"}
  ],
  "drawing_notes": "Good quality P&ID."
}
```"""
        registry, thinking = parse_response(raw, 1000, 1000)
        self.assertEqual(registry.total_count, 2)
        self.assertIn("Reasoning", thinking)
        self.assertEqual(registry.drawing_notes, "Good quality P&ID.")

        # Check tags
        tags = [e.tag for e in registry.equipment]
        self.assertIn("C-201", tags)
        self.assertIn("P-101", tags)

    def test_parse_boxes_only_no_json(self):
        """Should work with box primitives only, no structured JSON."""
        raw = '<|ref|>pump<|/ref|><|box|>[[50,60,70,80]]<|/box|> tag=P-101 confidence=high'
        registry, thinking = parse_response(raw, 100, 100)
        self.assertEqual(registry.total_count, 1)
        self.assertEqual(registry.equipment[0].tag, "P-101")

    def test_parse_empty_response(self):
        registry, thinking = parse_response("", 100, 100)
        self.assertEqual(registry.total_count, 0)
        self.assertEqual(thinking, "")


if __name__ == "__main__":
    unittest.main()
