"""Unit tests for visual-primitives schemas."""

import unittest

from garnet.visual_primitives.schemas import (
    EquipmentEntry,
    EquipmentRegistry,
    EquipmentClass,
    Confidence,
)


class TestEquipmentEntry(unittest.TestCase):

    def test_valid_entry(self):
        entry = EquipmentEntry(
            tag="C-201",
            equipment_class="distillation_column",
            global_bbox=[100, 50, 200, 600],
            confidence="high",
        )
        self.assertEqual(entry.tag, "C-201")
        self.assertEqual(entry.equipment_class, EquipmentClass.DISTILLATION_COLUMN)
        self.assertEqual(entry.global_bbox, [100, 50, 200, 600])
        self.assertEqual(entry.confidence, Confidence.HIGH)

    def test_default_confidence(self):
        entry = EquipmentEntry(
            tag="V-101",
            equipment_class="pressure_vessel",
            global_bbox=[10, 10, 50, 50],
        )
        self.assertEqual(entry.confidence, Confidence.MEDIUM)

    def test_invalid_bbox_out_of_range(self):
        with self.assertRaises(ValueError):
            EquipmentEntry(
                tag="E-301",
                equipment_class="heat_exchanger",
                global_bbox=[10, 10, 1000, 500],  # 1000 > 999
            )

    def test_invalid_bbox_order(self):
        with self.assertRaises(ValueError):
            EquipmentEntry(
                tag="P-401",
                equipment_class="pump",
                global_bbox=[200, 200, 100, 300],  # x2 < x1
            )

    def test_other_class_with_description(self):
        entry = EquipmentEntry(
            tag="unknown",
            equipment_class="other",
            global_bbox=[10, 10, 30, 30],
            description="Possibly a strainer or inline filter",
        )
        self.assertEqual(entry.equipment_class, EquipmentClass.OTHER)
        self.assertEqual(entry.description, "Possibly a strainer or inline filter")


class TestEquipmentRegistry(unittest.TestCase):

    def test_empty_registry(self):
        reg = EquipmentRegistry()
        self.assertEqual(reg.total_count, 0)
        self.assertEqual(reg.equipment, [])

    def test_auto_count(self):
        entries = [
            EquipmentEntry(tag="C-201", equipment_class="distillation_column", global_bbox=[1, 2, 3, 4]),
            EquipmentEntry(tag="E-202", equipment_class="heat_exchanger", global_bbox=[5, 6, 7, 8]),
        ]
        reg = EquipmentRegistry(equipment=entries)
        self.assertEqual(reg.total_count, 2)

    def test_explicit_count_override(self):
        entries = [EquipmentEntry(tag="C-201", equipment_class="distillation_column", global_bbox=[1, 2, 3, 4])]
        reg = EquipmentRegistry(equipment=entries, total_count=5)
        self.assertEqual(reg.total_count, 5)

    def test_serialization(self):
        entry = EquipmentEntry(tag="C-201", equipment_class="distillation_column", global_bbox=[100, 50, 200, 600])
        reg = EquipmentRegistry(equipment=[entry], drawing_notes="Test drawing")
        data = reg.model_dump()
        self.assertEqual(data["total_count"], 1)
        self.assertEqual(data["equipment"][0]["tag"], "C-201")
        self.assertEqual(data["equipment"][0]["equipment_class"], "distillation_column")


if __name__ == "__main__":
    unittest.main()
