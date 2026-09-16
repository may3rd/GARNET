import unittest

from garnet.ai_import import (
    AiImportError,
    Transform,
    check_frames,
    convert,
    fit_transform,
    merge_objects,
    project_port_to_bbox_edge,
)
from garnet.pid_extractor import EQUIPMENT_LABELS


def _ai_line(text: str, left: int, top: int) -> dict:
    return {"Object": "line number", "Left": left, "Top": top,
            "Width": 300, "Height": 20, "Score": 1.0, "Text": text}


def _job_line(text: str, left: int, top: int) -> dict:
    return {
        "id": f"line_number_{abs(hash(text)) % 999999:06d}",
        "normalized_text": text,
        "text": text,
        "bbox": {"x_min": left, "y_min": top, "x_max": left + 300, "y_max": top + 20},
    }


class TransformTests(unittest.TestCase):
    def test_apply_rounds_once(self) -> None:
        transform = Transform(dx=-146.5, dy=-106.5)
        self.assertEqual(transform.apply(1697.0, 884.0), (1550, 778))

    def test_check_frames_reports_mismatch_only(self) -> None:
        matching = {"image_width": 3961, "image_height": 3224}
        differing = {"coordinate_frame": {"width_px": 4962, "height_px": 3508}}
        self.assertEqual(check_frames([matching], (3961, 3224)), [])
        self.assertEqual(check_frames([matching, differing], (3961, 3224)), [(4962, 3508)])


class FitTransformTests(unittest.TestCase):
    """The real files are offset by a pure crop; alignment has to recover it."""

    OFFSET = (-146.5, -106.5)

    def _pairs(self) -> tuple[list[dict], list[dict]]:
        texts = [
            '6"-F-25-002019-L1A1-NI', '6"-F-25-002020-L1A1-NI',
            '2"-S12-66-059061-MS12-H', '1"-PL-25-002022-N2A1-NI',
            '3"-PL-24-015002-N2A1-NI', '2"-NAS-25-003004-B2A2-NI',
        ]
        ai, job = [], []
        for index, text in enumerate(texts):
            left, top = 500 + index * 400, 300 + index * 450
            ai.append(_ai_line(text, left, top))
            job.append(_job_line(text, int(left + self.OFFSET[0]), int(top + self.OFFSET[1])))
        return ai, job

    def test_recovers_known_translation(self) -> None:
        ai, job = self._pairs()
        transform, report = fit_transform(ai, job)
        self.assertAlmostEqual(transform.dx, self.OFFSET[0], delta=1.0)
        self.assertAlmostEqual(transform.dy, self.OFFSET[1], delta=1.0)
        self.assertEqual(transform.scale, 1.0)
        self.assertLessEqual(report.residual_median_x, 1.0)

    def test_duplicate_text_outlier_cannot_skew_the_fit(self) -> None:
        """A line number printed twice gives no way to tell which instance an OCR
        box belongs to, so those pairs must be excluded rather than averaged in."""
        ai, job = self._pairs()
        duplicate = '3"-CUL-25-002008-B1A2-NI'
        ai.append(_ai_line(duplicate, 3400, 1000))
        job.append(_job_line(duplicate, 100, 2900))   # wrong instance, ~2000px off
        job.append(_job_line(duplicate, 2800, 2500))  # the other instance
        transform, _ = fit_transform(ai, job)
        self.assertAlmostEqual(transform.dx, self.OFFSET[0], delta=1.0)
        self.assertAlmostEqual(transform.dy, self.OFFSET[1], delta=1.0)

    def test_refuses_when_too_few_pairs_match(self) -> None:
        with self.assertRaises(AiImportError):
            fit_transform([_ai_line('6"-F-25-002019-L1A1-NI', 100, 100)], [])


class ProjectPortTests(unittest.TestCase):
    """Supplied nozzle points sit on the equipment outline, which can be well
    inside the box (37px on the sample column). The tracer walks outward from
    the box edge, so they have to be projected onto it first."""

    BBOX = {"x_min": 1697, "y_min": 884, "x_max": 1951, "y_max": 2090}

    def test_inset_ports_move_out_to_the_facing_edge(self) -> None:
        self.assertEqual(project_port_to_bbox_edge(1734, 959, "LEFT", self.BBOX), (1697, 959))
        self.assertEqual(project_port_to_bbox_edge(1914, 959, "RIGHT", self.BBOX), (1951, 959))
        self.assertEqual(project_port_to_bbox_edge(1885, 896, "UP", self.BBOX), (1885, 884))
        self.assertEqual(project_port_to_bbox_edge(1824, 2000, "DOWN", self.BBOX), (1824, 2090))

    def test_ports_already_on_the_edge_are_untouched(self) -> None:
        self.assertEqual(project_port_to_bbox_edge(1951, 2008, "RIGHT", self.BBOX), (1951, 2008))
        self.assertEqual(project_port_to_bbox_edge(1824, 2090, "DOWN", self.BBOX), (1824, 2090))

    def test_along_edge_coordinate_is_clamped_into_the_box(self) -> None:
        # A nozzle marked past the corner still starts somewhere on that side.
        self.assertEqual(project_port_to_bbox_edge(9999, 959, "LEFT", self.BBOX), (1697, 959))
        self.assertEqual(project_port_to_bbox_edge(1800, 99, "LEFT", self.BBOX), (1697, 884))

    def test_unusable_direction_or_bbox_leaves_the_point_alone(self) -> None:
        self.assertEqual(project_port_to_bbox_edge(10, 20, "SIDEWAYS", self.BBOX), (10, 20))
        self.assertEqual(project_port_to_bbox_edge(10, 20, "LEFT", {}), (10, 20))


class ConvertEquipmentTests(unittest.TestCase):
    def _payload(self, **overrides) -> dict:
        item = {
            "Index": 1, "Object": "Column", "Tag": "V-2501",
            "Equipment_type": "Column",
            "Service": "Caustic Wash Column", "Size": "1400 mm",
            "Bounding_box_px": {"x_min": 100, "y_min": 200, "x_max": 300, "y_max": 900},
            "Ports": [
                {"mark": "AV", "size": '2"', "line_number": None, "side": "top",
                 "point_px": [200, 200]},
                {"mark": "AO", "size": '4"', "line_number": '4"-CUL-25-002007-B1A2-NI',
                 "side": "right", "point_px": [300, 400]},
            ],
        }
        item.update(overrides)
        return {"objects": [item]}

    def test_imports_equipment_with_prefixed_id_and_tag(self) -> None:
        result = convert(self._payload(), None, equipment_labels=EQUIPMENT_LABELS)
        self.assertEqual(len(result.objects), 1)
        obj = result.objects[0]
        self.assertTrue(obj["id"].startswith("equip_"))
        self.assertEqual(obj["class_name"], "column")
        self.assertEqual(obj["text"], "V-2501")
        self.assertEqual(obj["ai_import"]["Service"], "Caustic Wash Column")

    def test_static_mixer_maps_onto_a_known_equipment_label(self) -> None:
        payload = self._payload(Object="Static mixer", Equipment_type="Static mixer", Tag="X-2501")
        result = convert(payload, None, equipment_labels=EQUIPMENT_LABELS)
        self.assertEqual([o["class_name"] for o in result.objects], ["mixer"])

    def test_unknown_equipment_type_is_reported_not_dropped(self) -> None:
        payload = self._payload(Object="Sootblower", Equipment_type="Sootblower", Tag="Z-9")
        result = convert(payload, None, equipment_labels=EQUIPMENT_LABELS)
        self.assertEqual(result.objects, [])
        skipped = [r for r in result.report if r.status == "skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertIn("Sootblower", skipped[0].reason)

    def test_ports_convert_side_to_direction_and_follow_the_transform(self) -> None:
        result = convert(self._payload(), None, equipment_labels=EQUIPMENT_LABELS,
                         transform=Transform(dx=-10, dy=-20))
        ports = result.equipment_ports["equip_v_2501"]
        self.assertEqual([p["direction"] for p in ports], ["UP", "RIGHT"])
        self.assertEqual((ports[0]["x"], ports[0]["y"]), (190, 180))
        self.assertEqual(ports[1]["line_number"], '4"-CUL-25-002007-B1A2-NI')

    def test_port_with_unusable_side_is_reported(self) -> None:
        payload = self._payload(Ports=[{"mark": "EO", "side": None, "point_px": [1, 2]}])
        result = convert(payload, None, equipment_labels=EQUIPMENT_LABELS)
        self.assertEqual(result.equipment_ports, {})
        self.assertTrue(any(r.kind == "port" and r.status == "skipped" for r in result.report))

    def test_unresolved_notes_survive_as_info_rows(self) -> None:
        payload = self._payload()
        payload["unresolved"] = [{"what": "nozzle EO size", "why": "not annotated"}]
        payload["excluded_by_rule"] = {"manways": "no process line terminates"}
        result = convert(payload, None, equipment_labels=EQUIPMENT_LABELS)
        notes = [r for r in result.report if r.status == "info"]
        self.assertEqual({n.key for n in notes}, {"nozzle EO size", "manways"})


class ConvertLineNumberTests(unittest.TestCase):
    def test_entries_match_the_stage4_schema(self) -> None:
        payload = {"objects": [_ai_line('3"-PL-25-002013-B1A2-NI', 700, 2300)],
                   "image_width": 4962, "image_height": 3508}
        result = convert(None, payload, equipment_labels=EQUIPMENT_LABELS)
        entry = result.line_numbers[0]
        expected_keys = {
            "id", "source_object_id", "bbox", "text", "normalized_text", "ocr_region_id",
            "ocr_source", "score", "distance_px", "ocr_confirmed", "detection_confidence",
            "fused_confidence", "semantic_class", "review_state",
        }
        self.assertEqual(set(entry), expected_keys)
        self.assertEqual(entry["semantic_class"], "line_number")
        self.assertEqual(entry["review_state"], "ocr_confirmed")
        self.assertTrue(entry["ocr_confirmed"])

    def test_empty_text_is_reported_not_imported(self) -> None:
        payload = {"objects": [_ai_line("", 10, 10)]}
        result = convert(None, payload, equipment_labels=EQUIPMENT_LABELS)
        self.assertEqual(result.line_numbers, [])
        self.assertTrue(any(r.status == "skipped" for r in result.report))


class MergeObjectsTests(unittest.TestCase):
    def test_replaces_equipment_bucket_and_keeps_everything_else(self) -> None:
        existing = [
            {"id": "obj_000001", "class_name": "gate valve", "bbox": {}},
            {"id": "obj_000002", "class_name": "pump", "bbox": {}},
            {"id": "obj_000003", "class_name": "line number", "bbox": {}},
        ]
        imported = [{"id": "equip_p_2504a", "class_name": "pump", "bbox": {}}]
        merged = merge_objects(existing, imported, EQUIPMENT_LABELS)
        self.assertEqual(
            [o["id"] for o in merged],
            ["obj_000001", "obj_000003", "equip_p_2504a"],
        )


class Stage3DerivationTests(unittest.TestCase):
    """Guards the id contract the whole import depends on: stage5b classifies a
    terminal as equipment purely by the ``equip_`` prefix."""

    def test_derivation_preserves_prefixed_id_and_tag(self) -> None:
        import json
        import os
        import tempfile

        import api

        result = convert(
            {"objects": [{
                "Index": 1, "Object": "Pump", "Equipment_type": "Pump", "Tag": "P-2504A",
                "Bounding_box_px": {"x_min": 10, "y_min": 20, "x_max": 60, "y_max": 70},
            }]},
            None,
            equipment_labels=EQUIPMENT_LABELS,
        )
        with tempfile.TemporaryDirectory() as job_dir:
            api._derive_stage3_equipment_bboxes(job_dir, result.objects)
            with open(os.path.join(job_dir, "stage3_equipment_bboxes.json")) as handle:
                equipment = json.load(handle)["equipment"]
        self.assertEqual(len(equipment), 1)
        self.assertTrue(equipment[0]["id"].startswith("equip_"))
        self.assertEqual(equipment[0]["tag"], "P-2504A")


if __name__ == "__main__":
    unittest.main()
