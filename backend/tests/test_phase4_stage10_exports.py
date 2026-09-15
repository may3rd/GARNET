import json
import unittest

from garnet.stage10_process_exports import build_stage10_process_exports


class Phase4Stage10ExportTests(unittest.TestCase):
    def test_canonical_equipment_refs_and_ordered_route_occurrences(self):
        graph = {
            "nodes": [
                {"id": "legacy_pump", "type": "equipment_port"},
                {"id": "legacy_vessel", "type": "equipment"},
            ],
            "edges": [{
                "id": "edge_2",
                "source": "legacy_pump",
                "target": "legacy_vessel",
                "source_equipment_id": "equipment::P-101::pump_1",
                "source_port_id": "equipment::P-101::pump_1::port::01",
                "terminal_equipment_id": "equipment::P-101::vessel_1",
                "terminal_port_id": "equipment::P-101::vessel_1::port::xy_100_20",
                "effective_line_number_ids": ["L-1"],
                "attachments": {
                    "inline_objects": [{"id": "valve", "source_object_id": "valve", "class_name": "valve", "trace_distance_px": 20}],
                    "instrument_tags": [{"id": "PT-1", "relationship": "measures", "trace_distance_px": 5}],
                },
            }, {
                "id": "edge_1",
                "source": "legacy_pump",
                "target": "legacy_vessel",
                "source_equipment_id": "equipment::P-101::pump_1",
                "source_port_id": "equipment::P-101::pump_1::port::01",
                "terminal_equipment_id": "equipment::P-101::vessel_1",
                "terminal_port_id": "equipment::P-101::vessel_1::port::xy_100_20",
                "effective_line_number_ids": ["L-1"],
                "attachments": {
                    "inline_objects": [{"id": "valve", "source_object_id": "valve", "class_name": "valve", "trace_distance_px": 10}],
                    "instrument_tags": [{"id": "PT-1", "trace_distance_px": 2}],
                },
            }],
        }
        result = build_stage10_process_exports(image_id="P-101", corrected_graph_payload=graph)
        connectivity = result["equipment_connectivity_payload"]
        self.assertEqual(connectivity["connections"][0]["equipment_ids"], ["equipment::P-101::pump_1", "equipment::P-101::vessel_1"])
        item = result["inline_mto_payload"]["items"][0]
        self.assertEqual(item["canonical_id"], "inline::P-101::valve")
        self.assertEqual(item["route_occurrences"], [{"edge_id": "edge_1", "route_position_px": 10.0}, {"edge_id": "edge_2", "route_position_px": 20.0}])
        instrument = result["instrument_index_payload"]["items"][0]
        self.assertEqual(instrument["canonical_id"], "instrument::P-101::PT-1")
        self.assertEqual(instrument["relationship_state"], "observed")
        self.assertEqual(instrument["relationships"], [{"type": "measures", "target": "equipment::P-101::vessel_1", "evidence": "instrument_attachment"}])

    def test_canonical_only_connectivity_does_not_require_legacy_nodes(self):
        result = build_stage10_process_exports(
            image_id="P-101",
            corrected_graph_payload={"edges": [{
                "id": "e1",
                "source_equipment_id": "equipment::P-101::pump_1",
                "source_port_id": "equipment::P-101::pump_1::port::01",
                "terminal_equipment_id": "equipment::P-101::vessel_1",
                "terminal_port_id": "equipment::P-101::vessel_1::port::01",
                "effective_line_number_ids": ["L-1"],
            }]},
        )
        connectivity = result["equipment_connectivity_payload"]
        self.assertEqual(connectivity["connections"][0]["equipment_node_ids"], [])
        self.assertEqual(connectivity["connections"][0]["equipment_ids"], ["equipment::P-101::pump_1", "equipment::P-101::vessel_1"])
        self.assertEqual(connectivity["connections"][0]["port_ids"], ["equipment::P-101::pump_1::port::01", "equipment::P-101::vessel_1::port::01"])
        self.assertEqual(connectivity["direct_edge_connections"][0]["equipment_node_ids"], [])

    def test_instrument_without_explicit_relationship_stays_unresolved(self):
        result = build_stage10_process_exports(
            image_id="P-101",
            corrected_graph_payload={"edges": [{"id": "e1", "attachments": {"instrument_tags": [{"id": "FT-1"}]}}]},
        )
        instrument = result["instrument_index_payload"]["items"][0]
        self.assertEqual(instrument["relationship_state"], "unresolved")
        self.assertEqual(instrument["relationships"], [])

    def test_nonfinite_route_position_is_unresolved_and_json_safe(self):
        result = build_stage10_process_exports(
            image_id="P-101",
            corrected_graph_payload={"edges": [{
                "id": "e1",
                "attachments": {"inline_objects": [{"id": "valve", "trace_distance_px": "nan"}]},
            }]},
        )
        self.assertIsNone(result["inline_observations_payload"]["items"][0]["route_position_px"])
        json.dumps(result, allow_nan=False)

    def test_instrument_occurrences_with_same_normalized_tag_share_canonical_item(self):
        result = build_stage10_process_exports(
            image_id="P-101",
            corrected_graph_payload={"edges": [
                {"id": "e2", "attachments": {"instrument_tags": [{"id": "ocr_b", "normalized_text": "PT-101", "text": "PT-101"}]}},
                {"id": "e1", "attachments": {"instrument_tags": [{"id": "ocr_a", "normalized_text": "PT-101", "text": "PT-101"}]}},
            ]},
        )
        items = result["instrument_index_payload"]["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["instrument_id"], "ocr_a")
        self.assertEqual(items[0]["canonical_instrument_id"], "PT-101")
        self.assertEqual(items[0]["canonical_id"], "instrument::P-101::PT-101")
        self.assertEqual([item["id"] for item in items[0]["occurrences"]], ["ocr_a", "ocr_b"])

    def test_line_occurrences_with_same_normalized_text_share_canonical_line(self):
        def edge(edge_id, occurrence_id):
            return {
                "id": edge_id,
                "source": "equipment::P-101::pump",
                "target": "equipment::P-101::vessel",
                "source_equipment_id": "equipment::P-101::pump",
                "terminal_equipment_id": "equipment::P-101::vessel",
                "effective_line_number_ids": [occurrence_id],
                "effective_line_numbers": [{"id": occurrence_id, "normalized_text": "3-PL-101", "display_text": "3-PL-101"}],
            }
        result = build_stage10_process_exports(
            image_id="P-101",
            corrected_graph_payload={"edges": [edge("e2", "ocr_b"), edge("e1", "ocr_a")]},
        )
        lines = result["line_list_payload"]["canonical_lines"]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["line_number_ids"], ["ocr_a", "ocr_b"])
        self.assertEqual(lines[0]["edge_ids"], ["e1", "e2"])
        self.assertEqual(result["equipment_connectivity_payload"]["connections"][0]["canonical_line_ids"], ["line::P-101::3-PL-101"])
        canonical_connection = result["equipment_connectivity_payload"]["canonical_connections"][0]
        self.assertEqual(canonical_connection["canonical_line_id"], "line::P-101::3-PL-101")
        self.assertEqual(canonical_connection["line_number_ids"], ["ocr_a", "ocr_b"])
        self.assertEqual(canonical_connection["edge_ids"], ["e1", "e2"])

    def test_conflicting_line_records_on_one_edge_keep_separate_canonical_lines(self):
        graph = {"edges": [{
            "id": "e1",
            "source_equipment_id": "equipment::P-101::pump",
            "terminal_equipment_id": "equipment::P-101::vessel",
            "effective_line_number_ids": ["line_a", "line_b"],
            "effective_line_numbers": [
                {"id": "line_a", "normalized_text": "LINE-A", "display_text": "LINE-A"},
                {"id": "line_b", "normalized_text": "LINE-B", "display_text": "LINE-B"},
            ],
        }]}
        result = build_stage10_process_exports(image_id="P-101", corrected_graph_payload=graph)
        canonical = result["line_list_payload"]["canonical_lines"]
        self.assertEqual([item["canonical_line_id"] for item in canonical], ["line::P-101::LINE-A", "line::P-101::LINE-B"])


if __name__ == "__main__":
    unittest.main()
