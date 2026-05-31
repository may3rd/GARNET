import unittest

import numpy as np

from garnet.stage15_process_exports import build_stage15_process_exports, render_stage15_inline_mto_overlay


class Stage15ProcessExportsTests(unittest.TestCase):
    def test_build_stage15_exports_line_list_and_equipment_connectivity(self) -> None:
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [
                {"id": "equipment::pump_1", "type": "equipment", "position": {"x": 0, "y": 0}},
                {"id": "junction::j1", "type": "tee_junction", "position": {"x": 10, "y": 0}},
                {"id": "equipment::vessel_1", "type": "equipment", "position": {"x": 20, "y": 0}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "equipment::pump_1",
                    "target": "junction::j1",
                    "trace_length_px": 10,
                    "effective_line_number_ids": ["line_001"],
                    "effective_line_numbers": [
                        {
                            "id": "line_001",
                            "display_text": '3"_PL-26-003008-NZA1_Nl',
                            "normalized_text": '3"-PL-26-003008-NZA1-NL',
                        }
                    ],
                    "attachments": {
                        "inline_objects": [{"id": "valve_1", "source_object_id": "valve_1", "class_name": "gate valve"}],
                        "instrument_tags": [{"id": "inst_1", "source_object_id": "obj_inst_1"}],
                    },
                },
                {
                    "id": "e2",
                    "source": "junction::j1",
                    "target": "equipment::vessel_1",
                    "trace_length_px": 20,
                    "effective_line_number_ids": ["line_001"],
                },
            ],
        }

        result = build_stage15_process_exports(image_id="synthetic.png", corrected_graph_payload=graph_payload)

        self.assertEqual(result["line_list_payload"]["lines"][0]["line_number_id"], "line_001")
        self.assertEqual(result["line_list_payload"]["lines"][0]["display_texts"], ['3"_PL-26-003008-NZA1_Nl'])
        self.assertEqual(result["line_list_payload"]["lines"][0]["normalized_texts"], ['3"-PL-26-003008-NZA1-NL'])
        self.assertEqual(result["line_list_payload"]["lines"][0]["edge_ids"], ["e1", "e2"])
        self.assertEqual(result["line_list_payload"]["lines"][0]["total_length_px"], 30.0)
        self.assertEqual(result["equipment_connectivity_payload"]["connections"][0]["equipment_node_ids"], ["equipment::pump_1", "equipment::vessel_1"])
        self.assertEqual(result["inline_mto_payload"]["items"][0]["class_name"], "gate valve")
        self.assertEqual(result["inline_mto_payload"]["items"][0]["line_number_texts"], ['3"_PL-26-003008-NZA1_Nl'])
        self.assertEqual(result["instrument_index_payload"]["items"][0]["instrument_id"], "inst_1")
        self.assertEqual(result["summary"]["line_count"], 1)
        self.assertEqual(result["inline_mto_payload"]["items"][0]["material_basis"]["status"], "pending_line_property_data")
        self.assertEqual(result["inline_mto_payload"]["items"][0]["design_condition_basis"]["status"], "pending_line_property_data")

    def test_build_stage15_exports_groups_unassigned_edges(self) -> None:
        result = build_stage15_process_exports(
            image_id="synthetic.png",
            corrected_graph_payload={
                "image_id": "synthetic.png",
                "nodes": [],
                "edges": [{"id": "e1", "source": "n1", "target": "n2", "trace_length_px": 5}],
            },
        )

        line = result["line_list_payload"]["lines"][0]
        self.assertEqual(line["line_number_id"], "unassigned")
        self.assertEqual(line["assignment_state"], "missing")

    def test_build_stage15_exports_deduplicates_inline_and_instruments_globally(self) -> None:
        result = build_stage15_process_exports(
            image_id="synthetic.png",
            corrected_graph_payload={
                "image_id": "synthetic.png",
                "nodes": [],
                "edges": [
                    {
                        "id": "e1",
                        "attachments": {
                            "inline_objects": [{"id": "valve_1", "source_object_id": "valve_1", "class_name": "gate valve"}],
                            "instrument_tags": [{"id": "inst_1"}],
                        },
                    },
                    {
                        "id": "e2",
                        "attachments": {
                            "inline_objects": [{"id": "valve_1", "source_object_id": "valve_1", "class_name": "gate valve"}],
                            "instrument_tags": [{"id": "inst_1"}],
                        },
                    },
                ],
            },
        )

        self.assertEqual(len(result["inline_mto_payload"]["items"]), 1)
        self.assertEqual(result["inline_mto_payload"]["items"][0]["edge_ids"], ["e1", "e2"])
        self.assertEqual(len(result["instrument_index_payload"]["items"]), 1)
        self.assertEqual(result["instrument_index_payload"]["items"][0]["edge_ids"], ["e1", "e2"])

    def test_build_stage15_inline_mto_excludes_synthetic_tracer_hits(self) -> None:
        result = build_stage15_process_exports(
            image_id="synthetic.png",
            corrected_graph_payload={
                "image_id": "synthetic.png",
                "nodes": [],
                "edges": [
                    {
                        "id": "e1",
                        "effective_line_number_ids": ["line_001"],
                        "attachments": {
                            "inline_objects": [
                                {"id": "valve_1", "source_object_id": "valve_1", "class_name": "gate valve"},
                                {"id": "trace_1:hit_001", "source": "stage5b_hit", "class_name": "gate valve"},
                            ]
                        },
                    }
                ],
            },
        )

        self.assertEqual(len(result["inline_mto_payload"]["items"]), 1)
        self.assertEqual(result["inline_mto_payload"]["items"][0]["id"], "valve_1")
        self.assertEqual(len(result["inline_observations_payload"]["items"]), 2)
        self.assertEqual(result["summary"]["inline_item_count"], 1)
        self.assertEqual(result["summary"]["inline_observation_count"], 2)

    def test_build_stage15_inline_mto_selects_one_direct_line_when_effective_edge_has_conflict(self) -> None:
        result = build_stage15_process_exports(
            image_id="synthetic.png",
            corrected_graph_payload={
                "image_id": "synthetic.png",
                "nodes": [],
                "edges": [
                    {
                        "id": "e1",
                        "effective_line_number_ids": ["line_a", "line_b"],
                        "effective_line_numbers": [
                            {"id": "line_a", "display_text": "LINE-A", "normalized_text": "LINE-A"},
                            {"id": "line_b", "display_text": "LINE-B", "normalized_text": "LINE-B"},
                        ],
                        "line_numbers": [{"id": "line_a", "display_text": "LINE-A", "normalized_text": "LINE-A"}],
                        "attachments": {
                            "inline_objects": [
                                {
                                    "id": "valve_1",
                                    "source_object_id": "valve_1",
                                    "class_name": "gate valve",
                                    "trace_distance_px": 100,
                                }
                            ]
                        },
                    }
                ],
            },
        )

        item = result["inline_mto_payload"]["items"][0]
        self.assertEqual(item["line_number_ids"], ["line_a"])
        self.assertEqual(item["line_number_texts"], ["LINE-A"])
        self.assertEqual(item["line_number_assignment_state"], "selected")
        self.assertEqual([line["id"] for line in item["candidate_line_numbers"]], ["line_a", "line_b"])

    def test_render_stage15_inline_mto_overlay_draws_object_id_and_type(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        overlay = render_stage15_inline_mto_overlay(
            image,
            {
                "items": [
                    {
                        "id": "valve_1",
                        "class_name": "gate valve",
                        "bbox": {"x_min": 30, "y_min": 20, "x_max": 50, "y_max": 40},
                    }
                ]
            },
        )

        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(overlay.sum()), 0)


if __name__ == "__main__":
    unittest.main()
