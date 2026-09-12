import sys
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.graph_export_adapter import build_graph_v1_payload, reproject_polyline


def _payload() -> dict:
    return build_graph_v1_payload(
        stage12_graph={
            "image_id": "sample.png",
            "nodes": [
                {
                    "id": "equipment::obj_1",
                    "type": "pump",
                    "kind": "pump",
                    "position": {"x": 15.0, "y": 25.0},
                    "review_state": "provisional",
                },
                {
                    "id": "junction_1",
                    "type": "junction",
                    "kind": "junction",
                    "position": {"x": 50.0, "y": 60.0},
                    "review_state": "accepted",
                },
            ],
            "edges": [
                {
                    "id": "edge_1",
                    "source": "equipment::obj_1",
                    "target": "junction_1",
                    "review_state": "provisional",
                    "flow_direction": "source_to_target",
                    "flow_direction_confidence": 0.75,
                    "polyline": [{"col": 10, "row": 20}, {"col": 50, "row": 60}],
                }
            ],
        },
        objects_payload={
            "image_id": "sample.png",
            "objects": [
                {
                    "id": "obj_1",
                    "class_name": "pump",
                    "confidence": 0.82,
                    "bbox": {"x_min": 10, "y_min": 20, "x_max": 20, "y_max": 30},
                }
            ],
        },
        line_numbers_payload={
            "line_numbers": [
                {
                    "text": "10-P-100",
                    "normalized_text": "10-P-100",
                    "confidence": 0.7,
                    "bbox": {"x_min": 10, "y_min": 20, "x_max": 20, "y_max": 30},
                }
            ]
        },
        instrument_tags_payload={"instrument_tags": []},
        image_dimensions={"width": 100, "height": 80},
    )


class GraphExportAdapterTests(unittest.TestCase):
    def test_reproject_polyline_col_row_to_xy(self) -> None:
        self.assertEqual(reproject_polyline([{"col": 4, "row": 9}]), [{"x": 4.0, "y": 9.0}])

    def test_node_has_all_required_fields(self) -> None:
        node = _payload()["nodes"][0]
        self.assertTrue(
            {"id", "type", "bbox", "confidence", "text", "role", "provenance", "geometry", "patch_link", "tags"}
            <= set(node)
        )

    def test_edge_has_all_required_fields(self) -> None:
        edge = _payload()["edges"][0]
        self.assertTrue({"id", "src", "dst", "type", "confidence", "directed", "provenance", "geometry"} <= set(edge))

    def test_polyline_uses_xy_not_col_row(self) -> None:
        point = _payload()["edges"][0]["geometry"]["polyline"][0]
        self.assertEqual(set(point), {"x", "y"})

    def test_node_types_subset_of_enum(self) -> None:
        payload = _payload()
        enum_values = set(payload["classes"]["node_types"])
        self.assertTrue({node["type"] for node in payload["nodes"]} <= enum_values)

    def test_edge_types_are_solid_or_nonsolid(self) -> None:
        self.assertTrue({edge["type"] for edge in _payload()["edges"]} <= {"solid", "non_solid"})

    def test_directed_true_when_flow_direction_set(self) -> None:
        self.assertTrue(_payload()["edges"][0]["directed"])

    def test_confidence_bounds_0_1(self) -> None:
        payload = _payload()
        confidences = [item["confidence"] for item in payload["nodes"] + payload["edges"]]
        self.assertTrue(all(0.0 <= confidence <= 1.0 for confidence in confidences))

    def test_provenance_fields_present(self) -> None:
        provenance = _payload()["nodes"][0]["provenance"]
        self.assertTrue({"annotated_by", "annotated_at", "source", "notes"} <= set(provenance))

    def test_top_level_schema_fields(self) -> None:
        payload = _payload()
        self.assertTrue(
            {
                "schema_version",
                "coordinate_system",
                "document",
                "tiling",
                "classes",
                "nodes",
                "edges",
                "constraints",
                "recommended_defaults",
            }
            <= set(payload)
        )

    def test_page_connection_uses_connector_label_payload(self) -> None:
        payload = build_graph_v1_payload(
            stage12_graph={
                "image_id": "sample.png",
                "nodes": [
                    {
                        "id": "connection::obj_9",
                        "type": "page connection",
                        "kind": "equipment_attachment",
                        "position": {"x": 15.0, "y": 25.0},
                    }
                ],
                "edges": [],
            },
            objects_payload={
                "image_id": "sample.png",
                "objects": [
                    {
                        "id": "obj_9",
                        "class_name": "page connection",
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 20, "y_max": 30},
                    }
                ],
            },
            page_connector_labels_payload={
                "connectors": [
                    {
                        "object_id": "obj_9",
                        "labels": [
                            {
                                "normalized_text": "SHEET P-101",
                                "page_reference": {
                                    "reference_type": "sheet",
                                    "reference_value": "P-101",
                                    "matched_text": "SHEET P-101",
                                },
                            }
                        ],
                    }
                ]
            },
            image_dimensions={"width": 100, "height": 80},
        )

        node = payload["nodes"][0]
        self.assertEqual(node["text"], "SHEET P-101")
        self.assertEqual(node["tags"]["page_reference"]["reference_value"], "P-101")

    def test_off_page_connector_prefers_first_sorted_source_edge(self) -> None:
        payload = build_graph_v1_payload(
            stage12_graph={
                "image_id": "sample.png",
                "nodes": [
                    {
                        "id": "connection::obj_9",
                        "type": "page connection",
                        "kind": "page connection",
                        "position": {"x": 15.0, "y": 25.0},
                    }
                ],
                "edges": [
                    {
                        "id": "source_b",
                        "source": "connection::obj_9",
                        "target": "attach::obj_9",
                        "polyline": [{"x": 10, "y": 20}, {"x": 20, "y": 20}],
                    },
                    {
                        "id": "target_a",
                        "source": "endpoint_1",
                        "target": "connection::obj_9",
                        "polyline": [{"x": 0, "y": 0}, {"x": 10, "y": 10}],
                    },
                    {
                        "id": "source_a",
                        "source": "connection::obj_9",
                        "target": "attach::obj_10",
                        "polyline": [{"x": 10, "y": 20}, {"x": 30, "y": 20}],
                    },
                ],
            },
            objects_payload={
                "image_id": "sample.png",
                "objects": [
                    {
                        "id": "obj_9",
                        "class_name": "page connection",
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 20, "y_max": 30},
                    }
                ],
            },
            page_connector_labels_payload={
                "connectors": [
                    {
                        "object_id": "obj_9",
                        "labels": [
                            {
                                "normalized_text": "SHEET P-101",
                                "page_reference": {
                                    "reference_type": "sheet",
                                    "reference_value": "P-101",
                                    "matched_text": "SHEET P-101",
                                },
                            }
                        ],
                    }
                ]
            },
            image_dimensions={"width": 100, "height": 80},
        )

        edges_by_id = {e["id"]: e for e in payload["edges"]}
        off = edges_by_id["source_a"]["off_page_connector"]
        self.assertEqual(off["reference_type"], "sheet")
        self.assertEqual(off["reference_value"], "P-101")
        self.assertEqual(off["exit_terminal"], "source")
        self.assertEqual(off["direction"], "bidirectional")
        self.assertEqual(off["local_edge_id"], "source_a")
        self.assertNotIn("off_page_connector", edges_by_id["source_b"])
        self.assertNotIn("off_page_connector", edges_by_id["target_a"])

    def test_off_page_connector_exports_separate_reference_and_line_key(self) -> None:
        payload = build_graph_v1_payload(
            stage12_graph={
                "image_id": "DWG-100",
                "nodes": [{"id": "connection::obj_9", "type": "page connection"}],
                "edges": [{"id": "edge_1", "source": "connection::obj_9", "target": "node_1"}],
            },
            page_connector_labels_payload={
                "connectors": [{
                    "object_id": "obj_9",
                    "labels": [
                        {
                            "text": "10-P-100-A",
                            "normalized_text": "10-P-100-A",
                            "semantic_class": "line_number",
                            "distance_px": 4.0,
                        },
                        {
                            "text": "SEE DWG 200-02",
                            "normalized_text": "SEE DWG 200-02",
                            "semantic_class": "reference",
                            "distance_px": 9.0,
                            "page_reference": {
                                "reference_type": "drawing",
                                "reference_value": "200-02",
                                "matched_text": "DWG 200-02",
                            },
                        },
                    ],
                }]
            },
        )

        connector = payload["edges"][0]["off_page_connector"]
        self.assertEqual(connector["connector_key"], "10-P-100-A")
        self.assertEqual(connector["target_sheet_reference"], "200-02")
        self.assertEqual(connector["raw_reference_text"], "SEE DWG 200-02")
        self.assertEqual(payload["nodes"][0]["tags"]["page_reference"]["reference_value"], "200-02")

    def test_off_page_connector_uses_first_sorted_target_edge(self) -> None:
        payload = build_graph_v1_payload(
            stage12_graph={
                "nodes": [{"id": "connection::obj_9", "type": "page connection"}],
                "edges": [
                    {"id": "target_b", "source": "node_b", "target": "connection::obj_9"},
                    {"id": "target_a", "source": "node_a", "target": "connection::obj_9"},
                ],
            },
            page_connector_labels_payload={
                "connectors": [{
                    "object_id": "obj_9",
                    "labels": [{"page_reference": {"reference_type": "drawing", "reference_value": "D-12"}}],
                }]
            },
        )

        edges_by_id = {edge["id"]: edge for edge in payload["edges"]}
        off = edges_by_id["target_a"]["off_page_connector"]
        self.assertEqual(off["exit_terminal"], "destination")
        self.assertEqual(off["direction"], "bidirectional")
        self.assertNotIn("off_page_connector", edges_by_id["target_b"])

    def test_off_page_connector_without_reference_is_exported_for_merge_review(self) -> None:
        payload = build_graph_v1_payload(
            stage12_graph={
                "nodes": [{"id": "connection::obj_9", "type": "page connection"}],
                "edges": [{"id": "edge_1", "source": "connection::obj_9", "target": "node_1"}],
            },
            page_connector_labels_payload={"connectors": [{"object_id": "obj_9", "labels": []}]},
        )

        connector = payload["edges"][0]["off_page_connector"]
        self.assertEqual(connector["connector_key"], "")
        self.assertEqual(connector["target_sheet_reference"], "")
        self.assertEqual(connector["direction"], "bidirectional")

    def test_reviewed_connector_key_uses_corrected_line_label(self) -> None:
        payload = build_graph_v1_payload(
            stage12_graph={
                "nodes": [{"id": "connection::obj_9", "type": "page connection"}],
                "edges": [{
                    "id": "edge_1", "source": "connection::obj_9", "target": "node_1",
                    "line_number_review_state": "human_reviewed",
                    "effective_line_numbers": [{"id": "line_new", "normalized_text": "NEW-LINE"}],
                    "effective_line_number_ids": ["line_new"],
                }],
            },
            page_connector_labels_payload={"connectors": [{"object_id": "obj_9", "connector_key": "OLD-LINE", "page_reference": {"reference_type": "sheet", "reference_value": "P-101"}}]},
        )
        connector = payload["edges"][0]["off_page_connector"]
        self.assertEqual(connector["connector_key"], "NEW-LINE")
        self.assertEqual(connector["target_sheet_reference"], "P-101")

    def test_reviewed_connector_without_line_text_clears_stale_key(self) -> None:
        payload = build_graph_v1_payload(
            stage12_graph={
                "nodes": [{"id": "connection::obj_9", "type": "page connection"}],
                "edges": [{
                    "id": "edge_1", "source": "connection::obj_9", "target": "node_1",
                    "line_number_review_state": "human_reviewed",
                    "effective_line_numbers": [{"id": "line_new"}],
                    "effective_line_number_ids": ["line_new"],
                }],
            },
            page_connector_labels_payload={"connectors": [{"object_id": "obj_9", "connector_key": "OLD-LINE", "page_reference": {"reference_type": "sheet", "reference_value": "P-101"}}]},
        )
        self.assertEqual(payload["edges"][0]["off_page_connector"]["connector_key"], "")

    def test_semantic_projection_keeps_canonical_line_and_ocr_occurrences(self) -> None:
        payload = build_graph_v1_payload({
            "nodes": [],
            "edges": [
                {"id": "route_b", "source": "a", "target": "b", "line_number_ids": ["ln-1"],
                 "line_numbers": [{"id": "ln-1", "text": "10-P-100", "normalized_text": "10-P-100", "source": "ocr-b"}]},
                {"id": "route_a", "source": "c", "target": "d", "line_number_ids": ["ln-2"],
                 "line_numbers": [{"id": "ln-2", "text": "10 P 100", "normalized_text": "10-P-100", "source": "ocr-a"}]},
            ],
        })
        line = payload["lines"][0]
        self.assertEqual(line["id"], "line::unknown::10-P-100")
        self.assertEqual(line["normalized_text"], "10-P-100")
        self.assertEqual([item["edge_id"] for item in line["occurrences"]], ["route_a", "route_b"])
        self.assertEqual(payload["line_to_edges"]["ln-1"], ["route_b"])
        self.assertEqual(payload["line_to_edges"]["ln-2"], ["route_a"])
        self.assertEqual(payload["edges"][0]["canonical_line_ids"], ["line::unknown::10-P-100"])
        self.assertEqual(payload["canonical_line_to_edges"]["line::unknown::10-P-100"], ["route_a", "route_b"])

    def test_semantic_projection_orders_inline_route_and_types_relationships(self) -> None:
        payload = build_graph_v1_payload({
            "nodes": [],
            "edges": [{
                "id": "route-1", "source": "a", "target": "b",
                "polyline": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
                "attachments": {"inline_objects": [
                    {"id": "v-late", "source_object_id": "v-late", "class_name": "gate_valve", "trace_distance_px": 80},
                    {"id": "v-early", "source_object_id": "v-early", "class_name": "reducer", "trace_distance_px": 20},
                ]},
            }],
        })
        edge = payload["edges"][0]
        self.assertEqual(edge["ordered_inline_object_ids"], ["inline::unknown::v-early", "inline::unknown::v-late"])
        self.assertEqual(edge["ordered_inline_objects"][0]["inline_object_id"], "inline::unknown::v-early")
        inline = {item["id"]: item for item in payload["inline_objects"]}
        self.assertEqual([item["route_position_px"] for item in inline["inline::unknown::v-early"]["occurrences"]], [20.0])
        relation = [item for item in payload["relationships"] if item["type"] == "has_inline_object"]
        self.assertEqual(len(relation), 2)
        self.assertTrue(all(item["semantic_state"] == "observed" for item in relation))

    def test_semantic_projection_exports_canonical_instrument_and_unresolved_association(self) -> None:
        payload = build_graph_v1_payload({
            "nodes": [],
            "edges": [{
                "id": "route-1", "source": "a", "target": "b",
                "attachments": {"instrument_tags": [{
                    "id": "pt-101", "text": "PT-101", "normalized_text": "PT-101",
                    "projected_xy": [15, 20], "trace_distance_px": 15,
                }]},
            }],
        }, instrument_tags_payload={"instrument_tags": [{"id": "pt-101", "text": "PT-101"}]})
        instrument = payload["instruments"][0]
        self.assertEqual(instrument["id"], "instrument::unknown::PT-101")
        self.assertEqual(instrument["tag"], "PT-101")
        relations = [item for item in payload["relationships"] if item["type"] == "instrument_association"]
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["semantic_state"], "unresolved")
        self.assertEqual(payload["edges"][0]["instrument_ids"], ["instrument::unknown::PT-101"])

    def test_semantic_projection_exports_typed_instrument_relation_and_equipment_ports(self) -> None:
        payload = build_graph_v1_payload({
            "equipment_catalog": [{"id": "pump-1", "class_name": "pump"}],
            "equipment_ports": [{"id": "pump-1:out", "equipment_id": "pump-1"}],
            "nodes": [],
            "edges": [{"id": "route-1", "source": "pump-1", "target": "v-1",
                       "source_equipment_id": "pump-1", "source_port_id": "pump-1:out",
                       "terminal_equipment_id": "valve-1", "terminal_port_id": "valve-1:in",
                       "attachments": {"instrument_tags": [{"id": "ft-1", "text": "FT-1", "relationship_type": "measures"}]}}],
        })
        self.assertEqual(payload["equipment"][0]["id"], "pump-1")
        self.assertEqual(payload["equipment_ports"][0]["id"], "pump-1:out")
        edge = payload["edges"][0]
        self.assertEqual(edge["source_equipment_id"], "pump-1")
        self.assertEqual(edge["source_port_id"], "pump-1:out")
        self.assertEqual(edge["terminal_equipment_id"], "valve-1")
        self.assertEqual(edge["terminal_port_id"], "valve-1:in")
        relation = [item for item in payload["relationships"] if item["source"].lower().endswith("ft-1")][0]
        self.assertEqual(relation["type"], "measures")
        self.assertEqual(relation["semantic_state"], "observed")
        self.assertTrue(relation["source"].lower().endswith("ft-1"))
        self.assertEqual(relation["target"], "route-1")

    def test_instrument_occurrences_share_canonical_normalized_tag(self) -> None:
        payload = build_graph_v1_payload({"nodes": [], "edges": [
            {"id": "r1", "source": "a", "target": "b", "attachments": {"instrument_tags": [{"id": "ocr-1", "text": "PT 101", "normalized_text": "PT-101"}]}},
            {"id": "r2", "source": "c", "target": "d", "attachments": {"instrument_tags": [{"id": "ocr-2", "text": "PT-101", "normalized_text": "PT-101"}]}},
        ]})
        self.assertEqual(len(payload["instruments"]), 1)
        self.assertEqual(len(payload["instruments"][0]["occurrences"]), 2)
        self.assertEqual({item["occurrence_id"] for item in payload["instruments"][0]["occurrences"]}, {"ocr-1", "ocr-2"})

    def test_projection_does_not_mutate_catalog_and_deduplicates_attached_detector(self) -> None:
        graph = {"image_id": "sheet-a", "nodes": [], "equipment_catalog": [{"id": "eq-1"}], "edges": [
            {"id": "r1", "source": "a", "target": "b", "source_obj_id": "eq-1",
             "attachments": {"instrument_tags": [{"id": "pt-1", "text": "PT-1", "normalized_text": "PT-1"}]}}
        ]}
        original = __import__("copy").deepcopy(graph)
        payload = build_graph_v1_payload(graph, objects_payload={"objects": [{"id": "eq-1", "class_name": "pump", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}]}, instrument_tags_payload={"instrument_tags": [{"id": "pt-1", "text": "PT-1", "normalized_text": "PT-1"}]})
        self.assertEqual(graph, original)
        self.assertEqual(len(payload["instruments"][0]["occurrences"]), 1)
        self.assertEqual(payload["equipment"][0]["class_name"], "pump")

    def test_legacy_equipment_fallback_uses_edge_object_reference(self) -> None:
        payload = build_graph_v1_payload({"nodes": [], "edges": [{"id": "r1", "source": "a", "target": "b", "source_obj_id": "eq-1"}]}, objects_payload={"objects": [{"id": "eq-1", "class_name": "pump"}, {"id": "unrelated", "class_name": "valve"}]})
        self.assertEqual([item["id"] for item in payload["equipment"]], ["eq-1"])

    def test_repeated_attachment_occurrences_keep_evidence_but_dedupe_graph_relationships(self) -> None:
        payload = build_graph_v1_payload({"nodes": [], "edges": [{
            "id": "r1", "source": "a", "target": "b", "attachments": {
                "inline_objects": [{"id": "v1", "class_name": "valve", "trace_distance_px": 2}, {"id": "v1", "class_name": "valve", "trace_distance_px": 4}],
                "instrument_tags": [{"id": "pt1", "normalized_text": "PT-1"}, {"id": "pt1", "normalized_text": "PT-1"}],
            },
        }]})
        self.assertEqual(len(payload["inline_objects"][0]["occurrences"]), 2)
        self.assertEqual(len(payload["instruments"][0]["occurrences"]), 2)
        self.assertEqual(len([r for r in payload["relationships"] if r["type"] == "has_inline_object"]), 1)
        self.assertEqual(len([r for r in payload["relationships"] if r["type"] == "instrument_association"]), 1)
        self.assertEqual(len(payload["edges"][0]["ordered_inline_objects"]), 1)

    def test_legacy_edge_refs_and_nonfinite_route_positions_are_safe(self) -> None:
        payload = build_graph_v1_payload({"nodes": [], "edges": [{
            "id": "r1", "legacy_source": "old-a", "legacy_target": "old-b", "source": "eq-a", "target": "eq-b",
            "attachments": {"inline_objects": [{"id": "v1", "trace_distance_px": float("nan")}]},
        }]})
        edge = payload["edges"][0]
        self.assertEqual((edge["src"], edge["dst"]), ("old-a", "old-b"))
        self.assertEqual((edge["canonical_src"], edge["canonical_dst"]), ("eq-a", "eq-b"))
        self.assertIsNone(edge["ordered_inline_objects"][0]["route_position_px"])
        json.dumps(payload, allow_nan=False)

    def test_malformed_geometry_and_duplicate_catalog_refs_remain_strict_and_connected(self) -> None:
        payload = build_graph_v1_payload({"nodes": [], "equipment_catalog": [{"id": "eq-1"}, {"id": "eq-1"}], "equipment_ports": [{"id": "p-1", "equipment_id": "eq-1"}, {"id": "p-1", "equipment_id": "eq-1"}], "edges": [{"id": "r", "source": "eq-1", "target": "x", "terminal_equipment_id": "eq-2", "terminal_port_id": "p-2", "polyline": [{"x": float("nan"), "y": 1}, {"x": 2, "y": float("inf")}]}]})
        self.assertEqual(payload["edges"][0]["geometry"]["polyline"], [])
        self.assertEqual(len(payload["equipment"]), 2)
        self.assertEqual(len(payload["equipment_ports"]), 2)
        self.assertEqual(len({item["id"] for item in payload["relationships"]}), len(payload["relationships"]))
        json.dumps(payload, allow_nan=False)

    def test_strict_json_sanitizes_all_persisted_evidence(self) -> None:
        payload = build_graph_v1_payload({"nodes": [{"id": "n", "position": {"x": float("nan"), "y": float("inf")}}], "equipment_catalog": [{"id": "eq", "bbox": {"x": float("nan")}}], "equipment_ports": [{"id": "p", "equipment_id": "eq", "position": [float("inf"), 2]}], "edges": [{"id": "e", "source": "n", "target": "n", "line_numbers": [{"id": "l", "text": "L", "evidence": float("nan")}], "attachments": {"inline_objects": [{"id": "v", "projected_xy": [float("nan"), 1]}]}}]})
        json.dumps(payload, allow_nan=False)

    def test_missing_port_owner_is_synthesized_and_consistent(self) -> None:
        payload = build_graph_v1_payload({"nodes": [], "edges": [{"id": "e", "source": "a", "target": "b", "source_port_id": "p"}]})
        edge = payload["edges"][0]
        owner = edge["source_equipment_id"]
        self.assertTrue(owner)
        self.assertEqual(payload["equipment_ports"][0]["equipment_id"], owner)
        self.assertTrue(any(r["type"] == "has_port" and r["source"] == owner for r in payload["relationships"]))
        self.assertTrue(any(r["type"] == "connects_to" and r["equipment_id"] == owner for r in payload["relationships"]))

    def test_port_catalog_owner_wins_and_conflict_is_preserved(self) -> None:
        payload = build_graph_v1_payload({"nodes": [], "equipment_catalog": [{"id": "A"}, {"id": "B"}], "equipment_ports": [{"id": "p", "equipment_id": "A"}], "edges": [{"id": "e", "source": "a", "target": "b", "source_equipment_id": "B", "source_port_id": "p"}]})
        edge = payload["edges"][0]
        self.assertEqual(edge["source_equipment_id"], "A")
        self.assertEqual(edge["endpoint_conflicts"][0]["observed_equipment_id"], "B")
        self.assertTrue(any(r["type"] == "has_port" and r["source"] == "A" for r in payload["relationships"]))
        relation = next(r for r in payload["relationships"] if r["type"] == "connects_to")
        self.assertEqual(relation["equipment_id"], "A")
        self.assertEqual(relation["semantic_state"], "unresolved")

    def test_empty_catalog_owner_is_repaired_and_target_port_is_connected(self) -> None:
        payload = build_graph_v1_payload({"nodes": [], "equipment_ports": [{"id": "p"}], "edges": [{"id": "e", "source": "a", "target": "b", "target_port_id": "p"}]})
        owner = "unresolved_equipment::p"
        self.assertEqual(payload["equipment_ports"][0]["equipment_id"], owner)
        self.assertTrue(any(r["type"] == "has_port" and r["source"] == owner and r["semantic_state"] == "unresolved" for r in payload["relationships"]))
        self.assertTrue(any(r["type"] == "connects_to" and r["target"] == "p" and r["equipment_id"] == owner and r["endpoint"] == "target" for r in payload["relationships"]))


if __name__ == "__main__":
    unittest.main()
