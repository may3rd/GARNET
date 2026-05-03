import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
