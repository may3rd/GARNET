import unittest

from garnet.pipe_terminals import classify_pipe_edge_terminals


class PipeTerminalTests(unittest.TestCase):
    def test_classify_pipe_edge_terminals_accepts_junction_to_junction_edges(self) -> None:
        result = classify_pipe_edge_terminals(
            edges=[{"id": "edge_0", "source": "junction_0", "target": "junction_1"}],
            node_clusters=[
                {"id": "junction_0", "kind": "junction", "centroid": {"x": 10.0, "y": 20.0}},
                {"id": "junction_1", "kind": "junction", "centroid": {"x": 40.0, "y": 20.0}},
            ],
            object_regions=[],
            equipment_terminal_classes=("pump",),
            connection_terminal_classes=("connection", "page connection", "utility connection"),
            inline_passthrough_classes=("valve", "reducer"),
            match_distance_px=20.0,
        )

        edge = result["edge_terminals"][0]
        self.assertEqual(edge["terminal_status"], "validated")
        self.assertTrue(edge["is_internal_junction_edge"])
        self.assertFalse(edge["provisional_due_to_unresolved_terminal"])

    def test_classify_pipe_edge_terminals_accepts_equipment_to_connection_edges(self) -> None:
        result = classify_pipe_edge_terminals(
            edges=[{"id": "edge_0", "source": "endpoint_0", "target": "endpoint_1"}],
            node_clusters=[
                {"id": "endpoint_0", "kind": "endpoint", "centroid": {"x": 10.0, "y": 20.0}},
                {"id": "endpoint_1", "kind": "endpoint", "centroid": {"x": 90.0, "y": 20.0}},
            ],
            object_regions=[
                {
                    "id": "obj_1",
                    "class_name": "pump",
                    "bbox": {"x_min": 0, "y_min": 10, "x_max": 20, "y_max": 30},
                },
                {
                    "id": "obj_2",
                    "class_name": "page connection",
                    "bbox": {"x_min": 80, "y_min": 10, "x_max": 100, "y_max": 30},
                },
            ],
            equipment_terminal_classes=("pump",),
            connection_terminal_classes=("connection", "page connection", "utility connection"),
            inline_passthrough_classes=("valve", "reducer"),
            match_distance_px=20.0,
        )

        edge = result["edge_terminals"][0]
        self.assertEqual(edge["source_terminal"]["terminal_role"], "equipment_terminal")
        self.assertEqual(edge["destination_terminal"]["terminal_role"], "connection_terminal")
        self.assertEqual(edge["terminal_status"], "validated")

    def test_classify_pipe_edge_terminals_keeps_valve_only_endpoint_provisional(self) -> None:
        result = classify_pipe_edge_terminals(
            edges=[{"id": "edge_0", "source": "endpoint_0", "target": "endpoint_1"}],
            node_clusters=[
                {"id": "endpoint_0", "kind": "endpoint", "centroid": {"x": 10.0, "y": 20.0}},
                {"id": "endpoint_1", "kind": "endpoint", "centroid": {"x": 90.0, "y": 20.0}},
            ],
            object_regions=[
                {
                    "id": "obj_1",
                    "class_name": "valve",
                    "bbox": {"x_min": 0, "y_min": 10, "x_max": 20, "y_max": 30},
                },
                {
                    "id": "obj_2",
                    "class_name": "pump",
                    "bbox": {"x_min": 80, "y_min": 10, "x_max": 100, "y_max": 30},
                },
            ],
            equipment_terminal_classes=("pump",),
            connection_terminal_classes=("connection", "page connection", "utility connection"),
            inline_passthrough_classes=("valve", "reducer"),
            match_distance_px=20.0,
        )

        edge = result["edge_terminals"][0]
        self.assertEqual(edge["source_terminal"]["terminal_role"], "inline_passthrough")
        self.assertEqual(edge["destination_terminal"]["terminal_role"], "equipment_terminal")
        self.assertEqual(edge["terminal_status"], "provisional")
        self.assertTrue(edge["provisional_due_to_unresolved_terminal"])

    def test_classify_pipe_edge_terminals_prefers_equipment_over_inline_passthrough(self) -> None:
        result = classify_pipe_edge_terminals(
            edges=[{"id": "edge_0", "source": "endpoint_0", "target": "endpoint_1"}],
            node_clusters=[
                {"id": "endpoint_0", "kind": "endpoint", "centroid": {"x": 10.0, "y": 20.0}},
                {"id": "endpoint_1", "kind": "endpoint", "centroid": {"x": 90.0, "y": 20.0}},
            ],
            object_regions=[
                {
                    "id": "obj_1",
                    "class_name": "valve",
                    "bbox": {"x_min": 0, "y_min": 10, "x_max": 20, "y_max": 30},
                },
                {
                    "id": "obj_2",
                    "class_name": "pump",
                    "bbox": {"x_min": 2, "y_min": 12, "x_max": 22, "y_max": 32},
                },
                {
                    "id": "obj_3",
                    "class_name": "page connection",
                    "bbox": {"x_min": 80, "y_min": 10, "x_max": 100, "y_max": 30},
                },
            ],
            equipment_terminal_classes=("pump",),
            connection_terminal_classes=("connection", "page connection", "utility connection"),
            inline_passthrough_classes=("valve", "reducer"),
            match_distance_px=20.0,
        )

        edge = result["edge_terminals"][0]
        self.assertEqual(edge["source_terminal"]["terminal_role"], "equipment_terminal")
        self.assertEqual(edge["terminal_status"], "validated")


if __name__ == "__main__":
    unittest.main()
