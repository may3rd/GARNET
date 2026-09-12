import unittest

from garnet.trace_graph_builder import build_trace_graph_from_stage11


class Phase4EquipmentPortIdentityTests(unittest.TestCase):
    def test_graph_exposes_drawing_scoped_equipment_and_port_records(self):
        payload = {
            "image_id": "P-101.png",
            "trace_edges": [
                {
                    "trace_id": "pump_1_port_1",
                    "trace_kind": "port",
                    "source_obj_id": "pump_1",
                    "source_obj_type": "pump",
                    "port_index": 1,
                    "port": {"x": 25, "y": 40, "direction": "RIGHT"},
                    "terminal_type": "equipment",
                    "terminal_obj_id": "vessel_1",
                    "terminal_xy": [125, 40],
                    "polyline": [{"x": 25, "y": 40}, {"x": 125, "y": 40}],
                    "segments": [{"x1": 25, "y1": 40, "x2": 125, "y2": 40}],
                    "attachments": {"line_numbers": [{"id": "line_1", "review_state": "accepted"}]},
                }
            ],
        }

        graph = build_trace_graph_from_stage11(payload, image_id="P-101.png")["graph_payload"]

        pump_id = "equipment::P-101.png::pump_1"
        pump_port_id = f"{pump_id}::port::01"
        vessel_id = "equipment::P-101.png::vessel_1"
        self.assertEqual({item["id"] for item in graph["equipment"]}, {pump_id, vessel_id})
        pump_port = next(item for item in graph["ports"] if item["id"] == pump_port_id)
        self.assertEqual(pump_port["position"], {"x": 25.0, "y": 40.0})
        edge = graph["edges"][0]
        self.assertEqual(edge["source_equipment_id"], pump_id)
        self.assertEqual(edge["source_port_id"], pump_port_id)
        self.assertEqual(edge["source_port_xy"], {"x": 25.0, "y": 40.0})
        self.assertEqual(edge["terminal_equipment_id"], vessel_id)
        self.assertEqual(edge["terminal_port_id"], f"{vessel_id}::port::xy_125_40")
        self.assertEqual(edge["polyline"], [{"x": 25.0, "y": 40.0}, {"x": 125.0, "y": 40.0}])

    def test_missing_port_index_is_stable_across_trace_ids(self):
        def graph_for(trace_id: str, x: int) -> dict:
            return build_trace_graph_from_stage11(
                {
                    "image_id": "P-101.png",
                    "trace_edges": [{
                        "trace_id": trace_id,
                        "source_obj_id": "pump_1",
                        "source_obj_type": "pump",
                        "port": {"x": x, "y": 40, "direction": "RIGHT"},
                        "terminal_type": "dead_end",
                        "terminal_xy": [x + 10, 40],
                        "polyline": [{"x": x, "y": 40}, {"x": x + 10, "y": 40}],
                        "segments": [{"x1": x, "y1": 40, "x2": x + 10, "y2": 40}],
                        "attachments": {"line_numbers": [{"id": "line_1", "review_state": "accepted"}]},
                    }],
                },
                image_id="P-101.png",
            )["graph_payload"]

        first = graph_for("trace_a", 25)
        same_position = graph_for("trace_b", 25)
        other_position = graph_for("trace_c", 35)
        first_port_ids = {port["id"] for port in first["ports"]}
        same_port_ids = {port["id"] for port in same_position["ports"]}
        other_port_ids = {port["id"] for port in other_position["ports"]}
        self.assertEqual(first_port_ids, same_port_ids)
        self.assertNotEqual(first_port_ids, other_port_ids)

    def test_source_and_terminal_observations_share_position_based_port(self):
        edge = {
            "trace_id": "forward",
            "source_obj_id": "pump_1",
            "source_obj_type": "pump",
            "port_index": 1,
            "port": {"x": 25, "y": 40, "direction": "RIGHT"},
            "terminal_type": "equipment",
            "terminal_obj_id": "pump_1",
            "terminal_xy": [25, 40],
            "polyline": [{"x": 25, "y": 40}, {"x": 35, "y": 40}],
            "segments": [{"x1": 25, "y1": 40, "x2": 35, "y2": 40}],
            "attachments": {"line_numbers": [{"id": "line_1", "review_state": "accepted"}]},
        }
        graph = build_trace_graph_from_stage11({"image_id": "P-101", "trace_edges": [edge]}, image_id="P-101")["graph_payload"]
        port_ids = {port["id"] for port in graph["ports"]}
        self.assertEqual(port_ids, {"equipment::P-101::pump_1::port::01"})
        self.assertEqual(graph["edges"][0]["source_port_id"], graph["edges"][0]["terminal_port_id"])

    def test_terminal_jitter_matches_indexed_port_independent_of_edge_order(self):
        indexed_source = {
            "trace_id": "z_source",
            "source_obj_id": "pump_1",
            "source_obj_type": "pump",
            "port_index": 2,
            "port": {"x": 100, "y": 100, "direction": "RIGHT"},
            "terminal_type": "dead_end",
            "terminal_xy": [120, 100],
            "polyline": [{"x": 100, "y": 100}, {"x": 120, "y": 100}],
            "segments": [{"x1": 100, "y1": 100, "x2": 120, "y2": 100}],
            "attachments": {"line_numbers": [{"id": "line_1", "review_state": "accepted"}]},
        }
        terminal_observation = {
            "trace_id": "a_terminal",
            "source_obj_id": "other",
            "source_obj_type": "pump",
            "port": {"x": 80, "y": 100, "direction": "RIGHT"},
            "terminal_type": "equipment",
            "terminal_obj_id": "pump_1",
            "terminal_xy": [102, 102],
            "polyline": [{"x": 80, "y": 100}, {"x": 102, "y": 102}],
            "segments": [{"x1": 80, "y1": 100, "x2": 102, "y2": 102}],
            "attachments": {"line_numbers": [{"id": "line_2", "review_state": "accepted"}]},
        }
        graph = build_trace_graph_from_stage11(
            {"image_id": "P-101", "trace_edges": [terminal_observation, indexed_source]},
            image_id="P-101",
        )["graph_payload"]
        edges = {edge["trace_id"]: edge for edge in graph["edges"]}
        self.assertEqual(edges["a_terminal"]["terminal_port_id"], "equipment::P-101::pump_1::port::02")
        self.assertEqual(edges["z_source"]["source_port_id"], "equipment::P-101::pump_1::port::02")
        self.assertEqual(edges["a_terminal"]["target"], edges["z_source"]["source"])
        self.assertEqual(edges["a_terminal"]["target"], "equipment::P-101::pump_1::port::02")
        catalog_ids = {item["id"] for item in graph["ports"]}
        pump = next(item for item in graph["equipment"] if item["id"] == "equipment::P-101::pump_1")
        self.assertEqual(pump["ports"], ["equipment::P-101::pump_1::port::02"])
        self.assertTrue(set(pump["ports"]).issubset(catalog_ids))

    def test_excluded_indexed_trace_cannot_seed_valid_port_identity(self):
        excluded = {
            "trace_id": "excluded",
            "status": "skipped_existing_trace",
            "source_obj_id": "pump_1",
            "source_obj_type": "pump",
            "port_index": 3,
            "port": {"x": 100, "y": 100},
            "terminal_type": "dead_end",
            "terminal_xy": [110, 100],
            "segments": [{"x1": 100, "y1": 100, "x2": 110, "y2": 100}],
            "polyline": [{"x": 100, "y": 100}, {"x": 110, "y": 100}],
        }
        valid = {
            "trace_id": "valid",
            "source_obj_id": "pump_1",
            "source_obj_type": "pump",
            "port": {"x": 102, "y": 101},
            "terminal_type": "dead_end",
            "terminal_xy": [120, 101],
            "segments": [{"x1": 102, "y1": 101, "x2": 120, "y2": 101}],
            "polyline": [{"x": 102, "y": 101}, {"x": 120, "y": 101}],
            "attachments": {"line_numbers": [{"id": "line_1", "review_state": "accepted"}]},
        }
        graph = build_trace_graph_from_stage11({"image_id": "P-101", "trace_edges": [excluded, valid]}, image_id="P-101")["graph_payload"]
        self.assertEqual(graph["edges"][0]["source_port_id"], "equipment::P-101::pump_1::port::xy_102_101")
        port_ids = {port["id"] for port in graph["ports"]}
        self.assertEqual(port_ids, {"equipment::P-101::pump_1::port::xy_102_101"})

    def test_nonfinite_equipment_coordinates_are_malformed_without_port_seeding(self):
        for invalid_source, invalid_terminal in ((float("nan"), 100), (float("inf"), 100), (100, float("-inf"))):
            with self.subTest(invalid_source=invalid_source, invalid_terminal=invalid_terminal):
                trace = {
                    "trace_id": "bad",
                    "source_obj_id": "pump_1",
                    "source_obj_type": "pump",
                    "port_index": 1,
                    "port": {"x": invalid_source, "y": 20},
                    "terminal_type": "equipment",
                    "terminal_obj_id": "vessel_1",
                    "terminal_xy": [invalid_terminal, 20],
                    "polyline": [{"x": 0, "y": 20}, {"x": 10, "y": 20}],
                    "segments": [{"x1": 0, "y1": 20, "x2": 10, "y2": 20}],
                }
                graph = build_trace_graph_from_stage11(
                    {"image_id": "P-101", "trace_edges": [trace]}, image_id="P-101"
                )["graph_payload"]
                self.assertEqual(graph["edges"], [])
                self.assertEqual(graph["ports"], [])
                self.assertTrue(any(item["issue_type"] == "malformed_trace_geometry" for item in graph["review_queue"]))

    def test_nonfinite_interior_route_coordinate_invalidates_whole_trace(self):
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                trace = {
                    "trace_id": "bad_interior",
                    "source_obj_id": "pump_1",
                    "source_obj_type": "pump",
                    "port_index": 1,
                    "port": {"x": 0, "y": 20},
                    "terminal_type": "dead_end",
                    "terminal_xy": [20, 20],
                    "polyline": [{"x": 0, "y": 20}, {"x": invalid, "y": 20}, {"x": 20, "y": 20}],
                    "segments": [{"x1": 0, "y1": 20, "x2": 20, "y2": 20}],
                }
                graph = build_trace_graph_from_stage11(
                    {"image_id": "P-101", "trace_edges": [trace]}, image_id="P-101"
                )["graph_payload"]
                self.assertEqual(graph["edges"], [])
                self.assertEqual(graph["ports"], [])
                self.assertEqual(graph["metadata"]["excluded_trace_edges"][0]["status"], "malformed_trace_geometry")


if __name__ == "__main__":
    unittest.main()
