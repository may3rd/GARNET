import unittest

from garnet.trace_graph_builder import (
    _point_near_axis_segment,
    _split_polyline_at_points,
    build_trace_graph_from_stage11,
)


def _line_edge(trace_id, start, end, *, terminal_type="tee_junction", trace_kind="port"):
    x1, y1 = start
    x2, y2 = end
    direction = "RIGHT" if x2 >= x1 and y1 == y2 else "DOWN" if y2 >= y1 and x1 == x2 else "LEFT" if y1 == y2 else "UP"
    return {
        "trace_id": trace_id,
        "trace_kind": trace_kind,
        "source_obj_id": trace_id,
        "source_obj_type": "branch_candidate" if trace_kind == "branch" else "page_connection",
        "port": {"x": x1, "y": y1, "direction": direction},
        "terminal_type": terminal_type,
        "terminal_obj_id": f"terminal_{trace_id}",
        "terminal_xy": [x2, y2],
        "segments": [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "direction": direction, "length_px": abs(x2 - x1) + abs(y2 - y1)}],
        "polyline": [{"x": x1, "y": y1}, {"x": x2, "y": y2}],
        "attachments": {"line_numbers": [{"id": "line_1"}]},
        "status": "ok",
    }


class TraceGraphBuilderNormalizationTests(unittest.TestCase):
    def test_point_near_axis_segment_horizontal(self) -> None:
        point = {"x": 50, "y": 103}
        start = {"x": 0, "y": 100}
        end = {"x": 100, "y": 100}

        projected = _point_near_axis_segment(point, start, end, tolerance_px=5)

        self.assertEqual(projected, {"x": 50.0, "y": 100.0})

    def test_point_near_axis_segment_vertical(self) -> None:
        point = {"x": 97, "y": 50}
        start = {"x": 100, "y": 0}
        end = {"x": 100, "y": 100}

        projected = _point_near_axis_segment(point, start, end, tolerance_px=5)

        self.assertEqual(projected, {"x": 100.0, "y": 50.0})

    def test_split_polyline_at_interior_point(self) -> None:
        polyline = [{"x": 0, "y": 100}, {"x": 200, "y": 100}]

        parts = _split_polyline_at_points(polyline, [{"x": 100, "y": 100}], tolerance_px=4)

        self.assertEqual(parts, [[{"x": 0.0, "y": 100.0}, {"x": 100.0, "y": 100.0}], [{"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 100.0}]])

    def test_split_polyline_ignores_endpoint_duplicate(self) -> None:
        polyline = [{"x": 0, "y": 100}, {"x": 200, "y": 100}]

        parts = _split_polyline_at_points(polyline, [{"x": 0, "y": 100}, {"x": 200, "y": 100}], tolerance_px=4)

        self.assertEqual(parts, [[{"x": 0.0, "y": 100.0}, {"x": 200.0, "y": 100.0}]])

    def test_branch_start_on_main_trace_merges_into_junction_and_splits_main_edge(self) -> None:
        payload = {
            "image_id": "synthetic.png",
            "trace_source": "stage11_trace_associations",
            "trace_edges": [
                _line_edge("obj_main", (0, 100), (200, 100), terminal_type="tee_junction"),
                _line_edge("branch_000001", (100, 100), (100, 200), terminal_type="equipment", trace_kind="branch"),
            ],
        }

        result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
        graph = result["graph_payload"]

        branch_start_nodes = [node for node in graph["nodes"] if node["type"] == "branch_start"]
        self.assertEqual(branch_start_nodes, [])
        junction_nodes = [node for node in graph["nodes"] if node["type"] == "tee_junction"]
        source_junction = next(node for node in junction_nodes if node["position"] == {"x": 100.0, "y": 100.0})
        junction_id = source_junction["id"]
        degree = sum(1 for edge in graph["edges"] if edge["source"] == junction_id or edge["target"] == junction_id)
        self.assertGreaterEqual(degree, 3)
        self.assertEqual(len([edge for edge in graph["edges"] if edge["trace_id"].startswith("obj_main")]), 2)

    def test_duplicate_reverse_physical_path_collapses(self) -> None:
        forward = _line_edge("branch_000001", (0, 0), (100, 0), terminal_type="tee_junction", trace_kind="branch")
        reverse = _line_edge("branch_000002", (100, 0), (0, 0), terminal_type="tee_junction", trace_kind="branch")
        payload = {"image_id": "synthetic.png", "trace_edges": [forward, reverse]}

        result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
        graph = result["graph_payload"]

        self.assertEqual(len(graph["edges"]), 1)
        edge = graph["edges"][0]
        self.assertEqual(edge["merged_trace_ids"], ["branch_000001", "branch_000002"])
        self.assertEqual(result["summary"]["normalization_duplicate_edge_count"], 1)
        self.assertIn("duplicate_trace_collapsed", result["review_queue_summary"]["issue_counts"])

    def test_duplicate_branch_continuation_downgrades_synthetic_tee(self) -> None:
        main = _line_edge("obj_main", (0, 0), (200, 0), terminal_type="equipment")
        branch = _line_edge("branch_000001", (100, 0), (200, 0), terminal_type="equipment", trace_kind="branch")
        payload = {"image_id": "synthetic.png", "trace_edges": [main, branch]}

        result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
        graph = result["graph_payload"]

        split_nodes = [node for node in graph["nodes"] if node["position"] == {"x": 100.0, "y": 0.0}]
        self.assertEqual(len(split_nodes), 1)
        self.assertEqual(split_nodes[0]["type"], "junction")
        self.assertEqual(len(graph["edges"]), 2)

    def test_terminal_tee_without_object_id_merges_by_position(self) -> None:
        edge_a = _line_edge("obj_a", (0, 0), (100, 0), terminal_type="tee_junction")
        edge_b = _line_edge("obj_b", (100, 100), (100, 0), terminal_type="tee_junction")
        edge_a["terminal_obj_id"] = None
        edge_b["terminal_obj_id"] = None
        payload = {"image_id": "synthetic.png", "trace_edges": [edge_a, edge_b]}

        result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
        junction_nodes = [node for node in result["graph_payload"]["nodes"] if node["type"] == "tee_junction"]

        self.assertEqual(len(junction_nodes), 1)
        self.assertEqual(junction_nodes[0]["id"], "junction::xy::100::0")

    def test_equipment_port_node_is_distinct_from_equipment_terminal_node(self) -> None:
        edge = _line_edge("equip_1", (0, 0), (100, 0), terminal_type="equipment")
        edge["source_obj_id"] = "equip_1"
        edge["source_obj_type"] = "vessel"
        edge["port_index"] = 1
        edge["terminal_obj_id"] = "equip_1"
        payload = {"image_id": "synthetic.png", "trace_edges": [edge]}

        result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
        nodes = result["graph_payload"]["nodes"]
        node_ids = {node["id"] for node in nodes}

        self.assertIn("equipment::equip_1:port_01", node_ids)
        self.assertIn("equipment::equip_1", node_ids)

    def test_reviewed_line_number_propagates_to_connected_component(self) -> None:
        edge_a = _line_edge("obj_a", (0, 0), (200, 0), terminal_type="equipment")
        edge_b = _line_edge("branch_000001", (100, 0), (100, 100), terminal_type="equipment", trace_kind="branch")
        edge_a["attachments"]["line_numbers"] = [
            {
                "id": "line_1",
                "text": '3"_PL-26-003008-NZA1_Nl',
                "normalized_text": '3"-PL-26-003008-NZA1-NL',
                "review_state": "accepted",
            }
        ]
        edge_b["attachments"]["line_numbers"] = []
        payload = {"image_id": "synthetic.png", "trace_edges": [edge_a, edge_b]}

        result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
        edges = {edge["trace_id"]: edge for edge in result["graph_payload"]["edges"]}

        self.assertEqual(edges["obj_a::part_001"]["line_number_assignment_state"], "direct")
        self.assertEqual(edges["obj_a::part_001"]["direct_line_number_ids"], ["line_1"])
        self.assertEqual(edges["branch_000001"]["line_number_assignment_state"], "inferred")
        self.assertEqual(edges["branch_000001"]["inferred_line_number_ids"], ["line_1"])
        self.assertEqual(edges["branch_000001"]["effective_line_number_ids"], ["line_1"])
        self.assertEqual(edges["obj_a::part_001"]["direct_line_numbers"][0]["display_text"], '3"_PL-26-003008-NZA1_Nl')
        self.assertEqual(edges["branch_000001"]["effective_line_numbers"][0]["normalized_text"], '3"-PL-26-003008-NZA1-NL')

    def test_conflicting_reviewed_line_numbers_mark_component_conflict(self) -> None:
        edge_a = _line_edge("obj_a", (0, 0), (200, 0), terminal_type="equipment")
        edge_b = _line_edge("branch_000001", (100, 0), (100, 100), terminal_type="equipment", trace_kind="branch")
        edge_a["attachments"]["line_numbers"] = [{"id": "line_1", "review_state": "accepted"}]
        edge_b["attachments"]["line_numbers"] = [{"id": "line_2", "review_state": "accepted"}]
        payload = {"image_id": "synthetic.png", "trace_edges": [edge_a, edge_b]}

        result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
        edges = result["graph_payload"]["edges"]

        self.assertEqual({edge["line_number_assignment_state"] for edge in edges}, {"conflict"})
        self.assertEqual({tuple(edge["effective_line_number_ids"]) for edge in edges}, {("line_1", "line_2")})
        self.assertIn("line_number_conflict", result["review_queue_summary"]["issue_counts"])

    def test_missing_reviewed_line_number_remains_missing(self) -> None:
        edge = _line_edge("obj_a", (0, 0), (100, 0), terminal_type="equipment")
        edge["attachments"]["line_numbers"] = []
        payload = {"image_id": "synthetic.png", "trace_edges": [edge]}

        result = build_trace_graph_from_stage11(payload, image_id="synthetic.png")
        graph_edge = result["graph_payload"]["edges"][0]

        self.assertEqual(graph_edge["line_number_assignment_state"], "missing")
        self.assertEqual(graph_edge["effective_line_number_ids"], [])


if __name__ == "__main__":
    unittest.main()
