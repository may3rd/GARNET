import json
import unittest

from garnet.stage10_engineering_views import (
    build_phase8_engineering_views,
    build_process_boundary_candidates,
    build_test_package_candidates,
)


def _gate(ready=True):
    return {"release_ready": ready, "blocking_reasons": [] if ready else ["pending_review"]}


def _graph():
    return {
        "schema_version": "graph_v1",
        "document": {"doc_id": "P-101"},
        "nodes": [
            {"id": "pump", "type": "equipment", "position": {"x": 0, "y": 0}},
            {"id": "tee", "type": "tee_junction", "position": {"x": 50, "y": 0}},
            {"id": "vessel", "type": "equipment", "position": {"x": 100, "y": 0}},
            {"id": "branch", "type": "equipment", "position": {"x": 50, "y": 50}},
            {"id": "off", "type": "page_connection", "position": {"x": 200, "y": 0}},
        ],
        "edges": [
            {
                "id": "main",
                "source": "pump",
                "target": "vessel",
                "polyline": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
                "effective_line_number_ids": ["L-1"],
                "flow_direction_state": "forward",
                "confidence": 0.92,
                "attachments": {"inline_objects": [{"id": "iv-1", "class_name": "isolation valve", "trace_distance_px": 25}]},
            },
            {
                "id": "bypass",
                "source": "pump",
                "target": "vessel",
                "polyline": [{"x": 0, "y": 0}, {"x": 50, "y": -20}, {"x": 100, "y": 0}],
                "effective_line_number_ids": ["L-1"],
                "flow_direction_state": "forward",
                "confidence": 0.81,
            },
            {
                "id": "branch-route",
                "source": "vessel",
                "target": "branch",
                "polyline": [{"x": 100, "y": 0}, {"x": 50, "y": 50}],
                "effective_line_number_ids": ["L-2"],
                "flow_direction_state": "unknown",
            },
            {
                "id": "off-page",
                "source": "vessel",
                "target": "off",
                "geometry": {"polyline": [[100, 0], [200, 0]]},
                "effective_line_number_ids": ["L-3"],
                "flow_direction_state": "unknown",
                "off_page_connector": {
                    "id": "connector-1",
                    "edge_id": "off-page",
                    "reference_value": "P-202:10",
                    "exit_terminal": "target",
                    "review_state": "accepted",
                },
            },
            {
                "id": "uncertain",
                "source": "off",
                "target": "unknown-terminal",
                "polyline": [{"x": 200, "y": 0}, {"x": 250, "y": 0}],
                "flow_direction_state": "unknown",
            },
        ],
    }


class Phase8EngineeringViewsTests(unittest.TestCase):
    def test_release_gate_blocks_candidates(self):
        result = build_phase8_engineering_views(_graph(), release_gate_payload=_gate(False))
        self.assertFalse(result["release_ready"])
        self.assertEqual(result["process_boundaries"]["candidates"], [])
        self.assertEqual(result["test_packages"]["candidates"], [])
        self.assertEqual(result["blocked_reasons"], ["pending_review"])

    def test_boundary_retains_routes_connectors_and_isolation(self):
        result = build_process_boundary_candidates(_graph(), release_gate_payload=_gate())
        self.assertTrue(result["release_ready"])
        boundary = next(item for item in result["candidates"] if item["kind"] == "system")
        self.assertEqual(boundary["kind"], "system")
        self.assertIn("main", boundary["member_edge_ids"])
        self.assertEqual(boundary["isolation_elements"][0]["id"], "iv-1")
        self.assertEqual(boundary["isolation_elements"][0]["state"], "observed")
        connector = next(item for item in boundary["cut_points"] if item["kind"] == "off_page_connector")
        self.assertEqual(connector["connector_id"], "connector-1")
        route = next(item for item in boundary["routes"] if item["edge_id"] == "off-page")
        self.assertEqual(route["polyline"], [{"x": 100.0, "y": 0.0}, {"x": 200.0, "y": 0.0}])

    def test_packages_preserve_parallel_bypass_and_branch_routes(self):
        result = build_test_package_candidates(_graph(), release_gate_payload=_gate())
        by_id = {item["id"]: item for item in result["candidates"]}
        line_one = by_id["test_package::P-101::L-1"]
        self.assertEqual(line_one["route_kind"], "parallel_or_bypass")
        self.assertEqual(line_one["edge_ids"], ["bypass", "main"])
        self.assertEqual(line_one["equipment_ids"], ["pump", "vessel"])
        self.assertEqual(line_one["routes"][1]["polyline"][1], {"x": 100.0, "y": 0.0})
        line_two = by_id["test_package::P-101::L-2"]
        self.assertEqual(line_two["route_kind"], "branch")
        self.assertEqual(line_two["flow_direction_state"], "unknown")
        self.assertIn({"type": "uncertain_flow_direction", "edge_id": "branch-route", "state": "unknown"}, line_two["uncertainty"])

    def test_same_line_on_disconnected_components_stays_separate(self):
        graph = _graph()
        graph["nodes"].extend([
            {"id": "remote-a", "type": "equipment", "position": {"x": 500, "y": 0}},
            {"id": "remote-b", "type": "equipment", "position": {"x": 550, "y": 0}},
        ])
        graph["edges"].append({
            "id": "remote-line",
            "source": "remote-a",
            "target": "remote-b",
            "polyline": [{"x": 500, "y": 0}, {"x": 550, "y": 0}],
            "effective_line_number_ids": ["L-1"],
        })
        result = build_test_package_candidates(graph, release_gate_payload=_gate())
        line_one = [item for item in result["candidates"] if "L-1" in item["line_ids"]]
        self.assertEqual(len(line_one), 2)
        self.assertEqual(
            {tuple(item["edge_ids"]) for item in line_one},
            {("bypass", "main"), ("remote-line",)},
        )

    def test_embedded_blocked_gate_cannot_be_overridden_by_ready_argument(self):
        graph = _graph()
        graph["release_gate"] = _gate(False)
        result = build_phase8_engineering_views(graph, release_gate_payload=_gate(True))
        self.assertFalse(result["release_ready"])
        self.assertEqual(result["blocked_reasons"], ["pending_review"])

    def test_missing_line_and_direction_remain_uncertain(self):
        result = build_test_package_candidates(_graph(), release_gate_payload=_gate())
        uncertain = next(item for item in result["candidates"] if item["id"].endswith("edge::uncertain"))
        self.assertEqual(uncertain["line_assignment_state"], "missing")
        self.assertEqual(uncertain["line_ids"], [])
        self.assertEqual(uncertain["flow_direction_state"], "unknown")
        self.assertEqual(
            [(item["type"], item["edge_id"]) for item in uncertain["uncertainty"]],
            [("missing_line_number", "uncertain"), ("uncertain_flow_direction", "uncertain")],
        )
        self.assertEqual(uncertain["state"], "candidate")

    def test_empty_route_geometry_is_reported_as_missing_evidence(self):
        graph = _graph()
        graph["edges"][0]["polyline"] = []
        result = build_test_package_candidates(graph, release_gate_payload=_gate())
        package = next(item for item in result["candidates"] if "main" in item["edge_ids"])
        self.assertIn({"type": "missing_route_geometry", "edge_id": "main"}, package["uncertainty"])

    def test_output_is_deterministic_and_json_safe(self):
        graph_a = _graph()
        graph_a["edges"][0]["attachments"]["inline_objects"][0]["trace_distance_px"] = float("inf")
        graph_a["edges"][0]["effective_line_numbers"] = [{
            "id": "line-occurrence",
            "confidence": float("nan"),
            "evidence": {"z": 1, "a": 2},
        }]
        graph_b = _graph()
        graph_b["edges"][0]["attachments"]["inline_objects"][0]["trace_distance_px"] = float("inf")
        graph_b["edges"][0]["effective_line_numbers"] = [{
            "evidence": {"a": 2, "z": 1},
            "confidence": float("nan"),
            "id": "line-occurrence",
        }]
        graph_b["edges"] = list(reversed(graph_b["edges"]))
        first = build_phase8_engineering_views(graph_a, release_gate_payload=_gate())
        second = build_phase8_engineering_views(graph_b, release_gate_payload=_gate())
        self.assertEqual(json.dumps(first), json.dumps(second))
        isolation = first["process_boundaries"]["candidates"][0]["isolation_elements"][0]
        self.assertIsNone(isolation["route_position_px"])
        json.dumps(first, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
