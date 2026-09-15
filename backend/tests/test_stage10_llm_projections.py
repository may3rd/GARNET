import json
import unittest

from garnet.stage10_llm_projections import (
    build_hazop_context,
    build_llm_projections,
    build_process_description_context,
)


def _graph() -> dict:
    return {
        "schema_version": "graph_v1",
        "document": {"doc_id": "sheet-A"},
        "release_gate": {"release_ready": True, "blocking_reasons": []},
        "nodes": [
            {"id": "equipment::P-101", "type": "equipment", "position": {"x": 10, "y": 20}, "confidence": 0.91, "review_state": "accepted"},
            {"id": "junction::J1", "type": "crossing", "position": {"x": 30, "y": 20}},
        ],
        "edges": [{
            "id": "edge-2",
            "source": "equipment::P-101",
            "target": "junction::J1",
            "geometry": {"polyline": [{"x": 10, "y": 20}, {"x": 30, "y": 20}]},
            "trace_length_px": 20,
            "canonical_line_ids": ["line::sheet-A::3-PL-101"],
            "line_number_ids": ["ocr-line-1"],
            "source_equipment_id": "equipment::P-101",
            "flow_direction_state": "forward",
            "flow_direction_confidence": 0.8,
            "flow_direction_evidence": [{"id": "arrow-1", "position": {"x": 18, "y": 20}}],
            "attachments": {
                "inline_objects": [{"source_object_id": "valve-1", "class_name": "gate valve", "projected_xy": [20, 20]}],
                "instrument_tags": [{"source_object_id": "PT-101", "normalized_text": "PT-101"}],
            },
            "provenance": {"source": "reviewed_graph"},
            "review_state": "accepted",
        }],
        "relationships": [{"id": "rel-1", "type": "has_inline_object", "source": "edge-2", "target": "inline-1", "semantic_state": "observed"}],
    }


class Stage10LLMProjectionTests(unittest.TestCase):
    def test_process_context_keeps_routes_and_evidence_structured(self) -> None:
        result = build_process_description_context(
            graph_payload=_graph(),
            boundary_payload={"boundaries": [{"id": "boundary::unit-1", "member_ids": ["edge-2"], "review_state": "unresolved"}]},
            test_package_payload={"test_packages": [{"id": "tp-1", "line_ids": ["line::sheet-A::3-PL-101"]}]},
        )
        self.assertEqual(result["context_type"], "process_description")
        self.assertEqual(result["graph"]["routes"][0]["id"], "edge-2")
        self.assertEqual(result["graph"]["routes"][0]["line_ids"], ["line::sheet-A::3-PL-101", "ocr-line-1"])
        self.assertEqual(result["graph"]["routes"][0]["flow"]["state"], "forward")
        self.assertEqual(result["graph"]["routes"][0]["pixel_evidence"][0]["polyline"], [{"x": 10, "y": 20}, {"x": 30, "y": 20}])
        self.assertEqual(result["boundaries"][0]["member_ids"], ["edge-2"])
        self.assertEqual(result["test_packages"][0]["line_ids"], ["line::sheet-A::3-PL-101"])
        self.assertEqual(result["unresolved_assumptions"], [])
        json.dumps(result, allow_nan=False)

    def test_hazop_is_candidate_scaffold_without_engineering_claims(self) -> None:
        result = build_hazop_context(graph_payload=_graph())
        self.assertEqual(result["context_type"], "hazop_input_scaffold")
        self.assertEqual(result["candidate_segments"][0]["id"], "edge-2")
        self.assertTrue(all(item["status"] == "candidate" for item in result["deviation_dimensions"]))
        self.assertTrue(all(item["review_state"] == "unresolved" for item in result["deviation_dimensions"]))
        self.assertEqual(result["generation_policy"]["causes"], "not_generated")
        self.assertEqual(result["generation_policy"]["consequences"], "not_generated")
        self.assertEqual(result["generation_policy"]["safeguards"], "not_generated")
        self.assertEqual(result["unresolved_assumptions"], [])

    def test_accepts_full_engineering_view_and_propagates_candidate_uncertainty(self) -> None:
        engineering = {
            "process_boundaries": {
                "candidates": [{
                    "id": "boundary::sheet-A::edge-2",
                    "kind": "process",
                    "state": "candidate",
                    "review_state": "derived_from_released_graph",
                    "member_edge_ids": ["edge-2"],
                    "routes": [{"edge_id": "edge-2", "polyline": [[10, 20], [30, 20]]}],
                    "uncertainty": [{"type": "uncertain_flow_direction", "edge_id": "edge-2"}],
                }]
            },
            "test_packages": {
                "candidates": [{
                    "id": "test_package::sheet-A::line-1",
                    "state": "candidate",
                    "line_ids": ["line-1"],
                    "edge_ids": ["edge-2"],
                    "uncertainty": [{"type": "missing_route_geometry", "edge_id": "edge-2"}],
                }]
            },
        }
        result = build_process_description_context(
            graph_payload=_graph(),
            boundary_payload=engineering,
            test_package_payload=engineering,
        )
        self.assertEqual(result["boundaries"][0]["member_edge_ids"], ["edge-2"])
        self.assertEqual(result["test_packages"][0]["edge_ids"], ["edge-2"])
        self.assertEqual(result["boundaries"][0]["state"], "candidate")
        self.assertEqual(
            {(item["kind"], item["candidate_id"]) for item in result["unresolved_questions"]
             if item.get("candidate_id")},
            {("uncertain_flow_direction", "boundary::sheet-A::edge-2"), ("missing_route_geometry", "test_package::sheet-A::line-1")},
        )

    def test_empty_or_invalid_route_is_not_pixel_route_evidence(self) -> None:
        graph = _graph()
        graph["edges"][0]["polyline"] = []
        result = build_process_description_context(graph_payload=graph)
        self.assertEqual(result["graph"]["routes"][0]["pixel_evidence"], [])
        self.assertIn(
            {"id": "gap::pixel_route::edge-2", "kind": "pixel_route", "status": "unresolved", "route_id": "edge-2"},
            result["unresolved_questions"],
        )

    def test_blocked_or_missing_gate_is_explicitly_unreleased(self) -> None:
        graph = _graph()
        graph["release_gate"] = {"release_ready": False, "blocking_reasons": ["pending_review"]}
        result = build_llm_projections(graph_payload=graph)
        self.assertFalse(result["process_description"]["release_ready"])
        self.assertEqual(result["process_description"]["blocked_reasons"], ["pending_review"])
        self.assertEqual(result["hazop"]["state"], "blocked")
        self.assertEqual(result["process_description"]["graph"]["entities"], [])
        self.assertEqual(result["process_description"]["graph"]["routes"], [])
        self.assertEqual(result["process_description"]["graph"]["relationships"], [])
        self.assertEqual(result["process_description"]["boundaries"], [])
        self.assertEqual(result["process_description"]["test_packages"], [])
        self.assertEqual(result["hazop"]["candidate_nodes"], [])
        self.assertEqual(result["hazop"]["candidate_segments"], [])
        self.assertEqual(result["hazop"]["deviation_dimensions"], [])
        self.assertIn({
            "id": "gap::release_gate",
            "kind": "release_gate",
            "status": "unresolved",
            "reasons": ["pending_review"],
        }, result["process_description"]["unresolved_questions"])

        missing = build_llm_projections(graph_payload={"schema_version": "graph_v1", "nodes": [{"id": "source-fact"}]})
        self.assertFalse(missing["process_description"]["release_ready"])
        self.assertEqual(missing["process_description"]["blocked_reasons"], ["missing_release_gate"])
        self.assertEqual(missing["process_description"]["graph"]["entities"], [])
        self.assertEqual(missing["hazop"]["candidate_segments"], [])

    def test_combined_graph_and_input_order_produce_deterministic_output(self) -> None:
        graph = _graph()
        graph["combined_graph"] = {
            "schema_version": "graph_v2_combined",
            "sheets": ["sheet-B", "sheet-A"],
            "nodes": [{"id": "node::sheet-A::equipment::P-101", "type": "equipment", "sheet": "sheet-A"}],
            "edges": [{"id": "edge::sheet-A::edge-2", "source": "node::sheet-A::equipment::P-101", "target": "node::sheet-A::junction::J1", "polyline": [{"x": 0, "y": 0}]}],
            "relationships": [],
        }
        payload_a = build_llm_projections(graph_payload=graph, boundary_payload={"boundaries": [{"id": "b"}, {"id": "a"}]})
        payload_b = build_llm_projections(graph_payload=graph, boundary_payload={"boundaries": [{"id": "a"}, {"id": "b"}]})
        self.assertEqual(payload_a, payload_b)
        self.assertEqual(payload_a["process_description"]["graph"]["drawing_ids"], ["sheet-A", "sheet-B"])
        self.assertEqual(json.dumps(payload_a), json.dumps(payload_b))
        json.dumps(payload_a, allow_nan=False)

    def test_nested_mapping_order_is_stable_without_sort_keys(self) -> None:
        graph_a = _graph()
        graph_a["edges"][0]["flow_direction_review_state"] = float("nan")
        graph_a["edges"][0]["provenance"] = {"z": {"b": 2, "a": 1}, "a": 0}
        graph_b = _graph()
        graph_b["edges"][0]["flow_direction_review_state"] = float("nan")
        graph_b["edges"][0]["provenance"] = {"a": 0, "z": {"a": 1, "b": 2}}
        payload_a = build_process_description_context(graph_payload=graph_a)
        payload_b = build_process_description_context(graph_payload=graph_b)
        self.assertEqual(json.dumps(payload_a), json.dumps(payload_b))
        self.assertIsNone(payload_a["graph"]["routes"][0]["flow"]["review_state"])
        json.dumps(payload_a, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
