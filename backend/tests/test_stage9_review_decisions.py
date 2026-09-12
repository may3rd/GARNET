import unittest

from garnet.stage9_review_decisions import apply_stage9_review_decisions


class Stage9ReviewDecisionTests(unittest.TestCase):
    def test_apply_stage9_review_decisions_identity_pass(self) -> None:
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [{"id": "n1", "type": "tee_junction", "position": {"x": 10, "y": 20}}],
            "edges": [{"id": "e1", "source": "n1", "target": "n2", "polyline": []}],
        }
        review_items_payload = {
            "image_id": "synthetic.png",
            "review_items": [
                {
                    "id": "stage8::qa::tee_degree_mismatch::n1",
                    "category": "tee_degree_mismatch",
                    "priority": 10,
                    "status": "open",
                    "geometry": {"x": 10, "y": 20},
                }
            ],
        }

        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload=graph_payload,
            review_items_payload=review_items_payload,
            decisions_payload={"decisions": []},
        )

        self.assertEqual(result["corrected_graph_payload"]["nodes"], graph_payload["nodes"])
        self.assertEqual(result["corrected_graph_payload"]["edges"], graph_payload["edges"])
        self.assertEqual(result["summary"]["correction_count"], 0)
        self.assertEqual(result["summary"]["assumed_resolved_count"], 1)
        self.assertEqual(result["review_resolution_payload"]["resolutions"][0]["resolution_state"], "accepted_by_assumption")

    def test_apply_stage9_review_decisions_accepts_explicit_noop_decision(self) -> None:
        review_item_id = "stage8::qa::tee_degree_mismatch::n1"
        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload={"image_id": "synthetic.png", "nodes": [], "edges": []},
            review_items_payload={
                "image_id": "synthetic.png",
                "review_items": [
                    {
                        "id": review_item_id,
                        "category": "tee_degree_mismatch",
                        "priority": 10,
                        "status": "open",
                    }
                ],
            },
            decisions_payload={
                "decisions": [
                    {
                        "review_item_id": review_item_id,
                        "decision": "accept_as_is",
                        "reviewer": "human_assumed",
                        "note": "Known valid junction geometry.",
                    }
                ]
            },
        )

        resolution = result["review_resolution_payload"]["resolutions"][0]
        self.assertEqual(resolution["resolution_state"], "accept_as_is")
        self.assertEqual(resolution["decision_source"], "human_assumed")
        self.assertFalse(resolution["graph_changed"])
        self.assertEqual(result["summary"]["explicit_resolution_count"], 1)
        self.assertEqual(result["summary"]["assumed_resolved_count"], 0)

    def test_apply_stage9_review_decisions_marks_unknown_decision_unsupported(self) -> None:
        review_item_id = "stage8::qa::tee_degree_mismatch::n1"
        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload={"image_id": "synthetic.png", "nodes": [], "edges": []},
            review_items_payload={"review_items": [{"id": review_item_id, "category": "tee_degree_mismatch"}]},
            decisions_payload={"decisions": [{"review_item_id": review_item_id, "decision": "merge_nodes"}]},
        )

        resolution = result["review_resolution_payload"]["resolutions"][0]
        self.assertEqual(resolution["resolution_state"], "unsupported_decision")
        self.assertFalse(resolution["graph_changed"])
        self.assertEqual(result["summary"]["unsupported_decision_count"], 1)

    def test_apply_stage9_review_decisions_sets_line_number_on_selected_edges(self) -> None:
        review_item_id = "stage8::review::line_number_conflict::component_00001"
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [],
            "edges": [
                {"id": "e1", "effective_line_number_ids": ["line_old_a"]},
                {"id": "e2", "effective_line_number_ids": ["line_old_b"]},
                {"id": "e3", "effective_line_number_ids": ["line_unchanged"]},
            ],
        }

        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload=graph_payload,
            review_items_payload={
                "review_items": [
                    {
                        "id": review_item_id,
                        "category": "line_number_conflict",
                    }
                ]
            },
            decisions_payload={
                "decisions": [
                    {
                        "review_item_id": review_item_id,
                        "decision": "set_line_number",
                        "line_number_id": "line_123",
                        "edge_ids": ["e1", "e2"],
                        "reviewer": "human_assumed",
                    }
                ]
            },
        )

        edges = {edge["id"]: edge for edge in result["corrected_graph_payload"]["edges"]}
        self.assertEqual(edges["e1"]["effective_line_number_ids"], ["line_123"])
        self.assertEqual(edges["e2"]["effective_line_number_ids"], ["line_123"])
        self.assertEqual(edges["e1"]["line_number_review_state"], "human_reviewed")
        self.assertEqual(edges["e2"]["line_number_review_state"], "human_reviewed")
        self.assertEqual(edges["e3"]["effective_line_number_ids"], ["line_unchanged"])
        self.assertEqual(result["summary"]["correction_count"], 1)
        self.assertEqual(result["correction_audit_payload"]["corrections"][0]["affected_edge_ids"], ["e1", "e2"])
        self.assertTrue(result["review_resolution_payload"]["resolutions"][0]["graph_changed"])

    def test_apply_stage9_review_decisions_warns_for_missing_line_number_edge(self) -> None:
        review_item_id = "stage8::review::line_number_conflict::component_00001"
        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload={"image_id": "synthetic.png", "nodes": [], "edges": [{"id": "e1"}]},
            review_items_payload={"review_items": [{"id": review_item_id, "category": "line_number_conflict"}]},
            decisions_payload={
                "decisions": [
                    {
                        "review_item_id": review_item_id,
                        "decision": "set_line_number",
                        "line_number_id": "line_123",
                        "edge_ids": ["e1", "missing_edge"],
                    }
                ]
            },
        )

        self.assertEqual(result["summary"]["correction_count"], 1)
        self.assertEqual(result["summary"]["warning_count"], 1)
        self.assertEqual(result["correction_audit_payload"]["warnings"][0]["warning"], "missing_edge")
        self.assertEqual(result["correction_audit_payload"]["warnings"][0]["edge_id"], "missing_edge")

    def test_apply_stage9_review_decisions_sets_flow_direction_and_preserves_evidence(self) -> None:
        review_item_id = "stage8::flow_direction::e1"
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [],
            "edges": [
                {
                    "id": "e1",
                    "flow_direction_state": "conflicting",
                    "flow_direction_confidence": 0.42,
                    "flow_direction_evidence": [{"id": "arrow-a", "direction": "right"}],
                    "flow_direction_provenance": [{"source": "stage6", "state": "observed"}],
                }
            ],
        }

        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload=graph_payload,
            review_items_payload={"review_items": [{"id": review_item_id, "category": "flow_direction_conflict"}]},
            decisions_payload={
                "decisions": [
                    {
                        "review_item_id": review_item_id,
                        "decision": "set_flow_direction",
                        "edge_id": "e1",
                        "flow_direction": "reverse",
                        "reviewer": "eng-reviewer",
                    }
                ]
            },
        )

        edge = result["corrected_graph_payload"]["edges"][0]
        self.assertEqual(edge["flow_direction_state"], "reverse")
        self.assertEqual(edge["flow_direction_confidence"], 1.0)
        self.assertEqual(edge["flow_direction_review_state"], "human_reviewed")
        self.assertEqual(edge["flow_direction_evidence"], graph_payload["edges"][0]["flow_direction_evidence"])
        self.assertEqual(edge["observed_flow_direction_evidence"], graph_payload["edges"][0]["flow_direction_evidence"])
        self.assertEqual(edge["flow_direction_provenance"][0], graph_payload["edges"][0]["flow_direction_provenance"][0])
        self.assertEqual(result["summary"]["correction_count"], 1)
        self.assertEqual(result["correction_audit_payload"]["corrections"][0]["affected_edge_ids"], ["e1"])

    def test_apply_stage9_review_decisions_warns_and_ignores_invalid_flow_direction(self) -> None:
        review_item_id = "stage8::flow_direction::e1"
        graph_payload = {"edges": [{"id": "e1", "flow_direction_state": "unknown"}]}
        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload=graph_payload,
            review_items_payload={"review_items": [{"id": review_item_id, "category": "flow_direction_unknown"}]},
            decisions_payload={
                "decisions": [
                    {
                        "review_item_id": review_item_id,
                        "decision": "set_flow_direction",
                        "edge_id": "e1",
                        "flow_direction_state": "conflicting",
                    }
                ]
            },
        )

        self.assertEqual(result["corrected_graph_payload"], graph_payload)
        self.assertEqual(result["summary"]["correction_count"], 0)
        self.assertEqual(result["summary"]["warning_count"], 1)
        self.assertEqual(result["correction_audit_payload"]["warnings"][0]["warning"], "invalid_flow_direction")

    def test_apply_stage9_review_decisions_warns_and_ignores_missing_flow_edge(self) -> None:
        review_item_id = "stage8::flow_direction::missing"
        graph_payload = {"edges": [{"id": "e1", "flow_direction_state": "unknown"}]}
        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload=graph_payload,
            review_items_payload={"review_items": [{"id": review_item_id, "category": "flow_direction_unknown"}]},
            decisions_payload={
                "decisions": [
                    {
                        "review_item_id": review_item_id,
                        "decision": "set_flow_direction",
                        "edge_id": "missing-edge",
                        "flow_direction": "forward",
                    }
                ]
            },
        )

        self.assertEqual(result["corrected_graph_payload"], graph_payload)
        self.assertEqual(result["summary"]["correction_count"], 0)
        self.assertEqual(result["summary"]["warning_count"], 1)
        warning = result["correction_audit_payload"]["warnings"][0]
        self.assertEqual(warning["warning"], "missing_edge")
        self.assertEqual(warning["edge_id"], "missing-edge")


if __name__ == "__main__":
    unittest.main()
