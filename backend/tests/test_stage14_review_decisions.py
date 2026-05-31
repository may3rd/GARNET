import unittest

from garnet.stage14_review_decisions import apply_stage14_review_decisions


class Stage14ReviewDecisionTests(unittest.TestCase):
    def test_apply_stage14_review_decisions_identity_pass(self) -> None:
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [{"id": "n1", "type": "tee_junction", "position": {"x": 10, "y": 20}}],
            "edges": [{"id": "e1", "source": "n1", "target": "n2", "polyline": []}],
        }
        review_items_payload = {
            "image_id": "synthetic.png",
            "review_items": [
                {
                    "id": "stage13::qa::tee_degree_mismatch::n1",
                    "category": "tee_degree_mismatch",
                    "priority": 10,
                    "status": "open",
                    "geometry": {"x": 10, "y": 20},
                }
            ],
        }

        result = apply_stage14_review_decisions(
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

    def test_apply_stage14_review_decisions_accepts_explicit_noop_decision(self) -> None:
        review_item_id = "stage13::qa::tee_degree_mismatch::n1"
        result = apply_stage14_review_decisions(
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

    def test_apply_stage14_review_decisions_marks_unknown_decision_unsupported(self) -> None:
        review_item_id = "stage13::qa::tee_degree_mismatch::n1"
        result = apply_stage14_review_decisions(
            image_id="synthetic.png",
            graph_payload={"image_id": "synthetic.png", "nodes": [], "edges": []},
            review_items_payload={"review_items": [{"id": review_item_id, "category": "tee_degree_mismatch"}]},
            decisions_payload={"decisions": [{"review_item_id": review_item_id, "decision": "merge_nodes"}]},
        )

        resolution = result["review_resolution_payload"]["resolutions"][0]
        self.assertEqual(resolution["resolution_state"], "unsupported_decision")
        self.assertFalse(resolution["graph_changed"])
        self.assertEqual(result["summary"]["unsupported_decision_count"], 1)

    def test_apply_stage14_review_decisions_sets_line_number_on_selected_edges(self) -> None:
        review_item_id = "stage13::review::line_number_conflict::component_00001"
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [],
            "edges": [
                {"id": "e1", "effective_line_number_ids": ["line_old_a"]},
                {"id": "e2", "effective_line_number_ids": ["line_old_b"]},
                {"id": "e3", "effective_line_number_ids": ["line_unchanged"]},
            ],
        }

        result = apply_stage14_review_decisions(
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

    def test_apply_stage14_review_decisions_warns_for_missing_line_number_edge(self) -> None:
        review_item_id = "stage13::review::line_number_conflict::component_00001"
        result = apply_stage14_review_decisions(
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


if __name__ == "__main__":
    unittest.main()
