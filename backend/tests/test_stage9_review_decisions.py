import unittest

from garnet.stage9_review_decisions import apply_stage9_review_decisions


class Stage9ReviewDecisionTests(unittest.TestCase):
    def _topology_call(self, decision, graph=None, category="topology", blocking=True):
        item_id = "stage8::topology::1"
        return apply_stage9_review_decisions(
            image_id="synthetic.png", graph_payload=graph or {
                "nodes": [{"id": "n1", "type": "junction"}, {"id": "n2", "type": "junction"}, {"id": "n3", "type": "junction"}],
                "edges": [{"id": "e1", "src": "n1", "dst": "n2", "source": "n1", "target": "n2", "polyline": [[0, 0], [5, 0], [10, 0]]}],
            }, review_items_payload={"review_items": [{"id": item_id, "category": category, "release_blocking": blocking}]}, decisions_payload={"decisions": [{"review_item_id": item_id, **decision}]},
        )

    def test_topology_mutations_success(self):
        merged = self._topology_call({"decision": "merge_nodes", "node_ids": ["n1", "n2"], "keep_node_id": "n1"})
        self.assertEqual(len(merged["corrected_graph_payload"]["nodes"]), 2)
        self.assertEqual(merged["corrected_graph_payload"]["edges"], [])
        reconnected = self._topology_call({"decision": "reconnect_edge", "edge_id": "e1", "endpoint": "target", "node_id": "n3"})
        self.assertEqual(reconnected["corrected_graph_payload"]["edges"][0]["dst"], "n3")
        split = self._topology_call({"decision": "split_edge", "edge_id": "e1", "new_node_id": "mid", "split_index": 1})
        children = split["corrected_graph_payload"]["edges"]
        self.assertEqual([edge["polyline"] for edge in children], [[[0, 0], [5, 0]], [[5, 0], [10, 0]]])
        self.assertEqual(split["corrected_graph_payload"]["nodes"][-1]["position"], {"x": 5, "y": 0})
        deleted = self._topology_call({"decision": "delete_edge", "edge_id": "e1"})
        self.assertEqual(deleted["corrected_graph_payload"]["edges"], [])
        typed = self._topology_call({"decision": "set_node_type", "node_id": "n1", "node_type": "equipment"})
        self.assertEqual(typed["corrected_graph_payload"]["nodes"][0]["type"], "equipment")

    def test_invalid_topology_mutations_are_atomic(self):
        graph = {"nodes": [{"id": "n1"}, {"id": "n2"}], "edges": [{"id": "e1", "src": "n1", "dst": "n2", "polyline": [[0, 0], [10, 0]]}]}
        for decision in ({"decision": "split_edge", "edge_id": "e1", "new_node_id": "mid", "split_index": 0}, {"decision": "reconnect_edge", "edge_id": "e1", "endpoint": "target", "node_id": "n1"}, {"decision": "merge_nodes", "node_ids": ["n1", "missing"]}):
            result = self._topology_call(decision, graph=graph)
            self.assertEqual(result["corrected_graph_payload"], graph)
            self.assertFalse(result["release_gate_payload"]["release_ready"])

    def test_duplicate_and_orphan_decisions_block_release(self):
        base = {"nodes": [], "edges": []}
        result = apply_stage9_review_decisions(image_id="x", graph_payload=base, review_items_payload={"review_items": [{"id": "r", "category": "info", "release_blocking": False}]}, decisions_payload={"decisions": [{"review_item_id": "r", "decision": "accept_as_is"}, {"review_item_id": "r", "decision": "false_positive"}, {"review_item_id": "orphan", "decision": "accept_as_is"}]})
        self.assertFalse(result["release_gate_payload"]["release_ready"])

    def test_split_point_geometry_and_ready_gate(self):
        result = self._topology_call({"decision": "split_edge", "edge_id": "e1", "new_node_id": "mid", "point": {"x": 5, "y": 0}})
        self.assertTrue(result["release_gate_payload"]["release_ready"])
        unresolved_info = self._topology_call({"decision": "defer"}, category="info", blocking=False)
        self.assertTrue(unresolved_info["release_gate_payload"]["release_ready"])

    def test_split_audit_lists_only_changed_or_created_entities(self):
        result = self._topology_call({"decision": "split_edge", "edge_id": "e1", "new_node_id": "mid", "split_index": 1})
        audit = result["correction_audit_payload"]["corrections"][0]
        self.assertEqual(audit["affected_ids"], ["e1", "e1::a", "e1::b", "mid"])
        self.assertNotIn("n1", audit["affected_ids"])
        self.assertNotIn("n2", audit["affected_ids"])
        self.assertIsNotNone(audit["before"]["e1"])
        self.assertIsNone(audit["after"]["e1"])
        self.assertIsNotNone(audit["after"]["mid"])

    def test_reconnect_audit_does_not_fabricate_endpoint_node_changes(self):
        result = self._topology_call({"decision": "reconnect_edge", "edge_id": "e1", "endpoint": "source", "node_id": "n3"}, graph={
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [{"id": "e1", "source": "n1", "target": "n2", "polyline": [[0, 0], [10, 0]]}],
        })
        audit = result["correction_audit_payload"]["corrections"][0]
        self.assertEqual(audit["affected_ids"], ["e1"])
        self.assertNotIn("n1", audit["before"])
        self.assertNotIn("n3", audit["after"])
        self.assertEqual(audit["before"]["e1"]["source"], "n1")
        self.assertEqual(audit["after"]["e1"]["source"], "n3")

    def test_split_with_unlocated_off_page_connector_stays_unresolved(self):
        graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{
                "id": "e1", "source": "n1", "target": "n2",
                "off_page_connector": {"connector_key": "C-1"},
                "polyline": [[0, 0], [5, 0], [10, 0]],
            }],
        }
        result = self._topology_call({"decision": "split_edge", "edge_id": "e1", "new_node_id": "mid", "split_index": 1}, graph=graph, blocking=False)
        self.assertEqual(result["corrected_graph_payload"], graph)
        self.assertEqual(result["review_resolution_payload"]["resolutions"][0]["resolution_state"], "unresolved")
        self.assertFalse(result["release_gate_payload"]["release_ready"])
        self.assertEqual(result["correction_audit_payload"]["warnings"][0]["warning"], "invalid_off_page_connector")

    def test_reconnect_with_malformed_off_page_connector_stays_atomic(self):
        for connector in ({"connector_key": "C-1"}, {"connector_key": "C-1", "exit_terminal": "side"}, "malformed"):
            with self.subTest(connector=connector):
                graph = {
                    "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
                    "edges": [{"id": "e1", "source": "n1", "target": "n2", "off_page_connector": connector, "polyline": [[0, 0], [10, 0]]}],
                }
                result = self._topology_call({"decision": "reconnect_edge", "edge_id": "e1", "endpoint": "target", "node_id": "n3"}, graph=graph, blocking=False)
                self.assertEqual(result["corrected_graph_payload"], graph)
                self.assertEqual(result["review_resolution_payload"]["resolutions"][0]["resolution_state"], "unresolved")
                self.assertEqual(result["correction_audit_payload"]["warnings"][0]["warning"], "invalid_off_page_connector")
                self.assertFalse(result["release_gate_payload"]["release_ready"])

    def test_merge_with_malformed_off_page_connector_stays_atomic(self):
        for connector in ({"connector_key": "C-1"}, {"connector_key": "C-1", "exit_terminal": "side"}, "malformed"):
            with self.subTest(connector=connector):
                graph = {
                    "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
                    "edges": [{"id": "e1", "source": "n2", "target": "n3", "off_page_connector": connector, "polyline": [[0, 0], [10, 0]]}],
                }
                result = self._topology_call({"decision": "merge_nodes", "node_ids": ["n1", "n2"], "keep_node_id": "n1"}, graph=graph, blocking=False)
                self.assertEqual(result["corrected_graph_payload"], graph)
                self.assertEqual(result["review_resolution_payload"]["resolutions"][0]["resolution_state"], "unresolved")
                self.assertEqual(result["correction_audit_payload"]["warnings"][0]["warning"], "invalid_off_page_connector")
                self.assertFalse(result["release_gate_payload"]["release_ready"])

    def test_valid_connector_terminal_survives_opposite_endpoint_change(self):
        reconnect_graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [{"id": "e1", "source": "n1", "target": "n2", "off_page_connector": {"connector_key": "C-1", "exit_terminal": "source"}, "polyline": [[0, 0], [10, 0]]}],
        }
        reconnect = self._topology_call({"decision": "reconnect_edge", "edge_id": "e1", "endpoint": "target", "node_id": "n3"}, graph=reconnect_graph)
        self.assertEqual(reconnect["corrected_graph_payload"]["edges"][0]["off_page_connector"]["exit_terminal"], "source")

        merge_graph = {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [{"id": "e1", "source": "n2", "target": "n3", "off_page_connector": {"connector_key": "C-1", "exit_terminal": "source"}, "polyline": [[0, 0], [10, 0]]}],
        }
        merge = self._topology_call({"decision": "merge_nodes", "node_ids": ["n1", "n2"], "keep_node_id": "n1"}, graph=merge_graph)
        self.assertEqual(merge["corrected_graph_payload"]["edges"][0]["off_page_connector"]["exit_terminal"], "source")

    def test_split_edge_keeps_endpoint_aliases_and_scoped_metadata(self):
        item_id = "stage8::topology::split"
        graph = {
            "nodes": [{"id": "A", "type": "equipment"}, {"id": "B", "type": "equipment"}],
            "edges": [{
                "id": "e1", "source": "A", "target": "B", "trace_id": "trace-1",
                "source_obj_id": "pump-1", "source_obj_type": "pump", "source_port_id": "port-out",
                "source_equipment_id": "equipment-pump-1", "source_node_id": "A",
                "terminal_type": "equipment", "terminal_obj_id": "vessel-1", "terminal_port_id": "port-in",
                "terminal_equipment_id": "equipment-vessel-1", "terminal_node_id": "B",
                "off_page_connector": {"connector_key": "C-1", "exit_terminal": "destination"},
                "polyline": [[0, 0], [5, 0], [10, 0]],
            }],
        }
        result = apply_stage9_review_decisions(
            image_id="x", graph_payload=graph,
            review_items_payload={"review_items": [{"id": item_id, "category": "topology", "release_blocking": True}]},
            decisions_payload={"decisions": [{
                "review_item_id": item_id, "decision": "split_edge", "edge_id": "e1",
                "new_node_id": "mid", "split_index": 1, "reviewer": "qa", "note": "split",
                "reviewed_at": "2026-09-13T00:00:00Z",
            }]},
        )
        edges = {edge["id"]: edge for edge in result["corrected_graph_payload"]["edges"]}
        left, right = edges["e1::a"], edges["e1::b"]
        self.assertEqual((left["src"], left["source"], left["dst"], left["target"]), ("A", "A", "mid", "mid"))
        self.assertEqual((right["src"], right["source"], right["dst"], right["target"]), ("mid", "mid", "B", "B"))
        self.assertEqual(left["source_equipment_id"], "equipment-pump-1")
        self.assertIsNone(left["terminal_equipment_id"])
        self.assertIsNone(right["source_equipment_id"])
        self.assertEqual(right["terminal_equipment_id"], "equipment-vessel-1")
        self.assertNotIn("off_page_connector", left)
        self.assertEqual(right["off_page_connector"]["connector_key"], "C-1")
        self.assertEqual((left["trace_id"], right["trace_id"]), ("trace-1::part_000", "trace-1::part_001"))
        audit = result["correction_audit_payload"]["corrections"][0]
        self.assertEqual((audit["reviewer"], audit["note"], audit["reviewed_at"]), ("qa", "split", "2026-09-13T00:00:00Z"))

    def test_split_edge_uses_unique_trace_ids_when_prefix_is_already_present(self):
        result = self._topology_call({"decision": "split_edge", "edge_id": "e1", "new_node_id": "mid", "split_index": 1}, graph={
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"id": "e1", "source": "n1", "target": "n2", "trace_id": "trace-1", "polyline": [[0, 0], [5, 0], [10, 0]]},
                {"id": "e2", "source": "n2", "target": "n3", "trace_id": "trace-1::part_001", "polyline": [[10, 0], [15, 0], [20, 0]]},
            ],
        })
        trace_ids = [edge["trace_id"] for edge in result["corrected_graph_payload"]["edges"]]
        self.assertEqual(len(trace_ids), len(set(trace_ids)))
        self.assertEqual(sorted(trace_ids), ["trace-1::part_000", "trace-1::part_001", "trace-1::part_001::split_001"])

    def test_reconnect_clears_replaced_endpoint_identity(self):
        result = self._topology_call({"decision": "reconnect_edge", "edge_id": "e1", "endpoint": "source", "node_id": "n3"}, graph={
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [{"id": "e1", "source": "n1", "target": "n2", "source_equipment_id": "eq-1", "source_port_id": "p-1", "source_node_id": "n1", "terminal_equipment_id": "eq-2", "terminal_port_id": "p-2", "terminal_node_id": "n2", "polyline": [[0, 0], [10, 0]]}],
        })
        edge = result["corrected_graph_payload"]["edges"][0]
        self.assertEqual((edge["source"], edge["target"], edge["src"], edge["dst"]), ("n3", "n2", "n3", "n2"))
        self.assertIsNone(edge["source_equipment_id"])
        self.assertIsNone(edge["source_port_id"])
        self.assertEqual(edge["terminal_equipment_id"], "eq-2")

    def test_merge_audit_contains_surviving_rewired_edge_before_and_after(self):
        result = self._topology_call({"decision": "merge_nodes", "node_ids": ["n1", "n2"], "keep_node_id": "n1", "reviewer": "qa"}, graph={
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"id": "e1", "source": "n2", "target": "n3", "source_equipment_id": "old-eq", "polyline": [[0, 0], [1, 0]]},
                {"id": "e2", "source": "n1", "target": "n3", "polyline": [[0, 0], [1, 0]]},
            ],
        })
        audit = result["correction_audit_payload"]["corrections"][0]
        self.assertIn("e1", audit["before"])
        self.assertIn("e2", audit["before"])
        self.assertEqual(audit["before"]["e1"]["source"], "n2")
        self.assertEqual(audit["after"]["e1"]["source"], "n1")
        self.assertEqual(audit["after"]["e2"]["source"], "n1")
        self.assertIsNone(audit["after"]["n2"])
        self.assertIsNone(result["corrected_graph_payload"]["edges"][0]["source_equipment_id"])

    def test_missing_review_item_id_warns_and_blocks_even_when_review_resolves(self):
        result = apply_stage9_review_decisions(
            image_id="x", graph_payload={"nodes": [], "edges": []},
            review_items_payload={"review_items": [{"id": "r", "category": "qa", "release_blocking": True}]},
            decisions_payload={"decisions": [
                {"review_item_id": "r", "decision": "accept_as_is"},
                {"decision": "accept_as_is", "reviewer": "malformed"},
            ]},
        )
        self.assertFalse(result["release_gate_payload"]["release_ready"])
        self.assertIn("missing_review_item_id", result["release_gate_payload"]["blocking_reasons"])
        self.assertEqual(result["correction_audit_payload"]["warnings"][0]["warning"], "missing_review_item_id")

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
        self.assertEqual(result["summary"]["assumed_resolved_count"], 0)
        self.assertEqual(result["review_resolution_payload"]["resolutions"][0]["resolution_state"], "unresolved")

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
        self.assertEqual(resolution["resolution_state"], "unresolved")
        self.assertFalse(resolution["graph_changed"])
        self.assertEqual(result["summary"]["unsupported_decision_count"], 0)

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

        self.assertEqual(result["summary"]["correction_count"], 0)
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
