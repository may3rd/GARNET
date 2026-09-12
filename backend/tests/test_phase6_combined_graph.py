import json
import unittest
from copy import deepcopy

from garnet.graph_export_adapter import build_graph_v1_payload
from garnet.pipe_sheet_merge import _boundary_flow, resolve_merge_pairs


def graph(sheet, edge_id, key, direction="output", flow="forward", target="other"):
    return {
        "schema_version": "graph_v1", "document": {"doc_id": sheet},
        "nodes": [{"id": "n1", "type": "equipment"}, {"id": "n2", "type": "junction"}],
        "edges": [{"id": edge_id, "src": "n1", "dst": "n2", "flow_direction_state": flow,
                    "off_page_connector": {"local_edge_id": edge_id, "reference_type": "sheet",
                        "reference_value": key, "connector_key": key,
                        "direction": direction, "target_sheet_reference": target,
                        "exit_terminal": "source"}}],
    }


class CombinedGraphTests(unittest.TestCase):
    def test_duplicate_document_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate document.doc_id"):
            resolve_merge_pairs([graph(" Sheet-A ", "e1", "x"), graph("sheet-a", "e2", "y")])
        first = graph("A", "e1", "x")
        second = graph("B", "e2", "y")
        first["document"] = {}
        second["document"] = {}
        with self.assertRaisesRegex(ValueError, "duplicate document.doc_id"):
            resolve_merge_pairs([first, second])

    def test_boundary_flow_all_direction_endpoint_cases(self):
        self.assertEqual(_boundary_flow({"flow_direction_state": "forward"}, "source"), "incoming")
        self.assertEqual(_boundary_flow({"flow_direction_state": "forward"}, "target"), "outgoing")
        self.assertEqual(_boundary_flow({"flow_direction_state": "reverse"}, "source"), "outgoing")
        self.assertEqual(_boundary_flow({"flow_direction_state": "reverse"}, "target"), "incoming")

    def test_qualified_projection_and_continuity(self):
        a = graph("A", "e", "LINE-1", "output", "forward", "B")
        b = graph("B", "e", "LINE-1", "input", "reverse", "A")
        merged = resolve_merge_pairs([a, b]).to_dict()
        combined = merged["combined_graph"]
        self.assertEqual({n["id"] for n in combined["nodes"]}, {"node::A::n1", "node::A::n2", "node::B::n1", "node::B::n2"})
        self.assertEqual({c["id"] for c in combined["connectors"]}, {"connector::A::e", "connector::B::e"})
        self.assertEqual(len(combined["relationships"]), 1)
        relation = combined["relationships"][0]
        self.assertEqual(relation["type"], "cross_sheet_continues")
        self.assertEqual(relation["boundary_flow"][0]["flow"], "incoming")
        self.assertEqual(relation["boundary_flow"][1]["flow"], "outgoing")
        refs = {relation["source"], relation["target"]}
        self.assertTrue(refs <= {c["id"] for c in combined["connectors"]})

    def test_unmatched_connector_is_issue_without_relationship(self):
        result = resolve_merge_pairs([graph("A", "e", "LINE-1")]).to_dict()
        self.assertEqual(result["combined_graph"]["relationships"], [])
        self.assertTrue(result["combined_graph"]["issues"])

    def test_reversed_sheet_order_is_identical(self):
        a = graph("A", "e", "LINE-1", target="B")
        b = graph("B", "e", "LINE-1", direction="input", target="A")
        left = resolve_merge_pairs([a, b]).to_dict()
        right = resolve_merge_pairs([b, a]).to_dict()
        self.assertEqual(json.dumps(left, sort_keys=True), json.dumps(right, sort_keys=True))

    def test_catalog_and_relationship_references_are_qualified(self):
        a = graph("A", "e", "LINE-1", target="B")
        b = graph("B", "e", "LINE-1", direction="input", target="A")
        a.update({"equipment": [{"id": "same", "tag": "P-1"}],
                  "equipment_ports": [{"id": "p", "equipment_id": "same", "edge_id": "e"}],
                  "lines": [{"id": "line", "line_edges": [{"edge_id": "e"}]}],
                  "instruments": [{"id": "same", "equipment_id": "same"}],
                  "relationships": [{"id": "r", "type": "has_port", "source": "same", "target": "p", "edge_id": "e"}]})
        combined = resolve_merge_pairs([a, b]).to_dict()["combined_graph"]
        self.assertEqual(combined["equipment"][0]["id"], "equipment::A::same")
        port = combined["equipment_ports"][0]
        self.assertEqual(port["equipment_id"], "equipment::A::same")
        self.assertEqual(port["edge_id"], "edge::A::e")
        self.assertEqual(combined["lines"][0]["line_edges"][0]["edge_id"], "edge::A::e")
        rel = next(item for item in combined["relationships"] if item.get("type") == "has_port")
        self.assertEqual(rel["source"], "equipment::A::same")
        self.assertEqual(rel["target"], "port::A::p")

    def test_projection_preserves_provenance_and_review_state(self):
        a = graph("A", "left", "LINE-1", target="B")
        b = graph("B", "right", "LINE-1", direction="input", target="A")
        a["nodes"][0]["provenance"] = {"detector": "ocr"}
        merged = resolve_merge_pairs(
            [a, b],
            strict=True,
            manual_pairs=[{"left_connector_id": "A::left", "right_connector_id": "B::right"}],
        ).to_dict()["combined_graph"]
        self.assertEqual(merged["nodes"][0]["provenance"]["detector"], "ocr")
        relation = next(item for item in merged["relationships"] if item["type"] == "cross_sheet_continues")
        self.assertEqual(relation["semantic_state"], "reviewed")
        self.assertEqual(relation["review_state"], "accepted")

    def test_override_uses_local_connector_id_and_exposes_effective_fields(self):
        a = graph("A", "edge-a", "raw", target="B")
        b = graph("B", "edge-b", "raw", direction="input", target="A")
        a["edges"][0]["off_page_connector"]["local_edge_id"] = "connector-a"
        merged = resolve_merge_pairs(
            [a, b], strict=True,
            connector_overrides={"A::connector-a": {"connector_key": "fixed", "target_sheet_id": "B"},
                                 "B::edge-b": {"connector_key": "fixed", "target_sheet_id": "A"}},
        ).to_dict()["combined_graph"]
        connector = next(item for item in merged["connectors"] if item["sheet"] == "A")
        self.assertEqual(connector["connector_key"], "fixed")
        self.assertEqual(connector["target_sheet_id"], "B")
        self.assertEqual(connector["raw_connector_key"], "raw")

    def test_real_graph_v1_adapter_references_remain_resolvable(self):
        base = build_graph_v1_payload({
            "nodes": [
                {"id": "n1", "type": "junction", "position": {"x": 0, "y": 0}},
                {"id": "n2", "type": "junction", "position": {"x": 10, "y": 0}},
            ],
            "edges": [{
                "id": "route",
                "source": "n1",
                "target": "n2",
                "polyline": [[0, 0], [0, 10]],
                "attachments": {
                    "inline_objects": [{"id": "valve", "source_object_id": "valve"}],
                    "instrument_tags": [{"id": "ft", "text": "FT-1", "relationship_type": "measures"}],
                },
            }],
        })
        graphs = []
        for sheet, target, direction in (("A", "B", "output"), ("B", "A", "input")):
            payload = deepcopy(base)
            payload["document"]["doc_id"] = sheet
            payload["edges"][0]["off_page_connector"] = {
                "local_edge_id": "route",
                "reference_type": "drawing",
                "reference_value": target,
                "target_sheet_reference": target,
                "connector_key": "10-P-100-A",
                "direction": direction,
                "exit_terminal": "target",
            }
            graphs.append(payload)

        combined = resolve_merge_pairs(graphs, strict=True).to_dict()["combined_graph"]
        entity_ids = {
            item["id"]
            for catalog in ("nodes", "edges", "connectors", "equipment", "equipment_ports", "lines", "inline_objects", "instruments")
            for item in combined[catalog]
        }
        for edge in combined["edges"]:
            self.assertIn(edge["src"], entity_ids)
            self.assertIn(edge["dst"], entity_ids)
            for field in ("inline_object_ids", "ordered_inline_object_ids", "instrument_ids", "canonical_line_ids"):
                self.assertTrue(set(edge.get(field, [])) <= entity_ids)
            for item in edge.get("ordered_inline_objects", []):
                self.assertIn(item["inline_object_id"], entity_ids)
        for relationship in combined["relationships"]:
            self.assertIn(relationship["source"], entity_ids)
            self.assertIn(relationship["target"], entity_ids)
            if relationship.get("edge_id"):
                self.assertIn(relationship["edge_id"], entity_ids)

    def test_source_rejected_connector_never_creates_continuity(self):
        a = graph("A", "left", "LINE-1", target="B")
        b = graph("B", "right", "LINE-1", direction="input", target="A")
        a["edges"][0]["off_page_connector"]["review_state"] = "rejected"

        merged = resolve_merge_pairs([a, b], strict=True).to_dict()

        self.assertEqual(merged["cross_sheet_edges"], [])
        self.assertFalse(any(item["type"] == "cross_sheet_continues" for item in merged["combined_graph"]["relationships"]))
        self.assertIn("rejected_connector", {item["type"] for item in merged["merge_issues"]})

    def test_multiple_pairs_are_deterministic_under_sheet_reordering(self):
        graphs = [
            graph("A", "a", "LINE-1", target="B"),
            graph("B", "b", "LINE-1", direction="input", target="A"),
            graph("C", "c", "LINE-2", target="D"),
            graph("D", "d", "LINE-2", direction="input", target="C"),
        ]
        forward = resolve_merge_pairs(graphs).to_dict()
        reverse = resolve_merge_pairs(list(reversed(graphs))).to_dict()
        self.assertEqual(forward, reverse)

    def test_virtual_ids_are_unique_for_punctuation_distinct_connectors(self):
        a = graph("A", "a:b", "K-1", target="B")
        a["edges"].append(deepcopy(graph("A", "a/b", "K-2", target="B")["edges"][0]))
        b = graph("B", "c:d", "K-1", direction="input", target="A")
        b["edges"].append(deepcopy(graph("B", "c/d", "K-2", direction="input", target="A")["edges"][0]))

        merged = resolve_merge_pairs([a, b], strict=True).to_dict()

        edge_ids = [item["id"] for item in merged["cross_sheet_edges"]]
        relation_ids = [
            item["id"]
            for item in merged["combined_graph"]["relationships"]
            if item["type"] == "cross_sheet_continues"
        ]
        self.assertEqual(len(edge_ids), 2)
        self.assertEqual(len(edge_ids), len(set(edge_ids)))
        self.assertEqual(len(relation_ids), len(set(relation_ids)))


if __name__ == "__main__":
    unittest.main()
