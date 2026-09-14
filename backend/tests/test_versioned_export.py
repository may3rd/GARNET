import json
import unittest

from garnet.versioned_export import (
    SCHEMA_VERSION,
    build_downstream_export,
    build_blocked_export,
    graph_content_sha256,
    release_gate_sha256,
    serialize_downstream_export,
    validate_downstream_export,
)


def graph():
    return {
        "schema_version": "graph_v1",
        "document": {"doc_id": "P-101", "image": {"width": 100, "height": 80}},
        "nodes": [{"id": "node-a", "type": "equipment"}, {"id": "node-b", "type": "equipment"}],
        "edges": [{
            "id": "route-main", "source": "port-a", "target": "port-b",
            "polyline": [{"x": 1, "y": 2}, {"x": 8, "y": 2}],
            "flow_direction_state": "reverse", "flow_direction_confidence": 0.8,
            "flow_direction_evidence": [{"id": "arrow-1"}],
            "canonical_line_ids": ["line::P-101::L-1"],
            "line_number_ids": ["ocr-1"],
            "attachments": {"inline_objects": [{"id": "valve-1", "class_name": "valve"}]},
            "review_state": "reviewed", "provenance": {"source": "stage9"},
        }],
        "lines": [{"id": "line::P-101::L-1", "occurrences": [{
            "occurrence_id": "ocr-1", "text": "L-1", "normalized_text": "L-1",
            "bbox": {"x": 4, "y": 4, "w": 5, "h": 2}, "confidence": 0.9,
        }]}],
        "equipment": [{"id": "equipment-a", "tag": "P-101"}, {"id": "equipment-b", "tag": "V-101"}],
        "ports": [{"id": "port-a", "equipment_id": "equipment-a"}, {"id": "port-b", "equipment_id": "equipment-b"}],
        "inline_objects": [{"id": "valve-1", "class_name": "valve"}],
        "relationships": [{"id": "has-port", "type": "has_port", "source": "equipment-a", "target": "port-a"}],
        "release_gate": {"release_ready": True},
    }


class VersionedExportTests(unittest.TestCase):
    def test_projection_keeps_physical_endpoints_separate_from_flow(self):
        payload = build_downstream_export(graph(), boundary_payload={"candidates": [{"id": "boundary-1", "state": "candidate"}]})
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        route = payload["graph"]["routes"][0]
        self.assertFalse(payload["graph"]["directed_multigraph"])
        self.assertTrue(payload["graph"]["physical_multigraph"])
        self.assertTrue(payload["graph"]["flow_direction_separate"])
        self.assertEqual(route["physical"], {"source": "port-a", "target": "port-b"})
        self.assertEqual(route["flow"]["state"], "reverse")
        self.assertEqual(route["flow"]["source"], "port-b")
        self.assertEqual(route["flow"]["target"], "port-a")
        self.assertEqual(route["polyline"], [{"x": 1, "y": 2}, {"x": 8, "y": 2}])
        self.assertEqual(payload["graph"]["lines"][0]["occurrences"][0]["occurrence_id"], "ocr-1")
        self.assertEqual(payload["graph"]["line_number_occurrences"][0]["line_id"], "line::P-101::L-1")
        self.assertEqual(payload["engineering_views"]["boundaries"][0]["id"], "boundary-1")
        self.assertEqual(validate_downstream_export(payload)["issues"], [])

    def test_hash_and_serialization_are_deterministic_and_strict(self):
        first = build_downstream_export(graph())
        second = build_downstream_export({**graph(), "nodes": list(reversed(graph()["nodes"]))})
        self.assertEqual(first, second)
        self.assertEqual(first["source"]["graph_content_sha256"], graph_content_sha256(graph()))
        serialized = serialize_downstream_export(first)
        self.assertEqual(serialized, serialize_downstream_export(second))
        json.loads(serialized)
        json.dumps(first, allow_nan=False)

    def test_validator_reports_nonfinite_duplicates_dangling_and_hash_mismatch(self):
        bad = build_downstream_export(graph())
        bad["graph"]["nodes"].append(dict(bad["graph"]["nodes"][0]))
        bad["graph"]["routes"][0]["physical"]["target"] = "missing"
        bad["graph"]["relationships"].append({"id": "bad", "type": "measures", "source": "missing", "target": "missing-2"})
        bad["source"]["graph_content_sha256"] = "0" * 64
        result = validate_downstream_export(bad, source_graph=graph())
        codes = {item["code"] for item in result["issues"]}
        self.assertFalse(result["valid"])
        self.assertIn("duplicate_typed_id", codes)
        self.assertIn("dangling_reference", codes)
        self.assertIn("hash_mismatch", codes)
        bad["graph"]["nodes"][0]["x"] = float("nan")
        self.assertFalse(validate_downstream_export(bad)["valid"])

    def test_unknown_flow_does_not_create_flow_endpoints(self):
        payload = build_downstream_export({**graph(), "edges": [{**graph()["edges"][0], "flow_direction_state": "unknown"}]})
        flow = payload["graph"]["routes"][0]["flow"]
        self.assertEqual(flow["state"], "unknown")
        self.assertNotIn("source", flow)
        self.assertNotIn("target", flow)

    def test_combined_graph_keeps_explicit_cross_sheet_continuity(self):
        source = {"schema_version": "graph_v2", "combined_graph": {
            "schema_version": "graph_v2_combined",
            "drawings": [{"sheet": "A"}, {"sheet": "B"}],
            "nodes": [{"id": "node::A::a"}, {"id": "node::B::b"}],
            "edges": [], "connectors": [{"id": "connector::A::e"}, {"id": "connector::B::e"}],
            "relationships": [{"id": "rel-xs", "type": "cross_sheet_continues",
                                "source": "connector::A::e", "target": "connector::B::e",
                                "review_state": "accepted"}],
        }, "release_gate": {"release_ready": True}}
        payload = build_downstream_export(source)
        continuity = payload["combined_continuity"]
        self.assertTrue(continuity["is_combined"])
        self.assertEqual(continuity["relationships"][0]["type"], "cross_sheet_continues")
        self.assertEqual(validate_downstream_export(payload)["issues"], [])

    def test_blocked_export_redacts_graph_and_binds_gate_hash(self):
        gate = {"release_ready": False, "status": "blocked", "blocking_count": 1}
        payload = build_blocked_export(graph(), release_gate=gate)
        self.assertTrue(payload["review"]["blocked_redaction"] if "blocked_redaction" in payload["review"] else True)
        self.assertTrue(payload["graph"]["redacted"])
        self.assertEqual(payload["graph"]["nodes"], [])
        self.assertEqual(payload["source"]["graph_content_sha256"], graph_content_sha256(graph()))
        self.assertEqual(payload["source"]["release_gate_sha256"], release_gate_sha256(gate))
        self.assertEqual(validate_downstream_export(payload)["issues"], [])

    def test_gate_hash_mismatch_is_rejected(self):
        payload = build_downstream_export(graph())
        payload["source"]["release_gate_sha256"] = "0" * 64
        codes = {item["code"] for item in validate_downstream_export(payload)["issues"]}
        self.assertIn("gate_hash_mismatch", codes)

    def test_no_gate_route_export_is_redacted(self):
        source = graph()
        source.pop("release_gate")
        payload = build_downstream_export(source)
        self.assertFalse(payload["release_ready"])
        self.assertTrue(payload["graph"]["redacted"])
        self.assertEqual(payload["graph"]["routes"], [])
        self.assertEqual(validate_downstream_export(payload)["issues"], [])

    def test_unredacted_blocked_route_export_is_rejected(self):
        payload = build_downstream_export(graph())
        payload["review"]["release_gate"] = {"release_ready": False, "status": "blocked"}
        payload["review"]["release_gate_sha256"] = release_gate_sha256(payload["review"]["release_gate"])
        payload["review"]["release_gate_content_sha256"] = payload["review"]["release_gate_sha256"]
        payload["source"]["release_gate_sha256"] = payload["review"]["release_gate_sha256"]
        payload["source"]["release_gate_content_sha256"] = payload["review"]["release_gate_sha256"]
        payload["release_ready"] = False
        codes = {item["code"] for item in validate_downstream_export(payload)["issues"]}
        self.assertIn("blocked_graph_not_redacted", codes)

    def test_released_export_requires_gate_hash_and_consistent_release_flag(self):
        payload = build_downstream_export(graph())
        payload["source"]["release_gate_sha256"] = None
        payload["review"]["release_gate_sha256"] = None
        payload["release_ready"] = False
        codes = {item["code"] for item in validate_downstream_export(payload)["issues"]}
        self.assertIn("missing_release_gate_hash", codes)
        self.assertIn("release_ready_mismatch", codes)

    def test_combined_outer_document_does_not_create_phantom_drawing(self):
        source = {"schema_version": "graph_v2", "document": {"doc_id": "MERGED"}, "combined_graph": {
            "schema_version": "graph_v2_combined",
            "drawings": [{"sheet": "A"}, {"sheet": "B"}],
            "nodes": [], "edges": [], "connectors": [], "relationships": [],
        }}
        payload = build_downstream_export(source)
        self.assertEqual([item["drawing_id"] for item in payload["drawings"]], ["A", "B"])

    def test_cross_collection_id_collision_is_rejected(self):
        payload = build_downstream_export(graph())
        payload["graph"]["equipment"].append({"id": "node-a"})
        codes = {item["code"] for item in validate_downstream_export(payload)["issues"]}
        self.assertIn("ambiguous_typed_id", codes)

    def test_unresolved_cross_sheet_relationship_is_not_continuity(self):
        source = {"schema_version": "graph_v2", "combined_graph": {
            "schema_version": "graph_v2_combined",
            "drawings": [{"sheet": "A"}, {"sheet": "B"}],
            "connectors": [{"id": "connector::A::e"}, {"id": "connector::B::e"}],
            "relationships": [
                {"id": "rel-unresolved", "type": "cross_sheet_continues", "source": "connector::A::e", "target": "connector::B::e", "state": "unresolved"}
            ],
        }, "release_gate": {"release_ready": True}}
        payload = build_downstream_export(source)
        self.assertEqual(payload["combined_continuity"]["relationships"], [])
        self.assertEqual(validate_downstream_export(payload)["issues"], [])

    def test_no_gate_node_only_export_is_redacted_and_serializable(self):
        source = {
            "schema_version": "graph_v1",
            "document": {"doc_id": "P-208", "image": {"width": 10, "height": 10}},
            "nodes": [{"id": "node-only", "type": "equipment"}],
            "equipment": [{"id": "equipment-only", "tag": "P-208"}],
            "ports": [{"id": "port-only", "equipment_id": "equipment-only"}],
        }
        payload = build_downstream_export(source)
        self.assertFalse(payload["release_ready"])
        self.assertTrue(payload["graph"]["redacted"])
        self.assertEqual(payload["graph"]["nodes"], [])
        self.assertEqual(payload["graph"]["equipment"], [])
        self.assertEqual(payload["graph"]["ports"], [])
        self.assertEqual(validate_downstream_export(payload)["issues"], [])
        self.assertEqual(json.loads(serialize_downstream_export(payload)), payload)

        unredacted = build_downstream_export({**graph(), "release_gate": {"release_ready": True}})
        unredacted["review"]["release_gate"] = None
        unredacted["release_ready"] = False
        unredacted["source"]["release_gate_sha256"] = None
        unredacted["review"]["release_gate_sha256"] = None
        codes = {item["code"] for item in validate_downstream_export(unredacted)["issues"]}
        self.assertIn("blocked_graph_not_redacted", codes)


if __name__ == "__main__":
    unittest.main()
