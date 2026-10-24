import json
import unittest

from garnet.knowledge_graph import (
    SCHEMA_VERSION,
    build_knowledge_stack,
    release_state,
    serialize_knowledge_stack,
    validate_knowledge_stack,
)


def released_export():
    return {
        "schema_version": "garnet_downstream_export_v1",
        "drawings": [{"drawing_id": "P-101", "revision": "A"}],
        "source": {"schema_version": "graph_v1", "graph_content_sha256": "a" * 64},
        "graph": {
            "nodes": [{"id": "port-a"}, {"id": "port-b"}],
            "routes": [{
                "id": "route-1",
                "physical": {"source": "port-a", "target": "port-b"},
                "flow": {"state": "forward", "source": "port-a", "target": "port-b"},
                "canonical_line_ids": ["line-1"],
                "confidence": 0.9,
            }],
            "lines": [{
                "id": "line-1",
                "size": "6in",
                "spec": "A1A",
                "service": "process",
                "design_pressure": 19.0,
                "occurrences": [{"occurrence_id": "ocr-1", "normalized_text": "6-P-101-1001"}],
            }],
            "equipment": [{"id": "equipment-a", "tag": "P-101"}, {"id": "equipment-b", "tag": "V-101"}],
            "ports": [
                {"id": "port-a", "equipment_id": "equipment-a"},
                {"id": "port-b", "equipment_id": "equipment-b"},
            ],
            "relationships": [
                {"id": "has-port-a", "type": "has_port", "source": "equipment-a", "target": "port-a"},
                {"id": "has-port-b", "type": "has_port", "source": "equipment-b", "target": "port-b"},
            ],
        },
        "review": {"release_gate": {"release_ready": True}, "release_gate_sha256": "b" * 64, "release_ready": True},
        "release_ready": True,
        "blocked_reasons": [],
    }


def blocked_export():
    payload = released_export()
    payload["release_ready"] = False
    payload["review"] = {"release_gate": {"release_ready": False}, "release_ready": False}
    payload["blocked_reasons"] = ["stage9_release_gate_not_ready"]
    payload["graph"] = {"redacted": True, "nodes": [], "routes": [], "lines": [],
                        "equipment": [], "ports": [], "relationships": []}
    return payload


class KnowledgeStackTests(unittest.TestCase):
    def test_released_export_builds_topology_knowledge_and_jsonld(self):
        stack = build_knowledge_stack(released_export())
        self.assertEqual(stack["schema_version"], SCHEMA_VERSION)
        self.assertTrue(stack["release_ready"])
        self.assertFalse(stack["redacted"])

        topology = stack["physical_topology"]
        self.assertFalse(topology["directed"])
        self.assertTrue(topology["flow_direction_separate"])
        edge = topology["edges"][0]
        self.assertTrue(edge["undirected"])
        self.assertEqual(edge["source_equipment_id"], "equipment-a")
        self.assertEqual(edge["target_equipment_id"], "equipment-b")
        self.assertEqual(edge["line_ids"], ["line-1"])
        self.assertEqual(topology["adjacency"]["equipment-a"], ["equipment-b"])

        study_nodes = stack["process_knowledge"]["study_nodes"]
        self.assertEqual([node["id"] for node in study_nodes],
                         ["studynode::equipment-a", "studynode::equipment-b"])
        node_a = study_nodes[0]
        self.assertEqual(node_a["label"], "P-101")
        self.assertEqual(node_a["line_ids"], ["line-1"])
        self.assertEqual(node_a["route_ids"], ["route-1"])
        self.assertEqual(node_a["downstream_node_ids"], ["equipment-b"])
        self.assertEqual(node_a["process_properties"]["size"], "6in")
        self.assertEqual(node_a["process_properties"]["line_number"], "6-P-101-1001")
        self.assertTrue(stack["jsonld"]["@graph"])
        self.assertEqual(validate_knowledge_stack(stack)["issues"], [])

    def test_flow_evidence_never_creates_physical_direction(self):
        payload = released_export()
        payload["graph"]["routes"][0]["flow"] = {"state": "unknown"}
        stack = build_knowledge_stack(payload)
        edge = stack["physical_topology"]["edges"][0]
        self.assertTrue(edge["undirected"])
        self.assertEqual(edge["flow_state"], "unknown")
        self.assertIsNone(edge["flow_source"])
        self.assertIn("unknown_flow_direction:route-1", stack["issues"])

    def test_both_gate_flags_must_agree(self):
        payload = released_export()
        payload["review"]["release_gate"]["release_ready"] = False
        self.assertEqual(release_state(payload), (False, ["release_gate_not_ready"]))
        stack = build_knowledge_stack(payload)
        self.assertTrue(stack["redacted"])
        self.assertEqual(stack["physical_topology"]["edges"], [])

    def test_blocked_gate_redacts_every_layer(self):
        stack = build_knowledge_stack(blocked_export())
        self.assertFalse(stack["release_ready"])
        self.assertTrue(stack["redacted"])
        self.assertEqual(stack["blocked_reasons"], ["stage9_release_gate_not_ready"])
        self.assertEqual(stack["physical_topology"]["edges"], [])
        self.assertEqual(stack["process_knowledge"]["study_nodes"], [])
        self.assertEqual(stack["deviation_graph"]["nodes"], [])
        self.assertEqual(stack["jsonld"]["@graph"], [])
        self.assertEqual(validate_knowledge_stack(stack)["issues"], [])

    def test_deviation_graph_links_to_study_nodes_and_reports_unknown(self):
        stack = build_knowledge_stack(released_export(), deviations=[
            {"id": "dev-1", "study_node_id": "studynode::equipment-a", "guide_word": "MORE",
             "parameter": "flow", "cause": "pump runout", "consequence": "overpressure",
             "safeguard": "PSV-101", "safeguard_ids": ["instrument::PSV-101"]},
            {"id": "dev-2", "study_node_id": "studynode::missing", "guide_word": "LESS"},
        ])
        deviations = stack["deviation_graph"]
        self.assertEqual([node["id"] for node in deviations["nodes"]], ["dev-1"])
        self.assertEqual(deviations["nodes"][0]["guide_word"], "MORE")
        edge_types = {edge["id"]: edge["type"] for edge in deviations["edges"]}
        self.assertEqual(edge_types["applies_to::dev-1::studynode::equipment-a"], "applies_to")
        self.assertEqual(edge_types["protects::dev-1::instrument::PSV-101"], "protects")
        self.assertIn("unknown_study_node:studynode::missing", stack["issues"])
        self.assertEqual(validate_knowledge_stack(stack)["issues"], [])

    def test_paired_route_counts_on_both_study_nodes(self):
        payload = released_export()
        payload["graph"]["relationships"].append(
            {"id": "meas-1", "type": "measures", "source": "equipment-a", "target": "instrument::PI-101"}
        )
        stack = build_knowledge_stack(payload)
        node_a, node_b = stack["process_knowledge"]["study_nodes"]
        self.assertEqual(node_a["route_ids"], ["route-1"])
        self.assertEqual(node_b["route_ids"], ["route-1"])
        self.assertEqual(node_a["instrument_ids"], ["instrument::PI-101"])

    def test_serialization_is_deterministic_and_strict(self):
        first = build_knowledge_stack(released_export())
        payload = released_export()
        payload["graph"]["nodes"] = list(reversed(payload["graph"]["nodes"]))
        second = build_knowledge_stack(payload)
        self.assertEqual(first["physical_topology"], second["physical_topology"])
        serialized = serialize_knowledge_stack(first)
        self.assertEqual(serialized, serialize_knowledge_stack(first))
        json.loads(serialized)
        json.dumps(first, allow_nan=False)

    def test_validator_rejects_hash_mismatch_and_dangling_references(self):
        stack = build_knowledge_stack(released_export())
        self.assertFalse(validate_knowledge_stack(stack, source_export=blocked_export())["valid"])

        stack["physical_topology"]["edges"][0]["undirected"] = False
        stack["process_knowledge"]["study_nodes"][0]["route_ids"] = ["route-missing"]
        stack["process_knowledge"]["study_nodes"][0]["equipment_ids"] = ["equipment-missing"]
        codes = {item["code"] for item in validate_knowledge_stack(stack)["issues"]}
        self.assertIn("directed_physical_edge", codes)
        self.assertIn("dangling_route_reference", codes)
        self.assertIn("dangling_equipment_reference", codes)

    def test_wrong_input_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            build_knowledge_stack({"schema_version": "graph_v1"})
        with self.assertRaises(ValueError):
            build_knowledge_stack("not-an-envelope")


if __name__ == "__main__":
    unittest.main()
