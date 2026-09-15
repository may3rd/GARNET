import copy
import json
import unittest
from pathlib import Path

from garnet.phase9_benchmark import (
    BenchmarkContractError,
    evaluate_graph_payload,
    run_phase9_benchmark,
    validate_benchmark_case,
)
from garnet.versioned_export import build_downstream_export


FIXTURE = Path(__file__).parent / "fixtures" / "phase9_benchmark" / "benchmark.json"


class Phase9BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_representative_fixture_bundle_passes_without_detector_accuracy(self):
        report = run_phase9_benchmark(self.bundle, strict=True)
        self.assertTrue(report["passed"])
        self.assertEqual(report["case_count"], 4)
        self.assertEqual(report["detector_accuracy"]["status"], "not_scored")
        self.assertEqual(report["acceptance_thresholds"]["contract_violations"], 0)
        self.assertEqual(report["acceptance_thresholds"]["route_pixel_geometry_ratio"], 1.0)
        self.assertEqual(
            [item["case_id"] for item in report["cases"]],
            ["repeated_labels_unresolved_connectors", "representative_multi_sheet", "representative_single_sheet", "topology_and_instrument_semantics"],
        )

    def test_single_sheet_metrics_cover_routes_and_preserve_unresolved_evidence(self):
        case = next(item for item in self.bundle["cases"] if item["id"] == "representative_single_sheet")
        report = validate_benchmark_case(case, strict=True)
        metrics = report["metrics"]["coverage_validity"]
        self.assertEqual(metrics["route_pixel_geometry"]["ratio"], 1.0)
        self.assertEqual(metrics["physical_connectivity"]["ratio"], 1.0)
        self.assertEqual(metrics["line_assignment"]["ratio"], 1.0)
        self.assertEqual(metrics["line_occurrence_traceability"]["ratio"], 1.0)
        self.assertIn("parallel_route_preservation", {item["id"] for item in report["checks"]})
        self.assertTrue(any(item["kind"] == "relationship" for item in report["unresolved_evidence"]))

    def test_multi_sheet_continuity_requires_explicit_relationship(self):
        case = copy.deepcopy(next(item for item in self.bundle["cases"] if item["id"] == "representative_multi_sheet"))
        combined = case["payload"]["combined_graph"]
        combined["relationships"] = []
        report = validate_benchmark_case(case)
        self.assertFalse(report["passed"])
        self.assertIn("multi_sheet_connector_continuity", {item["check"] for item in report["violations"]})
        with self.assertRaises(BenchmarkContractError):
            validate_benchmark_case(case, strict=True)

    def test_geometry_violation_fails_but_unresolved_route_is_reported(self):
        case = copy.deepcopy(next(item for item in self.bundle["cases"] if item["id"] == "representative_single_sheet"))
        edge = case["payload"]["edges"][0]
        edge["polyline"] = []
        edge["state"] = "unresolved"
        report = validate_benchmark_case(case)
        self.assertFalse(report["passed"])
        self.assertTrue(any(item["kind"] == "route" and item["id"] == edge["id"] for item in report["unresolved_evidence"]))
        self.assertIn("route_pixel_geometry", {item["check"] for item in report["violations"]})

    def test_topology_and_unresolved_connector_contract_cases_pass(self):
        reports = {item["case_id"]: item for item in run_phase9_benchmark(self.bundle)["cases"]}
        self.assertTrue(reports["topology_and_instrument_semantics"]["passed"])
        self.assertTrue(reports["repeated_labels_unresolved_connectors"]["passed"])
        self.assertIn("crossing_without_junction", {item["id"] for item in reports["topology_and_instrument_semantics"]["checks"]})
        self.assertIn("unresolved_connector_no_continuity", {item["id"] for item in reports["repeated_labels_unresolved_connectors"]["checks"]})

    def test_report_is_deterministic_under_repeated_execution(self):
        first = json.dumps(run_phase9_benchmark(self.bundle), sort_keys=True)
        second = json.dumps(run_phase9_benchmark(self.bundle), sort_keys=True)
        self.assertEqual(first, second)

    def test_versioned_export_contract_and_source_hash_are_checked(self):
        source = {
            "schema_version": "graph_v1",
            "document": {"doc_id": "P-207", "image": {"width": 100, "height": 80}},
            "drawing": {"drawing_id": "P-207", "pixel_dimensions": {"width": 100, "height": 80}, "coordinate_system": "image_pixel_origin_top_left"},
            "nodes": [{"id": "n1"}, {"id": "n2"}],
            "edges": [{"id": "r1", "src": "n1", "dst": "n2", "polyline": [{"x": 1, "y": 1}, {"x": 2, "y": 2}]}],
            "lines": [{"id": "line-1", "edge_ids": ["r1"], "occurrences": [{"id": "ocr-1", "text": "L-1"}]}],
            "release_gate": {"release_ready": True, "status": "ready"},
        }
        payload = build_downstream_export(source)
        report = evaluate_graph_payload(payload, source_graph=source, case_id="versioned")
        self.assertTrue(report["passed"])
        payload["graph_content_sha256"] = "0" * 64
        payload["source"]["graph_content_sha256"] = "0" * 64
        report = evaluate_graph_payload(payload, source_graph=source, case_id="versioned-tampered")
        self.assertFalse(report["passed"])
        self.assertIn("versioned_export_contract", {item["check"] for item in report["violations"]})


if __name__ == "__main__":
    unittest.main()
