import json
import unittest
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_pid_graph" / "fixtures.json"


class CanonicalPidGraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_bundle_has_expected_cases(self):
        self.assertEqual(self.payload["schema_version"], "canonical_pid_graph_fixture_v1")
        self.assertEqual({item["id"] for item in self.payload["fixtures"]}, {"parallel_bypass", "crossing_and_tee", "repeated_labels_multisheet"})

    def test_parallel_routes_with_same_endpoints_are_preserved(self):
        item = next(item for item in self.payload["fixtures"] if item["id"] == "parallel_bypass")
        segments = item["segments"]
        self.assertEqual(len(segments), 2)
        self.assertEqual({(s["source"], s["target"]) for s in segments}, {("port:pump-out", "port:vessel-in")})
        self.assertNotEqual(segments[0]["polyline"], segments[1]["polyline"])

    def test_crossing_and_marked_tee_have_distinct_semantics(self):
        item = next(item for item in self.payload["fixtures"] if item["id"] == "crossing_and_tee")
        self.assertTrue(all(segment["state"] == "unresolved" for segment in item["segments"][:2]))
        tee = item["junctions"][0]
        self.assertEqual(len(tee["incident_segments"]), 3)
        self.assertEqual(item["direction"]["state"], "conflicting")

    def test_labels_and_uncertainty_remain_sheet_scoped(self):
        item = next(item for item in self.payload["fixtures"] if item["id"] == "repeated_labels_multisheet")
        self.assertEqual(len(item["lines"]), 2)
        self.assertEqual(item["lines"][0]["number"], item["lines"][1]["number"])
        self.assertEqual(item["connector"]["state"], "unresolved")

    def test_uncertainty_fixture_never_promotes_unresolved_facts(self):
        for item in self.payload["fixtures"]:
            for segment in item.get("segments", []):
                if segment["state"] == "unresolved":
                    self.assertNotEqual(segment.get("review_state"), "accepted")
            direction = item.get("direction", {})
            if direction.get("state") in {"unknown", "conflicting"}:
                self.assertNotEqual(direction.get("review_state"), "accepted")
            connector = item.get("connector", {})
            if connector.get("state") == "unresolved":
                self.assertNotEqual(connector.get("review_state"), "accepted")

    def test_segments_retain_pixel_geometry_and_drawing_identity(self):
        for item in self.payload["fixtures"]:
            for segment in item.get("segments", []):
                self.assertGreaterEqual(len(segment["polyline"]), 2)
                self.assertTrue(all({"x", "y"} <= set(point) for point in segment["polyline"]))
                if "drawing" in item:
                    self.assertEqual(item["drawing"]["coordinate_system"], "image_pixel_origin_top_left")


if __name__ == "__main__":
    unittest.main()
