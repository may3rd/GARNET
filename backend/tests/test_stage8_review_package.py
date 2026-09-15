import unittest

import numpy as np

from garnet.stage8_review_package import (
    _merge_review_items,
    build_stage8_review_package,
    render_stage8_review_overlay,
)


class Stage8ReviewPackageTests(unittest.TestCase):
    def test_build_review_package_converts_qa_issue(self) -> None:
        result = build_stage8_review_package(
            image_id="synthetic.png",
            graph_payload={
                "image_id": "synthetic.png",
                "nodes": [{"id": "junction::1", "type": "tee_junction", "position": {"x": 10, "y": 20}}],
                "edges": [],
                "review_queue": [],
            },
            stage7_qa_payload={
                "image_id": "synthetic.png",
                "issues": [
                    {
                        "id": "qa::tee_degree_mismatch::junction::1",
                        "category": "tee_degree_mismatch",
                        "severity": "high",
                        "node_id": "junction::1",
                        "geometry": {"x": 10, "y": 20},
                        "message": "Tee junction node has degree below 3.",
                    }
                ],
            },
            stage7_review_queue_payload={"review_queue": []},
        )

        item = result["review_items_payload"]["review_items"][0]
        self.assertEqual(item["review_item_type"], "topology")
        self.assertEqual(item["category"], "tee_degree_mismatch")
        self.assertEqual(item["priority"], 10)
        self.assertEqual(item["status"], "open")
        self.assertEqual(item["geometry"], {"x": 10, "y": 20})
        self.assertEqual(result["summary"]["review_item_count"], 1)

    def test_build_review_package_converts_stage7_review_queue(self) -> None:
        result = build_stage8_review_package(
            image_id="synthetic.png",
            graph_payload={"image_id": "synthetic.png", "nodes": [], "edges": []},
            stage7_qa_payload={"image_id": "synthetic.png", "issues": []},
            stage7_review_queue_payload={
                "review_queue": [
                    {
                        "id": "review::line_number_conflict::component_00001",
                        "issue_type": "line_number_conflict",
                        "severity": "review",
                        "message": "Connected trace component has multiple reviewed line numbers.",
                        "candidate_line_number_ids": ["line_1", "line_2"],
                        "component_edge_ids": ["trace::a", "trace::b"],
                    }
                ]
            },
        )

        item = result["review_items_payload"]["review_items"][0]
        self.assertEqual(item["category"], "line_number_conflict")
        self.assertEqual(item["review_item_type"], "line_number")
        self.assertEqual(item["priority"], 9)
        self.assertEqual(item["evidence"]["candidate_line_number_ids"], ["line_1", "line_2"])
        self.assertEqual(item["evidence"]["component_edge_ids"], ["trace::a", "trace::b"])

    def test_render_stage8_review_overlay_draws_issue(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        payload = {
            "image_id": "synthetic.png",
            "review_items": [
                {
                    "id": "stage8::tee_degree_mismatch::1",
                    "category": "tee_degree_mismatch",
                    "priority": 10,
                    "geometry": {"x": 40, "y": 40},
                }
            ],
        }

        overlay = render_stage8_review_overlay(image, payload)

        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(overlay.sum()), 0)

    def test_build_review_package_adds_unknown_and_conflicting_flow_items(self) -> None:
        result = build_stage8_review_package(
            image_id="synthetic.png",
            graph_payload={
                "edges": [
                    {
                        "id": "e-unknown",
                        "flow_direction_state": "unknown",
                        "flow_direction_evidence": [],
                        "polyline": [{"x": 2, "y": 3}, {"x": 12, "y": 3}],
                    },
                    {
                        "id": "e-conflicting",
                        "flow_direction_state": "conflicting",
                        "direction_evidence": [{"id": "arrow-a", "direction": "right"}],
                        "polyline": [{"x": 20, "y": 30}, {"x": 30, "y": 30}],
                    },
                    {"id": "e-forward", "flow_direction_state": "forward"},
                ]
            },
            stage7_qa_payload={"issues": []},
            stage7_review_queue_payload={"review_queue": []},
        )

        items = result["review_items_payload"]["review_items"]
        by_category = {item["category"]: item for item in items}
        self.assertEqual(set(by_category), {"flow_direction_unknown", "flow_direction_conflict"})
        self.assertEqual(by_category["flow_direction_unknown"]["evidence"]["edge_id"], "e-unknown")
        self.assertEqual(
            by_category["flow_direction_conflict"]["evidence"]["direction_evidence"],
            [{"id": "arrow-a", "direction": "right"}],
        )
        self.assertEqual(
            by_category["flow_direction_conflict"]["geometry"]["polyline"],
            [{"x": 20, "y": 30}, {"x": 30, "y": 30}],
        )

    def test_topology_items_promote_existing_targets_and_route_geometry(self) -> None:
        result = build_stage8_review_package(
            image_id="synthetic.png",
            graph_payload={
                "nodes": [
                    {"id": "n-source", "type": "junction", "position": {"x": 0, "y": 0}},
                    {"id": "n-target", "type": "junction", "position": {"x": 20, "y": 0}},
                ],
                "edges": [
                    {
                        "id": "edge-a",
                        "source": "n-source",
                        "target": "n-target",
                        "polyline": [{"x": 0, "y": 0}, {"x": 20, "y": 0}],
                    }
                ],
            },
            stage7_qa_payload={
                "issues": [
                    {
                        "id": "qa::duplicate_physical_path::edge-a",
                        "category": "duplicate_physical_path",
                        "severity": "high",
                        "edge_id": "edge-a",
                        "evidence": {"other_edge_id": "edge-b"},
                    }
                ]
            },
            stage7_review_queue_payload={"review_queue": []},
        )

        item = result["review_items_payload"]["review_items"][0]
        self.assertTrue(item["release_blocking"])
        self.assertEqual(item["release_relevance"], "blocking")
        self.assertEqual(item["source"], "n-source")
        self.assertEqual(item["target"], "n-target")
        self.assertEqual(item["other_edge_id"], "edge-b")
        self.assertEqual(item["target_ids"], {"node_ids": ["n-source", "n-target"], "edge_ids": ["edge-a", "edge-b"]})
        self.assertEqual(item["edge_geometry"], [{"x": 0, "y": 0}, {"x": 20, "y": 0}])

    def test_release_blocking_metadata_is_explicit_and_deterministic(self) -> None:
        result = build_stage8_review_package(
            image_id="synthetic.png",
            graph_payload={
                "edges": [
                    {"id": "flow-edge", "flow_direction_state": "unknown"},
                    {"id": "flow-info", "flow_direction_state": "forward"},
                ]
            },
            stage7_qa_payload={
                "issues": [
                    {"id": "qa::short", "category": "short_trace_edge", "severity": "info"},
                    {"id": "qa::terminal", "category": "unresolved_terminal_edge", "severity": "review", "edge_id": "edge-terminal"},
                ]
            },
            stage7_review_queue_payload={
                "review_queue": [
                    {"id": "review::missing", "issue_type": "missing_line_number", "severity": "review"}
                ]
            },
        )

        items = result["review_items_payload"]["review_items"]
        by_category = {item["category"]: item for item in items}
        self.assertFalse(by_category["short_trace_edge"]["release_blocking"])
        self.assertEqual(by_category["short_trace_edge"]["release_relevance"], "informational")
        for category in ("flow_direction_unknown", "unresolved_terminal_edge", "missing_line_number"):
            self.assertTrue(by_category[category]["release_blocking"])
            self.assertEqual(by_category[category]["release_relevance"], "blocking")
        self.assertEqual(result["summary"]["blocking_review_item_count"], 3)
        self.assertEqual(result["summary"]["release_blocking_review_item_count"], 3)
        self.assertEqual(result["summary"]["informational_review_item_count"], 1)

    def test_topology_item_does_not_invent_route_geometry(self) -> None:
        result = build_stage8_review_package(
            image_id="synthetic.png",
            graph_payload={"edges": [{"id": "edge-no-geometry", "source": "n1", "target": "n2"}]},
            stage7_qa_payload={
                "issues": [
                    {
                        "id": "qa::dead_end_not_expected::edge-no-geometry",
                        "category": "dead_end_not_expected",
                        "severity": "medium",
                        "edge_id": "edge-no-geometry",
                    }
                ]
            },
            stage7_review_queue_payload={"review_queue": []},
        )

        item = result["review_items_payload"]["review_items"][0]
        self.assertEqual(item["source"], "n1")
        self.assertEqual(item["target"], "n2")
        self.assertNotIn("edge_geometry", item)
        self.assertNotIn("geometry", item)

    def test_structural_qa_categories_and_unknown_review_items_block_release(self) -> None:
        categories = [
            "duplicate_node_id",
            "duplicate_edge_id",
            "self_loop_or_bad_endpoint",
            "dangling_equipment_port",
            "isolated_component",
            "unresolved_crossing",
            "line_number_split_components",
        ]
        result = build_stage8_review_package(
            image_id="synthetic.png",
            graph_payload={"nodes": [], "edges": []},
            stage7_qa_payload={
                "issues": [
                    {
                        "id": f"qa::{category}",
                        "category": category,
                        "severity": "high",
                    }
                    for category in categories
                ]
                + [
                    {
                        "id": "qa::new_future_issue",
                        "category": "new_future_issue",
                        "severity": "medium",
                    },
                    {
                        "id": "qa::missing_line_number_component",
                        "category": "missing_line_number_component",
                        "severity": "info",
                    },
                ]
            },
            stage7_review_queue_payload={"review_queue": []},
        )

        items = {item["category"]: item for item in result["review_items_payload"]["review_items"]}
        for category in categories + ["new_future_issue"]:
            self.assertTrue(items[category]["release_blocking"], category)
        self.assertEqual(items["duplicate_node_id"]["review_item_type"], "topology")
        self.assertEqual(items["line_number_split_components"]["review_item_type"], "line_number")
        self.assertEqual(items["missing_line_number_component"]["review_item_type"], "line_number")
        self.assertFalse(items["missing_line_number_component"]["release_blocking"])

    def test_merging_duplicate_sources_preserves_blocking_classification(self) -> None:
        merged = _merge_review_items(
            {
                "id": "stage8::same-source",
                "priority": 1,
                "release_blocking": True,
                "release_relevance": "blocking",
                "source_stage": "stage7_graph_qa",
                "evidence": {},
            },
            {
                "id": "stage8::same-source",
                "priority": 99,
                "release_blocking": False,
                "release_relevance": "informational",
                "source_stage": "stage7_review_queue",
                "evidence": {},
            },
        )

        self.assertTrue(merged["release_blocking"])
        self.assertEqual(merged["release_relevance"], "blocking")


if __name__ == "__main__":
    unittest.main()
