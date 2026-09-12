import unittest

import numpy as np

from garnet.stage8_review_package import build_stage8_review_package, render_stage8_review_overlay


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


if __name__ == "__main__":
    unittest.main()
