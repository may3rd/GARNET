import unittest

import numpy as np

from garnet.trace_graph_qa import render_stage12_graph_qa_overlay, run_stage12_trace_graph_qa


class TraceGraphQATests(unittest.TestCase):
    def test_tee_degree_counts_parallel_edges(self) -> None:
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [
                {"id": "junction::tee", "type": "tee_junction", "position": {"x": 0, "y": 0}},
                {"id": "equipment::a", "type": "equipment", "position": {"x": 100, "y": 0}},
                {"id": "equipment::b", "type": "equipment", "position": {"x": 0, "y": 100}},
            ],
            "edges": [
                {"id": "trace::a", "source": "junction::tee", "target": "equipment::a", "terminal_type": "equipment", "trace_length_px": 100, "polyline": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]},
                {"id": "trace::b", "source": "junction::tee", "target": "equipment::a", "terminal_type": "equipment", "trace_length_px": 100, "polyline": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]},
                {"id": "trace::c", "source": "junction::tee", "target": "equipment::b", "terminal_type": "equipment", "trace_length_px": 100, "polyline": [{"x": 0, "y": 0}, {"x": 0, "y": 100}]},
            ],
        }

        result = run_stage12_trace_graph_qa(
            image_id="synthetic.png",
            graph_payload=graph_payload,
            image_bgr=np.zeros((160, 160, 3), dtype=np.uint8),
        )

        issue_categories = [issue["category"] for issue in result["qa_payload"]["issues"]]
        self.assertNotIn("tee_degree_mismatch", issue_categories)
        self.assertEqual(result["summary"]["connected_component_count"], 1)

    def test_terminal_only_tee_is_unmerged_terminal_not_degree_mismatch(self) -> None:
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [
                {"id": "equipment::a", "type": "equipment", "position": {"x": 0, "y": 0}},
                {
                    "id": "junction::loose",
                    "type": "tee_junction",
                    "position": {"x": 100, "y": 0},
                    "evidence": [{"role": "terminal", "trace_id": "trace_a"}],
                },
            ],
            "edges": [
                {"id": "trace::a", "source": "equipment::a", "target": "junction::loose", "terminal_type": "tee_junction", "trace_length_px": 100, "polyline": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]},
            ],
        }

        result = run_stage12_trace_graph_qa(
            image_id="synthetic.png",
            graph_payload=graph_payload,
            image_bgr=np.zeros((160, 160, 3), dtype=np.uint8),
        )

        issue_categories = [issue["category"] for issue in result["qa_payload"]["issues"]]
        self.assertIn("unmerged_tee_terminal", issue_categories)
        self.assertNotIn("tee_degree_mismatch", issue_categories)

    def test_effective_line_number_prevents_missing_line_number_component(self) -> None:
        graph_payload = {
            "image_id": "synthetic.png",
            "nodes": [
                {"id": "equipment::a", "type": "equipment", "position": {"x": 0, "y": 0}},
                {"id": "equipment::b", "type": "equipment", "position": {"x": 120, "y": 0}},
            ],
            "edges": [
                {
                    "id": "trace::a",
                    "source": "equipment::a",
                    "target": "equipment::b",
                    "terminal_type": "equipment",
                    "trace_length_px": 120,
                    "polyline": [{"x": 0, "y": 0}, {"x": 120, "y": 0}],
                    "line_number_ids": [],
                    "effective_line_number_ids": ["line_1"],
                },
            ],
        }

        result = run_stage12_trace_graph_qa(
            image_id="synthetic.png",
            graph_payload=graph_payload,
            image_bgr=np.zeros((160, 160, 3), dtype=np.uint8),
        )

        issue_categories = [issue["category"] for issue in result["qa_payload"]["issues"]]
        self.assertNotIn("missing_line_number_component", issue_categories)

    def test_qa_overlay_uses_distinct_colors_for_distinct_line_numbers(self) -> None:
        image = np.full((220, 160, 3), 255, dtype=np.uint8)
        graph_payload = {
            "nodes": [],
            "edges": [
                {
                    "id": "edge_a",
                    "effective_line_number_ids": ["line_a"],
                    "polyline": [{"x": 10, "y": 120}, {"x": 150, "y": 120}],
                },
                {
                    "id": "edge_b",
                    "effective_line_number_ids": ["line_b"],
                    "polyline": [{"x": 10, "y": 170}, {"x": 150, "y": 170}],
                },
            ],
        }

        overlay = render_stage12_graph_qa_overlay(
            image_bgr=image,
            graph_payload=graph_payload,
            qa_payload={"issues": []},
        )
        color_a = tuple(int(value) for value in overlay[118, 30])
        color_b = tuple(int(value) for value in overlay[168, 30])

        self.assertNotEqual(color_a, (255, 255, 255))
        self.assertNotEqual(color_b, (255, 255, 255))
        self.assertNotEqual(color_a, color_b)


if __name__ == "__main__":
    unittest.main()
