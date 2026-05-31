import unittest

import numpy as np

from garnet.trace_graph_qa import run_stage12_trace_graph_qa


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


if __name__ == "__main__":
    unittest.main()
