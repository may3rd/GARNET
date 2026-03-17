import unittest

import numpy as np

from garnet.pipe_graph_qa import run_pipe_graph_qa_stage


class PipeGraphQaTests(unittest.TestCase):
    def test_run_pipe_graph_qa_stage_reports_components_and_isolated_nodes(self) -> None:
        graph_payload = {
            "nodes": [
                {"id": "n1", "type": "endpoint"},
                {"id": "n2", "type": "junction"},
                {"id": "n3", "type": "endpoint"},
            ],
            "edges": [
                {"id": "e1", "source": "n1", "target": "n2"},
            ],
            "edge_components": [
                ["e1"],
            ],
            "crossings": [
                {"id": "junction_1", "classification": "unresolved", "branch_count": 4, "unresolved_reasons": ["four_way_tie"]},
            ],
            "edge_terminals": [
                {
                    "edge_id": "e1",
                    "source_node_id": "n1",
                    "destination_node_id": "n2",
                    "source_terminal": {"terminal_role": "unresolved_terminal"},
                    "destination_terminal": {"terminal_role": "junction_terminal"},
                    "provisional_due_to_unresolved_terminal": True,
                }
            ],
        }

        result = run_pipe_graph_qa_stage(
            image_id="sample.png",
            graph_payload=graph_payload,
            image_bgr=np.zeros((20, 20, 3), dtype=np.uint8),
        )

        self.assertEqual(result["anomaly_report"]["connected_component_count"], 1)
        self.assertEqual(result["anomaly_report"]["isolated_node_count"], 1)
        self.assertEqual(result["anomaly_report"]["unresolved_crossing_count"], 1)
        self.assertEqual(result["anomaly_report"]["unresolved_terminal_edge_count"], 1)
        self.assertEqual(result["summary"]["review_queue_count"], 3)
        self.assertEqual(result["component_overlay_image"].shape, (20, 20, 3))
        unresolved_terminal_items = [
            item for item in result["review_queue"]["items"] if item["category"] == "unresolved_terminal_edge"
        ]
        self.assertEqual(len(unresolved_terminal_items), 1)
        self.assertEqual(unresolved_terminal_items[0]["edge_ids"], ["e1"])


if __name__ == "__main__":
    unittest.main()
