import unittest

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
        }

        result = run_pipe_graph_qa_stage(
            image_id="sample.png",
            graph_payload=graph_payload,
        )

        self.assertEqual(result["anomaly_report"]["connected_component_count"], 2)
        self.assertEqual(result["anomaly_report"]["isolated_node_count"], 1)
        self.assertEqual(result["summary"]["review_queue_count"], 1)


if __name__ == "__main__":
    unittest.main()
