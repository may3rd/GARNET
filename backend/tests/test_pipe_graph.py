import unittest

from garnet.pipe_graph import run_pipe_graph_stage


class PipeGraphTests(unittest.TestCase):
    def test_run_pipe_graph_stage_builds_nodes_edges_and_summary(self) -> None:
        clusters = [
            {"id": "endpoint_0", "kind": "endpoint", "centroid": {"x": 1.0, "y": 2.0}, "member_count": 1},
            {"id": "junction_0", "kind": "junction", "centroid": {"x": 5.0, "y": 6.0}, "member_count": 3},
        ]
        edges = [
            {"id": "edge_0", "source": "endpoint_0", "target": "junction_0", "pixel_length": 10, "polyline": []},
        ]
        confirmed_junctions = [{"id": "junction_0"}]
        unresolved_junctions = []

        result = run_pipe_graph_stage(
            image_id="sample.png",
            node_clusters=clusters,
            edges=edges,
            confirmed_junctions=confirmed_junctions,
            unresolved_junctions=unresolved_junctions,
        )

        self.assertEqual(result["summary"]["node_count"], 2)
        self.assertEqual(result["summary"]["edge_count"], 1)
        self.assertEqual(result["summary"]["connected_component_count"], 1)
        self.assertEqual(len(result["graph_payload"]["nodes"]), 2)
        self.assertEqual(len(result["graph_payload"]["edges"]), 1)


if __name__ == "__main__":
    unittest.main()
