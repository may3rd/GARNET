import unittest

from garnet.pipe_edge_connectivity import build_pipe_edge_connectivity


class PipeEdgeConnectivityTests(unittest.TestCase):
    def test_build_pipe_edge_connectivity_connects_edges_through_horizontal_inline_arrow(self) -> None:
        result = build_pipe_edge_connectivity(
            edges=[
                {
                    "id": "edge_left",
                    "source": "endpoint_0",
                    "target": "endpoint_1",
                    "polyline": [{"row": 10, "col": 0}, {"row": 10, "col": 20}],
                },
                {
                    "id": "edge_right",
                    "source": "endpoint_2",
                    "target": "endpoint_3",
                    "polyline": [{"row": 10, "col": 22}, {"row": 10, "col": 40}],
                },
                {
                    "id": "edge_top_noise",
                    "source": "endpoint_4",
                    "target": "endpoint_5",
                    "polyline": [{"row": 0, "col": 21}, {"row": 5, "col": 21}],
                },
            ],
            node_clusters=[],
            object_regions=[
                {
                    "id": "obj_arrow",
                    "class_name": "arrow",
                    "bbox": {"x_min": 16, "y_min": 8, "x_max": 28, "y_max": 14},
                }
            ],
            inline_connector_classes=("arrow", "valve", "reducer"),
            inline_match_distance_px=8.0,
        )

        self.assertEqual(result["summary"]["inline_element_connection_count"], 1)
        self.assertEqual(result["connections"][0]["kind"], "inline_element")
        self.assertEqual(result["connections"][0]["connector_class"], "arrow")
        self.assertEqual(
            tuple(sorted((result["connections"][0]["source_edge_id"], result["connections"][0]["target_edge_id"]))),
            ("edge_left", "edge_right"),
        )

    def test_build_pipe_edge_connectivity_forces_vertical_arrow_to_top_bottom_pair(self) -> None:
        result = build_pipe_edge_connectivity(
            edges=[
                {
                    "id": "edge_top",
                    "source": "endpoint_0",
                    "target": "endpoint_1",
                    "polyline": [{"row": 0, "col": 20}, {"row": 18, "col": 20}],
                },
                {
                    "id": "edge_bottom",
                    "source": "endpoint_2",
                    "target": "endpoint_3",
                    "polyline": [{"row": 22, "col": 20}, {"row": 40, "col": 20}],
                },
                {
                    "id": "edge_left_noise",
                    "source": "endpoint_4",
                    "target": "endpoint_5",
                    "polyline": [{"row": 20, "col": 0}, {"row": 20, "col": 14}],
                },
            ],
            node_clusters=[],
            object_regions=[
                {
                    "id": "obj_arrow",
                    "class_name": "arrow",
                    "bbox": {"x_min": 16, "y_min": 14, "x_max": 24, "y_max": 26},
                }
            ],
            inline_connector_classes=("arrow", "valve", "reducer"),
            inline_match_distance_px=8.0,
        )

        self.assertEqual(result["summary"]["inline_element_connection_count"], 1)
        self.assertEqual(
            tuple(sorted((result["connections"][0]["source_edge_id"], result["connections"][0]["target_edge_id"]))),
            ("edge_bottom", "edge_top"),
        )
        self.assertEqual(result["connections"][0]["alignment"], "vertical")

    def test_build_pipe_edge_connectivity_forces_horizontal_reducer_to_left_right_pair(self) -> None:
        result = build_pipe_edge_connectivity(
            edges=[
                {
                    "id": "edge_left",
                    "source": "endpoint_0",
                    "target": "endpoint_1",
                    "polyline": [{"row": 20, "col": 0}, {"row": 20, "col": 18}],
                },
                {
                    "id": "edge_right",
                    "source": "endpoint_2",
                    "target": "endpoint_3",
                    "polyline": [{"row": 20, "col": 22}, {"row": 20, "col": 40}],
                },
                {
                    "id": "edge_top_noise",
                    "source": "endpoint_4",
                    "target": "endpoint_5",
                    "polyline": [{"row": 0, "col": 20}, {"row": 14, "col": 20}],
                },
            ],
            node_clusters=[],
            object_regions=[
                {
                    "id": "obj_reducer",
                    "class_name": "reducer",
                    "bbox": {"x_min": 14, "y_min": 16, "x_max": 26, "y_max": 24},
                }
            ],
            inline_connector_classes=("arrow", "valve", "reducer"),
            inline_match_distance_px=8.0,
        )

        self.assertEqual(result["summary"]["inline_element_connection_count"], 1)
        self.assertEqual(
            tuple(sorted((result["connections"][0]["source_edge_id"], result["connections"][0]["target_edge_id"]))),
            ("edge_left", "edge_right"),
        )
        self.assertEqual(result["connections"][0]["alignment"], "horizontal")

    def test_build_pipe_edge_connectivity_connects_only_same_alignment_at_junction(self) -> None:
        result = build_pipe_edge_connectivity(
            edges=[
                {
                    "id": "edge_left",
                    "source": "endpoint_l",
                    "target": "junction_0",
                    "polyline": [{"row": 10, "col": 0}, {"row": 10, "col": 20}],
                },
                {
                    "id": "edge_right",
                    "source": "junction_0",
                    "target": "endpoint_r",
                    "polyline": [{"row": 10, "col": 20}, {"row": 10, "col": 40}],
                },
                {
                    "id": "edge_up",
                    "source": "endpoint_u",
                    "target": "junction_0",
                    "polyline": [{"row": 0, "col": 20}, {"row": 10, "col": 20}],
                },
                {
                    "id": "edge_down",
                    "source": "junction_0",
                    "target": "endpoint_d",
                    "polyline": [{"row": 10, "col": 20}, {"row": 30, "col": 20}],
                },
            ],
            node_clusters=[
                {"id": "junction_0", "kind": "junction", "centroid": {"x": 20.0, "y": 10.0}},
            ],
            object_regions=[],
            inline_connector_classes=("arrow", "valve", "reducer"),
            inline_match_distance_px=8.0,
        )

        pairs = {tuple(sorted((item["source_edge_id"], item["target_edge_id"]))) for item in result["connections"]}
        self.assertIn(("edge_left", "edge_right"), pairs)
        self.assertIn(("edge_down", "edge_up"), pairs)
        self.assertNotIn(("edge_left", "edge_up"), pairs)
        self.assertEqual(result["summary"]["junction_alignment_connection_count"], 2)

    def test_build_pipe_edge_connectivity_at_junction_keeps_only_best_opposite_pair_per_axis(self) -> None:
        result = build_pipe_edge_connectivity(
            edges=[
                {
                    "id": "edge_up",
                    "source": "endpoint_u",
                    "target": "junction_0",
                    "polyline": [{"row": 0, "col": 20}, {"row": 10, "col": 20}],
                },
                {
                    "id": "edge_down",
                    "source": "junction_0",
                    "target": "endpoint_d",
                    "polyline": [{"row": 10, "col": 20}, {"row": 30, "col": 20}],
                },
                {
                    "id": "edge_down_offset",
                    "source": "junction_0",
                    "target": "endpoint_d2",
                    "polyline": [{"row": 10, "col": 20}, {"row": 24, "col": 22}],
                },
            ],
            node_clusters=[
                {"id": "junction_0", "kind": "junction", "centroid": {"x": 20.0, "y": 10.0}},
            ],
            object_regions=[],
            inline_connector_classes=("arrow", "valve", "reducer"),
            inline_match_distance_px=8.0,
        )

        vertical_pairs = {
            tuple(sorted((item["source_edge_id"], item["target_edge_id"])))
            for item in result["connections"]
            if item["alignment"] == "vertical"
        }
        self.assertEqual(len(vertical_pairs), 1)
        self.assertIn(("edge_down", "edge_up"), vertical_pairs)
        self.assertNotIn(("edge_down_offset", "edge_up"), vertical_pairs)

    def test_build_pipe_edge_connectivity_adds_gap_continuation_for_broken_straight_run(self) -> None:
        result = build_pipe_edge_connectivity(
            edges=[
                {
                    "id": "edge_left",
                    "source": "endpoint_0",
                    "target": "endpoint_1",
                    "polyline": [{"row": 20, "col": 0}, {"row": 20, "col": 18}],
                },
                {
                    "id": "edge_right",
                    "source": "endpoint_2",
                    "target": "endpoint_3",
                    "polyline": [{"row": 20, "col": 22}, {"row": 20, "col": 40}],
                },
            ],
            node_clusters=[],
            object_regions=[],
            inline_connector_classes=("arrow", "valve", "reducer"),
            inline_match_distance_px=8.0,
        )

        gap_pairs = {
            tuple(sorted((item["source_edge_id"], item["target_edge_id"])))
            for item in result["connections"]
            if item["kind"] in {"gap_continuation", "connection_seeded_continuation"}
        }
        self.assertIn(("edge_left", "edge_right"), gap_pairs)
        self.assertEqual(result["summary"]["gap_continuation_connection_count"], 1)

    def test_build_pipe_edge_connectivity_allows_longer_gap_for_connection_seeded_edges(self) -> None:
        result = build_pipe_edge_connectivity(
            edges=[
                {
                    "id": "edge_seed",
                    "source": "endpoint_0",
                    "target": "endpoint_1",
                    "polyline": [{"row": 20, "col": 0}, {"row": 20, "col": 18}],
                },
                {
                    "id": "edge_far",
                    "source": "endpoint_2",
                    "target": "endpoint_3",
                    "polyline": [{"row": 20, "col": 57}, {"row": 20, "col": 80}],
                },
            ],
            node_clusters=[],
            object_regions=[],
            inline_connector_classes=("arrow", "valve", "reducer"),
            inline_match_distance_px=8.0,
            connection_seed_edge_ids={"edge_seed"},
        )

        seeded_pairs = {
            tuple(sorted((item["source_edge_id"], item["target_edge_id"])))
            for item in result["connections"]
            if item["kind"] == "connection_seeded_continuation"
        }
        self.assertIn(("edge_far", "edge_seed"), seeded_pairs)
        self.assertEqual(result["summary"]["connection_seeded_continuation_count"], 1)

    def test_build_pipe_edge_connectivity_propagates_connection_seeded_continuation(self) -> None:
        result = build_pipe_edge_connectivity(
            edges=[
                {
                    "id": "edge_seed",
                    "source": "endpoint_0",
                    "target": "endpoint_1",
                    "polyline": [{"row": 20, "col": 0}, {"row": 20, "col": 18}],
                },
                {
                    "id": "edge_mid",
                    "source": "endpoint_2",
                    "target": "endpoint_3",
                    "polyline": [{"row": 20, "col": 57}, {"row": 20, "col": 80}],
                },
                {
                    "id": "edge_farther",
                    "source": "endpoint_4",
                    "target": "endpoint_5",
                    "polyline": [{"row": 20, "col": 119}, {"row": 20, "col": 140}],
                },
            ],
            node_clusters=[],
            object_regions=[],
            inline_connector_classes=("arrow", "valve", "reducer"),
            inline_match_distance_px=8.0,
            connection_seed_edge_ids={"edge_seed"},
        )

        seeded_pairs = {
            tuple(sorted((item["source_edge_id"], item["target_edge_id"])))
            for item in result["connections"]
            if item["kind"] == "connection_seeded_continuation"
        }
        self.assertIn(("edge_mid", "edge_seed"), seeded_pairs)
        self.assertIn(("edge_farther", "edge_mid"), seeded_pairs)
        self.assertEqual(result["summary"]["connection_seeded_continuation_count"], 2)


if __name__ == "__main__":
    unittest.main()
