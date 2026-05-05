import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.edge_split import split_edges_at_inline_elements


def _edge(edge_id: str, polyline: list[dict], **kwargs) -> dict:
    return {
        "id": edge_id,
        "source": f"{edge_id}_src",
        "target": f"{edge_id}_tgt",
        "polyline": polyline,
        "pixel_length": len(polyline),
        **kwargs,
    }


def _object(obj_id: str, class_name: str, bbox: dict) -> dict:
    return {"id": obj_id, "class_name": class_name, "bbox": bbox}


def _inline_connection(
    connector_id: str,
    connector_class: str,
    source_edge_id: str,
    target_edge_id: str,
    distance: float = 5.0,
) -> dict:
    return {
        "kind": "inline_element",
        "connector_id": connector_id,
        "connector_class": connector_class,
        "source_edge_id": source_edge_id,
        "target_edge_id": target_edge_id,
        "distance_px": distance,
        "inline_match_distance_px": 36.0,
    }


def _line_points(count: int = 11) -> list[dict]:
    return [{"row": 10, "col": col * 10} for col in range(count)]


class EdgeSplitTests(unittest.TestCase):
    def test_valve_on_single_edge_splits_edge(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e1")],
            objects=[_object("v1", "valve", {"x_min": 45, "y_min": 5, "x_max": 55, "y_max": 15})],
        )

        self.assertEqual(len(result["edges_payload"]["edges"]), 2)
        self.assertEqual(len(result["split_nodes"]), 1)
        self.assertEqual(result["split_report"][0]["status"], "split")

    def test_split_creates_correct_upstream_edge(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e1")],
            objects=[_object("v1", "valve", {"x_min": 45, "y_min": 5, "x_max": 55, "y_max": 15})],
        )

        upstream = next(edge for edge in result["edges_payload"]["edges"] if edge["split_position"] == "upstream")
        self.assertEqual(upstream["source"], "e1_src")
        self.assertEqual(upstream["target"], "inline::v1")
        self.assertEqual(upstream["inline_node_id"], "inline::v1")

    def test_split_creates_correct_downstream_edge(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e1")],
            objects=[_object("v1", "valve", {"x_min": 45, "y_min": 5, "x_max": 55, "y_max": 15})],
        )

        downstream = next(edge for edge in result["edges_payload"]["edges"] if edge["split_position"] == "downstream")
        self.assertEqual(downstream["source"], "inline::v1")
        self.assertEqual(downstream["target"], "e1_tgt")
        self.assertEqual(downstream["inline_node_id"], "inline::v1")

    def test_split_preserves_flow_direction_on_both_edges(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points(), flow_direction="forward", simplified_pixel_length=3)],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e1")],
            objects=[_object("v1", "valve", {"x_min": 45, "y_min": 5, "x_max": 55, "y_max": 15})],
        )

        for edge in result["edges_payload"]["edges"]:
            self.assertEqual(edge["flow_direction"], "forward")
            self.assertEqual(edge["simplified_pixel_length"], 3)

    def test_valve_between_two_different_edges_not_split(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points()), _edge("e2", _line_points())],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e2")],
            objects=[_object("v1", "valve", {"x_min": 45, "y_min": 5, "x_max": 55, "y_max": 15})],
        )

        self.assertEqual(len(result["edges_payload"]["edges"]), 2)
        self.assertEqual(result["split_report"][0]["status"], "skipped_already_connected")

    def test_low_confidence_valve_not_split(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e1", distance=30.0)],
            objects=[_object("v1", "valve", {"x_min": 45, "y_min": 5, "x_max": 55, "y_max": 15})],
            confidence_threshold=0.5,
        )

        self.assertEqual(len(result["edges_payload"]["edges"]), 1)
        self.assertEqual(result["split_report"][0]["status"], "low_confidence")

    def test_split_at_edge_start_skipped(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e1")],
            objects=[_object("v1", "valve", {"x_min": -5, "y_min": 5, "x_max": 5, "y_max": 15})],
        )

        self.assertEqual(result["split_report"][0]["status"], "skipped_edge_too_short")

    def test_split_at_edge_end_skipped(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e1")],
            objects=[_object("v1", "valve", {"x_min": 95, "y_min": 5, "x_max": 105, "y_max": 15})],
        )

        self.assertEqual(result["split_report"][0]["status"], "skipped_edge_too_short")

    def test_multiple_valves_on_same_edge_split_once(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[
                _inline_connection("v1", "valve", "e1", "e1"),
                _inline_connection("v2", "valve", "e1", "e1"),
            ],
            objects=[
                _object("v1", "valve", {"x_min": 25, "y_min": 5, "x_max": 35, "y_max": 15}),
                _object("v2", "valve", {"x_min": 65, "y_min": 5, "x_max": 75, "y_max": 15}),
            ],
        )

        self.assertEqual(result["summary"]["edges_split"], 2)
        self.assertEqual(result["summary"]["nodes_created"], 2)
        self.assertEqual(len(result["edges_payload"]["edges"]), 3)

    def test_summary_counts_correct(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points()), _edge("e2", _line_points()), _edge("e3", _line_points())],
            inline_connections=[
                _inline_connection("v1", "valve", "e1", "e1"),
                _inline_connection("v2", "valve", "e2", "e2"),
                _inline_connection("v3", "valve", "e3", "e3", distance=30.0),
            ],
            objects=[
                _object("v1", "valve", {"x_min": 25, "y_min": 5, "x_max": 35, "y_max": 15}),
                _object("v2", "valve", {"x_min": 65, "y_min": 5, "x_max": 75, "y_max": 15}),
                _object("v3", "valve", {"x_min": 45, "y_min": 5, "x_max": 55, "y_max": 15}),
            ],
        )

        self.assertEqual(result["summary"]["edges_split"], 2)
        self.assertEqual(result["summary"]["nodes_created"], 2)
        self.assertEqual(result["summary"]["skipped_low_confidence"], 1)

    def test_non_inline_connection_passed_through(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[{"kind": "junction_alignment", "source_edge_id": "e1", "target_edge_id": "e1"}],
            objects=[],
        )

        self.assertEqual(len(result["edges_payload"]["edges"]), 1)
        self.assertEqual(result["split_report"], [])

    def test_missing_object_for_connector_skipped(self) -> None:
        result = split_edges_at_inline_elements(
            edges=[_edge("e1", _line_points())],
            inline_connections=[_inline_connection("v1", "valve", "e1", "e1")],
            objects=[],
        )

        self.assertEqual(result["split_report"][0]["status"], "skipped_missing_object")


if __name__ == "__main__":
    unittest.main()
