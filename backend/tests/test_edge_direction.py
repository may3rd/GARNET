import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.edge_direction import compute_arrow_direction, run_edge_direction_stage


def _horizontal_edge(edge_id: str, row: int = 10, start: int = 0, end: int = 100) -> dict:
    return {
        "id": edge_id,
        "source": "n1",
        "target": "n2",
        "pixel_length": abs(end - start) + 1,
        "polyline": [{"row": row, "col": col} for col in range(start, end + 1, 10)],
    }


def _vertical_edge(edge_id: str, col: int = 10, start: int = 0, end: int = 100) -> dict:
    return {
        "id": edge_id,
        "source": "n1",
        "target": "n2",
        "pixel_length": abs(end - start) + 1,
        "polyline": [{"row": row, "col": col} for row in range(start, end + 1, 10)],
    }


class EdgeDirectionTests(unittest.TestCase):
    def test_horizontal_arrow_rightward(self) -> None:
        direction = compute_arrow_direction(
            {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 20, "direction": "right"}
        )

        self.assertEqual(direction, "forward")

    def test_horizontal_arrow_leftward(self) -> None:
        direction = compute_arrow_direction(
            {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 20, "direction": "left"}
        )

        self.assertEqual(direction, "reverse")

    def test_vertical_arrow_downward(self) -> None:
        direction = compute_arrow_direction(
            {"x_min": 10, "y_min": 10, "x_max": 20, "y_max": 60, "direction": "down"}
        )

        self.assertEqual(direction, "forward")

    def test_vertical_arrow_upward(self) -> None:
        direction = compute_arrow_direction(
            {"x_min": 10, "y_min": 10, "x_max": 20, "y_max": 60, "direction": "up"}
        )

        self.assertEqual(direction, "reverse")

    def test_ambiguous_aspect_returns_forward(self) -> None:
        # Square arrows (aspect=1.0) are classified as vertical and canonical forward.
        # This is intentional: portrait/square arrows on pipes represent downward
        # or forward flow, and should be assigned a direction, not skipped.
        direction = compute_arrow_direction({"x_min": 10, "y_min": 10, "x_max": 30, "y_max": 30})
        self.assertEqual(direction, "forward")

    def test_diagonal_aspect_returns_forward(self) -> None:
        # Arrows with diagonal aspect (1.0 < aspect < 1.5) are classified as diagonal
        # and assigned canonical forward direction.
        direction = compute_arrow_direction({"x_min": 10, "y_min": 10, "x_max": 40, "y_max": 30})
        self.assertEqual(direction, "forward")

    def test_edge_with_nearby_arrow_gets_direction(self) -> None:
        result = run_edge_direction_stage(
            edges=[_horizontal_edge("e1")],
            objects=[
                {
                    "id": "a1",
                    "class_name": "arrow",
                    "bbox": {"x_min": 40, "y_min": 5, "x_max": 70, "y_max": 15},
                    "direction": "right",
                }
            ],
            image_id="sample.png",
        )

        edge = result["edges_payload"]["edges"][0]
        self.assertEqual(edge["flow_direction"], "forward")
        self.assertEqual(edge["assigned_arrow_id"], "a1")
        self.assertGreater(edge["flow_direction_confidence"], 0.0)

    def test_edge_without_nearby_arrow_gets_null(self) -> None:
        result = run_edge_direction_stage(
            edges=[_horizontal_edge("e1")],
            objects=[
                {
                    "id": "a1",
                    "class_name": "arrow",
                    "bbox": {"x_min": 200, "y_min": 200, "x_max": 240, "y_max": 215},
                    "direction": "right",
                }
            ],
            image_id="sample.png",
        )

        edge = result["edges_payload"]["edges"][0]
        self.assertIsNone(edge["flow_direction"])
        self.assertIsNone(edge["assigned_arrow_id"])
        self.assertEqual(edge["flow_direction_confidence"], 0.0)

    def test_multiple_arrows_picks_closest(self) -> None:
        result = run_edge_direction_stage(
            edges=[_horizontal_edge("e1")],
            objects=[
                {
                    "id": "far",
                    "class_name": "arrow",
                    "bbox": {"x_min": 40, "y_min": 38, "x_max": 70, "y_max": 48},
                    "direction": "left",
                },
                {
                    "id": "near",
                    "class_name": "arrow",
                    "bbox": {"x_min": 40, "y_min": 5, "x_max": 70, "y_max": 15},
                    "direction": "right",
                },
            ],
            image_id="sample.png",
            arrow_proximity_px=40.0,
        )

        edge = result["edges_payload"]["edges"][0]
        self.assertEqual(edge["assigned_arrow_id"], "near")
        self.assertEqual(edge["flow_direction"], "forward")

    def test_summary_counts_accurate(self) -> None:
        result = run_edge_direction_stage(
            edges=[
                _horizontal_edge("e1", row=10),
                _horizontal_edge("e2", row=50),
                _horizontal_edge("e3", row=100),
            ],
            objects=[
                {
                    "id": "a1",
                    "class_name": "arrow",
                    "bbox": {"x_min": 40, "y_min": 5, "x_max": 70, "y_max": 15},
                    "direction": "right",
                },
                {
                    "id": "a2",
                    "class_name": "arrow",
                    "bbox": {"x_min": 40, "y_min": 45, "x_max": 70, "y_max": 55},
                    "direction": "left",
                },
            ],
            image_id="sample.png",
        )

        summary = result["summary"]
        self.assertEqual(summary["total_edges"], 3)
        self.assertEqual(summary["edges_with_forward_direction"], 1)
        self.assertEqual(summary["edges_with_reverse_direction"], 1)
        self.assertEqual(summary["edges_without_direction"], 1)
        self.assertEqual(summary["arrows_assigned_to_edge"], 2)
        self.assertEqual(summary["arrows_unassigned"], 0)

    def test_vertical_edge_with_horizontal_arrow_handled(self) -> None:
        result = run_edge_direction_stage(
            edges=[_vertical_edge("e1")],
            objects=[
                {
                    "id": "a1",
                    "class_name": "arrow",
                    "bbox": {"x_min": 5, "y_min": 40, "x_max": 45, "y_max": 50},
                    "direction": "right",
                }
            ],
            image_id="sample.png",
        )

        edge = result["edges_payload"]["edges"][0]
        self.assertIsNone(edge["flow_direction"])
        self.assertIsNone(edge["assigned_arrow_id"])


if __name__ == "__main__":
    unittest.main()
