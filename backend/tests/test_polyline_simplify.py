import unittest

from garnet.polyline_simplify import run_polyline_simplification_stage


def _point(row: int, col: int) -> dict[str, int]:
    return {"row": row, "col": col}


def _edge(edge_id: str, polyline: list[dict[str, int]]) -> dict[str, object]:
    return {
        "id": edge_id,
        "source": f"{edge_id}_source",
        "target": f"{edge_id}_target",
        "polyline": polyline,
        "pixel_length": len(polyline),
    }


class PolylineSimplifyTests(unittest.TestCase):
    def test_straight_line_collapses_to_endpoints(self) -> None:
        edge = _edge("edge_0", [_point(5, col) for col in range(100)])

        result = run_polyline_simplification_stage(edges=[edge], image_id="sample.png")

        simplified = result["edges_payload"]["edges"][0]["polyline"]
        self.assertEqual(simplified, [_point(5, 0), _point(5, 99)])
        self.assertEqual(result["edges_payload"]["edges"][0]["pixel_length"], 100)

    def test_corner_is_preserved(self) -> None:
        polyline = [_point(0, col) for col in range(6)] + [_point(row, 5) for row in range(1, 6)]
        edge = _edge("edge_0", polyline)

        result = run_polyline_simplification_stage(edges=[edge], image_id="sample.png", epsilon=0.5)

        simplified = result["edges_payload"]["edges"][0]["polyline"]
        self.assertEqual(simplified, [_point(0, 0), _point(0, 5), _point(5, 5)])

    def test_short_polyline_unchanged(self) -> None:
        polyline = [_point(1, 2), _point(3, 4)]
        edge = _edge("edge_0", polyline)

        result = run_polyline_simplification_stage(edges=[edge], image_id="sample.png")

        self.assertEqual(result["edges_payload"]["edges"][0]["polyline"], polyline)

    def test_epsilon_zero_preserves_all(self) -> None:
        polyline = [_point(0, col) for col in range(10)]
        edge = _edge("edge_0", polyline)

        result = run_polyline_simplification_stage(edges=[edge], image_id="sample.png", epsilon=0)

        self.assertEqual(result["edges_payload"]["edges"][0]["polyline"], polyline)

    def test_large_epsilon_collapses_everything(self) -> None:
        polyline = [_point(0, 0), _point(0, 5), _point(5, 5), _point(5, 10)]
        edge = _edge("edge_0", polyline)

        result = run_polyline_simplification_stage(edges=[edge], image_id="sample.png", epsilon=999)

        self.assertEqual(result["edges_payload"]["edges"][0]["polyline"], [_point(0, 0), _point(5, 10)])

    def test_multiple_edges_all_simplified(self) -> None:
        edges = [
            _edge("edge_0", [_point(0, col) for col in range(10)]),
            _edge("edge_1", [_point(row, 5) for row in range(12)]),
            _edge("edge_2", [_point(i, i) for i in range(8)]),
        ]

        result = run_polyline_simplification_stage(edges=edges, image_id="sample.png")

        simplified_edges = result["edges_payload"]["edges"]
        self.assertTrue(all(len(edge["polyline"]) == 2 for edge in simplified_edges))
        self.assertEqual(result["summary"]["edges_simplified_count"], 3)

    def test_simplified_pixel_length_field_added(self) -> None:
        edges = [_edge("edge_0", [_point(0, col) for col in range(10)])]

        result = run_polyline_simplification_stage(edges=edges, image_id="sample.png")

        edge = result["edges_payload"]["edges"][0]
        self.assertEqual(edge["pixel_length"], 10)
        self.assertEqual(edge["simplified_pixel_length"], 2)

    def test_compression_summary_accurate(self) -> None:
        edges = [
            _edge("edge_0", [_point(0, col) for col in range(10)]),
            _edge("edge_1", [_point(2, col) for col in range(4)]),
        ]

        result = run_polyline_simplification_stage(edges=edges, image_id="sample.png")

        summary = result["summary"]
        self.assertEqual(summary["total_original_point_count"], 14)
        self.assertEqual(summary["total_simplified_point_count"], 4)
        self.assertAlmostEqual(summary["compression_ratio"], 4 / 14)
        self.assertEqual(summary["edges_simplified_count"], 2)
        self.assertAlmostEqual(summary["mean_compression_per_edge"], ((2 / 10) + (2 / 4)) / 2)
        self.assertAlmostEqual(summary["median_compression_per_edge"], ((2 / 10) + (2 / 4)) / 2)
        self.assertEqual(summary["epsilon"], 2.0)


if __name__ == "__main__":
    unittest.main()
