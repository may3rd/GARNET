import unittest

from garnet.topology_markers import run_topology_marker_router


class TopologyMarkerRouterTests(unittest.TestCase):
    def test_routes_only_arrow_and_node_classes(self) -> None:
        result = run_topology_marker_router(
            image_id="sample.png",
            objects=[
                {
                    "id": "obj_1",
                    "class_name": "arrow",
                    "confidence": 0.91,
                    "bbox": {"x_min": 1, "y_min": 2, "x_max": 11, "y_max": 12},
                },
                {
                    "id": "obj_2",
                    "class_name": "connection",
                    "confidence": 0.88,
                    "bbox": {"x_min": 4, "y_min": 5, "x_max": 14, "y_max": 15},
                },
                {
                    "id": "obj_3",
                    "class_name": "page connection",
                    "confidence": 0.87,
                    "bbox": {"x_min": 6, "y_min": 7, "x_max": 16, "y_max": 17},
                },
                {
                    "id": "obj_4",
                    "class_name": "utility connection",
                    "confidence": 0.86,
                    "bbox": {"x_min": 6, "y_min": 7, "x_max": 16, "y_max": 17},
                },
                {
                    "id": "obj_5",
                    "class_name": "node",
                    "confidence": 0.85,
                    "bbox": {"x_min": 7, "y_min": 8, "x_max": 17, "y_max": 18},
                },
                {
                    "id": "obj_6",
                    "class_name": "pump",
                    "confidence": 0.95,
                    "bbox": {"x_min": 10, "y_min": 11, "x_max": 20, "y_max": 21},
                },
            ],
        )

        markers = result["topology_markers_payload"]["topology_markers"]
        self.assertEqual(len(markers), 2)
        self.assertEqual(result["summary"]["class_counts"]["arrow"], 1)
        self.assertEqual(result["summary"]["class_counts"]["node"], 1)
        self.assertEqual(result["summary"]["role_counts"]["flow_marker"], 1)
        self.assertEqual(result["summary"]["role_counts"]["junction_marker"], 1)
        self.assertNotIn("connection", result["summary"]["class_counts"])
        self.assertNotIn("page connection", result["summary"]["class_counts"])
        self.assertNotIn("utility connection", result["summary"]["class_counts"])


if __name__ == "__main__":
    unittest.main()
