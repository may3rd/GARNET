import unittest

from garnet.pipe_equipment_attachment import run_pipe_equipment_attachment_stage


class PipeEquipmentAttachmentTests(unittest.TestCase):
    def test_run_pipe_equipment_attachment_stage_accepts_simple_pump_case(self) -> None:
        objects = [
            {
                "id": "obj_1",
                "class_name": "pump",
                "confidence": 0.9,
                "bbox": {"x_min": 5, "y_min": 5, "x_max": 15, "y_max": 15},
            }
        ]
        edges = [
            {
                "id": "edge_1",
                "source": "n1",
                "target": "n2",
                "polyline": [
                    {"row": 10, "col": 0},
                    {"row": 10, "col": 20},
                ],
            }
        ]

        result = run_pipe_equipment_attachment_stage(
            image_id="sample.png",
            objects=objects,
            edges=edges,
            attachment_classes=("pump",),
            max_distance_px=20.0,
            k_candidate_edges=4,
        )

        self.assertEqual(result["summary"]["equipment_candidates"], 1)
        self.assertEqual(result["summary"]["accepted_attachment_count"], 1)
        self.assertEqual(len(result["attachments_payload"]["accepted"]), 1)
        attachment = result["attachments_payload"]["accepted"][0]
        self.assertIsNotNone(attachment["connection_anchor_xy"])
        self.assertIsNotNone(attachment["attachment_stub_xy"])


if __name__ == "__main__":
    unittest.main()
