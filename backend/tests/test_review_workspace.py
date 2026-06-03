import json
import tempfile
import unittest
from pathlib import Path

from garnet.review_workspace import (
    build_workspace_from_artifacts,
    empty_review_workspace,
    load_review_workspace,
    save_review_workspace,
    workspace_to_stage3_equipment,
    workspace_to_stage4_objects,
)


class ReviewWorkspaceTests(unittest.TestCase):
    def test_empty_review_workspace_contains_all_sections(self) -> None:
        payload = empty_review_workspace("job_123")

        self.assertEqual(payload["job_id"], "job_123")
        self.assertEqual(payload["version"], 1)
        self.assertIn("updated_at", payload)
        for key in (
            "objects",
            "equipment",
            "manual_ports",
            "deleted_entities",
            "line_association_overrides",
            "trace_overrides",
        ):
            self.assertEqual(payload[key], [])

    def test_save_and_load_review_workspace_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_review_workspace(
                tmp,
                {
                    "job_id": "job_123",
                    "objects": [{"id": "obj_001", "class_name": "gate_valve"}],
                    "equipment": [{"id": "equip_001", "class_name": "vessel"}],
                },
            )

            self.assertTrue(saved.exists())
            loaded = load_review_workspace(tmp)
            self.assertEqual(loaded["job_id"], "job_123")
            self.assertEqual(loaded["objects"][0]["id"], "obj_001")
            self.assertEqual(loaded["equipment"][0]["id"], "equip_001")
            self.assertEqual(loaded["manual_ports"], [])

    def test_build_workspace_from_artifacts_preserves_stage_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            (job_dir / "stage4_objects.json").write_text(
                json.dumps(
                    {
                        "image_id": "Test-01.png",
                        "objects": [
                            {
                                "id": "obj_001",
                                "class_name": "gate_valve",
                                "bbox": {"x_min": 1, "y_min": 2, "x_max": 11, "y_max": 22},
                                "confidence": 0.7,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "stage3_equipment_bboxes.json").write_text(
                json.dumps(
                    {
                        "equipment": [
                            {
                                "id": "equip_001",
                                "class_name": "vessel",
                                "bbox": {"x_min": 30, "y_min": 40, "x_max": 130, "y_max": 240},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "stage5_connection_ports.json").write_text(
                json.dumps({"equipment": {"equip_001": [{"port_id": "p01", "x": 30, "y": 60}]}}),
                encoding="utf-8",
            )
            (job_dir / "stage6_line_number_review.json").write_text(
                json.dumps({"accepted": [{"trace_id": "obj_001", "text": "2-CUL-001"}]}),
                encoding="utf-8",
            )

            workspace = build_workspace_from_artifacts(job_dir)

            self.assertEqual(workspace["image_id"], "Test-01.png")
            self.assertEqual(workspace["objects"][0]["id"], "obj_001")
            self.assertEqual(workspace["equipment"][0]["id"], "equip_001")
            self.assertEqual(workspace["manual_ports"][0]["port_id"], "p01")
            self.assertEqual(workspace["line_association_overrides"][0]["trace_id"], "obj_001")

    def test_workspace_to_stage3_equipment_omits_rejected_equipment(self) -> None:
        artifact = workspace_to_stage3_equipment(
            {
                "equipment": [
                    {"id": "equip_001", "class_name": "vessel", "bbox": {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}},
                    {
                        "id": "equip_002",
                        "class_name": "pump",
                        "bbox": {"x_min": 5, "y_min": 6, "x_max": 7, "y_max": 8},
                        "review_state": "rejected",
                    },
                ]
            }
        )

        self.assertEqual(len(artifact["equipment"]), 1)
        self.assertEqual(artifact["equipment"][0]["id"], "equip_001")
        self.assertEqual(artifact["equipment"][0]["source"], "hitl")

    def test_workspace_to_stage4_objects_omits_rejected_objects(self) -> None:
        artifact = workspace_to_stage4_objects(
            {
                "objects": [
                    {"id": "obj_001", "class_name": "gate_valve", "bbox": {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}},
                    {
                        "id": "obj_002",
                        "class_name": "node",
                        "bbox": {"x_min": 5, "y_min": 6, "x_max": 7, "y_max": 8},
                        "review_state": "rejected",
                    },
                ]
            },
            image_id="Test-01.png",
        )

        self.assertEqual(artifact["image_id"], "Test-01.png")
        self.assertEqual(len(artifact["objects"]), 1)
        self.assertEqual(artifact["objects"][0]["id"], "obj_001")
        self.assertEqual(artifact["objects"][0]["source"], "hitl")


if __name__ == "__main__":
    unittest.main()
