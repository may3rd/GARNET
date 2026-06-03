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
    workspace_to_stage5_ports,
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
                            },
                            {
                                "id": "obj_pump",
                                "class_name": "pump",
                                "bbox": {"x_min": 40, "y_min": 50, "x_max": 90, "y_max": 120},
                                "confidence": 0.9,
                            },
                            {
                                "id": "obj_line",
                                "class_name": "line_number",
                                "bbox": {"x_min": 140, "y_min": 150, "x_max": 190, "y_max": 170},
                                "confidence": 0.8,
                            },
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
                json.dumps({"equip_001": [[30, 60, "RIGHT"]]}),
                encoding="utf-8",
            )
            (job_dir / "stage6_line_number_review.json").write_text(
                json.dumps({"accepted": [{"trace_id": "obj_001", "text": "2-CUL-001"}]}),
                encoding="utf-8",
            )

            workspace = build_workspace_from_artifacts(job_dir)

            self.assertEqual(workspace["image_id"], "Test-01.png")
            self.assertEqual(workspace["objects"][0]["id"], "obj_001")
            self.assertEqual(len(workspace["objects"]), 1)
            self.assertEqual(workspace["equipment"][0]["id"], "equip_001")
            self.assertEqual(workspace["manual_ports"][0]["port_id"], "equip_001:port_01")
            self.assertEqual(workspace["line_association_overrides"][0]["trace_id"], "obj_001")

    def test_build_workspace_from_artifacts_moves_stage4_pumps_to_equipment_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            (job_dir / "stage4_objects.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {"id": "obj_pump", "class_name": "pump", "bbox": {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}},
                            {"id": "obj_line", "class_name": "line number", "bbox": {"x_min": 5, "y_min": 6, "x_max": 7, "y_max": 8}},
                            {"id": "obj_valve", "class_name": "gate_valve", "bbox": {"x_min": 9, "y_min": 10, "x_max": 11, "y_max": 12}},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            workspace = build_workspace_from_artifacts(job_dir)

            self.assertEqual([item["id"] for item in workspace["equipment"]], ["obj_pump"])
            self.assertEqual([item["id"] for item in workspace["objects"]], ["obj_valve"])

    def test_workspace_to_stage5_ports_exports_reviewed_manual_ports(self) -> None:
        artifact = workspace_to_stage5_ports(
            {
                "manual_ports": [
                    {
                        "port_id": "equip_001:port_01",
                        "owner_id": "equip_001",
                        "owner_type": "equipment",
                        "x": 30.4,
                        "y": 60.2,
                        "direction": "RIGHT",
                    },
                    {
                        "port_id": "equip_001:port_02",
                        "owner_id": "equip_001",
                        "owner_type": "equipment",
                        "x": 40,
                        "y": 70,
                        "direction": "LEFT",
                        "review_state": "rejected",
                    },
                    {"owner_id": "equip_002", "x": 1, "y": 2, "direction": "diagonal"},
                ]
            }
        )

        self.assertEqual(artifact, {"equip_001": [[30, 60, "RIGHT"]]})

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
                    {"id": "obj_pump", "class_name": "pump", "bbox": {"x_min": 9, "y_min": 10, "x_max": 11, "y_max": 12}},
                    {"id": "obj_line", "class_name": "line_number", "bbox": {"x_min": 13, "y_min": 14, "x_max": 15, "y_max": 16}},
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
