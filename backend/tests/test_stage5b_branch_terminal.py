import unittest

from garnet.pid_extractor import PIDPipeline


class Stage5bBranchTerminalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = PIDPipeline.__new__(PIDPipeline)

    def test_paired_node_terminal_is_not_promoted_when_far_from_traced_endpoint(self) -> None:
        branch = {
            "terminal_type": "branch_connection",
            "terminal_obj_id": "branch_000015",
            "terminal_x": 2903,
            "terminal_y": 1704,
            "segments": [
                {"x1": 2587, "y1": 1704, "x2": 2903, "y2": 1704, "direction": "RIGHT", "length_px": 316}
            ],
        }
        paired_candidate = {
            "id": "branch_000015",
            "x": 2908,
            "y": 853,
            "node_obj_id": "obj_000068",
        }

        promoted = self.pipeline._promote_stage5b_paired_node_terminal_if_attached(branch, paired_candidate)

        self.assertFalse(promoted)
        self.assertEqual(branch["terminal_type"], "branch_connection")
        self.assertEqual(branch["terminal_obj_id"], "branch_000015")
        self.assertEqual((branch["terminal_x"], branch["terminal_y"]), (2903, 1704))
        self.assertEqual((branch["segments"][-1]["x2"], branch["segments"][-1]["y2"]), (2903, 1704))

    def test_paired_node_terminal_is_promoted_when_attached_to_traced_endpoint(self) -> None:
        branch = {
            "terminal_type": "branch_connection",
            "terminal_obj_id": "branch_000002",
            "terminal_x": 100,
            "terminal_y": 50,
            "trace_length_px": 100,
            "segments": [
                {"x1": 0, "y1": 50, "x2": 100, "y2": 50, "direction": "RIGHT", "length_px": 100}
            ],
        }
        paired_candidate = {
            "id": "branch_000002",
            "x": 106,
            "y": 51,
            "node_obj_id": "obj_node",
        }

        promoted = self.pipeline._promote_stage5b_paired_node_terminal_if_attached(branch, paired_candidate)

        self.assertTrue(promoted)
        self.assertEqual(branch["terminal_type"], "tee_junction")
        self.assertEqual(branch["terminal_obj_id"], "obj_node")
        self.assertEqual((branch["terminal_x"], branch["terminal_y"]), (106, 51))
        self.assertEqual((branch["segments"][-1]["x2"], branch["segments"][-1]["y2"]), (106, 50))
        self.assertEqual(branch["segments"][-1]["length_px"], 106)
        self.assertEqual(branch["trace_length_px"], 106)


class Stage5bEquipmentBboxLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.pipeline = PIDPipeline.__new__(PIDPipeline)
        self.pipeline.out_dir = self.root / "out"
        self.pipeline.out_dir.mkdir()
        self.pipeline.image_path = str(self.root / "pid.png")
        (self.root / "pid.png").write_bytes(b"placeholder")

    def test_stage3_equipment_bboxes_take_priority_over_labelme_fallback(self) -> None:
        import json

        (self.root / "pid.json").write_text(
            json.dumps({
                "shapes": [
                    {"label": "pump", "points": [[1, 2], [11, 22]]},
                ]
            })
        )
        (self.pipeline.out_dir / "stage3_equipment_bboxes.json").write_text(
            json.dumps({
                "equipment": [
                    {
                        "id": "equip_hitl_001",
                        "class_name": "vessel",
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 220},
                        "review_state": "accepted",
                    }
                ]
            })
        )

        equipment = self.pipeline._load_equipment_bboxes_for_stage5b()

        self.assertEqual(len(equipment), 1)
        self.assertEqual(equipment[0]["id"], "equip_hitl_001")
        self.assertEqual(equipment[0]["source"], "hitl")
        self.assertEqual(equipment[0]["bbox"]["x_max"], 110)

    def test_labelme_equipment_bboxes_are_fallback_when_stage3_missing(self) -> None:
        import json

        (self.root / "pid.json").write_text(
            json.dumps({
                "shapes": [
                    {"label": "pump", "points": [[1, 2], [11, 22]]},
                    {"label": "not equipment", "points": [[100, 100], [120, 120]]},
                ]
            })
        )

        equipment = self.pipeline._load_equipment_bboxes_for_stage5b()

        self.assertEqual(len(equipment), 1)
        self.assertEqual(equipment[0]["class_name"], "pump")
        self.assertEqual(equipment[0]["source"], "labelme_fallback")
        self.assertEqual(equipment[0]["bbox"], {"x_min": 1, "y_min": 2, "x_max": 11, "y_max": 22})


if __name__ == "__main__":
    unittest.main()

class Stage5bIndividualTraceImageTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.pipeline = PIDPipeline.__new__(PIDPipeline)
        self.pipeline.out_dir = Path(self.tmpdir.name)
        self.pipeline._current_stage_artifacts = []

    def test_safe_stage5b_trace_image_name_replaces_path_unsafe_characters(self) -> None:
        self.assertEqual(
            self.pipeline._safe_stage5b_trace_image_name("equip_1_vessel:port_01"),
            "equip_1_vessel_port_01",
        )

    def test_write_stage5b_individual_trace_images_writes_only_segmented_traces(self) -> None:
        import numpy as np

        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        all_results = {
            "obj_000001": {
                "status": "ok",
                "port": {"x": 10, "y": 20, "direction": "RIGHT"},
                "terminal_type": "tee_junction",
                "terminal_x": 80,
                "terminal_y": 20,
                "segments": [
                    {"x1": 10, "y1": 20, "x2": 80, "y2": 20, "direction": "RIGHT", "length_px": 70}
                ],
                "turns": [],
                "trace_length_px": 70,
            },
            "equip_1_vessel:port_01": {
                "status": "ok",
                "port": {"x": 10, "y": 40, "direction": "RIGHT"},
                "terminal_type": "equipment",
                "terminal_x": 10,
                "terminal_y": 40,
                "segments": [],
                "turns": [],
                "trace_length_px": 0,
            },
        }
        branch_results = {
            "branch_000001": {
                "status": "traced",
                "port": {"x": 30, "y": 30, "direction": "DOWN"},
                "terminal_type": "dead_end",
                "terminal_x": 30,
                "terminal_y": 70,
                "segments": [
                    {"x1": 30, "y1": 30, "x2": 30, "y2": 70, "direction": "DOWN", "length_px": 40}
                ],
                "turns": [],
                "trace_length_px": 40,
            },
            "branch_000002": {"status": "skipped", "segments": []},
        }

        written = self.pipeline._write_stage5b_individual_trace_images(image, all_results, branch_results)

        trace_dir = self.pipeline.out_dir / "stage5b_traced_path"
        self.assertEqual(written, 2)
        self.assertTrue((trace_dir / "obj_000001.png").exists())
        self.assertTrue((trace_dir / "branch_000001.png").exists())
        self.assertFalse((trace_dir / "equip_1_vessel_port_01.png").exists())
        self.assertIn("stage5b_traced_path", self.pipeline._current_stage_artifacts)

class Stage5bBranchPairingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = PIDPipeline.__new__(PIDPipeline)

    def test_branch_connection_is_not_attached_to_far_candidate_start(self) -> None:
        result = {
            "terminal_type": "branch_connection",
            "terminal_x": 2903,
            "terminal_y": 1704,
            "terminal_obj_id": "branch_000015",
        }
        candidate = {
            "id": "branch_000015",
            "x": 2908,
            "y": 853,
        }

        self.assertFalse(
            self.pipeline._stage5b_branch_connection_attached_to_candidate(result, candidate)
        )

    def test_branch_connection_is_attached_to_near_candidate_start(self) -> None:
        result = {
            "terminal_type": "branch_connection",
            "terminal_x": 2903,
            "terminal_y": 1704,
            "terminal_obj_id": "branch_000029",
        }
        candidate = {
            "id": "branch_000029",
            "x": 2908,
            "y": 1704,
        }

        self.assertTrue(
            self.pipeline._stage5b_branch_connection_attached_to_candidate(result, candidate)
        )

class Stage5bBranchCandidateDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = PIDPipeline.__new__(PIDPipeline)

    def test_node_branch_candidate_is_done_when_direction_already_traced(self) -> None:
        import numpy as np

        mask = np.zeros((120, 140), dtype=np.uint8)
        mask[50, 0:101] = 255
        mask[0:101, 100] = 255
        all_results = {
            "branch_source": {
                "terminal_type": "equipment",
                "terminal_x": 100,
                "terminal_y": 100,
                "segments": [
                    {"x1": 100, "y1": 0, "x2": 100, "y2": 100, "direction": "DOWN", "length_px": 100}
                ],
                "turns": [],
            },
            "branch_existing": {
                "terminal_type": "equipment",
                "terminal_x": 0,
                "terminal_y": 50,
                "segments": [
                    {"x1": 0, "y1": 50, "x2": 100, "y2": 50, "direction": "RIGHT", "length_px": 100}
                ],
                "turns": [],
            },
        }
        node_symbols = [
            {"id": "node_1", "bbox": {"x_min": 96, "y_min": 46, "x_max": 104, "y_max": 54}}
        ]

        candidates = self.pipeline._detect_stage5b_branch_candidates(
            mask,
            all_results,
            inline_symbols=[],
            node_symbols=node_symbols,
            equipment_objects=[],
            min_branch_run=25,
        )

        left_candidates = [
            c for c in candidates
            if c.get("node_obj_id") == "node_1" and c.get("branch_direction") == "LEFT"
        ]
        self.assertEqual(len(left_candidates), 1)
        self.assertEqual(left_candidates[0]["status"], "done_already_traced")
        self.assertEqual(left_candidates[0]["reason"], "branch_direction_covered_by_existing_segment")

class Stage5bTurnMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = PIDPipeline.__new__(PIDPipeline)

    def test_rebuild_turns_uses_final_segments_only(self) -> None:
        segments = [
            {"x1": 1485, "y1": 792, "x2": 1070, "y2": 792, "direction": "LEFT", "length_px": 415},
            {"x1": 1070, "y1": 787, "x2": 1070, "y2": 777, "direction": "UP", "length_px": 10},
            {"x1": 1070, "y1": 739, "x2": 1070, "y2": 405, "direction": "UP", "length_px": 334},
        ]

        turns = self.pipeline._rebuild_stage5b_turns_from_segments(segments)

        self.assertEqual(turns, [{"x": 1070, "y": 792, "new_dir": "UP"}])
