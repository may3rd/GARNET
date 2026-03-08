import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from garnet import pid_extractor


class FakePipeline(pid_extractor.PIDPipeline):
    def __init__(self, out_dir: str | Path, fail_stage: int | None = None) -> None:
        super().__init__("image.png", out_dir=out_dir)
        self.called: list[str] = []
        self.fail_stage = fail_stage

    def _record(self, name: str) -> None:
        self.called.append(name)
        self._save_json(f"{name}_artifact", {"stage": name})
        if self.fail_stage is not None and name == f"stage{self.fail_stage}":
            raise RuntimeError(f"{name} failed")

    def stage1_input_normalization(self) -> None:
        self._record("stage1")

    def stage2_ocr_discovery(self) -> None:
        self._record("stage2")

    def stage4_object_detection(self) -> None:
        self._record("stage4")

    def stage5_pipe_mask(self) -> None:
        self._record("stage5")

class PIDPipelineRunnerTests(unittest.TestCase):
    def test_stage_definitions_follow_master_plan_order(self) -> None:
        pipe = FakePipeline(tempfile.mkdtemp())

        stage_names = [name for _, name, _ in pipe._stage_definitions()]

        self.assertEqual(
            stage_names,
            [
                "stage1_input_normalization",
                "stage2_ocr_discovery",
                "stage4_object_detection",
                "stage5_pipe_mask",
            ],
        )

    def test_run_stops_after_requested_stage_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            pipe.run(stop_after=5)

            self.assertEqual(pipe.called, ["stage1", "stage2", "stage4", "stage5"])
            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["stop_after"], 5)
            self.assertEqual(
                [item["name"] for item in manifest["stages"]],
                [
                    "stage1_input_normalization",
                    "stage2_ocr_discovery",
                    "stage4_object_detection",
                    "stage5_pipe_mask",
                ],
            )
            self.assertTrue(all(item["status"] == "completed" for item in manifest["stages"]))

    def test_run_rejects_stop_after_past_last_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            with self.assertRaisesRegex(ValueError, "stop_after must be one of"):
                pipe.run(stop_after=6)

    def test_run_writes_failed_stage_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp, fail_stage=2)

            with self.assertRaisesRegex(RuntimeError, "stage2 failed"):
                pipe.run(stop_after=5)

            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["stages"][0]["status"], "completed")
            self.assertEqual(manifest["stages"][1]["status"], "failed")
            self.assertIn("stage2 failed", manifest["stages"][1]["error"])

    def test_stage2_uses_plain_gray_artifact_as_ocr_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe._save_img("stage1_gray", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_img("stage1_gray_equalized", np.ones((20, 20), dtype=np.uint8) * 255)

            with patch("garnet.pid_extractor.run_easyocr_sahi") as mock_ocr:
                mock_ocr.return_value = {
                    "regions_payload": {"image_id": "", "pass_type": "sheet", "text_regions": []},
                    "summary": {"image_id": "", "pass_type": "sheet"},
                    "exception_candidates": [],
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                }

                pipe.stage2_ocr_discovery()

            self.assertEqual(Path(mock_ocr.call_args.args[0]).name, "stage1_gray.png")

    def test_pipeline_config_defaults_to_easyocr_route(self) -> None:
        cfg = pid_extractor.PipelineConfig()

        self.assertEqual(cfg.ocr_route, "easyocr")
        self.assertEqual(cfg.gemini_postprocess_match_threshold, 0.1)

    def test_load_pipeline_env_reads_root_then_backend_env(self) -> None:
        with patch("garnet.pid_extractor.load_dotenv") as mock_load_dotenv:
            pid_extractor.load_pipeline_env()

        self.assertEqual(mock_load_dotenv.call_count, 2)
        self.assertEqual(mock_load_dotenv.call_args_list[0].args[0], pid_extractor.ROOT_DIR / ".env")
        self.assertEqual(mock_load_dotenv.call_args_list[1].args[0], pid_extractor.BACKEND_DIR / ".env")
        self.assertTrue(all(call.kwargs["override"] is False for call in mock_load_dotenv.call_args_list))

    def test_stage2_dispatches_to_easyocr_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline(
                "image.png",
                out_dir=tmp,
                cfg=pid_extractor.PipelineConfig(ocr_route="easyocr"),
            )
            pipe._save_img("stage1_gray", np.zeros((20, 20), dtype=np.uint8))

            with patch("garnet.pid_extractor.run_easyocr_sahi") as mock_easyocr, patch(
                "garnet.pid_extractor.run_gemini_ocr_sahi"
            ) as mock_gemini:
                mock_easyocr.return_value = {
                    "regions_payload": {"image_id": "", "pass_type": "sheet", "text_regions": []},
                    "summary": {"image_id": "", "pass_type": "sheet"},
                    "exception_candidates": [],
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                }

                pipe.stage2_ocr_discovery()

            mock_easyocr.assert_called_once()
            mock_gemini.assert_not_called()

    def test_stage2_dispatches_to_gemini_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline(
                "image.png",
                out_dir=tmp,
                cfg=pid_extractor.PipelineConfig(
                    ocr_route="gemini",
                    gemini_postprocess_match_threshold=0.17,
                ),
            )
            pipe._save_img("stage1_gray", np.zeros((20, 20), dtype=np.uint8))

            with patch("garnet.pid_extractor.run_easyocr_sahi") as mock_easyocr, patch(
                "garnet.pid_extractor.run_gemini_ocr_sahi"
            ) as mock_gemini:
                mock_gemini.return_value = {
                    "regions_payload": {"image_id": "", "pass_type": "sheet", "text_regions": []},
                    "summary": {"image_id": "", "pass_type": "sheet"},
                    "exception_candidates": [],
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                }

                pipe.stage2_ocr_discovery()

            mock_easyocr.assert_not_called()
            mock_gemini.assert_called_once()
            self.assertEqual(mock_gemini.call_args.kwargs["cfg"].postprocess_match_threshold, 0.17)

    def test_stage4_writes_object_detection_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)

            with patch("garnet.pid_extractor.run_object_detection_sahi") as mock_detect:
                mock_detect.return_value = {
                    "objects_payload": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "objects": [
                            {
                                "id": "obj_000001",
                                "class_name": "valve",
                                "confidence": 0.91,
                                "bbox": {"x_min": 1, "y_min": 2, "x_max": 11, "y_max": 12},
                                "source_model": "ultralytics",
                                "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
                            }
                        ],
                    },
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "route": "ultralytics",
                        "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
                    },
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                }

                pipe.stage4_object_detection()

            mock_detect.assert_called_once()
            self.assertTrue((Path(tmp) / "stage4_objects.json").exists())
            self.assertTrue((Path(tmp) / "stage4_objects_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage4_objects_overlay.png").exists())
            summary = json.loads((Path(tmp) / "stage4_objects_summary.json").read_text())
            self.assertEqual(summary["source_weight"], "yolo_weights/yolo11n_PPCL_640_20250204.pt")

    def test_stage5_writes_pipe_mask_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_img("stage1_gray", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_img("stage1_binary_adaptive", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_img("stage1_binary_otsu", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_json("stage2_ocr_regions", {"text_regions": []})
            pipe._save_json("stage4_objects", {"objects": []})

            with patch("garnet.pid_extractor.run_pipe_mask_stage") as mock_pipe_mask:
                mock_pipe_mask.return_value = {
                    "mask_image": np.zeros((20, 20), dtype=np.uint8),
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "mask_pixel_count": 15,
                        "source_artifacts": [
                            "stage1_gray.png",
                            "stage2_ocr_regions.json",
                            "stage4_objects.json",
                        ],
                    },
                }

                pipe.stage5_pipe_mask()

            mock_pipe_mask.assert_called_once()
            self.assertTrue((Path(tmp) / "stage5_pipe_mask.png").exists())
            self.assertTrue((Path(tmp) / "stage5_pipe_mask_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage5_pipe_mask_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
