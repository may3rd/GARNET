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

    def stage6_morphological_sealing(self) -> None:
        self._record("stage6")

    def stage7_skeleton_generation(self) -> None:
        self._record("stage7")

    def stage8_skeleton_node_detection(self) -> None:
        self._record("stage8")

    def stage9_node_clustering(self) -> None:
        self._record("stage9")

    def stage10_edge_tracing(self) -> None:
        self._record("stage10")

    def stage11_junction_review(self) -> None:
        self._record("stage11")

    def stage12_graph_assembly(self) -> None:
        self._record("stage12")

    def stage13_graph_qa(self) -> None:
        self._record("stage13")

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
                "stage6_morphological_sealing",
                "stage7_skeleton_generation",
                "stage8_skeleton_node_detection",
                "stage9_node_clustering",
                "stage10_edge_tracing",
                "stage11_junction_review",
                "stage12_graph_assembly",
                "stage13_graph_qa",
            ],
        )

    def test_run_stops_after_requested_stage_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            pipe.run(stop_after=13)

            self.assertEqual(pipe.called, ["stage1", "stage2", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9", "stage10", "stage11", "stage12", "stage13"])
            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["stop_after"], 13)
            self.assertEqual(
                [item["name"] for item in manifest["stages"]],
                [
                    "stage1_input_normalization",
                    "stage2_ocr_discovery",
                    "stage4_object_detection",
                    "stage5_pipe_mask",
                    "stage6_morphological_sealing",
                    "stage7_skeleton_generation",
                    "stage8_skeleton_node_detection",
                    "stage9_node_clustering",
                    "stage10_edge_tracing",
                    "stage11_junction_review",
                    "stage12_graph_assembly",
                    "stage13_graph_qa",
                ],
            )
            self.assertTrue(all(item["status"] == "completed" for item in manifest["stages"]))

    def test_run_rejects_stop_after_past_last_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            with self.assertRaisesRegex(ValueError, "stop_after must be one of"):
                pipe.run(stop_after=14)

    def test_run_writes_failed_stage_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp, fail_stage=2)

            with self.assertRaisesRegex(RuntimeError, "stage2 failed"):
                pipe.run(stop_after=13)

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

    def test_stage6_writes_pipe_seal_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_img("stage5_pipe_mask", np.zeros((20, 20), dtype=np.uint8))

            with patch("garnet.pid_extractor.run_pipe_seal_stage") as mock_pipe_seal:
                mock_pipe_seal.return_value = {
                    "sealed_mask_image": np.zeros((20, 20), dtype=np.uint8),
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "mask_pixel_count": 15,
                        "source_artifacts": ["stage5_pipe_mask.png"],
                    },
                }

                pipe.stage6_morphological_sealing()

            mock_pipe_seal.assert_called_once()
            self.assertTrue((Path(tmp) / "stage6_pipe_mask_sealed.png").exists())
            self.assertTrue((Path(tmp) / "stage6_pipe_mask_sealed_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage6_pipe_mask_sealed_summary.json").exists())

    def test_stage7_writes_skeleton_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_img("stage6_pipe_mask_sealed", np.zeros((20, 20), dtype=np.uint8))

            with patch("garnet.pid_extractor.run_pipe_skeleton_stage") as mock_pipe_skeleton:
                mock_pipe_skeleton.return_value = {
                    "skeleton_image": np.zeros((20, 20), dtype=np.uint8),
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "skeleton_pixel_count": 12,
                        "source_artifacts": ["stage6_pipe_mask_sealed.png"],
                    },
                }

                pipe.stage7_skeleton_generation()

            mock_pipe_skeleton.assert_called_once()
            self.assertTrue((Path(tmp) / "stage7_pipe_skeleton.png").exists())
            self.assertTrue((Path(tmp) / "stage7_pipe_skeleton_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage7_pipe_skeleton_summary.json").exists())

    def test_stage8_writes_node_detection_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_img("stage7_pipe_skeleton", np.zeros((20, 20), dtype=np.uint8))

            with patch("garnet.pid_extractor.run_pipe_node_stage") as mock_pipe_nodes:
                mock_pipe_nodes.return_value = {
                    "endpoint_image": np.zeros((20, 20), dtype=np.uint8),
                    "junction_image": np.zeros((20, 20), dtype=np.uint8),
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "endpoint_count": 4,
                        "junction_count": 1,
                    },
                }

                pipe.stage8_skeleton_node_detection()

            mock_pipe_nodes.assert_called_once()
            self.assertTrue((Path(tmp) / "stage8_endpoints.png").exists())
            self.assertTrue((Path(tmp) / "stage8_junctions.png").exists())
            self.assertTrue((Path(tmp) / "stage8_nodes_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage8_node_summary.json").exists())

    def test_stage9_writes_node_cluster_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_img("stage8_endpoints", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_img("stage8_junctions", np.zeros((20, 20), dtype=np.uint8))

            with patch("garnet.pid_extractor.run_pipe_node_cluster_stage") as mock_pipe_clusters:
                mock_pipe_clusters.return_value = {
                    "endpoint_cluster_image": np.zeros((20, 20), dtype=np.uint8),
                    "junction_cluster_image": np.zeros((20, 20), dtype=np.uint8),
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "clusters_payload": {"clusters": []},
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "endpoint_cluster_count": 2,
                        "junction_cluster_count": 1,
                    },
                }

                pipe.stage9_node_clustering()

            mock_pipe_clusters.assert_called_once()
            self.assertTrue((Path(tmp) / "stage9_endpoint_clusters.png").exists())
            self.assertTrue((Path(tmp) / "stage9_junction_clusters.png").exists())
            self.assertTrue((Path(tmp) / "stage9_node_clusters_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage9_node_clusters.json").exists())
            self.assertTrue((Path(tmp) / "stage9_node_cluster_summary.json").exists())

    def test_stage10_writes_edge_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_img("stage7_pipe_skeleton", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_json("stage9_node_clusters", {"clusters": []})

            with patch("garnet.pid_extractor.run_pipe_edge_stage") as mock_pipe_edges:
                mock_pipe_edges.return_value = {
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "edges_payload": {"edges": []},
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "edge_count": 0,
                    },
                }

                pipe.stage10_edge_tracing()

            mock_pipe_edges.assert_called_once()
            self.assertTrue((Path(tmp) / "stage10_pipe_edges_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage10_pipe_edges.json").exists())
            self.assertTrue((Path(tmp) / "stage10_pipe_edge_summary.json").exists())

    def test_stage11_writes_junction_review_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_img("stage7_pipe_skeleton", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_json("stage9_node_clusters", {"clusters": []})

            with patch("garnet.pid_extractor.run_pipe_junction_stage") as mock_pipe_junctions:
                mock_pipe_junctions.return_value = {
                    "confirmed_junction_image": np.zeros((20, 20), dtype=np.uint8),
                    "unresolved_junction_image": np.zeros((20, 20), dtype=np.uint8),
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "junctions_payload": {"confirmed_junctions": [], "unresolved_junctions": []},
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "confirmed_junction_count": 0,
                        "unresolved_junction_count": 0,
                    },
                }

                pipe.stage11_junction_review()

            mock_pipe_junctions.assert_called_once()
            self.assertTrue((Path(tmp) / "stage11_confirmed_junctions.png").exists())
            self.assertTrue((Path(tmp) / "stage11_unresolved_junctions.png").exists())
            self.assertTrue((Path(tmp) / "stage11_junction_review_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage11_junctions.json").exists())
            self.assertTrue((Path(tmp) / "stage11_junction_review_summary.json").exists())

    def test_stage12_writes_graph_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe._save_json("stage4_objects", {"objects": []})
            pipe._save_json("stage2_ocr_regions", {"text_regions": []})
            pipe._save_json("stage9_node_clusters", {"clusters": []})
            pipe._save_json("stage10_pipe_edges", {"edges": []})
            pipe._save_json("stage11_junctions", {"confirmed_junctions": [], "unresolved_junctions": []})

            with patch("garnet.pid_extractor.run_pipe_equipment_attachment_stage") as mock_pipe_attachment, patch(
                "garnet.pid_extractor.run_pipe_text_attachment_stage"
            ) as mock_pipe_text_attachment, patch(
                "garnet.pid_extractor.run_pipe_graph_stage"
            ) as mock_pipe_graph:
                mock_pipe_attachment.return_value = {
                    "attachments_payload": {"accepted": [], "rejected": []},
                    "summary": {"accepted_attachment_count": 0},
                }
                mock_pipe_text_attachment.return_value = {
                    "attachments_payload": {"accepted": [], "rejected": []},
                    "summary": {"accepted_attachment_count": 0},
                }
                mock_pipe_graph.return_value = {
                    "graph_payload": {"nodes": [], "edges": []},
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "node_count": 0,
                        "edge_count": 0,
                    },
                }

                pipe.stage12_graph_assembly()

            mock_pipe_attachment.assert_called_once()
            mock_pipe_text_attachment.assert_called_once()
            mock_pipe_graph.assert_called_once()
            self.assertTrue((Path(tmp) / "stage12_equipment_attachments.json").exists())
            self.assertTrue((Path(tmp) / "stage12_equipment_attachment_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_text_attachments.json").exists())
            self.assertTrue((Path(tmp) / "stage12_text_attachment_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_graph.json").exists())
            self.assertTrue((Path(tmp) / "stage12_graph_summary.json").exists())

    def test_stage13_writes_graph_qa_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe._save_json("stage12_graph", {"nodes": [], "edges": []})

            with patch("garnet.pid_extractor.run_pipe_graph_qa_stage") as mock_pipe_graph_qa:
                mock_pipe_graph_qa.return_value = {
                    "anomaly_report": {"connected_component_count": 0},
                    "review_queue": {"items": []},
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "review_queue_count": 0,
                    },
                }

                pipe.stage13_graph_qa()

            mock_pipe_graph_qa.assert_called_once()
            self.assertTrue((Path(tmp) / "stage13_graph_anomalies.json").exists())
            self.assertTrue((Path(tmp) / "stage13_review_queue.json").exists())
            self.assertTrue((Path(tmp) / "stage13_graph_qa_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
