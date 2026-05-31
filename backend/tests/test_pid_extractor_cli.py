import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    def stage4_line_number_fusion(self) -> None:
        self._record("stage4")

    def stage4_instrument_tag_fusion(self) -> None:
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

    def stage10b_polyline_simplification(self) -> None:
        self._record("stage10b")

    def stage10c_edge_direction(self) -> None:
        self._record("stage10c")

    def stage10d_edge_split(self) -> None:
        self._record("stage10d")

    def stage11_junction_review(self) -> None:
        self._record("stage11")

    def stage12_graph_assembly(self) -> None:
        self._record("stage12")

    def stage12c_page_connector_labeling(self) -> None:
        self._record("stage12c")

    def stage12b_graph_export(self) -> None:
        self._record("stage12b")

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
                "stage4_line_number_fusion",
                "stage4_instrument_tag_fusion",
                "stage4_equipment_tag_fusion",
                "stage5_pipe_mask",
                "stage6_morphological_sealing",
                "stage7_skeleton_generation",
                "stage8_skeleton_node_detection",
                "stage9_node_clustering",
                "stage10_edge_tracing",
                "stage10b_polyline_simplification",
                "stage10c_edge_direction",
                "stage10d_edge_split",
                "stage11_junction_review",
                "stage12_graph_assembly",
                "stage12c_page_connector_labeling",
                "stage12b_graph_export",
                "stage13_graph_qa",
            ],
        )

    def test_run_stops_after_requested_stage_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            pipe.run(stop_after=13)

            self.assertEqual(
                pipe.called,
                [
                    "stage1",
                    "stage2",
                    "stage4",
                    "stage4",
                    "stage4",
                    "stage5",
                    "stage6",
                    "stage7",
                    "stage8",
                    "stage9",
                    "stage10",
                    "stage10b",
                    "stage10c",
                    "stage10d",
                    "stage11",
                    "stage12",
                    "stage12c",
                    "stage12b",
                    "stage13",
                ],
            )
            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["stop_after"], 13)
            self.assertEqual(
                [item["name"] for item in manifest["stages"]],
                [
                    "stage1_input_normalization",
                    "stage2_ocr_discovery",
                    "stage4_object_detection",
                    "stage4_line_number_fusion",
                    "stage4_instrument_tag_fusion",
                    "stage4_equipment_tag_fusion",
                    "stage5_pipe_mask",
                    "stage6_morphological_sealing",
                    "stage7_skeleton_generation",
                    "stage8_skeleton_node_detection",
                    "stage9_node_clustering",
                    "stage10_edge_tracing",
                    "stage10b_polyline_simplification",
                    "stage10c_edge_direction",
                    "stage10d_edge_split",
                    "stage11_junction_review",
                    "stage12_graph_assembly",
                    "stage12c_page_connector_labeling",
                    "stage12b_graph_export",
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
            pipe = pid_extractor.PIDPipeline(
                "image.png",
                out_dir=tmp,
                cfg=pid_extractor.PipelineConfig(ocr_route="easyocr"),
            )
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

    def test_pipeline_config_defaults_to_ocrmac_route(self) -> None:
        cfg = pid_extractor.PipelineConfig()

        self.assertEqual(cfg.ocr_route, "ocrmac")
        self.assertEqual(cfg.gemini_postprocess_match_threshold, 0.1)
        self.assertEqual(cfg.polyline_simplify_epsilon, 2.0)
        self.assertEqual(cfg.arrow_proximity_px, 40.0)
        self.assertEqual(cfg.inline_split_confidence_threshold, 0.5)

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

    def test_stage2_dispatches_to_ocrmac_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline(
                "image.png",
                out_dir=tmp,
                cfg=pid_extractor.PipelineConfig(ocr_route="ocrmac"),
            )
            pipe._save_img("stage1_gray", np.zeros((20, 20), dtype=np.uint8))

            with patch("garnet.pid_extractor.run_ocrmac_sahi") as mock_ocrmac:
                mock_ocrmac.return_value = {
                    "regions_payload": {"image_id": "", "pass_type": "sheet", "text_regions": []},
                    "summary": {"image_id": "", "pass_type": "sheet"},
                    "exception_candidates": [],
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                }

                pipe.stage2_ocr_discovery()

            mock_ocrmac.assert_called_once()
            self.assertTrue(mock_ocrmac.call_args.kwargs["cfg"].enable_rotated_ocr)

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
                                "class_name": "arrow",
                                "confidence": 0.91,
                                "bbox": {"x_min": 1, "y_min": 2, "x_max": 11, "y_max": 12},
                                "source_model": "ultralytics",
                                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
                            }
                        ],
                    },
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "route": "ultralytics",
                        "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
                    },
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                }

                pipe.stage4_object_detection()

            mock_detect.assert_called_once()
            self.assertTrue((Path(tmp) / "stage4_objects.json").exists())
            self.assertTrue((Path(tmp) / "stage4_objects_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage4_objects_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage4_topology_markers.json").exists())
            self.assertTrue((Path(tmp) / "stage4_topology_marker_summary.json").exists())
            summary = json.loads((Path(tmp) / "stage4_objects_summary.json").read_text())
            topology_summary = json.loads((Path(tmp) / "stage4_topology_marker_summary.json").read_text())
            self.assertEqual(summary["source_weight"], "yolo_weights/yolo26n_PPCL_640_20260227.pt")
            self.assertEqual(topology_summary["topology_marker_count"], 1)

    def test_stage4_line_number_fusion_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_json("stage4_objects", {"objects": []})
            pipe._save_json("stage2_ocr_regions", {"text_regions": []})

            with patch("garnet.pid_extractor.run_line_number_fusion_stage") as mock_line_fusion:
                mock_line_fusion.return_value = {
                    "line_numbers_payload": {"line_numbers": [], "rejected": []},
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "summary": {"matched_line_number_count": 0},
                }

                pipe.stage4_line_number_fusion()

            mock_line_fusion.assert_called_once()
            self.assertTrue((Path(tmp) / "stage4_line_numbers.json").exists())
            self.assertTrue((Path(tmp) / "stage4_line_number_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage4_line_number_overlay.png").exists())

    def test_stage4_instrument_tag_fusion_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_json("stage4_objects", {"objects": []})
            pipe._save_json("stage2_ocr_regions", {"text_regions": []})

            with patch("garnet.pid_extractor.run_instrument_tag_fusion_stage") as mock_tag_fusion:
                mock_tag_fusion.return_value = {
                    "instrument_tags_payload": {"instrument_tags": [], "rejected": []},
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "summary": {"matched_instrument_tag_count": 0},
                }

                pipe.stage4_instrument_tag_fusion()

            mock_tag_fusion.assert_called_once()
            self.assertTrue((Path(tmp) / "stage4_instrument_tags.json").exists())
            self.assertTrue((Path(tmp) / "stage4_instrument_tag_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage4_instrument_tag_overlay.png").exists())

    def test_stage4_equipment_tag_fusion_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_json(
                "stage2_ocr_regions",
                {
                    "text_regions": [
                        {
                            "id": "ocr_1",
                            "text": "V-101",
                            "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 30},
                            "confidence": 0.9,
                        }
                    ]
                },
            )

            with patch("garnet.pid_extractor.run_equipment_tag_fusion_stage") as mock_tag_fusion:
                mock_tag_fusion.return_value = {
                    "equipment_tags_payload": {"equipment_tags": [], "rejected": []},
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "summary": {"matched_equipment_tag_count": 0},
                }

                pipe.stage4_equipment_tag_fusion()

            mock_tag_fusion.assert_called_once()
            self.assertTrue((Path(tmp) / "stage4_equipment_tags.json").exists())
            self.assertTrue((Path(tmp) / "stage4_equipment_tag_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage4_equipment_tag_overlay.png").exists())

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
            self.assertEqual(
                mock_pipe_mask.call_args.kwargs["preserve_ocr_classes"],
                pipe.cfg.pipe_mask_preserve_ocr_classes,
            )
            self.assertEqual(
                mock_pipe_mask.call_args.kwargs["preserve_object_classes"],
                pipe.cfg.pipe_mask_preserve_object_classes,
            )
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
            pipe._save_img("stage6_pipe_mask_sealed", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_img("stage7_pipe_skeleton", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_json("stage9_node_clusters", {"clusters": []})
            pipe._save_json("stage4_topology_markers", {"topology_markers": [{"id": "topology_marker::obj_1"}]})

            with patch("garnet.pid_extractor.run_pipe_crossing_stage") as mock_crossings, patch(
                "garnet.pid_extractor.run_pipe_edge_stage"
            ) as mock_pipe_edges:
                mock_crossings.return_value = {
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                    "crossings_payload": {"candidates": []},
                    "summary": {
                        "image_id": "image.png",
                        "pass_type": "sheet",
                        "candidate_count": 0,
                    },
                }
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

            mock_crossings.assert_called_once()
            self.assertEqual(mock_crossings.call_args.kwargs["topology_markers"], [{"id": "topology_marker::obj_1"}])
            mock_pipe_edges.assert_called_once()
            self.assertTrue((Path(tmp) / "stage10_crossing_resolution_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage10_crossing_resolution.json").exists())
            self.assertTrue((Path(tmp) / "stage10_crossing_resolution_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage10_pipe_edges_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage10_pipe_edges.json").exists())
            self.assertTrue((Path(tmp) / "stage10_pipe_edge_summary.json").exists())

    def test_stage10b_writes_simplified_edge_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe._save_json(
                "stage10_pipe_edges",
                {
                    "edges": [
                        {
                            "id": "edge_0",
                            "source": "endpoint_0",
                            "target": "endpoint_1",
                            "polyline": [{"row": 5, "col": col} for col in range(10)],
                            "pixel_length": 10,
                        }
                    ]
                },
            )

            pipe.stage10b_polyline_simplification()

            simplified_payload = json.loads((Path(tmp) / "stage10b_pipe_edges_simplified.json").read_text())
            summary = json.loads((Path(tmp) / "stage10b_polyline_simplification_summary.json").read_text())
            self.assertEqual(len(simplified_payload["edges"][0]["polyline"]), 2)
            self.assertEqual(simplified_payload["edges"][0]["pixel_length"], 10)
            self.assertEqual(simplified_payload["edges"][0]["simplified_pixel_length"], 2)
            self.assertEqual(summary["total_original_point_count"], 10)
            self.assertEqual(summary["total_simplified_point_count"], 2)

    def test_stage10c_writes_edge_direction_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe._save_json(
                "stage10b_pipe_edges_simplified",
                {
                    "edges": [
                        {
                            "id": "edge_0",
                            "source": "endpoint_0",
                            "target": "endpoint_1",
                            "polyline": [{"row": 5, "col": col} for col in range(0, 50, 10)],
                            "pixel_length": 50,
                            "simplified_pixel_length": 5,
                        }
                    ]
                },
            )
            pipe._save_json(
                "stage4_objects",
                {
                    "objects": [
                        {
                            "id": "arrow_0",
                            "class_name": "arrow",
                            "bbox": {"x_min": 20, "y_min": 0, "x_max": 45, "y_max": 10},
                            "direction": "right",
                        }
                    ]
                },
            )

            pipe.stage10c_edge_direction()

            directed_payload = json.loads((Path(tmp) / "stage10c_edge_direction.json").read_text())
            assignments = json.loads((Path(tmp) / "stage10c_arrow_assignments.json").read_text())
            summary = json.loads((Path(tmp) / "stage10c_edge_direction_summary.json").read_text())
            self.assertEqual(directed_payload["edges"][0]["flow_direction"], "forward")
            self.assertEqual(directed_payload["edges"][0]["assigned_arrow_id"], "arrow_0")
            self.assertEqual(assignments["arrow_assignments"][0]["edge_id"], "edge_0")
            self.assertEqual(summary["edges_with_forward_direction"], 1)

    def test_stage10d_writes_split_edge_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe._save_json(
                "stage10c_edge_direction",
                {
                    "edges": [
                        {
                            "id": "edge_0",
                            "source": "endpoint_0",
                            "target": "endpoint_1",
                            "polyline": [{"row": 5, "col": col} for col in range(0, 100, 10)],
                            "pixel_length": 10,
                            "flow_direction": "forward",
                        }
                    ]
                },
            )
            pipe._save_json(
                "stage12_edge_connections",
                {
                    "edge_connections": [
                        {
                            "kind": "inline_element",
                            "connector_id": "valve_0",
                            "connector_class": "valve",
                            "source_edge_id": "edge_0",
                            "target_edge_id": "edge_0",
                            "distance_px": 4.0,
                        }
                    ]
                },
            )
            pipe._save_json(
                "stage4_objects",
                {
                    "objects": [
                        {
                            "id": "valve_0",
                            "class_name": "valve",
                            "bbox": {"x_min": 45, "y_min": 0, "x_max": 55, "y_max": 10},
                        }
                    ]
                },
            )

            pipe.stage10d_edge_split()

            split_edges = json.loads((Path(tmp) / "stage10d_split_edges.json").read_text())
            split_nodes = json.loads((Path(tmp) / "stage10d_split_nodes.json").read_text())
            split_report = json.loads((Path(tmp) / "stage10d_split_report.json").read_text())
            split_summary = json.loads((Path(tmp) / "stage10d_split_summary.json").read_text())
            self.assertEqual(len(split_edges["edges"]), 2)
            self.assertEqual(split_nodes["nodes"][0]["id"], "inline::valve_0")
            self.assertEqual(split_report[0]["status"], "split")
            self.assertEqual(split_summary["edges_split"], 1)

    def test_stage11_writes_junction_review_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_json("stage10_crossing_resolution", {"candidates": []})

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
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_json("stage4_objects", {"objects": []})
            pipe._save_json("stage2_ocr_regions", {"text_regions": []})
            pipe._save_json("stage4_line_numbers", {"line_numbers": []})
            pipe._save_json("stage4_instrument_tags", {"instrument_tags": []})
            pipe._save_json("stage4_equipment_tags", {"equipment_tags": []})
            pipe._save_json("stage9_node_clusters", {"clusters": []})
            pipe._save_json("stage10_crossing_resolution", {"candidates": []})
            pipe._save_json("stage10d_split_edges", {"edges": []})
            pipe._save_json("stage10d_split_nodes", {"nodes": []})
            pipe._save_json(
                "stage10b_polyline_simplification_summary",
                {"total_original_point_count": 0, "total_simplified_point_count": 0, "compression_ratio": 1.0},
            )
            pipe._save_json(
                "stage10c_edge_direction_summary",
                {"total_edges": 0, "edges_without_direction": 0, "arrows_assigned_to_edge": 0},
            )
            pipe._save_json(
                "stage10d_split_summary",
                {"total_inline_connections": 0, "edges_split": 0, "nodes_created": 0},
            )
            pipe._save_json("stage10c_arrow_assignments", {"arrow_assignments": []})
            pipe._save_json("stage11_junctions", {"confirmed_junctions": [], "unresolved_junctions": []})

            with patch("garnet.pid_extractor.run_pipe_equipment_attachment_stage") as mock_pipe_attachment, patch(
                "garnet.pid_extractor.classify_pipe_edge_terminals"
            ) as mock_pipe_terminals, patch(
                "garnet.pid_extractor.run_pipe_text_attachment_stage"
            ) as mock_pipe_text_attachment, patch(
                "garnet.pid_extractor.run_pipe_graph_stage"
            ) as mock_pipe_graph:
                mock_pipe_attachment.return_value = {
                    "attachments_payload": {"accepted": [], "rejected": []},
                    "summary": {"accepted_attachment_count": 0},
                }
                mock_pipe_terminals.return_value = {
                    "edge_terminals": [],
                    "summary": {"edge_count": 0, "validated_edge_count": 0, "provisional_edge_count": 0},
                }
                mock_pipe_text_attachment.return_value = {
                    "attachments_payload": {"accepted": [], "rejected": []},
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
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

            self.assertEqual(mock_pipe_attachment.call_count, 2)
            mock_pipe_terminals.assert_called_once()
            self.assertEqual(mock_pipe_text_attachment.call_count, 2)
            mock_pipe_graph.assert_called_once()
            self.assertTrue((Path(tmp) / "stage12_equipment_attachments.json").exists())
            self.assertTrue((Path(tmp) / "stage12_equipment_attachment_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_connection_attachments.json").exists())
            self.assertTrue((Path(tmp) / "stage12_connection_attachment_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_connection_attachment_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage12_junction_decision_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage12_edge_terminals.json").exists())
            self.assertTrue((Path(tmp) / "stage12_edge_terminal_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_arrow_assignments.json").exists())
            self.assertTrue((Path(tmp) / "stage12_edge_connections.json").exists())
            self.assertTrue((Path(tmp) / "stage12_edge_connection_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_candidate_links.json").exists())
            self.assertTrue((Path(tmp) / "stage12_candidate_link_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_selected_candidate_links.json").exists())
            self.assertTrue((Path(tmp) / "stage12_candidate_link_diff.json").exists())
            self.assertTrue((Path(tmp) / "stage12_candidate_link_selection_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_candidate_link_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage12_rejected_junction_connections.json").exists())
            self.assertTrue((Path(tmp) / "stage12_rejected_junction_connection_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_text_attachments.json").exists())
            self.assertTrue((Path(tmp) / "stage12_text_attachment_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_text_attachment_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage12_overlay_edges_filtered.json").exists())
            self.assertTrue((Path(tmp) / "stage12_overlay_edges_filtered_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_instrument_tag_attachments.json").exists())
            self.assertTrue((Path(tmp) / "stage12_instrument_tag_attachment_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_equipment_tag_attachments.json").exists())
            self.assertTrue((Path(tmp) / "stage12_equipment_tag_attachment_summary.json").exists())
            self.assertTrue((Path(tmp) / "stage12_equipment_tag_attachment_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage12_graph.json").exists())
            self.assertTrue((Path(tmp) / "stage12_graph_summary.json").exists())
            graph_summary = json.loads((Path(tmp) / "stage12_graph_summary.json").read_text())
            self.assertIn("polyline_simplification", graph_summary)
            self.assertIn("edge_direction", graph_summary)
            self.assertIn("edge_split", graph_summary)

    def test_stage12c_page_connector_labeling_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe._save_json(
                "stage12_connection_attachments",
                {
                    "accepted": [
                        {
                            "det_id": "obj_1",
                            "class_name": "page connection",
                            "bbox": [100, 100, 120, 120],
                        }
                    ]
                },
            )
            pipe._save_json("stage12_equipment_attachments", {"accepted": []})
            pipe._save_json(
                "stage2_ocr_regions",
                {
                    "text_regions": [
                        {
                            "id": "ocr_1",
                            "text": "SHEET P-101",
                            "bbox": {"x_min": 115, "y_min": 100, "x_max": 135, "y_max": 120},
                        }
                    ]
                },
            )

            pipe.stage12c_page_connector_labeling()

            labels = json.loads((Path(tmp) / "stage12_page_connector_labels.json").read_text())
            summary = json.loads((Path(tmp) / "stage12_page_connector_labels_summary.json").read_text())
            self.assertEqual(labels["connectors"][0]["object_id"], "obj_1")
            self.assertEqual(labels["connectors"][0]["labels"][0]["page_reference"]["reference_value"], "P-101")
            self.assertEqual(summary["total_connectors"], 1)

    def test_stage13_writes_graph_qa_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", out_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_json("stage12_graph", {"nodes": [], "edges": []})

            with patch("garnet.pid_extractor.run_pipe_graph_qa_stage") as mock_pipe_graph_qa:
                mock_pipe_graph_qa.return_value = {
                    "anomaly_report": {"connected_component_count": 0},
                    "component_overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
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
            self.assertTrue((Path(tmp) / "stage13_graph_components_overlay.png").exists())
            self.assertTrue((Path(tmp) / "stage13_review_queue.json").exists())
            self.assertTrue((Path(tmp) / "stage13_graph_qa_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
