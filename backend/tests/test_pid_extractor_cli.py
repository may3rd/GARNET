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
    def __init__(
        self,
        out_dir: str | Path,
        fail_stage: int | None = None,
        cfg: pid_extractor.PipelineConfig | None = None,
    ) -> None:
        self._input_path = Path(out_dir) / "image.png"
        if not self._input_path.exists():
            self._input_path.write_bytes(b"placeholder")
        super().__init__(str(self._input_path), output_dir=out_dir, cfg=cfg)
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

    def stage5b_pipe_trace(self) -> None:
        self._record("stage5b")

    def stage6_trace_associations(self) -> None:
        self._record("stage6")

    def stage7_geometric_graph_assembly(self) -> None:
        self._record("stage7")

    def stage7c_page_connector_labeling(self) -> None:
        self._record("stage7c")

    def stage7b_graph_export(self) -> None:
        self._record("stage7b")

    def stage8_graph_qa(self) -> None:
        self._record("stage8")

    def stage9_apply_review_decisions(self) -> None:
        self._record("stage9")

    def stage10_process_exports(self) -> None:
        self._record("stage10")

    def stage11_connection_overlay(self) -> None:
        self._record("stage11")

class PIDPipelineRunnerTests(unittest.TestCase):
    def test_document_id_defaults_to_filename_and_accepts_confirmed_sheet_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            default_pipe = pid_extractor.PIDPipeline("input.png", output_dir=tmp)
            system_pipe = pid_extractor.PIDPipeline("input.png", output_dir=tmp, document_id="DWG-200-02")

            self.assertEqual(default_pipe._image_id(), "input.png")
            self.assertEqual(system_pipe._image_id(), "DWG-200-02")

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
                "stage5_pipe_mask",
                "stage5b_pipe_trace",
                "stage6_trace_associations",
                "stage7_geometric_graph_assembly",
                "stage7c_page_connector_labeling",
                "stage7b_graph_export",
                "stage8_graph_qa",
                "stage9_apply_review_decisions",
                "stage10_process_exports",
                "stage11_connection_overlay",
            ],
        )

    def test_stage10_writes_phase8_views_and_registers_the_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            image_path = out_dir / "sheet.png"
            image_path.write_bytes(b"placeholder")
            (out_dir / "stage9_corrected_graph.json").write_text(
                json.dumps({
                    "schema_version": "graph_v1",
                    "graph_revision": "stage9-r1",
                    "document": {"doc_id": "sheet"},
                    "nodes": [
                        {"id": "n1", "type": "equipment", "position": {"x": 0, "y": 0}},
                        {"id": "n2", "type": "terminal", "position": {"x": 10, "y": 0}},
                    ],
                    "edges": [{
                        "id": "edge-1",
                        "source": "n1",
                        "target": "n2",
                        "polyline": [[0, 0], [10, 0]],
                        "effective_line_number_ids": ["line-1"],
                        "flow_direction_state": "forward",
                    }],
                    "relationships": [],
                }),
                encoding="utf-8",
            )
            (out_dir / "stage9_release_gate.json").write_text(
                json.dumps({"release_ready": True, "status": "ready"}),
                encoding="utf-8",
            )
            pipeline = pid_extractor.PIDPipeline(str(image_path), output_dir=out_dir)
            pipeline.image_bgr = np.zeros((4, 12, 3), dtype=np.uint8)
            pipeline._current_stage_artifacts = []
            with patch(
                "garnet.stage10_process_exports.render_stage10_inline_mto_overlay",
                return_value=pipeline.image_bgr,
            ), patch(
                "garnet.stage10_process_exports.render_stage10_line_number_overlay",
                return_value=pipeline.image_bgr,
            ):
                pipeline.stage10_process_exports()

            expected = {
                "stage10_process_boundaries.json",
                "stage10_test_package_candidates.json",
                "stage10_engineering_view_summary.json",
                "stage10_llm_projections.json",
            }
            self.assertTrue(all((out_dir / name).is_file() for name in expected))
            self.assertTrue(expected.issubset(set(pipeline._current_stage_artifacts)))
            summary = json.loads((out_dir / "stage10_engineering_view_summary.json").read_text())
            self.assertEqual(summary["graph_revision"], "stage9-r1")
            self.assertRegex(summary["graph_content_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(summary["source_graph_content_sha256"], summary["graph_content_sha256"])
            self.assertEqual(summary["graph_counts"], {"node_count": 2, "edge_count": 1, "relationship_count": 0})
            boundary = json.loads((out_dir / "stage10_process_boundaries.json").read_text())
            packages = json.loads((out_dir / "stage10_test_package_candidates.json").read_text())
            self.assertEqual(boundary["source_graph_content_sha256"], summary["graph_content_sha256"])
            self.assertEqual(packages["source_graph_content_sha256"], summary["graph_content_sha256"])
            process_summary = json.loads((out_dir / "stage10_process_export_summary.json").read_text())
            self.assertEqual(process_summary["graph_summary"]["graph_revision"], "stage9-r1")
            llm = json.loads((out_dir / "stage10_llm_projections.json").read_text())
            self.assertEqual(llm["graph_summary"]["edge_count"], 1)
            self.assertEqual(llm["graph_summary"]["graph_content_sha256"], summary["graph_content_sha256"])
            self.assertEqual(llm["source_graph_content_sha256"], summary["graph_content_sha256"])

    def test_run_stops_after_requested_stage_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            pipe.run(stop_after=11)

            self.assertEqual(
                pipe.called,
                [
                    "stage1",
                    "stage2",
                    "stage4",
                    "stage4",
                    "stage4",
                    "stage5",
                    "stage5b",
                    "stage6",
                    "stage7",
                    "stage7c",
                    "stage7b",
                    "stage8",
                    "stage9",
                    "stage10",
                    "stage11",
                ],
            )
            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["stop_after"], 11)
            self.assertEqual(
                [item["name"] for item in manifest["stages"]],
                [
                    "stage1_input_normalization",
                    "stage2_ocr_discovery",
                    "stage4_object_detection",
                    "stage4_line_number_fusion",
                    "stage4_instrument_tag_fusion",
                    "stage5_pipe_mask",
                    "stage5b_pipe_trace",
                    "stage6_trace_associations",
                    "stage7_geometric_graph_assembly",
                    "stage7c_page_connector_labeling",
                    "stage7b_graph_export",
                    "stage8_graph_qa",
                    "stage9_apply_review_decisions",
                    "stage10_process_exports",
                    "stage11_connection_overlay",
                ],
            )
            self.assertTrue(all(item["status"] == "completed" for item in manifest["stages"]))

    def test_run_rejects_stop_after_past_last_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            with self.assertRaisesRegex(ValueError, "stop_after must be one of"):
                pipe.run(stop_after=12)

    def test_run_writes_failed_stage_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp, fail_stage=2)

            with self.assertRaisesRegex(RuntimeError, "stage2 failed"):
                pipe.run(stop_after=11)

            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["stages"][0]["status"], "completed")
            self.assertEqual(manifest["stages"][1]["status"], "failed")
            self.assertIn("stage2 failed", manifest["stages"][1]["error"])

    def test_run_manifest_records_v2_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)

            pipe.run(stop_after=1)

            manifest = json.loads((Path(tmp) / "stage_manifest.json").read_text())
            self.assertEqual(manifest["manifest_version"], 2)
            self.assertEqual(manifest["run_signature"]["input"]["size_bytes"], len(b"placeholder"))
            self.assertEqual(manifest["run_signature"]["config"]["ocr_route"], "ocrmac")
            self.assertIn("sha256", manifest["run_signature"]["input"])
            self.assertIn("git_revision", manifest["run_signature"])

    def test_resume_rejects_changed_input_before_rewriting_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)
            pipe.run(stop_after=1)
            manifest_path = Path(tmp) / "stage_manifest.json"
            pipe._input_path.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "input.sha256"):
                FakePipeline(tmp).run(stop_after=4, resume=True)

            self.assertEqual(json.loads(manifest_path.read_text())["stop_after"], 1)

    def test_resume_rejects_changed_config_before_rewriting_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)
            pipe.run(stop_after=1)
            manifest_path = Path(tmp) / "stage_manifest.json"

            with self.assertRaisesRegex(ValueError, "config.ocr_route"):
                FakePipeline(tmp, cfg=pid_extractor.PipelineConfig(ocr_route="easyocr")).run(
                    stop_after=4,
                    resume=True,
                )

            self.assertEqual(json.loads(manifest_path.read_text())["stop_after"], 1)

    def test_resume_rejects_changed_weight_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weight_path = Path(tmp) / "model.pt"
            weight_path.write_bytes(b"first")
            cfg = pid_extractor.PipelineConfig(detection_weight_path=str(weight_path))
            pipe = FakePipeline(tmp, cfg=cfg)
            pipe.run(stop_after=1)
            weight_path.write_bytes(b"second")

            with self.assertRaisesRegex(ValueError, "detection_weight.sha256"):
                FakePipeline(tmp, cfg=cfg).run(stop_after=1, resume=True)

    def test_resume_rejects_missing_completed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)
            pipe.run(stop_after=1)
            (Path(tmp) / "stage1_artifact.json").unlink()

            with self.assertRaisesRegex(ValueError, "stage1_artifact.json"):
                FakePipeline(tmp).run(stop_after=1, resume=True)

    def test_resume_rejects_missing_phase8_stage10_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            FakePipeline(tmp).run(stop_after=11)
            manifest_path = Path(tmp) / "stage_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            stage10 = next(item for item in manifest["stages"] if item["name"] == "stage10_process_exports")
            phase8_names = [
                "stage10_process_boundaries.json",
                "stage10_test_package_candidates.json",
                "stage10_engineering_view_summary.json",
                "stage10_llm_projections.json",
            ]
            for name in phase8_names:
                (Path(tmp) / name).write_text("{}", encoding="utf-8")
            stage10["artifacts"].extend(phase8_names)
            manifest_path.write_text(json.dumps(manifest))
            (Path(tmp) / phase8_names[-1]).unlink()

            with self.assertRaisesRegex(ValueError, "stage10_llm_projections.json"):
                FakePipeline(tmp).run(stop_after=11, resume=True)

    def test_stage10_writes_phase8_artifacts_from_released_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline(str(Path(tmp) / "image.png"), output_dir=tmp)
            (Path(tmp) / "image.png").write_bytes(b"placeholder")
            (Path(tmp) / "stage9_corrected_graph.json").write_text(
                json.dumps({"schema_version": "graph_v1", "document": {"doc_id": "P-101"}, "nodes": [], "edges": []}),
                encoding="utf-8",
            )
            (Path(tmp) / "stage9_release_gate.json").write_text(
                json.dumps({"release_ready": True, "status": "ready"}), encoding="utf-8"
            )
            with patch.object(pipe, "_ensure_image_loaded", return_value=np.zeros((4, 4, 3), dtype=np.uint8)):
                pipe.stage10_process_exports()

            for name in (
                "stage10_process_boundaries.json",
                "stage10_test_package_candidates.json",
                "stage10_engineering_view_summary.json",
                "stage10_llm_projections.json",
            ):
                self.assertTrue((Path(tmp) / name).is_file(), name)
            summary = json.loads((Path(tmp) / "stage10_process_export_summary.json").read_text())
            self.assertEqual(summary["phase8"]["source_release_gate_artifact"], "stage9_release_gate.json")

    def test_resume_rejects_non_contiguous_completed_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)
            pipe.run(stop_after=4)
            manifest_path = Path(tmp) / "stage_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            next(item for item in manifest["stages"] if item["name"] == "stage2_ocr_discovery")["status"] = "stale"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "contiguous"):
                FakePipeline(tmp).run(stop_after=4, resume=True)

    def test_resume_rejects_unsafe_artifact_name(self) -> None:
        for artifact in ("../outside.json", ".."):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as tmp:
                FakePipeline(tmp).run(stop_after=1)
                manifest_path = Path(tmp) / "stage_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["stages"][0]["artifacts"] = [artifact]
                manifest_path.write_text(json.dumps(manifest))

                with self.assertRaisesRegex(ValueError, "invalid artifact name"):
                    FakePipeline(tmp).run(stop_after=1, resume=True)

    def test_resume_rejects_symlink_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            FakePipeline(tmp).run(stop_after=1)
            manifest_path = Path(tmp) / "stage_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            real_path = Path(tmp) / "real.json"
            real_path.write_text("{}")
            (Path(tmp) / "link.json").symlink_to(real_path)
            manifest["stages"][0]["artifacts"] = ["link.json"]
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "not a regular file"):
                FakePipeline(tmp).run(stop_after=1, resume=True)

    def test_resume_allows_git_revision_change_and_runs_next_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("garnet.pid_extractor._git_revision", return_value="old"):
                FakePipeline(tmp).run(stop_after=1)

            resumed = FakePipeline(tmp)
            with patch("garnet.pid_extractor._git_revision", return_value="new"):
                resumed.run(stop_after=2, resume=True)

            self.assertEqual(resumed.called, ["stage2"])

    def test_resume_rejects_legacy_manifest_without_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = FakePipeline(tmp)
            (Path(tmp) / "stage_manifest.json").write_text(json.dumps({"stages": []}))

            with self.assertRaisesRegex(ValueError, "manifest version 2"):
                pipe.run(stop_after=1, resume=True)

    def test_save_img_rejects_failed_cv2_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=tmp)

            with patch("garnet.pid_extractor.cv2.imwrite", return_value=False):
                with self.assertRaisesRegex(OSError, "Failed to save image"):
                    pipe._save_img("failed", np.zeros((2, 2), dtype=np.uint8))

            self.assertEqual(pipe._current_stage_artifacts, [])

    def test_save_json_failure_preserves_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=tmp)
            artifact_path = Path(tmp) / "artifact.json"
            artifact_path.write_text('{"old": true}', encoding="utf-8")

            with self.assertRaises(TypeError):
                pipe._save_json("artifact", {"bad": object()})

            self.assertEqual(json.loads(artifact_path.read_text(encoding="utf-8")), {"old": True})
            self.assertFalse((Path(tmp) / ".artifact.json.tmp").exists())

    def test_stage2_uses_plain_gray_artifact_as_ocr_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline(
                "image.png",
                output_dir=tmp,
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

    def test_stage2_reruns_when_ocr_artifact_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline(
                "image.png",
                output_dir=tmp,
                cfg=pid_extractor.PipelineConfig(ocr_route="easyocr"),
            )
            pipe._save_img("stage1_gray", np.zeros((20, 20), dtype=np.uint8))
            pipe._save_json("stage2_ocr_regions", {"text_regions": [{"text": "stale"}]})

            with patch("garnet.pid_extractor.run_easyocr_sahi") as mock_ocr:
                mock_ocr.return_value = {
                    "regions_payload": {"image_id": "", "pass_type": "sheet", "text_regions": []},
                    "summary": {"image_id": "", "pass_type": "sheet"},
                    "exception_candidates": [],
                    "overlay_image": np.zeros((20, 20, 3), dtype=np.uint8),
                }

                pipe.stage2_ocr_discovery()

            mock_ocr.assert_called_once()
            payload = json.loads((Path(tmp) / "stage2_ocr_regions.json").read_text())
            self.assertEqual(payload["text_regions"], [])

    def test_pipeline_rejects_unknown_constructor_keyword(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                pid_extractor.PIDPipeline("image.png", output_dir=tmp, out_dir="ignored")

    def test_stage7c_labels_non_rejected_stage4_page_connections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=tmp)
            pipe._save_json(
                "stage4_objects",
                {
                    "objects": [
                        {
                            "id": "accepted",
                            "class_name": "page connection",
                            "review_state": "accepted",
                            "bbox": {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20},
                        },
                        {
                            "id": "rejected",
                            "class_name": "page connection",
                            "review_state": "rejected",
                            "bbox": {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20},
                        },
                        {
                            "id": "pump",
                            "class_name": "pump",
                            "bbox": {"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20},
                        },
                    ]
                },
            )
            pipe._save_json(
                "stage2_ocr_regions",
                {
                    "text_regions": [{
                        "id": "ocr_1",
                        "text": "SHEET P-101",
                        "bbox": {"x_min": 20, "y_min": 0, "x_max": 40, "y_max": 20},
                    }]
                },
            )

            pipe.stage7c_page_connector_labeling()

            payload = json.loads((Path(tmp) / "stage7_page_connector_labels.json").read_text())
            summary = json.loads((Path(tmp) / "stage7_page_connector_labels_summary.json").read_text())
            self.assertEqual([item["object_id"] for item in payload["connectors"]], ["accepted"])
            self.assertEqual(payload["connectors"][0]["labels"][0]["normalized_text"], "SHEET P-101")
            self.assertEqual(summary["total_connectors"], 1)

    def test_stage11_renders_current_graph_without_legacy_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=tmp)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_json("stage7_graph", {"nodes": [], "edges": []})

            with patch(
                "garnet.trace_graph_builder.render_stage12_graph_overlay",
                return_value=np.zeros((20, 20, 3), dtype=np.uint8),
            ):
                pipe.stage11_connection_overlay()

            self.assertTrue((Path(tmp) / "stage11_connection_pipeline_overlay.png").is_file())
            for name in (
                "stage7_connection_attachments.json",
                "stage7_edge_connections.json",
                "stage7_edge_terminals.json",
            ):
                self.assertFalse((Path(tmp) / name).exists())

    def test_pipeline_config_defaults_to_ocrmac_route(self) -> None:
        cfg = pid_extractor.PipelineConfig()

        self.assertEqual(cfg.ocr_route, "ocrmac")
        self.assertEqual(cfg.gemini_postprocess_match_threshold, 0.1)

    def test_pipeline_config_rejects_removed_fields(self) -> None:
        removed = (
            "equipment_tag_fusion_max_distance_px",
            "equipment_tag_attachment_max_distance_px",
            "pipe_mask_continuity_ocr_padding",
            "pipe_mask_continuity_min_component_area",
            "pipe_seal_horizontal_close_kernel",
            "pipe_seal_vertical_close_kernel",
            "pipe_seal_min_component_area",
            "node_cluster_eps",
            "node_cluster_min_samples",
            "min_edge_length_px",
            "crossing_branch_stub_length_px",
            "crossing_branch_merge_angle_tolerance_deg",
            "crossing_opposite_angle_tolerance_deg",
            "crossing_center_blob_radius_px",
            "crossing_center_blob_threshold",
            "crossing_stage4_marker_match_distance_px",
            "polyline_simplify_epsilon",
            "arrow_proximity_px",
            "inline_split_confidence_threshold",
            "equipment_attachment_classes",
            "equipment_attachment_max_distance_px",
            "equipment_attachment_k_candidate_edges",
            "connection_attachment_classes",
            "connection_attachment_max_distance_px",
            "connection_attachment_k_candidate_edges",
            "line_text_attachment_max_distance_px",
            "terminal_equipment_classes",
            "terminal_connection_classes",
            "terminal_inline_passthrough_classes",
            "terminal_match_distance_px",
            "graph_inline_connector_classes",
            "graph_inline_connector_match_distance_px",
        )
        self.assertEqual(len(removed), 32)
        for name in removed:
            with self.subTest(name=name), self.assertRaises(TypeError):
                pid_extractor.PipelineConfig(**{name: 1})

    def test_resolve_cli_weight_file_accepts_backend_relative_path(self) -> None:
        weight_dir = pid_extractor.BACKEND_DIR / "yolo_weights"
        weight_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=weight_dir,
            suffix=".pt",
        ) as tmp_weight:
            weight_path = Path(tmp_weight.name)

            resolved = pid_extractor._resolve_cli_weight_file(
                f"yolo_weights/{weight_path.name}"
            )

        self.assertEqual(resolved, f"yolo_weights/{weight_path.name}")

    def test_resolve_cli_weight_file_rejects_missing_path(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Weight file not found"):
            pid_extractor._resolve_cli_weight_file("yolo_weights/does-not-exist.pt")

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
                output_dir=tmp,
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
                output_dir=tmp,
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
                output_dir=tmp,
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
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=tmp)

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
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=tmp)
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
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=tmp)
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

    def test_stage6_materializes_reviewed_stage4_line_numbers_before_association(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=out_dir)
            pipe.image_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
            pipe._save_json("stage4_objects", {"objects": []})
            pipe._save_json("stage4_line_numbers", {"image_id": "image.png", "line_numbers": [], "rejected": []})
            pipe._save_json("stage4_instrument_tags", {"instrument_tags": []})
            pipe._save_json("stage5b_trace_results", {"traces": []})
            pipe._save_json("stage5b_branch_trace_results", {"branches": {}})
            pipe._save_json("stage5_connection_ports", {})
            (out_dir / "stage_review_state.json").write_text(
                json.dumps(
                    {
                        "workspace_objects": {
                            "stage4_line_number": [
                                {
                                    "Object": "line_number",
                                    "SourceItemId": "line_number_000001",
                                    "Text": "4-CUL-25-004007-L1A1-NI",
                                    "Left": 10,
                                    "Top": 20,
                                    "Width": 200,
                                    "Height": 30,
                                    "Score": 1,
                                    "ReviewStatus": "accepted",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch("garnet.pid_extractor.build_trace_associations") as mock_associations:
                mock_associations.return_value = {
                    "trace_associations_payload": {"trace_edges": []},
                    "trace_association_summary": {},
                    "line_number_review_payload": {"accepted": []},
                    "line_number_review_summary": {},
                    "trace_edges": [],
                    "associations": {},
                }
                pipe.stage6_trace_associations()

            line_payload = json.loads((out_dir / "stage4_line_numbers.json").read_text(encoding="utf-8"))
            self.assertEqual(line_payload["line_numbers"][0]["id"], "line_number_000001")
            self.assertEqual(
                mock_associations.call_args.kwargs["line_numbers"][0]["normalized_text"],
                "4-CUL-25-004007-L1A1-NI",
            )

    def test_stage5_writes_pipe_mask_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipe = pid_extractor.PIDPipeline("image.png", output_dir=tmp)
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



if __name__ == "__main__":
    unittest.main()
