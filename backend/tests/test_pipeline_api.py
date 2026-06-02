import time
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import numpy as np

try:
    from api import app
except ModuleNotFoundError as exc:
    if exc.name == "pdf2image":
        app = None
    else:
        raise


@unittest.skipIf(app is None, "pdf2image is not installed in this test environment")
class PipelineApiTests(unittest.TestCase):
    def test_pipeline_stage_status_reports_stale_after_stage3_artifact_update(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "stage_state_job_1"
            manifest = {
                "stages": [
                    {"num": 1, "name": "stage1_input_normalization", "status": "completed"},
                    {"num": 2, "name": "stage2_ocr_discovery", "status": "completed"},
                    {"num": 4, "name": "stage4_object_detection", "status": "completed"},
                    {"num": 5, "name": "stage5_pipe_mask", "status": "completed"},
                    {"num": 5, "name": "stage5b_pipe_trace", "status": "completed"},
                    {"num": 6, "name": "stage6_trace_associations", "status": "completed"},
                ]
            }
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage6_trace_associations",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 6,
                "ocr_route": "ocrmac",
                "gemini_postprocess_match_threshold": 0.1,
                "weight_file": "yolo_weights/model.pt",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/artifacts/stage3_equipment_bboxes.json",
                    json={"equipment": [{"id": "equip_001", "class_name": "vessel", "bbox": {"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 20}}]},
                )
                self.assertEqual(response.status_code, 200)

                status_response = client.get(f"/api/pipeline/jobs/{job_id}/stage-status")

            self.assertEqual(status_response.status_code, 200)
            stages = {item["name"]: item for item in status_response.json()["stages"]}
            self.assertEqual(stages["stage5b_pipe_trace"]["status"], "stale")
            self.assertEqual(stages["stage6_trace_associations"]["status"], "stale")
            self.assertEqual(stages["stage4_object_detection"]["status"], "completed")
            self.assertTrue((Path(tmp) / "stage3_equipment_bboxes.json").exists())
            self.assertFalse((Path(tmp) / "stage5_connection_ports.json").exists())

    def test_pipeline_resume_from_stage_reruns_from_requested_stage(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            image_path.write_bytes(b"placeholder")
            manifest = {
                "image_path": str(image_path),
                "stages": [
                    {"num": 1, "name": "stage1_input_normalization", "status": "completed"},
                    {"num": 2, "name": "stage2_ocr_discovery", "status": "completed"},
                    {"num": 5, "name": "stage5_pipe_mask", "status": "completed"},
                    {"num": 5, "name": "stage5b_pipe_trace", "status": "stale"},
                ],
            }
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            run_calls: list[tuple[int, bool]] = []

            class FakeResumePipeline:
                def __init__(self, image_path: str, output_dir: str, stage_callback=None, cfg=None) -> None:
                    self.stage_manifest = {"stages": [{"name": "stage5b_pipe_trace"}]}

                def run(self, stop_after: int, resume: bool = False) -> None:
                    run_calls.append((stop_after, resume))

            job_id = "stage_state_job_2"
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage5b_pipe_trace",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 5,
                "ocr_route": "ocrmac",
                "gemini_postprocess_match_threshold": 0.1,
                "weight_file": "yolo_weights/model.pt",
            }}, clear=False), patch("api.PIDPipeline", FakeResumePipeline):
                response = client.post(f"/api/pipeline/jobs/{job_id}/resume-from/stage5b_pipe_trace")
                self.assertEqual(response.status_code, 200)
                deadline = time.time() + 5
                while time.time() < deadline and not run_calls:
                    time.sleep(0.05)

            self.assertEqual(run_calls, [(5, True)])

    def test_pipeline_review_state_get_returns_empty_default(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_job_1"
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage13_graph_qa",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 13,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.get(f"/api/pipeline/jobs/{job_id}/review-state")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["job_id"], Path(tmp).name)
            self.assertEqual(payload["items"], [])

    def test_pipeline_review_state_put_persists_payload(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_job_2"
            with open(Path(tmp) / "stage_manifest.json", "w", encoding="utf-8") as f:
                json.dump({"image_path": "sample.png"}, f)
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage13_graph_qa",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 13,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/review-state",
                    json={
                        "items": [
                            {
                                "item_id": "stage4_line_number:line_number_000001",
                                "bucket": "stage4_line_number",
                                "source_stage": "stage4_line_number_fusion",
                                "source_artifact": "stage4_line_numbers.json",
                                "entity_id": "line_number_000001",
                                "decision": "accepted",
                            }
                        ],
                        "workspace_objects": {"stage4_line_number": [{"Object": "line_number"}]},
                    },
                )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(len(payload["items"]), 1)
            self.assertEqual(payload["items"][0]["decision"], "accepted")
            self.assertTrue((Path(tmp) / "stage_review_state.json").exists())

    def test_pipeline_review_state_put_rejects_invalid_bucket(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            job_id = "review_job_3"
            with patch.dict("api.PIPELINE_JOBS", {job_id: {
                "job_id": job_id,
                "status": "completed",
                "current_stage": "stage13_graph_qa",
                "error": None,
                "job_dir": tmp,
                "created_at": time.time(),
                "stop_after": 13,
                "ocr_route": "ocrmac",
            }}, clear=False):
                response = client.put(
                    f"/api/pipeline/jobs/{job_id}/review-state",
                    json={
                        "items": [
                            {
                                "item_id": "bad",
                                "bucket": "bad_bucket",
                                "decision": "accepted",
                            }
                        ],
                        "workspace_objects": {},
                    },
                )
            self.assertEqual(response.status_code, 400)

    def test_pipeline_job_runs_stage2_and_reports_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "text_regions": [
                    {
                        "id": "ocr_000001",
                        "text": "P-1001",
                        "normalized_text": "P-1001",
                        "class": "line_number",
                        "confidence": 0.91,
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 90, "y_max": 40},
                        "rotation": 0,
                        "reading_direction": "horizontal",
                        "legibility": "clear",
                    }
                ],
            },
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "tile_count": 1,
                "raw_detection_count": 1,
                "merged_region_count": 1,
                "exception_candidate_count": 0,
                "slice_height": 1600,
                "slice_width": 1600,
                "overlap_height_ratio": 0.2,
                "overlap_width_ratio": 0.2,
            },
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "2", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIn("job_id", payload)
            job_id = payload["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["current_stage"], "stage2_ocr_discovery")
            self.assertEqual(job_payload["ocr_route"], "easyocr")
            self.assertEqual(len(job_payload["manifest"]["stages"]), 2)
            self.assertEqual(job_payload["manifest"]["ocr_route"], "easyocr")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage1_gray.png", artifact_names)
            self.assertIn("stage1_normalization_summary.json", artifact_names)
            self.assertIn("stage2_ocr_regions.json", artifact_names)
            self.assertIn("stage2_ocr_summary.json", artifact_names)
            self.assertIn("stage2_ocr_exception_candidates.json", artifact_names)

    def test_pipeline_job_accepts_ocrmac_route(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "route": "ocrmac"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }

        with patch("garnet.pid_extractor.run_ocrmac_sahi", return_value=fake_ocr_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "2", "ocr_route": "ocrmac"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["ocr_route"], "ocrmac")

    def test_pipeline_job_runs_stage4_and_reports_object_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "tile_count": 1,
                "raw_detection_count": 0,
                "merged_region_count": 0,
                "exception_candidate_count": 0,
                "slice_height": 1600,
                "slice_width": 1600,
                "overlap_height_ratio": 0.2,
                "overlap_width_ratio": 0.2,
            },
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "objects": [
                    {
                        "id": "obj_000001",
                        "class_name": "valve",
                        "confidence": 0.88,
                        "bbox": {"x_min": 5, "y_min": 6, "x_max": 20, "y_max": 30},
                        "source_model": "ultralytics",
                        "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
                    }
                ],
            },
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 1,
                "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_line_number_fusion_result = {
            "line_numbers_payload": {"line_numbers": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"matched_line_number_count": 0},
        }
        fake_instrument_tag_fusion_result = {
            "instrument_tags_payload": {"instrument_tags": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"matched_instrument_tag_count": 0},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch(
            "garnet.pid_extractor.run_line_number_fusion_stage", return_value=fake_line_number_fusion_result
        ), patch(
            "garnet.pid_extractor.run_instrument_tag_fusion_stage", return_value=fake_instrument_tag_fusion_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "4", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["current_stage"], "stage4_instrument_tag_fusion")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage4_objects.json", artifact_names)
            self.assertIn("stage4_objects_summary.json", artifact_names)
            self.assertIn("stage4_objects_overlay.png", artifact_names)
            self.assertIn("stage4_line_numbers.json", artifact_names)
            self.assertIn("stage4_line_number_summary.json", artifact_names)
            self.assertIn("stage4_line_number_overlay.png", artifact_names)
            self.assertIn("stage4_instrument_tags.json", artifact_names)
            self.assertIn("stage4_instrument_tag_summary.json", artifact_names)
            self.assertIn("stage4_instrument_tag_overlay.png", artifact_names)

    def test_pipeline_job_uses_selected_weight_file(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"
        selected_weight = "yolo_weights/custom-selected.pt"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": selected_weight,
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_line_number_fusion_result = {
            "line_numbers_payload": {"line_numbers": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"matched_line_number_count": 0},
        }
        fake_instrument_tag_fusion_result = {
            "instrument_tags_payload": {"instrument_tags": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"matched_instrument_tag_count": 0},
        }

        with patch("api.resolve_pipeline_weight_file", return_value=selected_weight), patch(
            "garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result
        ), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch(
            "garnet.pid_extractor.run_line_number_fusion_stage", return_value=fake_line_number_fusion_result
        ), patch(
            "garnet.pid_extractor.run_instrument_tag_fusion_stage", return_value=fake_instrument_tag_fusion_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "4", "ocr_route": "easyocr", "weight_file": selected_weight},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["weight_file"], selected_weight)
            self.assertEqual(job_payload["manifest"]["detection_weight_path"], selected_weight)

    def test_pipeline_job_runs_stage5_and_reports_pipe_mask_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "mask_pixel_count": 42,
                "source_artifacts": [
                    "stage1_gray.png",
                    "stage2_ocr_regions.json",
                    "stage4_objects.json",
                ],
            },
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "5", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

            deadline = time.time() + 10
            job_payload = None
            while time.time() < deadline:
                poll = client.get(f"/api/pipeline/jobs/{job_id}")
                self.assertEqual(poll.status_code, 200)
                job_payload = poll.json()
                if job_payload["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            self.assertIsNotNone(job_payload)
            assert job_payload is not None
            self.assertEqual(job_payload["status"], "completed")
            self.assertEqual(job_payload["current_stage"], "stage5b_pipe_trace")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage5_pipe_mask.png", artifact_names)
            self.assertIn("stage5_pipe_mask_overlay.png", artifact_names)
            self.assertIn("stage5_pipe_mask_summary.json", artifact_names)



if __name__ == "__main__":
    unittest.main()
